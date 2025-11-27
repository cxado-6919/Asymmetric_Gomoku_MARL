from typing import Callable
from tensordict import TensorDict
from tensordict.nn import TensorDictModule, set_interaction_type, InteractionType
import torch
import torch.nn.functional as F

from torchrl.data.tensor_specs import (
    CompositeSpec,
    DiscreteTensorSpec,
    BinaryDiscreteTensorSpec,
    UnboundedContinuousTensorSpec,
)
import time

from src.utils.policy import _policy_t
from src.utils.misc import add_prefix
from src.utils.log import get_log_func
from src.envs.core import Gomoku

from collections import defaultdict


class GomokuEnv:
    def __init__(
        self,
        num_envs: int,
        board_size: int,
        device=None,
    ):
        """Initializes a parallel Gomoku environment."""
        self.gomoku = Gomoku(
            num_envs=num_envs, board_size=board_size, device=device
        )

        self.observation_spec = CompositeSpec(
            {
                "observation": UnboundedContinuousTensorSpec(
                    device=self.device,
                    shape=[num_envs, 3, board_size, board_size],
                ),
                "action_mask": BinaryDiscreteTensorSpec(
                    n=board_size * board_size,
                    device=self.device,
                    shape=[num_envs, board_size * board_size],
                    dtype=torch.bool,
                ),
            },
            shape=[
                num_envs,
            ],
            device=self.device,
        )
        self.action_spec = DiscreteTensorSpec(
            board_size * board_size,
            shape=[
                num_envs,
            ],
            device=self.device,
        )
        self.reward_spec = UnboundedContinuousTensorSpec(
            shape=[num_envs, 1],
            device=self.device,
        )

        self._post_step: Callable[
            [
                TensorDict,
            ],
            None,
        ] | None = None

    @property
    def batch_size(self):
        return torch.Size((self.num_envs,))

    @property
    def board_size(self):
        return self.gomoku.board_size

    @property
    def device(self):
        return self.gomoku.device

    @property
    def num_envs(self):
        return self.gomoku.num_envs

    def _max_line_in_five(self, mask: torch.Tensor) -> torch.Tensor:
        """
        mask (E, B, B): 현재 플레이어 혹은 상대 플레이어 돌 위치 (bool 또는 0/1 float)

        가로 / 세로 / 양 대각선으로 5칸 윈도우를 슬라이딩 하면서
        한 윈도우 안에 포함된 최대 연속 돌 개수(<=5)를 계산.
        """
        if mask.dtype not in (torch.float32, torch.float64):
            x = mask.float().unsqueeze(1)  # (E,1,B,B)
        else:
            x = mask.unsqueeze(1)

        gomoku = self.gomoku

        out_h = F.conv2d(x, gomoku.kernel_horizontal)  # (E,1,B-4,B)
        out_v = F.conv2d(x, gomoku.kernel_vertical)    # (E,1,B,B-4)
        out_d = F.conv2d(x, gomoku.kernel_diagonal)    # (E,2,B-4,B-4)

        max_h = out_h.flatten(start_dim=1).amax(dim=1)  # (E,)
        max_v = out_v.flatten(start_dim=1).amax(dim=1)  # (E,)
        max_d = out_d.flatten(start_dim=1).amax(dim=1)  # (E,)

        return torch.stack([max_h, max_v, max_d], dim=1).amax(dim=1)

    def reset(self, env_indices: torch.Tensor | None = None) -> TensorDict:
        self.gomoku.reset(env_indices=env_indices)
        tensordict = TensorDict(
            {
                "observation": self.gomoku.get_encoded_board(),
                "action_mask": self.gomoku.get_action_mask(),
            },
            self.batch_size,
            device=self.device,
        )
        return tensordict

    def step(
        self,
        tensordict: TensorDict,
    ) -> TensorDict:
        action: torch.Tensor = tensordict.get("action")
        env_mask: torch.Tensor | None = tensordict.get("env_mask", None)

        # env_mask 기본값: 모든 env에 대해 수를 둔다고 가정
        if env_mask is None:
            env_mask = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)

        # ----- step 이전 상태 백업 (보상 계산용) -----
        board_before = self.gomoku.board.clone()      # (E,B,B)
        turn_before = self.gomoku.turn.clone()        # (E,)
        move_count_before = self.gomoku.move_count.clone()  # (E,)
        # -----------------------------------------

        episode_len = move_count_before + 1
        win, illegal = self.gomoku.step(action=action, env_mask=env_mask)

        # 현재 구현에서는 illegal move가 나오지 않도록 설계
        assert not illegal.any()

        done = win
        black_win = win & (episode_len % 2 == 1)
        white_win = win & (episode_len % 2 == 0)

        # ========= 보상 쉐이핑 =========
        # 현재 수를 둔 플레이어의 stone 값 (1: black, -1: white)
        piece = torch.where(
            turn_before == 0,
            torch.ones_like(turn_before, dtype=torch.long),
            -torch.ones_like(turn_before, dtype=torch.long),
        )  # (E,)

        piece_view = piece.view(-1, 1, 1)  # (E,1,1)

        board_after = self.gomoku.board  # step 이후 보드 (E,B,B)

        # 자기 / 상대 돌 마스크 (step 전/후)
        own_before = board_before == piece_view
        opp_before = board_before == -piece_view
        own_after = board_after == piece_view
        opp_after = board_after == -piece_view

        # 최장 연속 줄 길이 (최대 5)
        own_max_before = self._max_line_in_five(own_before)  # (E,)
        own_max_after = self._max_line_in_five(own_after)    # (E,)
        opp_max_before = self._max_line_in_five(opp_before)  # (E,)
        opp_max_after = self._max_line_in_five(opp_after)    # (E,)

        # 자기 줄 증가 / 상대 줄 차단 정도
        delta_self = (own_max_after - own_max_before).clamp(min=0.0)
        delta_block = (opp_max_before - opp_max_after).clamp(min=0.0)

        # 중앙에 둘수록 작은 보너스 (L1 거리 기준 정규화)
        board_size = self.board_size
        x = action // board_size
        y = action % board_size
        center = (board_size - 1) / 2.0
        dist_center = (x.float() - center).abs() + (y.float() - center).abs()
        max_dist = 2.0 * (board_size - 1)
        center_reward = -0.03 * (dist_center / max_dist)  # 중앙에 가까울수록 0에 가깝고, 바깥으로 갈수록 -0.03에 근접

        # 실제로 수를 둔 env에만 보상 반영
        valid_mask = env_mask & (~illegal)

        # 하이퍼파라미터 (필요하면 튜닝)
        w_len_self = 0.04   # 자기 줄 길이 증가 보상 가중치
        w_len_block = 0.06  # 상대 줄 차단 보상 가중치
        step_penalty = -0.002  # 매 수마다 작은 페널티

        shaping = (
            w_len_self * delta_self
            + w_len_block * delta_block
            + center_reward
            + step_penalty
        )

        shaping = shaping * valid_mask.float()

        # 승리 시 +1 추가 보상 (현재 수를 둔 플레이어가 5목을 완성한 경우)
        win_reward = win.float()

        reward = (win_reward + shaping).unsqueeze(-1)  # (E,1)
        # ===============================

        tensordict = TensorDict({}, self.batch_size, device=self.device)
        tensordict.update(
            {
                "observation": self.gomoku.get_encoded_board(),
                "action_mask": self.gomoku.get_action_mask(),
                "reward": reward,
                "done": done,
                "win": win,
                "stats": {
                    "episode_len": episode_len,
                    "black_win": black_win,
                    "white_win": white_win,
                    # 디버그/로그용 추가 통계
                    "own_max_before": own_max_before,
                    "own_max_after": own_max_after,
                    "opp_max_before": opp_max_before,
                    "opp_max_after": opp_max_after,
                    "delta_self": delta_self,
                    "delta_block": delta_block,
                },
            }
        )
        if self._post_step:
            self._post_step(tensordict)
        return tensordict

    def step_and_maybe_reset(
        self,
        tensordict: TensorDict,
        env_mask: torch.Tensor | None = None,
    ) -> TensorDict:
        if env_mask is not None:
            tensordict.set("env_mask", env_mask)
        next_tensordict = self.step(tensordict=tensordict)
        tensordict.exclude("env_mask", inplace=True)

        done: torch.Tensor = next_tensordict.get("done")
        env_ids = done.nonzero().squeeze(0)
        reset_td = self.reset(env_indices=env_ids)
        next_tensordict.update(reset_td)
        return next_tensordict

    def set_post_step(self, post_step: Callable[[TensorDict], None] | None = None):
        self._post_step = post_step
