import abc
from collections import defaultdict
import torch
from tensordict import TensorDict
import time
from tensordict.nn import TensorDictModule, set_interaction_type, InteractionType

from src.utils.policy import _policy_t
from src.utils.log import get_log_func
from src.utils.augment import augment_transition
from src.envs.gomoku_env import GomokuEnv 

# ==============================
# Reward Shaping (Potential 기반)
# ==============================

# 보상 쉐이핑 세기 (0.0이면 쉐이핑 꺼짐)
REWARD_SHAPING_LAMBDA: float = 0.02
POT_A: float = 1.0
POT_B: float = 1.2  # 방어 쪽을 조금 더 강조하려면 POT_B > POT_A

def _max_line_length_1d(line: torch.Tensor, player_value: int) -> int:
    """
    1차원 라인(가로/세로/대각선 한 줄)에서 해당 플레이어 돌의
    최장 연속 길이를 반환함.
    line: 1D tensor, 값은 {player_value, 다른 값들}
    """
    mask = (line == player_value).to(torch.int32)
    max_len = 0
    cur = 0
    for v in mask.tolist():
        if v == 1:
            cur += 1
            if cur > max_len:
                max_len = cur
        else:
            cur = 0
    return max_len


def _get_max_line_length(board: torch.Tensor, player_value: int) -> int:
    b = board.detach().cpu()
    if b.dim() != 2:
        raise ValueError(f"board must be 2D, got shape {b.shape}")

    H, W = b.shape
    max_len = 0

    # 가로
    for i in range(H):
        max_len = max(max_len, _max_line_length_1d(b[i, :], player_value))
    # 세로
    for j in range(W):
        max_len = max(max_len, _max_line_length_1d(b[:, j], player_value))

    # / 대각선 (왼쪽 위 ↘ 오른쪽 아래)
    for offset in range(-H + 1, W):
        diag = b.diagonal(offset=offset)  # <-- 여기
        if diag.numel() > 0:
            max_len = max(max_len, _max_line_length_1d(diag, player_value))

    # \ 대각선 (왼쪽 아래 ↗ 오른쪽 위)
    flipped = torch.flip(b, dims=[1])
    for offset in range(-H + 1, W):
        diag = flipped.diagonal(offset=offset)  # <-- 여기
        if diag.numel() > 0:
            max_len = max(max_len, _max_line_length_1d(diag, player_value))

    return max_len


def _extract_board_from_observation(obs: torch.Tensor) -> torch.Tensor:
    """
    GomokuEnv.get_encoded_board() 구조에 맞춰 보드를 추출함.

    - obs shape: (E, 3, B, B)
      채널 0: 현재 플레이어 돌 (0/1)
      채널 1: 상대 플레이어 돌 (0/1)
      채널 2: 마지막 수 위치 (0/1)  (여기서는 사용 안 함)

    반환:
      board: (E, B, B), 값은 {1, 0, -1}
             1  = 현재 플레이어 돌
            -1  = 상대 플레이어 돌
             0  = 빈칸
    """
    if obs.dim() != 4:
        raise ValueError(f"Expected obs shape (E, 3, B, B), got {obs.shape}")

    my = obs[:, 0]   # 현재 플레이어 돌 (0/1)
    opp = obs[:, 1]  # 상대 플레이어 돌 (0/1)
    board = my - opp  # {1, 0, -1}
    return board


def compute_potential_from_obs_batch(
    obs: torch.Tensor,
    a: float = POT_A,
    b: float = POT_B,
) -> torch.Tensor:
    """
    Phi(s) = a * my_max_line - b * opp_max_line
    을 배치(obs) 전체에 대해 계산함.

    - obs: (E, 3, B, B)
    - 반환: (E,)
    """
    boards = _extract_board_from_observation(obs).detach().cpu()
    if boards.dim() == 2:
        boards = boards.unsqueeze(0)
    B = boards.shape[0]

    phi_vals = torch.empty(B, dtype=torch.float32)
    for i in range(B):
        board = boards[i]
        # board에서 +1이 '현재 플레이어 돌', -1이 '상대 돌'이므로
        my_max = _get_max_line_length(board, 1)
        opp_max = _get_max_line_length(board, -1)
        phi_vals[i] = a * float(my_max) - b * float(opp_max)

    return phi_vals.to(obs.device)


def make_transition(
    tensordict_t_minus_1: TensorDict,
    tensordict_t: TensorDict,
    tensordict_t_plus_1: TensorDict,
) -> TensorDict:
    """
    Constructs a transition tensor dictionary for a two-player game...
    """

    # 1) 원래 승/패 기반 보상
    reward: torch.Tensor = (
        tensordict_t.get("win").float() -
        tensordict_t_plus_1.get("win").float()
    ).unsqueeze(-1)  # shape: (E, 1)

    # 2) terminal step에서만 보상 쉐이핑 추가
    obs_prev = tensordict_t_minus_1.get("observation", None)
    obs_next = tensordict_t_plus_1.get("observation", None)
    done_flag = tensordict_t_plus_1.get("done", None)

    if (
        REWARD_SHAPING_LAMBDA != 0.0
        and obs_prev is not None
        and obs_next is not None
        and done_flag is not None
    ):
        # done_flag shape: (E, 1) or (E,)
        done_mask = done_flag.view(-1).bool()
        if done_mask.any():
            # 에피소드가 끝나는 env들만 추려서 potential 계산
            phi_prev = compute_potential_from_obs_batch(obs_prev[done_mask])  # (E_done,)
            phi_next = compute_potential_from_obs_batch(obs_next[done_mask])  # (E_done,)

            shaping = REWARD_SHAPING_LAMBDA * (phi_next - phi_prev)           # (E_done,)

            # reward: (E, 1) 에 terminal 위치만 더해줌
            reward_flat = reward.view(-1, 1)
            reward_flat[done_mask, 0] = reward_flat[done_mask, 0] + shaping
            reward = reward_flat.view_as(reward)

    # 3) 나머지 transition 구성 (원래 코드 그대로)
    transition: TensorDict = tensordict_t_minus_1.select(
        "observation",
        "action_mask",
        "action",
        "sample_log_prob",
        "state_value",
        strict=False,
    )
    transition.set(
        "next",
        tensordict_t_plus_1.select(
            "observation", "action_mask", "state_value", strict=False
        ),
    )
    transition.set(("next", "reward"), reward)
    done = tensordict_t_plus_1["done"] | tensordict_t["done"]
    transition.set(("next", "done"), done)
    return transition



def round(env: GomokuEnv,
          policy_black: _policy_t,
          policy_white: _policy_t,
          tensordict_t_minus_1: TensorDict,
          tensordict_t: TensorDict,
          return_black_transitions: bool = True,
          return_white_transitions: bool = True,):
    """Executes two sequential steps in the Gomoku environment..."""
    
    tensordict_t_plus_1 = env.step_and_maybe_reset(
        tensordict=tensordict_t)
    with set_interaction_type(type=InteractionType.RANDOM):
        tensordict_t_plus_1 = policy_white(tensordict_t_plus_1)
    if return_white_transitions:
        transition_white = make_transition(
            tensordict_t_minus_1, tensordict_t, tensordict_t_plus_1
        )
        invalid: torch.Tensor = tensordict_t_minus_1["done"]
        transition_white["next", "done"] = (
            invalid | transition_white["next", "done"]
        )
        transition_white.set("invalid", invalid)
    else:
        transition_white = None

    tensordict_t_plus_2 = env.step_and_maybe_reset(
        tensordict_t_plus_1,
        env_mask=~tensordict_t_plus_1.get("done"),
    )
    with set_interaction_type(type=InteractionType.RANDOM):
        tensordict_t_plus_2 = policy_black(tensordict_t_plus_2)

    if return_black_transitions:
        transition_black = make_transition(
            tensordict_t, tensordict_t_plus_1, tensordict_t_plus_2
        )
        transition_black.set(
            "invalid",
            torch.zeros(env.num_envs, device=env.device,
                        dtype=torch.bool),
        )
    else:
        transition_black = None

    return (
        transition_black,
        transition_white,
        tensordict_t_plus_1,
        tensordict_t_plus_2,
    )


def self_play_step(
    env: GomokuEnv,
    policy: _policy_t,
    tensordict_t_minus_1: TensorDict,
    tensordict_t: TensorDict,
):
    """Executes a single step of self-play..."""
    
    tensordict_t_plus_1 = env.step_and_maybe_reset(
        tensordict=tensordict_t
    )
    with set_interaction_type(type=InteractionType.RANDOM):
        tensordict_t_plus_1 = policy(tensordict_t_plus_1)
    transition = make_transition(
        tensordict_t_minus_1, tensordict_t, tensordict_t_plus_1
    )
    return (
        transition,
        tensordict_t,
        tensordict_t_plus_1,
    )

class Collector(abc.ABC):
    @abc.abstractmethod
    def rollout(self, steps: int) -> tuple[TensorDict, dict]:
        ...
    @abc.abstractmethod
    def reset(self):
        ...

class SelfPlayCollector(Collector):
    def __init__(self, env: GomokuEnv, policy: _policy_t, out_device=None, augment: bool = False):
        """Initializes a collector for self-play data..."""
        
        self._env = env
        self._policy = policy
        self._out_device = out_device or self._env.device
        self._augment = augment
        self._t = None
        self._t_minus_1 = None
    def reset(self):
        
        self._env.reset()
        self._t = None
        self._t_minus_1 = None
    @torch.no_grad()
    def rollout(self, steps: int) -> tuple[TensorDict, dict]:
        """Executes a rollout in the environment..."""
        
        info: defaultdict[str, float] = defaultdict(float)
        self._env.set_post_step(get_log_func(info))
        tensordicts = []
        start = time.perf_counter()
        if self._t_minus_1 is None and self._t is None:
            self._t_minus_1 = self._env.reset()
            with set_interaction_type(type=InteractionType.RANDOM):
                self._t_minus_1 = self._policy(self._t_minus_1)
            self._t = self._env.step(self._t_minus_1)
            with set_interaction_type(type=InteractionType.RANDOM):
                self._t = self._policy(self._t)
        for i in range(steps - 1):
            (
                transition,
                self._t_minus_1,
                self._t,
            ) = self_play_step(self._env, self._policy, self._t_minus_1, self._t)
            if i == steps-2:
                transition["next", "done"] = torch.ones(
                    transition["next", "done"].shape, dtype=torch.bool, device=transition.device)
            if self._augment:
                transition = augment_transition(transition)
            tensordicts.append(transition.to(self._out_device))
        end = time.perf_counter()
        fps = (steps * self._env.num_envs) / (end - start)
        self._env.set_post_step(None)
        tensordicts = torch.stack(tensordicts, dim=-1)
        info.update({"fps": fps})
        return tensordicts, dict(info)

class VersusPlayCollector(Collector):
    def __init__(self, env: GomokuEnv, policy_black: _policy_t, policy_white: _policy_t, out_device=None, augment: bool = False):
        """Initializes a collector for versus play data..."""
        
        self._env = env
        self._policy_black = policy_black
        self._policy_white = policy_white
        self._out_device = out_device or self._env.device
        self._augment = augment
        self._t_minus_1 = None
        self._t = None
    def reset(self):
        
        self._env.reset()
        self._t_minus_1 = None
        self._t = None
    @torch.no_grad()
    def rollout(self, steps: int) -> tuple[TensorDict, TensorDict, dict]:
        """Executes a rollout in the environment..."""
        
        steps = (steps//2)*2
        info: defaultdict[str, float] = defaultdict(float)
        self._env.set_post_step(get_log_func(info))
        blacks = []
        whites = []
        start = time.perf_counter()
        if self._t_minus_1 is None and self._t is None:
            self._t_minus_1 = self._env.reset()
            self._t = self._env.reset()
            self._t_minus_1.update(
                {
                    "done": torch.ones(
                        self._env.num_envs, dtype=torch.bool, device=self._env.device
                    ),
                    "win": torch.zeros(
                        self._env.num_envs, dtype=torch.bool, device=self._env.device
                    ),
                }
            )
            with set_interaction_type(type=InteractionType.RANDOM):
                self._t = self._policy_black(self._t)
            self._t .update(
                {
                    "done": torch.zeros(
                        self._env.num_envs, dtype=torch.bool, device=self._env.device
                    ),
                    "win": torch.zeros(
                        self._env.num_envs, dtype=torch.bool, device=self._env.device
                    ),
                }
            )
        for i in range(steps//2):
            (
                transition_black,
                transition_white,
                self._t_minus_1,
                self._t,
            ) = round(self._env, self._policy_black, self._policy_white, self._t_minus_1, self._t)
            if i == steps//2-1:
                transition_black["next", "done"] = torch.ones(
                    transition_black["next", "done"].shape, dtype=torch.bool, device=transition_black.device)
                transition_white["next", "done"] = torch.ones(
                    transition_white["next", "done"].shape, dtype=torch.bool, device=transition_white.device)
            if self._augment:
                transition_black = augment_transition(transition_black)
                if i != 0:
                    transition_white = augment_transition(transition_white)
            blacks.append(transition_black.to(self._out_device))
            if i != 0:
                whites.append(transition_white.to(self._out_device))
        blacks = torch.stack(blacks, dim=-1) if blacks else None
        whites = torch.stack(whites, dim=-1) if whites else None
        end = time.perf_counter()
        fps = (steps * self._env.num_envs) / (end - start)
        self._env.set_post_step(None)
        info.update({"fps": fps})
        return blacks, whites, dict(info)

class BlackPlayCollector(Collector):
    def __init__(self, env: GomokuEnv, policy_black: _policy_t, policy_white: _policy_t, out_device=None, augment: bool = False):
        """Initializes a collector for capturing game transitions (black player)..."""
        
        self._env = env
        self._policy_black = policy_black
        self._policy_white = policy_white
        self._out_device = out_device or self._env.device
        self._augment = augment
        self._t_minus_1 = None
        self._t = None
    def reset(self):
        
        self._env.reset()
        self._t_minus_1 = None
        self._t = None
    @torch.no_grad()
    def rollout(self, steps: int) -> tuple[TensorDict, dict]:
        """Executes a data collection session... (black player)"""
        
        steps = (steps//2)*2
        info: defaultdict[str, float] = defaultdict(float)
        self._env.set_post_step(get_log_func(info))
        blacks = []
        start = time.perf_counter()
        if self._t_minus_1 is None and self._t is None:
            self._t_minus_1 = self._env.reset()
            self._t = self._env.reset()
            self._t_minus_1.update(
                {
                    "done": torch.ones(
                        self._env.num_envs, dtype=torch.bool, device=self._env.device
                    ),
                    "win": torch.zeros(
                        self._env.num_envs, dtype=torch.bool, device=self._env.device
                    ),
                }
            )
            with set_interaction_type(type=InteractionType.RANDOM):
                self._t = self._policy_black(self._t)
            self._t .update(
                {
                    "done": torch.zeros(
                        self._env.num_envs, dtype=torch.bool, device=self._env.device
                    ),
                    "win": torch.zeros(
                        self._env.num_envs, dtype=torch.bool, device=self._env.device
                    ),
                }
            )
        for i in range(steps//2):
            (
                transition_black,
                transition_white,
                self._t_minus_1,
                self._t,
            ) = round(self._env, self._policy_black, self._policy_white, self._t_minus_1, self._t, return_black_transitions=True, return_white_transitions=False)
            if i == steps//2-1:
                transition_black["next", "done"] = torch.ones(
                    transition_black["next", "done"].shape, dtype=torch.bool, device=transition_black.device)
            if self._augment:
                transition_black = augment_transition(transition_black)
            blacks.append(transition_black.to(self._out_device))
        blacks = torch.stack(blacks, dim=-1) if blacks else None
        end = time.perf_counter()
        fps = (steps * self._env.num_envs) / (end - start)
        self._env.set_post_step(None)
        info.update({"fps": fps})
        return blacks, dict(info)

class WhitePlayCollector(Collector):
    def __init__(self, env: GomokuEnv, policy_black: _policy_t, policy_white: _policy_t, out_device=None, augment: bool = False):
        """Initializes a collector focused on capturing game transitions (white player)..."""
        self._env = env
        self._policy_black = policy_black
        self._policy_white = policy_white
        self._out_device = out_device or self._env.device
        self._augment = augment
        self._t_minus_1 = None
        self._t = None
    def reset(self):
        self._env.reset()
        self._t_minus_1 = None
        self._t = None
    @torch.no_grad()
    def rollout(self, steps: int) -> tuple[TensorDict, dict]:
        """Performs a data collection session... (white player)"""
        steps = (steps//2)*2
        info: defaultdict[str, float] = defaultdict(float)
        self._env.set_post_step(get_log_func(info))
        whites = []
        start = time.perf_counter()
        if self._t_minus_1 is None and self._t is None:
            self._t_minus_1 = self._env.reset()
            self._t = self._env.reset()
            self._t_minus_1.update(
                {
                    "done": torch.ones(
                        self._env.num_envs, dtype=torch.bool, device=self._env.device
                    ),
                    "win": torch.zeros(
                        self._env.num_envs, dtype=torch.bool, device=self._env.device
                    ), "action": -torch.ones(
                        self._env.num_envs, dtype=torch.long, device=self._env.device  #  이 부분 수정 후 테스트 예정
                    ),
                }
            )
            with set_interaction_type(type=InteractionType.RANDOM):
                self._t = self._policy_black(self._t)
            self._t .update(
                {
                    "done": torch.zeros(
                        self._env.num_envs, dtype=torch.bool, device=self._env.device
                    ),
                    "win": torch.zeros(
                        self._env.num_envs, dtype=torch.bool, device=self._env.device
                    ),
                }
            )
        for i in range(steps//2):
            (
                transition_black,
                transition_white,
                self._t_minus_1,
                self._t,
            ) = round(self._env, self._policy_black, self._policy_white, self._t_minus_1, self._t, return_black_transitions=False, return_white_transitions=True)
            if i == steps//2-1:
                transition_white["next", "done"] = torch.ones(
                    transition_white["next", "done"].shape, dtype=torch.bool, device=transition_white.device)
            if self._augment:
                if i != 0 and len(transition_white) > 0:
                    transition_white = augment_transition(transition_white)
            if i != 0:
                whites.append(transition_white.to(self._out_device))
        whites = torch.stack(whites, dim=-1) if whites else None
        end = time.perf_counter()
        fps = (steps * self._env.num_envs) / (end - start)
        self._env.set_post_step(None)
        info.update({"fps": fps})
        return whites, dict(info)
