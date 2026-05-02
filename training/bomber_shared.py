"""Shared Bomber IL utilities: observation encoding, demo collection, replay buffer.

Used by SQIL (DQfD) and bc_ppo_lstm (BC + PPO). No CUDA or training side effects.
"""

import sys
from pathlib import Path
from typing import Callable, Optional, Tuple, Any, Sequence, Union

import numpy as np
from tqdm import tqdm

parent_dir = Path(__file__).resolve().parent.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

from engine import BomberEnv, Map, Player
from agent import (
    SimpleRuleAgent,
    SmarterRuleAgent,
    TacticalRuleAgent,
    GeniusRuleAgent,
    BoxFarmerAgent,
)

BOMB_MAX_TIMER = 7
NUM_ACTIONS = 6

AGENT_LOOKUP = {
    "genius": GeniusRuleAgent,
    "tactical": TacticalRuleAgent,
    "smarter": SmarterRuleAgent,
    "simple": SimpleRuleAgent,
    "box_farmer": BoxFarmerAgent,
}


def _make_agent(name: str, agent_id: int):
    cls = AGENT_LOOKUP.get(name)
    if cls is None:
        raise ValueError(f"Unknown agent type: {name}")
    return cls(agent_id)


def normalize_opponent_names(names: Union[str, Sequence[str]], opp_ids: Sequence[int]) -> list[str]:
    """Broadcast a single type to all opponents, or require one name per ``opp_ids`` slot.

    Order matches ``opp_ids`` (ascending enemy ids for the usual ``range(4)`` minus learner).
    """
    if isinstance(names, str):
        lst = [names]
    else:
        lst = list(names)
    n = len(opp_ids)
    if len(lst) == 1:
        return lst * n
    if len(lst) == n:
        return lst
    raise ValueError(
        f"Expected 1 or {n} opponent types, got {len(lst)}: {lst!r} (opp_ids={list(opp_ids)})"
    )


class ReplayBuffer:
    """Pre-allocated numpy circular buffer."""

    def __init__(self, capacity: int, map_shape, aux_dim: int):
        self.capacity = capacity
        self.pos = 0
        self.size = 0
        self.map_shape = tuple(map_shape)
        self.aux_dim = int(aux_dim)
        self.map_states = np.zeros((capacity, *self.map_shape), dtype=np.float32)
        self.aux_states = np.zeros((capacity, self.aux_dim), dtype=np.float32)
        self.next_map_states = np.zeros((capacity, *self.map_shape), dtype=np.float32)
        self.next_aux_states = np.zeros((capacity, self.aux_dim), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)

    def __len__(self):
        return self.size

    def push(self, map_state, aux_state, action, reward, next_map_state, next_aux_state, done):
        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
        self.map_states[self.pos] = map_state
        self.aux_states[self.pos] = aux_state
        self.next_map_states[self.pos] = next_map_state
        self.next_aux_states[self.pos] = next_aux_state
        self.actions[self.pos] = action
        self.rewards[self.pos] = reward
        self.dones[self.pos] = done

    def sample(self, batch_size: int):
        idx = np.random.randint(0, self.size, size=batch_size)
        return (
            self.map_states[idx],
            self.aux_states[idx],
            self.next_map_states[idx],
            self.next_aux_states[idx],
            self.actions[idx],
            self.rewards[idx],
            self.dones[idx],
        )


def encode_obs(obs, agent_ids):
    """Encode raw observation into (map_feat [C,H,W], aux_feat [A,])."""
    if obs is None:
        raise ValueError("obs should not be None")
    user_id = int(agent_ids[0])
    players = obs["players"]
    num_players = int(players.shape[0])

    if len(agent_ids) > 1:
        ordered_ids = [int(i) for i in agent_ids]
    else:
        ordered_ids = [user_id] + [i for i in range(num_players) if i != user_id]

    opp_ids = [i for i in ordered_ids[1:] if i != user_id]
    while len(opp_ids) < 3:
        opp_ids.append(-1)

    grid = obs["map"]
    bombs = obs["bombs"]
    H, W = grid.shape

    map_channels = [(grid == v).astype(np.float32) for v in (
        Map.GRASS, Map.WALL, Map.BOX, Map.ITEM_RADIUS, Map.ITEM_CAPACITY)]

    my_x, my_y, my_alive, my_bombs_left, my_radius_bonus = players[user_id]
    my_pos = np.zeros((H, W), dtype=np.float32)
    if int(my_alive) == 1:
        my_pos[int(my_x), int(my_y)] = 1.0

    opp_pos_planes = []
    opp_alive_flags = []
    for oid in opp_ids[:3]:
        plane = np.zeros((H, W), dtype=np.float32)
        alive_flag = 0.0
        if 0 <= oid < num_players:
            ox, oy, o_alive, _, _ = players[oid]
            if int(o_alive) == 1:
                plane[int(ox), int(oy)] = 1.0
                alive_flag = 1.0
        opp_pos_planes.append(plane)
        opp_alive_flags.append(alive_flag)

    bomb_timer = np.zeros((H, W), dtype=np.float32)
    bomb_owned = np.zeros((H, W), dtype=np.float32)
    for b in bombs:
        bx, by, timer, owner_id = b
        bx, by = int(bx), int(by)
        bomb_timer[bx, by] = max(bomb_timer[bx, by], float(timer) / BOMB_MAX_TIMER)
        bomb_owned[bx, by] = 1.0 if int(owner_id) == user_id else 0.0

    scalar = np.array([
        float(my_bombs_left) / Player.MAX_BOMB_CAPACITY,
        float(my_radius_bonus) / Player.MAX_BOMB_RADIUS,
        *opp_alive_flags[:3],
    ], dtype=np.float32)

    map_feat = np.stack([
        *map_channels,
        my_pos,
        *opp_pos_planes[:3],
        bomb_timer,
        bomb_owned,
    ], axis=0).astype(np.float32)
    return map_feat, scalar


_HFLIP_ACTION = np.array([0, 2, 1, 3, 4, 5], dtype=np.int64)
_VFLIP_ACTION = np.array([0, 1, 2, 4, 3, 5], dtype=np.int64)
_HVFLIP_ACTION = np.array([0, 2, 1, 4, 3, 5], dtype=np.int64)


def augment_transition(map_state, aux_state, action):
    """Return 3 augmented copies: h-flip, v-flip, hv-flip."""
    h_map = np.flip(map_state, axis=1).copy()
    v_map = np.flip(map_state, axis=2).copy()
    hv_map = np.flip(np.flip(map_state, axis=1), axis=2).copy()
    return [
        (h_map, aux_state.copy(), int(_HFLIP_ACTION[action])),
        (v_map, aux_state.copy(), int(_VFLIP_ACTION[action])),
        (hv_map, aux_state.copy(), int(_HVFLIP_ACTION[action])),
    ]


def collect_demonstrations(
    expert_type: str,
    opponent_type: Union[str, Sequence[str]],
    num_episodes: int,
    max_steps: int,
    seed: int,
    augment: bool = True,
    store_dqfd_buffer: bool = True,
    reward_fn: Optional[Callable[..., float]] = None,
) -> Tuple[dict, Optional[ReplayBuffer], Tuple[Any, int]]:
    """Collect expert demonstrations (win-filtered). Optionally skip DQfD replay buffer.

    ``opponent_type`` is either one key in ``AGENT_LOOKUP`` (all opponents) or one name
    per opponent id in ``[1,2,3]`` (expert is always slot 0 in demos).

    reward_fn(prev_obs, next_obs, agent_id) -> float; required if store_dqfd_buffer True.
    """
    if store_dqfd_buffer and reward_fn is None:
        raise ValueError("reward_fn is required when store_dqfd_buffer is True")

    env = BomberEnv(max_steps=max_steps, seed=seed)
    expert_id = 0
    expert = _make_agent(expert_type, agent_id=expert_id)
    opp_ids = [i for i in range(4) if i != expert_id]
    opp_types = normalize_opponent_names(opponent_type, opp_ids)
    opponents = [_make_agent(t, agent_id=i) for t, i in zip(opp_types, opp_ids)]
    agent_ids = [expert_id, *opp_ids]

    dummy_obs = env.reset(seed=seed)
    sample = encode_obs(dummy_obs, agent_ids)
    input_spec = (sample[0].shape, sample[1].shape[0])

    bc_maps, bc_auxs, bc_actions = [], [], []
    demo_buffer: Optional[ReplayBuffer] = None
    if store_dqfd_buffer:
        capacity = num_episodes * max_steps * (4 if augment else 1)
        demo_buffer = ReplayBuffer(
            capacity=capacity, map_shape=input_spec[0], aux_dim=input_spec[1]
        )

    wins, total_eps = 0, 0
    for ep in tqdm(range(num_episodes), desc="Collecting demos"):
        obs = env.reset(seed=seed + ep)
        map_state, aux_state = encode_obs(obs, agent_ids)
        prev_obs = None
        ep_transitions = []

        for _ in range(max_steps):
            actions = [None] * 4
            expert_action = expert.act(obs)
            actions[expert_id] = expert_action
            for opp in opponents:
                actions[opp.agent_id] = opp.act(obs)

            next_obs, terminated, truncated = env.step(actions)
            done = terminated or truncated
            next_map_state, next_aux_state = encode_obs(next_obs, agent_ids)
            r = 0.0
            if reward_fn is not None:
                r = float(reward_fn(prev_obs, next_obs, agent_id=expert_id))

            ep_transitions.append((
                map_state, aux_state, expert_action, r,
                next_map_state, next_aux_state, float(done),
            ))

            prev_obs = obs
            obs = next_obs
            map_state = next_map_state
            aux_state = next_aux_state
            if done:
                break

        total_eps += 1
        alive_final = next_obs["players"][:, 2].astype(np.int8)
        expert_alive = int(alive_final[expert_id]) == 1
        opp_alive_any = any(int(alive_final[i]) == 1 for i in opp_ids)
        expert_won = expert_alive and (not opp_alive_any)

        if not expert_won:
            continue
        wins += 1

        for ms, axs, act, rew, nms, naxs, d in ep_transitions:
            if store_dqfd_buffer and demo_buffer is not None:
                demo_buffer.push(ms, axs, act, rew, nms, naxs, d)
            bc_maps.append(ms)
            bc_auxs.append(axs)
            bc_actions.append(act)
            if augment:
                for aug_ms, aug_axs, aug_act in augment_transition(ms, axs, act):
                    bc_maps.append(aug_ms)
                    bc_auxs.append(aug_axs)
                    bc_actions.append(aug_act)

    buf_len = len(demo_buffer) if demo_buffer is not None else 0
    print(
        f"Demo collection: {wins}/{total_eps} winning episodes, "
        f"{len(bc_maps)} BC samples (augmented), {buf_len} replay transitions"
    )

    bc_data = {
        "map": np.array(bc_maps, dtype=np.float32),
        "aux": np.array(bc_auxs, dtype=np.float32),
        "action": np.array(bc_actions, dtype=np.int64),
    }
    return bc_data, demo_buffer, input_spec
