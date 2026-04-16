"""ICEC 2019 Bomberman-style rewards (Goulart et al., Table 1).

Scaled to this engine's observations. Kill credit is approximate: the env does
not attribute deaths to a specific bomb owner, so we use +1.0 per net enemy
death while the learner remains alive (same pragmatic choice as reward.py).

Reference: Learning How to Play Bomberman with Deep RL and IL (hal-03652029).
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

parent_dir = Path(__file__).resolve().parent.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

from engine import Map

from training.reward import (
    _any_bombs,
    _blast_status_at,
    _bomb_radius_from_obs,
    _enemy_alive_count,
    _explosion_tiles_for_bomb,
    _manhattan_to_nearest_alive_enemy,
    _parse_bomb_row,
)

# Table 1 (paper) — numeric values as published
R_AGENT_DEATH = -0.5
R_ENEMY_DEATH = 1.0
R_LAST_ALIVE = 1.0
R_BLOCK_DESTROYED = 0.1
R_CLOSEST_SO_FAR = 0.1
R_CLOSER_STEP = 0.002
R_FARTHER_STEP = -0.002
R_TIME_STEP = -0.02
R_IN_BLAST = -0.000666
R_SAFE_NEAR_BOMB = 0.002
R_PLANT_BOMB_TACTICAL = 0.05
R_PLANT_BOMB_UNSAFE = -0.08
R_OWN_BOMB_BLAST_PENALTY = -0.01
R_IDLE_STREAK_PENALTY_PER_STEP = -0.004
R_IDLE_STREAK_PENALTY_CAP = -0.04

# Manhattan distance to nearest bomb center to count as "bomb nearby" when safe
SAFE_NEAR_BOMB_RADIUS = 4
IDLE_STREAK_GRACE_STEPS = 6

# Anti–reward-hacking: dense shaping cap per episode, decay with time, and
# movement bonuses only when improving global best distance to an enemy.
# Cap raised so the shaping signal lasts more of the episode before saturating.
DENSE_SHAPING_CAP_PER_EPISODE = 0.8
DENSE_DECAY_TAU_STEPS = 280.0

# Small bonus per box or alive enemy covered by a newly planted bomb.
PLANT_TARGET_BONUS_PER_TARGET = 0.02
PLANT_TARGET_BONUS_CAP = 0.08


@dataclass
class EpisodeRewardState:
    """Mutable per-episode state for closest-enemy-so-far shaping."""

    best_dist: Optional[float] = None
    step_idx: int = 0
    dense_cumulative: float = 0.0
    idle_streak: int = 0
    last_pos: Optional[Tuple[int, int]] = None

    def reset(self) -> None:
        self.best_dist = None
        self.step_idx = 0
        self.dense_cumulative = 0.0
        self.idle_streak = 0
        self.last_pos = None


def _box_count(obs) -> int:
    return int(np.sum(obs["map"] == Map.BOX))


def _min_manhattan_to_bomb_center(obs, x: int, y: int) -> Optional[int]:
    bombs = obs["bombs"]
    if bombs is None or np.asarray(bombs).size == 0:
        return None
    arr = np.asarray(bombs)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    best = None
    for i in range(arr.shape[0]):
        parsed = _parse_bomb_row(arr[i])
        if parsed is None:
            continue
        bx, by, _, _ = parsed
        d = abs(int(x) - int(bx)) + abs(int(y) - int(by))
        best = d if best is None else min(best, d)
    return best


def _is_walkable_for_escape(grid, row: int, col: int) -> bool:
    cell = int(grid[row, col])
    return cell not in (Map.WALL, Map.BOX)


def _has_escape_route_after_plant(obs, row: int, col: int, radius: int, fuse_steps: int = 7) -> bool:
    """True if there is a reachable tile outside this bomb's blast before detonation.

    Uses a shortest-path BFS on the current grid while treating existing bomb tiles as
    blocked (except starting tile). This is a local heuristic used only for reward shaping.
    """
    grid = obs["map"]
    h, w = grid.shape
    blast_tiles = _explosion_tiles_for_bomb(grid, row, col, radius)
    max_steps = max(0, int(fuse_steps) - 1)

    blocked = set()
    bombs = np.asarray(obs["bombs"])
    if bombs.size > 0:
        if bombs.ndim == 1:
            bombs = bombs.reshape(1, -1)
        for i in range(bombs.shape[0]):
            parsed = _parse_bomb_row(bombs[i])
            if parsed is None:
                continue
            bx, by, _, _ = parsed
            blocked.add((int(bx), int(by)))
    blocked.discard((row, col))

    q = [(row, col, 0)]
    visited = {(row, col)}
    head = 0
    while head < len(q):
        r, c, d = q[head]
        head += 1

        if d <= max_steps and (r, c) not in blast_tiles:
            return True
        if d >= max_steps:
            continue

        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = r + dr, c + dc
            if not (0 <= nr < h and 0 <= nc < w):
                continue
            if (nr, nc) in visited:
                continue
            if not _is_walkable_for_escape(grid, nr, nc):
                continue
            if (nr, nc) in blocked:
                continue
            visited.add((nr, nc))
            q.append((nr, nc, d + 1))
    return False


def _count_plant_targets(grid, row: int, col: int, radius: int, players, agent_id: int) -> int:
    """Count boxes + alive enemies in blast for a bomb planted at (row, col)."""
    blast_tiles = _explosion_tiles_for_bomb(grid, row, col, radius)
    boxes = sum(1 for tr, tc in blast_tiles if int(grid[tr, tc]) == Map.BOX)
    alive_enemies = 0
    for pid in range(len(players)):
        if pid == agent_id:
            continue
        if int(players[pid][2]) != 1:
            continue
        px, py = int(players[pid][0]), int(players[pid][1])
        if (px, py) in blast_tiles:
            alive_enemies += 1
    return boxes + alive_enemies


def compute_reward_icec(
    prev_obs,
    curr_obs,
    agent_id: int,
    episode_state: EpisodeRewardState,
    out_components: Optional[Dict[str, float]] = None,
) -> Tuple[float, EpisodeRewardState]:
    """Return (reward, episode_state). Mutates ``episode_state`` in place.

    Movement shaping (closer / closest-so-far) only applies when the step
    **improves** the running best Manhattan distance to a living enemy, so
    oscillating left/right cannot farm dense rewards. Positive dense shaping is
    decayed over time and capped per episode.

    If ``out_components`` is a dict, it is cleared and filled with this step's
    contribution per component (before episode cap; ``dense_applied`` is after).
    """
    if prev_obs is None:
        if out_components is not None:
            out_components.clear()
        return 0.0, episode_state

    episode_state.step_idx += 1
    decay = math.exp(-episode_state.step_idx / max(DENSE_DECAY_TAU_STEPS, 1e-6))

    if out_components is not None:
        out_components.clear()

    def _add_comp(key: str, val: float) -> None:
        if out_components is not None:
            out_components[key] = out_components.get(key, 0.0) + val

    prev_players = prev_obs["players"]
    curr_players = curr_obs["players"]
    prev_alive = int(prev_players[agent_id][2])
    curr_alive = int(curr_players[agent_id][2])

    if prev_alive == 1 and curr_alive == 0:
        _add_comp("agent_death", float(R_AGENT_DEATH))
        if out_components is not None:
            out_components["total"] = float(R_AGENT_DEATH)
        return float(R_AGENT_DEATH), episode_state

    reward = 0.0

    prev_enemies_alive = _enemy_alive_count(prev_players, agent_id)
    curr_enemies_alive = _enemy_alive_count(curr_players, agent_id)

    if curr_alive == 1 and curr_enemies_alive < prev_enemies_alive:
        v = R_ENEMY_DEATH * float(prev_enemies_alive - curr_enemies_alive)
        reward += v
        _add_comp("enemy_death", v)
    if (
        curr_alive == 1
        and curr_enemies_alive == 0
        and prev_enemies_alive > 0
    ):
        reward += R_LAST_ALIVE
        _add_comp("last_alive", float(R_LAST_ALIVE))

    prev_boxes = _box_count(prev_obs)
    curr_boxes = _box_count(curr_obs)
    if curr_boxes < prev_boxes:
        v = R_BLOCK_DESTROYED * float(prev_boxes - curr_boxes)
        reward += v
        _add_comp("block_destroyed", v)

    reward += R_TIME_STEP
    _add_comp("time_step", float(R_TIME_STEP))

    prev_x = int(prev_players[agent_id][0])
    prev_y = int(prev_players[agent_id][1])
    curr_x = int(curr_players[agent_id][0])
    curr_y = int(curr_players[agent_id][1])
    moved = prev_x != curr_x or prev_y != curr_y

    dense_pos_this_step = 0.0

    # Anti-camping: penalize repeated idling only when enemies are still alive and
    # there is no immediate bomb context to justify waiting.
    if curr_alive == 1 and curr_enemies_alive > 0:
        if moved:
            episode_state.idle_streak = 0
        else:
            if episode_state.last_pos == (curr_x, curr_y):
                episode_state.idle_streak += 1
            else:
                episode_state.idle_streak = 1
        episode_state.last_pos = (curr_x, curr_y)
        if (
            episode_state.idle_streak > IDLE_STREAK_GRACE_STEPS
            and not _any_bombs(curr_obs)
        ):
            over = episode_state.idle_streak - IDLE_STREAK_GRACE_STEPS
            v = max(
                R_IDLE_STREAK_PENALTY_CAP,
                R_IDLE_STREAK_PENALTY_PER_STEP * float(over),
            )
            reward += v
            _add_comp("idle_streak_penalty", float(v))
    else:
        episode_state.idle_streak = 0
        episode_state.last_pos = (curr_x, curr_y)

    if curr_alive == 1 and moved:
        prev_d = _manhattan_to_nearest_alive_enemy(prev_players, agent_id, prev_x, prev_y)
        curr_d = _manhattan_to_nearest_alive_enemy(curr_players, agent_id, curr_x, curr_y)
        if prev_d is not None and curr_d is not None:
            improving_best = episode_state.best_dist is None or curr_d < episode_state.best_dist

            if curr_d < prev_d and improving_best:
                v = R_CLOSER_STEP * decay
                reward += v
                dense_pos_this_step += v
                _add_comp("closer_step", v)
            elif curr_d > prev_d:
                v = R_FARTHER_STEP * decay
                reward += v
                _add_comp("farther_step", v)

            if improving_best:
                v = R_CLOSEST_SO_FAR * decay
                reward += v
                dense_pos_this_step += v
                _add_comp("closest_so_far", v)
                episode_state.best_dist = float(curr_d)

    prev_bombs_left = int(prev_players[agent_id][3])
    curr_bombs_left = int(curr_players[agent_id][3])
    planted_this_step = curr_alive == 1 and curr_bombs_left < prev_bombs_left
    if planted_this_step:
        radius = _bomb_radius_from_obs(curr_players, agent_id)
        targets = _count_plant_targets(curr_obs["map"], curr_x, curr_y, radius, curr_players, agent_id)
        target_bonus = min(PLANT_TARGET_BONUS_CAP, PLANT_TARGET_BONUS_PER_TARGET * float(targets))
        has_escape = _has_escape_route_after_plant(curr_obs, curr_x, curr_y, radius, fuse_steps=7)
        if has_escape:
            v = (R_PLANT_BOMB_TACTICAL + target_bonus) * decay
            reward += v
            dense_pos_this_step += v
            _add_comp("plant_bomb_tactical", v)
            if target_bonus > 0:
                _add_comp("plant_target_bonus", target_bonus * decay)
        else:
            v = R_PLANT_BOMB_UNSAFE
            reward += v
            _add_comp("plant_bomb_unsafe", float(v))

    if curr_alive == 1:
        in_blast, _ = _blast_status_at(curr_obs, curr_x, curr_y)
        if in_blast:
            reward += R_IN_BLAST
            _add_comp("in_blast", float(R_IN_BLAST))
        elif _any_bombs(curr_obs):
            dbomb = _min_manhattan_to_bomb_center(curr_obs, curr_x, curr_y)
            if dbomb is not None and dbomb <= SAFE_NEAR_BOMB_RADIUS:
                v = R_SAFE_NEAR_BOMB * decay
                reward += v
                dense_pos_this_step += v
                _add_comp("safe_near_bomb", v)

        own_blast_timer = None
        for b in curr_obs["bombs"]:
            parsed = _parse_bomb_row(b)
            if parsed is None:
                continue
            bx, by, timer, owner_id = parsed
            if int(owner_id) != int(agent_id):
                continue
            radius = _bomb_radius_from_obs(curr_players, owner_id)
            if (curr_x, curr_y) in _explosion_tiles_for_bomb(curr_obs["map"], bx, by, radius):
                own_blast_timer = timer if own_blast_timer is None else min(own_blast_timer, timer)
        if own_blast_timer is not None:
            urgency = (8 - int(own_blast_timer)) / 7.0
            v = R_OWN_BOMB_BLAST_PENALTY * max(0.15, urgency)
            reward += v
            _add_comp("own_bomb_blast", float(v))

    # Cap only positive dense shaping (not time / death / blocks / kills).
    cap_room = max(0.0, DENSE_SHAPING_CAP_PER_EPISODE - episode_state.dense_cumulative)
    if dense_pos_this_step > cap_room:
        scale = (cap_room / dense_pos_this_step) if dense_pos_this_step > 0 else 0.0
        if out_components is not None:
            for key in ("closer_step", "closest_so_far", "safe_near_bomb", "plant_bomb_tactical"):
                if key in out_components:
                    out_components[key] *= scale
        reward -= dense_pos_this_step * (1.0 - scale)
        dense_pos_this_step = cap_room

    episode_state.dense_cumulative += dense_pos_this_step
    if out_components is not None:
        out_components["dense_applied"] = dense_pos_this_step
        out_components["dense_decay"] = decay
        out_components["total"] = float(reward)

    return float(reward), episode_state


if __name__ == "__main__":
    es = EpisodeRewardState()

    # Death
    po = {"map": np.zeros((3, 3), dtype=np.int8), "players": np.array([[1, 1, 1, 1, 0]], dtype=np.int8), "bombs": np.zeros((0, 4), dtype=np.int8)}
    co = {"map": np.zeros((3, 3), dtype=np.int8), "players": np.array([[1, 1, 0, 1, 0]], dtype=np.int8), "bombs": np.zeros((0, 4), dtype=np.int8)}
    r, _ = compute_reward_icec(po, co, 0, es)
    assert abs(r - R_AGENT_DEATH) < 1e-9, r

    # Time step only (standing still, alive, no bombs)
    es2 = EpisodeRewardState()
    r, _ = compute_reward_icec(po, po, 0, es2)
    assert abs(r - R_TIME_STEP) < 1e-9, r

    # Movement + closer enemy (two players)
    m = np.zeros((5, 5), dtype=np.int8)
    po3 = {
        "map": m,
        "players": np.array([[1, 2, 1, 1, 0], [3, 2, 1, 1, 0]], dtype=np.int8),
        "bombs": np.zeros((0, 4), dtype=np.int8),
    }
    co3 = {
        "map": m,
        "players": np.array([[2, 2, 1, 1, 0], [3, 2, 1, 1, 0]], dtype=np.int8),
        "bombs": np.zeros((0, 4), dtype=np.int8),
    }
    es3 = EpisodeRewardState()
    r, _ = compute_reward_icec(po3, co3, 0, es3)
    decay1 = math.exp(-1.0 / DENSE_DECAY_TAU_STEPS)
    # time + decayed closer + decayed closest-so-far (step 1)
    want = R_TIME_STEP + (R_CLOSER_STEP + R_CLOSEST_SO_FAR) * decay1
    assert abs(r - want) < 1e-9, (r, want)

    # Oscillation: after reaching best distance, stepping closer again without a new record gives no dense movement bonus.
    es4 = EpisodeRewardState()
    _, es4 = compute_reward_icec(po3, co3, 0, es4)
    assert es4.best_dist == 1.0
    r_b, es4 = compute_reward_icec(co3, po3, 0, es4)
    decay2 = math.exp(-2.0 / DENSE_DECAY_TAU_STEPS)
    want_b = R_TIME_STEP + R_FARTHER_STEP * decay2
    assert abs(r_b - want_b) < 1e-9, (r_b, want_b)
    r_c, es4 = compute_reward_icec(po3, co3, 0, es4)
    # Closer again to dist=1 but not strictly better than best_dist=1 → only time penalty
    assert abs(r_c - R_TIME_STEP) < 1e-9, (r_c, es4.best_dist)

    print("reward_02 self-tests passed.")
