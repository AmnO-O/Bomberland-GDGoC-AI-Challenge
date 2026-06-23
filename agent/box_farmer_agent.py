import random
from collections import deque

import numpy as np


class BoxFarmerAgent:
    """
    Safer box-farming agent.

    Changes vs your version:
    - prefers tiles with earliest explosion time > 7,
    - chooses movement by maximizing safety margin first,
    - bombs only when a real escape route exists,
    - avoids "looks safe next turn but becomes a dead end" moves.
    """

    team_id = "BoxFarmerAgent"

    MOVES = {
        0: (0, 0),
        1: (0, -1),
        2: (0, 1),
        3: (-1, 0),
        4: (1, 0),
    }

    SAFE_TIME_TARGET = 7      # prefer tiles safe for > 7 turns
    MIN_ESCAPE_MARGIN = 2     # stricter than just > 0
    MAX_ESCAPE_DEPTH = 8

    def __init__(self, agent_id: int):
        self.agent_id = int(agent_id)

    def act(self, obs):
        grid = obs["map"]
        players = obs["players"]
        bombs = obs["bombs"]

        if self.agent_id >= len(players) or int(players[self.agent_id][2]) != 1:
            return 0

        my_r = int(players[self.agent_id][0])
        my_c = int(players[self.agent_id][1])
        my_pos = (my_r, my_c)

        bombs_left = int(players[self.agent_id][3])
        bomb_radius = max(1, 1 + int(players[self.agent_id][4]))
        bomb_positions = self._bomb_positions(bombs)

        time_plane = self._time_to_explosion_plane(grid, players, bombs)
        safe_mask = self._safe_mask(grid, bomb_positions, time_plane)

        # If the current tile is already risky, escape first.
        current_margin = self._escape_margin_from(
            grid=grid,
            players=players,
            bombs=bombs,
            start=my_pos,
            bomb_positions=bomb_positions,
            max_depth=self.MAX_ESCAPE_DEPTH,
        )
        if time_plane[my_r, my_c] <= 1 or current_margin < self.MIN_ESCAPE_MARGIN:
            escape = self._best_escape_action(
                grid=grid,
                start=my_pos,
                bombs=bombs,
                players=players,
                bomb_positions=bomb_positions,
                time_plane=time_plane,
            )
            if escape is not None:
                return int(escape)

            fallback = self._best_local_safe_move(
                my_pos=my_pos,
                safe_mask=safe_mask,
                grid=grid,
                bomb_radius=bomb_radius,
                time_plane=time_plane,
            )
            return int(fallback) if fallback is not None else 0

        # Bomb only if it hits boxes and we can still escape safely.
        if bombs_left > 0 and my_pos not in bomb_positions:
            boxes_here = self._count_boxes_in_blast(grid, my_pos, bomb_radius)
            if boxes_here > 0 and self._can_escape_after_placing(
                grid=grid,
                my_pos=my_pos,
                bombs=bombs,
                players=players,
                bomb_positions=bomb_positions,
            ):
                return 5

        # Resource-starved? Prefer items first.
        low_resource = (bombs_left <= 1) or (bomb_radius <= 1)

        item_tiles = self._item_tiles(
            grid,
            prefer_capacity=bombs_left <= 1,
            prefer_radius=int(players[self.agent_id][4]) <= 1,
        )
        box_spots = self._box_bomb_spots(grid, bomb_positions)

        if low_resource:
            move = self._best_move_to_targets(
                grid=grid,
                start=my_pos,
                targets=item_tiles,
                safe_mask=safe_mask,
                score_fn=lambda pos, dist: self._item_score(grid, pos, bombs_left, bomb_radius)
                + self._safety_bonus(time_plane, pos)
                - 0.15 * dist,
            )
            if move is not None:
                return int(move)

            move = self._best_move_to_targets(
                grid=grid,
                start=my_pos,
                targets=box_spots,
                safe_mask=safe_mask,
                score_fn=lambda pos, dist: self._box_spot_score(grid, pos, bomb_radius)
                + self._safety_bonus(time_plane, pos)
                - 0.20 * dist,
            )
            if move is not None:
                return int(move)
        else:
            move = self._best_move_to_targets(
                grid=grid,
                start=my_pos,
                targets=box_spots,
                safe_mask=safe_mask,
                score_fn=lambda pos, dist: self._box_spot_score(grid, pos, bomb_radius)
                + self._safety_bonus(time_plane, pos)
                - 0.20 * dist,
            )
            if move is not None:
                return int(move)

            move = self._best_move_to_targets(
                grid=grid,
                start=my_pos,
                targets=item_tiles,
                safe_mask=safe_mask,
                score_fn=lambda pos, dist: self._item_score(grid, pos, bombs_left, bomb_radius)
                + self._safety_bonus(time_plane, pos)
                - 0.15 * dist,
            )
            if move is not None:
                return int(move)

        best = self._best_local_safe_move(
            my_pos=my_pos,
            safe_mask=safe_mask,
            grid=grid,
            bomb_radius=bomb_radius,
            time_plane=time_plane,
        )
        if best is not None:
            return int(best)

        return 0

    # ------------------------------------------------------------------
    # Basic helpers
    # ------------------------------------------------------------------
    def _next_pos(self, pos, action):
        dr, dc = self.MOVES[int(action)]
        return pos[0] + dr, pos[1] + dc

    def _in_bounds(self, grid, r, c):
        return 0 <= r < grid.shape[0] and 0 <= c < grid.shape[1]

    def _passable(self, grid, r, c):
        return self._in_bounds(grid, r, c) and int(grid[r, c]) in (0, 3, 4)

    def _bomb_positions(self, bombs):
        if bombs is None or len(bombs) == 0:
            return set()
        return {(int(b[0]), int(b[1])) for b in bombs}

    def _blast_tiles(self, grid, bx, by, radius):
        tiles = {(bx, by)}
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            for d in range(1, radius + 1):
                r, c = bx + dr * d, by + dc * d
                if not self._in_bounds(grid, r, c):
                    break
                cell = int(grid[r, c])
                if cell == 1:
                    break
                tiles.add((r, c))
                if cell == 2:
                    break
        return tiles

    def _bomb_radius_for_owner(self, players, owner_id):
        if 0 <= owner_id < len(players) and int(players[owner_id][2]) == 1:
            return max(1, 1 + int(players[owner_id][4]))
        return 1

    # ------------------------------------------------------------------
    # Bomb danger model
    # ------------------------------------------------------------------
    def _effective_bomb_times(self, grid, players, bombs):
        if bombs is None or len(bombs) == 0:
            return np.zeros((0,), dtype=np.int32)

        n = len(bombs)
        times = np.array([max(0, int(b[2])) for b in bombs], dtype=np.int32)
        blasts = []

        for i in range(n):
            owner = int(bombs[i][3]) if bombs.shape[1] > 3 else -1
            radius = self._bomb_radius_for_owner(players, owner)
            blasts.append(self._blast_tiles(grid, int(bombs[i][0]), int(bombs[i][1]), radius))

        q = deque(range(n))
        in_q = [True] * n

        while q:
            i = q.popleft()
            in_q[i] = False
            ti = int(times[i])

            for j in range(n):
                if i == j:
                    continue
                bj = (int(bombs[j][0]), int(bombs[j][1]))
                if bj in blasts[i] and int(times[j]) > ti:
                    times[j] = ti
                    if not in_q[j]:
                        q.append(j)
                        in_q[j] = True

        return times

    def _time_to_explosion_plane(self, grid, players, bombs):
        plane = np.full((grid.shape[0], grid.shape[1]), 9999, dtype=np.int32)
        if bombs is None or len(bombs) == 0:
            return plane

        eff = self._effective_bomb_times(grid, players, bombs)
        for i, b in enumerate(bombs):
            owner = int(b[3]) if bombs.shape[1] > 3 else -1
            radius = self._bomb_radius_for_owner(players, owner)
            bm = self._blast_tiles(grid, int(b[0]), int(b[1]), radius)
            rows, cols = zip(*bm)
            plane[list(rows), list(cols)] = np.minimum(
                plane[list(rows), list(cols)],
                int(max(0, eff[i]))
            )
        return plane

    def _safety_bonus(self, time_plane, pos):
        """
        Bonus for tiles with explosion time > 7.
        The farther above 7, the better.
        """
        t = int(time_plane[pos[0], pos[1]])
        if t <= self.SAFE_TIME_TARGET:
            return 0.0
        return min(3.0, 0.15 * float(t - self.SAFE_TIME_TARGET))

    def _safe_mask(self, grid, bomb_positions, time_plane):
        walkable = np.isin(grid, [0, 3, 4])
        safe = walkable & (time_plane > 1)
        if bomb_positions:
            for r, c in bomb_positions:
                if self._in_bounds(grid, r, c):
                    safe[r, c] = False
        return safe

    def _escape_margin_from(self, grid, players, bombs, start, bomb_positions, max_depth=8):
        plane = self._time_to_explosion_plane(grid, players, bombs)

        q = deque([(start, 0)])
        seen = {start}
        best = -10**9

        while q:
            pos, dist = q.popleft()

            if pos != start and pos in bomb_positions:
                continue

            margin = int(plane[pos[0], pos[1]]) - dist
            best = max(best, margin)

            if dist >= max_depth:
                continue

            for a in (1, 2, 3, 4):
                npos = self._next_pos(pos, a)
                if npos in seen:
                    continue
                if not self._passable(grid, npos[0], npos[1]):
                    continue
                if npos in bomb_positions:
                    continue

                if int(plane[npos[0], npos[1]]) <= dist + 1:
                    continue

                seen.add(npos)
                q.append((npos, dist + 1))

        return -1.0 if best < -1000 else float(best)

    def _best_escape_action(self, grid, start, bombs, players, bomb_positions, time_plane):
        best_action = None
        best_score = -10**9

        for a in (1, 2, 3, 4):
            npos = self._next_pos(start, a)
            if not self._passable(grid, npos[0], npos[1]):
                continue
            if npos in bomb_positions:
                continue

            margin = self._escape_margin_from(
                grid=grid,
                players=players,
                bombs=bombs,
                start=npos,
                bomb_positions=bomb_positions,
                max_depth=self.MAX_ESCAPE_DEPTH,
            )
            score = 2.0 * margin + self._safety_bonus(time_plane, npos)

            if score > best_score:
                best_score = score
                best_action = a

        return best_action

    def _can_escape_after_placing(self, grid, my_pos, bombs, players, bomb_positions):
        hyp_bombs = self._add_hypothetical_bomb(bombs, my_pos, self.agent_id)

        # Need a real escape route from at least one neighboring tile.
        for a in (1, 2, 3, 4):
            npos = self._next_pos(my_pos, a)
            if not self._passable(grid, npos[0], npos[1]):
                continue
            if npos in bomb_positions:
                continue

            margin = self._escape_margin_from(
                grid=grid,
                players=players,
                bombs=hyp_bombs,
                start=npos,
                bomb_positions=bomb_positions | {my_pos},
                max_depth=self.MAX_ESCAPE_DEPTH,
            )
            if margin >= self.MIN_ESCAPE_MARGIN:
                return True

        return False

    def _add_hypothetical_bomb(self, bombs, pos, owner_id, timer=7):
        row = np.array([[pos[0], pos[1], timer, owner_id]], dtype=np.int8)
        if bombs is None or len(bombs) == 0:
            return row
        return np.concatenate([bombs, row], axis=0)

    # ------------------------------------------------------------------
    # Item / box targeting
    # ------------------------------------------------------------------
    def _item_tiles(self, grid, prefer_capacity=False, prefer_radius=False):
        preferred_values = set()
        if prefer_radius:
            preferred_values.add(3)
        if prefer_capacity:
            preferred_values.add(4)

        preferred = {
            (x, y)
            for x in range(grid.shape[0])
            for y in range(grid.shape[1])
            if int(grid[x, y]) in preferred_values
        }
        if preferred:
            return preferred

        return {
            (x, y)
            for x in range(grid.shape[0])
            for y in range(grid.shape[1])
            if int(grid[x, y]) in (3, 4)
        }

    def _count_boxes_in_blast(self, grid, my_pos, radius):
        return sum(
            1
            for x, y in self._blast_tiles(grid, my_pos[0], my_pos[1], radius)
            if int(grid[x, y]) == 2
        )

    def _box_bomb_spots(self, grid, bomb_positions):
        spots = set()
        for x in range(grid.shape[0]):
            for y in range(grid.shape[1]):
                if int(grid[x, y]) != 2:
                    continue
                for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nx, ny = x + dx, y + dy
                    if self._passable(grid, nx, ny) and (nx, ny) not in bomb_positions:
                        spots.add((nx, ny))
        return spots

    def _item_score(self, grid, pos, bombs_left, bomb_radius):
        tile = int(grid[pos[0], pos[1]])
        if tile == 3:
            return 3.0 if bomb_radius <= 2 else 1.5
        if tile == 4:
            return 3.0 if bombs_left <= 1 else 1.5
        return 1.0

    def _box_spot_score(self, grid, pos, bomb_radius):
        return float(self._count_boxes_in_blast(grid, pos, bomb_radius))

    def _best_move_to_targets(self, grid, start, targets, safe_mask, score_fn, max_depth=12):
        if not targets:
            return None

        q = deque([(start, 0, None)])
        seen = {start}
        best_actions = []
        best_score = -10**9

        while q:
            pos, dist, first_action = q.popleft()

            if pos in targets and first_action is not None:
                score = float(score_fn(pos, dist))
                if score > best_score + 1e-9:
                    best_score = score
                    best_actions = [first_action]
                elif abs(score - best_score) <= 1e-9:
                    best_actions.append(first_action)

            if dist >= max_depth:
                continue

            for a in (1, 2, 3, 4):
                npos = self._next_pos(pos, a)
                if npos in seen:
                    continue
                if not self._passable(grid, npos[0], npos[1]):
                    continue
                if not safe_mask[npos[0], npos[1]]:
                    continue

                seen.add(npos)
                q.append((npos, dist + 1, a if first_action is None else first_action))

        if not best_actions:
            return None
        return random.choice(best_actions)

    def _best_local_safe_move(self, my_pos, safe_mask, grid=None, bomb_radius=1, time_plane=None):
        candidates = []
        box_coords = None
        if grid is not None:
            box_coords = np.argwhere(np.array(grid) == 2)

        for a in (1, 2, 3, 4):
            npos = self._next_pos(my_pos, a)
            if not self._in_bounds(safe_mask, npos[0], npos[1]):
                continue
            if not safe_mask[npos[0], npos[1]]:
                continue

            score = 0.0
            if grid is not None:
                score += 1.5 * self._box_spot_score(grid, npos, bomb_radius)

                if len(box_coords) > 0:
                    nearest_box = min(
                        abs(int(r) - npos[0]) + abs(int(c) - npos[1])
                        for r, c in box_coords
                    )
                    score += max(0.0, 2.0 - 0.10 * nearest_box)

            if time_plane is not None:
                t = int(time_plane[npos[0], npos[1]])
                if t > self.SAFE_TIME_TARGET:
                    score += 0.5 * min(10.0, float(t - self.SAFE_TIME_TARGET))
                else:
                    score -= 1.0

            candidates.append((score, a))

        if not candidates:
            return None

        best_score = max(s for s, _ in candidates)
        best = [a for s, a in candidates if abs(s - best_score) <= 1e-9]
        return random.choice(best) if best else None