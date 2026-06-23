"""
agent.py — V8-fix value-scorer hẹp (sinh tự động bởi model_v8/fix_v8/build_agent_v8.py).
KHÔNG sửa tay. Sửa anh5/submission_v6.py / features.py / numpy_forward.py rồi build lại.
Sàn an toàn: thiếu model_value.pkl hoặc USE_NET=False ⇒ chạy y hệt v6 rule.
"""
"""
agent.py -- Bomberland Rule-Based Agent v4
Base: submission_v3 (v6 core + _project_future_state idea).

Root cause of "agent stands still" in v3:
  _project_future_state assumes ALL live armed enemies place a bomb every
  single step simultaneously.  This floods future_danger_any with half the
  map every turn, so _escape_timed can never find a "safe" cell, making
  _action_survives_future return False for EVERY action.  Priorities 2, 3,
  and 5 all require _action_survives_future → all fall through → only
  _safe_fallback runs → agent oscillates / stands still.

Fixes applied (surgical — minimum changes to the working v3 base):

  FIX-1  _action_survives_future: enemies only place hypothetical bombs if
         they are within BFS_THREAT_RADIUS walkable steps.  Far-away enemies
         flooding the future map is the #1 cause of paralysis.

  FIX-2  _action_survives_future: the "survivable continuation" check uses
         a dedicated _can_reach_safety BFS (returns bool) instead of
         _escape_timed (returns action), which avoids the ambiguity of
         "None means unreachable" vs "None because already safe".

  FIX-3  _reachable_safe_timed replaces _reachable_safe_count in
         _safe_fallback.  Each cell at BFS depth d is compared against its
         earliest explosion time:
           SAFE : explode_time > d or never dangerous  → +REWARD_SAFE (10)
           TEMP : explode_time <= d (corridor closes)   → +REWARD_TEMP  (3)
           DEAD : would explode at arrival (pruned)
           Dead-end leaf (no onward moves, d < depth)   → -REWARD_SAFE
         Per-cell min-explode-time dict built once by _build_danger_timed
         for O(1) lookup inside BFS.

  FIX-4  _build_danger_timed now also returns cell_explode_min dict.
         Signature: returns (danger_by_time, danger_any, cell_explode_min).
         All callers updated.

  FIX-5  Combat detection switched from Manhattan ≤2 to BFS ≤3 through
         walkable cells (_count_nearby_armed_bfs already existed, just
         wasn't being used for combat_mode).  Radius lowered 4→3 to avoid
         over-triggering in opening game.

  FIX-6  _safe_fallback repeat penalty raised 15→20 so it stays dominant
         over the reach score (no ×10 multiplier added).

Everything else (C1–C4, _can_escape_after_bomb, _count_escape_first_moves,
_bfs_timed, _bfs_escape, _escape_timed, _project_future_state structure,
all priority logic) is unchanged from v3.
"""
import time
from collections import deque
import numpy as np

# ---------------------------------------------------------------------------
# Action / map / game constants  (unchanged)
# ---------------------------------------------------------------------------
STOP, LEFT, RIGHT, UP, DOWN, PLACE_BOMB = 0, 1, 2, 3, 4, 5

MOVES = {
    STOP:  ( 0,  0),
    LEFT:  (-1,  0),
    RIGHT: ( 1,  0),
    UP:    ( 0, -1),
    DOWN:  ( 0,  1),
}
MOVE_ACTIONS = [LEFT, RIGHT, UP, DOWN]

GRASS, WALL, BOX, ITEM_RADIUS, ITEM_CAPACITY = 0, 1, 2, 3, 4

BOMB_TIMER   = 7
MAX_RADIUS   = 5
MAX_CAPACITY = 5

TIME_BUDGET_S             = 0.070
BFS_DEPTH_CAP             = 30
ESCAPE_DEPTH              = 25
BOMB_MIN_SCORE            = 1
BOMB_COMBAT_CLUSTER_SCORE = 4   # C2

# FIX-5: BFS radius for combat_mode detection (replaces Manhattan 2)
BFS_COMBAT_RADIUS = 3

# FIX-1: only enemies within this BFS radius are assumed to bomb in future sim
BFS_THREAT_RADIUS = 20

# FIX-3: tiered rewards for _reachable_safe_timed
REWARD_SAFE = 10
REWARD_TEMP =  3
SAFE_DEPTH  =  6


# ===========================================================================
# Geometry helpers  (unchanged)
# ===========================================================================

def _blast_tiles(bx, by, radius, grid):
    h, w  = grid.shape
    tiles = {(bx, by)}
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        for r in range(1, radius + 1):
            tr, tc = bx + dr * r, by + dc * r
            if not (0 <= tr < h and 0 <= tc < w):
                break
            cell = grid[tr, tc]
            if cell == WALL:
                break
            tiles.add((tr, tc))
            if cell == BOX:
                break
    return tiles


def _walkable(r, c, grid, bomb_pos):
    h, w = grid.shape
    if not (0 <= r < h and 0 <= c < w):
        return False
    if grid[r, c] in (WALL, BOX):
        return False
    if (r, c) in bomb_pos:
        return False
    return True


# ===========================================================================
# Timer-aware danger map  (FIX-4: also returns cell_explode_min)
# ===========================================================================

def _build_danger_timed(obs):
    """
    Returns (danger_by_time, danger_any, cell_explode_min).
    danger_by_time[t] = set of tiles exploding at step t.
    danger_any        = union of all danger tiles.
    cell_explode_min  = {cell: earliest_explosion_t}  — O(1) lookup for FIX-3.
    Chain reaction: bomb B inside blast of A → B.timer = min(A, B).
    """
    grid    = obs["map"]
    bombs_a = obs["bombs"]
    players = obs["players"]

    if len(bombs_a) == 0:
        return {}, set(), {}

    n = len(players)
    bomb_list = []
    for b in bombs_a:
        bx, by, timer, oid = int(b[0]), int(b[1]), int(b[2]), int(b[3])
        oid_s  = max(0, min(n - 1, oid))
        radius = max(1, min(MAX_RADIUS, 1 + int(players[oid_s][4])))
        tiles  = _blast_tiles(bx, by, radius, grid)
        bomb_list.append({'pos': (bx, by), 'timer': timer, 'tiles': tiles})

    changed = True
    while changed:
        changed = False
        for i, b1 in enumerate(bomb_list):
            for j, b2 in enumerate(bomb_list):
                if i == j:
                    continue
                if b2['pos'] in b1['tiles'] and b1['timer'] < b2['timer']:
                    b2['timer'] = b1['timer']
                    changed = True

    danger_by_time   = {}
    danger_any       = set()
    cell_explode_min = {}
    for b in bomb_list:
        t = b['timer']
        if t not in danger_by_time:
            danger_by_time[t] = set()
        danger_by_time[t] |= b['tiles']
        danger_any        |= b['tiles']
        for cell in b['tiles']:
            if cell not in cell_explode_min or t < cell_explode_min[cell]:
                cell_explode_min[cell] = t

    return danger_by_time, danger_any, cell_explode_min


# ===========================================================================
# Timer-aware BFS  (unchanged from v3)
# ===========================================================================

def _bfs_timed(start, targets, grid, bomb_pos, danger_by_time,
               depth_cap=BFS_DEPTH_CAP):
    sr, sc = start
    if start in targets:
        return STOP

    visited = {(sr, sc)}
    queue   = deque()

    for a in MOVE_ACTIONS:
        dr, dc = MOVES[a]
        nr, nc = sr + dr, sc + dc
        if not _walkable(nr, nc, grid, bomb_pos):
            continue
        if (nr, nc) in visited:
            continue
        if (nr, nc) in danger_by_time.get(1, set()):
            continue
        visited.add((nr, nc))
        queue.append((nr, nc, a, 1))

    while queue:
        r, c, first_a, t = queue.popleft()
        if (r, c) in targets:
            return first_a
        if t >= depth_cap:
            continue
        for a in MOVE_ACTIONS:
            dr, dc = MOVES[a]
            nr, nc = r + dr, c + dc
            t_next = t + 1
            if not _walkable(nr, nc, grid, bomb_pos):
                continue
            if (nr, nc) in visited:
                continue
            if (nr, nc) in danger_by_time.get(t_next, set()):
                continue
            visited.add((nr, nc))
            queue.append((nr, nc, first_a, t_next))

    return None


def _bfs_escape(start, targets, grid, bomb_pos, danger_by_time,
                depth_cap=ESCAPE_DEPTH):
    sr, sc = start
    if start in targets:
        return STOP

    visited = {(sr, sc, 0)}
    queue   = deque([(sr, sc, None, 0)])

    while queue:
        r, c, first_a, t = queue.popleft()
        if t >= depth_cap:
            continue
        t_next      = t + 1
        danger_next = danger_by_time.get(t_next, set())

        for a in (LEFT, RIGHT, UP, DOWN, STOP):
            if a == STOP:
                nr, nc = r, c
            else:
                dr, dc = MOVES[a]
                nr, nc = r + dr, c + dc
                if not _walkable(nr, nc, grid, bomb_pos):
                    continue
            if (nr, nc, t_next) in visited:
                continue
            if (nr, nc) in danger_next:
                continue
            fa = a if first_a is None else first_a
            if (nr, nc) in targets:
                return fa
            visited.add((nr, nc, t_next))
            queue.append((nr, nc, fa, t_next))

    return None


# ===========================================================================
# Escape  (unchanged from v3 — returns None only when truly cornered)
# ===========================================================================

def _escape_timed(pos, grid, bomb_pos, danger_by_time, danger_any):
    h, w = grid.shape

    safe = set()
    for rr in range(h):
        for cc in range(w):
            if (grid[rr, cc] not in (WALL, BOX)
                    and (rr, cc) not in bomb_pos
                    and (rr, cc) not in danger_any):
                safe.add((rr, cc))

    if safe:
        a = _bfs_escape(pos, safe, grid, bomb_pos, danger_by_time,
                        depth_cap=ESCAPE_DEPTH)
        if a is not None:
            return a

    urgent     = danger_by_time.get(1, set()) | danger_by_time.get(2, set())
    less_risky = set()
    for rr in range(h):
        for cc in range(w):
            if (grid[rr, cc] not in (WALL, BOX)
                    and (rr, cc) not in bomb_pos
                    and (rr, cc) not in urgent):
                less_risky.add((rr, cc))

    if less_risky:
        sr, sc  = pos
        visited = {(sr, sc)}
        queue   = deque()
        for a in MOVE_ACTIONS:
            dr, dc = MOVES[a]
            nr, nc = sr + dr, sc + dc
            if _walkable(nr, nc, grid, bomb_pos) and (nr, nc) not in visited:
                visited.add((nr, nc))
                queue.append((nr, nc, a))
        while queue:
            r, c, first_a = queue.popleft()
            if (r, c) in less_risky:
                return first_a
            for a in MOVE_ACTIONS:
                dr, dc = MOVES[a]
                nr, nc = r + dr, c + dc
                if _walkable(nr, nc, grid, bomb_pos) and (nr, nc) not in visited:
                    visited.add((nr, nc))
                    queue.append((nr, nc, first_a))

    return None


# ===========================================================================
# Bomb helpers  (unchanged from v3)
# ===========================================================================

def _enemy_predicted_blast(my_r, my_c, players, agent_id, grid):
    """C1: blast union for armed enemies within Manhattan 2."""
    predicted = set()
    for i, p in enumerate(players):
        if i == agent_id or int(p[2]) != 1 or int(p[3]) <= 0:
            continue
        er, ec = int(p[0]), int(p[1])
        if abs(er - my_r) + abs(ec - my_c) > 2:
            continue
        radius = max(1, min(MAX_RADIUS, 1 + int(p[4])))
        predicted |= _blast_tiles(er, ec, radius, grid)
    return predicted


def _can_escape_after_bomb(my_r, my_c, my_radius, grid, bomb_pos,
                            danger_by_time, extra_danger_tiles=None):
    new_blast  = _blast_tiles(my_r, my_c, my_radius, grid)
    new_bomb_p = set(bomb_pos) | {(my_r, my_c)}

    eff_t = BOMB_TIMER - 1
    for t, tiles in danger_by_time.items():
        if (my_r, my_c) in tiles and t < eff_t:
            eff_t = t

    mod = {t: set(s) for t, s in danger_by_time.items()}
    if eff_t not in mod:
        mod[eff_t] = set()
    mod[eff_t] |= new_blast
    if extra_danger_tiles:
        mod[eff_t] |= extra_danger_tiles

    new_danger_any = set()
    for tiles in mod.values():
        new_danger_any |= tiles

    h, w = grid.shape
    safe_cells = set()
    for rr in range(h):
        for cc in range(w):
            if (grid[rr, cc] not in (WALL, BOX)
                    and (rr, cc) not in new_bomb_p
                    and (rr, cc) not in new_danger_any):
                safe_cells.add((rr, cc))

    if not safe_cells:
        return False

    visited = {(my_r, my_c, 0)}
    queue   = deque([(my_r, my_c, 0)])

    while queue:
        r, c, t = queue.popleft()
        if (r, c) in safe_cells:
            return True
        if t >= eff_t:
            continue
        t_next      = t + 1
        danger_next = mod.get(t_next, set())
        for a in (LEFT, RIGHT, UP, DOWN, STOP):
            if a == STOP:
                nr, nc = r, c
            else:
                dr, dc = MOVES[a]
                nr, nc = r + dr, c + dc
                if not _walkable(nr, nc, grid, new_bomb_p):
                    continue
            if (nr, nc, t_next) in visited:
                continue
            if (nr, nc) in danger_next:
                continue
            visited.add((nr, nc, t_next))
            queue.append((nr, nc, t_next))

    return False


def _count_escape_first_moves(my_r, my_c, my_radius, grid, bomb_pos,
                               danger_by_time, extra_danger_tiles=None):
    new_blast  = _blast_tiles(my_r, my_c, my_radius, grid)
    new_bomb_p = set(bomb_pos) | {(my_r, my_c)}

    eff_t = BOMB_TIMER - 1
    for t, tiles in danger_by_time.items():
        if (my_r, my_c) in tiles and t < eff_t:
            eff_t = t

    mod = {t: set(s) for t, s in danger_by_time.items()}
    if eff_t not in mod:
        mod[eff_t] = set()
    mod[eff_t] |= new_blast
    if extra_danger_tiles:
        mod[eff_t] |= extra_danger_tiles

    new_danger_any = set()
    for tiles in mod.values():
        new_danger_any |= tiles

    h, w = grid.shape
    safe_cells = set()
    for rr in range(h):
        for cc in range(w):
            if (grid[rr, cc] not in (WALL, BOX)
                    and (rr, cc) not in new_bomb_p
                    and (rr, cc) not in new_danger_any):
                safe_cells.add((rr, cc))

    if not safe_cells:
        return 0

    count = 0
    for first_a in (LEFT, RIGHT, UP, DOWN, STOP):
        if first_a == STOP:
            r1, c1 = my_r, my_c
        else:
            dr, dc = MOVES[first_a]
            r1, c1 = my_r + dr, my_c + dc
            if not _walkable(r1, c1, grid, new_bomb_p):
                continue

        if (r1, c1) in mod.get(1, set()):
            continue
        if (r1, c1) in safe_cells:
            count += 1
            continue

        visited2 = {(r1, c1, 1)}
        queue2   = deque([(r1, c1, 1)])
        found    = False
        while queue2 and not found:
            r, c, t = queue2.popleft()
            if t >= eff_t:
                continue
            t_next      = t + 1
            danger_next = mod.get(t_next, set())
            for a2 in (LEFT, RIGHT, UP, DOWN, STOP):
                if a2 == STOP:
                    nr, nc = r, c
                else:
                    dr2, dc2 = MOVES[a2]
                    nr, nc   = r + dr2, c + dc2
                    if not _walkable(nr, nc, grid, new_bomb_p):
                        continue
                if (nr, nc, t_next) in visited2:
                    continue
                if (nr, nc) in danger_next:
                    continue
                if (nr, nc) in safe_cells:
                    found = True
                    break
                visited2.add((nr, nc, t_next))
                queue2.append((nr, nc, t_next))
        if found:
            count += 1

    return count


# ===========================================================================
# FIX-3: Timer-aware reachable safe score
# ===========================================================================

def _reachable_safe_timed(pos, grid, bomb_pos, cell_explode_min,
                           danger_by_time, depth=SAFE_DEPTH):
    """
    BFS up to `depth` steps from pos, tracking arrival time d.
    Each visited cell classified using cell_explode_min (O(1) lookup):
      SAFE : explode_time > d  OR  never dangerous  → +REWARD_SAFE
      TEMP : explode_time <= d (corridor closes)    → +REWARD_TEMP
      Prune: cell in danger_by_time[d] (explodes on arrival, skip)
      Dead-end leaf (no forward moves, d < depth)   → -REWARD_SAFE
    Returns float score. Higher = more open safe space ahead.
    """
    visited = {pos}
    queue   = deque([(pos[0], pos[1], 0)])
    score   = 0.0

    while queue:
        r, c, d = queue.popleft()

        expl_t = cell_explode_min.get((r, c))
        if expl_t is None or expl_t > d:
            score += REWARD_SAFE
        else:
            score += REWARD_TEMP

        if d >= depth:
            continue

        d_next     = d + 1
        danger_nxt = danger_by_time.get(d_next, set())
        has_fwd    = False

        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if (nr, nc) in visited:
                continue
            if not _walkable(nr, nc, grid, bomb_pos):
                continue
            if (nr, nc) in danger_nxt:
                # Would explode exactly when we step there — don't go there
                continue
            has_fwd = True
            visited.add((nr, nc))
            queue.append((nr, nc, d_next))

        if not has_fwd:
            score -= REWARD_SAFE   # dead-end penalty

    return score


# ===========================================================================
# Combat detection  (FIX-5: use BFS, not Manhattan)
# ===========================================================================

def _count_nearby_armed_bfs(my_r, my_c, players, agent_id, grid,
                             radius=BFS_COMBAT_RADIUS):
    """Count live armed enemies reachable within `radius` BFS steps."""
    armed_enemy_pos = set()
    for i, p in enumerate(players):
        if i == agent_id or int(p[2]) != 1 or int(p[3]) <= 0:
            continue
        armed_enemy_pos.add((int(p[0]), int(p[1])))

    if not armed_enemy_pos:
        return 0

    h, w    = grid.shape
    visited = {(my_r, my_c)}
    queue   = deque([(my_r, my_c, 0)])
    count   = 0

    while queue:
        r, c, d = queue.popleft()
        if (r, c) in armed_enemy_pos:
            count += 1
        if d >= radius:
            continue
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if (nr, nc) in visited:
                continue
            if not (0 <= nr < h and 0 <= nc < w):
                continue
            if grid[nr, nc] in (WALL, BOX):
                continue
            visited.add((nr, nc))
            queue.append((nr, nc, d + 1))

    return count


# ===========================================================================
# Manhattan proximity (kept for _enemy_adjacent and _enemy_predicted_blast)
# ===========================================================================

def _count_nearby_armed_manhattan(my_r, my_c, players, agent_id):
    """Count live armed enemies within Manhattan 2 (kept for C1/C2 guard)."""
    count = 0
    for i, p in enumerate(players):
        if i == agent_id or int(p[2]) != 1 or int(p[3]) <= 0:
            continue
        if abs(int(p[0]) - my_r) + abs(int(p[1]) - my_c) <= 2:
            count += 1
    return count


def _enemy_adjacent(my_r, my_c, players, agent_id):
    """True if any live armed enemy is within Manhattan 2."""
    for i, p in enumerate(players):
        if i == agent_id or int(p[2]) != 1 or int(p[3]) <= 0:
            continue
        if abs(int(p[0]) - my_r) + abs(int(p[1]) - my_c) <= 2:
            return True
    return False


# ===========================================================================
# Bomb value  (unchanged from v3)
# ===========================================================================

def _bomb_value(my_r, my_c, my_radius, grid, players, agent_id, danger_any):
    blast   = _blast_tiles(my_r, my_c, my_radius, grid)
    boxes   = sum(1 for r, c in blast if grid[r, c] == BOX)
    enemies = sum(
        1 for i in range(len(players))
        if i != agent_id
        and int(players[i][2]) == 1
        and (int(players[i][0]), int(players[i][1])) in blast
    )
    
    score = boxes * 1 + enemies * 3 
    if not any(grid[r][c] == BOX for r in range(13) for c in range(13)):
        score += 1
    return score, boxes > 0, enemies > 0


# ===========================================================================
# FIX-1 + FIX-2: future state projection and survival check
# ===========================================================================

def _resolve_chain_reaction(bomb_list):
    changed = True
    while changed:
        changed = False
        for i, b1 in enumerate(bomb_list):
            for j, b2 in enumerate(bomb_list):
                if i == j:
                    continue
                if b2["pos"] in b1["tiles"] and b1["timer"] < b2["timer"]:
                    b2["timer"] = b1["timer"]
                    changed = True
    return bomb_list


def _project_future_state(obs, agent_id, our_action, my_r, my_c,
                           my_radius, nearby_threat_pos):
    """
    FIX-1: Project ONE tick ahead.
    Enemies only place hypothetical bombs if they are in nearby_threat_pos
    (within BFS_THREAT_RADIUS walkable steps).  Far enemies are NOT assumed
    to bomb — that was flooding future_danger_any and paralysing all movement.

    Returns (future_pos, future_bomb_pos, future_dbt, future_da).
    """
    grid    = obs["map"]
    bombs_a = obs["bombs"]
    players = obs["players"]
    n       = len(players)

    current_bomb_pos = {(int(b[0]), int(b[1])) for b in bombs_a}

    # Our future position
    if our_action in MOVE_ACTIONS:
        dr, dc = MOVES[our_action]
        nr, nc = my_r + dr, my_c + dc
        future_pos = (nr, nc) if _walkable(nr, nc, grid, current_bomb_pos) \
                               else (my_r, my_c)
    else:
        future_pos = (my_r, my_c)

    # Tick existing bombs down by 1
    future_bombs = []
    for b in bombs_a:
        bx, by, timer, oid = int(b[0]), int(b[1]), int(b[2]), int(b[3])
        oid_s  = max(0, min(n - 1, oid))
        radius = max(1, min(MAX_RADIUS, 1 + int(players[oid_s][4])))
        future_bombs.append({
            "pos":   (bx, by),
            "timer": max(0, timer - 1),
            "tiles": _blast_tiles(bx, by, radius, grid),
        })

    # FIX-1: only nearby enemies place hypothetical bombs
    for i, p in enumerate(players):
        if i == agent_id or int(p[2]) != 1 or int(p[3]) <= 0:
            continue
        ep = (int(p[0]), int(p[1]))
        if ep not in nearby_threat_pos:          # ← key change
            continue
        if ep in current_bomb_pos:
            continue
        radius = max(1, min(MAX_RADIUS, 1 + int(p[4])))
        future_bombs.append({
            "pos":   ep,
            "timer": BOMB_TIMER - 1,
            "tiles": _blast_tiles(ep[0], ep[1], radius, grid),
        })

    # Our own bomb if PLACE_BOMB
    if (our_action == PLACE_BOMB
            and int(players[agent_id][3]) > 0
            and (my_r, my_c) not in current_bomb_pos):
        future_bombs.append({
            "pos":   (my_r, my_c),
            "timer": BOMB_TIMER - 1,
            "tiles": _blast_tiles(my_r, my_c, my_radius, grid),
        })

    if not future_bombs:
        return future_pos, current_bomb_pos, {}, set(), {}

    future_bombs = _resolve_chain_reaction(future_bombs)

    future_bomb_pos = set()
    future_dbt      = {}
    future_da       = set()
    future_cell_explode_min = {}       

    for b in future_bombs:
        t = b["timer"]
        future_bomb_pos.add(b["pos"])
        future_dbt.setdefault(t, set()).update(b["tiles"])
        future_da.update(b["tiles"])
        for cell in b["tiles"]:
            if cell not in future_cell_explode_min or t < future_cell_explode_min[cell]:
                future_cell_explode_min[cell] = t

    return future_pos, future_bomb_pos, future_dbt, future_da, future_cell_explode_min


def _can_reach_safety(start, grid, bomb_pos, danger_by_time, danger_any,
                      depth=ESCAPE_DEPTH):
    """
    FIX-2: Pure bool BFS — True iff `start` can reach a cell outside
    danger_any within `depth` steps.  No action needed; no fallback to
    random move; no ambiguity.
    """
    # Already safe
    if start not in danger_any:
        return True

    h, w = grid.shape
    safe_cells = {(r, c) for r in range(h) for c in range(w)
                  if grid[r, c] not in (WALL, BOX)
                  and (r, c) not in bomb_pos
                  and (r, c) not in danger_any}
    if not safe_cells:
        return False

    sr, sc  = start
    visited = {(sr, sc, 0)}
    queue   = deque([(sr, sc, 0)])

    while queue:
        r, c, t = queue.popleft()
        if (r, c) in safe_cells:
            return True
        if t >= depth:
            continue
        t_next      = t + 1
        danger_next = danger_by_time.get(t_next, set())
        for a in (LEFT, RIGHT, UP, DOWN, STOP):
            if a == STOP:
                nr, nc = r, c
            else:
                dr, dc = MOVES[a]
                nr, nc = r + dr, c + dc
                if not _walkable(nr, nc, grid, bomb_pos):
                    continue
            if (nr, nc, t_next) in visited:
                continue
            if (nr, nc) in danger_next:
                continue
            visited.add((nr, nc, t_next))
            queue.append((nr, nc, t_next))

    return False


def _action_survives_future(future_pos, grid, future_bomb_pos,
                             future_dbt, future_da):
    """
    FIX-2: True if future_pos is not killed at tick 0 AND can reach safety.
    Uses _can_reach_safety (bool BFS) instead of _escape_timed (action BFS).
    """
    # Immediate death: future_pos is in a zone exploding at timer=0
    if future_pos in future_dbt.get(0, set()):
        return False
    return _can_reach_safety(future_pos, grid, future_bomb_pos,
                              future_dbt, future_da)


# ===========================================================================
# FIX-1 helper: build the set of enemy positions within BFS_THREAT_RADIUS
# ===========================================================================

def _nearby_threat_positions(my_r, my_c, players, agent_id, grid,
                              radius=BFS_THREAT_RADIUS):
    """
    Return set of (r,c) of live armed enemies reachable within `radius`
    walkable BFS steps from (my_r, my_c).
    """
    armed = {(int(p[0]), int(p[1]))
             for i, p in enumerate(players)
             if i != agent_id and int(p[2]) == 1 and int(p[3]) > 0}
    if not armed:
        return set()

    h, w    = grid.shape
    visited = {(my_r, my_c)}
    queue   = deque([(my_r, my_c, 0)])
    result  = set()

    while queue:
        r, c, d = queue.popleft()
        if (r, c) in armed:
            result.add((r, c))
        if d >= radius:
            continue
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if (nr, nc) not in visited and 0 <= nr < h and 0 <= nc < w \
                    and grid[nr, nc] not in (WALL, BOX):
                visited.add((nr, nc))
                queue.append((nr, nc, d + 1))

    return result


# ===========================================================================
# Safe fallback  (FIX-3: _reachable_safe_timed; FIX-6: repeat_pen=20)
# ===========================================================================

def _safe_fallback(pos, grid, bomb_pos, danger_by_time, cell_explode_min,
                   step_count, agent_id, last_pos=None):
    sr, sc     = pos
    imm_danger = danger_by_time.get(1, set())
    any_danger = set()
    for tiles in danger_by_time.values():
        any_danger |= tiles

    def _score(r, c, is_stop, a):
        reach  = _reachable_safe_timed(
            (r, c), grid, bomb_pos, cell_explode_min, danger_by_time,
            depth=SAFE_DEPTH)
        open_n = sum(
            1 for dr2, dc2 in ((-1, 0), (1, 0), (0, -1), (0, 1))
            if _walkable(r + dr2, c + dc2, grid, bomb_pos)
        )
        penalty    = REWARD_SAFE if (r, c) in any_danger else 0
        repeat_pen = 20 if (not is_stop) and last_pos is not None \
                          and (r, c) == last_pos else 0   # FIX-6
        tb = (step_count * 13 + agent_id * 7 + r * 5 + c * 3 + a * 11) % 97
        return reach + open_n - penalty - repeat_pen, open_n, tb

    candidates          = []
    dead_end_candidates = []

    if (sr, sc) not in imm_danger:
        s, o, tb = _score(sr, sc, True, STOP)
        candidates.append((STOP, s, True, o, tb))

    for a in MOVE_ACTIONS:
        dr, dc = MOVES[a]
        nr, nc = sr + dr, sc + dc
        if not _walkable(nr, nc, grid, bomb_pos):
            continue
        if (nr, nc) in imm_danger:
            continue
        s, o, tb = _score(nr, nc, False, a)
        if o <= 1:
            dead_end_candidates.append((a, s, False, o, tb))
        else:
            candidates.append((a, s, False, o, tb))

    if not candidates:
        candidates = dead_end_candidates
    if not candidates:
        return STOP

    best_score = max(c[1] for c in candidates)
    best       = [c for c in candidates if c[1] == best_score]
    non_stop   = [c for c in best if not c[2]]
    pool       = non_stop if non_stop else best
    pool.sort(key=lambda x: x[4])
    return pool[0][0]


def _reachable_cells_with_dist(start, grid, bomb_pos, danger_by_time, depth_cap=BFS_DEPTH_CAP):
    """
    Time-aware BFS from start.
    Returns a dict: (r, c) -> shortest safe distance from start.
    Only walks through cells that are currently walkable and not exploding
    at the arrival time.
    """
    sr, sc = start
    visited = {(sr, sc)}
    dist_map = {(sr, sc): 0}
    q = deque([(sr, sc, 0)])

    while q:
        r, c, d = q.popleft()
        if d >= depth_cap:
            continue

        d_next = d + 1
        danger_next = danger_by_time.get(d_next, set())

        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if (nr, nc) in visited:
                continue
            if not _walkable(nr, nc, grid, bomb_pos):
                continue
            if (nr, nc) in danger_next:
                continue

            visited.add((nr, nc))
            dist_map[(nr, nc)] = d_next
            q.append((nr, nc, d_next))

    return dist_map


def _box_farm_score(
    r, c, grid, players, bombs_a, my_id,
    my_r, my_c, my_radius, bomb_pos, danger_any, bfs_dist
):
    """
    Higher score = better box farm spot.

    local spots get a bonus because they are more certain:
    - you can reach them sooner
    - you can keep collecting nearby boxes/items on the way
    - far spots are only worth it if they are clearly better
    """
    if grid[r, c] in (WALL, BOX) or (r, c) in bomb_pos:
        return -1e9

    blast = _blast_tiles(r, c, my_radius, grid)

    boxes_hit = sum(1 for x, y in blast if grid[x, y] == BOX)
    safe_boxes = sum(1 for x, y in blast if grid[x, y] == BOX and (x,y) not in danger_any)

    if boxes_hit == 0:
        return -1e9

    # How many escape routes around this bomb spot?
    open_exits = 0
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        nr, nc = r + dr, c + dc
        if 0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1]:
            if grid[nr, nc] not in (WALL, BOX) and (nr, nc) not in bomb_pos:
                open_exits += 1

    # Nearby enemy pressure
    theft_risk = 0
    for pid, p in enumerate(players):
        if pid == my_id or int(p[2]) != 1:
            continue
        for (x, y) in blast:
            if grid[x, y] == BOX:
                d_enemy_box = abs(int(p[0]) - x) + abs(int(p[1]) - y)
                if d_enemy_box <= 2:     
                    theft_risk += 3
                elif d_enemy_box <= 3:
                    theft_risk += 1

    # Nearby bombs mean the area may be contested / unsafe
    bomb_risk = 0
    for b in bombs_a:
        bx, by, timer, oid = int(b[0]), int(b[1]), int(b[2]), int(b[3])
        if oid == my_id:
            continue

        if 0 <= oid < len(players):
            radius = max(1, min(MAX_RADIUS, 1 + int(players[oid][4])))
        else:
            radius = 2

        for (x, y) in blast:
            if grid[x, y] == BOX and abs(x - bx) + abs(y - by) <= radius:
                if timer <= bfs_dist + abs(x - r) + abs(y - c) + 4:  
                    bomb_risk += 2

    # Prefer local, certain opportunities.
    # Farther spots are possible, but must be clearly better.
    local_bonus = 2.0 if bfs_dist <= 2 else (1.0 if bfs_dist <= 4 else 0.0)
    travel_penalty = 0.65 * bfs_dist + 0.06 * (bfs_dist ** 2)

    danger_pen = 2.5 if (r, c) in danger_any else 0.0

    score = (
        4.5 * safe_boxes +
        1.2 * open_exits +
        local_bonus +
        0.6 * boxes_hit * max(0, 4 - bfs_dist) -
        0.9 * theft_risk -
        0.9 * bomb_risk -
        travel_penalty -
        danger_pen
    )
    return score


# ===========================================================================
# Agent
# ===========================================================================


# === V8 NET config (tinh chỉnh qua bench/server) ===
USE_NET = True              # False => agent == v6 thuần (regression test)
USE_NET_BOMB = True         # FIX #3: nhánh net-assisted bomb (đòn bẩy chính)
USE_FARM_RERANK = False     # hook farm cũ (gây thụ động) -> TẮT (A/B sau nếu cần)
NET_TIME_BUDGET = 0.045     # chỉ chạy net nếu còn ngân sách thời gian (s)
BIG_MARGIN = 2.0            # (farm rerank) chỉ can thiệp khi top-2 rule sát
LAMBDA = 1.5               # (farm rerank) trọng số V trong combined
MAX_CANDIDATES = 6
NET_FARM_DIST_CAP = 8       # chỉ xét ô farm trong bán kính BFS này (chặn đuôi thời gian)
TOL_BOMB = 0.02            # net chỉ cần KHÔNG phản đối rõ rệt: bom nếu V_bomb >= V_pass - TOL
BOMB_CLUSTER_TOL_BONUS = 0.05  # cluster >=2 địch -> rule rất chắc -> nới TOL thêm

# === NEURAL ZONE: encode planes (NHÚNG từ features.py, bit-exact) ===
# Sinh tự động bởi build_agent_v8.py — KHÔNG sửa tay. Sửa features.py rồi build lại.
N_PLANES = 16

def _nn_normalize_obs(obs):
    """Convert obs (JSON list / numpy) -> dict numpy chuẩn cho _build_danger_timed."""
    grid = np.asarray(obs["map"], dtype=np.int16)
    players = np.asarray(obs["players"], dtype=np.int16)
    bombs_raw = obs.get("bombs", [])
    if bombs_raw is None or len(bombs_raw) == 0:
        bombs = np.zeros((0, 4), dtype=np.int16)
    else:
        bombs = np.asarray(bombs_raw, dtype=np.int16)
        if bombs.ndim == 1:                       # shape (4,) -> (1,4)
            bombs = bombs.reshape(1, -1)
    return grid, players, bombs


def _nn_encode(obs, agent_id, step, bomb_radius_map=None):
    """
    obs: dict {map, players, bombs}. map 13x13; players (4,5); bombs (N,>=4).
    agent_id: slot của mình (0..3).
    step: số thứ tự step (để plane 14).
    bomb_radius_map: dict {(row,col): radius} ước lượng (tùy chọn). Nếu None,
        plane 9 dùng cột radius trong bombs (nếu có) hoặc 1+bonus owner hiện tại.
    Trả về planes (16,13,13) float32.
    """
    grid, players, bombs = _nn_normalize_obs(obs)
    H, W = grid.shape
    n_players = players.shape[0]
    P = np.zeros((N_PLANES, H, W), dtype=np.float32)

    # 0-3: map one-hot
    P[0] = (grid == WALL)
    P[1] = (grid == BOX)
    P[2] = (grid == ITEM_RADIUS)
    P[3] = (grid == ITEM_CAPACITY)

    # 4: vị trí mình (nếu còn sống)
    me = players[agent_id]
    my_r, my_c, my_alive, my_bombs_left, my_bonus = (
        int(me[0]), int(me[1]), int(me[2]), int(me[3]), int(me[4])
    )
    if my_alive and 0 <= my_r < H and 0 <= my_c < W:
        P[4, my_r, my_c] = 1.0

    # 5-6: enemy alive / enemy armed
    for i in range(n_players):
        if i == agent_id:
            continue
        r, c, alive, bl = int(players[i][0]), int(players[i][1]), int(players[i][2]), int(players[i][3])
        if alive and 0 <= r < H and 0 <= c < W:
            P[5, r, c] = 1.0
            if bl > 0:
                P[6, r, c] = 1.0

    # 7-9: bomb presence / timer / radius
    for b in bombs:
        br, bc, timer, owner = int(b[0]), int(b[1]), int(b[2]), int(b[3])
        if not (0 <= br < H and 0 <= bc < W):
            continue
        P[7, br, bc] = 1.0
        P[8, br, bc] = max(0.0, min(1.0, timer / BOMB_TIMER))
        if bomb_radius_map is not None and (br, bc) in bomb_radius_map:
            radius = bomb_radius_map[(br, bc)]
        elif b.shape[0] >= 5:                      # bombs có sẵn cột radius
            radius = int(b[4])
        else:                                      # fallback: 1 + bonus owner hiện tại
            oid = max(0, min(n_players - 1, owner))
            radius = 1 + int(players[oid][4])
        radius = max(1, min(MAX_RADIUS, radius))
        P[9, br, bc] = radius / MAX_RADIUS

    # 10-11: danger map — đi qua đúng hàm của v4 (không copy công thức)
    if len(bombs) > 0:
        dobs = {"map": grid, "players": players, "bombs": bombs}
        _dbt, danger_any, cell_explode_min = _build_danger_timed(dobs)
        for (r, c) in danger_any:
            if 0 <= r < H and 0 <= c < W:
                P[10, r, c] = 1.0
        for (r, c), t in cell_explode_min.items():
            if 0 <= r < H and 0 <= c < W:
                P[11, r, c] = max(0.0, 1.0 - t / 8.0)

    # 12-15: scalar broadcast cả plane
    P[12, :, :] = min(1.0, my_bombs_left / 5.0)
    P[13, :, :] = min(1.0, (1 + my_bonus) / 5.0)
    P[14, :, :] = min(1.0, step / 500.0)
    n_alive = int(sum(int(players[i][2]) for i in range(n_players)))
    P[15, :, :] = n_alive / 4.0

    return P

# === END NEURAL ZONE encode ===

# === NEURAL ZONE: numpy-forward ValueCNN (NHÚNG từ numpy_forward.py) ===
"""
numpy_forward.py (fix_v8) — Forward ValueCNN BẰNG NUMPY THUẦN (0 phụ thuộc torch).

Đọc trọng số đã FOLD BatchNorm. Hỗ trợ 2 loader:
  * from_npz  : .npz (giữ để export_weights verify trên Kaggle).
  * from_pkl  : .pkl (allowlist cuộc thi) — chỉ cần numpy + pickle.

Phần SẼ ĐƯỢC NHÚNG vào agent.py khi build (cắt phần self-test).

_NN_KEYS (do export_weights.py tạo):
  stem_W (32,16,3,3) stem_b (32,)
  b1_W (64,32,3,3) b1_b (64,) ... red_W (16,64,1,1) red_b (16,)
  fc1_W (128,2704) fc1_b (128,) fc2_W (1,128) fc2_b (1,)
"""

_NN_KEYS = ["stem_W", "stem_b", "b1_W", "b1_b", "b2_W", "b2_b",
        "b3_W", "b3_b", "red_W", "red_b", "fc1_W", "fc1_b", "fc2_W", "fc2_b"]


def _nn_im2col_3x3(x):
    """x: (B,C,H,W) -> patches (B, H*W, C*9) theo thứ tự (c, kh, kw); pad=1."""
    B, C, H, W = x.shape
    xp = np.zeros((B, C, H + 2, W + 2), dtype=x.dtype)
    xp[:, :, 1:H + 1, 1:W + 1] = x
    cols = np.empty((B, C, 9, H, W), dtype=x.dtype)
    idx = 0
    for kh in range(3):
        for kw in range(3):
            cols[:, :, idx] = xp[:, :, kh:kh + H, kw:kw + W]
            idx += 1
    cols = cols.transpose(0, 3, 4, 1, 2).reshape(B, H * W, C * 9)
    return cols


def _nn_conv3x3(x, W, b):
    """x:(B,Cin,H,W) W:(Cout,Cin,3,3) b:(Cout,) -> (B,Cout,H,W) pad=1."""
    B, Cin, H, Wd = x.shape
    Cout = W.shape[0]
    cols = _nn_im2col_3x3(x)
    Wm = W.reshape(Cout, Cin * 9)
    out = cols @ Wm.T + b
    return out.transpose(0, 2, 1).reshape(B, Cout, H, Wd)


def _nn_conv1x1(x, W, b):
    """x:(B,Cin,H,W) W:(Cout,Cin,1,1) b:(Cout,) -> (B,Cout,H,W)."""
    B, Cin, H, Wd = x.shape
    Cout = W.shape[0]
    Wm = W.reshape(Cout, Cin)
    out = np.einsum("bchw,oc->bohw", x, Wm) + b[None, :, None, None]
    return out


def _nn_relu(x):
    return np.maximum(x, 0.0, out=x)


class _NPValueNet:
    """Forward numpy thuần. .forward(planes) -> values (B,) float32."""

    def __init__(self, weights):
        self.w = {k: np.ascontiguousarray(weights[k].astype(np.float32)) for k in _NN_KEYS}

    @classmethod
    def from_npz(cls, path):
        data = np.load(path)
        missing = [k for k in _NN_KEYS if k not in data]
        if missing:
            raise KeyError(f"npz thiếu keys: {missing}")
        return cls({k: data[k] for k in _NN_KEYS})

    @classmethod
    def from_pkl(cls, path):
        """Loader .pkl (allowlist cuộc thi). pickle dict {key: np.ndarray}.
        Chỉ cần numpy + pickle (KHÔNG cần torch). __init__ tự cast fp16->fp32."""
        import pickle
        with open(path, "rb") as f:
            data = pickle.load(f)
        missing = [k for k in _NN_KEYS if k not in data]
        if missing:
            raise KeyError(f"pkl thiếu keys: {missing}")
        return cls({k: data[k] for k in _NN_KEYS})

    def forward(self, planes):
        """planes: (B,16,13,13) float32 -> (B,) float32."""
        w = self.w
        x = np.asarray(planes, dtype=np.float32)
        if x.ndim == 3:
            x = x[None]
        x = _nn_relu(_nn_conv3x3(x, w["stem_W"], w["stem_b"]))
        x = _nn_relu(_nn_conv3x3(x, w["b1_W"], w["b1_b"]))
        x = _nn_relu(_nn_conv3x3(x, w["b2_W"], w["b2_b"]))
        x = _nn_relu(_nn_conv3x3(x, w["b3_W"], w["b3_b"]))
        x = _nn_relu(_nn_conv1x1(x, w["red_W"], w["red_b"]))
        B = x.shape[0]
        flat = x.reshape(B, -1)
        h = flat @ w["fc1_W"].T + w["fc1_b"]
        h = _nn_relu(h)
        out = h @ w["fc2_W"].T + w["fc2_b"]
        return out[:, 0].astype(np.float32)



# === END NEURAL ZONE npforward ===


class Agent:
    def __init__(self, agent_id: int):
        self.agent_id    = int(agent_id)
        self.step_count  = 0
        self.pos_history = deque(maxlen=20)
        self.last_pos    = None
        # === V8 NET init ===
        self._net = None
        self._bomb_radius_map = {}
        self._radius_keys = {}
        self._net_used = 0
        try:
            import os as _os
            _here = _os.path.dirname(_os.path.abspath(__file__))
            _pkl = _os.path.join(_here, "model_value.pkl")
            if USE_NET and _os.path.isfile(_pkl):
                self._net = _NPValueNet.from_pkl(_pkl)
        except Exception:
            self._net = None
        # === END V8 NET init ===

    def _project_obs_for_net(self, obs, action, my_r, my_c, my_radius, bombs_left):
        """Dựng dict {map,players,bombs} mô phỏng state SAU `action` (1 tick) để encode."""
        grid = obs["map"]
        players = obs["players"]
        bombs_a = obs["bombs"]
        cur_bp = {(int(b[0]), int(b[1])) for b in bombs_a}
        if action in MOVE_ACTIONS:
            dr, dc = MOVES[action]
            nr, nc = my_r + dr, my_c + dc
            fpos = (nr, nc) if _walkable(nr, nc, grid, cur_bp) else (my_r, my_c)
        else:
            fpos = (my_r, my_c)
        pl = np.array(players, dtype=np.int16)
        pl[self.agent_id, 0] = fpos[0]
        pl[self.agent_id, 1] = fpos[1]
        placed = (action == PLACE_BOMB and bombs_left > 0 and (my_r, my_c) not in cur_bp)
        if placed:
            pl[self.agent_id, 3] = max(0, int(pl[self.agent_id, 3]) - 1)
        bombs5 = []
        for b in bombs_a:
            br, bc, tm, ow = int(b[0]), int(b[1]), int(b[2]), int(b[3])
            rad = self._bomb_radius_map.get((br, bc))
            if rad is None:
                oid = max(0, min(len(players) - 1, ow))
                rad = max(1, min(MAX_RADIUS, 1 + int(players[oid][4])))
            bombs5.append([br, bc, max(0, tm - 1), ow, int(rad)])
        if placed:
            bombs5.append([my_r, my_c, BOMB_TIMER - 1, self.agent_id, int(my_radius)])
        return {"map": grid, "players": pl, "bombs": bombs5}

    def _best_safe_farm_move(self, obs, grid, players, bombs_a, my_r, my_c, my_radius,
                             bomb_pos, danger_by_time, danger_any, nearby_threat_pos):
        """Nước DI CHUYỂN farm RULE-SAFE tốt nhất (first move) hoặc None — nước 'thụ động'."""
        pos = (my_r, my_c)
        reachable = _reachable_cells_with_dist(pos, grid, bomb_pos, danger_by_time)
        best_a, best_s = None, -1e18
        for (r, c), bfs_dist in reachable.items():
            if bfs_dist > NET_FARM_DIST_CAP:
                continue
            if grid[r, c] in (WALL, BOX):
                continue
            if (r, c) in danger_by_time.get(0, set()):
                continue
            if (r, c) in danger_by_time.get(1, set()) and bfs_dist <= 1:
                continue
            a = _bfs_timed(pos, {(r, c)}, grid, bomb_pos, danger_by_time)
            if a is None or a == STOP:
                continue
            score = _box_farm_score(r, c, grid, players, bombs_a, self.agent_id,
                                    my_r, my_c, my_radius, bomb_pos, danger_any, bfs_dist)
            if score <= best_s:
                continue
            proj = _project_future_state(obs, self.agent_id, a, my_r, my_c,
                                         my_radius, nearby_threat_pos)
            if not _action_survives_future(proj[0], grid, proj[1], proj[2], proj[3]):
                continue
            best_s, best_a = score, a
        return best_a

    def _net_assisted_bomb(self, obs, grid, players, bombs_a, my_r, my_c, my_radius,
                           bombs_left, bomb_pos, danger_by_time, danger_any,
                           nearby_armed, combat_mode, extra_danger,
                           nearby_threat_pos, t0):
        """FIX #3 — trả PLACE_BOMB (bất đối xứng, CHỈ thêm bom an toàn) hoặc None."""
        bval, hits_box, hits_enemy = _bomb_value(
            my_r, my_c, my_radius, grid, players, self.agent_id, danger_any)
        if not hits_enemy:
            return None
        threshold = BOMB_COMBAT_CLUSTER_SCORE if nearby_armed >= 2 else BOMB_MIN_SCORE
        if bval >= threshold:
            return None        # v6 đã tự đặt ở PRIORITY 3 -> không phải vùng của ta
        can_esc = _can_escape_after_bomb(
            my_r, my_c, my_radius, grid, bomb_pos,
            danger_by_time, extra_danger_tiles=extra_danger)
        if not can_esc:
            return None
        if combat_mode:
            if _count_escape_first_moves(
                    my_r, my_c, my_radius, grid, bomb_pos,
                    danger_by_time, extra_danger_tiles=extra_danger) < 2:
                return None
        proj = _project_future_state(obs, self.agent_id, PLACE_BOMB, my_r, my_c,
                                     my_radius, nearby_threat_pos)
        if not _action_survives_future(proj[0], grid, proj[1], proj[2], proj[3]):
            return None
        a_pass = self._best_safe_farm_move(
            obs, grid, players, bombs_a, my_r, my_c, my_radius,
            bomb_pos, danger_by_time, danger_any, nearby_threat_pos)
        if a_pass is None:
            a_pass = STOP
        if (time.perf_counter() - t0) > NET_TIME_BUDGET:
            return None
        o_bomb = self._project_obs_for_net(obs, PLACE_BOMB, my_r, my_c, my_radius, bombs_left)
        o_pass = self._project_obs_for_net(obs, a_pass, my_r, my_c, my_radius, bombs_left)
        P = np.stack([
            _nn_encode(o_bomb, self.agent_id, self.step_count + 1, None),
            _nn_encode(o_pass, self.agent_id, self.step_count + 1, None),
        ]).astype(np.float32)
        V = self._net.forward(P)
        v_bomb, v_pass = float(V[0]), float(V[1])
        tol = TOL_BOMB + (BOMB_CLUSTER_TOL_BONUS if nearby_armed >= 2 else 0.0)
        if v_bomb >= v_pass - tol:
            return PLACE_BOMB
        return None

    def _net_rerank(self, obs, grid, players, bombs_a, my_r, my_c, my_radius,
                    bombs_left, bomb_pos, danger_by_time, danger_any,
                    nearby_threat_pos, t0):
        """(LEGACY, mặc định TẮT qua USE_FARM_RERANK) re-rank farm move. Giữ để A/B."""
        pos = (my_r, my_c)
        reachable = _reachable_cells_with_dist(pos, grid, bomb_pos, danger_by_time)
        best_by_action = {}
        for (r, c), bfs_dist in reachable.items():
            if bfs_dist > NET_FARM_DIST_CAP:
                continue
            if grid[r, c] in (WALL, BOX):
                continue
            if (r, c) in danger_by_time.get(0, set()):
                continue
            if (r, c) in danger_by_time.get(1, set()) and bfs_dist <= 1:
                continue
            a = _bfs_timed(pos, {(r, c)}, grid, bomb_pos, danger_by_time)
            if a is None or a == STOP:
                continue
            score = _box_farm_score(r, c, grid, players, bombs_a, self.agent_id,
                                    my_r, my_c, my_radius, bomb_pos, danger_any, bfs_dist)
            if a not in best_by_action or score > best_by_action[a]:
                proj = _project_future_state(obs, self.agent_id, a, my_r, my_c,
                                             my_radius, nearby_threat_pos)
                if not _action_survives_future(proj[0], grid, proj[1], proj[2], proj[3]):
                    continue
                best_by_action[a] = score
        if len(best_by_action) < 2:
            return None
        cands = sorted(best_by_action.items(), key=lambda kv: -kv[1])
        if cands[0][1] - cands[1][1] >= BIG_MARGIN:
            return None
        if (time.perf_counter() - t0) > NET_TIME_BUDGET:
            return None
        cands = cands[:MAX_CANDIDATES]
        planes = []
        for a, _s in cands:
            o2 = self._project_obs_for_net(obs, a, my_r, my_c, my_radius, bombs_left)
            planes.append(_nn_encode(o2, self.agent_id, self.step_count + 1, None))
        P = np.stack(planes).astype(np.float32)
        V = self._net.forward(P)
        scores = np.array([s for _a, s in cands], dtype=np.float32)
        smin, smax = float(scores.min()), float(scores.max())
        rnorm = (scores - smin) / (smax - smin) if smax > smin else np.zeros_like(scores)
        combined = rnorm + LAMBDA * V
        return cands[int(np.argmax(combined))][0]

    def act(self, obs: dict) -> int:
        try:
            result = self._act_impl(obs)
            return int(result)
        except Exception:
            return STOP
        finally:
            self.step_count += 1

    def _act_impl(self, obs: dict) -> int:
        t0 = time.perf_counter()

        grid    = obs["map"]
        players = obs["players"]
        bombs_a = obs["bombs"]

        me           = players[self.agent_id]
        my_r, my_c   = int(me[0]), int(me[1])
        alive        = int(me[2])
        bombs_left   = int(me[3])
        radius_bonus = int(me[4])

        if not alive:
            return STOP

        my_radius = max(1, min(MAX_RADIUS, 1 + radius_bonus))
        pos       = (my_r, my_c)

        # C4: position history for stuck detection and anti-repeat
        self.pos_history.append(pos)
        prev_pos      = self.last_pos
        self.last_pos = pos

        # FIX-4: danger map now also returns cell_explode_min
        danger_by_time, danger_any, cell_explode_min = _build_danger_timed(obs)

        bomb_pos = {(int(b[0]), int(b[1])) for b in bombs_a}

        # === V8 bomb-radius tracker (mỗi step, plane 9 train-parity) ===
        try:
            _nk = {}
            _brm = {}
            for _b in bombs_a:
                _br, _bc, _bo = int(_b[0]), int(_b[1]), int(_b[3])
                _key = (_br, _bc, _bo)
                if _key in self._radius_keys:
                    _rad = self._radius_keys[_key]
                else:
                    _oid = max(0, min(len(players) - 1, _bo))
                    _rad = max(1, min(MAX_RADIUS, 1 + int(players[_oid][4])))
                _nk[_key] = _rad
                _brm[(_br, _bc)] = _rad
            self._radius_keys = _nk
            self._bomb_radius_map = _brm
        except Exception:
            self._bomb_radius_map = {}
        # === END tracker ===

        # FIX-5: combat mode via BFS, not Manhattan
        nearby_armed = _count_nearby_armed_bfs(
            my_r, my_c, players, self.agent_id, grid)
        combat_mode  = nearby_armed >= 1

        # C1: predicted enemy blasts (unchanged — Manhattan ≤2 is intentionally
        #     conservative here: we want to be cautious about adjacent enemies)
        extra_danger = set()
        if combat_mode:
            extra_danger = _enemy_predicted_blast(
                my_r, my_c, players, self.agent_id, grid)

        # C4a: stuck detection
        is_stuck = (
            len(self.pos_history) >= 16
            and len(set(self.pos_history)) <= 3
            and len(bombs_a) == 0
            and pos not in danger_any
        )

        # FIX-1: compute nearby_threat_pos ONCE; reuse in all future projections
        nearby_threat_pos = _nearby_threat_positions(
            my_r, my_c, players, self.agent_id, grid)

        # -------------------------------------------------------------------
        # PRIORITY 1: ESCAPE (current real danger — unchanged)
        # -------------------------------------------------------------------
        if pos in danger_by_time.get(1, set()) or pos in danger_any:
            esc = _escape_timed(pos, grid, bomb_pos, danger_by_time, danger_any)
            best_score = -1e9
            best_action = None

            for action in MOVE_ACTIONS:
                dr, dc = MOVES[action]

                if not _walkable(my_r + dr, my_c + dc, grid, bomb_pos):
                    continue


                f_pos, f_bp, f_dbt, f_da, future_cell_explode_min  = _project_future_state(
                            obs, self.agent_id, action, my_r, my_c, my_radius,
                            nearby_threat_pos)

                score = _reachable_safe_timed(f_pos, grid, f_bp, 
                                              future_cell_explode_min, f_dbt, depth=SAFE_DEPTH)


                if _action_survives_future(f_pos, grid, f_bp, f_dbt, f_da):
                    if score > best_score:
                        best_score = score
                        best_action = action
                
            if best_action is not None:
                return best_action
            
            if esc is not None:
                return esc

            for a in MOVE_ACTIONS:
                dr, dc = MOVES[a]
                if _walkable(my_r + dr, my_c + dc, grid, bomb_pos):
                    return a
            return PLACE_BOMB
        
        # -------------------------------------------------------------------
        # PRIORITY 2: PICK UP ITEM
        # -------------------------------------------------------------------
        if time.perf_counter() - t0 < TIME_BUDGET_S:
            h, w  = grid.shape
            items = {(r, c) for r in range(h) for c in range(w)
                     if grid[r, c] in (ITEM_RADIUS, ITEM_CAPACITY)}
            if items:
                
                a = _bfs_timed(pos, items, grid, bomb_pos, danger_by_time)
                if a is not None and a != STOP:
                    f_pos, f_bp, f_dbt, f_da, future_cell_explode_min = _project_future_state(
                        obs, self.agent_id, a, my_r, my_c, my_radius,
                        nearby_threat_pos)
                    if _action_survives_future(f_pos, grid, f_bp, f_dbt, f_da):
                        return a

        # -------------------------------------------------------------------
        # C4b: UNSTUCK BOMB (farm mode only)
        # -------------------------------------------------------------------
        if is_stuck and not combat_mode and bombs_left > 0 and pos not in bomb_pos:
            if time.perf_counter() - t0 < TIME_BUDGET_S:
                blast   = _blast_tiles(my_r, my_c, my_radius, grid)
                has_box = any(grid[r][c] == BOX for r, c in blast)
                if has_box and _can_escape_after_bomb(
                        my_r, my_c, my_radius, grid, bomb_pos,
                        danger_by_time, extra_danger_tiles=None):
                    f_pos, f_bp, f_dbt, f_da, future_cell_explode_min = _project_future_state(
                        obs, self.agent_id, PLACE_BOMB, my_r, my_c, my_radius,
                        nearby_threat_pos)
                    if _action_survives_future(f_pos, grid, f_bp, f_dbt, f_da):
                        return PLACE_BOMB

        # -------------------------------------------------------------------
        # PRIORITY 3: PLACE BOMB
        # -------------------------------------------------------------------
        if bombs_left > 0 and pos not in bomb_pos:
            if time.perf_counter() - t0 < TIME_BUDGET_S:
                bval, hits_box, hits_enemy = _bomb_value(
                    my_r, my_c, my_radius, grid, players,
                    self.agent_id, danger_any)

                threshold = BOMB_COMBAT_CLUSTER_SCORE if nearby_armed >= 2 \
                            else BOMB_MIN_SCORE

                if bval >= threshold and (hits_box or hits_enemy):
                    can_esc = _can_escape_after_bomb(
                        my_r, my_c, my_radius, grid, bomb_pos,
                        danger_by_time, extra_danger_tiles=extra_danger)

                    if can_esc and combat_mode:
                        can_esc = _count_escape_first_moves(
                            my_r, my_c, my_radius, grid, bomb_pos,
                            danger_by_time,
                            extra_danger_tiles=extra_danger) >= 2

                    if can_esc:
                        f_pos, f_bp, f_dbt, f_da, future_cell_explode_min = _project_future_state(
                            obs, self.agent_id, PLACE_BOMB, my_r, my_c,
                            my_radius, nearby_threat_pos)
                        if _action_survives_future(f_pos, grid, f_bp, f_dbt, f_da):
                            return PLACE_BOMB



        # === V8 NET-ASSISTED BOMB (sau PRIORITY 3, trước PRIORITY 4) ===
        # Bất đối xứng: net CHỈ có thể THÊM một cú bom mà rule đã xác nhận trúng địch,
        # qua đủ escape-check của v6, và net không phản đối. KHÔNG bao giờ làm agent
        # thụ động hơn / kém an toàn hơn v6. USE_NET_BOMB=False => bỏ qua hoàn toàn.
        if (USE_NET_BOMB and self._net is not None and bombs_left > 0
                and pos not in bomb_pos
                and (time.perf_counter() - t0) < NET_TIME_BUDGET):
            try:
                _nb = self._net_assisted_bomb(
                    obs, grid, players, bombs_a, my_r, my_c, my_radius,
                    bombs_left, bomb_pos, danger_by_time, danger_any,
                    nearby_armed, combat_mode, extra_danger, nearby_threat_pos, t0)
                if _nb is not None:
                    self._net_used += 1
                    return int(_nb)
            except Exception:
                pass
        # === END V8 NET-ASSISTED BOMB ===

        # === V8 FARM RE-RANK HOOK (LEGACY; USE_FARM_RERANK=False mặc định) ===
        if (USE_NET and USE_FARM_RERANK and self._net is not None
                and (time.perf_counter() - t0) < NET_TIME_BUDGET):
            try:
                _na = self._net_rerank(obs, grid, players, bombs_a, my_r, my_c,
                                       my_radius, bombs_left, bomb_pos,
                                       danger_by_time, danger_any, nearby_threat_pos, t0)
                if _na is not None:
                    self._net_used += 1
                    return int(_na)
            except Exception:
                pass
        # === END V8 FARM HOOK ===

        # -------------------------------------------------------------------
        # PRIORITY 4: FARM BOXES (reachable + local-first + far only if better)
        # -------------------------------------------------------------------
        if time.perf_counter() - t0 < TIME_BUDGET_S:
            reachable = _reachable_cells_with_dist(pos, grid, bomb_pos, danger_by_time)

            candidates = []
            h, w = grid.shape

            for (r, c), bfs_dist in reachable.items():
                if grid[r, c] in (WALL, BOX):
                    continue
                if (r, c) in danger_by_time.get(0, set()):
                    continue  # chết ngay khi đứng lên
                if (r, c) in danger_by_time.get(1, set()) and bfs_dist <= 1:
                    continue 

                score = _box_farm_score(
                    r, c, grid, players, bombs_a,
                    self.agent_id, my_r, my_c, my_radius,
                    bomb_pos, danger_any, bfs_dist
                )
                candidates.append((score, bfs_dist, (r, c)))

            if candidates:
                # Split into local vs far.
                # Local = more certain, should win unless far is clearly better.
                local = [x for x in candidates if x[1] <= 4]
                far   = [x for x in candidates if x[1] > 4]

                best_local = max(local, key=lambda x: x[0]) if local else None
                best_far   = max(far,   key=lambda x: x[0]) if far   else None

                chosen = None

                if best_local is not None and best_far is not None:
                    # Far spot must beat local by a clear margin
                    if best_far[0] >= best_local[0] + 2.5:
                        chosen = best_far
                    else:
                        chosen = best_local
                elif best_local is not None:
                    chosen = best_local
                elif best_far is not None:
                    chosen = best_far

                if chosen is not None:
                    _, _, target = chosen
                    a = _bfs_timed(pos, {target}, grid, bomb_pos, danger_by_time)

                    if a is not None and a != STOP:
                        f_pos, f_bp, f_dbt, f_da, future_cell_explode_min = _project_future_state(
                                obs, self.agent_id, a, my_r, my_c,
                                my_radius, nearby_threat_pos)
                        
                        if _action_survives_future(f_pos, grid, f_bp, f_dbt, f_da):
                                return a


        # -------------------------------------------------------------------
        # PRIORITY 5: CHASE NEAREST ENEMY
        # -------------------------------------------------------------------
        if time.perf_counter() - t0 < TIME_BUDGET_S:
            if not _enemy_adjacent(my_r, my_c, players, self.agent_id):
                enemies = {
                    (int(players[i][0]), int(players[i][1]))
                    for i in range(len(players))
                    if i != self.agent_id and int(players[i][2]) == 1
                }
                if enemies:
                    a = _bfs_timed(pos, enemies, grid, bomb_pos, danger_by_time)
                    if a is not None and a != STOP:
                        f_pos, f_bp, f_dbt, f_da, future_cell_explode_min = _project_future_state(
                            obs, self.agent_id, a, my_r, my_c, my_radius,
                            nearby_threat_pos)
                        if _action_survives_future(
                                f_pos, grid, f_bp, f_dbt, f_da):
                            return a

        # -------------------------------------------------------------------
        # FALLBACK: timer-aware safe move  (FIX-3 + FIX-6)
        # -------------------------------------------------------------------

        return _safe_fallback(pos, grid, bomb_pos, danger_by_time,
                              cell_explode_min, self.step_count,
                              self.agent_id, last_pos=prev_pos)