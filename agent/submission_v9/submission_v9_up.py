"""
agent.py -- Bomberland Rule-Based Agent v5
Base: submission_v6 (v4 core + all FIX-1..6).

New in v5: Enemy Intent Tracking
=================================
Motivation
----------
The v4 agent used only static Manhattan distance to estimate theft risk and
enemy proximity.  This ignores where enemies are *heading*, causing the agent
to:
  - Chase items that an enemy is already sprinting toward (losing race).
  - Under-value placing a bomb when an enemy is walking straight into the
    blast zone from a box the enemy is farming.
  - Over-penalise items the enemy is actually moving *away* from.

Design
------
EnemyTracker (stored in Agent):
  - Stores a deque(maxlen=3) of recent positions per enemy ID.
  - Called at the start of every act() to record current positions.

_enemy_intent(tracker, enemy_id, enemy_pos, grid, bomb_pos, obs):
  - BFS from enemy_pos to every item and box on the map (up to depth cap).
  - For each BFS shortest-path, examine the *first step* of that path.
  - If the enemy's recent movement direction aligns with that first step,
    multiply that target's weight by an alignment bonus (1 + ALIGN_BONUS).
  - Final weight per target: base_weight / bfs_dist * alignment_multiplier
    where base_weight = ITEM_WEIGHT for items, BOX_WEIGHT for boxes.
  - Returns dict {target_pos: intent_weight}.

_enemy_threat_weights(tracker, obs, grid, bomb_pos):
  - Aggregates intent across all live enemies.
  - Returns dict {cell: total_enemy_intent_weight}  (union over all enemies).

Integration
-----------
FIX-7  Priority 2 (item pickup): item score = base_bfs_score
       minus  ITEM_THEFT_SCALE * enemy_intent[item_pos].
       Items enemies are heading toward are deprioritised; we pick the
       safest, least-contested item first.

FIX-8  _box_farm_score: theft_risk replaced by trajectory-aware
       _traj_theft_risk(blast, enemy_intent_weights).
       Each box in blast is penalised by the sum of enemy intent weights
       pointing at it (or at any cell in the blast within range).

FIX-9  Priority 3 (place bomb): bomb_value gets a small additive bonus
       BOMB_CHASE_BONUS if any enemy's predicted trajectory passes through
       our current blast zone in the next FWD_STEPS steps.

All FIX-1..6 from v4 are unchanged.
"""
import time
from collections import deque

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

TIME_BUDGET_S             = 0.10
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
SAFE_DEPTH  =  10

# ---------------------------------------------------------------------------
# FIX-7/8/9: Enemy Intent Tracking constants
# ---------------------------------------------------------------------------
ENEMY_HISTORY_LEN  = 3    # steps of position history kept per enemy

# BFS depth cap when computing enemy intent (keep cheap: map is 13×13)
INTENT_DEPTH_CAP   = 12

# Base weights for different target types in intent BFS
INTENT_ITEM_WEIGHT = 3.0  # items are high-value targets
INTENT_BOX_WEIGHT  = 1.0  # boxes are lower-value targets

# How much an aligned movement direction multiplies that path's weight
INTENT_ALIGN_BONUS = 1.5  # aligned direction → weight × (1 + 1.5) = ×2.5

# FIX-7: scale factor applied to enemy intent when penalising Priority 2 item score
ITEM_THEFT_SCALE   = 2.5

# FIX-8: scale factor applied to enemy intent when penalising box farm theft_risk
BOX_THEFT_SCALE    = 1.5

# FIX-9: bonus added to bomb_value when enemy trajectory enters our blast zone
BOMB_CHASE_BONUS   = 2.0
# How many steps ahead we project the enemy's intended path for FIX-9
INTENT_FWD_STEPS   = 4

# === Aggression knobs from Guide A ===
AGGR                = True
ITEM_MAX_CHASE_DIST = 6          # A2: only chase items within this BFS distance
TRAP_ENEMY_MAX_ESC  = 2          # A1: enemy is "trappable" if escape cells ≤ this
TRAP_BFS_RADIUS     = 6          # A1: consider enemies within this many steps


# ===========================================================================
# Geometry helpers  (unchanged)
# ===========================================================================
def _traj_theft_risk(blast_tiles, enemy_intent_weights):
    """
    FIX-8: Trajectory-aware theft risk for _box_farm_score.
    Sums enemy intent weights for every cell in blast_tiles.
    Replaces the old Manhattan-distance loop.
    """
    risk = 0.0
    for cell in blast_tiles:
        risk += enemy_intent_weights.get(cell, 0.0)
    return risk


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
            score -= REWARD_SAFE + 4  # dead-end penalty

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
def _enemy_escape_cells(er, ec, grid, bomb_pos, blast_set):
    """Return list of cells enemy can reach (stop + 4 moves) that are outside blast_set."""
    escapes = []
    # stay
    if _walkable(er, ec, grid, bomb_pos) and (er, ec) not in blast_set:
        escapes.append((er, ec))
    for dr, dc in ((-1,0),(1,0),(0,-1),(0,1)):
        nr, nc = er+dr, ec+dc
        if _walkable(nr, nc, grid, bomb_pos) and (nr, nc) not in blast_set:
            escapes.append((nr, nc))
    return escapes

def _trap_kill_score(my_r, my_c, my_radius, grid, players, agent_id, bomb_pos, reachable_dist=None):
    """Returns (n_trappable, n_enemy_in_blast)."""
    blast = _blast_tiles(my_r, my_c, my_radius, grid)
    blast_set = set(blast)
    n_trap = 0
    n_in = 0
    for i, p in enumerate(players):
        if i == agent_id or int(p[2]) != 1: continue
        er, ec = int(p[0]), int(p[1])
        # distance filter (use precomputed reachable if available)
        dist = reachable_dist.get((er, ec)) if reachable_dist else None
        if dist is None or dist > TRAP_BFS_RADIUS:
            continue
        if (er, ec) in blast_set:
            n_in += 1
            escapes = _enemy_escape_cells(er, ec, grid, bomb_pos, blast_set)
            if len(escapes) <= TRAP_ENEMY_MAX_ESC:
                n_trap += 1
    return n_trap, n_in


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
    return score, boxes > 0, enemies > 0

def _bomb_value_advanced(my_r, my_c, my_radius, grid, players, agent_id, danger_any,
                tracker=None, obs=None, bomb_pos=None):
    """
    Aggressive bomb scoring using enemy intent.
    - Enemy in blast now: high value, multiplied by cornered factor and bomb count.
    - Enemy not in blast but heading into blast (intent): predictive bonus.
    - Boxes have value, but also block enemy escape routes.
    - Chain reaction bonus.
    - Suicide penalty.
    """
    blast = _blast_tiles(my_r, my_c, my_radius, grid)
    score = 0.0
    enemy_positions = set()
    boxes_hit = 0
    enemies_hit = 0

    # Helper to count escape moves from a cell given blast zone
    def escape_moves_from_cell(r, c, blast_set, grid, bomb_pos):
        escapes = 0
        for dr, dc in ((-1,0),(1,0),(0,-1),(0,1)):
            nr, nc = r+dr, c+dc
            if _walkable(nr, nc, grid, bomb_pos) and (nr, nc) not in blast_set:
                escapes += 1
        return escapes

    # --- Enemy scoring ---
    if tracker is not None and obs is not None:
        for i, p in enumerate(players):
            if i == agent_id or int(p[2]) != 1:
                continue
            er, ec = int(p[0]), int(p[1])
            enemy_positions.add((er, ec))
            enemy_bombs = int(p[3])

            if (er, ec) in blast:
                # Enemy currently inside blast
                enemies_hit += 1
                # Cornered factor: fewer escape moves outside blast = deadlier
                escapes = escape_moves_from_cell(er, ec, blast, grid, bomb_pos)
                corner_bonus = 1.5 if escapes <= 1 else 1.0
                # More dangerous if enemy still has bombs
                bomb_bonus = 1.2 if enemy_bombs > 0 else 1.0
                score += 5.0 * corner_bonus * bomb_bonus
            else:
                # Enemy not in blast – will they walk into it?
                # Use intent BFS from enemy, look for blast cells
                intent_map = _enemy_intent(tracker, i, (er, ec), grid, bomb_pos, obs)
                for target, weight in intent_map.items():
                    if target in blast:
                        # Predictive bonus: enemy intends to come here
                        score += 2.0 * weight
    else:
        # Fallback to original simple scoring (no intent data)
        for i, p in enumerate(players):
            if i == agent_id or int(p[2]) != 1:
                continue
            if (int(p[0]), int(p[1])) in blast:
                enemies_hit += 1
        boxes_hit = sum(1 for r,c in blast if grid[r,c] == BOX)
        score = boxes_hit * 1 + enemies_hit * 3


    # --- Box scoring (aggressive twist: boxes block enemy escape) ---
    boxes_hit = 0
    for r,c in blast:
        if grid[r,c] == BOX:
            boxes_hit += 1
            # Does this box block an enemy escape route?
            blocks = 0
            for (er, ec) in enemy_positions:
                # If box is adjacent to enemy and enemy is not inside blast
                if abs(er - r) + abs(ec - c) == 1 and (er, ec) not in blast:
                    blocks += 1
            score += 1.0 + 0.5 * min(blocks, 2)

    # --- Chain reaction: if blast includes another bomb that we can detonate early ---
    if bomb_pos and obs is not None:
        # Build a quick map from bomb position to its owner and timer
        bomb_owners = {}
        for b in obs["bombs"]:
            bx, by, timer, oid = int(b[0]), int(b[1]), int(b[2]), int(b[3])
            bomb_owners[(bx, by)] = (oid, timer)
        
        for (bx, by) in bomb_pos:
            # Skip our own bomb position (we are about to place one there, but it doesn't exist yet)
            if (bx, by) == (my_r, my_c):
                continue
            
            # Check if the existing bomb is inside our blast → we will detonate it
            if (bx, by) in blast:
                owner, timer = bomb_owners.get((bx, by), (None, None))
                if owner is not None and owner != agent_id:
                    # Enemy bomb: get its radius from the player's power
                    enemy_radius = my_radius  # fallback
                    for i, p in enumerate(players):
                        if i == owner and int(p[2]) == 1:
                            enemy_radius = max(1, min(MAX_RADIUS, 1 + int(p[4])))
                            break
                    # Compute the blast of that enemy bomb
                    other_blast = _blast_tiles(bx, by, enemy_radius, grid)
                    # Bonus for detonating an enemy bomb: base + value from what it hits
                    score += 3.0
                    # Optionally add extra for boxes/enemies the enemy bomb would destroy
                    for (x, y) in other_blast:
                        if grid[x, y] == BOX:
                            score += 0.5   # each box destroyed by chain
                        # Check if any live enemy is inside that blast (other than the bomb owner)
                        for ii, pp in enumerate(players):
                            if ii == agent_id or int(pp[2]) != 1:
                                continue
                            if (int(pp[0]), int(pp[1])) == (x, y):
                                score += 1.5
                                break
                else:
                    # Our own bomb (already placed elsewhere) – small bonus for chain
                    score += 1

    return score, boxes_hit > 0, enemies_hit > 0

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
    
    grid_copy = grid.copy()

    for r in range(13):
        for c in range(13):
            if (r,c) in future_dbt.get(0, set()) and grid_copy[r, c] == BOX:
                grid_copy[r, c] = GRASS


    return _can_reach_safety(future_pos, grid_copy, future_bomb_pos,
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
    my_r, my_c, my_radius, bomb_pos, danger_any, bfs_dist,
    enemy_intent_weights=None   # FIX-8: trajectory-aware theft risk
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

    # FIX-8: trajectory-aware theft risk replaces old Manhattan loop
    if enemy_intent_weights is not None:
        theft_risk = BOX_THEFT_SCALE * _traj_theft_risk(blast, enemy_intent_weights)
    else:
        # Fallback: original Manhattan-based theft risk (kept for safety)
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

    danger_pen = 1.5 if (r, c) in danger_any else 0.0

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
# FIX-7/8/9: Enemy Intent Tracking
# ===========================================================================

class EnemyTracker:
    """
    Stores the last ENEMY_HISTORY_LEN positions for each enemy agent.
    Updated once per act() call before any decision logic runs.
    """
    def __init__(self, n_agents: int):
        # history[i] = deque of (row, col) for agent i
        self.history = {i: deque(maxlen=ENEMY_HISTORY_LEN) for i in range(n_agents)}

    def update(self, players):
        """Record current positions of all live enemies."""
        for i, p in enumerate(players):
            if int(p[2]) == 1:  # alive
                self.history[i].append((int(p[0]), int(p[1])))

    def movement_direction(self, enemy_id) -> tuple:
        """
        Returns the dominant recent movement direction (dr, dc) for enemy_id,
        computed as the sign of (latest_pos - oldest_pos).
        Returns (0, 0) if history has fewer than 2 entries or enemy is stationary.
        """
        h = self.history[enemy_id]
        if len(h) < 2:
            return (0, 0)
        r0, c0 = h[0]   # oldest in window
        r1, c1 = h[-1]  # most recent
        dr = r1 - r0
        dc = c1 - c0
        # Normalise to unit direction (dominant axis wins on diagonals)
        if dr == 0 and dc == 0:
            return (0, 0)
        if abs(dr) >= abs(dc):
            return (1 if dr > 0 else -1, 0)
        else:
            return (0, 1 if dc > 0 else -1)


def _enemy_intent(tracker, enemy_id, enemy_pos, grid, bomb_pos, obs):
    """
    BFS from enemy_pos to every reachable item and box.

    For each target T found at BFS depth d:
      base_weight = INTENT_ITEM_WEIGHT  if T is an item
                  = INTENT_BOX_WEIGHT   if T is a box-adjacent grass cell
                                         (i.e. a potential bomb-placement spot)
      path_weight = base_weight / d

    The first step of the BFS shortest path is compared to the enemy's
    recent movement direction.  If they align, path_weight is multiplied
    by (1 + INTENT_ALIGN_BONUS).

    Returns dict {target_pos: accumulated_weight}.
    """
    er, ec       = enemy_pos
    move_dir     = tracker.movement_direction(enemy_id)
    grid_map     = obs["map"]
    h, w         = grid_map.shape

    # BFS: state = (r, c, first_step_dir)
    # first_step_dir = (dr, dc) of the very first move taken from enemy_pos
    visited    = {(er, ec)}
    queue      = deque()
    intent_map = {}   # target_pos -> weight

    # Seed from enemy position
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        nr, nc = er + dr, ec + dc
        if not _walkable(nr, nc, grid_map, bomb_pos):
            continue
        if (nr, nc) in visited:
            continue
        visited.add((nr, nc))
        queue.append((nr, nc, 1, (dr, dc)))  # (r, c, depth, first_dir)

    while queue:
        r, c, d, first_dir = queue.popleft()
        if d > INTENT_DEPTH_CAP:
            continue

        cell = grid_map[r, c]

        # Items on the ground: direct pickup target
        if cell in (ITEM_RADIUS, ITEM_CAPACITY):
            base_w = INTENT_ITEM_WEIGHT / d
            align  = (first_dir == move_dir and move_dir != (0, 0))
            w_val  = base_w * (1 + INTENT_ALIGN_BONUS) if align else base_w
            intent_map[(r, c)] = intent_map.get((r, c), 0.0) + w_val

        # Grass cells adjacent to a box: enemy can farm from here
        if cell == GRASS:
            adj_boxes = sum(
                1 for ddr, ddc in ((-1,0),(1,0),(0,-1),(0,1))
                if 0 <= r+ddr < h and 0 <= c+ddc < w
                and grid_map[r+ddr, c+ddc] == BOX
            )
            if adj_boxes > 0:
                base_w = INTENT_BOX_WEIGHT * adj_boxes / d
                align  = (first_dir == move_dir and move_dir != (0, 0))
                w_val  = base_w * (1 + INTENT_ALIGN_BONUS) if align else base_w
                intent_map[(r, c)] = intent_map.get((r, c), 0.0) + w_val

        # Continue BFS
        if d < INTENT_DEPTH_CAP:
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nr, nc = r + dr, c + dc
                if (nr, nc) in visited:
                    continue
                if not _walkable(nr, nc, grid_map, bomb_pos):
                    continue
                visited.add((nr, nc))
                queue.append((nr, nc, d + 1, first_dir))

    return intent_map


def _all_enemy_intent(tracker, obs, grid, bomb_pos, agent_id):
    """
    Aggregate intent weights across all live enemies (excluding self).
    Returns dict {cell: total_weight}.
    """
    players    = obs["players"]
    combined   = {}
    for i, p in enumerate(players):
        if i == agent_id or int(p[2]) != 1:
            continue
        ep      = (int(p[0]), int(p[1]))
        weights = _enemy_intent(tracker, i, ep, grid, bomb_pos, obs)
        for cell, w in weights.items():
            combined[cell] = combined.get(cell, 0.0) + w
    return combined


def _enemy_bfs_dist(enemy_pos, target, grid, bomb_pos, depth_cap=INTENT_DEPTH_CAP):
    """
    Returns the BFS distance (walkable steps) from enemy_pos to target,
    or None if unreachable within depth_cap.
    Used to check whether we can beat an enemy to an item.
    """
    er, ec = enemy_pos
    tr, tc = target
    if (er, ec) == (tr, tc):
        return 0
    visited = {(er, ec)}
    queue   = deque([(er, ec, 0)])
    while queue:
        r, c, d = queue.popleft()
        if d >= depth_cap:
            continue
        for dr, dc in ((-1,0),(1,0),(0,-1),(0,1)):
            nr, nc = r+dr, c+dc
            if (nr, nc) in visited:
                continue
            if not _walkable(nr, nc, grid, bomb_pos):
                continue
            if (nr, nc) == (tr, tc):
                return d + 1
            visited.add((nr, nc))
            queue.append((nr, nc, d+1))
    return None


def _race_adjusted_intent(item_pos, my_bfs_dist, enemy_intent_weights,
                           tracker, obs, grid, bomb_pos, agent_id):
    """
    FIX-7 (improved): Returns theft penalty for an item, scaled by whether
    we actually win the race to it.

    For each live enemy that has non-zero intent toward item_pos:
      - Compute enemy BFS distance to item_pos.
      - lead = enemy_dist - my_bfs_dist
          lead > 0  → we arrive first  → penalty = 0
          lead == 0 → tie              → penalty × 0.5 (item destroyed on tie)
          lead < 0  → enemy arrives first → full penalty

    Returns the adjusted total theft penalty (to be multiplied by ITEM_THEFT_SCALE).
    """
    players  = obs["players"]
    adjusted = 0.0
    for i, p in enumerate(players):
        if i == agent_id or int(p[2]) != 1:
            continue
        ep      = (int(p[0]), int(p[1]))
        w       = _enemy_intent(tracker, i, ep, grid, bomb_pos, obs).get(item_pos, 0.0)
        if w <= 0:
            continue
        e_dist = _enemy_bfs_dist(ep, item_pos, grid, bomb_pos)
        if e_dist is None:
            continue   # enemy can't even reach it
        lead = e_dist - my_bfs_dist
        if lead > 0:
            scale = 0.0          # we beat them there
        elif lead == 0:
            scale = 0.5          # tie → item destroyed, half penalty
        else:
            # Enemy arrives ahead by |lead| steps — penalty grows with their lead
            scale = min(1.0, 0.4 + 0.15 * abs(lead))
        adjusted += w * scale
    return adjusted



def _enemy_entering_blast(blast_tiles, tracker, obs, grid, bomb_pos, agent_id):
    """
    FIX-9: Returns True if any live enemy's BFS intent path passes through
    our blast zone within INTENT_FWD_STEPS steps.
    Used to add BOMB_CHASE_BONUS to bomb_value.
    """
    players = obs["players"]
    for i, p in enumerate(players):
        if i == agent_id or int(p[2]) != 1:
            continue
        ep = (int(p[0]), int(p[1]))
        # Short BFS from enemy, check if any node in first FWD_STEPS lands in blast
        visited = {ep}
        queue   = deque([(ep[0], ep[1], 0)])
        while queue:
            r, c, d = queue.popleft()
            if (r, c) in blast_tiles and (r, c) != ep:
                return True
            if d >= INTENT_FWD_STEPS:
                continue
            for dr, dc in ((-1,0),(1,0),(0,-1),(0,1)):
                nr, nc = r+dr, c+dc
                if (nr, nc) in visited:
                    continue
                if not _walkable(nr, nc, grid, bomb_pos):
                    continue
                visited.add((nr, nc))
                queue.append((nr, nc, d+1))
    return False


def _bfs_path(start, target, grid, bomb_pos, danger_by_time, depth_cap=BFS_DEPTH_CAP):
    """
    Returns a list of (r, c) cells from start to target (inclusive) if reachable,
    or None if not found within depth_cap.
    Uses the same walkability rules as _bfs_timed.
    """
    sr, sc = start
    tr, tc = target
    if start == target:
        return [start]

    visited = {(sr, sc)}
    queue = deque()
    # store (r, c, path_so_far)
    # we'll store parent pointers to reconstruct path later
    parent = {(sr, sc): None}

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
        parent[(nr, nc)] = (sr, sc)
        queue.append((nr, nc, 1))

    while queue:
        r, c, t = queue.popleft()
        if (r, c) == target:
            # reconstruct path
            path = []
            cur = (r, c)
            while cur is not None:
                path.append(cur)
                cur = parent.get(cur)
            path.reverse()
            return path

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
            parent[(nr, nc)] = (r, c)
            queue.append((nr, nc, t_next))

    return None


def _open_neighbors(r, c, grid, bomb_pos):
    """
    Count walkable neighbouring cells (4 directions) for cell (r, c).
    Walkable means: inside bounds, not a wall, not a box, and not occupied by a bomb.
    """
    h, w = grid.shape
    cnt = 0
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        nr, nc = r + dr, c + dc
        # Check bounds
        if not (0 <= nr < h and 0 <= nc < w):
            continue
        # Check if cell is walkable
        if grid[nr, nc] in (WALL, BOX):
            continue
        if (nr, nc) in bomb_pos:
            continue
        cnt += 1
    return cnt

def _reachable_cells_with_path(start, grid, bomb_pos, danger_by_time, depth_cap=BFS_DEPTH_CAP):
    """
    BFS from start, returns (paths, distances).
    paths[cell] = list of cells from start to cell (inclusive).
    distances[cell] = number of steps.
    """
    sr, sc = start
    visited = {(sr, sc)}
    parent = {(sr, sc): None}
    dist = {(sr, sc): 0}
    q = deque([(sr, sc, 0)])

    while q:
        r, c, d = q.popleft()
        if d >= depth_cap:
            continue
        d_next = d + 1
        danger_next = danger_by_time.get(d_next, set())
        for dr, dc in ((-1,0),(1,0),(0,-1),(0,1)):
            nr, nc = r+dr, c+dc
            if (nr, nc) in visited:
                continue
            if not _walkable(nr, nc, grid, bomb_pos):
                continue
            if (nr, nc) in danger_next:
                continue
            visited.add((nr, nc))
            parent[(nr, nc)] = (r, c)
            dist[(nr, nc)] = d_next
            q.append((nr, nc, d_next))

    # Reconstruct paths
    paths = {}
    for cell in visited:
        path = []
        cur = cell
        while cur is not None:
            path.append(cur)
            cur = parent.get(cur)
        path.reverse()
        paths[cell] = path
    return paths, dist

# ===========================================================================
# Agent
# ===========================================================================

class Agent:
    def __init__(self, agent_id: int):
        self.agent_id    = int(agent_id)
        self.step_count  = 0
        self.pos_history = deque(maxlen=20)
        self.last_pos    = None
        # FIX-7/8/9: enemy intent tracker (4 agents in Bomberland)
        self.enemy_tracker = EnemyTracker(n_agents=4)

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

        # FIX-7/8/9: update enemy tracker, compute intent weights ONCE per step
        self.enemy_tracker.update(players)
        bomb_pos_for_intent = {(int(b[0]), int(b[1])) for b in bombs_a}
        enemy_intent_weights = _all_enemy_intent(
            self.enemy_tracker, obs, grid, bomb_pos_for_intent, self.agent_id)

        # FIX-4: danger map now also returns cell_explode_min
        danger_by_time, danger_any, cell_explode_min = _build_danger_timed(obs)

        bomb_pos = {(int(b[0]), int(b[1])) for b in bombs_a}

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
        


        # ===== A1: TRAP-KILL (aggressive bomb before item) =====
        if AGGR and bombs_left > 0 and pos not in bomb_pos \
                and time.perf_counter() - t0 < TIME_BUDGET_S:
            # Use reachable distances computed earlier (or compute cheap now)
            reachable = _reachable_cells_with_dist(pos, grid, bomb_pos, danger_by_time)
            n_trap, n_in = _trap_kill_score(
                my_r, my_c, my_radius, grid, players, self.agent_id,
                bomb_pos, reachable_dist=reachable)
            if n_trap >= 1:
                can_esc = _can_escape_after_bomb(
                    my_r, my_c, my_radius, grid, bomb_pos,
                    danger_by_time, extra_danger_tiles=extra_danger)
                if can_esc:
                    f_pos, f_bp, f_dbt, f_da, _ = _project_future_state(
                        obs, self.agent_id, PLACE_BOMB, my_r, my_c,
                        my_radius, nearby_threat_pos)
                    if _action_survives_future(f_pos, grid, f_bp, f_dbt, f_da):
                        return PLACE_BOMB
        # ===== END A1 =====

        # -------------------------------------------------------------------
        # -------------------------------------------------------------------
        # PRIORITY 2: PICK UP ITEM (FIX-7) + DENIAL
        # -------------------------------------------------------------------
        if time.perf_counter() - t0 < TIME_BUDGET_S:
            h, w = grid.shape
            items = [(r, c) for r in range(h) for c in range(w)
                    if grid[r, c] in (ITEM_RADIUS, ITEM_CAPACITY)]
            if items:
                # Compute all reachable paths and distances once
                paths, distances = _reachable_cells_with_path(pos, grid, bomb_pos, danger_by_time)

                # Filter items that are close enough (within ITEM_MAX_CHASE_DIST)
                close_items = [it for it in items if distances.get(it, 999) <= ITEM_MAX_CHASE_DIST]
                no_box_left = not any(grid[r][c] == BOX for r in range(13) for c in range(13))

                candidate_items = close_items if (close_items or no_box_left) else []

                if candidate_items:
                    my_blast = _blast_tiles(my_r, my_c, my_radius, grid)

                    scored_items = []
                    for item_pos in candidate_items:
                        # Basic reward for picking up item (adjustable)
                        raw_score = 3.0

                        # Get precomputed path
                        path = paths.get(item_pos)
                        if path is None:
                            continue   # unreachable (should not happen if in distances)

                        # Count narrow cells along path (skip start)
                        narrow_count = 0
                        for cell in path[1:11]:   # exclude current position
                            if _open_neighbors(cell[0], cell[1], grid, bomb_pos) <= 2:
                                narrow_count += 1

                        path_penalty = min(1.5 * narrow_count, 6.0)
                        final_score = raw_score - path_penalty

                        if final_score >= 0:
                            scored_items.append((final_score, item_pos))

                    scored_items.sort(key=lambda x: x[0], reverse=True)

                    for score, item_pos in scored_items:
                        my_dist = distances.get(item_pos, 999)   # use precomputed distance

                        # --- DENIAL: enemy will get it first? ---
                        if bombs_left > 0 and pos not in bomb_pos:
                            enemy_min_dist = float('inf')
                            for i, p in enumerate(players):
                                if i == self.agent_id or int(p[2]) != 1:
                                    continue
                                ep = (int(p[0]), int(p[1]))
                                e_dist = _enemy_bfs_dist(ep, item_pos, grid, bomb_pos)
                                if e_dist is not None and e_dist < enemy_min_dist:
                                    enemy_min_dist = e_dist

                            if enemy_min_dist <= my_dist and item_pos in my_blast:
                                if _can_escape_after_bomb(my_r, my_c, my_radius, grid,
                                                        bomb_pos, danger_by_time):
                                    f_pos, f_bp, f_dbt, f_da, _ = _project_future_state(
                                        obs, self.agent_id, PLACE_BOMB, my_r, my_c,
                                        my_radius, nearby_threat_pos)
                                    if _action_survives_future(f_pos, grid, f_bp, f_dbt, f_da):
                                        return PLACE_BOMB

                        # --- Normal pickup attempt ---
                        a = _bfs_timed(pos, {item_pos}, grid, bomb_pos, danger_by_time)
                        if a is None or a == STOP:
                            continue
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
        # PRIORITY 3: PLACE BOMB  (FIX-9: chase bonus when enemy heads our way)
        # -------------------------------------------------------------------
        if bombs_left > 0 and pos not in bomb_pos:
            if time.perf_counter() - t0 < TIME_BUDGET_S:
                bval, hits_box, hits_enemy = _bomb_value_advanced(
                    my_r, my_c, my_radius, grid, players, self.agent_id, danger_any,
                    tracker=self.enemy_tracker, obs=obs, bomb_pos=bomb_pos) 

                # FIX-9: if an enemy is walking into our blast zone, boost value
                blast_now = _blast_tiles(my_r, my_c, my_radius, grid)
                if _enemy_entering_blast(blast_now, self.enemy_tracker, obs,
                                         grid, bomb_pos, self.agent_id):
                    bval += BOMB_CHASE_BONUS

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

        # -------------------------------------------------------------------
        # PRIORITY 4: FARM BOXES (reachable + local-first + far only if better)
        # -------------------------------------------------------------------
        if time.perf_counter() - t0 < TIME_BUDGET_S:
            # One BFS to get both distances and full paths
            paths, distances = _reachable_cells_with_path(pos, grid, bomb_pos, danger_by_time)

            raw_candidates = []
            for (r, c), bfs_dist in distances.items():
                # Only consider cells within a reasonable distance (≤8)
                if bfs_dist > 8:
                    continue
                # Skip walls/boxes and cells that explode immediately
                if grid[r, c] in (WALL, BOX):
                    continue
                if (r, c) in danger_by_time.get(0, set()):
                    continue
                if (r, c) in danger_by_time.get(1, set()) and bfs_dist <= 1:
                    continue

                # Compute raw score (no path penalty yet)
                raw_score = _box_farm_score(
                    r, c, grid, players, bombs_a,
                    self.agent_id, my_r, my_c, my_radius,
                    bomb_pos, danger_any, bfs_dist,
                    enemy_intent_weights=enemy_intent_weights
                )
                raw_candidates.append((raw_score, bfs_dist, (r, c)))

            # Keep only the top 20 raw scores (to limit expensive path processing)
            raw_candidates.sort(key=lambda x: x[0], reverse=True)
            raw_candidates = raw_candidates[:20]

            # Now compute path penalty for each kept candidate
            candidates = []
            for raw_score, bfs_dist, cell in raw_candidates:
                path = paths.get(cell)
                if path is None:
                    continue   # unreachable

                # Count narrow cells along the first 10 steps (exclude start)
                narrow_count = 0
                for step in path[1:11]:   # up to 10 cells after start
                    if _open_neighbors(step[0], step[1], grid, bomb_pos) <= 2:
                        narrow_count += 1

                penalty = min(1.5 * narrow_count, 6.0)
                final_score = raw_score - penalty
                candidates.append((final_score, bfs_dist, cell))

            if candidates:
                # Split into local vs far (same as before)
                local = [x for x in candidates if x[1] <= 4]
                far = [x for x in candidates if x[1] > 4]

                best_local = max(local, key=lambda x: x[0]) if local else None
                best_far = max(far, key=lambda x: x[0]) if far else None

                chosen = None
                if best_local is not None and best_far is not None:
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
                            my_radius, nearby_threat_pos
                        )
                        if _action_survives_future(f_pos, grid, f_bp, f_dbt, f_da):
                            return a
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
        # PRIORITY 5: CHASE NEAREST ENEMY
        # -------------------------------------------------------------------
        if time.perf_counter() - t0 < TIME_BUDGET_S:
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