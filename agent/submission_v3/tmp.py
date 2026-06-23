
"""
agent.py -- Bomberland Rule-Based Agent v8 "Precision Safety"
Base: v6 (enemy-bomb prediction C1, combat cluster C2, escape redundancy C3,
           stuck-detector / unstuck-bomb / anti-repeat C4).

Changes over v6 (and fixes for v7 regressions):

  E1. _reachable_safe_timed  (replaces v6 _reachable_safe_count)
      ---------------------------------------------------------------
      BFS up to SAFE_DEPTH steps from a position, tracking arrival time d.
      For each visited cell, look up its earliest explosion time expl_t:
        SAFE   : expl_t is None OR expl_t > d  → +REWARD_SAFE
        TEMP   : 0 < expl_t <= d               → +REWARD_TEMP  (cell blows up
                 before we arrive, but corridor was open when we passed through)
        DEAD   : cell in danger at arrival t=d  → not visited at all (pruned)
      Dead-end branches (leaf nodes with no onward moves) get -REWARD_SAFE.
      Key fix over v7: cells are only pruned if they explode AT step d (arrival),
      not merely because some future neighbour is dangerous; this prevents the
      v7 bug of cutting off entire branches prematurely.
      Pre-built per-cell explosion lookup (dict) avoids O(n) scan per cell.

  E2. _count_nearby_armed_bfs  (replaces v6 Manhattan version)
      ---------------------------------------------------------------
      BFS distance through walkable cells, radius BFS_COMBAT_RADIUS=3.
      Tighter than v7's radius=4 to avoid over-triggering combat mode in
      opening games where corridors make enemies far in practice.

  E3. Hypothetical danger map (_build_hypothetical_danger)
      ---------------------------------------------------------------
      Kept from v7 but used ONLY for Priority 1b (pre-emptive escape).
      NOT injected into bomb-placement escape check (that was v7's mistake —
      it made bomb placement nearly impossible because hypothetical enemy bombs
      always covered the escape routes).
      Priority 1b trigger tightened: only fires if our cell is in
      hypo_dbt.get(1) or hypo_dbt.get(2)  (imminent, not merely "any danger").

  E4. Item spawn probability bonus (_item_spawn_bonus)
      ---------------------------------------------------------------
      P = 0.0003 * (step / 165) per grass cell per step.
      Cells adjacent to many grass cells receive a small future-value bonus
      when navigating during farm mode (encourages open-area positioning).

  Unchanged from v6:
      C1 enemy-blast prediction, C2 combat-cluster threshold (>=2 nearby),
      C3 escape-direction count (>=2 dirs), C4 stuck/unstuck/anti-repeat.
"""
import time
from collections import deque

# --- Action constants ---------------------------------------------------------
STOP, LEFT, RIGHT, UP, DOWN, PLACE_BOMB = 0, 1, 2, 3, 4, 5

MOVES = {
    STOP:  ( 0,  0),
    LEFT:  (-1,  0),
    RIGHT: ( 1,  0),
    UP:    ( 0, -1),
    DOWN:  ( 0,  1),
}
MOVE_ACTIONS = [LEFT, RIGHT, UP, DOWN]

# --- Map cell types -----------------------------------------------------------
GRASS, WALL, BOX, ITEM_RADIUS, ITEM_CAPACITY = 0, 1, 2, 3, 4

# --- Game constants -----------------------------------------------------------
BOMB_TIMER   = 7
MAX_RADIUS   = 5
MAX_CAPACITY = 5

# --- Tuning ------------------------------------------------------------------
TIME_BUDGET_S             = 0.070
BFS_DEPTH_CAP             = 30
ESCAPE_DEPTH              = 25
BOMB_MIN_SCORE            = 1
BOMB_COMBAT_CLUSTER_SCORE = 4   # C2: threshold when >=2 armed enemies nearby
BFS_COMBAT_RADIUS         = 3   # E2: BFS steps for combat detection

# E1: tiered rewards for _reachable_safe_timed
REWARD_SAFE  = 10
REWARD_TEMP  =  3
SAFE_DEPTH   =  20

# E3: hypothetical danger – only trigger pre-emptive escape within this horizon
HYPO_IMMINENCE = 2   # check hypo_dbt.get(1) | hypo_dbt.get(2)

# E4: item spawn probability weight (multiplied by step count later)
ITEM_SPAWN_WEIGHT = 0.0003 / 165.0


# =============================================================================
# Geometry helpers
# =============================================================================

def _blast_tiles(bx, by, radius, grid):
    """Tiles in this bomb's blast zone. Stops at WALL, stops+includes BOX."""
    h, w = grid.shape
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
    """Can the agent step onto (r,c)? Blocked by WALL, BOX, live bombs."""
    h, w = grid.shape
    if not (0 <= r < h and 0 <= c < w):
        return False
    if grid[r, c] in (WALL, BOX):
        return False
    if (r, c) in bomb_pos:
        return False
    return True


# =============================================================================
# Timer-aware danger map
# =============================================================================

def _build_danger_timed(obs):
    """
    Returns (danger_by_time, danger_any, cell_explode_min).
    danger_by_time[t]  = set of tiles that explode at step t.
    danger_any         = union of all danger tiles.
    cell_explode_min   = dict mapping cell -> earliest explosion time  (E1 lookup).
    Chain reaction: if bomb B's position is in bomb A's blast, B.timer = min(A,B).
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


# =============================================================================
# E3: Hypothetical danger map (used ONLY for pre-emptive escape, not bombing)
# =============================================================================

def _build_hypothetical_danger(grid, bombs_a, players, agent_id, our_action):
    """
    E3: Simulate one tick ahead:
      - Existing bomb timers decremented by 1.
      - Every live armed enemy places a bomb at their current position
        (timer = BOMB_TIMER-1 after the tick, i.e. BOMB_TIMER-1 steps remain).
      - Chain reactions resolved.
    Returns (hypo_dbt, hypo_da, hypo_pos).
    hypo_pos: where we land after our_action (clamped to current if blocked).
    """
    n = len(players)
    me = players[agent_id]
    my_r, my_c = int(me[0]), int(me[1])

    cur_bomb_pos = {(int(b[0]), int(b[1])) for b in bombs_a}
    if our_action in MOVES and our_action != STOP:
        dr, dc = MOVES[our_action]
        nr, nc = my_r + dr, my_c + dc
        hypo_pos = (nr, nc) if _walkable(nr, nc, grid, cur_bomb_pos) \
                             else (my_r, my_c)
    else:
        hypo_pos = (my_r, my_c)

    # Tick existing bombs down
    bomb_list = []
    for b in bombs_a:
        bx, by, timer, oid = int(b[0]), int(b[1]), int(b[2]), int(b[3])
        oid_s  = max(0, min(n - 1, oid))
        radius = max(1, min(MAX_RADIUS, 1 + int(players[oid_s][4])))
        tiles  = _blast_tiles(bx, by, radius, grid)
        bomb_list.append({'pos': (bx, by), 'timer': max(0, timer - 1),
                          'tiles': tiles})

    existing_pos = {e['pos'] for e in bomb_list}
    for i, p in enumerate(players):
        if i == agent_id or int(p[2]) != 1 or int(p[3]) <= 0:
            continue
        ep = (int(p[0]), int(p[1]))
        if ep in existing_pos:
            continue
        radius = max(1, min(MAX_RADIUS, 1 + int(p[4])))
        tiles  = _blast_tiles(ep[0], ep[1], radius, grid)
        bomb_list.append({'pos': ep, 'timer': BOMB_TIMER - 1, 'tiles': tiles})

    if not bomb_list:
        return {}, set(), hypo_pos

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

    hypo_dbt = {}
    hypo_da  = set()
    for b in bomb_list:
        t = b['timer']
        if t not in hypo_dbt:
            hypo_dbt[t] = set()
        hypo_dbt[t] |= b['tiles']
        hypo_da     |= b['tiles']

    hypo_bomb_pos = {b['pos'] for b in bomb_list if b['timer'] > 0}

    return hypo_dbt, hypo_da, hypo_pos, hypo_bomb_pos


# =============================================================================
# Timer-aware BFS (navigation)
# =============================================================================

def _bfs_timed(start, targets, grid, bomb_pos, danger_by_time,
               depth_cap=BFS_DEPTH_CAP):
    """
    BFS tracking arrival time. Step is valid iff cell not in danger_by_time[t].
    Returns first action toward nearest target, or None.
    """
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
    """
    Time-expanded escape BFS; allows WAIT. visited keyed (r,c,t).
    Returns first action toward nearest safe target, or None.
    """
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


# =============================================================================
# Escape
# =============================================================================

def _escape_timed(pos, grid, bomb_pos, danger_by_time, danger_any):
    """3-pass escape: fully safe -> urgency-filtered -> any walkable move."""
    h, w = grid.shape

    safe = {(rr, cc) for rr in range(h) for cc in range(w)
            if grid[rr, cc] not in (WALL, BOX)
            and (rr, cc) not in bomb_pos
            and (rr, cc) not in danger_any}

    if safe:
        a = _bfs_escape(pos, safe, grid, bomb_pos, danger_by_time,
                        depth_cap=ESCAPE_DEPTH)
        if a is not None:
            return a

    urgent     = danger_by_time.get(1, set()) | danger_by_time.get(2, set())
    less_risky = {(rr, cc) for rr in range(h) for cc in range(w)
                  if grid[rr, cc] not in (WALL, BOX)
                  and (rr, cc) not in bomb_pos
                  and (rr, cc) not in urgent}

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

    sr, sc = pos
    for a in MOVE_ACTIONS:
        dr, dc = MOVES[a]
        nr, nc = sr + dr, sc + dc
        if _walkable(nr, nc, grid, bomb_pos):
            return a
    return STOP


# =============================================================================
# Bomb helpers
# =============================================================================

def _enemy_predicted_blast(my_r, my_c, players, agent_id, grid):
    """
    C1: Blast union for armed enemies within Manhattan BFS_COMBAT_RADIUS+1.
    Conservative over-estimate (Manhattan) keeps this fast.
    """
    predicted = set()
    for i, p in enumerate(players):
        if i == agent_id or int(p[2]) != 1 or int(p[3]) <= 0:
            continue
        er, ec = int(p[0]), int(p[1])
        if abs(er - my_r) + abs(ec - my_c) > BFS_COMBAT_RADIUS + 1:
            continue
        radius = max(1, min(MAX_RADIUS, 1 + int(p[4])))
        predicted |= _blast_tiles(er, ec, radius, grid)
    return predicted


def _can_escape_after_bomb(my_r, my_c, my_radius, grid, bomb_pos,
                            danger_by_time, extra_danger_tiles=None):
    """
    True if we can reach a fully-safe cell within eff_t steps after placing
    a bomb here.  extra_danger_tiles (C1) added at eff_t.
    """
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
    safe_cells = {(rr, cc) for rr in range(h) for cc in range(w)
                  if grid[rr, cc] not in (WALL, BOX)
                  and (rr, cc) not in new_bomb_p
                  and (rr, cc) not in new_danger_any}

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
    """
    C3: Count distinct first-move directions that reach safety after our bomb.
    """
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
    safe_cells = {(rr, cc) for rr in range(h) for cc in range(w)
                  if grid[rr, cc] not in (WALL, BOX)
                  and (rr, cc) not in new_bomb_p
                  and (rr, cc) not in new_danger_any}

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


# =============================================================================
# E1: Timer-aware reachable safe count
# =============================================================================

def _reachable_safe_timed(pos, grid, bomb_pos, cell_explode_min,
                           danger_by_time, depth=SAFE_DEPTH):
    """
    E1: BFS up to `depth` steps. For each cell reached at arrival time d:
      - Prune if cell explodes AT step d (danger_by_time.get(d)) — we'd die.
      - SAFE  : cell_explode_min.get(cell) > d  OR  cell not in danger  → +REWARD_SAFE
      - TEMP  : cell_explode_min.get(cell) <= d (explodes before arrival) → +REWARD_TEMP
      - Dead-end penalty: leaf node (d < depth, no onward moves)         → -REWARD_SAFE

    cell_explode_min pre-built by _build_danger_timed for O(1) lookup.
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
                # Would explode exactly when we step there — skip
                continue
            has_fwd = True
            visited.add((nr, nc))
            queue.append((nr, nc, d_next))

        if not has_fwd and d < depth:
            score -= REWARD_SAFE   # dead-end penalty

    return score


# =============================================================================
# E2: BFS-based nearby armed enemy count
# =============================================================================

def _count_nearby_armed_bfs(my_r, my_c, players, agent_id, grid,
                             radius=BFS_COMBAT_RADIUS):
    """
    E2: Count live armed enemies reachable within `radius` BFS steps.
    Does NOT include bomb_pos in blocking — enemies can stand on empty cells.
    """
    armed_enemy_pos = set()
    for i, p in enumerate(players):
        if i == agent_id or int(p[2]) != 1 or int(p[3]) <= 0:
            continue
        armed_enemy_pos.add((int(p[0]), int(p[1])))

    if not armed_enemy_pos:
        return 0

    visited = {(my_r, my_c)}
    queue   = deque([(my_r, my_c, 0)])
    count   = 0
    h, w    = grid.shape

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


# =============================================================================
# E4: Item spawn bonus
# =============================================================================

def _item_spawn_bonus(r, c, grid, step_count):
    """
    E4: Expected future item value at (r,c).
    Counts adjacent+self grass cells and scales by spawn probability.
    Returns a small float bonus to add to position scores during farm navigation.
    """
    h, w  = grid.shape
    p_per = ITEM_SPAWN_WEIGHT * step_count   # P per grass cell per step
    grass = sum(
        1 for dr, dc in ((0,0),(-1,0),(1,0),(0,-1),(0,1))
        if (0 <= r+dr < h and 0 <= c+dc < w and grid[r+dr, c+dc] == GRASS)
    )
    return p_per * grass * 2.0   # *2 for small nudge toward open areas


# =============================================================================
# Bomb value scoring
# =============================================================================

def _bomb_value(my_r, my_c, my_radius, grid, players, agent_id, danger_any):
    """
    Score a potential bomb at (my_r, my_c).
    Returns (score, hits_box, hits_enemy).
    boxes*1 + enemies*3 - corridor_pen(2 if <=1 safe adj tile).
    """
    blast   = _blast_tiles(my_r, my_c, my_radius, grid)
    boxes   = sum(1 for r, c in blast if grid[r, c] == BOX)
    enemies = sum(
        1 for i in range(len(players))
        if i != agent_id
        and int(players[i][2]) == 1
        and (int(players[i][0]), int(players[i][1])) in blast
    )
    h, w = grid.shape
    adj_safe = sum(
        1 for dr, dc in ((-1,0),(1,0),(0,-1),(0,1))
        if (0 <= my_r+dr < h and 0 <= my_c+dc < w
            and grid[my_r+dr,my_c+dc] == GRASS
            and (my_r+dr, my_c+dc) not in danger_any)
    )
    corridor_pen = 2 if adj_safe <= 1 else 0
    score = boxes * 1 + enemies * 3 - corridor_pen
    return score, boxes > 0, enemies > 0


# =============================================================================
# Enemy proximity helpers
# =============================================================================

def _enemy_adjacent(my_r, my_c, players, agent_id):
    """True if any live armed enemy is within Manhattan 2."""
    for i, p in enumerate(players):
        if i == agent_id or int(p[2]) != 1 or int(p[3]) <= 0:
            continue
        er, ec = int(p[0]), int(p[1])
        if abs(er - my_r) + abs(ec - my_c) <= 2:
            return True
    return False


# =============================================================================
# Safe fallback
# =============================================================================

def _safe_fallback(pos, grid, bomb_pos, danger_by_time, cell_explode_min,
                   step_count, agent_id, last_pos=None):
    """
    Score each candidate move with _reachable_safe_timed + E4 item bonus.
    Hard-skip dead-end moves (open_neighbors <= 1) unless all are dead-ends.
    Prefer non-STOP; deterministic tiebreak. C4c: penalise last_pos by -15.
    """
    sr, sc     = pos
    imm_danger = danger_by_time.get(1, set())
    any_danger = set()
    for tiles in danger_by_time.values():
        any_danger |= tiles

    def _score(r, c, is_stop, a):
        reach  = _reachable_safe_timed(
            (r, c), grid, bomb_pos, cell_explode_min,
            danger_by_time, depth=SAFE_DEPTH)
        open_n = sum(
            1 for dr2, dc2 in ((-1,0),(1,0),(0,-1),(0,1))
            if _walkable(r+dr2, c+dc2, grid, bomb_pos)
        )
        spawn_b    = _item_spawn_bonus(r, c, grid, step_count)
        penalty    = REWARD_SAFE if (r, c) in any_danger else 0
        repeat_pen = 15 if (not is_stop) and last_pos is not None \
                          and (r, c) == last_pos else 0
        tb = (step_count * 13 + agent_id * 7 + r * 5 + c * 3 + a * 11) % 97
        return reach + open_n + spawn_b - penalty - repeat_pen, open_n, tb

    candidates          = []
    dead_end_candidates = []

    if (sr, sc) not in imm_danger:
        sc_s, sc_o, sc_tb = _score(sr, sc, True, STOP)
        candidates.append((STOP, sc_s, True, sc_o, sc_tb))

    for a in MOVE_ACTIONS:
        dr, dc = MOVES[a]
        nr, nc = sr + dr, sc + dc
        if not _walkable(nr, nc, grid, bomb_pos):
            continue
        if (nr, nc) in imm_danger:
            continue
        mv_s, mv_o, mv_tb = _score(nr, nc, False, a)
        if mv_o <= 1:
            dead_end_candidates.append((a, mv_s, False, mv_o, mv_tb))
        else:
            candidates.append((a, mv_s, False, mv_o, mv_tb))

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

def _cell_explode_min_from_dbt(danger_by_time):
    cem = {}
    for t, cells in danger_by_time.items():
        for cell in cells:
            if cell not in cem or t < cem[cell]:
                cem[cell] = t
    return cem

# =============================================================================
# Agent
# =============================================================================

class Agent:
    def __init__(self, agent_id: int):
        self.agent_id    = int(agent_id)
        self.step_count  = 0
        self.pos_history = deque(maxlen=20)
        self.last_pos    = None

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

        me = players[self.agent_id]
        my_r, my_c   = int(me[0]), int(me[1])
        alive        = int(me[2])
        bombs_left   = int(me[3])
        radius_bonus = int(me[4])

        if not alive:
            return STOP

        my_radius = max(1, min(MAX_RADIUS, 1 + radius_bonus))
        pos       = (my_r, my_c)

        self.pos_history.append(pos)
        prev_pos      = self.last_pos
        self.last_pos = pos

        # Real danger map (with cell_explode_min for O(1) E1 lookup)
        danger_by_time, danger_any, cell_explode_min = _build_danger_timed(obs)
        bomb_pos = {(int(b[0]), int(b[1])) for b in bombs_a}

        # E2: BFS-based combat mode
        nearby_armed = _count_nearby_armed_bfs(
            my_r, my_c, players, self.agent_id, grid)
        combat_mode  = nearby_armed >= 1

        # C1: predicted enemy blasts (combat only)
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

        # -----------------------------------------------------------------------
        # PRIORITY 1: ESCAPE (real danger)
        # -----------------------------------------------------------------------
        in_immediate = pos in danger_by_time.get(1, set())
        in_any       = pos in danger_any

        if in_immediate or in_any:
            return _escape_timed(pos, grid, bomb_pos, danger_by_time, danger_any)

        # -----------------------------------------------------------------------
        # PRIORITY 1b: PRE-EMPTIVE ESCAPE (E3 hypothetical, combat only)
        # Only triggers if our current cell becomes imminently dangerous
        # (within HYPO_IMMINENCE ticks) under the worst-case enemy bomb scenario.
        # -----------------------------------------------------------------------
        if combat_mode and time.perf_counter() - t0 < TIME_BUDGET_S:
            hypo_dbt, hypo_da, _, hypo_bomb_pos = _build_hypothetical_danger(
                grid, bombs_a, players, self.agent_id, STOP)
            
            hypo_cem = _cell_explode_min_from_dbt(hypo_dbt)

            imm_hypo = set()
            for t in range(1, HYPO_IMMINENCE + 1):
                imm_hypo |= hypo_dbt.get(t, set())

            if pos in imm_hypo:
                # Pick the move that maximises safe space under hypothetical danger
                best_a     = None
                best_score = -1e9
                for action in MOVE_ACTIONS:
                    dr, dc = MOVES[action]
                    nr, nc = my_r + dr, my_c + dc
                    if not _walkable(nr, nc, grid, bomb_pos):
                        continue
                    if (nr, nc) in hypo_dbt.get(1, set()):
                        continue
                    sc = _reachable_safe_timed(
                        (nr, nc), grid, hypo_bomb_pos, hypo_cem,
                        hypo_dbt, depth=SAFE_DEPTH)
                    if sc > best_score:
                        best_score = sc
                        best_a     = action
                if best_a is not None:
                    return best_a

        # -----------------------------------------------------------------------
        # PRIORITY 2: PICK UP ITEM
        # -----------------------------------------------------------------------
        if time.perf_counter() - t0 < TIME_BUDGET_S:
            h, w  = grid.shape
            items = {(r, c) for r in range(h) for c in range(w)
                     if grid[r, c] in (ITEM_RADIUS, ITEM_CAPACITY)}
            if items:
                a = _bfs_timed(pos, items, grid, bomb_pos, danger_by_time)
                if a is not None and a != STOP:
                    return a

        # -----------------------------------------------------------------------
        # C4b: UNSTUCK BOMB (farm mode only)
        # -----------------------------------------------------------------------
        if is_stuck and not combat_mode and bombs_left > 0 and pos not in bomb_pos:
            if time.perf_counter() - t0 < TIME_BUDGET_S:
                blast   = _blast_tiles(my_r, my_c, my_radius, grid)
                has_box = any(grid[r][c] == BOX for r, c in blast)
                if has_box and _can_escape_after_bomb(
                        my_r, my_c, my_radius, grid, bomb_pos,
                        danger_by_time, extra_danger_tiles=None):
                    return PLACE_BOMB

        # -----------------------------------------------------------------------
        # PRIORITY 3: PLACE BOMB
        # Uses real danger_by_time (NOT hypothetical) for escape check — avoids
        # the v7 regression where hypothetical enemy bombs blocked all escapes.
        # -----------------------------------------------------------------------
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
                        danger_by_time,           # real map, not hypothetical
                        extra_danger_tiles=extra_danger)

                    if can_esc and combat_mode:
                        can_esc = _count_escape_first_moves(
                            my_r, my_c, my_radius, grid, bomb_pos,
                            danger_by_time,
                            extra_danger_tiles=extra_danger) >= 2

                    if can_esc:
                        return PLACE_BOMB

        # -----------------------------------------------------------------------
        # PRIORITY 4: FARM BOXES
        # -----------------------------------------------------------------------
        if time.perf_counter() - t0 < TIME_BUDGET_S:
            h, w      = grid.shape
            box_spots = set()
            for r in range(h):
                for c in range(w):
                    if grid[r, c] != BOX:
                        continue
                    for dr, dc in ((-1,0),(1,0),(0,-1),(0,1)):
                        nr, nc = r + dr, c + dc
                        if (0 <= nr < h and 0 <= nc < w
                                and grid[nr, nc] not in (WALL, BOX)
                                and (nr, nc) not in bomb_pos
                                and (nr, nc) not in danger_any):
                            box_spots.add((nr, nc))

            if box_spots and pos not in box_spots:
                a = _bfs_timed(pos, box_spots, grid, bomb_pos, danger_by_time)
                if a is not None and a != STOP:
                    return a

        # -----------------------------------------------------------------------
        # PRIORITY 5: CHASE NEAREST ENEMY
        # -----------------------------------------------------------------------
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
                        return a

        # -----------------------------------------------------------------------
        # FALLBACK: anti-corner safe move
        # -----------------------------------------------------------------------
        return _safe_fallback(pos, grid, bomb_pos, danger_by_time,
                              cell_explode_min, self.step_count,
                              self.agent_id, last_pos=prev_pos)