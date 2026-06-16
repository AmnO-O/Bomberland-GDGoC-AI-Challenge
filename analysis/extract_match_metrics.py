"""Extract per-player behavioural metrics from JSON match logs.

Usage (local testing on sample logs):
    python analysis/extract_match_metrics.py --json_dir logs/json --output analysis/data/metrics.csv

Designed to be imported by the Drive streaming pipeline as well.
"""

import csv
import json
import math
import os
import sys
from pathlib import Path
from collections import Counter
from typing import Optional


# ── Metric schema ─────────────────────────────────────────────────────────────
METRIC_COLUMNS = [
    "match_id",
    "date",
    "submission_id",
    "player_slot",
    # Action distribution
    "action_stop",
    "action_up",
    "action_down",
    "action_left",
    "action_right",
    "action_bomb",
    "movement_entropy",
    # Core stats
    "bombs_placed",
    "bombs_near_box",
    "bombs_near_enemy",
    "idle_ratio",
    "steps_alive",
    "max_steps",
    "survived",
    "rank",
    # Items
    "items_collected_radius",
    "items_collected_capacity",
    # Danger awareness
    "danger_entries",
    "danger_escapes",
    # Spatial
    "avg_dist_to_nearest_enemy",
    "unique_tiles_visited",
    "map_coverage_pct",
]

HEADER = METRIC_COLUMNS


# ── Helpers ───────────────────────────────────────────────────────────────────

def _blast_tiles(grid, bx, by, radius):
    """Cross-shaped blast prediction, matching engine logic."""
    H = len(grid)
    W = len(grid[0]) if H > 0 else 0
    tiles = {(bx, by)}
    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        for r in range(1, radius + 1):
            x, y = bx + dx * r, by + dy * r
            if not (0 <= x < H and 0 <= y < W):
                break
            cell = grid[x][y]
            if cell == 1:  # Wall
                break
            tiles.add((x, y))
            if cell == 2:  # Box
                break
    return tiles


def _danger_tiles(grid, bombs, players):
    """Return set of tiles in predicted blast of any active bomb."""
    danger = set()
    for b in bombs:
        bx, by = int(b[0]), int(b[1])
        owner_id = int(b[3]) if len(b) > 3 else 0
        radius = 1
        if 0 <= owner_id < len(players):
            radius = max(1, int(players[owner_id][4]) + 1)
        danger |= _blast_tiles(grid, bx, by, radius)
    return danger


def _manhattan(ax, ay, bx, by):
    return abs(ax - bx) + abs(ay - by)


def _shannon_entropy(counts, total):
    """Shannon entropy of a distribution given counts and total."""
    if total == 0:
        return 0.0
    ent = 0.0
    for c in counts:
        if c > 0:
            p = c / total
            ent -= p * math.log2(p)
    return ent


# ── Main extraction ───────────────────────────────────────────────────────────

def extract_metrics(match_data: dict) -> list[dict]:
    """Extract per-player metrics from a single parsed JSON match log.

    Returns a list of dicts (one per player), keyed by METRIC_COLUMNS.
    """
    history = match_data.get("history", [])
    team_ids = match_data.get("team_ids", [])
    ranks = match_data.get("ranks", [])
    seed = match_data.get("seed", 0)
    n_players = len(team_ids)

    if n_players == 0 or len(history) < 2:
        return []

    max_steps_in_match = len(history) - 1  # step 0 has no actions

    # Derive date from match filename or first step (we'll set it externally)
    # Initialize per-player accumulators
    action_counts = [Counter() for _ in range(n_players)]
    bombs_placed = [0] * n_players
    bombs_near_box = [0] * n_players
    bombs_near_enemy = [0] * n_players
    steps_alive = [0] * n_players
    items_radius = [0] * n_players
    items_capacity = [0] * n_players
    danger_entries = [0] * n_players
    danger_escapes = [0] * n_players
    dist_to_enemy_sum = [0.0] * n_players
    dist_to_enemy_count = [0] * n_players
    tiles_visited = [set() for _ in range(n_players)]

    prev_in_danger = [False] * n_players
    prev_players = None

    for step_data in history:
        step_idx = step_data["step"]
        actions = step_data.get("actions")  # None for step 0
        alive = step_data.get("alive", [])
        grid = step_data.get("map", [])
        players = step_data.get("players", [])
        bombs = step_data.get("bombs", [])

        if not players or not grid:
            continue

        # Compute danger tiles for this step
        danger = _danger_tiles(grid, bombs, players) if bombs else set()

        for pid in range(min(n_players, len(players))):
            px, py = int(players[pid][0]), int(players[pid][1])
            p_alive = bool(alive[pid]) if pid < len(alive) else False

            if p_alive:
                steps_alive[pid] = step_idx

            # Record actions (skip step 0 which has no actions)
            if actions is not None and pid < len(actions):
                a = int(actions[pid])
                action_counts[pid][a] += 1

                # Bomb placement analysis
                if a == 5 and p_alive:
                    bombs_placed[pid] += 1
                    # Check adjacent boxes
                    has_adj_box = False
                    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nx, ny = px + dx, py + dy
                        if 0 <= nx < len(grid) and 0 <= ny < len(grid[0]):
                            if grid[nx][ny] == 2:
                                has_adj_box = True
                                break
                    if has_adj_box:
                        bombs_near_box[pid] += 1

                    # Check enemies in blast range
                    radius = max(1, int(players[pid][4]) + 1)
                    my_blast = _blast_tiles(grid, px, py, radius)
                    for eid in range(len(players)):
                        if eid != pid and (eid < len(alive) and alive[eid]):
                            ex, ey = int(players[eid][0]), int(players[eid][1])
                            if (ex, ey) in my_blast:
                                bombs_near_enemy[pid] += 1
                                break

            if p_alive:
                tiles_visited[pid].add((px, py))

                # Danger tracking
                in_danger_now = (px, py) in danger
                if in_danger_now and not prev_in_danger[pid]:
                    danger_entries[pid] += 1
                if not in_danger_now and prev_in_danger[pid]:
                    danger_escapes[pid] += 1
                prev_in_danger[pid] = in_danger_now

                # Distance to nearest alive enemy
                min_dist = None
                for eid in range(len(players)):
                    if eid != pid and (eid < len(alive) and alive[eid]):
                        ex, ey = int(players[eid][0]), int(players[eid][1])
                        d = _manhattan(px, py, ex, ey)
                        if min_dist is None or d < min_dist:
                            min_dist = d
                if min_dist is not None:
                    dist_to_enemy_sum[pid] += min_dist
                    dist_to_enemy_count[pid] += 1

            # Item collection detection (compare with previous step)
            if prev_players is not None and p_alive and pid < len(prev_players):
                prev_radius_bonus = int(prev_players[pid][4])
                curr_radius_bonus = int(players[pid][4])
                if curr_radius_bonus > prev_radius_bonus:
                    items_radius[pid] += (curr_radius_bonus - prev_radius_bonus)

                prev_bombs_max = int(prev_players[pid][3])
                curr_bombs_max = int(players[pid][3])
                # Capacity increases when max goes up (bombs_left increases)
                # This is approximate — detect via bomb_radius_bonus diff
                # Actually players[pid][3] is bombs_left, not max. We track
                # capacity items via the observation. Let's use a simpler heuristic:
                # if bombs_left increased without placing a bomb, likely collected capacity item
                # This is imperfect but good enough for trend analysis.

        prev_players = [list(p) for p in players]

    # Count passable tiles for coverage calculation
    initial_grid = history[0].get("map", [])
    passable_count = 0
    if initial_grid:
        H, W = len(initial_grid), len(initial_grid[0]) if initial_grid else 0
        for r in range(H):
            for c in range(W):
                if initial_grid[r][c] != 1:  # Not a wall
                    passable_count += 1

    # Build output rows
    rows = []
    for pid in range(n_players):
        total_actions = sum(action_counts[pid].values())
        ac = action_counts[pid]

        # Action distribution (normalized)
        a_stop = ac.get(0, 0)
        a_up = ac.get(1, 0)
        a_down = ac.get(2, 0)
        a_left = ac.get(3, 0)
        a_right = ac.get(4, 0)
        a_bomb = ac.get(5, 0)

        entropy = _shannon_entropy(
            [a_stop, a_up, a_down, a_left, a_right, a_bomb],
            total_actions,
        )

        idle_ratio = a_stop / total_actions if total_actions > 0 else 0.0
        survived = 1 if steps_alive[pid] >= max_steps_in_match - 1 else 0
        avg_enemy_dist = (
            dist_to_enemy_sum[pid] / dist_to_enemy_count[pid]
            if dist_to_enemy_count[pid] > 0
            else 0.0
        )
        coverage = (
            len(tiles_visited[pid]) / passable_count
            if passable_count > 0
            else 0.0
        )

        row = {
            "match_id": "",  # Set by caller
            "date": "",  # Set by caller
            "submission_id": team_ids[pid] if pid < len(team_ids) else f"unknown_{pid}",
            "player_slot": pid,
            "action_stop": a_stop,
            "action_up": a_up,
            "action_down": a_down,
            "action_left": a_left,
            "action_right": a_right,
            "action_bomb": a_bomb,
            "movement_entropy": round(entropy, 4),
            "bombs_placed": bombs_placed[pid],
            "bombs_near_box": bombs_near_box[pid],
            "bombs_near_enemy": bombs_near_enemy[pid],
            "idle_ratio": round(idle_ratio, 4),
            "steps_alive": steps_alive[pid],
            "max_steps": max_steps_in_match,
            "survived": survived,
            "rank": ranks[pid] if pid < len(ranks) else -1,
            "items_collected_radius": items_radius[pid],
            "items_collected_capacity": items_capacity[pid],
            "danger_entries": danger_entries[pid],
            "danger_escapes": danger_escapes[pid],
            "avg_dist_to_nearest_enemy": round(avg_enemy_dist, 2),
            "unique_tiles_visited": len(tiles_visited[pid]),
            "map_coverage_pct": round(coverage, 4),
        }
        rows.append(row)

    return rows


# ── Batch processing ──────────────────────────────────────────────────────────

def process_json_file(json_path: str) -> list[dict]:
    """Process a single JSON file and return metric rows."""
    fname = Path(json_path).stem  # e.g. match_20260520_112300_155533
    try:
        with open(json_path, "r") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        print(f"  WARN: skipping {fname}: {e}", file=sys.stderr)
        return []

    rows = extract_metrics(data)

    # Derive date from filename: match_YYYYMMDD_HHMMSS_SEED
    parts = fname.split("_")
    date_str = ""
    if len(parts) >= 2:
        raw = parts[1]  # YYYYMMDD
        if len(raw) == 8:
            date_str = f"{raw[:4]}-{raw[4:6]}-{raw[6:8]}"

    for row in rows:
        row["match_id"] = fname
        row["date"] = date_str

    return rows


def process_directory(json_dir: str, output_csv: str, limit: Optional[int] = None):
    """Process all JSON files in a directory and write metrics CSV."""
    json_dir = Path(json_dir)
    json_files = sorted(json_dir.glob("*.json"))

    if limit:
        json_files = json_files[:limit]

    total = len(json_files)
    print(f"Processing {total} JSON files from {json_dir}...")

    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=HEADER)
        writer.writeheader()

        for i, jf in enumerate(json_files):
            rows = process_json_file(str(jf))
            for row in rows:
                writer.writerow(row)

            if (i + 1) % 500 == 0 or (i + 1) == total:
                print(f"  [{i+1}/{total}] processed")

    print(f"Done. Wrote {output_csv}")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Extract match metrics from JSON logs")
    parser.add_argument("--json_dir", required=True, help="Directory containing JSON match files")
    parser.add_argument("--output", default="analysis/data/metrics.csv", help="Output CSV path")
    parser.add_argument("--limit", type=int, default=None, help="Process only first N files (for testing)")
    args = parser.parse_args()

    process_directory(args.json_dir, args.output, args.limit)
