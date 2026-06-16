"""Analyze the production competition.db for aggregate stats and rating timelines.

This script expects the PRODUCTION database (with match_results populated).
On the VM, this is `competition.db` in the project root.
Locally, you'll need to download a backup from Drive first.

Usage:
    python analysis/db_analysis.py --db_path competition.db --output_dir analysis/data
"""

import csv
import json
import os
import sqlite3
import sys
from pathlib import Path


def analyze_db(db_path: str, output_dir: str):
    """Run all DB analyses and write results to output_dir."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()

    # ── 1. Basic stats ────────────────────────────────────────────────────────
    print("=== Basic Stats ===")

    c.execute("SELECT COUNT(*) FROM teams WHERE status='active'")
    n_teams = c.fetchone()[0]
    print(f"  Active teams: {n_teams}")

    c.execute("SELECT COUNT(*) FROM submissions WHERE is_baseline=0")
    n_student_subs = c.fetchone()[0]
    c.execute("SELECT COUNT(*) FROM submissions WHERE is_baseline=1")
    n_baseline_subs = c.fetchone()[0]
    print(f"  Student submissions: {n_student_subs}")
    print(f"  Baseline submissions: {n_baseline_subs}")

    c.execute("SELECT COUNT(*) FROM match_results")
    n_matches = c.fetchone()[0]
    print(f"  Total matches: {n_matches}")

    c.execute("SELECT MIN(created_at), MAX(created_at) FROM match_results")
    row = c.fetchone()
    print(f"  Date range: {row[0]} to {row[1]}")

    # ── 2. Matches per day ────────────────────────────────────────────────────
    print("\n=== Matches Per Day ===")
    c.execute("""
        SELECT substr(created_at, 1, 10) as day, COUNT(*) as matches
        FROM match_results
        GROUP BY day ORDER BY day
    """)
    daily_rows = c.fetchall()
    daily_csv = output_dir / "matches_per_day.csv"
    with open(daily_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["date", "matches"])
        for row in daily_rows:
            writer.writerow([row["day"], row["matches"]])
            print(f"  {row['day']}: {row['matches']}")
    print(f"  → {daily_csv}")

    # ── 3. Match types ────────────────────────────────────────────────────────
    print("\n=== Match Types ===")
    c.execute("SELECT match_type, COUNT(*) as cnt FROM match_results GROUP BY match_type")
    for row in c.fetchall():
        print(f"  {row['match_type']}: {row['cnt']}")

    # ── 4. Submissions per team ───────────────────────────────────────────────
    print("\n=== Submissions Per Team ===")
    c.execute("""
        SELECT t.team_name, t.canonical_team_id,
               COUNT(s.submission_id) as n_subs,
               s.is_baseline
        FROM teams t
        JOIN submissions s ON t.canonical_team_id = s.canonical_team_id
        GROUP BY t.canonical_team_id
        ORDER BY n_subs DESC
    """)
    teams_csv = output_dir / "submissions_per_team.csv"
    with open(teams_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["team_name", "team_id", "submissions", "is_baseline"])
        for row in c.fetchall():
            writer.writerow([row["team_name"], row["canonical_team_id"],
                             row["n_subs"], row["is_baseline"]])
            if not row["is_baseline"]:
                print(f"  {row['team_name']}: {row['n_subs']} submissions")
    print(f"  → {teams_csv}")

    # ── 5. Current leaderboard snapshot ───────────────────────────────────────
    print("\n=== Leaderboard Snapshot ===")
    c.execute("""
        SELECT t.team_name, s.submission_id, s.mu, s.sigma,
               (s.mu - 3*s.sigma) as score, s.n_games,
               s.wins, s.draws, s.losses, s.is_baseline, s.is_team_best
        FROM submissions s
        JOIN teams t ON s.canonical_team_id = t.canonical_team_id
        WHERE s.validation_status = 'valid' AND s.is_team_best = 1
        ORDER BY score DESC
    """)
    lb_rows = c.fetchall()
    lb_csv = output_dir / "leaderboard_snapshot.csv"
    with open(lb_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["rank", "team_name", "submission_id", "mu", "sigma",
                          "score", "n_games", "wins", "draws", "losses", "is_baseline"])
        for i, row in enumerate(lb_rows, 1):
            writer.writerow([i, row["team_name"], row["submission_id"],
                             f"{row['mu']:.2f}", f"{row['sigma']:.2f}",
                             f"{row['score']:.2f}", row["n_games"],
                             row["wins"], row["draws"], row["losses"],
                             row["is_baseline"]])
            tag = " [B]" if row["is_baseline"] else ""
            print(f"  #{i} {row['team_name']}{tag}: score={row['score']:.2f} "
                  f"(mu={row['mu']:.2f}, σ={row['sigma']:.2f}) games={row['n_games']}")
    print(f"  → {lb_csv}")

    # ── 6. Submission timeline ────────────────────────────────────────────────
    print("\n=== Submission Timeline ===")
    c.execute("""
        SELECT s.submission_id, t.team_name, s.canonical_team_id,
               s.created_at, s.validation_status, s.is_baseline,
               s.mu, s.sigma, s.n_games
        FROM submissions s
        JOIN teams t ON s.canonical_team_id = t.canonical_team_id
        WHERE s.is_baseline = 0
        ORDER BY s.created_at
    """)
    timeline_rows = c.fetchall()
    timeline_csv = output_dir / "submission_timeline.csv"
    with open(timeline_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["submission_id", "team_name", "team_id", "created_at",
                          "validation_status", "mu", "sigma", "n_games"])
        for row in timeline_rows:
            writer.writerow([row["submission_id"], row["team_name"],
                             row["canonical_team_id"], row["created_at"],
                             row["validation_status"],
                             f"{row['mu']:.2f}" if row["mu"] else "",
                             f"{row['sigma']:.2f}" if row["sigma"] else "",
                             row["n_games"]])
    print(f"  {len(timeline_rows)} student submissions")
    print(f"  → {timeline_csv}")

    # ── 7. Per-submission win rates (for all active subs) ─────────────────────
    print("\n=== Per-Submission Performance ===")
    c.execute("""
        SELECT s.submission_id, t.team_name, s.canonical_team_id,
               s.mu, s.sigma, (s.mu - 3*s.sigma) as score,
               s.n_games, s.wins, s.draws, s.losses,
               s.is_baseline, s.is_active, s.created_at
        FROM submissions s
        JOIN teams t ON s.canonical_team_id = t.canonical_team_id
        WHERE s.validation_status = 'valid' AND s.n_games > 0
        ORDER BY score DESC
    """)
    perf_rows = c.fetchall()
    perf_csv = output_dir / "submission_performance.csv"
    with open(perf_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["submission_id", "team_name", "team_id", "mu", "sigma",
                          "score", "n_games", "wins", "draws", "losses",
                          "win_rate", "is_baseline", "is_active", "created_at"])
        for row in perf_rows:
            wr = row["wins"] / row["n_games"] if row["n_games"] > 0 else 0
            writer.writerow([
                row["submission_id"], row["team_name"], row["canonical_team_id"],
                f"{row['mu']:.4f}", f"{row['sigma']:.4f}", f"{row['score']:.4f}",
                row["n_games"], row["wins"], row["draws"], row["losses"],
                f"{wr:.4f}", row["is_baseline"], row["is_active"], row["created_at"],
            ])
    print(f"  {len(perf_rows)} submissions with games")
    print(f"  → {perf_csv}")

    # ── 8. Export raw match_results for timeline reconstruction ────────────────
    print("\n=== Exporting Match Results (for timeline) ===")
    c.execute("""
        SELECT match_id, created_at, player_submission_ids_csv,
               ranks_csv, match_type, seed
        FROM match_results
        ORDER BY created_at
    """)
    match_rows = c.fetchall()
    matches_csv = output_dir / "match_results_export.csv"
    with open(matches_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["match_id", "created_at", "player_submission_ids_csv",
                          "ranks_csv", "match_type", "seed"])
        for row in match_rows:
            writer.writerow([row["match_id"], row["created_at"],
                             row["player_submission_ids_csv"],
                             row["ranks_csv"], row["match_type"],
                             row["seed"]])
    print(f"  {len(match_rows)} matches exported")
    print(f"  → {matches_csv}")

    conn.close()

    # ── Summary JSON ──────────────────────────────────────────────────────────
    summary = {
        "active_teams": n_teams,
        "student_submissions": n_student_subs,
        "baseline_submissions": n_baseline_subs,
        "total_matches": n_matches,
        "date_range": [str(daily_rows[0]["day"]) if daily_rows else None,
                       str(daily_rows[-1]["day"]) if daily_rows else None],
    }
    summary_path = output_dir / "db_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary → {summary_path}")

    return summary


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Analyze production competition.db")
    parser.add_argument("--db_path", default="competition.db", help="Path to production DB")
    parser.add_argument("--output_dir", default="analysis/data", help="Output directory")
    args = parser.parse_args()

    analyze_db(args.db_path, args.output_dir)
