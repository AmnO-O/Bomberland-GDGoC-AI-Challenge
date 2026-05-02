"""Seed baseline agents as explicit DB rows with fixed ratings.

Usage:
    python -m evaluation.seed_baselines --db_path competition.db
"""

import argparse
import hashlib
import json
import sqlite3
from pathlib import Path
from datetime import datetime, timezone

ROOT_DIR = Path(__file__).resolve().parent.parent

from competition.config import load_env
load_env()
DEFAULT_DB_PATH = str(ROOT_DIR / "competition.db")


BASELINES = [
    {
        "name": "tactical_rule_agent",
        "team_id": "baseline_tactical_rule_agent",
        "submission_id": "baseline_tactical_rule_agent_v1",
        "response_id": "baseline:tactical_rule_agent:v1",
        "drive_file_id": "baseline:tactical_rule_agent:v1",
        "agent_path": ROOT_DIR / "submissions" / "baseline_tactical_rule_agent" / "baseline_tactical_rule_agent_v1" / "agent.py",
        "mu": 31.95,
        "sigma": 0.70,
        "n_games": 486,
    },
    {
        "name": "genius_rule_agent",
        "team_id": "baseline_genius_rule_agent",
        "submission_id": "baseline_genius_rule_agent_v1",
        "response_id": "baseline:genius_rule_agent:v1",
        "drive_file_id": "baseline:genius_rule_agent:v1",
        "agent_path": ROOT_DIR / "submissions" / "baseline_genius_rule_agent" / "baseline_genius_rule_agent_v1" / "agent.py",
        "mu": 30.80,
        "sigma": 0.70,
        "n_games": 459,
    },
    {
        "name": "smarter_rule_agent",
        "team_id": "baseline_smarter_rule_agent",
        "submission_id": "baseline_smarter_rule_agent_v1",
        "response_id": "baseline:smarter_rule_agent:v1",
        "drive_file_id": "baseline:smarter_rule_agent:v1",
        "agent_path": ROOT_DIR / "submissions" / "baseline_smarter_rule_agent" / "baseline_smarter_rule_agent_v1" / "agent.py",
        "mu": 25.17,
        "sigma": 0.70,
        "n_games": 552,
    },
    {
        "name": "box_farmer_agent",
        "team_id": "baseline_box_farmer_agent",
        "submission_id": "baseline_box_farmer_agent_v1",
        "response_id": "baseline:box_farmer_agent:v1",
        "drive_file_id": "baseline:box_farmer_agent:v1",
        "agent_path": ROOT_DIR / "submissions" / "baseline_box_farmer_agent" / "baseline_box_farmer_agent_v1" / "agent.py",
        "mu": 25.17,
        "sigma": 0.70,
        "n_games": 494,
    },
    {
        "name": "simple_rule_agent",
        "team_id": "baseline_simple_rule_agent",
        "submission_id": "baseline_simple_rule_agent_v1",
        "response_id": "baseline:simple_rule_agent:v1",
        "drive_file_id": "baseline:simple_rule_agent:v1",
        "agent_path": ROOT_DIR / "submissions" / "baseline_simple_rule_agent" / "baseline_simple_rule_agent_v1" / "agent.py",
        "mu": 24.90,
        "sigma": 0.69,
        "n_games": 523,
    },
    {
        "name": "random_agent",
        "team_id": "baseline_random_agent",
        "submission_id": "baseline_random_agent_v1",
        "response_id": "baseline:random_agent:v1",
        "drive_file_id": "baseline:random_agent:v1",
        "agent_path": ROOT_DIR / "submissions" / "baseline_random_agent" / "baseline_random_agent_v1" / "agent.py",
        "mu": 16.62,
        "sigma": 0.84,
        "n_games": 526,
    },
]


def _score(mu: float, sigma: float) -> float:
    return mu - 3.0 * sigma


def seed_baselines(db_path: str):
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()

        for baseline in BASELINES:
            agent_path = baseline["agent_path"]
            if not agent_path.exists():
                raise FileNotFoundError(f"Missing baseline agent.py: {agent_path}")

            extracted_path = str(agent_path.parent)
            manifest = {"agent.py": agent_path.stat().st_size}

            # Upsert baseline team in the same transaction to avoid SQLite lock contention.
            token_hash = hashlib.sha256(
                f"{baseline['team_id']}_token".encode("utf-8")
            ).hexdigest()
            cursor.execute(
                """
                INSERT INTO teams (
                    canonical_team_id,
                    team_name,
                    primary_email,
                    submission_token_hash,
                    status,
                    created_at
                ) VALUES (?, ?, ?, ?, 'active', ?)
                ON CONFLICT(canonical_team_id) DO UPDATE SET
                    team_name = excluded.team_name,
                    primary_email = excluded.primary_email,
                    submission_token_hash = excluded.submission_token_hash,
                    status = 'active'
                """,
                (
                    baseline["team_id"],
                    baseline["team_id"],
                    "baseline@local",
                    token_hash,
                    datetime.now(timezone.utc).isoformat(),
                ),
            )

            # Upsert submission row and pin baseline properties.
            cursor.execute(
                """
                INSERT INTO submissions (
                    submission_id,
                    canonical_team_id,
                    response_id,
                    drive_file_id,
                    original_filename,
                    sha256,
                    uploaded_at,
                    created_at,
                    validation_status,
                    validation_reason,
                    extracted_path,
                    extracted_manifest_json,
                    is_baseline,
                    is_active,
                    is_team_best,
                    is_team_recent,
                    is_top_global,
                    mu,
                    sigma,
                    n_games,
                    wins,
                    draws,
                    losses,
                    total_rank,
                    total_steps
                ) VALUES (?, ?, ?, ?, ?, NULL, NULL, datetime('now'), 'valid', NULL, ?, ?, 1, 1, 1, 1, 1, ?, ?, ?, 0, 0, 0, 0, 0)
                ON CONFLICT(submission_id) DO UPDATE SET
                    canonical_team_id = excluded.canonical_team_id,
                    response_id = excluded.response_id,
                    drive_file_id = excluded.drive_file_id,
                    original_filename = excluded.original_filename,
                    validation_status = 'valid',
                    validation_reason = NULL,
                    extracted_path = excluded.extracted_path,
                    extracted_manifest_json = excluded.extracted_manifest_json,
                    is_baseline = 1,
                    is_active = 1,
                    is_team_best = 1,
                    is_team_recent = 1,
                    is_top_global = 1,
                    mu = excluded.mu,
                    sigma = excluded.sigma,
                    n_games = excluded.n_games
                """,
                (
                    baseline["submission_id"],
                    baseline["team_id"],
                    baseline["response_id"],
                    baseline["drive_file_id"],
                    f"{baseline['name']}.py",
                    extracted_path,
                    json.dumps(manifest, sort_keys=True),
                    baseline["mu"],
                    baseline["sigma"],
                    baseline["n_games"],
                ),
            )

        conn.commit()

    print("Seeded baselines:")
    for baseline in BASELINES:
        print(
            f"- {baseline['submission_id']}: mu={baseline['mu']:.2f} "
            f"sigma={baseline['sigma']:.2f} score={_score(baseline['mu'], baseline['sigma']):.2f} n_games={baseline['n_games']}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--db_path", default=DEFAULT_DB_PATH)
    args = parser.parse_args()
    seed_baselines(db_path=args.db_path)
