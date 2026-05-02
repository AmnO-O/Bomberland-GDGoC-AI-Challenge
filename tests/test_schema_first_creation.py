import sqlite3
import tempfile
from pathlib import Path

from competition.storage import SubmissionStore


def test_schema_first_creation_has_full_submission_model():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name

    try:
        SubmissionStore(db_path=db_path)

        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("PRAGMA table_info(submissions)")
            cols = {row[1] for row in cursor.fetchall()}

            required = {
                "submission_id",
                "canonical_team_id",
                "response_id",
                "drive_file_id",
                "validation_status",
                "is_baseline",
                "is_active",
                "is_team_best",
                "is_team_recent",
                "is_top_global",
                "mu",
                "sigma",
                "n_games",
                "wins",
                "draws",
                "losses",
                "total_rank",
                "total_steps",
            }
            assert required.issubset(cols)

            cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='match_results'"
            )
            assert cursor.fetchone() is not None
    finally:
        Path(db_path).unlink()
