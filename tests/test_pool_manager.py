import sqlite3
import tempfile
import uuid
from pathlib import Path

from competition.storage import SubmissionStore
from competition.evaluation.pool_manager import PoolManager


def _save_valid_submission(store: SubmissionStore, team_id: str, response_id: str):
    sid = str(uuid.uuid4())
    store.save_submission(
        submission_id=sid,
        canonical_team_id=team_id,
        response_id=response_id,
        drive_file_id=response_id,
        original_filename="agent.zip",
        sha256="hash",
        uploaded_at="2026-01-01T00:00:00Z",
        validation_status="valid",
        validation_reason=None,
        extracted_path=f"submissions/{team_id}/{sid}",
        extracted_manifest_json='{"agent.py": 10}',
    )
    return sid


def test_pool_manager_sets_best_recent_top_and_active_flags():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name

    try:
        store = SubmissionStore(db_path=db_path)
        store.register_team("team_a", "Team A", "a@example.com", "tok_a")
        store.register_team("team_b", "Team B", "b@example.com", "tok_b")

        s1 = _save_valid_submission(store, "team_a", "resp_a_1")
        s2 = _save_valid_submission(store, "team_a", "resp_a_2")
        s3 = _save_valid_submission(store, "team_b", "resp_b_1")

        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("UPDATE submissions SET mu = 35, sigma = 3, n_games = 12 WHERE submission_id = ?", (s1,))
            cursor.execute("UPDATE submissions SET mu = 26, sigma = 8, n_games = 3 WHERE submission_id = ?", (s2,))
            cursor.execute("UPDATE submissions SET mu = 30, sigma = 5, n_games = 11 WHERE submission_id = ?", (s3,))
            conn.commit()

        summary = PoolManager(db_path=db_path).recompute_active_pool(recent_per_team=1, top_k=1)
        assert summary["total_valid"] == 3
        assert summary["total_active"] >= 2

        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT submission_id, is_team_best, is_team_recent, is_top_global, is_active FROM submissions"
            )
            rows = {row[0]: row[1:] for row in cursor.fetchall()}

        assert rows[s1][0] == 1
        assert rows[s2][1] == 1
        assert rows[s3][0] == 1
        assert rows[s1][3] == 1
        assert rows[s2][3] == 1
        assert rows[s3][3] == 1
    finally:
        Path(db_path).unlink()
