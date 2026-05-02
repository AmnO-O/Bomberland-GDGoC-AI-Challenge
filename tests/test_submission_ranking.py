import sqlite3
import tempfile
import uuid
from pathlib import Path

from competition.storage import SubmissionStore
from competition.evaluation.ranking import RankingSystem


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


def test_submission_centric_rating_updates_stats_and_match_results():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name

    try:
        store = SubmissionStore(db_path=db_path)
        store.register_team("team_a", "Team A", "a@example.com", "tok_a")
        store.register_team("team_b", "Team B", "b@example.com", "tok_b")

        s1 = _save_valid_submission(store, "team_a", "resp_a_1")
        s2 = _save_valid_submission(store, "team_b", "resp_b_1")

        ranking = RankingSystem(db_path=db_path)
        ranking.update_ratings(
            submission_ids=[s1, s2],
            ranks=[0, 1],
            steps=[100, 80],
            seed=123,
            json_path="logs/json/test.json",
            match_type="submission_batch",
        )

        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT n_games, wins, draws, losses, total_rank, total_steps, mu, sigma FROM submissions WHERE submission_id = ?",
                (s1,),
            )
            r1 = cursor.fetchone()
            cursor.execute(
                "SELECT n_games, wins, draws, losses, total_rank, total_steps, mu, sigma FROM submissions WHERE submission_id = ?",
                (s2,),
            )
            r2 = cursor.fetchone()
            cursor.execute("SELECT COUNT(*) FROM match_results")
            match_count = cursor.fetchone()[0]

        assert r1[0] == 1 and r2[0] == 1
        assert r1[1] == 1 and r2[3] == 1
        assert r1[5] == 100 and r2[5] == 80
        assert r1[6] != 25.0 or r1[7] != 8.333
        assert match_count == 1

        leaderboard = ranking.get_leaderboard(include_baseline=True)
        assert len(leaderboard) >= 2
        assert leaderboard[0]["score"] >= leaderboard[1]["score"]
    finally:
        Path(db_path).unlink()


def test_baseline_submission_keeps_fixed_rating_and_stats():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name

    try:
        store = SubmissionStore(db_path=db_path)
        store.register_team("team_a", "Team A", "a@example.com", "tok_a")
        store.register_team("team_b", "Team B", "b@example.com", "tok_b")

        baseline_sid = _save_valid_submission(store, "team_a", "resp_base_1")
        challenger_sid = _save_valid_submission(store, "team_b", "resp_b_1")

        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                UPDATE submissions
                SET is_baseline = 1, mu = 31.95, sigma = 0.70, n_games = 486
                WHERE submission_id = ?
                """,
                (baseline_sid,),
            )
            conn.commit()

        ranking = RankingSystem(db_path=db_path)
        ranking.update_ratings(
            submission_ids=[baseline_sid, challenger_sid],
            ranks=[1, 0],
            steps=[80, 120],
            seed=321,
            json_path="logs/json/test_baseline.json",
            match_type="submission_batch",
        )

        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT mu, sigma, n_games, wins, draws, losses FROM submissions WHERE submission_id = ?",
                (baseline_sid,),
            )
            baseline_row = cursor.fetchone()
            cursor.execute(
                "SELECT n_games FROM submissions WHERE submission_id = ?",
                (challenger_sid,),
            )
            challenger_games = cursor.fetchone()[0]

        assert baseline_row[0] == 31.95
        assert baseline_row[1] == 0.70
        assert baseline_row[2] == 486
        assert baseline_row[3] == 0
        assert baseline_row[4] == 0
        assert baseline_row[5] == 0
        assert challenger_games == 1
    finally:
        Path(db_path).unlink()
