import argparse
import logging
import random
import time
import os
import sqlite3
from concurrent.futures import ProcessPoolExecutor, as_completed


from competition.config import load_env
load_env()

from competition.storage import SubmissionStore
from competition.evaluation.match_runner import MatchRunner
from competition.evaluation.ranking import RankingSystem

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

def run_single_calibration_match(
    match_index: int,
    participants: list[str],
    db_path: str,
    enable_gif: bool = False,
    timeout_s: float = 15.0
):
    agent_paths = []
    team_ids = []
    
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        for sid in participants:
            cursor.execute(
                """
                SELECT s.extracted_path, t.team_name 
                FROM submissions s
                JOIN teams t ON s.canonical_team_id = t.canonical_team_id
                WHERE s.submission_id = ?
                """,
                (sid,)
            )
            row = cursor.fetchone()
            if row:
                agent_path = os.path.join(row[0], "agent.py")
                agent_paths.append(agent_path)
                team_ids.append(row[1])
            else:
                return {"status": "error", "reason": f"submission {sid} not found"}

    runner = MatchRunner(
        log_dir="logs",
        enable_gif=enable_gif,
    )
    
    try:
        ranks, survival_steps, gif_path, json_path, gif_drive_url, json_drive_url = runner.run_match(
            agent_paths=agent_paths,
            team_ids=participants, # Keep submission IDs for ranking system mapping
            seed=random.randint(0, 999999),
            max_steps=500,
            inference_timeout_s=0.1,
            startup_timeout_s=timeout_s
        )
        return {
            "status": "success",
            "participants": participants,
            "ranks": ranks,
            "steps": survival_steps,
            "json_path": json_path,
            "gif_path": gif_path,
        }
    except Exception as e:
        return {
            "status": "error",
            "participants": participants,
            "reason": str(e)
        }

def calibrate_baselines(db_path: str, matches: int, parallel_workers: int):
    # Fetch baselines
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT submission_id FROM submissions WHERE is_baseline = 1 AND validation_status = 'valid'")
        baseline_ids = [row[0] for row in cursor.fetchall()]

    if len(baseline_ids) < 4:
        logger.error(f"Need at least 4 valid baselines to calibrate. Found {len(baseline_ids)}.")
        return

    logger.info(f"Found {len(baseline_ids)} baselines. Starting {matches} calibration matches using {parallel_workers} workers.")
    
    match_queue = []
    for _ in range(matches):
        match_queue.append(random.sample(baseline_ids, 4))
        
    ranking = RankingSystem(db_path=db_path)
    
    success_count = 0
    error_count = 0
    
    start_time = time.time()
    
    with ProcessPoolExecutor(max_workers=parallel_workers) as executor:
        futures = {
            executor.submit(run_single_calibration_match, i, p, db_path): p 
            for i, p in enumerate(match_queue)
        }
        
        for future in as_completed(futures):
            res = future.result()
            if res["status"] == "success":
                success_count += 1
                # The critical part: update ratings AND allow baselines to be updated
                ranking.update_ratings(
                    submission_ids=res["participants"],
                    ranks=res["ranks"],
                    steps=res["steps"],
                    json_path=res["json_path"],
                    gif_path=res["gif_path"],
                    match_type="baseline_calibration",
                    allow_baseline_updates=True
                )
            else:
                error_count += 1
                logger.error(f"Calibration match failed: {res.get('reason')}")
            
            completed = success_count + error_count
            if completed % 10 == 0 or completed == matches:
                logger.info(f"Progress: {completed}/{matches} matches complete")
                
    duration = time.time() - start_time
    logger.info(f"Calibration complete in {duration:.1f}s. Success: {success_count}, Errors: {error_count}")
    
    # Print updated baseline stats
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT t.team_name, s.mu, s.sigma, s.n_games 
            FROM submissions s 
            JOIN teams t ON s.canonical_team_id = t.canonical_team_id 
            WHERE s.is_baseline = 1 
            ORDER BY s.mu DESC
            """
        )
        print("\n--- Final Baseline Ratings ---")
        for row in cursor.fetchall():
            print(f"{row[0]:<35} Mu: {row[1]:.2f} | Sigma: {row[2]:.2f} | Games: {row[3]}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--matches", type=int, default=600, help="Number of calibration matches to run")
    parser.add_argument("--workers", type=int, default=4, help="Number of parallel workers")
    parser.add_argument("--db_path", type=str, default="competition.db", help="Path to DB")
    args = parser.parse_args()
    
    calibrate_baselines(args.db_path, args.matches, args.workers)
