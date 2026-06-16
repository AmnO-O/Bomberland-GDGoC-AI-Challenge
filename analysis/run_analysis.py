"""Main orchestrator for Bomberland strategy trend analysis.

This script coordinates all analysis phases:
  Track A: DB analysis + code classification (no heavy downloads)
  Track B: JSON metrics extraction from Drive (streaming)
  Visualizations: Charts from all collected data

Usage on VM:
    cd /home/vltisme/Bomberland   # or wherever project root is
    conda activate aic_gdgoc
    python analysis/run_analysis.py track_a          # DB + code analysis
    python analysis/run_analysis.py track_b          # Drive streaming (10% sample)
    python analysis/run_analysis.py track_b_full     # Drive streaming (100%)
    python analysis/run_analysis.py visualize        # Generate all plots
    python analysis/run_analysis.py all              # Run everything

Usage locally (testing with sample data):
    python analysis/run_analysis.py local_test       # Uses local logs/json + submissions
"""

import argparse
import os
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def run_track_a(db_path: str = "competition.db", output_dir: str = "analysis/data"):
    """Track A: DB analysis + source code classification."""
    print("=" * 60)
    print("TRACK A: DB Analysis + Code Classification")
    print("=" * 60)

    # 1. DB analysis
    print("\n── Phase 1: DB Analysis ──")
    if os.path.exists(db_path):
        from analysis.db_analysis import analyze_db
        analyze_db(db_path, output_dir)
    else:
        print(f"  WARNING: {db_path} not found.")
        print(f"  On VM, ensure you're running from the project root with competition.db present.")
        print(f"  To download from Drive backup, run:")
        print(f"    python analysis/download_db_backup.py")

    # 2. Source code classification
    print("\n── Phase 2: Source Code Classification ──")
    submissions_dir = os.path.join(str(PROJECT_ROOT), "submissions")
    if os.path.isdir(submissions_dir):
        from analysis.classify_submissions import classify_all_submissions
        classify_all_submissions(submissions_dir, os.path.join(output_dir, "code_classification.csv"))
    else:
        print(f"  WARNING: {submissions_dir} not found.")

    print("\n✅ Track A complete.")


def run_track_b(sample_rate: float = 0.10, output: str = "analysis/data/metrics.csv",
                start_date: str = "", end_date: str = ""):
    """Track B: Stream JSON metrics from Drive."""
    print("=" * 60)
    print(f"TRACK B: Drive JSON Streaming (sample_rate={sample_rate:.0%})")
    print("=" * 60)

    from analysis.drive_stream_pipeline import run_pipeline
    run_pipeline(
        sample_rate=sample_rate,
        start_date=start_date,
        end_date=end_date,
        output_csv=output,
        in_memory=True,
    )

    print("\n✅ Track B complete.")


def run_visualize(data_dir: str = "analysis/data", plot_dir: str = "analysis/plots"):
    """Generate all visualizations."""
    print("=" * 60)
    print("VISUALIZATIONS")
    print("=" * 60)

    from analysis.visualize import generate_all_plots
    generate_all_plots(data_dir, plot_dir)

    print("\n✅ Visualization complete.")


def run_local_test(output_dir: str = "analysis/data"):
    """Local test mode: use sample logs and local submissions."""
    print("=" * 60)
    print("LOCAL TEST MODE")
    print("=" * 60)

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # 1. Extract metrics from local sample JSONs
    json_dir = os.path.join(str(PROJECT_ROOT), "logs", "json")
    if os.path.isdir(json_dir):
        from analysis.extract_match_metrics import process_directory
        metrics_csv = os.path.join(output_dir, "metrics.csv")
        process_directory(json_dir, metrics_csv, limit=200)  # First 200 for quick test
    else:
        print(f"  WARNING: {json_dir} not found")

    # 2. Classify submissions
    submissions_dir = os.path.join(str(PROJECT_ROOT), "submissions")
    if os.path.isdir(submissions_dir):
        from analysis.classify_submissions import classify_all_submissions
        classify_all_submissions(submissions_dir, os.path.join(output_dir, "code_classification.csv"))
    else:
        print(f"  WARNING: {submissions_dir} not found")

    # 3. DB analysis (use local DB even if it's a dev copy)
    db_path = os.path.join(str(PROJECT_ROOT), "competition.db")
    if os.path.exists(db_path):
        from analysis.db_analysis import analyze_db
        analyze_db(db_path, output_dir)

    # 4. Visualize
    from analysis.visualize import generate_all_plots
    generate_all_plots(output_dir, os.path.join(str(PROJECT_ROOT), "analysis", "plots"))

    print("\n✅ Local test complete.")


def main():
    parser = argparse.ArgumentParser(
        description="Bomberland Strategy Trend Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Commands:
  track_a       DB analysis + code classification (no downloads needed)
  track_b       Drive JSON streaming with 10%% sampling (~20 min)
  track_b_full  Drive JSON streaming at 100%% (~3-4 hours)
  visualize     Generate all plots from existing data
  local_test    Quick test using local sample data
  all           Run track_a + track_b + visualize
        """,
    )
    parser.add_argument("command", choices=["track_a", "track_b", "track_b_full",
                                            "visualize", "local_test", "all"])
    parser.add_argument("--db_path", default="competition.db")
    parser.add_argument("--output_dir", default="analysis/data")
    parser.add_argument("--plot_dir", default="analysis/plots")
    parser.add_argument("--start_date", default="")
    parser.add_argument("--end_date", default="")

    args = parser.parse_args()

    if args.command == "track_a":
        run_track_a(args.db_path, args.output_dir)

    elif args.command == "track_b":
        run_track_b(sample_rate=0.10, output=os.path.join(args.output_dir, "metrics.csv"),
                    start_date=args.start_date, end_date=args.end_date)

    elif args.command == "track_b_full":
        run_track_b(sample_rate=1.0, output=os.path.join(args.output_dir, "metrics.csv"),
                    start_date=args.start_date, end_date=args.end_date)

    elif args.command == "visualize":
        run_visualize(args.output_dir, args.plot_dir)

    elif args.command == "local_test":
        run_local_test(args.output_dir)

    elif args.command == "all":
        run_track_a(args.db_path, args.output_dir)
        run_track_b(sample_rate=0.10, output=os.path.join(args.output_dir, "metrics.csv"),
                    start_date=args.start_date, end_date=args.end_date)
        run_visualize(args.output_dir, args.plot_dir)


if __name__ == "__main__":
    main()
