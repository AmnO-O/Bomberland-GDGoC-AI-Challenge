"""Generate visualizations from analysis data.

Usage:
    python analysis/visualize.py --data_dir analysis/data --plot_dir analysis/plots
"""

import csv
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")  # Non-interactive backend for VM
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from matplotlib.ticker import MaxNLocator
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("WARNING: matplotlib not available. Skipping plot generation.", file=sys.stderr)


# ── Color palette ─────────────────────────────────────────────────────────────
COLORS = {
    "rule_based": "#4CAF50",
    "ml_pytorch": "#FF5722",
    "ml_tensorflow": "#2196F3",
    "ml_onnx": "#9C27B0",
    "hybrid": "#FF9800",
    "copy_baseline": "#9E9E9E",
    "unknown": "#607D8B",
}

BASELINE_COLOR = "#BDBDBD"
STUDENT_COLORS = [
    "#E91E63", "#3F51B5", "#009688", "#FF5722", "#795548",
    "#2196F3", "#4CAF50", "#FFC107", "#9C27B0", "#00BCD4",
    "#FF9800", "#8BC34A", "#673AB7", "#F44336", "#03A9F4",
    "#CDDC39", "#607D8B", "#E91E63", "#3F51B5", "#009688",
]


def _load_csv(path: str) -> list[dict]:
    """Load CSV file into list of dicts."""
    with open(path, "r") as f:
        return list(csv.DictReader(f))


def _save_fig(fig, path: str, dpi: int = 150):
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  → {path}")


def _get_sid_to_team(data_dir: str) -> dict:
    perf_csv = Path(data_dir) / "submission_performance.csv"
    if not perf_csv.exists():
        return {}
    return {r["submission_id"]: r["team_name"] for r in _load_csv(str(perf_csv))}


# ── Plot 1: Code type distribution ───────────────────────────────────────────

def plot_code_type_distribution(data_dir: str, plot_dir: str):
    """Pie chart of submission code types."""
    if not HAS_MPL:
        return
    csv_path = Path(data_dir) / "code_classification.csv"
    if not csv_path.exists():
        print("  SKIP: code_classification.csv not found")
        return

    rows = _load_csv(str(csv_path))
    # Only count student submissions (exclude baselines)
    rows = [r for r in rows if not r["team_id"].startswith("baseline_")]

    type_counts = Counter(r["code_type"] for r in rows)

    fig, ax = plt.subplots(figsize=(8, 6))
    labels = list(type_counts.keys())
    sizes = list(type_counts.values())
    colors = [COLORS.get(l, "#999999") for l in labels]

    # Format labels
    display_labels = []
    for l, s in zip(labels, sizes):
        pct = 100 * s / sum(sizes)
        pretty = l.replace("_", " ").title()
        display_labels.append(f"{pretty}\n({s}, {pct:.0f}%)")

    wedges, texts = ax.pie(sizes, labels=display_labels, colors=colors,
                           startangle=90, textprops={"fontsize": 10})
    ax.set_title("Submission Code Types (Student Submissions)", fontsize=14, fontweight="bold")

    _save_fig(fig, str(Path(plot_dir) / "code_type_distribution.png"))


# ── Plot 2: Matches per day ──────────────────────────────────────────────────

def plot_matches_per_day(data_dir: str, plot_dir: str):
    """Bar chart of matches per day."""
    if not HAS_MPL:
        return
    csv_path = Path(data_dir) / "matches_per_day.csv"
    if not csv_path.exists():
        print("  SKIP: matches_per_day.csv not found")
        return

    rows = _load_csv(str(csv_path))
    dates = [r["date"] for r in rows]
    counts = [int(r["matches"]) for r in rows]

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.bar(range(len(dates)), counts, color="#42A5F5", edgecolor="#1E88E5")
    ax.set_xticks(range(len(dates)))
    ax.set_xticklabels(dates, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Matches")
    ax.set_title("Matches Per Day", fontsize=14, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)

    _save_fig(fig, str(Path(plot_dir) / "matches_per_day.png"))


# ── Plot 3: Leaderboard bar chart ────────────────────────────────────────────

def plot_leaderboard(data_dir: str, plot_dir: str):
    """Horizontal bar chart of leaderboard scores."""
    if not HAS_MPL:
        return
    csv_path = Path(data_dir) / "leaderboard_snapshot.csv"
    if not csv_path.exists():
        print("  SKIP: leaderboard_snapshot.csv not found")
        return

    rows = _load_csv(str(csv_path))
    rows.reverse()  # Lowest score at bottom

    names = [r["team_name"] for r in rows]
    scores = [float(r["score"]) for r in rows]
    is_baseline = [int(r["is_baseline"]) for r in rows]
    colors = [BASELINE_COLOR if b else "#42A5F5" for b in is_baseline]

    fig, ax = plt.subplots(figsize=(10, max(6, len(rows) * 0.4)))
    bars = ax.barh(range(len(names)), scores, color=colors, edgecolor="#333", linewidth=0.5)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel("Score (μ - 3σ)")
    ax.set_title("Leaderboard Snapshot", fontsize=14, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)

    # Add score labels
    for bar, score in zip(bars, scores):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                f"{score:.1f}", va="center", fontsize=8)

    _save_fig(fig, str(Path(plot_dir) / "leaderboard_snapshot.png"))


# ── Plot 4: Behavioral metrics heatmap ────────────────────────────────────────

def plot_behavioral_heatmap(data_dir: str, plot_dir: str):
    """Heatmap of normalized behavioral metrics per submission."""
    if not HAS_MPL:
        return
    csv_path = Path(data_dir) / "metrics.csv"
    if not csv_path.exists():
        print("  SKIP: metrics.csv not found")
        return

    rows = _load_csv(str(csv_path))
    if not rows:
        return

    # Aggregate per submission_id
    metric_keys = [
        "bombs_placed", "bombs_near_box", "bombs_near_enemy",
        "idle_ratio", "movement_entropy", "danger_entries", "danger_escapes",
        "avg_dist_to_nearest_enemy", "map_coverage_pct",
    ]

    sub_metrics = defaultdict(lambda: {k: [] for k in metric_keys})
    for r in rows:
        sid = r["submission_id"]
        for k in metric_keys:
            try:
                sub_metrics[sid][k].append(float(r[k]))
            except (ValueError, KeyError):
                pass

    # Average per submission
    subs = sorted(sub_metrics.keys())
    # Filter to those with enough data
    subs = [s for s in subs if len(sub_metrics[s][metric_keys[0]]) >= 5]
    if not subs:
        print("  SKIP: not enough data for heatmap")
        return

    # Limit to top 20 by match count
    subs = sorted(subs, key=lambda s: len(sub_metrics[s][metric_keys[0]]), reverse=True)[:20]

    matrix = np.zeros((len(subs), len(metric_keys)))
    for i, sid in enumerate(subs):
        for j, k in enumerate(metric_keys):
            vals = sub_metrics[sid][k]
            matrix[i, j] = np.mean(vals) if vals else 0

    # Normalize columns to [0, 1]
    col_min = matrix.min(axis=0)
    col_max = matrix.max(axis=0)
    col_range = col_max - col_min
    col_range[col_range == 0] = 1
    norm_matrix = (matrix - col_min) / col_range

    fig, ax = plt.subplots(figsize=(12, max(6, len(subs) * 0.4)))
    im = ax.imshow(norm_matrix, aspect="auto", cmap="YlOrRd")

    # Map to team names
    sid_map = _get_sid_to_team(data_dir)
    display_names = [sid_map.get(s, s[:8]) for s in subs]
    pretty_keys = [k.replace("_", " ").title() for k in metric_keys]

    ax.set_xticks(range(len(metric_keys)))
    ax.set_xticklabels(pretty_keys, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(len(subs)))
    ax.set_yticklabels(display_names, fontsize=8)
    ax.set_title("Behavioral Metrics Heatmap (Normalized)", fontsize=14, fontweight="bold")

    fig.colorbar(im, ax=ax, fraction=0.02, pad=0.04)

    _save_fig(fig, str(Path(plot_dir) / "behavioral_heatmap.png"))


# ── Plot 5: Strategy cluster scatter ─────────────────────────────────────────

def plot_strategy_clusters(data_dir: str, plot_dir: str):
    """PCA scatter plot of submission behavioral fingerprints."""
    if not HAS_MPL:
        return
    csv_path = Path(data_dir) / "metrics.csv"
    if not csv_path.exists():
        print("  SKIP: metrics.csv not found")
        return

    rows = _load_csv(str(csv_path))
    if not rows:
        return

    metric_keys = [
        "bombs_placed", "bombs_near_box", "bombs_near_enemy",
        "idle_ratio", "movement_entropy", "danger_entries", "danger_escapes",
        "avg_dist_to_nearest_enemy", "map_coverage_pct",
    ]

    # Aggregate per submission
    sub_metrics = defaultdict(lambda: {k: [] for k in metric_keys})
    for r in rows:
        sid = r["submission_id"]
        for k in metric_keys:
            try:
                sub_metrics[sid][k].append(float(r[k]))
            except (ValueError, KeyError):
                pass

    # Find the top submission ID per team
    perf_csv = Path(data_dir) / "submission_performance.csv"
    best_sid_per_team = set()
    if perf_csv.exists():
        perf_rows = _load_csv(str(perf_csv))
        team_best = {}
        for r in perf_rows:
            team = r.get("team_name")
            sid = r["submission_id"]
            try:
                score = float(r["score"])
                if team not in team_best or score > team_best[team]["score"]:
                    team_best[team] = {"sid": sid, "score": score}
            except (ValueError, KeyError):
                pass
        best_sid_per_team = {v["sid"] for v in team_best.values()}

    # Filter to only the best agent per team (or baselines)
    subs = [s for s in sub_metrics if len(sub_metrics[s][metric_keys[0]]) >= 10]
    if best_sid_per_team:
        subs = [s for s in subs if s in best_sid_per_team or "baseline" in s]

    if len(subs) < 3:
        print("  SKIP: not enough submissions for PCA")
        return

    matrix = np.zeros((len(subs), len(metric_keys)))
    for i, sid in enumerate(subs):
        for j, k in enumerate(metric_keys):
            vals = sub_metrics[sid][k]
            matrix[i, j] = np.mean(vals) if vals else 0

    # Standardize
    mean = matrix.mean(axis=0)
    std = matrix.std(axis=0)
    std[std == 0] = 1
    Z = (matrix - mean) / std

    # PCA (2D) via SVD
    U, S, Vt = np.linalg.svd(Z, full_matrices=False)
    pca = Z @ Vt[:2].T  # Project onto first 2 components

    # Color by baseline vs student
    is_baseline = ["baseline" in s for s in subs]
    sid_map = _get_sid_to_team(data_dir)

    fig, ax = plt.subplots(figsize=(10, 8))
    for i, sid in enumerate(subs):
        color = BASELINE_COLOR if is_baseline[i] else STUDENT_COLORS[i % len(STUDENT_COLORS)]
        marker = "s" if is_baseline[i] else "o"
        ax.scatter(pca[i, 0], pca[i, 1], c=color, marker=marker, s=80, edgecolors="black",
                   linewidth=0.5, zorder=3)
        label = sid_map.get(sid, sid[:8])
        ax.annotate(label, (pca[i, 0], pca[i, 1]),
                    textcoords="offset points", xytext=(5, 5), fontsize=7, alpha=0.8)

    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("Strategy Cluster Scatter (PCA)", fontsize=14, fontweight="bold")
    ax.grid(alpha=0.3)

    _save_fig(fig, str(Path(plot_dir) / "strategy_clusters_pca.png"))


# ── Plot 6: Code complexity vs rating ────────────────────────────────────────

def plot_complexity_vs_rating(data_dir: str, plot_dir: str):
    """Scatter plot of code complexity (LOC) vs TrueSkill score."""
    if not HAS_MPL:
        return
    code_csv = Path(data_dir) / "code_classification.csv"
    perf_csv = Path(data_dir) / "submission_performance.csv"
    if not code_csv.exists() or not perf_csv.exists():
        print("  SKIP: need code_classification.csv + submission_performance.csv")
        return

    code_rows = {r["submission_id"]: r for r in _load_csv(str(code_csv))}
    perf_rows = _load_csv(str(perf_csv))

    xs, ys, colors, labels = [], [], [], []
    for pr in perf_rows:
        sid = pr["submission_id"]
        if sid in code_rows and not int(pr.get("is_baseline", 0)):
            loc = int(code_rows[sid]["total_py_loc"])
            score = float(pr["score"])
            ctype = code_rows[sid]["code_type"]
            xs.append(loc)
            ys.append(score)
            colors.append(COLORS.get(ctype, "#999"))
            labels.append(pr.get("team_name", sid[:8]))

    if not xs:
        print("  SKIP: no matched data")
        return

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.scatter(xs, ys, c=colors, s=60, edgecolors="black", linewidth=0.5, alpha=0.8)
    for x, y, label in zip(xs, ys, labels):
        ax.annotate(label, (x, y), textcoords="offset points", xytext=(4, 4), fontsize=6, alpha=0.7)

    ax.set_xlabel("Total Python LOC")
    ax.set_ylabel("Score (μ - 3σ)")
    ax.set_title("Code Complexity vs. Rating", fontsize=14, fontweight="bold")
    ax.grid(alpha=0.3)

    _save_fig(fig, str(Path(plot_dir) / "complexity_vs_rating.png"))


# ── Plot 7: Submission timeline ──────────────────────────────────────────────

def plot_submission_timeline(data_dir: str, plot_dir: str):
    """Timeline showing when each team submitted."""
    if not HAS_MPL:
        return
    csv_path = Path(data_dir) / "submission_timeline.csv"
    if not csv_path.exists():
        print("  SKIP: submission_timeline.csv not found")
        return

    rows = _load_csv(str(csv_path))
    if not rows:
        return

    # Group by team
    team_subs = defaultdict(list)
    for r in rows:
        if r["created_at"]:
            team_subs[r["team_name"]].append(r["created_at"][:10])

    teams = sorted(team_subs.keys(), key=lambda t: min(team_subs[t]))

    fig, ax = plt.subplots(figsize=(14, max(4, len(teams) * 0.35)))

    for i, team in enumerate(teams):
        dates = team_subs[team]
        for d in dates:
            ax.plot(d, i, "o", color=STUDENT_COLORS[i % len(STUDENT_COLORS)],
                    markersize=6, zorder=3)

    ax.set_yticks(range(len(teams)))
    ax.set_yticklabels(teams, fontsize=8)
    ax.set_title("Submission Timeline by Team", fontsize=14, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)

    # Rotate x labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=8)

    _save_fig(fig, str(Path(plot_dir) / "submission_timeline.png"))


# ── Plot 8: Daily Submissions Line Chart ──────────────────────────────────────

def plot_daily_submissions(data_dir: str, plot_dir: str):
    """Line chart of number of submissions per day."""
    if not HAS_MPL:
        return
    csv_path = Path(data_dir) / "submission_timeline.csv"
    if not csv_path.exists():
        return

    rows = _load_csv(str(csv_path))
    if not rows:
        return

    dates = [r["created_at"][:10] for r in rows if r["created_at"]]
    date_counts = Counter(dates)
    
    sorted_dates = sorted(date_counts.keys())
    counts = [date_counts[d] for d in sorted_dates]

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(sorted_dates, counts, marker="o", linestyle="-", color="#E91E63", linewidth=2)
    ax.fill_between(sorted_dates, counts, alpha=0.1, color="#E91E63")
    
    ax.set_xticks(range(len(sorted_dates)))
    ax.set_xticklabels(sorted_dates, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Number of Submissions")
    ax.set_title("Daily Submissions (Line Chart)", fontsize=14, fontweight="bold")
    ax.grid(alpha=0.3)

    for i, count in enumerate(counts):
        ax.annotate(str(count), (sorted_dates[i], counts[i]), textcoords="offset points", xytext=(0, 5), ha="center", fontsize=8)

    _save_fig(fig, str(Path(plot_dir) / "daily_submissions.png"))


# ── Main ──────────────────────────────────────────────────────────────────────

def generate_all_plots(data_dir: str = "analysis/data", plot_dir: str = "analysis/plots"):
    """Generate all available plots."""
    Path(plot_dir).mkdir(parents=True, exist_ok=True)

    print("Generating plots...")
    plot_code_type_distribution(data_dir, plot_dir)
    plot_matches_per_day(data_dir, plot_dir)
    plot_leaderboard(data_dir, plot_dir)
    plot_behavioral_heatmap(data_dir, plot_dir)
    plot_strategy_clusters(data_dir, plot_dir)
    plot_complexity_vs_rating(data_dir, plot_dir)
    plot_submission_timeline(data_dir, plot_dir)
    plot_daily_submissions(data_dir, plot_dir)
    print("\nAll plots generated.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate analysis visualizations")
    parser.add_argument("--data_dir", default="analysis/data")
    parser.add_argument("--plot_dir", default="analysis/plots")
    args = parser.parse_args()

    generate_all_plots(args.data_dir, args.plot_dir)
