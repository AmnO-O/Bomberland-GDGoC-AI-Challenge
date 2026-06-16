"""Classify submission source code into strategy types.

Scans each submission's agent.py and supporting files to determine:
- Code type: rule_based, ml_pytorch, ml_tensorflow, ml_onnx, hybrid, unknown
- Complexity metrics: LOC, function count, class count
- Strategy signals: has BFS, has danger model, has escape logic, etc.

Usage:
    python analysis/classify_submissions.py --submissions_dir submissions --output analysis/data/code_classification.csv
"""

import ast
import csv
import os
import re
import sys
from pathlib import Path
from typing import Optional


COLUMNS = [
    "team_id",
    "submission_id",
    "submission_path",
    "code_type",          # rule_based, ml_pytorch, ml_tensorflow, ml_onnx, hybrid, copy_baseline, unknown
    "baseline_origin",    # Which baseline it's copied from, if any
    "has_model_file",     # .pth, .onnx, .pt, .pkl, .h5, etc.
    "model_file_type",    # pth, onnx, etc.
    "model_file_size_mb",
    "total_py_loc",
    "agent_py_loc",
    "num_functions",
    "num_classes",
    "has_bfs",
    "has_danger_model",
    "has_escape_logic",
    "has_enemy_targeting",
    "has_item_collection",
    "has_pathfinding",
    "imports_torch",
    "imports_tensorflow",
    "imports_onnx",
    "imports_numpy",
    "num_py_files",
    "num_total_files",
]

# Known baseline class names & signatures
BASELINE_SIGNATURES = {
    "RandomAgent": "random_agent",
    "SimpleRuleAgent": "simple_rule_agent",
    "SmarterRuleAgent": "smarter_rule_agent",
    "GeniusRuleAgent": "genius_rule_agent",
    "BoxFarmerAgent": "box_farmer_agent",
    "TacticalRuleAgent": "tactical_rule_agent",
}

MODEL_EXTENSIONS = {".pth", ".pt", ".onnx", ".pkl", ".h5", ".keras", ".tflite", ".bin", ".pb"}


def _count_loc(filepath: str) -> int:
    """Count non-blank, non-comment lines of code."""
    try:
        with open(filepath, "r", errors="ignore") as f:
            lines = f.readlines()
    except OSError:
        return 0
    count = 0
    for line in lines:
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            count += 1
    return count


def _parse_ast_info(filepath: str) -> dict:
    """Extract AST-level info from a Python file."""
    info = {
        "num_functions": 0,
        "num_classes": 0,
        "class_names": [],
        "imports": set(),
    }
    try:
        with open(filepath, "r", errors="ignore") as f:
            source = f.read()
        tree = ast.parse(source)
    except (SyntaxError, OSError, ValueError):
        return info

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) or isinstance(node, ast.AsyncFunctionDef):
            info["num_functions"] += 1
        elif isinstance(node, ast.ClassDef):
            info["num_classes"] += 1
            info["class_names"].append(node.name)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                info["imports"].add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                info["imports"].add(node.module.split(".")[0])

    return info


def _detect_patterns(source_code: str) -> dict:
    """Search for strategy-related patterns in source code."""
    src_lower = source_code.lower()

    return {
        "has_bfs": bool(re.search(r"deque|bfs|breadth.?first|queue", src_lower)),
        "has_danger_model": bool(re.search(r"danger|blast|explosion|timer", src_lower)),
        "has_escape_logic": bool(re.search(r"escape|safe|flee|evade|avoid", src_lower)),
        "has_enemy_targeting": bool(re.search(r"enemy|opponent|target|hunt|attack|chase", src_lower)),
        "has_item_collection": bool(re.search(r"item|power.?up|collect|pickup|capacity|radius", src_lower)),
        "has_pathfinding": bool(re.search(r"a.?star|dijkstra|pathfind|shortest.?path|heapq", src_lower)),
    }


def _detect_baseline_copy(class_names: list[str], source_code: str) -> Optional[str]:
    """Check if a submission is a copy of a baseline agent."""
    for cls_name, baseline_name in BASELINE_SIGNATURES.items():
        if cls_name in class_names:
            # Extra check: if the source is very similar to baselines (same team_id attribute)
            if f'team_id = "{cls_name}"' in source_code:
                return baseline_name
    return None


def classify_submission(submission_dir: str) -> dict:
    """Classify a single submission directory."""
    sub_path = Path(submission_dir)
    team_id = sub_path.parent.name  # submissions/{team_id}/{sub_id}/
    submission_id = sub_path.name

    agent_py = sub_path / "agent.py"
    if not agent_py.exists():
        return None

    # Collect all files
    all_files = list(sub_path.iterdir())
    py_files = [f for f in all_files if f.suffix == ".py"]
    model_files = [f for f in all_files if f.suffix.lower() in MODEL_EXTENSIONS]

    # Read agent.py
    try:
        with open(agent_py, "r", errors="ignore") as f:
            agent_source = f.read()
    except OSError:
        return None

    # AST analysis on agent.py
    ast_info = _parse_ast_info(str(agent_py))

    # Pattern detection across ALL .py files
    all_source = agent_source
    total_py_loc = _count_loc(str(agent_py))
    total_functions = ast_info["num_functions"]
    total_classes = ast_info["num_classes"]

    for pyf in py_files:
        if pyf.name != "agent.py":
            try:
                with open(pyf, "r", errors="ignore") as f:
                    extra_source = f.read()
                all_source += "\n" + extra_source
                total_py_loc += _count_loc(str(pyf))
                extra_ast = _parse_ast_info(str(pyf))
                total_functions += extra_ast["num_functions"]
                total_classes += extra_ast["num_classes"]
                ast_info["imports"] |= extra_ast["imports"]
            except OSError:
                pass

    patterns = _detect_patterns(all_source)

    # Import classification
    imports_torch = "torch" in ast_info["imports"]
    imports_tf = "tensorflow" in ast_info["imports"] or "tf" in ast_info["imports"]
    imports_onnx = "onnxruntime" in ast_info["imports"] or "onnx" in ast_info["imports"]
    imports_numpy = "numpy" in ast_info["imports"] or "np" in ast_info["imports"]

    # Model file detection
    has_model_file = len(model_files) > 0
    model_file_type = ""
    model_file_size_mb = 0.0
    if model_files:
        biggest = max(model_files, key=lambda f: f.stat().st_size)
        model_file_type = biggest.suffix.lstrip(".")
        model_file_size_mb = round(biggest.stat().st_size / (1024 * 1024), 2)

    # Baseline copy detection
    baseline_origin = _detect_baseline_copy(ast_info["class_names"], agent_source)

    # Code type classification
    if baseline_origin and not has_model_file and not imports_torch and not imports_tf:
        code_type = "copy_baseline"
    elif has_model_file or imports_torch or imports_tf or imports_onnx:
        if patterns["has_bfs"] or patterns["has_danger_model"]:
            code_type = "hybrid"
        elif imports_torch:
            code_type = "ml_pytorch"
        elif imports_tf:
            code_type = "ml_tensorflow"
        elif imports_onnx:
            code_type = "ml_onnx"
        else:
            code_type = "ml_pytorch"  # has model file
    elif patterns["has_bfs"] or patterns["has_danger_model"] or patterns["has_escape_logic"]:
        code_type = "rule_based"
    else:
        code_type = "unknown"

    return {
        "team_id": team_id,
        "submission_id": submission_id,
        "submission_path": str(sub_path),
        "code_type": code_type,
        "baseline_origin": baseline_origin or "",
        "has_model_file": int(has_model_file),
        "model_file_type": model_file_type,
        "model_file_size_mb": model_file_size_mb,
        "total_py_loc": total_py_loc,
        "agent_py_loc": _count_loc(str(agent_py)),
        "num_functions": total_functions,
        "num_classes": total_classes,
        "has_bfs": int(patterns["has_bfs"]),
        "has_danger_model": int(patterns["has_danger_model"]),
        "has_escape_logic": int(patterns["has_escape_logic"]),
        "has_enemy_targeting": int(patterns["has_enemy_targeting"]),
        "has_item_collection": int(patterns["has_item_collection"]),
        "has_pathfinding": int(patterns["has_pathfinding"]),
        "imports_torch": int(imports_torch),
        "imports_tensorflow": int(imports_tf),
        "imports_onnx": int(imports_onnx),
        "imports_numpy": int(imports_numpy),
        "num_py_files": len(py_files),
        "num_total_files": len(all_files),
    }


def classify_all_submissions(submissions_dir: str, output_csv: str):
    """Classify all submissions and write CSV."""
    submissions_root = Path(submissions_dir)
    results = []

    # Structure: submissions/{team_id}/{submission_id}/agent.py
    team_dirs = sorted([d for d in submissions_root.iterdir() if d.is_dir()])

    for team_dir in team_dirs:
        try:
            sub_dirs = sorted([d for d in team_dir.iterdir() if d.is_dir()])
        except PermissionError:
            print(f"  SKIP (permission denied): {team_dir.name}", file=sys.stderr)
            continue

        for sub_dir in sub_dirs:
            try:
                result = classify_submission(str(sub_dir))
                if result:
                    results.append(result)
            except PermissionError:
                print(f"  SKIP (permission denied): {sub_dir}", file=sys.stderr)
            except Exception as e:
                print(f"  ERROR classifying {sub_dir}: {e}", file=sys.stderr)

    # Write CSV
    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        for row in results:
            writer.writerow(row)

    # Print summary
    print(f"\nClassified {len(results)} submissions → {output_csv}")
    type_counts = {}
    for r in results:
        t = r["code_type"]
        type_counts[t] = type_counts.get(t, 0) + 1
    print("Code type distribution:")
    for t, c in sorted(type_counts.items(), key=lambda x: -x[1]):
        print(f"  {t}: {c}")

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Classify submission source code")
    parser.add_argument("--submissions_dir", default="submissions", help="Root submissions directory")
    parser.add_argument("--output", default="analysis/data/code_classification.csv", help="Output CSV path")
    args = parser.parse_args()

    classify_all_submissions(args.submissions_dir, args.output)
