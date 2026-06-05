"""Aggregate per-submission benchmark results into dual-dataset leaderboard.json.

Scans results/<submission>-{2016,master2017}.json, builds the leaderboard
with both datasets, and regenerates embedded tables.

Usage:
    uv run python scripts/aggregate_leaderboard.py [--check] [--write]
"""

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "results"
LEADERBOARD_PATH = RESULTS_DIR / "leaderboard.json"

SUBMISSIONS = [
    "2dcnn-baseline",
    "bottom-top-lstm",
    "class-balanced-loss",
    "deep-mlp-baseline",
    "fast-mlp",
    "focal-loss",
    "gradient-boost-baseline",
    "lstm-baseline",
    "multichannel-2dcnn",
    "ordinal-regression",
    "perceptron-baseline",
    "ridge-baseline",
    "transformer-encoder",
    "tree-baseline",
]

SUBMISSION_META: dict[str, dict[str, str]] = {
    "2dcnn-baseline": {
        "model": "2D CNN Baseline",
        "model_class": "3-layer 2D CNN",
        "features": "3-layer grid (3x18x11)",
    },
    "bottom-top-lstm": {
        "model": "Bottom-to-Top LSTM",
        "model_class": "Bidirectional LSTM with bottom-to-top token ordering",
        "features": "Hold name tokens",
    },
    "class-balanced-loss": {
        "model": "Class-Balanced Loss (MLP)",
        "model_class": "3-layer MLP (198->256->128->13) + ClassBalancedLoss",
        "features": "198-dim binary hold vector, per-fold standardized",
    },
    "deep-mlp-baseline": {
        "model": "DeepMLP (ensemble)",
        "model_class": "4-layer DeepMLP + 5-model softmax ensemble",
        "features": "656-dim: section-separated + bigram + meta",
    },
    "fast-mlp": {
        "model": "FastMLP",
        "model_class": "3-layer MLP (198->256->128->13)",
        "features": "198-dim binary hold vector, per-fold standardized",
    },
    "focal-loss": {
        "model": "Focal Loss (MLP)",
        "model_class": "3-layer MLP (198->256->128->13) + FocalLoss(gamma=2.0)",
        "features": "198-dim binary hold vector, per-fold standardized",
    },
    "gradient-boost-baseline": {
        "model": "Gradient Boost",
        "model_class": "GradientBoostingClassifier (sklearn)",
        "features": "164-dim compressed grid vector via GridMapper or 198-dim hold vector",
    },
    "lstm-baseline": {
        "model": "LSTM Baseline",
        "model_class": "2-layer LSTM + attention",
        "features": "Hold name tokens",
    },
    "multichannel-2dcnn": {
        "model": "Multi-Channel 2D CNN",
        "model_class": "Multi-channel 2D CNN with per-layer channels",
        "features": "3-layer grid (3x18x11)",
    },
    "ordinal-regression": {
        "model": "Ordinal Regression (CORAL)",
        "model_class": "CORAL ordinal regression MLP (198->256->128->13)",
        "features": "198-dim binary hold vector, per-fold standardized",
    },
    "perceptron-baseline": {
        "model": "Perceptron Baseline",
        "model_class": "2-layer MLP (198->128->13)",
        "features": "198-dim binary hold vector",
    },
    "ridge-baseline": {
        "model": "Ridge Regression",
        "model_class": "RidgeClassifierCV (sklearn)",
        "features": "164-dim compressed grid vector via GridMapper or 198-dim hold vector",
    },
    "transformer-encoder": {
        "model": "Transformer Encoder",
        "model_class": "Transformer encoder with masked mean pooling",
        "features": "Hold name tokens",
    },
    "tree-baseline": {
        "model": "Random Forest",
        "model_class": "RandomForestClassifier (sklearn)",
        "features": "164-dim compressed grid vector via GridMapper or 198-dim hold vector",
    },
}

DATASET_INFO: dict[str, dict[str, str | int]] = {
    "2016": {
        "num_routes": 25738,
        "cv_folds": 5,
        "description": "2016 hold setup (25,738 unique routes)",
    },
    "master2017": {
        "num_routes": 19363,
        "cv_folds": 5,
        "description": "Masters 2017 hold setup (19,363 unique routes after adding 6B)",
    },
}


def load_results(submission: str, dataset: str) -> dict | None:
    """Load benchmark results for a submission on a given dataset."""
    path = RESULTS_DIR / f"{submission}-{dataset}.json"
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return None
    return data


def build_entry(submission: str, results: dict) -> dict:
    """Build a leaderboard entry from benchmark results."""
    mean_scores = results.get("mean_scores", {})
    std_scores = results.get("std_scores", {})
    meta = SUBMISSION_META.get(submission, {"model": submission, "model_class": "", "features": ""})

    entry: dict[str, str | dict] = {
        "submission": submission,
        "model": meta["model"],
        "model_class": meta["model_class"],
        "features": meta["features"],
    }

    for metric in ("exact_accuracy", "within_one_grade", "within_two_grades", "macro_f1"):
        entry[metric] = {
            "mean": round(mean_scores.get(metric, 0.0), 4),
            "std": round(std_scores.get(metric, 0.0), 4),
        }

    if "training_time" in results:
        entry["training_time"] = results["training_time"]

    return entry


def build_leaderboard() -> dict:
    """Build the full dual-dataset leaderboard."""
    datasets: dict[str, dict] = {}

    for dataset_key in ("2016", "master2017"):
        entries = []
        for sub in SUBMISSIONS:
            results = load_results(sub, dataset_key)
            if results is None:
                continue
            entry = build_entry(sub, results)
            entries.append(entry)

        # Sort by exact_accuracy mean descending
        entries.sort(key=lambda e: e["exact_accuracy"]["mean"], reverse=True)

        info = DATASET_INFO.get(dataset_key, {})
        datasets[dataset_key] = {
            "num_routes": info.get("num_routes", 0),
            "cv_folds": info.get("cv_folds", 5),
            "description": info.get("description", ""),
            "entries": entries,
        }

    return {
        "version": "2.0",
        "datasets": datasets,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate benchmark results into dual-dataset leaderboard"
    )
    parser.add_argument(
        "--check", action="store_true",
        help="Dry-run mode: check what would change without writing",
    )
    parser.add_argument(
        "--write", action="store_true",
        help="Write updated leaderboard.json to disk",
    )
    args = parser.parse_args()

    board = build_leaderboard()

    if not board["datasets"]:
        print("ERROR: no results found. Run benchmarks first.")
        sys.exit(1)

    if args.write:
        LEADERBOARD_PATH.write_text(json.dumps(board, indent=2) + "\n")
        print(f"Wrote {LEADERBOARD_PATH}")
    elif args.check:
        print("Dry-run: leaderboard would be written to disk. Pass --write to apply.")
    else:
        print("Pass --write to write leaderboard.json or --check for dry-run.")

    # Print summary
    for ds_name, ds_data in board["datasets"].items():
        entries = ds_data["entries"]
        print(f"\nDataset: {ds_name} ({len(entries)} submissions)")
        print(f"{'Submission':<25} {'Exact':>8} {'±1':>8} {'±2':>8} {'Macro-F1':>8}")
        print("-" * 57)
        for e in entries[:5]:  # Top 5
            ea = e["exact_accuracy"]["mean"]
            w1 = e["within_one_grade"]["mean"]
            w2 = e["within_two_grades"]["mean"]
            mf = e["macro_f1"]["mean"]
            print(f"{e['submission']:<25} {ea:>8.4f} {w1:>8.4f} {w2:>8.4f} {mf:>8.4f}")
        print(f"  ... and {len(entries) - 5} more" if len(entries) > 5 else "")


if __name__ == "__main__":
    main()
