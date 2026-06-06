#!/usr/bin/env python3
"""Rebuild leaderboard 2016 entries from re-run benchmark results."""
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "results"
LEADERBOARD_PATH = RESULTS_DIR / "leaderboard.json"

# Submissions that were re-run
SUBMISSIONS = [
    "perceptron-baseline",
    "class-balanced-loss",
    "focal-loss",
    "fast-mlp",
    "deep-mlp-baseline",
    "lstm-baseline",
    "bottom-top-lstm",
    "transformer-encoder",
    "2dcnn-baseline",
    "multichannel-2dcnn",
    "tree-baseline",
    "ridge-baseline",
    "gradient-boost-baseline",
    "ordinal-regression",
]

def main():
    with open(LEADERBOARD_PATH, encoding="utf-8") as f:
        leaderboard = json.load(f)
    
    entries_2016 = leaderboard["datasets"]["2016"]["entries"]
    
    # Build lookup by submission name
    entry_by_name = {e["submission"]: e for e in entries_2016}
    
    updated = 0
    for sub in SUBMISSIONS:
        result_path = RESULTS_DIR / f"{sub}-2016.json"
        if not result_path.exists():
            print(f"  SKIP {sub}: no result file")
            continue
        
        with open(result_path) as f:
            result = json.load(f)
        
        ms = result["mean_scores"]
        ss = result["std_scores"]
        
        if sub not in entry_by_name:
            print(f"  SKIP {sub}: not in leaderboard")
            continue
        
        entry = entry_by_name[sub]
        entry["exact_accuracy"]["mean"] = ms["exact_accuracy"]
        entry["exact_accuracy"]["std"] = ss["exact_accuracy"]
        entry["within_one_grade"]["mean"] = ms["within_one_grade"]
        entry["within_one_grade"]["std"] = ss["within_one_grade"]
        entry["within_two_grades"]["mean"] = ms["within_two_grades"]
        entry["within_two_grades"]["std"] = ss["within_two_grades"]
        entry["macro_f1"]["mean"] = ms["macro_f1"]
        entry["macro_f1"]["std"] = ss["macro_f1"]
        updated += 1
        print(f"  UPDATED {sub}: macro_f1={ms['macro_f1']:.4f}")
    
    # Sort entries by exact_accuracy descending
    leaderboard["datasets"]["2016"]["entries"] = sorted(
        entries_2016,
        key=lambda e: e["exact_accuracy"]["mean"],
        reverse=True,
    )
    
    with open(LEADERBOARD_PATH, "w") as f:
        json.dump(leaderboard, f, indent=2)
    
    print(f"\nUpdated {updated} entries. Leaderboard written.")

if __name__ == "__main__":
    main()
