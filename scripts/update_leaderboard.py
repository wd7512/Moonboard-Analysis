#!/usr/bin/env python3
"""Update class-balanced-loss results in leaderboard.json after re-benchmark."""
import json
from pathlib import Path

LEADERBOARD = Path(__file__).resolve().parent.parent / "results" / "leaderboard.json"

NEW_RESULTS = {
    "exact_accuracy": {"mean": 0.3970, "std": 0.0090},
    "within_one_grade": {"mean": 0.6426, "std": 0.0076},
    "within_two_grades": {"mean": 0.8314, "std": 0.0096},
    "macro_f1": {"mean": 0.1798, "std": 0.0121},
}

with open(LEADERBOARD) as f:
    data = json.load(f)

for entry in data["entries"]:
    if entry["submission"] == "class-balanced-loss":
        old = {k: dict(entry[k]) for k in NEW_RESULTS}
        entry.update(NEW_RESULTS)
        print("Updated class-balanced-loss:")
        for k in NEW_RESULTS:
            print(f"  {k}: {old[k]} -> {NEW_RESULTS[k]}")
        break

with open(LEADERBOARD, "w") as f:
    json.dump(data, f, indent=2)
    f.write("\n")
print("leaderboard.json written")
