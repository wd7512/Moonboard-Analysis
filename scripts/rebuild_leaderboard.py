#!/usr/bin/env python3
"""Rebuild leaderboard from all available result JSONs."""
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "results"
LEADERBOARD_PATH = RESULTS_DIR / "leaderboard.json"

SUBS = sorted([
    "perceptron-baseline", "class-balanced-loss", "focal-loss", "fast-mlp",
    "deep-mlp-baseline", "lstm-baseline", "bottom-top-lstm", "transformer-encoder",
    "2dcnn-baseline", "multichannel-2dcnn", "tree-baseline",
    "ridge-baseline", "gradient-boost-baseline", "ordinal-regression",
    "coral-engineered", "coral-engineered-10class", "coral-engineered-10ens",
    "coral-engineered-f1loss", "coral-engineered-f1loss-10class",
    "coral-engineered-gamma1", "coral-engineered-gamma3",
    "coral-engineered-stacking", "coral-engineered-temp",
])

# Metadata for new submissions
META = {
    "coral-engineered": {"model": "Coral Engineered", "model_class": "Focal ordinal ensemble (gamma=2.0)", "features": "198-dim binary hold vector"},
    "coral-engineered-10class": {"model": "Coral 10-Class", "model_class": "Focal ordinal ensemble, 10 classes", "features": "198-dim binary hold vector"},
    "coral-engineered-10ens": {"model": "Coral 10-Class Ensemble", "model_class": "10-class focal + F1-loss ensemble", "features": "198-dim binary hold vector"},
    "coral-engineered-f1loss": {"model": "Coral F1-Loss", "model_class": "Direct soft-F1 loss optimization", "features": "198-dim binary hold vector"},
    "coral-engineered-f1loss-10class": {"model": "Coral F1-Loss 10-Class", "model_class": "F1-loss, 10 classes", "features": "198-dim binary hold vector"},
    "coral-engineered-gamma1": {"model": "Coral Gamma=1.0", "model_class": "Focal ordinal (gamma=1.0)", "features": "198-dim binary hold vector"},
    "coral-engineered-gamma3": {"model": "Coral Gamma=3.0", "model_class": "Focal ordinal (gamma=3.0)", "features": "198-dim binary hold vector"},
    "coral-engineered-stacking": {"model": "Coral Stacking", "model_class": "3-model stacking meta-learner", "features": "198-dim binary hold vector"},
    "coral-engineered-temp": {"model": "Coral Temperature", "model_class": "Focal ordinal with temperature scaling", "features": "198-dim binary hold vector"},
}

def build_entry(sub, dataset):
    f = RESULTS_DIR / f"{sub}-{dataset}.json"
    if not f.exists():
        return None
    d = json.loads(f.read_text())
    ms = d["mean_scores"]
    ss = d["std_scores"]
    
    # Get metadata from existing leaderboard or use defaults
    entry = {
        "submission": sub,
        "model": META.get(sub, {}).get("model", sub),
        "model_class": META.get(sub, {}).get("model_class", ""),
        "features": META.get(sub, {}).get("features", ""),
        "exact_accuracy": {"mean": ms["exact_accuracy"], "std": ss["exact_accuracy"]},
        "within_one_grade": {"mean": ms["within_one_grade"], "std": ss["within_one_grade"]},
        "within_two_grades": {"mean": ms["within_two_grades"], "std": ss["within_two_grades"]},
        "macro_f1": {"mean": ms["macro_f1"], "std": ss["macro_f1"]},
    }
    return entry

with open(LEADERBOARD_PATH) as f:
    lb = json.load(f)

# Update 2016 entries
entries_2016 = []
for sub in SUBS:
    e = build_entry(sub, "2016")
    if e:
        entries_2016.append(e)

entries_2016.sort(key=lambda e: e["exact_accuracy"]["mean"], reverse=True)
lb["datasets"]["2016"]["entries"] = entries_2016
lb["datasets"]["2016"]["num_routes"] = 25738

# Update 2017 entries
entries_2017 = []
for sub in SUBS:
    e = build_entry(sub, "2017")
    if e:
        entries_2017.append(e)

entries_2017.sort(key=lambda e: e["exact_accuracy"]["mean"], reverse=True)
lb["datasets"]["master2017"]["entries"] = entries_2017
lb["datasets"]["master2017"]["num_routes"] = 19642

with open(LEADERBOARD_PATH, "w") as f:
    json.dump(lb, f, indent=2)

print(f"2016: {len(entries_2016)} entries")
print(f"2017: {len(entries_2017)} entries")
print("Leaderboard written.")
