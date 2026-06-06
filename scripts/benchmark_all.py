#!/usr/bin/env python3
"""Benchmark all missing submissions on 2016 and 2017.

Runs sequentially to avoid CPU contention.
Skips submissions that already have results.
"""
import subprocess
import json
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SUBMISSIONS_DIR = REPO_ROOT / "submissions"
RESULTS_DIR = REPO_ROOT / "results"
DATA_2016 = REPO_ROOT / "Raw" / "moonboard_problems_setup_2016.json"
DATA_2017 = REPO_ROOT / "Raw" / "moonboard_problems_setup_master2017.json"

# All submissions (excluding non-submission dirs)
EXCLUDE = {"check_experiment.py", "README.md"}
SUBMISSIONS = sorted([
    d.name for d in SUBMISSIONS_DIR.iterdir()
    if d.is_dir() and d.name not in EXCLUDE
])

def run_benchmark(submission, dataset, data_path):
    out_json = RESULTS_DIR / f"{submission}-{dataset}.json"
    out_md = RESULTS_DIR / f"{submission}-{dataset}.md"
    
    if out_json.exists():
        return submission, dataset, "SKIP", "already exists"
    
    cmd = [
        "uv", "run", "python", "-m", "moonboard_analysis.scripts.benchmark",
        "--submission-dir", str(SUBMISSIONS_DIR / submission),
        "--data-path", str(data_path),
        "--output-json", str(out_json),
        "--output-markdown", str(out_md),
    ]
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=900,
            cwd=str(REPO_ROOT)
        )
        if result.returncode == 0:
            d = json.loads(out_json.read_text())
            mf1 = d["mean_scores"]["macro_f1"]
            return submission, dataset, "PASS", f"macro_f1={mf1:.4f}"
        else:
            err = (result.stderr + result.stdout)[-300:]
            return submission, dataset, "FAIL", err[:200]
    except subprocess.TimeoutExpired:
        return submission, dataset, "TIMEOUT", "900s exceeded"
    except Exception as e:
        return submission, dataset, "ERROR", str(e)[:200]

def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    
    # Build todo list
    todo_2016 = []
    todo_2017 = []
    for sub in SUBMISSIONS:
        if not (RESULTS_DIR / f"{sub}-2016.json").exists():
            todo_2016.append(sub)
        if not (RESULTS_DIR / f"{sub}-2017.json").exists():
            todo_2017.append(sub)
    
    print(f"2016: {len(todo_2016)} to run, 2017: {len(todo_2017)} to run")
    print(f"Already done: {len(SUBMISSIONS) - len(todo_2016)} (2016), {len(SUBMISSIONS) - len(todo_2017)} (2017)")
    
    results = []
    
    # Run 2016 first
    if todo_2016:
        print(f"\n=== 2016 benchmarks ({len(todo_2016)} submissions) ===")
        for sub in todo_2016:
            print(f"  Starting {sub}...", flush=True)
            start = time.time()
            s, d, status, msg = run_benchmark(sub, "2016", DATA_2016)
            elapsed = time.time() - start
            print(f"  [{status}] {sub} ({elapsed:.0f}s): {msg}")
            results.append((s, d, status, msg))
    
    # Then 2017
    if todo_2017:
        print(f"\n=== 2017 benchmarks ({len(todo_2017)} submissions) ===")
        for sub in todo_2017:
            print(f"  Starting {sub}...", flush=True)
            start = time.time()
            s, d, status, msg = run_benchmark(sub, "2017", DATA_2017)
            elapsed = time.time() - start
            print(f"  [{status}] {sub} ({elapsed:.0f}s): {msg}")
            results.append((s, d, status, msg))
    
    # Summary
    passed = sum(1 for _, _, s, _ in results if s == "PASS")
    failed = sum(1 for _, _, s, _ in results if s in ("FAIL", "ERROR", "TIMEOUT"))
    skipped = sum(1 for _, _, s, _ in results if s == "SKIP")
    print(f"\nDone: {passed} passed, {failed} failed, {skipped} skipped")
    
    if failed > 0:
        print("\nFailed:")
        for s, d, status, msg in results:
            if status in ("FAIL", "ERROR", "TIMEOUT"):
                print(f"  {s} ({d}): {msg[:200]}")

if __name__ == "__main__":
    main()
