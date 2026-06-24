# Moonboard AutoResearch Loop — Implementation Plan

**Date:** 2026-06-24
**Branch:** `research/moonboard-autoresearch-20260624`
**Status:** Approved — ready for implementation

## Overview

Build an autoresearch-style autonomous research loop inside `research-loop/` within the Moonboard-Analysis repo. An opencode agent iterates overnight on model/training code to maximize macro_F1 on the Moonboard grade classification benchmark.

## Design Decisions (from self-grill)

| Decision | Choice |
|----------|--------|
| Location | `research-loop/` subdirectory in Moonboard-Analysis repo |
| Branch | `research/moonboard-autoresearch-20260624` |
| Agent | opencode (big-pickle, local) |
| Working dir | repo root (agent sees full project) |
| Metric | macro_F1 (2016 5-fold CV) |
| Keep/discard | Pareto gate: F1 improves AND exact_acc > champion - 2pp AND exact_acc > 30% |
| Time budget | 10 min hard cap per experiment |
| Smoke gate | 3 min, 1 fold, 2500 samples — must pass before full CV |
| Dashboard | FastAPI + chart.js (served on localhost:8000) |
| Git | commit-per-run on branch, squash-merge top results to main at end |
| Safety | holdout year (2017), fixed seed=42, no rm, edit restricted to submissions/ + research-loop/ |
| Overfit check | every 5th experiment: 80/20 fresh-sample split |

## Directory Structure

```
research-loop/
├── AGENT.md               # Agent instructions (the brain)
├── runs.jsonl             # Append-only experiment log
├── harness/               # IMMUTABLE — evaluation only
│   └── evaluate.py        # CLI: smoke test + full CV for a submission
├── submissions/           # Agent creates/edits these
│   └── baseline/          # Starting point (copy of best known approach)
├── dashboard/
│   ├── main.py            # FastAPI app
│   ├── templates/
│   │   └── index.html     # Dashboard (Chart.js)
│   └── static/            # CSS/JS
└── checkpoints/           # Best model state per experiment
```

(opencode.json lives at repo root, not in research-loop/)

## Loop Flow (per iteration)

1. Agent reads `runs.jsonl` for current best metrics
2. Agent creates/modifies `submissions/<name>/main.py`
3. Runs smoke test: `uv run python -m harness.evaluate submissions/<name> --smoke`
   - fail → log status=smoke_fail, discard, goto 1
4. Runs full CV: `uv run python -m harness.evaluate submissions/<name> --full`
   - fail/timeout → log status=crash/timeout, discard, goto 1
5. Evaluate Pareto gate against champion:
   - pass → new champion, log, commit (msg: `ex:<name> macro_F1=XX.XX%`)
   - fail → log as also-ran, revert submission
6. Every 5th run: fresh-sample overfitting check
   - if CV improves but fresh-sample degrades → flag OVERFIT, discard
7. Loop until time budget exhausted or human stops

## opencode.json (repo root)

```json
{
  "$schema": "https://opencode.ai/config.json",
  "model": "opencode/big-pickle",
  "permissions": {
    "edit": {
      "*": "deny",
      "research-loop/**/*": "allow",
      "submissions/**/*": "allow"
    },
    "bash": {
      "uv run *": "allow",
      "git add *": "allow",
      "git commit *": "allow",
      "git status *": "allow",
      "git diff *": "allow",
      "git checkout *": "allow",
      "rm *": "deny"
    },
    "read": {
      "*": "allow"
    }
  }
}
```

## runs.jsonl Schema

```json
{
  "run_id": "0000000001",
  "timestamp": "2026-06-24T22:15:00Z",
  "model_name": "mlp-dropout0.3-batchnorm",
  "status": "success",
  "duration_s": 342,
  "macro_f1": 21.03,
  "exact_accuracy": 38.5,
  "within_one_grade": 64.2,
  "within_two_grades": 83.1,
  "per_class_f1": [...],
  "hyperparams": {"lr": 0.001, "dropout": 0.3, "batch_size": 128},
  "commit_sha": "abc123...",
  "overfit_flag": false,
  "is_champion": true
}
```

## Dashboard Views

| View | Content |
|------|---------|
| Progress | macro_F1 over run number, champion highlighted, 25% target line |
| Leaderboard | All runs sorted by macro_F1 |
| Run Detail | Per-class F1 bar chart, confusion matrix, hyperparams |
| Health | Success/crash/timeout ratio, experiments/hour, last run |

## Implementation Steps

1. Create `research-loop/` directory structure
2. Write `harness/evaluate.py` (wraps existing BenchmarkHarness)
3. Write `AGENT.md` (agent instructions)
4. Write `opencode.json` at repo root
5. Build `dashboard/` (FastAPI + chart.js)
6. Create baseline submission (copy of best known approach)
7. Integration test: run baseline through the loop manually
8. Write launch script (`run_loop.sh`)
9. Launch first overnight run

## Risks

- Agent may plateau on one approach (like fractalsearch's Claude Opus)
- Agent may produce inscrutable optimized code
- Local model (big-pickle) may lack reasoning for complex architecture changes
- Cost: ~$50-200 per overnight run depending on model/provider
