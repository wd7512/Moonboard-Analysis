# Moonboard AutoResearch Loop — Implementation Plan

**Date:** 2026-06-24
**Branch:** `research/moonboard-autoresearch-20260624`
**Status:** Post-grill — 21 findings applied, ready for implementation

## Overview

Build an autoresearch-style autonomous research loop inside the Moonboard-Analysis repo. An opencode agent iterates overnight on model/training code to maximize macro_F1 on the Moonboard grade classification benchmark. A frozen controller scores, gates, and commits. The agent proposes; the arbiter disposes.

## Architecture: Proposal ← → Judgment Separation

The core design principle: **the agent NEVER judges its own work.**

```
Agent (research-loop/)          Controller (orchestrator/)       Harness (harness/)
─────────────────────           ──────────────────────────      ──────────────────
Creates/modify submissions/     Reads feedback.jsonl            Loads data (2016/2017)
Calls run_experiment.sh         Spawns run_experiment.pins      Splits folds
                                Reads harness output            Evaluates model
                                Applies Pareto gate             Scores with seed=42
                                Commits if champion             Writes to runs.jsonl
                                Reverts if not                 Returns result
                                Runs stuck detection
```

**The agent sees:** `logs/feedback.jsonl` only (binary status + rank + delta bucket + trajectory).
**The controller sees:** everything. **The harness sees:** frozen, never modified.

## Design Decisions (from self-grill)

| Decision | Choice |
|----------|--------|
| Location | `research-loop/` subdirectory + `orchestrator/` + `harness/` at repo root |
| Branch | `research/moonboard-autoresearch-20260624` |
| Agent | opencode (big-pickle, local) |
| Working dir | repo root (agent sees full project) |
| Metric | macro_F1 (2016 5-fold CV) |
| Keep/discard | Pareto gate: F1 improves AND exact_acc >= 35% (HARD floor, no relative 2pp) |
| Time budget | 10 min hard cap per experiment |
| Smoke gate | 3 min, 80/20 split, 2500 samples, 30 epochs max |
| Reproducibility | seed=42 pinned at fit time, reproducibility check (2x run, reject if drift > 0.5pp) |
| Dashboard | FastAPI + chart.js (localhost:8000, MORNING REVIEW ONLY — agent never reads it) |
| Git | commit-per-run by controller on branch, squash-merge top results to main at end |
| Safety | holdout year (2017), fixed seed=42, no rm, edit restricted to submissions/ + research-loop/ |
| Overfit check | every 5th experiment: 80/20 fresh-sample split |
| Data isolation | Sandboxed cwd — global CSV not visible to eval subprocess; model receives rootless arrays (no hold_id) |
| Agent feedback | Quantized: rank + delta_bucket + trajectory (NOT raw F1, NOT binary) |
| Process isolation | evaluate.py spawns subprocess per submission, never import in-process |
| Package ban | Agent must NOT run `uv add` or install new packages |
| Dead uv ban | Agent cannot run `uv add` — all deps from existing venv |

## Directory Structure

```
harness/                         # IMMUTABLE — evaluation only (outside agent edit scope)
├── evaluate.py                  # CLI: smoke test + full CV for a submission
│                               # Pins RNG at fit time: np.random.seed(42), random.seed(42), torch.manual_seed(42)
│                               # Passes only 2016 frame to model, 2017 for final score only
│                               # Passes rootless numpy arrays (no hold_id, no row keys)
│                               # Spawns submission as subprocess (never import in-process)
│                               # Writes scores to logs/runs.jsonl
│                               # Reads/writes no agent-writable paths
│
orchestrator/                    # IMMUTABLE — decision logic (outside agent edit scope)
├── control_loop.py              # Reads feedback.jsonl, applies Pareto gate, commits/reverts
│                               # Runs stuck detection (3 fails → revert, 5 → stop)
│                               # Re-evaluates champion (2x run reproducibility check)
│                               # Manages overnight time budget
│                               # NEVER modified by agent
│
research-loop/                   # AGENT-WORKSPACE (agent can edit)
├── AGENT.md                     # Agent instructions (the brain)
├── submissions/                 # Agent creates/edits these
│   └── baseline/               # Starting point (copy of best known approach)
├── dashboard/
│   ├── main.py                 # FastAPI app
│   ├── templates/
│   │   └── index.html          # Dashboard (Chart.js)
│   └── static/                 # CSS/JS
└── checkpoints/                 # Best model state per experiment
|
logs/                           # OUTSIDE agent edit scope (deny-read for agent)
├── runs.jsonl                   # Full experiment log (macro_F1, exact_acc, per_class_f1, etc.)
│                               # Written by harness. Read by controller + dashboard. Agent DENY READ.
└── feedback.jsonl               # Agent-readable: run_id, status, rank, delta_bucket, trajectory
                               # Written by controller. Read by agent.
```

(opencode.json lives at repo root, not in research-loop/)

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
      "python *": "allow",
      "subprocess.run *": "allow",
      "git add *": "allow",
      "git commit *": "allow",
      "git status *": "allow",
      "git diff *": "allow",
      "git checkout *": "allow",
      "cat *": "allow",
      "tail *": "allow",
      "grep *": "allow",
      "echo >> logs/feedback.jsonl": "allow",
      "rm *": "deny",
      "uv add *": "deny",
      "pip install *": "deny"
    },
    "read": {
      "logs/runs.jsonl": "deny",
      "harness/**": "deny",
      "orchestrator/**": "deny",
      "*": "allow"
    }
  }
}
```

**Key:** Agent is DENY READ on `logs/runs.jsonl`, `harness/**`, `orchestrator/**`. Agent can ONLY read `logs/feedback.jsonl` for its experiment results.

## Data Visibility Contract

The harness/evaluate.py enforces strict data boundaries:

1. **Features.py contract:** `features.py` exports `transform(X, y) -> X_transformed`. Receives ONLY 2016 training data. NEVER sees 2017. Pure function — no file I/O, no global state.
2. **Model.py contract:** Model receives `(X: np.ndarray, y: np.ndarray)`. No `hold_id`, no row index, no stable key. Cannot memorize by lookup — would need feature-vector equality.
3. **Sandboxed cwd:** Eval subprocess runs in a cloistered directory. ONLY fold CSVs the harness explicitly passes exist. `pd.read_csv('data/moonboard.csv')` → FileNotFoundError.
4. **Smoke gate grep:** Before training, smoke gate greps ALL `submissions/` files for: `read_csv`, `open(`, `np.load`, `pd.read_`, `os.system`, `subprocess`, `np.random.seed(None)`, `random.seed(os.urandom`, `random_state=None`. Any match → smoke fail.
5. **RNG pinning:** Before calling `model.fit(X, y)`, harness does:
   ```python
   import numpy as np, random
   np.random.seed(42)
   random.seed(42)
   try:
       import torch; torch.manual_seed(42)
   except: pass
   ```

## Feedback Channel Design

Agent sees `logs/feedback.jsonl` with:
```json
{
  "run_id": "0000000023",
  "status": "new_champion",
  "rank": "3 of 12 since last champion",
  "delta_bucket": "new_champion",
  "trajectory": "improving (12→8→5→3 [champion])"
}
```

Status values: `new_champion`, `also_ran`, `smoke_fail`, `crash`, `timeout`, `nondeterministic`
Delta buckets: `far`, `close`, `new_champion`
Trajectory: `improving (<rank sequence>)`, `plateau (<rank sequence>)`, `regressing (<rank sequence>)`

This is leak-safe: rank reveals position but not precision. Delta bucket is 1-bit. Agent can hill-climb but cannot overfit a threshold.

## runs.jsonl Schema (agent DENY READ)

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
  "is_champion": true,
  "reproducibility_delta_f1": 0.0
}
```

## Loop Flow (per iteration)

1. Agent reads `logs/feedback.jsonl` (tail-50 + `logs/runs_summary.json`)
2. Agent creates/modifies `submissions/<name>/main.py` + optionally `submissions/<name>/features.py`
3. Agent calls: `python orchestrator/run_experiment.py <name>`
4. Controller (frozen):
   a. Spawns `sandbox-exec -f eval.sb harness/evaluate.py submissions/<name> --smoke` — Seatbelt restricts FS/network at kernel level
   b. Smoke gate: transitive grep for forbidden patterns → defense-in-depth, fail if found
   c. Smoke gate fails → log to runs.jsonl + feedback.jsonl, return status=smoke_fail
   d. Smoke gate passes → spawn `harness/evaluate.py submissions/<name> --full`
   e. Full CV runs with RNG pinned at fit time
   f. Harness writes scores to `logs/runs.jsonl` (agent cannot read)
   g. Reproducibility check: run again with same seed, compare F1
   h. If |F1_run1 - F1_run2| > 0.5pp → log status=nondeterministic, reject
   i. Apply Pareto gate: macro_F1 > champion AND exact_acc >= 35%
   j. Pass → commit, update champion, write feedback.jsonl (status=new_champion)
   k. Fail → revert submission, write feedback.jsonl (status=also_ran)
5. Every 5th run: fresh-sample overfitting check (80/20 split within 2016)
   - If CV improves but fresh-sample degrades → flag OVERFIT, discard
6. Stuck detection (controller):
   - 3 consecutive non-champion results → revert to champion as base, agent makes small change
   - 5 consecutive non-champion results → stop loop, log "reasoning spiral"
7. Loop until time budget exhausted or human stops

## runs_summary.json (agent-readable)

```json
{
  "total_runs": 47,
  "total_champions": 3,
  "total_failures": 12,
  "runs_since_last_champion": 7,
  "last_champion_id": "0000000041",
  "status": "plateau",
  "experiments_per_hour": 6.2
}
```

NO numeric metrics. Only counts and status. Agent knows "stuck" (high runs_since_last_champion) but not "how close."

## Smoke Gate Detail

The smoke gate is a CRASH FILTER, not a CV proxy. It catches:
- Syntax errors in submission code
- Shape mismatches / dimension errors
- NaN losses / infinite loops
- OOM / memory errors
- Forbidden code patterns (file I/O, unseeded RNG)

It does NOT predict final CV score. If the submission trains without crashing for 3 min on 2500 samples, it passes. The full 5-fold CV is the real metric.

## Implementation Steps ( Ordered by Dependency)

1. Create `harness/evaluate.py` — evaluation entry point (frozen, immutable)
   - Implements subprocess isolation, RNG pinning, data visibility contract
   - Implements smoke gate + transitive grep + reproducibility check
   - Writes to `logs/runs.jsonl`
2. Create `orchestrator/control_loop.py` — frozen controller
   - Implements Pareto gate (hard floor)
   - Implements stuck detection
   - Implements commit/revert logic
   - Writes to `logs/feedback.jsonl`
3. Create `logs/` directory and seed `runs.jsonl`, `feedback.jsonl`, `runs_summary.json`
4. Create `research-loop/` directory structure
5. Write `research-loop/AGENT.md` — agent instructions
   - Specifies: agent sees only `logs/feedback.jsonl` + `logs/runs_summary.json`
   - Complexity budget: hidden dim ≤ 512, num_layers ≤ 4, num_epochs ≤ 50 (soft constraint)
   - No `uv add`, no package installs
   - Stuck behavior: when `runs_since_last_champion > 5`, revert to champion + small change
6. Write `opencode.json` at repo root with full read + edit permissions
7. Create `submissions/baseline/` — copy of best known approach
8. Integration test: run baseline through the full loop manually
9. Launch first overnight run

## Containerization Decision: Skip Docker, Use Seatbelt

**Decision:** Do NOT use Docker for v1. Use macOS `sandbox-exec` (Seatbelt) instead.

**Rationale:**
- Docker Desktop on macOS requires a running daemon — single point of failure on a laptop that may sleep overnight
- Docker Desktop consumes ~2GB RAM — starves eval subprocess on a memory-constrained MacBook
- Container startup overhead (1-3s × 30 evals) is acceptable but unnecessary when Seatbelt is free
- Seatbelt provides kernel-level FS + network isolation at fork/exec time — cannot be bypassed by Python string tricks
- Seatbelt has zero daemon, zero image, zero RAM overhead — built into macOS

**Seatbelt approach:**
- `sandbox-exec -f eval.sb python submissions/<name>/main.py`
- `.sb` profile restricts: only Python stdlib + venv + eval tmpdir visible
- Global CSV unreachable at kernel level (no path in allowed subpaths)
- Network outbound denied by default in profile
- One-time setup (write `.sb` file), not per-eval

**Keep grep as defense-in-depth:** catches bugs in the sandbox profile itself, and provides fast-fail before spawning the subprocess.

## Risks and Mitigations

| Risk | Mitigation |
|------|------------|
| Agent plateau on one approach | Stuck detection → revert champion + small change |
| Agent produces inscrutable optimized code | Quantized feedback prevents overfitting to specific thresholds |
| Agent memorizes test folds | Seatbelt sandbox + rootless arrays + transitive grep (3 layers) |
| Agent corrupts own log | logs/runs.jsonl outside agent read scope |
| Agent hacks the judge | harness/ and orchestrator/ outside agent edit scope |
| Agent sees scores it shouldn't | Read-scope permissions in opencode.json |
| Champion is lucky draw (stochastic) | Reproducibility check (2x run, reject if drift > 0.5pp) |
| No champion produced all night | Quantized rank+trajectory gives ~10-20% acceptance rate |
| Binary feedback makes loop sterile | Middle-ground: rank + delta bucket + trajectory (not raw, not binary) |
| Docker daemon sleeps on laptop | Seatbelt has zero daemon — always available |
| Sandbox profile too restrictive for Python | One-time tuning of .sb allowlist, then frozen |

## Cost Estimate

- ~24-30 experiments/night (accounting for 2x reproducibility check on champions)
- ~$50-200 per overnight run depending on model/provider
- Expected yield: 2-5 champions per night

## Grill Log

Full 21-question grill transcript saved to session history. Key architectural changes from grill:
- Separation of proposal (Agent) from judgment (Controller)
- Read-scope boundary (agent denied read on scores)
- Data leakage prevention (sandbox + rootless arrays)
- Training determinism (RNG pinning at fit time)
- Quantized feedback channel (rank + delta bucket + trajectory)
