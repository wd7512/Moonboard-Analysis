# Submissions

This directory holds self-contained model submissions for the Moonboard Analysis benchmark leaderboard.

## Code-Only Requirement

Submissions must be **code-only** — no saved model weights are allowed. The
benchmark retrains a fresh model from scratch on each cross-validation fold,
so pre-trained weights are neither used nor accepted.

## Format

Each submission lives in its own folder under `submissions/<model-name>/` and must follow this structure:

```
submissions/<model-name>/
├── main.py            # Required entry point (see contracts below)
├── requirements.txt   # Optional extra dependencies (beyond project deps)
└── config.json        # Optional hyperparameters or metadata
```

## Two Contracts

Every `main.py` must satisfy **both** contracts below.

### A. `train_and_evaluate()` — used by the CV benchmark harness

```python
def train_and_evaluate(
    sequences: list[list[str]],
    grades: list[int],
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    seed: int = 42,
    **kwargs,
) -> dict[str, float]:
    """Train a fresh model on the training fold and evaluate on the test fold.

    Args:
        sequences: Full preprocessed route sequences. Each sequence includes
            hold tokens followed by the grade string at index [-2] and
            'GRADE_END' at index [-1].
        grades: Integer-encoded grade labels (parallel to sequences).
        train_idx: Indices into sequences/grades for the training fold.
        test_idx: Indices into sequences/grades for the test fold.
        seed: Random seed for reproducibility.

    Returns:
        Dict with at minimum the keys:
            "exact_accuracy": float
            "within_one_grade": float
            "within_two_grades": float
    """
```

The harness calls this function once per CV fold with different `train_idx`/`test_idx`. It must **not** load any pre-trained weights — it must train a fresh model each time.

### B. `main()` — standalone training script

The submission must also be runnable as a standalone script:

```bash
uv run python submissions/<model-name>/main.py [ARGS]
```

#### Required CLI Arguments

| Argument | Description |
|----------|-------------|
| `--data-path` | Path to raw Moonboard JSON data |
| `--output-dir` | Directory to save model weights and results |
| `--seed` | Random seed for reproducibility |

#### Expected Behavior

1. **Load** data from `--data-path`.
2. **Preprocess** using the project's preprocessing utilities.
3. **Train** the model on an 80/20 stratified split.
4. **Print** evaluation metrics (exact, within-1, within-2 accuracy).
5. **Save** trained model weights to `--output-dir` (for local inspection; weights are not used by the benchmark).

## Reference Submissions

- **`lstm-baseline/`** — 3-layer LSTM grade classifier. Good starting point for
  PyTorch-based sequence models.
- **`perceptron-baseline/`** — 3-layer MLP on binary hold vectors. Good starting
  point for feed-forward architectures.
- **`tree-baseline/`** — Random Forest classifier on 164-dim feature vectors.
  Demonstrates non-PyTorch submissions.
- **`ridge-baseline/`** — Ridge regression on 164-dim binary hold vectors.
  Simple linear baseline; predicts grade index via rounded regression output.

## Before You Submit — Gate Check

Every new submission MUST pass the experiment gate checker:

```bash
uv run python submissions/check_experiment.py --submission-dir submissions/<your-model-name>
```

The checker runs these gates (all must pass for exit code 0):

| Gate | Description | Type |
|------|-------------|------|
| Interface contract | `main.py` exists with `train_and_evaluate(sequences, grades, train_idx, test_idx, ...)` | PASS/FAIL |
| No pre-trained weights | No `.pth`, `.joblib`, `.h5`, `.onnx` files in submission dir | PASS/FAIL |
| Uses `set_seeds` | Calls `set_seeds(seed)` for reproducibility | PASS/FAIL |
| Required metrics | Returns `exact_accuracy`, `within_one_grade`, `within_two_grades` | PASS/FAIL |
| Feature redundancy | Warns if feature representation duplicates existing submission | WARN |
| Training time | Warns if estimated time is slow at 250K samples | PASS/WARN |
| Code quality | Runs `uv run ruff check` on submission | PASS/WARN |

Exit 0 = no failures (warnings ok). Exit 1 = gate failure.

**Also required:** Read [`EXPERIMENTS.md`](../EXPERIMENTS.md) and verify your experiment is novel BEFORE coding. If your planned model + feature combo is already in the decision matrix as "Exhausted" or "Banned", provide a documented novel variation or choose a different direction.
