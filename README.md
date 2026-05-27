# Moonboard Analysis

Machine learning analysis of Moonboard climbing route data — route compression via autoencoders and grade classification via LSTMs, MLPs, Random Forests, and 2D CNNs.

## Project Structure

```
src/moonboard_analysis/     # Core Python package
  config.py                 # Hyperparameter dataclasses
  data/                     # Data loading, preprocessing, PyTorch datasets
  models/                   # Autoencoder, LSTM, PCA wrapper
  training/                 # Training loops, evaluation metrics, benchmark harness
  utils/                    # Reproducibility seeding, path helpers
notebooks/                  # Jupyter notebooks for exploration
archive/Legacy/             # Previous analysis notebooks (archived)
Raw/                        # Raw Moonboard API JSON data
submissions/                # Model submissions for cross-validation benchmarking
```

## Installation

```bash
uv sync
```

## Components

### Autoencoder Route Compression

Compresses 164-dimensional binary hold vectors into a low-dimensional
bottleneck and reconstructs them. Compared against PCA across 7 compression
ratios.

```bash
moonboard-train-ae
moonboard-compare-pca
```

Key results (autoencoder vs PCA at 5% bottleneck):
| Metric | Autoencoder | PCA |
|--------|-------------|-----|
| Binary Accuracy | 97.6% | 95.1% |
| Exact Match | 2.8% | 0.06% |

### Grade Classification Benchmark

Predicts route grade (6B+ through 8A) from hold configurations using a
retrain-per-fold 5-fold cross-validation framework. Each fold trains a
fresh model from scratch.

#### Usage

```bash
# Run 5-fold CV benchmark on a submission
moonboard-benchmark --submission-dir submissions/lstm-baseline

# Use a subset for faster iteration
moonboard-benchmark --submission-dir submissions/tree-baseline --max-samples 10000

# Use a different data source
moonboard-benchmark --submission-dir submissions/2dcnn-baseline --data-path Raw/moonboard_problems_setup_master2017.json
```

#### Leaderboard (5-fold CV, 2016 dataset)

| Model | Exact (%) | Within ±1 (%) | Within ±2 (%) |
|-------|-----------|---------------|---------------|
| Random Forest | **49.55** (±0.4) | **69.65** (±0.9) | **82.88** (±0.7) |
| Perceptron (MLP) | **45.26** (±0.7) | 45.26 (±0.7) | 70.89 (±1.0) |
| LSTM | **35.46** (±1.9) | 35.46 (±1.9) | 66.31 (±1.0) |
| 2DCNN | **27.23** (±5.3) | 27.23 (±5.3) | 55.62 (±8.7) |

Results are mean ± std across 5 stratified folds on 10K sampled routes from the 2016 dataset (25K raw / 92K preprocessed). All submissions train with early stopping (patience=10, max 50 epochs).

#### Submissions

New models go in `submissions/<model-name>/main.py` and must expose a `train_and_evaluate()` function. See existing submissions for the interface:

```python
def train_and_evaluate(
    sequences: list[list[str]],
    grades: list[int],
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    seed: int = 42,
    **kwargs,
) -> dict[str, float]:
    ...
```

## Reproducibility

- All random seeds are set explicitly (`utils/reproducibility.py`)
- Dependencies are pinned with version ranges in `pyproject.toml`
- MLflow tracks every run's hyperparameters and metrics
- Model weights are not committed (code-only submission policy)

## Related Work

See [`docs/research-overview.md`](docs/research-overview.md) for a summary of existing Moonboard grade prediction papers and a reproduction roadmap.

## License

MIT