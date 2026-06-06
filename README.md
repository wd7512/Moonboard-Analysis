# Moonboard Analysis & ML Benchmark

[![CI](https://github.com/wd7512/Moonboard-Analysis/actions/workflows/ci.yml/badge.svg)](https://github.com/wd7512/Moonboard-Analysis/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![Docs](https://img.shields.io/badge/docs-gh--pages-blue)](https://wd7512.github.io/Moonboard-Analysis/)

Predict climbing route grades from Moonboard hold configurations. Benchmark
compares LSTM, CNN, MLP, Random Forest, Ridge Regression, and autoencoder
models under 5-fold cross-validation with pinned dependencies and MLflow
tracking.

[Moonboard](https://moonboard.com) routes are defined by subsets of a fixed
144-hold grid and graded on the Fontainebleau scale (6B+ to 8A+). The
dataset is imbalanced — most routes cluster around 7A to 7B+.

---

## Route Compression with Autoencoders

Compress 164-dimensional binary hold vectors into a low-dimensional bottleneck and reconstruct them. Compared against PCA across 7 compression ratios.

```bash
moonboard-train-ae
moonboard-compare-pca
```

At 5% bottleneck:

| Metric | Autoencoder | PCA |
|--------|-------------|-----|
| Binary Accuracy | 97.6% | 95.1% |
| Exact Match | 2.8% | 0.06% |

The autoencoder does noticeably better than PCA at low compression ratios.

---

## Quick Start

```bash
git clone https://github.com/wd7512/Moonboard-Analysis.git
cd Moonboard-Analysis
uv sync
```

Run a benchmark:

```bash
moonboard-benchmark --submission-dir submissions/lstm-baseline
```

Run on a subset for faster iteration:

```bash
moonboard-benchmark --submission-dir submissions/tree-baseline --max-samples 10000
```

Use a different Moonboard dataset:

```bash
moonboard-benchmark --submission-dir submissions/2dcnn-baseline --data-path Raw/moonboard_problems_setup_master2017.json
```

---

## Leaderboard

Results are mean +/- std across 5 stratified folds on the full dataset (25,738 unique routes after deduplication and preprocessing). Within-metrics use the corrected (bug-fixed) calculation.

### Full Dataset (26K routes)

<!-- LEADERBOARD-START -->

## 2016 Hold Setup

| Model | Exact (%) | Within +/-1 (%) | Within +/-2 (%) | Macro-F1 (%) |
|-------|-----------|-----------------|-----------------|--------------|
| Coral Stacking | 41.50 ± 0.85 | 66.59 ± 0.46 | 85.16 ± 0.30 | 15.08 ± 0.73 |
| deep-mlp-baseline | 40.52 ± 0.63 | 65.11 ± 0.60 | 84.35 ± 0.61 | 19.58 ± 0.68 |
| fast-mlp | 40.26 ± 0.74 | 65.34 ± 0.67 | 84.38 ± 0.68 | 19.13 ± 0.95 |
| Coral 10-Class Ensemble | 40.08 ± 0.67 | 69.81 ± 0.22 | 90.16 ± 0.21 | 19.53 ± 0.29 |
| bottom-top-lstm | 40.07 ± 0.80 | 64.42 ± 1.47 | 83.37 ± 1.52 | 18.14 ± 0.61 |
| focal-loss | 39.87 ± 1.53 | 63.70 ± 2.70 | 82.84 ± 2.18 | 17.66 ± 2.74 |
| transformer-encoder | 39.75 ± 0.94 | 64.67 ± 0.31 | 84.62 ± 0.49 | 18.90 ± 0.57 |
| perceptron-baseline | 39.65 ± 0.60 | 65.62 ± 0.42 | 84.89 ± 0.64 | 20.20 ± 0.70 |
| Coral Engineered | 39.61 ± 0.39 | 65.47 ± 0.46 | 85.28 ± 0.29 | 17.34 ± 0.34 |
| class-balanced-loss | 39.54 ± 0.57 | 63.84 ± 1.12 | 82.55 ± 0.88 | 20.71 ± 1.36 |
| gradient-boost-baseline | 39.49 ± 0.48 | 62.20 ± 0.44 | 80.78 ± 0.31 | 15.40 ± 0.42 |
| multichannel-2dcnn | 39.44 ± 0.78 | 62.61 ± 0.98 | 81.44 ± 0.74 | 17.60 ± 1.20 |
| Coral F1-Loss 10-Class | 39.16 ± 0.39 | 71.06 ± 0.46 | 91.01 ± 0.27 | 18.55 ± 0.26 |
| Coral 10-Class | 38.92 ± 0.75 | 72.11 ± 0.71 | 91.20 ± 0.42 | 20.00 ± 0.46 |
| Coral F1-Loss | 38.55 ± 0.21 | 68.07 ± 0.56 | 87.41 ± 0.29 | 18.04 ± 0.49 |
| tree-baseline | 38.51 ± 0.46 | 62.08 ± 0.41 | 80.77 ± 0.45 | 16.15 ± 0.56 |
| 2dcnn-baseline | 37.84 ± 1.93 | 62.17 ± 2.83 | 82.01 ± 2.93 | 18.18 ± 0.76 |
| ordinal-regression | 36.81 ± 0.99 | 68.46 ± 0.83 | 87.54 ± 0.33 | 22.29 ± 0.71 |
| lstm-baseline | 36.32 ± 0.74 | 64.22 ± 0.64 | 84.50 ± 1.00 | 22.08 ± 0.50 |
| Coral Gamma=3.0 | 36.10 ± 0.60 | 69.34 ± 0.54 | 88.39 ± 0.46 | 17.75 ± 0.61 |
| Coral Gamma=1.0 | 36.06 ± 0.83 | 69.35 ± 0.73 | 88.36 ± 0.49 | 17.45 ± 0.55 |
| Coral Temperature | 35.73 ± 0.62 | 69.30 ± 0.49 | 88.45 ± 0.27 | 22.67 ± 0.48 |
| ridge-baseline | 23.94 ± 0.64 | 63.63 ± 0.55 | 86.01 ± 0.24 | 11.91 ± 0.35 |

## Masters 2017 Hold Setup

| Model | Exact (%) | Within +/-1 (%) | Within +/-2 (%) | Macro-F1 (%) |
|-------|-----------|-----------------|-----------------|--------------|
| Coral Stacking | 33.97 ± 0.88 | 59.80 ± 0.55 | 80.88 ± 0.53 | 15.47 ± 0.68 |
| Coral 10-Class Ensemble | 33.39 ± 0.88 | 59.47 ± 0.60 | 83.01 ± 0.54 | 18.96 ± 0.68 |
| Coral F1-Loss 10-Class | 33.34 ± 0.81 | 59.59 ± 0.90 | 83.19 ± 0.16 | 17.49 ± 0.43 |
| deep-mlp-baseline | 33.12 ± 0.69 | 57.70 ± 0.70 | 78.61 ± 0.72 | 15.61 ± 0.71 |
| focal-loss | 32.75 ± 1.00 | 58.06 ± 0.76 | 78.90 ± 0.36 | 15.94 ± 0.55 |
| gradient-boost-baseline | 32.64 ± 0.32 | 55.86 ± 0.62 | 75.99 ± 0.67 | 17.23 ± 0.51 |
| bottom-top-lstm | 32.48 ± 0.77 | 57.13 ± 1.17 | 78.86 ± 0.82 | 14.69 ± 0.54 |
| Coral F1-Loss | 32.36 ± 0.65 | 57.86 ± 0.67 | 80.88 ± 0.48 | 15.90 ± 0.73 |
| fast-mlp | 32.35 ± 0.82 | 57.64 ± 0.81 | 78.43 ± 0.32 | 15.38 ± 0.50 |
| transformer-encoder | 32.12 ± 0.65 | 56.87 ± 1.26 | 77.95 ± 1.19 | 14.55 ± 0.42 |
| perceptron-baseline | 32.11 ± 0.60 | 57.56 ± 0.67 | 78.99 ± 0.46 | 16.74 ± 0.63 |
| tree-baseline | 31.41 ± 0.33 | 54.98 ± 0.20 | 74.88 ± 0.67 | 17.11 ± 0.66 |
| class-balanced-loss | 31.19 ± 0.79 | 55.37 ± 0.66 | 76.12 ± 1.34 | 16.39 ± 1.19 |
| multichannel-2dcnn | 31.02 ± 0.95 | 54.76 ± 1.51 | 76.08 ± 1.31 | 13.49 ± 1.50 |
| lstm-baseline | 29.77 ± 0.52 | 56.34 ± 1.38 | 77.53 ± 0.24 | 16.19 ± 1.10 |
| Coral 10-Class | 29.21 ± 0.43 | 62.13 ± 0.30 | 84.30 ± 0.64 | 21.03 ± 0.31 |
| Coral Engineered | 28.84 ± 0.85 | 60.44 ± 0.50 | 81.82 ± 0.52 | 17.34 ± 0.83 |
| Coral Gamma=1.0 | 28.53 ± 0.35 | 60.14 ± 0.62 | 81.91 ± 0.65 | 17.20 ± 0.48 |
| 2dcnn-baseline | 28.48 ± 2.35 | 51.30 ± 4.12 | 72.17 ± 5.94 | 13.26 ± 1.25 |
| Coral Temperature | 28.42 ± 0.52 | 60.49 ± 0.61 | 81.95 ± 0.72 | 17.31 ± 0.40 |
| Coral Gamma=3.0 | 28.11 ± 0.73 | 60.42 ± 0.79 | 82.24 ± 0.65 | 17.02 ± 0.71 |
| ordinal-regression | 26.92 ± 0.77 | 59.62 ± 0.95 | 80.97 ± 0.53 | 15.86 ± 0.50 |
| ridge-baseline | 19.95 ± 0.63 | 56.17 ± 0.61 | 79.37 ± 0.85 | 12.11 ± 0.31 |

## Side-by-Side Comparison

| Model | 2016 Exact (%) | 2017 Exact (%) | 2016 Macro-F1 (%) | 2017 Macro-F1 (%) |
|-------|---------------|---------------|-------------------|-------------------|
| 2dcnn-baseline | 37.84 ± 1.93 | 28.48 ± 2.35 | 18.18 ± 0.76 | 13.26 ± 1.25 |
| bottom-top-lstm | 40.07 ± 0.80 | 32.48 ± 0.77 | 18.14 ± 0.61 | 14.69 ± 0.54 |
| class-balanced-loss | 39.54 ± 0.57 | 31.19 ± 0.79 | 20.71 ± 1.36 | 16.39 ± 1.19 |
| Coral Engineered | 39.61 ± 0.39 | 28.84 ± 0.85 | 17.34 ± 0.34 | 17.34 ± 0.83 |
| Coral 10-Class | 38.92 ± 0.75 | 29.21 ± 0.43 | 20.00 ± 0.46 | 21.03 ± 0.31 |
| Coral 10-Class Ensemble | 40.08 ± 0.67 | 33.39 ± 0.88 | 19.53 ± 0.29 | 18.96 ± 0.68 |
| Coral F1-Loss | 38.55 ± 0.21 | 32.36 ± 0.65 | 18.04 ± 0.49 | 15.90 ± 0.73 |
| Coral F1-Loss 10-Class | 39.16 ± 0.39 | 33.34 ± 0.81 | 18.55 ± 0.26 | 17.49 ± 0.43 |
| Coral Gamma=1.0 | 36.06 ± 0.83 | 28.53 ± 0.35 | 17.45 ± 0.55 | 17.20 ± 0.48 |
| Coral Gamma=3.0 | 36.10 ± 0.60 | 28.11 ± 0.73 | 17.75 ± 0.61 | 17.02 ± 0.71 |
| Coral Stacking | 41.50 ± 0.85 | 33.97 ± 0.88 | 15.08 ± 0.73 | 15.47 ± 0.68 |
| Coral Temperature | 35.73 ± 0.62 | 28.42 ± 0.52 | 22.67 ± 0.48 | 17.31 ± 0.40 |
| deep-mlp-baseline | 40.52 ± 0.63 | 33.12 ± 0.69 | 19.58 ± 0.68 | 15.61 ± 0.71 |
| fast-mlp | 40.26 ± 0.74 | 32.35 ± 0.82 | 19.13 ± 0.95 | 15.38 ± 0.50 |
| focal-loss | 39.87 ± 1.53 | 32.75 ± 1.00 | 17.66 ± 2.74 | 15.94 ± 0.55 |
| gradient-boost-baseline | 39.49 ± 0.48 | 32.64 ± 0.32 | 15.40 ± 0.42 | 17.23 ± 0.51 |
| lstm-baseline | 36.32 ± 0.74 | 29.77 ± 0.52 | 22.08 ± 0.50 | 16.19 ± 1.10 |
| multichannel-2dcnn | 39.44 ± 0.78 | 31.02 ± 0.95 | 17.60 ± 1.20 | 13.49 ± 1.50 |
| ordinal-regression | 36.81 ± 0.99 | 26.92 ± 0.77 | 22.29 ± 0.71 | 15.86 ± 0.50 |
| perceptron-baseline | 39.65 ± 0.60 | 32.11 ± 0.60 | 20.20 ± 0.70 | 16.74 ± 0.63 |
| ridge-baseline | 23.94 ± 0.64 | 19.95 ± 0.63 | 11.91 ± 0.35 | 12.11 ± 0.31 |
| transformer-encoder | 39.75 ± 0.94 | 32.12 ± 0.65 | 18.90 ± 0.57 | 14.55 ± 0.42 |
| tree-baseline | 38.51 ± 0.46 | 31.41 ± 0.33 | 16.15 ± 0.56 | 17.11 ± 0.66 |

<!-- LEADERBOARD-END -->

DeepMLP trains a 5-model softmax ensemble. Neural submissions use early stopping. Ridge and Random Forest have no epoch-based training.


---

## Add Your Own Model

New models go in `submissions/<model-name>/main.py` and must expose a `train_and_evaluate()` function:

```python
def train_and_evaluate(
    sequences: list[list[str]],
    grades: list[int],
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    seed: int = 42,
    **kwargs,
) -> dict[str, float]:
    # Return exact, within_one_grade, within_two_grades, macro_f1 (all floats 0-1)
    ...
```

Reference submissions:
- `submissions/lstm-baseline/` -- LSTM grade classifier
- `submissions/tree-baseline/` -- Random Forest classifier
- `submissions/2dcnn-baseline/` -- 2D CNN classifier
- `submissions/deep-mlp-baseline/` -- Deep MLP ensemble

---

## Project Structure

```
src/moonboard_analysis/     Core Python package
  config.py                 Hyperparameter dataclasses
  data/                     Data loading, preprocessing, PyTorch datasets
  models/                   Autoencoder, LSTM, PCA wrapper
  training/                 Training loops, evaluation metrics, benchmark harness
  utils/                    Reproducibility seeding, path helpers
notebooks/                  Jupyter notebooks for exploration
archive/Legacy/             Previous analysis notebooks (archived)
Raw/                        Raw Moonboard API JSON data
submissions/                Model submissions for cross-validation benchmarking
docs/                       GitHub Pages site content
```

---

## Reproducibility

- All random seeds set explicitly (`utils/reproducibility.py`)
- Dependencies pinned with version ranges in `pyproject.toml`
- MLflow tracks every run's hyperparameters and metrics
- Model weights not committed (code-only submission policy)
- GitHub Actions CI runs lint, typecheck, and tests on every push

---

## Related Work

See [`docs/research-overview.md`](docs/research-overview.md) for a summary of existing Moonboard grade prediction papers.

---

## Citation

If you use this benchmark in research:

```bibtex
@software{moonboard_ml_benchmark_2026,
  title = {Moonboard Analysis & Machine Learning Benchmark},
  author = {Dennis, William},
  year = {2026},
  url = {https://github.com/wd7512/Moonboard-Analysis}
}
```

---

## License

MIT — see [LICENSE](LICENSE).
