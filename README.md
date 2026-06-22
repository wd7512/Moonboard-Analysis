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
| DeepMLP (ensemble) | 40.52 ± 0.63 | 65.11 ± 0.60 | 84.35 ± 0.61 | 19.58 ± 0.68 |
| FastMLP | 40.26 ± 0.74 | 65.34 ± 0.67 | 84.38 ± 0.68 | 19.13 ± 0.95 |
| Bottom-to-Top LSTM | 40.07 ± 0.80 | 64.42 ± 1.47 | 83.37 ± 1.52 | 18.14 ± 0.61 |
| Focal Loss (MLP) | 39.87 ± 1.53 | 63.70 ± 2.70 | 82.84 ± 2.18 | 17.66 ± 2.74 |
| Transformer Encoder | 39.75 ± 0.94 | 64.67 ± 0.31 | 84.62 ± 0.49 | 18.90 ± 0.57 |
| Perceptron Baseline | 39.65 ± 0.60 | 65.62 ± 0.42 | 84.89 ± 0.64 | 20.20 ± 0.70 |
| Class-Balanced Loss (MLP) | 39.54 ± 0.57 | 63.84 ± 1.12 | 82.55 ± 0.88 | 20.71 ± 1.36 |
| Gradient Boost | 39.49 ± 0.48 | 62.20 ± 0.44 | 80.78 ± 0.31 | 15.40 ± 0.42 |
| Multi-Channel 2D CNN | 39.44 ± 0.78 | 62.61 ± 0.98 | 81.44 ± 0.74 | 17.60 ± 1.20 |
| Random Forest | 38.51 ± 0.46 | 62.08 ± 0.41 | 80.77 ± 0.45 | 16.15 ± 0.56 |
| 2D CNN Baseline | 37.84 ± 1.93 | 62.17 ± 2.83 | 82.01 ± 2.93 | 18.18 ± 0.76 |
| CORAL-DeepMLP Ensemble | 39.95 ± 0.95 | **72.50 ± 0.66** | **91.53 ± 0.39** | **28.78 ± 0.55** |
| Ordinal Regression (CORAL) | 36.81 ± 0.99 | 68.46 ± 0.83 | 87.54 ± 0.33 | 22.29 ± 0.71 |
| LSTM Baseline | 36.32 ± 0.74 | 64.22 ± 0.64 | 84.50 ± 1.00 | 22.08 ± 0.50 |
| Ridge Regression | 23.94 ± 0.64 | 63.63 ± 0.55 | 86.01 ± 0.24 | 11.91 ± 0.35 |

## Masters 2017 Hold Setup

| Model | Exact (%) | Within +/-1 (%) | Within +/-2 (%) | Macro-F1 (%) |
|-------|-----------|-----------------|-----------------|--------------|
| DeepMLP (ensemble) | 32.14 ± 0.18 | 56.29 ± 0.02 | 76.67 ± 0.40 | 13.94 ± 0.18 |
| Focal Loss (MLP) | 31.96 ± 0.20 | 56.56 ± 0.37 | 77.13 ± 0.19 | 14.94 ± 0.02 |
| FastMLP | 31.81 ± 0.03 | 55.80 ± 0.13 | 76.06 ± 0.07 | 13.49 ± 0.92 |
| Bottom-to-Top LSTM | 31.65 ± 0.32 | 56.33 ± 0.07 | 76.86 ± 0.14 | 14.18 ± 0.26 |
| Perceptron Baseline | 31.57 ± 0.21 | 56.71 ± 0.27 | 78.17 ± 0.07 | 16.66 ± 0.07 |
| Random Forest | 31.41 ± 0.33 | 54.98 ± 0.20 | 74.88 ± 0.67 | 17.11 ± 0.66 |
| Transformer Encoder | 31.34 ± 0.36 | 56.29 ± 0.10 | 76.91 ± 0.96 | 14.06 ± 0.55 |
| Multi-Channel 2D CNN | 30.77 ± 0.41 | 54.22 ± 0.11 | 74.97 ± 0.77 | 13.05 ± 0.01 |
| Gradient Boost | 30.46 ± 0.02 | 54.80 ± 0.24 | 75.58 ± 0.63 | 17.12 ± 0.19 |
| LSTM Baseline | 29.61 ± 0.19 | 54.80 ± 0.21 | 76.67 ± 1.52 | 16.29 ± 0.32 |
| CORAL-DeepMLP Ensemble | pending | — | pending | — |
| Class-Balanced Loss (MLP) | 28.94 ± 0.13 | 52.92 ± 0.97 | 71.98 ± 1.06 | 15.86 ± 0.61 |
| 2D CNN Baseline | 28.20 ± 2.52 | 49.73 ± 3.84 | 70.90 ± 4.24 | 12.38 ± 0.00 |
| Ordinal Regression (CORAL) | 22.06 ± 0.27 | 56.96 ± 0.12 | 78.79 ± 0.37 | 13.93 ± 0.26 |
| Ridge Regression | 19.95 ± 0.63 | 56.17 ± 0.61 | 79.37 ± 0.85 | 12.11 ± 0.31 |

## Side-by-Side Comparison

| Model | 2016 Exact (%) | 2017 Exact (%) | 2016 Macro-F1 (%) | 2017 Macro-F1 (%) |
|-------|---------------|---------------|-------------------|-------------------|
| 2D CNN Baseline | 37.84 ± 1.93 | 28.20 ± 2.52 | 18.18 ± 0.76 | 12.38 ± 0.00 |
| Bottom-to-Top LSTM | 40.07 ± 0.80 | 31.65 ± 0.32 | 18.14 ± 0.61 | 14.18 ± 0.26 |
| Class-Balanced Loss (MLP) | 39.54 ± 0.57 | 28.94 ± 0.13 | 20.71 ± 1.36 | 15.86 ± 0.61 |
| DeepMLP (ensemble) | 40.52 ± 0.63 | 32.14 ± 0.18 | 19.58 ± 0.68 | 13.94 ± 0.18 |
| FastMLP | 40.26 ± 0.74 | 31.81 ± 0.03 | 19.13 ± 0.95 | 13.49 ± 0.92 |
| Focal Loss (MLP) | 39.87 ± 1.53 | 31.96 ± 0.20 | 17.66 ± 2.74 | 14.94 ± 0.02 |
| Gradient Boost | 39.49 ± 0.48 | 30.46 ± 0.02 | 15.40 ± 0.42 | 17.12 ± 0.19 |
| LSTM Baseline | 36.32 ± 0.74 | 29.61 ± 0.19 | 22.08 ± 0.50 | 16.29 ± 0.32 |
| Multi-Channel 2D CNN | 39.44 ± 0.78 | 30.77 ± 0.41 | 17.60 ± 1.20 | 13.05 ± 0.01 |
| Ordinal Regression (CORAL) | 36.81 ± 0.99 | 22.06 ± 0.27 | 22.29 ± 0.71 | 13.93 ± 0.26 |
| CORAL-DeepMLP Ensemble | 39.95 ± 0.95 | pending | 28.78 ± 0.55 | pending |
| Perceptron Baseline | 39.65 ± 0.60 | 31.57 ± 0.21 | 20.20 ± 0.70 | 16.66 ± 0.07 |
| Ridge Regression | 23.94 ± 0.64 | 19.95 ± 0.63 | 11.91 ± 0.35 | 12.11 ± 0.31 |
| Transformer Encoder | 39.75 ± 0.94 | 31.34 ± 0.36 | 18.90 ± 0.57 | 14.06 ± 0.55 |
| Random Forest | 38.51 ± 0.46 | 31.41 ± 0.33 | 16.15 ± 0.56 | 17.11 ± 0.66 |

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
