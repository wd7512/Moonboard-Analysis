# 🧗 Moonboard Analysis & Machine Learning (ML) Benchmark

[![CI](https://github.com/wd7512/Moonboard-Analysis/actions/workflows/ci.yml/badge.svg)](https://github.com/wd7512/Moonboard-Analysis/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![GitHub Pages](https://img.shields.io/badge/GitHub%20Pages-live-green)](https://wd7512.github.io/Moonboard-Analysis/)

> **A machine learning benchmark for Moonboard climbing route grade prediction and route compression.**
>
> Predict climbing route grades from hold configurations using LSTM, CNN, Random Forest, MLP, and autoencoder models. Built for reproducibility with 5-fold cross-validation, pinned dependencies, and MLflow tracking.

---

## Table of Contents

- [What is Moonboard?](#what-is-moonboard)
- [Moonboard ML Benchmark](#moonboard-ml-benchmark)
- [Route Compression with Autoencoders](#route-compression-with-autoencoders)
- [Quick Start](#quick-start)
- [Leaderboard](#leaderboard)
- [Add Your Own Model](#add-your-own-model)
- [Project Structure](#project-structure)
- [Reproducibility](#reproducibility)
- [Related Work](#related-work)
- [Citation](#citation)
- [License](#license)

---

## What is Moonboard?

The [Moonboard](https://moonboard.com) is a **standardized climbing wall** used by climbers worldwide for training. It features a fixed grid of 144 holds (11 rows × 18 columns minus corner cutoffs), and climbing routes are defined by subsets of these holds. Routes are graded on the Fontainebleau scale from **6B+ to 8A+** based on difficulty.

This project uses the Moonboard as a **machine learning benchmark platform** — route data is publicly available via the Moonboard API, making it ideal for reproducible experiments in **grade classification** and **route compression**.

---

## Moonboard ML Benchmark

The **Moonboard ML Benchmark** is a standardized framework for evaluating machine learning models on the task of **predicting climbing route grades from hold configurations**.

### Key Features

- **5-fold stratified cross-validation** — each fold trains a fresh model from scratch
- **Reproducible** — all random seeds set, dependencies pinned, MLflow tracks every run
- **Extensible** — add new models via the submission format
- **Multiple architectures evaluated** — LSTM, CNN, Random Forest, MLP, Ridge, Deep Ensemble

### Supported Models

| Category | Models |
|----------|--------|
| **Deep Learning** | LSTM, 2D CNN, MLP, DeepMLP Ensemble, FastMLP |
| **Classical ML** | Random Forest, Ridge Regression |
| **Dimensionality Reduction** | Autoencoder, PCA |

### Moonboard Grade Classification Task

Given a binary vector of length 164 (each dimension = one hold on the Moonboard grid), predict the route's Fontainebleau grade among 17 classes (6B+ through 8A). This is a **multi-class classification problem** with imbalanced classes — making it a challenging benchmark for ML practitioners.

---

## Route Compression with Autoencoders

Compress 164-dimensional **binary hold vectors** into a low-dimensional bottleneck and reconstruct them. The autoencoder architecture is compared against PCA across 7 compression ratios.

```bash
moonboard-train-ae
moonboard-compare-pca
```

### Autoencoder vs PCA Results

| Metric | Autoencoder (5% bottleneck) | PCA (5% bottleneck) |
|--------|---------------------------|---------------------|
| Binary Accuracy | **97.6%** | 95.1% |
| Exact Match | **2.8%** | 0.06% |

The autoencoder significantly outperforms PCA at low compression ratios, preserving route structure better than linear methods.

---

## Quick Start

### Installation

```bash
git clone https://github.com/wd7512/Moonboard-Analysis.git
cd Moonboard-Analysis
uv sync
```

### Run a Benchmark

```bash
# Run 5-fold CV benchmark on a submission
moonboard-benchmark --submission-dir submissions/lstm-baseline

# Use a subset for faster iteration
moonboard-benchmark --submission-dir submissions/tree-baseline --max-samples 10000

# Use a different Moonboard dataset (2016 vs 2017)
moonboard-benchmark --submission-dir submissions/2dcnn-baseline --data-path Raw/moonboard_problems_setup_master2017.json
```

### Train an Autoencoder

```bash
moonboard-train-ae
```

### Compare Autoencoder vs PCA

```bash
moonboard-compare-pca
```

---

## Leaderboard

Results are mean ± std across 5 stratified folds on 10K sampled routes from the 2016 Moonboard dataset (25K raw / 92K preprocessed).

### 5-Fold CV Leaderboard

| Model | Exact (%) | Within ±1 (%) | Within ±2 (%) |
|-------|-----------|---------------|---------------|
| DeepMLP (ensemble) | **49.60** (±0.7) | **49.60** (±0.7) | **70.95** (±0.7) |
| Random Forest | **49.55** (±0.4) | **69.65** (±0.9) | **82.88** (±0.7) |
| FastMLP | **46.61** (±0.8) | 46.61 (±0.8) | 71.10 (±1.0) |
| Perceptron (MLP) | **45.26** (±0.7) | 45.26 (±0.7) | 70.89 (±1.0) |
| LSTM | **35.46** (±1.9) | 35.46 (±1.9) | 66.31 (±1.0) |
| 2DCNN | **27.23** (±5.3) | 27.23 (±5.3) | 55.62 (±8.7) |
| Ridge Regression | **20.39** (±0.7) | 55.60 (±1.1) | 80.60 (±0.9) |

### Full-Data Results (92K preprocessed routes)

| Model | Exact (%) | Within ±1 (%) | Within ±2 (%) | Training Time |
|-------|-----------|---------------|---------------|---------------|
| FastMLP | **82.56** (±0.4) | 82.56 (±0.4) | 90.01 (±0.4) | ~5 min |

**Notes:** DeepMLP trains a 5-model softmax ensemble; all neural submissions use early stopping; Ridge and Random Forest have no epoch-based training.

[Detailed results](results.md) | [DeepMLP results](results_deepmlp.md)

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
    """Train a model and return evaluation metrics.

    Returns dict with keys: exact, within_1, within_2 (all floats 0-1)
    """
    ...
```

See existing submissions for reference:
- `submissions/lstm-baseline/` — LSTM grade classifier
- `submissions/tree-baseline/` — Random Forest classifier
- `submissions/2dcnn-baseline/` — 2D CNN classifier
- `submissions/deep-mlp-baseline/` — Deep MLP ensemble

---

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
docs/                       # GitHub Pages site content
```

---

## Reproducibility

- All random seeds are set explicitly (`utils/reproducibility.py`)
- Dependencies are pinned with version ranges in `pyproject.toml`
- MLflow tracks every run's hyperparameters and metrics
- Model weights are not committed (code-only submission policy)
- GitHub Actions CI runs lint, typecheck, and tests on every push

---

## Related Work

See [`docs/research-overview.md`](docs/research-overview.md) for a summary of existing Moonboard grade prediction papers, including:

- Deep learning approaches to climbing route classification
- Sequence models for hold-chain analysis
- Comparison of CNN vs LSTM architectures for route grading

---

## Citation

If you use this benchmark in research, please cite:

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

MIT License — see [LICENSE](LICENSE).
