# Moonboard Analysis & ML Benchmark

[![CI](https://github.com/wd7512/Moonboard-Analysis/actions/workflows/ci.yml/badge.svg)](https://github.com/wd7512/Moonboard-Analysis/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)

A machine learning benchmark for Moonboard climbing route grade prediction and route compression. Predict route grades from hold configurations using LSTM, CNN, Random Forest, MLP, and autoencoder models. Uses 5-fold cross-validation with pinned dependencies and MLflow tracking.

---

## What is Moonboard?

The [Moonboard](https://moonboard.com) is a standardized climbing wall used for training. It has a fixed grid of 144 holds (11 rows x 18 columns minus corner cutoffs). Routes are defined by subsets of these holds and graded on the Fontainebleau scale from 6B+ to 8A+.

Route data is publicly available via the Moonboard API, which makes it a useful testbed for reproducible ML experiments on grade classification and route compression.

---

## Moonboard ML Benchmark

A framework for evaluating ML models on the task of predicting climbing route grades from hold configurations. Input is a binary vector of length 164 (one dimension per hold), output is one of 17 grade classes (6B+ through 8A). The dataset is imbalanced — most routes cluster around 7A to 7B+.

The benchmark uses 5-fold stratified cross-validation with a retrain-per-fold design. Models available so far:

- Deep learning: LSTM, 2D CNN, MLP, DeepMLP Ensemble, FastMLP
- Classical ML: Random Forest, Ridge Regression
- Dimensionality reduction: Autoencoder, PCA

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

Results are mean +/- std across 5 stratified folds on a 10K stratified subsample of the 2016 Moonboard dataset.

### 5-Fold CV

| Model | Exact (%) | Within +/-1 (%) | Within +/-2 (%) | Macro-F1 (%) |
|-------|-----------|-----------------|-----------------|--------------|
| DeepMLP (ensemble) | 49.60 (0.7) | 49.60 (0.7) | 70.95 (0.7) | TBD |
| Random Forest | 49.55 (0.4) | 69.65 (0.9) | 82.88 (0.7) | TBD |
| FastMLP | 46.61 (0.8) | 46.61 (0.8) | 71.10 (1.0) | TBD |
| Perceptron (MLP) | 45.26 (0.7) | 45.26 (0.7) | 70.89 (1.0) | TBD |
| LSTM | 35.46 (1.9) | 35.46 (1.9) | 66.31 (1.0) | TBD |
| 2DCNN | 27.23 (5.3) | 27.23 (5.3) | 55.62 (8.7) | TBD |
| Ridge Regression | 20.39 (0.7) | 55.60 (1.1) | 80.60 (0.9) | TBD |

DeepMLP trains a 5-model softmax ensemble. Neural submissions use early stopping. Ridge and Random Forest have no epoch-based training.

[Detailed results](results.md) | [DeepMLP results](results_deepmlp.md)

---

## Add Your Own Model

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
    # Return exact, within_1, within_2 (all floats 0-1)
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
