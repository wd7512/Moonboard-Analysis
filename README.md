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
| Model | Exact (%) | Within +/-1 (%) | Within +/-2 (%) | Macro-F1 (%) |
|-------|-----------|-----------------|-----------------|--------------|
| DeepMLP (ensemble) | 40.76 ± 0.62 | 65.46 ± 0.55 | 84.49 ± 0.53 | 16.99 ± 0.32 |
| FastMLP | 40.50 ± 0.50 | 64.89 ± 0.69 | 83.69 ± 0.73 | 15.44 ± 0.82 |
| Perceptron (MLP) | 40.18 ± 0.51 | 65.67 ± 0.93 | 84.80 ± 0.82 | 17.29 ± 0.42 |
| Gradient Boost | 39.49 ± 0.48 | 62.20 ± 0.44 | 80.78 ± 0.31 | 16.69 ± 0.45 |
| Random Forest | 38.51 ± 0.46 | 62.08 ± 0.41 | 80.77 ± 0.45 | 17.50 ± 0.61 |
| LSTM | 37.05 ± 1.25 | 63.40 ± 0.91 | 83.76 ± 0.57 | 17.81 ± 0.85 |
| 2DCNN | 36.81 ± 5.30 | 60.66 ± 5.89 | 79.99 ± 4.91 | 14.24 ± 1.14 |
| Ridge Regression | 23.94 ± 0.64 | 63.63 ± 0.55 | 86.30 ± 0.25 | 12.90 ± 0.38 |
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
