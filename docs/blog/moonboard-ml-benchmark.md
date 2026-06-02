---
layout: default
title: Moonboard ML Benchmark - Open Source Machine Learning for Climbing Route Grade Prediction
---

# Building an Open-Source ML Benchmark for Moonboard Climbing Route Grade Prediction

*Published June 2026*

## What is the Moonboard?

The [Moonboard](https://moonboard.com) is a **standardized climbing wall** used by climbers worldwide for training. It features a fixed grid of 144 holds arranged in an 11×18 pattern. Climbing routes — called "problems" — are defined by subsets of these holds, and each route is assigned a difficulty grade on the Fontainebleau scale from **6B+ to 8A+**.

Because every Moonboard is identical and route data is publicly available, it's an excellent platform for **machine learning research**. The question we can ask a computer is simple: *given the holds used in a route, can you predict its grade?*

## The Moonboard ML Benchmark

I built the [Moonboard Analysis & ML Benchmark](https://github.com/wd7512/Moonboard-Analysis) as an open-source framework for evaluating ML models on this task. Here's what it provides:

### Grade Classification

Given a binary vector of length 164 (one dimension per hold on the Moonboard grid), predict the route's grade among 17 classes. This is a **multi-class classification problem** with imbalanced classes — most routes cluster around grades 7A to 7B+, with fewer at the extremes.

The benchmark uses **5-fold stratified cross-validation** with a retrain-per-fold design, ensuring every model is evaluated fairly without data leakage.

### Route Compression via Autoencoders

Each route can be represented as a 164-dimensional binary vector. We use **autoencoders** to compress these into a much smaller bottleneck (down to 5% of original dimensionality) and reconstruct them. This is compared against PCA across 7 compression ratios.

### Current Leaderboard

| Model | Exact (%) | Within ±1 (%) | Within ±2 (%) |
|-------|-----------|---------------|---------------|
| DeepMLP (ensemble) | **49.60** | **49.60** | **70.95** |
| Random Forest | **49.55** | **69.65** | **82.88** |
| FastMLP | **46.61** | 46.61 | 71.10 |
| LSTM | **35.46** | 35.46 | 66.31 |
| 2D CNN | **27.23** | 27.23 | 55.62 |

On the full dataset (92K preprocessed routes), FastMLP achieves **82.56% exact accuracy** in ~5 minutes of training.

## Technical Details

The project is built in Python using PyTorch for deep learning models and scikit-learn for classical ML. It uses:

- **[uv](https://github.com/astral-sh/uv)** for dependency management
- **MLflow** for experiment tracking
- **GitHub Actions** for CI (lint, typecheck, test)
- **pytest** with 99 unit tests

### Adding Your Own Model

The submission format is simple — create a `main.py` with a `train_and_evaluate()` function:

```python
def train_and_evaluate(
    sequences: list[list[str]],
    grades: list[int],
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    seed: int = 42,
    **kwargs,
) -> dict[str, float]:
    # Your model here
    return {"exact": 0.45, "within_1": 0.70, "within_2": 0.85}
```

### Autoencoder Architecture Results

At 5% bottleneck compression:
- **Autoencoder**: 97.6% binary accuracy, 2.8% exact match
- **PCA**: 95.1% binary accuracy, 0.06% exact match

The autoencoder's non-linear compression preserves route structure significantly better than linear PCA.

## Try It Yourself

```bash
git clone https://github.com/wd7512/Moonboard-Analysis.git
cd Moonboard-Analysis
uv sync
moonboard-benchmark --submission-dir submissions/lstm-baseline
```

The full project is available on GitHub: [github.com/wd7512/Moonboard-Analysis](https://github.com/wd7512/Moonboard-Analysis)

## Related Work

Several researchers have explored Moonboard grade prediction:
- Early work used hand-crafted features from hold positions
- Deep learning approaches (LSTM, CNN) have shown promise
- The [2016 Moonboard dataset](https://github.com/FingerprintGod/Moonboard_Data) has become a standard benchmark

This project aims to standardize evaluation with a reproducible cross-validation framework and a simple submission format.

## Contributing

Contributions welcome! The project needs:
- New model submissions (Transformer, Graph Neural Network, etc.)
- Additional datasets (2017, 2019, 2020 Moonboard setups)
- Analysis notebooks comparing architectures
- Documentation and tutorials

See [CONTRIBUTING.md](https://github.com/wd7512/Moonboard-Analysis/blob/main/CONTRIBUTING.md) for guidelines.

## Citation

```bibtex
@software{moonboard_ml_benchmark_2026,
  title = {Moonboard Analysis & Machine Learning Benchmark},
  author = {Dennis, William},
  year = {2026},
  url = {https://github.com/wd7512/Moonboard-Analysis}
}
```

---

*Built by [William Dennis](https://github.com/wd7512). Licensed under MIT.*
