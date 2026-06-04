---
layout: default
title: Moonboard Analysis & ML Benchmark
---

# Moonboard Analysis & ML Benchmark

A machine learning benchmark for Moonboard climbing route grade prediction and route compression.

[View on GitHub](https://github.com/wd7512/Moonboard-Analysis)

## Overview

Route Compression via Autoencoders: 164-dimensional binary hold vectors compressed to a low-dimensional bottleneck and reconstructed. Compared against PCA across 7 ratios.

Grade Classification: 5-fold cross validation benchmark predicting Fontainebleau grades (6B+ through 8A) from hold configurations. 17 classes, imbalanced distribution.

Standardized submission format for contributing new models.

## Leaderboard

Results are mean +/- std across 5 stratified folds on the full dataset (25,738 unique routes after deduplication and preprocessing).

<!-- LEADERBOARD-START -->
| Model | Exact (%) | Within +/-1 (%) | Within +/-2 (%) | Macro-F1 (%) |
|-------|-----------|-----------------|-----------------|--------------|
| DeepMLP (ensemble) | 40.76 ± 0.62 | 65.46 ± 0.55 | 84.49 ± 0.53 | 16.99 ± 0.32 |
| Focal Loss (MLP) | 40.73 ± 0.71 | 65.56 ± 0.46 | 84.44 ± 0.51 | 16.72 ± 1.34 |
| FastMLP | 40.50 ± 0.50 | 64.89 ± 0.69 | 83.69 ± 0.73 | 15.44 ± 0.82 |
| Perceptron (MLP) | 40.18 ± 0.51 | 65.67 ± 0.93 | 84.80 ± 0.82 | 17.29 ± 0.42 |
| Bottom-to-Top LSTM | 40.17 ± 0.53 | 65.10 ± 0.44 | 83.40 ± 1.06 | 14.86 ± 0.68 |
| Class-Balanced Loss (MLP) | 39.70 ± 0.90 | 64.26 ± 0.76 | 83.14 ± 0.96 | 17.98 ± 1.21 |
| Transformer Encoder | 39.51 ± 0.87 | 64.12 ± 1.01 | 82.59 ± 1.01 | 15.18 ± 0.70 |
| Gradient Boost | 39.49 ± 0.48 | 62.20 ± 0.44 | 80.78 ± 0.31 | 16.69 ± 0.45 |
| Multi-Channel 2DCNN | 39.31 ± 0.70 | 62.41 ± 0.94 | 81.20 ± 0.65 | 14.17 ± 0.22 |
| Random Forest | 38.51 ± 0.46 | 62.08 ± 0.41 | 80.77 ± 0.45 | 17.50 ± 0.61 |
| LSTM | 37.05 ± 1.25 | 63.40 ± 0.91 | 83.76 ± 0.57 | 17.81 ± 0.85 |
| CORAL Ordinal Regression | 37.02 ± 0.81 | 68.16 ± 0.56 | 87.23 ± 0.24 | 18.19 ± 0.76 |
| 2DCNN | 36.81 ± 5.30 | 60.66 ± 5.89 | 79.99 ± 4.91 | 14.24 ± 1.14 |
| Ridge Regression | 23.94 ± 0.64 | 63.63 ± 0.55 | 86.30 ± 0.25 | 12.90 ± 0.38 |
<!-- LEADERBOARD-END -->

## Quick Start

```bash
git clone https://github.com/wd7512/Moonboard-Analysis.git
cd Moonboard-Analysis
uv sync
moonboard-benchmark --submission-dir submissions/lstm-baseline
```

## Citation

```bibtex
@software{moonboard_ml_benchmark_2026,
  title = {Moonboard Analysis & Machine Learning Benchmark},
  author = {Dennis, William},
  year = {2026},
  url = {https://github.com/wd7512/Moonboard-Analysis}
}
```

## License

MIT
