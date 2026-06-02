---
layout: default
title: Moonboard Analysis & ML Benchmark
---

# Moonboard Analysis & ML Benchmark

A machine learning benchmark for Moonboard climbing route grade prediction and route compression.

## Overview

Route Compression via Autoencoders: 164-dimensional binary hold vectors compressed to a low-dimensional bottleneck and reconstructed. Compared against PCA across 7 ratios.

Grade Classification: 5-fold cross validation benchmark predicting Fontainebleau grades (6B+ through 8A) from hold configurations. 17 classes, imbalanced distribution.

Standardized submission format for contributing new models.

## Leaderboard

| Model | Exact (%) | Within +/-1 (%) | Within +/-2 (%) |
|-------|-----------|-----------------|-----------------|
| DeepMLP (ensemble) | 49.60 (0.7) | 49.60 (0.7) | 70.95 (0.7) |
| Random Forest | 49.55 (0.4) | 69.65 (0.9) | 82.88 (0.7) |
| FastMLP | 46.61 (0.8) | 46.61 (0.8) | 71.10 (1.0) |
| Perceptron (MLP) | 45.26 (0.7) | 45.26 (0.7) | 70.89 (1.0) |
| LSTM | 35.46 (1.9) | 35.46 (1.9) | 66.31 (1.0) |
| 2DCNN | 27.23 (5.3) | 27.23 (5.3) | 55.62 (8.7) |
| Ridge Regression | 20.39 (0.7) | 55.60 (1.1) | 80.60 (0.9) |

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
