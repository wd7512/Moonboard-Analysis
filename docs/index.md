---
layout: default
title: Moonboard Analysis & ML Benchmark
---

# Moonboard Analysis & Machine Learning (ML) Benchmark

> A machine learning benchmark for Moonboard climbing route grade prediction and route compression.

## What is This?

The **Moonboard ML Benchmark** is an open-source, reproducible framework for evaluating machine learning models on the [Moonboard](https://moonboard.com) — a standardized climbing wall used by climbers worldwide.

This project provides:

- **Route Compression** via Autoencoders (164-dimensional binary hold vectors → low-dimensional bottleneck)
- **Grade Classification** benchmark using 5-fold cross-validation (grades 6B+ through 8A)
- **Standardized submissions format** so anyone can contribute a model
- **Leaderboard** tracking Exact, Within-±1, and Within-±2 accuracy across 7 model architectures

## Moonboard ML Benchmark Leaderboard

| Model | Exact (%) | Within ±1 (%) | Within ±2 (%) |
|-------|-----------|---------------|---------------|
| DeepMLP (ensemble) | **49.60** (±0.7) | **49.60** (±0.7) | **70.95** (±0.7) |
| Random Forest | **49.55** (±0.4) | **69.65** (±0.9) | **82.88** (±0.7) |
| FastMLP | **46.61** (±0.8) | 46.61 (±0.8) | 71.10 (±1.0) |
| Perceptron (MLP) | **45.26** (±0.7) | 45.26 (±0.7) | 70.89 (±1.0) |
| LSTM | **35.46** (±1.9) | 35.46 (±1.9) | 66.31 (±1.0) |
| 2DCNN | **27.23** (±5.3) | 27.23 (±5.3) | 55.62 (±8.7) |
| Ridge Regression | **20.39** (±0.7) | 55.60 (±1.1) | 80.60 (±0.9) |

## Key Search Terms

This project targets these search queries:

- moonboard machine learning
- moonboard ml benchmark
- moonboard grade prediction
- moonboard analysis
- climbing route grade classification
- moonboard LSTM
- climbing machine learning benchmark
- moonboard autoencoder
- moonboard deep learning
- climbing AI grade prediction
- route compression autoencoder
- hold configuration classification

## Documentation

- [Getting Started Guide](getting-started.md) — installation, running benchmarks, understanding results
- [Blog: Building the Moonboard ML Benchmark](blog/moonboard-ml-benchmark.md) — the story behind the project

## Quick Start

```bash
git clone https://github.com/wd7512/Moonboard-Analysis.git
cd Moonboard-Analysis
uv sync
moonboard-benchmark --submission-dir submissions/lstm-baseline
```

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

## License

MIT License — see [LICENSE](https://github.com/wd7512/Moonboard-Analysis/blob/main/LICENSE)
