---
layout: default
title: Getting Started with the Moonboard ML Benchmark
---

# Getting Started with the Moonboard ML Benchmark

A step-by-step guide to running the [Moonboard Analysis & ML Benchmark](https://github.com/wd7512/Moonboard-Analysis).

## Prerequisites

- Python 3.12+
- [uv](https://github.com/astral-sh/uv) package manager
- ~2GB RAM for training

## Installation

```bash
git clone https://github.com/wd7512/Moonboard-Analysis.git
cd Moonboard-Analysis
uv sync
```

## Quick Start

Run the LSTM baseline benchmark:

```bash
moonboard-benchmark --submission-dir submissions/lstm-baseline
```

This will:
1. Download and preprocess Moonboard 2016 data
2. Run 5-fold stratified cross-validation
3. Report Exact, Within-±1, and Within-±2 accuracy

## Running Other Baselines

```bash
# Random Forest (fast, no GPU needed)
moonboard-benchmark --submission-dir submissions/tree-baseline

# 2D CNN
moonboard-benchmark --submission-dir submissions/2dcnn-baseline

# Deep MLP Ensemble (best performing)
moonboard-benchmark --submission-dir submissions/deep-mlp-baseline

# Fast MLP (best speed/accuracy tradeoff)
moonboard-benchmark --submission-dir submissions/fast-mlp-baseline
```

## Training an Autoencoder

```bash
moonboard-train-ae
moonboard-compare-pca
```

Run on a data subset for faster iteration:

```bash
moonboard-benchmark --submission-dir submissions/lstm-baseline --max-samples 5000
```

## Understanding the Output

```
Overall Metrics
| Metric          | Mean ± Std  |
|-----------------|-------------|
| exact_accuracy  | 0.3546 ± 0.0190 |
| within_one_grade| 0.3546 ± 0.0190 |
| within_two_grades| 0.6631 ± 0.0100 |
```

- **exact_accuracy**: percentage of routes where the predicted grade matches exactly
- **within_one_grade**: percentage within ±1 grade of correct
- **within_two_grades**: percentage within ±2 grades of correct

## Moonboard grades explained

The Moonboard uses the Fontainebleau grading system:

| Grade | Difficulty |
|-------|-----------|
| 6B+   | Beginner  |
| 6C    | Beginner+ |
| 6C+   | Intermediate |
| 7A    | Intermediate+ |
| 7A+   | Advanced  |
| 7B    | Advanced+ |
| 7B+   | Expert    |
| 7C    | Expert+   |
| 7C+   | Elite     |
| 8A    | Elite+    |

The benchmark treats these as 17 classes (6B+ through 8A) for the classification task.

## Next Steps

- [Add your own model](https://github.com/wd7512/Moonboard-Analysis/blob/main/CONTRIBUTING.md)
- [View the leaderboard](https://wd7512.github.io/Moonboard-Analysis/)
- [Read the research overview](https://github.com/wd7512/Moonboard-Analysis/blob/main/docs/research-overview.md)
