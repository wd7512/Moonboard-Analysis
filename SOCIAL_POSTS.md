# Moonboard ML Benchmark - Social Post Drafts

Last updated: June 2026

---

## Reddit: r/Moonboard

**Title:** I built an open-source ML benchmark for predicting Moonboard route grades

**Body:**

Hey r/Moonboard,

I've been working on a machine learning project that I thought this community might find interesting.

The idea: given the holds used in a Moonboard problem, can a machine learning model predict its grade?

I've built the **Moonboard ML Benchmark** — an open-source framework that:

- Uses the public Moonboard 2016 dataset (~25K problems, 92K preprocessed routes)
- Evaluates models using 5-fold stratified cross-validation
- Supports LSTM, CNN, Random Forest, MLP, and autoencoder architectures
- Provides a simple submission format so anyone can add their model

**Current best results:**
- Deep MLP Ensemble: 49.6% exact accuracy (within ±1 grade: 70.9%)
- FastMLP: 82.6% exact accuracy on the full 92K dataset in ~5 min
- Autoencoder route compression: 97.6% binary accuracy at 5% bottleneck

The project is MIT licensed and available here:
https://github.com/wd7512/Moonboard-Analysis

I'd love contributions — especially if anyone has ideas for features that would help with climbing training analysis, or wants to try different model architectures.

Also has an autoencoder module for route compression if you're into dimensionality reduction / creative route generation.

---

## Reddit: r/climbing

**Title:** [Project] Machine learning for Moonboard grade prediction — open-source benchmark

**Body:**

I built an open-source benchmark for evaluating machine learning models on Moonboard climbing route grade prediction.

The Moonboard is great for ML because every board is identical and route data is public. The task: given which holds a route uses (out of 144 on the grid), predict the Fontainebleau grade.

**What it includes:**
- 5-fold cross-validation benchmark harness
- LSTM, CNN, Random Forest, MLP baselines
- Autoencoder for route compression (compresses 164-dim hold vectors down to 5%)
- CI, tests, and experiment tracking

**Best results so far:**
- 49.6% exact grade prediction (5-fold CV, 10K routes)
- 82.6% on full dataset
- Autoencoder beats PCA by 2.5%+ at low compression ratios

If any ML folks here are route-setters or coaches, I'd be curious whether hold-sequence features (not just hold presence) could improve predictions.

Repo: https://github.com/wd7512/Moonboard-Analysis

---

## Reddit: r/MachineLearning

**Title:** [P] Moonboard ML Benchmark — an open-source reproducible benchmark for climbing route grade classification

**Body:**

Released an open-source benchmark for a niche but interesting multi-class classification task: predicting climbing route grades on the Moonboard from hold configurations.

**Task:** Multi-class classification (17 classes, 6B+ through 8A). Input is a 164-dimensional binary vector representing which holds are used on the standardized Moonboard grid. Data from the public Moonboard API (2016 setup, ~25K routes, 92K preprocessed with duplicates).

**Framework features:**
- 5-fold stratified CV with retrain-per-fold design
- Simple submission format (one function: `train_and_evaluate`)
- MLflow experiment tracking
- Pinned dependencies, reproducible seeds, CI

**Results:**

| Model | Exact (%) | Within ±1 (%) | Within ±2 (%) |
|-------|-----------|---------------|---------------|
| DeepMLP (ensemble) | 49.60 | 49.60 | 70.95 |
| Random Forest | 49.55 | 69.65 | 82.88 |
| FastMLP | 46.61 | 46.61 | 71.10 |
| LSTM | 35.46 | 35.46 | 66.31 |
| 2D CNN | 27.23 | 27.23 | 55.62 |

Also includes an autoencoder compression module (compresses routes to 5% of original dimensionality, outperforms PCA).

I think this could be a good benchmark for comparing sequence models vs set models vs tree-based approaches on a real-world, small-ish dataset.

Code: https://github.com/wd7512/Moonboard-Analysis

---

## Hacker News: Show HN

**Title:** Show HN: Moonboard ML Benchmark — predict climbing route grades with machine learning

**Body:**

I built an open-source benchmark for predicting Moonboard climbing route grades from hold configurations.

The Moonboard is a standardized climbing wall (144 holds in a fixed grid) used worldwide. Route data is publicly available, which makes it a nice ML benchmark: multi-class classification (17 grades) from 164-dim binary vectors.

The repo includes:
- 5-fold CV benchmark with retrain-per-fold design
- LSTM, CNN, Random Forest, and MLP baselines
- Autoencoder route compression
- A simple submission format so anyone can add their model

Best result: 49.6% exact grade prediction (82.6% on full data), with Random Forest nearly tied. I'm curious whether transformers or GNNs could do better on this.

https://github.com/wd7512/Moonboard-Analysis

---

## X/Twitter

**Tweet 1:**
I built an open-source ML benchmark for predicting Moonboard climbing route grades 🧗🤖

Given which holds a route uses (out of 144 on the grid), predict its difficulty grade.

Best model: 49.6% exact accuracy (5-fold CV). Autoencoder compression for route embeddings too.

github.com/wd7512/Moonboard-Analysis

#MachineLearning #Climbing #OpenSource

**Tweet 2:**
New open-source benchmark: Moonboard climbing route grade prediction

17 classes (6B+ to 8A), 164-dim binary input vectors
5-fold CV, reproducible, easy submission format
LSTM, CNN, RF, MLP baselines included
Autoencoder route compression beats PCA by 2.5%+

github.com/wd7512/Moonboard-Analysis

#ML #DeepLearning #Climbing #DataScience

---

## LinkedIn Post

**Title:** Moonboard ML Benchmark — Open Source Machine Learning for Climbing

I've released an open-source benchmark exploring an unusual intersection: machine learning and rock climbing.

The Moonboard is a standardized climbing wall used by climbers worldwide. Because every board is identical and route data is public, it turns out to be a great testbed for ML models.

The benchmark evaluates LSTM, CNN, Random Forest, and MLP architectures on the task of predicting climbing route grades from hold configurations (multi-class classification with 17 grade categories).

**Key results:**
- 49.6% exact grade prediction (5-fold cross-validation)
- 82.6% accuracy on the full dataset
- Autoencoder-based route compression outperforms PCA at low compression ratios

Built in Python with PyTorch, scikit-learn, and MLflow for experiment tracking.

I'd love to hear from anyone at the ML/sports science intersection — particularly those interested in how route-setting could benefit from ML analysis.

Repository: https://github.com/wd7512/Moonboard-Analysis

#MachineLearning #OpenSource #Climbing #DataScience #AI
