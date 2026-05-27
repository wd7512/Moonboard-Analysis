# Moonboard ML Research Overview

A summary of existing work on Moonboard climbing route grade prediction and generation, plus concrete next steps for reproducing and extending results.

---

## Existing Work

### 1. Duh & Chang (2021) — RNN for MoonBoard Route Classification and Generation

**Source:** [arXiv:2102.01788](https://arxiv.org/abs/2102.01788) (cs.LG / cs.CV)
**Also published as:** Stanford CS230 project report, Spring 2020 ([PDF](http://cs230.stanford.edu/projects_spring_2020/reports/38850664.pdf))

**Key idea:** Introduced "BetaMove", a preprocessing pipeline that converts Moonboard hold sequences into human-like move sequences (mimicking a climber's hand sequence). Trained an RNN (LSTM) for both grade prediction and route generation.

**Results:**
| Metric | GradeNet (RNN + BetaMove) | Human Performance |
|--------|---------------------------|-------------------|
| Exact match | 46.7% | ~45% |
| Within ±1 grade | 84.7% | ~85% |

**Significance:** First model to reach near-human-level grade prediction on Moonboard data. Also demonstrated route generation (DeepRouteSet) producing higher-quality routes than prior LSTM-based generators. The BetaMove preprocessing was shown to be critical — an RNN without move-sequence preprocessing reached only 34.7% exact match.

**Dataset:** 25,096 problems scraped from Moonboard app (filtered for quality), grades V4–V13 (Hueco scale), 80/10/10 train/dev/test split.

---

### 2. Petashvili & Rodda (2023) — Board-to-Board: Evaluating Moonboard Grade Prediction Generalization

**Source:** [arXiv:2311.12419](https://arxiv.org/abs/2311.12419) (cs.LG / cs.CV)

**Key idea:** Evaluated classical ML and deep learning models for grade prediction across multiple Moonboard editions (2016, 2017, 2019). Introduced a novel vision-based grade prediction approach using rendered route images. Focused on **cross-edition generalization** — can a model trained on one Moonboard configuration predict grades on a different one?

**Results (single-edition, 2016 dataset):**
| Model | MAE | RMSE | Exact Acc. | Within ±1 |
|-------|-----|------|------------|-----------|
| 2DCNN (best) | 0.86 | 1.12 | 42% | 84% |
| LSTM | ~0.95 | ~1.20 | — | — |
| RBF SVM (best classical) | 0.98 | 1.28 | — | — |

**Cross-edition generalization (trained on 2 editions, tested on 3rd):**
| Model | MAE | RMSE |
|-------|-----|------|
| LSTM (best generalizer) | 2.35 | 2.93 |

**Vision-based approach (ResNet50 on rendered route images):** 1.84 MAE / 2.30 RMSE — promising direction but currently well below tabular/sequential methods.

**Key finding:** The 2DCNN architecture excels at single-edition prediction by learning spatial relationships between holds through convolutional filters. However, cross-edition generalization remains an open problem — all models perform significantly worse on unseen board configurations.

**Code:** https://github.com/a1773620/Moonboard-Grade-Prediction

---

### 3. Drummond & Popinga (2021) — Bayesian Inference of the Climbing Grade Scale

**Source:** [arXiv:2111.08140](https://arxiv.org/abs/2111.08140) (stat.AP / cs.LG)

**Key idea:** Applied a dynamic Bradley-Terry model (whole-history rating) to climbing ascent data to estimate the fundamental difficulty scale. Used MCMC inference on a curated dataset of regular climbers.

**Key results:**
- Climbing grade scales are **logarithmic** in difficulty (analogous to decibels or stellar magnitude)
- Each grade increment corresponds to a ~2.1× increase in difficulty (Ewbank, French, UIAA systems)
- The V-scale (bouldering) corresponds to a ~3.17× increase per grade increment
- Results align with Weber-Fechner psychophysical laws

**Significance:** Provides a statistically rigorous foundation for treating climbing grades as an ordinal scale with known mathematical properties. Relevant for choosing loss functions and evaluation metrics in ML models.

---

### 4. Dobles (2017) — Machine Learning Methods for Climbing Route Classification

**Source:** [Semantic Scholar](https://api.semanticscholar.org/CorpusID:44500)

**Key idea:** Early application of ML to climbing route classification using the Moonboard dataset. One of the first systematic attempts at automated grade prediction.

**Results:** ~34% exact accuracy using CNN-based approaches.

---

### 5. Kempen (2019) — A Fair Grade: Assessing Difficulty of Climbing Routes Through Machine Learning

**Source:** University of Twente, Formal Methods and Tools

**Key idea:** Manually decomposed routes into individual moves and used human-labeled move sequences for binary (easy/hard) classification.

**Results:** ~64% binary classification accuracy using k-fold cross-validation.

**Limitation:** Manual move decomposition doesn't scale and introduces human bias.

---

### 6. Houghton et al. — LSTM Route Generator (cited in Duh & Chang)

**Source:** Cited in Duh & Chang (2021); original work on LSTM-based Moonboard route generation.

**Key idea:** Generated Moonboard routes using LSTM trained on raw hold sequences without move preprocessing.

**Limitation:** Generated routes contained redundant holds and unnatural move sequences. Duh & Chang's BetaMove preprocessing significantly improved generation quality.

---

### 7. Phillips et al. — "StrangeBeta" Automatic Route Setter

**Key idea:** Non-ML approach using mathematical characteristics of strange attractors to generate climbing routes. Used a specialized route description language (CRDL).

**Limitation:** Not applied to MoonBoard; CRDL language is ambiguous even to climbing experts.

---

## This Repository: Current State

The **Moonboard-Analysis** repo implements:

1. **Autoencoder route compression** — Compresses 164-dim binary hold vectors into low-dimensional bottlenecks. Outperforms PCA at all compression ratios (e.g., 97.6% vs 95.1% binary accuracy at 5% bottleneck).

2. **LSTM grade classification** — 3-layer LSTM with 16-dim embeddings, 128-dim hidden state, class-weighted cross-entropy loss.

3. **Benchmark harness** — 5-fold cross-validation framework with stratified splits, MLflow tracking, and standardized evaluation metrics.

4. **Baseline submissions** — LSTM, perceptron, and tree baselines in `submissions/`.

### Current Leaderboard

| Model | Exact (%) | Within ±1 (%) | Within ±2 (%) |
|-------|-----------|---------------|---------------|
| Random Forest | 49.55 | 69.65 | 82.88 |
| Perceptron (MLP) | 45.26 | 45.26 | 70.89 |
| LSTM Baseline | 35.46 | 35.46 | 66.31 |
| 2DCNN | 27.23 | 27.23 | 55.62 |
| Ridge Regression | 20.39 | 55.60 | 80.60 |

> **Note:** All results are from the 5-fold retrain-per-fold CV benchmark (10K stratified subsample of 2016 data). The previous 96.43% RF result was caused by data leakage (multiple variants of the same route in both train and test folds) and has been corrected.

---

## Next Steps: Reproducing and Extending Results

### Phase 1: Reproduce Duh & Chang (2021)

**Goal:** Reproduce the ~46.7% exact / ~84.7% within-±1 results from the GradeNet paper.

- [x] **Train the LSTM baseline to completion** — Done. 5-fold retrain-per-fold CV: 35.46% exact, 66.31% within-±2. Config: 16-dim embeddings, 128-dim hidden, 3 layers, Adam lr=0.001, early stopping (patience=10).
- [ ] **Verify BetaMove preprocessing equivalence** — Compare the repo's `preprocessing.py` tokenization against the BetaMove pipeline described in the paper. The repo uses sorted hold descriptions with special tokens (`START_END`, `MIDDLE_END`, etc.) which appears to be a variant of the same concept.
- [ ] **Evaluate with the same metrics** — Report exact match and within-±1 grade accuracy on a held-out test set with the same grade distribution as the paper.
- [ ] **Compare against reported human baseline** — The paper estimates human-level performance at ~45% exact / ~85% within-±1. Verify whether the trained model approaches this.

**Expected effort:** ~1-2 days (primarily compute time for training).

### Phase 2: Reproduce Petashvili & Rodda (2023)

**Goal:** Reproduce the 2DCNN results (0.86 MAE / 1.12 RMSE) and cross-edition generalization experiments.

- [x] **Implement 2DCNN architecture** — 4-layer CNN with 3×3 kernels, trained with Adam optimizer and MSE loss, early stopping (patience=20). Input: one-hot encoded 11×18 binary hold matrix. Implemented at `submissions/2dcnn-baseline/`. Smoke test: 40% exact at 2 epochs (paper reports 42%).
- [x] **Implement Ridge regression baseline** — Simple linear baseline on 164-dim binary hold vectors. Implemented at `submissions/ridge-baseline/`. 5-fold CV on 10K samples: 20.39% exact, 80.60% within-±2.
- [ ] **Train and evaluate on 2016 dataset** — Target: 0.86 MAE / 1.12 RMSE / 42% exact / 84% within-±1.
- [ ] **Cross-edition generalization experiment** — Train on 2016+2017, test on 2019 (and all permutations). Target: LSTM should generalize best with ~2.35 MAE.
- [ ] **Vision-based baseline** — Generate route images and train ResNet50 / MaxViT backbones. Target: ~1.84 MAE (current SOTA for vision approach, but well below tabular methods).

**Expected effort:** ~2-3 days (architecture implementation + multi-dataset training).

### Phase 3: Extend and Improve

**Goal:** Beat existing results with new approaches.

- [ ] **Transformer-based grade prediction** — Replace LSTM with a transformer encoder that attends over hold positions. The Moonboard grid has natural spatial structure that self-attention may capture better than sequential models.
- [ ] **Graph neural network approach** — Model the Moonboard as a graph where holds are nodes and edges represent physical adjacency/distance. GNNs could learn spatial relationships more naturally than CNNs or RNNs.
- [ ] **Multi-task learning** — Jointly predict grade and route quality/style. Duh & Chang suggested this as future work for climbing style classification.
- [ ] **Cross-edition generalization improvements** — The biggest open problem. Approaches could include:
  - Edition-conditional training (provide board configuration as input)
  - Domain adaptation techniques
  - Hold-level feature learning (learning what makes a hold "hard" independent of position)
- [ ] **Vision-based grade prediction (improved)** — The ResNet50 baseline (1.84 MAE) has significant room for improvement. Try:
    - Pre-trained backbones (ImageNet → fine-tune)
    - Multi-channel images encoding hold type, orientation, and difficulty
    - Wall geometry encoding for non-flat surfaces
- [ ] **Probabilistic grade prediction** — Following Drummond & Popinga's Bayesian framework, model grade as a continuous latent variable rather than discrete classes. This could better capture the inherently subjective nature of climbing grades.

### Phase 4: Infrastructure and Reproducibility

- [ ] **Standardized dataset splits** — Create canonical train/test splits for each Moonboard edition to enable fair comparison across papers.
- [ ] **Unified evaluation protocol** — Agree on MAE, RMSE, exact match, and within-±k metrics as standard benchmarks.
- [ ] **Dataset versioning** — Track which Moonboard data dump each result uses (the dataset changes over time as routes are added/graded).

---

## Summary of Key Metrics Across Papers

| Paper | Model | Exact Acc. | Within ±1 | MAE | RMSE |
|-------|-------|------------|-----------|-----|------|
| Dobles (2017) | CNN | ~34% | — | — | — |
| Kempen (2018) | Various (binary) | ~60% | — | — | — |
| Duh & Chang (2021) | RNN + BetaMove | 46.7% | 84.7% | — | — |
| Petashvili & Rodda (2023) | 2DCNN | 42% | 84% | 0.86 | 1.12 |
| Petashvili & Rodda (2023) | LSTM | — | — | ~0.95 | ~1.20 |
| Petashvili & Rodda (2023) | ResNet50 (vision) | — | — | 1.84 | 2.30 |
|| **This repo** | **Random Forest** | **49.55%** | **69.65%** | — | — |
|| **This repo** | **Perceptron (MLP)** | **45.26%** | **45.26%** | — | — |
|| **This repo** | **LSTM** | **35.46%** | **35.46%** | — | — |
|| **This repo** | **2DCNN** | **27.23%** | **27.23%** | — | — |
|| **This repo** | **Ridge Regression** | **20.39%** | **55.60%** | — | — |

> **Note:** All this-repo results from 5-fold retrain-per-fold CV (10K stratified subsample, 2016 data). Ridge is from a 2-fold smoke test (2K samples). The previous 96.43% RF result was caused by data leakage and has been corrected.
