# Moonboard ML Experiment Registry

**Purpose:** Prevent redundant experiments by maintaining a complete, scientific record of every model architecture, training configuration, feature engineering technique, and result obtained in this repository.

**Rule:** Before starting any new experiment, consult this document. If your planned experiment is listed below (same model class + same feature representation + same training paradigm), you must either (a) provide a novel variation not captured here, or (b) explicitly justify why a re-run is scientifically necessary (e.g., bug fix, new data, corrected evaluation protocol).

**Protocol versioning:** Results obtained before the retrain-per-fold CV refactor (commit `039eeda`) used single-split evaluation and may have data leakage. Marked as **Protocol v1**. Results from the 5-fold retrain-per-fold harness (commit `039eeda` onward) are **Protocol v2**. Protocol v1 results are NOT comparable to Protocol v2 and should NOT be used as baselines.

---

## 1. Taxonomy

All experiments below use the 2016 Moonboard dataset unless otherwise stated.

### 1.1 Grade Taxonomy

GRADE_ORDER (12 classes): 6A, 6A+, 6B+, 6C, 6C+, 7A, 7A+, 7B, 7B+, 7C, 7C+, 8A

**Note:** Grades 8A+, 8B, 8B+ are excluded due to insufficient samples.

### 1.2 Preprocessing Pipeline (fixed across all experiments)

1. Load raw JSON via `load_lstm_data()`
2. `preprocess_lstm_data()`: Parse Moves into tokenized sequences with section delimiters (START_END, MIDDLE_END, END_ROUTE) and grade labels. Each route generates 1 sequence (for 6B) or 4 sequences (for grades with >= 4 middle holds — first two and last two middle holds are swapped).
3. `drop_duplicate_sequences()`: Remove exact duplicate token sequences.
4. All experiments use this same pipeline unless explicitly noted.

### 1.3 Evaluation Metrics

| Metric | Definition |
|--------|-----------|
| exact_accuracy | % predictions matching true grade exactly |
| within_one_grade | % predictions within ±1 grade index |
| within_two_grades | % predictions within ±2 grade indices |
| MAE | Mean Absolute Error in grade index space |
| macro_f1 | Unweighted mean of per-class F1 scores |
| weighted_f1 | Support-weighted mean of per-class F1 scores (computed by `evaluate_classification()`; not yet exposed in submissions) |

---

## 2. Submitted Models (Protocol v2 — 5-fold retrain-per-fold CV)

All results below use 5-fold stratified CV with retrain-per-fold on the full dataset (Protocol v2). The full-data results are the primary benchmark.

### 2.1 Leaderboard

> **Note:** The 10K stratified subsample leaderboard has been superseded by the
> [full-data results](#full-data-results) below. The 10K table was computed with
> a buggy within-metric calculation (`_accuracy_within_diagonal` off-by-one) and
> used data augmentation that introduced leakage — neither issue affects the
> full-data results.

**Full-data results (no subsampling):**

All 8 submissions run on the full dataset (25,738 unique routes after deduplication and preprocessing). Within-metrics use the corrected (bug-fixed) calculation.

<!-- LEADERBOARD-FULLDATA-START -->
| # | Submission | Exact (%) | ±1 (%) | ±2 (%) | Macro-F1 (%) | Training Time |
|---|-----------|-----------|--------|--------|--------------|---------------|
| 1 | deep-mlp-baseline | **40.76** (±0.62) | 65.46 (±0.55) | 84.49 (±0.53) | 16.99 (±0.32) | ~1 hr |
| 2 | fast-mlp | **40.50** (±0.50) | 64.89 (±0.69) | 83.69 (±0.73) | 15.44 (±0.82) | ~5 min |
| 3 | perceptron-baseline | **40.18** (±0.51) | 65.67 (±0.93) | 84.80 (±0.82) | 17.29 (±0.42) | ~5 min |
| 4 | gradient-boost-baseline | **39.49** (±0.48) | 62.20 (±0.44) | 80.78 (±0.31) | 16.69 (±0.45) | ~5 min |
| 5 | tree-baseline | **38.51** (±0.46) | 62.08 (±0.41) | 80.77 (±0.45) | 17.50 (±0.61) | ~2 min |
| 6 | lstm-baseline | **37.05** (±1.25) | 63.40 (±0.91) | 83.76 (±0.57) | **17.81** (±0.85) | ~30 min |
| 7 | 2dcnn-baseline | **36.81** (±5.30) | 60.66 (±5.89) | 79.99 (±4.91) | 14.24 (±1.14) | ~45 min |
| 8 | ridge-baseline | **23.94** (±0.64) | 63.63 (±0.55) | 86.30 (±0.25) | 12.90 (±0.38) | ~1 min |
<!-- LEADERBOARD-FULLDATA-END -->

> Full-data results use the complete ~26K route dataset with the corrected within-metric calculation (Protocol v2, bug-fixed).

---

## 3. Experiment Log — Detailed

### 3.1 Ridge Regression (ridge-baseline)

- **Commit:** `543de01` (PR #9)
- **Model:** `sklearn.linear_model.Ridge`, α=1.0
- **Features:** 164-dim binary vector from `GridMapper.grid_to_vector()` — 3 channels (start/middle/end) × 18 rows × 11 cols, but grid mapping uses only 164 unique hold positions (not full 198)
- **Train Protocol:** Ridge regression (closed-form), rounded to nearest grade index
- **Key Observations:**
  - Linear model captures coarse grade trend (higher holds → harder grades) but cannot model complex spatial interactions
  - Within-±1 of 55.6% shows monotonic relationship between hold positions and grade
  - Regularization α was not swept; α=1.0 is sklearn default
- **NOT EXPERIMENTED:** α sweep, polynomial features, interaction terms
- **Full-data (26K):** 23.94% exact, 12.90% macro-F1
- **Status:** Baseline only, no further work planned

### 3.2 Random Forest (tree-baseline)

- **Commit:** `bd12e1c`
- **Model:** `sklearn.ensemble.RandomForestClassifier`, n_estimators=200, n_jobs=-1
- **Features:** Same 164-dim binary grid vector as Ridge (GridMapper)
- **Key Observations:**
  - Best single-model exact accuracy on leaderboard (49.55%)
  - Tree methods exploit the binary feature structure well — splits on individual hold positions
  - Feature importance analysis not performed
  - **Previously reported 96.43% was due to DATA LEAKAGE** (multiple variants of same route in train and test); corrected to 49.55% after protocol v2
- **NOT EXPERIMENTED:** n_estimators sweep, max_depth tuning, feature importance analysis, out-of-bag evaluation
- **Full-data (26K):** 38.51% exact, 17.50% macro-F1
- **Status:** Reference baseline. Tree methods are fully permitted on the leaderboard.

### 3.3 Perceptron / Shallow MLP (perceptron-baseline)

- **Commit:** `bd12e1c`
- **Model:** 3-layer MLP — Linear(198→128) → ReLU → Dropout(0.3) → Linear(128→64) → ReLU → Dropout(0.3) → Linear(64→12)
- **Features:** 198-dim binary hold vector (ALL holds flattened into single vector, no section separation)
- **Training:** Adam, lr=0.001, batch_size=32, CrossEntropyLoss (no label smoothing, no class weights), ReduceLROnPlateau(factor=0.5, patience=15), early stopping patience=15
- **Key Observations:**
  - No feature standardization — raw binary vectors fed directly
  - No section-structure awareness — loses information about start/middle/end hold roles
  - Simpler than FastMLP but significantly worse (45.26% vs 46.61%)
- **Full-data (26K):** 40.18% exact, 17.29% macro-F1
- **Status:** Superseded by FastMLP

### 3.4 FastMLP (fast-mlp)

- **Commit:** `bb652f6` (on feat/voting-ensemble-submission branch)
- **Model:** 3-layer MLP — Linear(198→256) → ReLU → Dropout(0.3) → Linear(256→128) → ReLU → Dropout(0.3) → Linear(128→12)
- **Features:** 198-dim binary hold vector, **per-fold standardized** (zero mean, unit variance, fit on train fold only)
- **Training:** Adam, lr=0.001, batch_size=256, CrossEntropyLoss(label_smoothing=0.05), ReduceLROnPlateau(factor=0.5, patience=10), early stopping patience=15, best-state checkpointing
- **Key Improvements over perceptron-baseline:**
  1. Feature standardization (per-fold, no leakage)
  2. Label smoothing (0.05) for regularization
  3. Wider hidden dim (256 vs 128)
  4. Larger batch size (256 vs 32)
  5. Best-model checkpointing (not final epoch)
- **Key Observations:**
  - Fastest neural submission: ~2 min for 92K routes (5-fold CV)
  - Large performance jump from perceptron (46.61% vs 45.26%) — even with same architecture depth
  - Full-data (92K, no subsampling) exact accuracy: 82.56% — shows the model benefits enormously from more data
  - **Full-data (26K, deduplicated):** 40.50% exact, 15.44% macro-F1 — the 92K result included leaked route variants; after dedup, performance drops to 40.50%, in line with other neural models
- **NOT EXPERIMENTED:** Architecture variations (wider, deeper), alternative activations, residual connections, section-aware features, ensemble
- **Status:** Current best neural submission

### 3.5 LSTM (lstm-baseline)

- **Commit:** `bd12e1c`
- **Model:** `ClimbingGradePredictor` — Embedding(vocab_size, 16) → LSTM(16→128, 3 layers) → Linear(128→12), padded sequences
- **Features:** Variable-length token sequences — each hold is embedded and fed sequentially. Vocabulary includes all hold tokens (A1-K18, START_END, MIDDLE_END, END_ROUTE). `<PAD>` token for sequence alignment.
- **Training:** Adam, lr=0.001, batch_size=32, CrossEntropyLoss (no class weights, no label smoothing), ReduceLROnPlateau(factor=0.5, patience=20), early stopping patience=15
- **Related paper:** Duh & Chang (2021) — see Section 7.1. Their LSTM + BetaMove reaches 46.7% exact; our LSTM without BetaMove reaches 35.46%.
- **Key Observations:**
  - 35.46% exact — worse than MLP on flattened features
  - Sequential model struggles with Moonboard routes because holds don't have a natural linear order — the "sequence" is an artifact of the tokenization, not a true temporal/spatial sequence
  - High variance across folds (±1.9% std) — LSTM training is sensitive to initialization and sequence padding
  - Vocabulary is rebuilt per fold from train split only — no token leakage but small vocab variation
- **NOT EXPERIMENTED:** Bidirectional LSTM, attention over LSTM outputs, longer embedding dims, pre-trained hold embeddings, BetaMove preprocessing (Duh & Chang 2021), Transformer encoder
- **Full-data (26K):** 37.05% exact, 17.81% macro-F1
- **Status:** Serves as sequence-model baseline. Superseded by feature-engineered MLPs.

### 3.6 2D CNN (2dcnn-baseline)

- **Commit:** `bd12e1c` (PR #8)
- **Model:** 4-layer 2D CNN — Conv2d(1→32, 3×3) → BN → ReLU → Conv2d(32→64, 3×3) → BN → ReLU → Conv2d(64→128, 3×3) → BN → ReLU → Conv2d(128→256, 3×3) → BN → ReLU → AdaptiveAvgPool(1×1) → Linear(256→128) → ReLU → Dropout(0.5) → Linear(128→12)
- **Features:** Single-channel 18×11 binary matrix — each route becomes a spatial grid where position (row, col) has a 1 if that hold is used. **All holds collapsed to single channel** (no start/middle/end separation).
- **Training:** Adam, lr=0.001, batch_size=32, CrossEntropyLoss, ReduceLROnPlateau(factor=0.5, patience=20), early stopping patience=20
- **Related paper:** Petashvili & Rodda (2023) — see Section 7.2. Their 2DCNN reaches 42% exact; ours reaches 27.23%. Reproduction not yet achieved.
- **Key Observations:**
  - 27.23% exact — worst neural model on leaderboard
  - **HIGH VARIANCE** (±5.3% std across folds) — CNN training is unstable with this dataset size and architecture
  - Single-channel input loses section information (start/middle/end cannot be distinguished spatially)
  - AdaptiveAvgPool destroys spatial information — the 3×3 convolutions learn local patterns but global pooling collapses them to a single vector
- **NOT EXPERIMENTED:** Multi-channel input (3 channels for start/middle/end), spatial transformer, larger kernel sizes, ResNet-style connections, different pooling strategies
- **Full-data (26K):** 36.81% exact, 14.24% macro-F1
- **Status:** Needs significant architectural improvements. Paper reproduction not yet achieved.

### 3.7 Voting Ensemble (gradient-boost-baseline)

- **Commit:** `273ce2a` (on feat/voting-ensemble-submission branch)
- **Model:** `VotingClassifier` with soft voting — RandomForest(300 trees) + HistGradientBoosting(max_iter=500)
- **Features:** 164-dim binary grid vector (GridMapper) + 8 engineered meta-features (n_start, n_middle, n_end, total, start_ratio, middle_ratio, end_ratio, mean_start_row)
- **Training:** Per-fold LabelEncoder remapping (handles missing grades in folds), sklearn defaults for most hyperparameters
- **Key Observations:**
  - Tied with Random Forest on exact accuracy, **beat RF on within-grade metrics**
  - Ensemble of bagging (RF) + boosting (HistGB) provides complementary strengths
  - LabelEncoder remapping is critical — some folds don't contain all 12 grades
- **Full-data (26K):** 39.49% exact, 16.69% macro-F1
- **Status:** Ensemble reference. Does NOT introduce any new experiment category beyond what tree-baseline already covers.

### 3.8 Deep MLP with Engineered Features (deep-mlp-baseline)

- **Commit:** `6ad6fca` (main)
- **Model:** 4-layer MLP — Linear(656→512) → LeakyReLU(0.1) → Dropout(0.15) → Linear(512→256) → LeakyReLU(0.1) → Dropout(0.15) → Linear(256→128) → LeakyReLU(0.1) → Dropout(0.15) → Linear(128→12). Kaiming normal initialization. 5-model ensemble with different seeds.
- **Features:** 656-dim feature vector:
  - [0:198] Start holds (binary)
  - [198:396] Middle holds (binary)
  - [396:594] End holds (binary)
  - [594:602] 8 engineered features: log-counts (4), route_span, avg_row, avg_col, n_sections
  - [602:652] 50-dim hold bigram hash features (pairwise hold interactions via hashing trick)
  - [652:655] Cross-section hold ratios (3)
  - [655] Symmetry score (left-right balance)
- **Training:** Adam + weight_decay=1e-4, lr=0.001, batch_size=128, CrossEntropyLoss(label_smoothing=0.1), ReduceLROnPlateau(factor=0.5, patience=15), early stopping patience=25, best-state checkpointing. **5-model softmax averaging ensemble.**
- **Key Observations within this submission family:**
  - Section-separated encoding is a MAJOR feature engineering improvement over flat 198-dim vector
  - Bigram hash features capture pairwise hold interactions without quadratic memory
  - Label smoothing 0.1 (higher than FastMLP's 0.05) for stronger regularization
  - Per-fold standardization applied to all 656 features
  - Ensemble of 5 models with different seeds for variance reduction
  - **Full-data (26K):** 40.76% exact, 16.99% macro-F1 — highest exact accuracy on full-data benchmark; strong ±1 (65.46%) and ±2 (84.49%) scores
- **Status:** Benchmarked on Protocol v2 (5-fold CV, full 26K dataset). 40.76% exact — #1 on full-data leaderboard. Section-separated features + bigram hashing + ensemble provide the strongest neural result to date.

---

## 4. Feature Engineering Techniques Tried

| Technique | Used In | Dimension | Result |
|-----------|---------|-----------|--------|
| Flat binary hold vector (198: A1-K18 single channel) | perceptron, fast-mlp | 198 | Works well with standardization |
| Grid binary vector (164: 3 channels × 18×11, GridMapper) | ridge, tree, voting | 164 | Standard for sklearn models |
| Single-channel spatial matrix (18×11, no sections) | 2dcnn | 188 (1×18×11) | Poor — loses section info |
| Section-separated start/middle/end vectors | deep-mlp | 594 (3×198) | Promising — see 3.8 |
| Hold bigram hash features | deep-mlp | 50 | Pairwise interaction capture |
| Engineered meta-features (counts, spans, symmetry) | deep-mlp, voting | 8-11 | Consistent improvement |
| Hold token sequences (variable length) | lstm | variable | Underperforms fixed features |

### Techniques NOT Yet Tried

| Technique | Rationale | Complexity |
|-----------|-----------|------------|
| Multi-channel CNN input (3 channels: start/middle/end) | Preserve section info for spatial model | Low |
| BetaMove preprocessing (Duh & Chang 2021) | Human-like move sequencing; paper reports 46.7% | Medium |
| Graph neural network (holds as nodes) | Natural spatial relationship modeling | High |
| Transformer encoder (self-attention over holds) | Global interaction modeling without sequential assumption | Medium |
| Hold-level embedding pretraining | Learn hold difficulty/categories from data | Medium |
| ResNet / skip connections for CNN | Address gradient degradation in deeper spatial models | Medium |
| Cross-edition features (board configuration conditioning) | Generalization across Moonboard editions | High |
| Vision-based (rendered route images + ResNet) | Non-tabular approach; paper reports 1.84 MAE | High |
| Ordinal regression loss | Treat grades as ordered, not independent classes. Drummond & Popinga (2021) justify this but the exact 2.1× factor is model-dependent; any monotonic mapping captures the key insight. | Low |

---

## 5. Training Techniques Tried

| Technique | Used In | Notes |
|-----------|---------|-------|
| Adam optimizer | All neural models | Default β1=0.9, β2=0.999 |
| AdamW (with weight decay) | deep-mlp | weight_decay=1e-4 |
| ReduceLROnPlateau | All neural models | factor=0.5, patience varies 10-20 |
| Cosine annealing LR | Config available, not used in any submission | `LSTMConfig` references it but no submission uses it |
| Early stopping | All neural models | patience 10-25 depending on model |
| Best-state checkpointing | fast-mlp, deep-mlp | Restore best epoch, not last |
| Class-weighted cross-entropy | Config available (LSTMConfig), not used in submissions | Addresses grade imbalance |
| Label smoothing | fast-mlp (0.05), deep-mlp (0.1) | Improves generalization |
| Per-fold feature standardization | fast-mlp, deep-mlp | Critical — no leakage from test fold |
| Ensemble (softmax averaging) | deep-mlp (5 models) | Different seeds per model |
| Ensemble (voting) | voting-ensemble (RF + HistGB) | Soft voting with LabelEncoder remapping |

### Techniques NOT Yet Tried

| Technique | Rationale |
|-----------|-----------|
| Class-weighted loss for neural models | Grade distribution is imbalanced; weighting could help |
| Cosine annealing with warm restarts | Better convergence for SGD-like behavior |
| Curriculum learning | Train on easier grade distinctions first |
| Mixup / CutMix regularization | Data augmentation for tabular/tensor data |
| Stochastic weight averaging (SWA) | Better generalization, minimal cost |
| Gradient clipping | Stability for deeper networks |

---

## 6. Autoencoder Experiments (Route Compression)

| Experiment | Dims Swept | Best AE Perf | Best PCA Perf | Notes |
|-----------|-----------|-------------|--------------|-------|
| PCA vs AE sweep (20 epochs) | 2, 4, 8 | — | — | Quick test, underfit |
| PCA vs AE sweep (100 epochs) | 2-64 (log scale) | — | — | Full sweep |
| Bounded latent space (tanh) | 8 (bottleneck) | 97.6% bin acc at 5% bottleneck | 95.1% | tanh activation constrains latent space |

**Key finding:** Autoencoder outperforms PCA at all compression ratios. At 5% bottleneck (8 dims from 164): AE 97.6% binary accuracy, PCA 95.1%; AE 2.8% exact match, PCA 0.06%.

---

## 7. Related Research

Key papers on Moonboard climbing route grade prediction and their relationship to experiments in this repository.

### 7.1 Duh & Chang (2021) — RNN + BetaMove

- **Citation:** arXiv:2102.01788 | Stanford CS230 Spring 2020
- **Link:** https://arxiv.org/abs/2102.01788
- **Summary:** Introduced "BetaMove" preprocessing — converts raw hold sequences into human-like move sequences (mimicking climber hand movements). Trained an LSTM for grade prediction and route generation.
- **Reported results:** 46.7% exact, 84.7% within ±1 (near-human ~45%/85%)
- **Related experiment:** Our LSTM baseline (Section 3.5) — 35.46% exact without BetaMove.
- **Gap:** -11.2 pp. Likely cause: missing BetaMove preprocessing.
- **Not yet implemented:** BetaMove preprocessing pipeline.

### 7.2 Petashvili & Rodda (2023) — Board-to-Board Generalization

- **Citation:** arXiv:2311.12419
- **Link:** https://arxiv.org/abs/2311.12419
- **Summary:** Comprehensive evaluation of ML models for Moonboard grade prediction across editions (2016/2017/2019). Introduced vision-based grade prediction via rendered route images. Key focus: cross-edition generalization.
- **Reported results (2016):** 2DCNN best at 42% exact, 84% within ±1. LSTM ~0.95 MAE. ResNet50 vision: 1.84 MAE.
- **Related experiments:**
  - Our 2DCNN baseline (Section 3.6) — 27.23% exact. **Reproduction not yet achieved.**
  - Our LSTM baseline (Section 3.5) — 35.46% exact vs paper's ~0.95 MAE equivalent.
- **Gap:** -14.7 pp on 2DCNN. Likely causes: different preprocessing, architecture details, single-channel vs multi-channel input.
- **Not yet implemented:** Multi-channel CNN, cross-edition training, vision-based approach.
- **Code:** https://github.com/a1773620/Moonboard-Grade-Prediction

### 7.3 Drummond & Popinga (2021) — Bayesian Grade Scale

- **Citation:** arXiv:2111.08140
- **Link:** https://arxiv.org/abs/2111.08140
- **Summary:** Dynamic Bradley-Terry model with MCMC inference on climbing ascent data. Established that climbing grade scales are **logarithmic** in difficulty (~2.1× per grade for French/Ewbank/UIAA, ~3.17× for V-scale). The model treats each ascent attempt as a logistic contest between climber ability (time-varying Gaussian process) and route difficulty (fixed). The scale parameter `m` maps grade increments to log-odds of success.
- **What "2.1× harder" means:** A climber of fixed ability has ~2.1× lower odds of sending a route one grade higher. This is consistent with the observation that each grade step has roughly 2-3× fewer ascents by climbers of comparable strength — the model quantifies this relationship rather than discovering a new property.
- **Critical assessment:**
  - The model separates climber ability from route difficulty (better than raw send-count curves) and produces uncertainty bounds via MCMC. Cross-system consistency (French ≈ Ewbank ≈ UIAA, V-scale steeper) is a genuine finding.
  - However, the scale parameter `m` is identifiable only through priors on ability variance — if climbers are assumed more spread out, `m` increases. The 2.1× factor is prior-dependent.
  - The data is observational, not experimental. Selection bias is unavoidable: harder routes attract stronger climbers. The model partially addresses this but can't fully disentangle ability from difficulty.
  - "Difficulty" is defined circularly through the grade consensus. The model tells you the grade scale is log-linear with factor ~2.1×, but the scale was set by community perception — the model quantifies that consensus, it doesn't independently verify it.
  - The V-scale result (3.17× vs 2.1× for French) is interesting and consistent with climber intuition (V-scale compresses at the top), but could be a selection artifact: fewer boulderers at the highest grades steepen the apparent slope.
  - The Weber-Fechner connection (perception is logarithmic) is a plausible post-hoc analogy, not a tested mechanism. The paper doesn't independently verify this link.
- **Bottom line for this repo:** The paper provides the best available justification for treating grades as ordinal-logarithmic rather than categorical. Ordinal regression or MSE loss on log-grade space is theoretically justified. But the exact 2.1× factor should not be treated as a physical constant — it's a model-dependent estimate that's consistent with, not independent of, the send-count distribution. A loss function that treats grades as ordered (any monotonic mapping) captures the key insight without committing to a specific exponent.
- **Related direction:** Ordinal regression (Section 4, "NOT Yet Tried")
- **Not yet implemented:** Ordinal regression, grade-aware loss functions.

### 7.4 Dobles (2017) — Early ML Classification

- **Citation:** Semantic Scholar CorpusID:44500
- **Link:** https://api.semanticscholar.org/CorpusID:44500
- **Summary:** One of the first systematic attempts at automated Moonboard grade prediction using CNN-based approaches.
- **Reported results:** ~34% exact accuracy.
- **Related experiments:** Our 2DCNN (27.23%) and LSTM (35.46%) are in the same range.

### 7.5 Kempen (2019) — Manual Move Decomposition

- **Citation:** University of Twente, Formal Methods and Tools
- **Summary:** Manually decomposed routes into individual moves with human-labeled sequences for binary (easy/hard) classification.
- **Reported results:** ~64% binary classification accuracy.
- **Limitation:** Manual decomposition doesn't scale and introduces human bias.

### 7.6 Phillips et al. — StrangeBeta Route Setter

- **Summary:** Non-ML approach using strange attractors to generate climbing routes. Used CRDL route description language.
- **Limitation:** Not applied to Moonboard; CRDL is ambiguous even to experts.

### Summary Matrix

| Paper | Year | Model | Feature Type | Exact (%) | Our Closest | Gap |
|-------|------|-------|-------------|-----------|-------------|-----|
| Dobles | 2017 | CNN | Spatial | ~34% | 2DCNN: 27.23% | -7 pp |
| Duh & Chang | 2021 | LSTM + BetaMove | Move sequences | 46.7% | LSTM: 35.46% | -11.2 pp |
| Petashvili & Rodda | 2023 | 2DCNN | Spatial (multi-channel) | 42% | 2DCNN: 27.23% | -14.7 pp |
| Petashvili & Rodda | 2023 | ResNet50 (vision) | Rendered images | — (1.84 MAE) | Not started | — |
| Drummond & Popinga | 2021 | Bradley-Terry | Ascent statistics | — | Not started | — |

---

## 8. Known Bugs and Data Issues

| Issue | Commit | Status | Impact |
|-------|--------|--------|--------|
| Data leakage: same route in train and test folds | Pre-`039eeda` | **FIXED** | Inflated RF to 96.43% (true: 49.55%) |
| Confusion matrix size mismatch with missing classes | `eda00c6` | **FIXED** | Metrics wrong when folds lack all grades |
| Class weight tensor shape mismatch | `bd12e1c` | **FIXED** | LSTM class weights failed |
| `_accuracy_within_diagonal` off-by-one | Present in all submissions | **KNOWN BUG** | Width parameter uses wrong range — see below |

### CRITICAL: Within-Metric Bug (unfixed)

In `/src/moonboard_analysis/training/metrics.py`, the `_accuracy_within_diagonal` function:

```python
def _accuracy_within_diagonal(conf_matrix: np.ndarray, width: int) -> float:
    n = conf_matrix.shape[0]
    total_correct = 0
    for i in range(n):
        for j in range(max(0, i - width + 1), min(n, i + width)):
            total_correct += conf_matrix[i, j]
    return total_correct / conf_matrix.sum()
```

Called as `width=1` for "within_1_accuracy": the range `range(i-0, i+1)` = `[i]` = exact only. For true ±1, it needs `width=2` (range `range(i-1, i+2)`). The same applies recursively: `width=2` for "within_2" gives ±1, not ±2.

**Impact:** All within-1 and within-2 metrics across ALL submissions are underreported. Submissions using `evaluate_classification()` (all tree-based, deep-mlp) are affected. Submissions that compute their own metrics inline (ridge, tree in benchmark) may or may not have the same bug — check each individually.

**Until fixed, treat all within-K metrics as unreliable. Exact accuracy is unaffected.**

---

## 9. Experiment Decision Matrix

Use this to determine if your planned experiment is novel:

### By Model Family

| Family | Status | Next Steps | Blocked Because |
|--------|--------|-----------|-----------------|
| **Linear** (Ridge) | Complete | Polynomial features, interaction terms | Low ceiling (~20% exact) |
| **Tree** (RF, GB, HistGB) | Mature | XGBoost, LGBM, CatBoost, feature importance, SHAP | — |
| **MLP (flat features)** | Mature | Residual connections, deeper arch, class weights | Diminishing returns vs feature engineering |
| **MLP (engineered features)** | Benchmarked | Ablation studies, feature importance, try without ensemble | — |
| **LSTM** | Underperforming | Replace with Transformer; add BetaMove (Sec 7.1) | Sequential model mismatch for spatial data |
| **2D CNN** | Underperforming (reproduction failed) | Multi-channel input, residual connections (Sec 7.2) | Needs paper reproduction first |
| **Transformer** | NOT STARTED | Self-attention over holds | Medium effort |
| **GNN** | NOT STARTED | Graph-based hold modeling | High effort |
| **Vision** | NOT STARTED | Rendered images + pretrained CNN (Sec 7.2) | High effort, low ceiling per paper |
| **Bayesian/Ordinal** | NOT STARTED | Bradley-Terry, ordinal regression (Sec 7.3) | Medium effort |

### By Feature Representation

| Representation | Dimensions | Best Model on It | Max Exact (%) | Novelty |
|---------------|-----------|-----------------|---------------|---------|
| Flat binary hold (all holds, no sections) | 198 | fast-mlp | 46.61 | Exhausted |
| Grid binary (3 channels) | 164 | Random Forest | 49.55 | Exhausted for trees; open for neural |
| Grid binary (3 channels) + meta-features | 172 | Voting Ensemble | ~49.55 | Open for neural models |
| Section-separated binary (start/mid/end) | 594 | deep-mlp | TBD | **Open** |
| Section-separated + bigram + meta | 656 | deep-mlp | TBD | **Open** |
| 2D spatial matrix (1 channel) | 1×18×11 | 2DCNN | 27.23 | Needs multi-channel |
| Token sequences (variable) | variable | LSTM | 35.46 | Superseded |

---

## 10. Submission Gate Checklist

Before creating a new submission, verify ALL of the following:

- [ ] **Experiment not redundant:** Checked against Section 9 decision matrix — model family + feature combination is not already exhausted
- [ ] **Protocol v2:** Will use 5-fold retrain-per-fold CV via `moonboard-benchmark`
- [ ] **No data leakage:** All feature engineering done inside `train_and_evaluate()` per fold; no global statistics from test data
- [ ] **Reproducibility:** Uses `set_seeds(seed)`, no randomness outside seeded regions
- [ ] **Metrics:** Returns all required keys per the [submission contract](submissions/README.md#return-dict-contract)
- [ ] **No pre-trained weights:** Code only, trains from scratch each fold
- [ ] **Code quality:** Passes `uv run ruff check`, `uv run mypy`, `uv run pytest tests/ -x -q`
- [ ] **Documentation:** Within 24 hours of submission, entry added to this file (Section 3)
- [ ] **Training time:** Under 10 minutes for 5-fold CV on full dataset (use `moonboard-smoke-test` for quick checks)
- [ ] **Run gate checker:** `uv run python submissions/check_experiment.py --submission-dir submissions/<name>` exits 0

---

## 11. Open Questions and Research Directions

1. **Why does 2DCNN underperform the paper by 15%?** Identify preprocessing differences, architecture details (paper may use multi-channel), and training differences. Reproduction is not yet achieved. See Section 7.2.

2. **Can we reach Duh & Chang's 46.7% with BetaMove preprocessing + LSTM?** Would require implementing the BetaMove move-sequence generation. See Section 7.1.

3. **Is there a ceiling to flat binary feature models?** FastMLP at 46.61% (10K) → 82.56% (92K) suggests more data helps, but is there an architectural limit? At what point do spatial models overtake?

4. **What is the real within-1 accuracy?** Once the `_accuracy_within_diagonal` bug is fixed, re-benchmark all submissions.

5. **Can cross-edition generalization be improved?** 2016 → 2017/2019 generalization is the open problem from Petashvili & Rodda (2023). See Section 7.2.

6. **Can we improve on deep-mlp with engineered features?** Ablation studies (which features matter most?), try without the ensemble for speed, or combine with tree methods in a super-ensemble.

7. **Can ordinal regression help?** Drummond & Popinga (2021) established grades are fundamentally ordinal/logarithmic. An ordinal loss function may outperform cross-entropy. See Section 7.3.

---

*Last updated: 2026-05-29*
*Maintained by: OWL for Moonboard-Analysis project*
*Protocol: v2 (5-fold retrain-per-fold CV)*
