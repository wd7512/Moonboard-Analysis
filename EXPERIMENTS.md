# Moonboard ML Experiment Registry

**Purpose:** Prevent redundant experiments by maintaining a complete, scientific record of every model architecture, training configuration, feature engineering technique, and result obtained in this repository.

**Rule:** Before starting any new experiment, consult this document. If your planned experiment is listed below (same model class + same feature representation + same training paradigm), you must either (a) provide a novel variation not captured here, or (b) explicitly justify why a re-run is scientifically necessary (e.g., bug fix, new data, corrected evaluation protocol).

**Protocol versioning:** Results obtained before the retrain-per-fold CV refactor (commit `039eeda`) used single-split evaluation and may have data leakage. Marked as **Protocol v1**. Results from the 5-fold retrain-per-fold harness (commit `039eeda` onward) are **Protocol v2**. Protocol v1 results are NOT comparable to Protocol v2 and should NOT be used as baselines.

---

## 1. Taxonomy

All experiments below use the 2016 Moonboard dataset unless otherwise stated.

### 1.1 Grade Taxonomy

GRADE_ORDER (13 classes): 6A, 6A+, 6B, 6B+, 6C, 6C+, 7A, 7A+, 7B, 7B+, 7C, 7C+, 8A

**CRITICAL CORRECTION (2026-06-06):** The actual 2016 dataset contains only 10 of these 13 classes. Classes 6A, 6A+, and 6B have ZERO samples in the dataset. The actual distribution after dedup:
- 6B+: 8,369 (32.5%) — NOT 6B as previously assumed
- 6C+: 3,838 (14.9%)
- 7A: 3,564 (13.8%)
- 7A+: 2,765 (10.7%)
- 7B+: 1,541 (6.0%)
- 6C: 2,605 (10.1%)
- 7B: 1,213 (4.7%)
- 7C: 1,211 (4.7%)
- 7C+: 434 (1.7%)
- 8A: 198 (0.8%)

**This is NOT a bell curve centered on 7A-7B+.** It is heavily right-skewed with 6B+ as the mode (32.5%). The previous assumption about the distribution was WRONG and led to incorrect conclusions about the F1 ceiling.

Grades 8A+, 8B, 8B+ are excluded due to insufficient samples.

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
> data augmentation that introduced leakage — this issue does not affect the
> full-data results.

**Full-data results (no subsampling):**

All 14 submissions run on the full dataset (25,738 unique routes after deduplication and preprocessing).

<!-- LEADERBOARD-FULLDATA-START -->

## 2016 Hold Setup

| # | Submission | Exact (%) | ±1 (%) | ±2 (%) | Macro-F1 (%) | Training Time |
|---|-----------|-----------|--------|--------|--------------|---------------|
| 1 | fast-mlp | **40.26** (±0.74) | 65.34 (±0.67) | 84.38 (±0.68) | 14.71 (±0.73) |  |
| 2 | focal-loss | **39.87** (±1.53) | 63.70 (±2.70) | 82.84 (±2.18) | 13.58 (±2.11) |  |
| 3 | bottom-top-lstm | **39.86** (±0.55) | 64.09 (±0.66) | 83.35 (±1.01) | 13.73 (±0.47) |  |
| 4 | transformer-encoder | **39.76** (±1.03) | 64.74 (±0.30) | 84.56 (±0.62) | 14.53 (±0.58) |  |
| 5 | perceptron-baseline | **39.65** (±0.60) | 65.62 (±0.42) | 84.89 (±0.64) | 15.54 (±0.54) |  |
| 6 | class-balanced-loss | **39.54** (±0.57) | 63.84 (±1.12) | 82.55 (±0.88) | 15.93 (±1.04) |  |
| 7 | multichannel-2dcnn | **39.44** (±0.78) | 62.61 (±0.98) | 81.44 (±0.74) | 13.54 (±0.93) |  |
| 8 | deep-mlp-baseline | **39.39** (±0.40) | 64.48 (±0.58) | 83.88 (±0.20) | 16.78 (±0.31) |  |
| 9 | gradient-boost-baseline | **38.69** (±0.54) | 61.16 (±0.82) | 80.09 (±0.37) | 15.24 (±0.61) |  |
| 10 | tree-baseline | **38.51** (±0.46) | 62.08 (±0.41) | 80.77 (±0.45) | 16.15 (±0.56) |  |
| 11 | lstm-baseline | **37.99** (±1.67) | 63.38 (±0.52) | 83.46 (±0.44) | 15.35 (±1.33) |  |
| 12 | 2dcnn-baseline | **37.15** (±1.99) | 61.13 (±0.10) | 80.86 (±1.81) | 12.90 (±0.38) |  |
| 13 | ordinal-regression | **36.81** (±0.99) | 68.46 (±0.83) | 87.54 (±0.33) | **17.15** (±0.54) |  |
| 14 | ridge-baseline | **23.94** (±0.64) | 63.63 (±0.55) | 86.01 (±0.24) | 11.91 (±0.35) |  |

## Masters 2017 Hold Setup

| # | Submission | Exact (%) | ±1 (%) | ±2 (%) | Macro-F1 (%) | Training Time |
|---|-----------|-----------|--------|--------|--------------|---------------|
| 1 | deep-mlp-baseline | **32.14** (±0.18) | 56.29 (±0.02) | 76.67 (±0.40) | 13.94 (±0.18) |  |
| 2 | focal-loss | **31.96** (±0.20) | 56.56 (±0.37) | 77.13 (±0.19) | 14.94 (±0.02) |  |
| 3 | fast-mlp | **31.81** (±0.03) | 55.80 (±0.13) | 76.06 (±0.07) | 13.49 (±0.92) |  |
| 4 | bottom-top-lstm | **31.65** (±0.32) | 56.33 (±0.07) | 76.86 (±0.14) | 14.18 (±0.26) |  |
| 5 | perceptron-baseline | **31.57** (±0.21) | 56.71 (±0.27) | 78.17 (±0.07) | 16.66 (±0.07) |  |
| 6 | tree-baseline | **31.41** (±0.33) | 54.98 (±0.20) | 74.88 (±0.67) | 17.11 (±0.66) |  |
| 7 | transformer-encoder | **31.34** (±0.36) | 56.29 (±0.10) | 76.91 (±0.96) | 14.06 (±0.55) |  |
| 8 | multichannel-2dcnn | **30.77** (±0.41) | 54.22 (±0.11) | 74.97 (±0.77) | 13.05 (±0.01) |  |
| 9 | gradient-boost-baseline | **30.46** (±0.02) | 54.80 (±0.24) | 75.58 (±0.63) | **17.12** (±0.19) |  |
| 10 | lstm-baseline | **29.61** (±0.19) | 54.80 (±0.21) | 76.67 (±1.52) | 16.29 (±0.32) |  |
| 11 | class-balanced-loss | **28.94** (±0.13) | 52.92 (±0.97) | 71.98 (±1.06) | 15.86 (±0.61) |  |
| 12 | 2dcnn-baseline | **28.20** (±2.52) | 49.73 (±3.84) | 70.90 (±4.24) | 12.38 (±0.00) |  |
| 13 | ordinal-regression | **22.06** (±0.27) | 56.96 (±0.12) | 78.79 (±0.37) | 13.93 (±0.26) |  |
| 14 | ridge-baseline | **19.95** (±0.63) | 56.17 (±0.61) | 79.37 (±0.85) | 12.11 (±0.31) |  |

<!-- LEADERBOARD-FULLDATA-END -->

> Full-data results use the complete ~26K route dataset (Protocol v2).

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

### 3.9 Multi-Channel 2D CNN (multichannel-2dcnn)

- **Commit:** `feat/multichannel-2dcnn` (unmerged as of this writing)
- **Model:** Compact 2D CNN with 3 input channels (start/middle/end) — Conv2d(3→16, 3×3) → BN → ReLU → Conv2d(16→16, 3×3) → BN → ReLU → MaxPool(2) → Conv2d(16→32, 3×3) → BN → ReLU → MaxPool(2) → AdaptiveAvgPool(1) → Linear(32→12)
- **Features:** 3-channel 18×11 binary spatial matrices — each channel corresponds to one hold section (start, middle, end). Section boundaries identified via START_END, MIDDLE_END, END_ROUTE tokens in the sequence.
- **Training:** Adam, lr=0.001, batch_size=64, CrossEntropyLoss (no label smoothing), ReduceLROnPlateau(factor=0.5, patience=10), early stopping patience=10, best-state checkpointing
- **Key Observations:**
  - 39.31% exact — significant improvement over single-channel 2DCNN (36.81%, +2.5pp) and variant-stable (±0.70% vs ±5.30%)
  - Multi-channel input alone accounts for the gain — section awareness helps the spatial model
  - Compact architecture (3→16→16→32, ~13K params) is dramatically smaller than the 4-layer single-channel model (1→32→64→128→256, ~800K params) yet outperforms it
  - Still below paper target of 42% (Petashvili & Rodda 2023) — gap narrowed from -5.2pp to -2.7pp
  - Within-1 (62.41%) and within-2 (81.20%) are competitive with tree-based models
  - Macro-F1 (14.17%) lags — grade imbalance more pronounced in spatial features
- **NOT EXPERIMENTED:** Wider channels, residual connections, larger kernels, spatial transformer, label smoothing, class weights
- **Full-data (26K):** 39.31% exact, 62.41% within-1, 81.20% within-2, 14.17% macro-F1
- **Status:** Best CNN submission. Successfully narrows the paper reproduction gap.

### 3.10 Focal Loss (focal-loss)

- **Commit:** `feat/focal-loss` (unmerged as of this writing)
- **Model:** FastMLP (198-dim binary hold vector → 256 → 128 → 12) — identical to fast-mlp architecture
- **Loss Function:** FocalLoss(γ=2.0) instead of CrossEntropyLoss(label_smoothing=0.05)
- **Training:** Adam, lr=0.001, batch_size=256, ReduceLROnPlateau(factor=0.5, patience=10), early stopping patience=15, best-state checkpointing
- **Key Observations:**
  - 40.73% exact — beats fast-mlp (40.50%) by +0.23pp, ranks #3 overall
  - Focal Loss focuses training on hard misclassified examples, helping with grade imbalance
  - Within-1 (65.56%) and within-2 (84.44%) also improved over fast-mlp (64.89%, 83.69%)
  - Macro-F1 (16.72%) improved over fast-mlp (15.44%) — loss function directly impacts per-class performance
  - Same training speed as fast-mlp (~5 min)
  - No hyperparameter sweep on γ — γ=2.0 from literature default
- **NOT EXPERIMENTED:** γ sweep, α weighting per class, combined label smoothing + focal loss
- **Full-data (26K):** 40.73% exact, 65.56% within-1, 84.44% within-2, 16.72% macro-F1
- **Status:** Best loss-function-only improvement. Suggests class imbalance is a significant factor.

### 3.11 Class-Balanced Loss (class-balanced-loss)

- **Commit:** `feat/class-balanced-loss` (unmerged as of this writing)
- **Model:** FastMLP (198-dim binary hold vector → 256 → 128 → 12) — identical to fast-mlp architecture
- **Loss Function:** ClassBalancedLoss (Cui et al. CVPR 2019) — effective number of samples weighting: w_y = (1-β)/(1-β^{n_y})
- **Training:** Adam, lr=0.001, batch_size=256, ReduceLROnPlateau(factor=0.5, patience=10), early stopping patience=15, best-state checkpointing
- **β Sweep Results:**

| β | Exact (%) | Within-1 (%) | Within-2 (%) | Macro-F1 (%) |
|---|-----------|-------------|-------------|--------------|
| 0.9 | **39.87** (±0.75) | 65.03 (±0.89) | 83.52 (±0.48) | **18.72** (±1.04) |
| 0.99 | 35.71 (±0.70) | 65.26 (±1.14) | 83.69 (±1.13) | **19.48** (±1.10) |
| 0.999 | 34.01 (±0.79) | 64.64 (±1.34) | 82.87 (±1.12) | **19.02** (±0.94) |

- **Key Observations:**
  - Best exact accuracy at β=0.9 (39.87%), but still below fast-mlp (40.50%)
  - Best macro-F1 at β=0.99 (19.48%) — significantly higher than any submission (best prior: lstm 17.81%)
  - As β increases, class weights approach inverse frequency (n_effective → n), which heavily penalizes rare classes → exact accuracy drops, macro-F1 improves
  - The trade-off is meaningful: class-balanced loss is the best approach for balanced per-class performance
  - Base architecture is identical to fast-mlp — the only difference is the loss function
- **NOT EXPERIMENTED:** Per-class α weighting combined with β, label smoothing + class-balanced loss, class-balanced focal loss
- **Full-data (26K, β=0.9):** 39.87% exact, 65.03% within-1, 83.52% within-2, 18.72% macro-F1
- **Status:** Best macro-F1 among all submissions. Class balancing is effective for per-class metrics but slightly reduces overall accuracy.

### 3.12 Bottom-to-Top LSTM (bottom-top-lstm)

- **Commit:** `feat/bottom-top-lstm` (unmerged as of this writing)
- **Model:** Compact LSTM — Embedding(→8) → LSTM(hid=64, 1 layer) → Linear(64→12). Much smaller than lstm-baseline (emb=16, hid=128, 3 layers).
- **Features:** Token sequences re-ordered by row ascending (bottom-to-top) within each section. Ties in same row randomized.
- **Training:** Adam, lr=0.001, batch_size=64, epochs=30, ReduceLROnPlateau(factor=0.5, patience=8), early stopping patience=8
- **Key Observations:**
  - 40.17% exact — massive improvement over lstm-baseline (37.05%, +3.12pp) despite MUCH smaller model
  - Primarily driven by bottom-to-top ordering making sequences more natural for LSTM processing
  - Lower variance than baseline (±0.53% vs ±1.25%) — increased training stability
  - Only ~8 min training time (vs baseline ~30 min) thanks to compact architecture
  - Compact model with bottom-to-top ordering nearly matches perceptron (40.18%)
  - Suggests that sequence order matters significantly for LSTM performance on sparse spatial data
- **NOT EXPERIMENTED:** Bidirectional LSTM, attention mechanism, BetaMove preprocessing, deeper LSTM with bottom-to-top ordering
- **Full-data (26K):** 40.17% exact, 65.10% within-1, 83.40% within-2, 14.86% macro-F1
- **Status:** LSTM is competitive with MLPs when sequences are properly ordered. Bottom-to-top ordering is a simple but effective preprocessing.

### 3.13 Transformer Encoder (transformer-encoder)

- **Commit:** `feat/transformer-encoder` (unmerged as of this writing)
- **Model:** Compact TransformerEncoder — Embedding(→64) → PositionalEncoding → TransformerEncoder(d_model=64, nhead=2, num_layers=2, ff=128) → MeanPool → Linear(64→12)
- **Features:** Same bottom-to-top re-ordered token sequences (as Goal 3.12). Vocabulary built per fold.
- **Training:** Adam, lr=0.001, batch_size=64, epochs=30, ReduceLROnPlateau(factor=0.5, patience=8), early stopping patience=8, CrossEntropyLoss
- **Key Observations:**
  - 39.51% exact — competitive with gradient-boost (39.49%) but below bottom-top-lstm (40.17%)
  - Transformer underperforms LSTM on this data despite self-attention's theoretical advantages
  - Likely reasons: short sequences don't benefit from long-range attention; small d_model limits capacity; no pretraining
  - Mean pooling discards positional information after self-attention
  - Training stable with low variance (±0.87%)
  - Runs in ~8 min (within time budget)
- **NOT EXPERIMENTED:** Learned pooling (CLS token), bidirectional LSTM baseline for comparison, pretrained token embeddings, larger d_model, more layers
- **Full-data (26K):** 39.51% exact, 64.12% within-1, 82.59% within-2, 15.18% macro-F1
- **Status:** Transformer is functional but doesn't outperform simpler sequence models on this data.

### 3.14 Ordinal Regression — CORAL (ordinal-regression)

- **Commit:** `feat/ordinal-regression` (unmerged as of this writing)
- **Model:** FastMLP with CORAL head — 198-dim input → 256 → 128 → (K-1=11) binary logits. CORAL uses shared weight with cumulative biases for monotonic threshold prediction.
- **Loss Function:** BCEWithLogitsLoss over ordinal thresholds (11 binary outputs per sample)
- **Training:** Adam, lr=0.001, batch_size=256, ReduceLROnPlateau(factor=0.5, patience=10), early stopping patience=15, best-state checkpointing
- **Key Observations:**
  - 37.02% exact — lower than fast-mlp (40.50%), similar to LSTM baseline
  - **Best within-1 accuracy (68.16%)** and **best within-2 accuracy (87.23%)** across ALL submissions
  - Ordinal regression makes errors that are closer to the true grade — precisely what theory predicts
  - Highest within-2 of any submission by +2.73pp (previous best: ridge at 86.30%)
  - Macro-F1 (18.19%) is excellent — second only to class-balanced-loss (18.72%)
  - CORAL architecture is theoretically principled (Drummond & Popinga 2021): grades are ordered, not categorical
  - Training time ~5 min
- **NOT EXPERIMENTED:** CORAL + Focal loss (ordinal focal), CORAL with class-balanced loss, deeper CORAL head, different threshold strategies
- **Full-data (26K):** 37.02% exact, **68.16%** within-1, **87.23%** within-2, 18.19% macro-F1
- **Status:** Best within-grade metrics on leaderboard. Ordinal regression is the right loss for the grade prediction task — it just doesn't maximize exact accuracy.

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

4. **Can ordinal regression beat cross-entropy with strong features?** CORAL with 198-dim features underperforms (36.8% exact). The real test: ordinal loss on DeepMLP's 656-dim features (section-separated + bigram + meta). Drummond & Popinga (2021) showed grades are fundamentally ordinal — we haven't tested ordinal loss with a strong feature representation.

5. **Can cross-edition generalization be improved?** 2016 → 2017/2019 generalization is the open problem from Petashvili & Rodda (2023). See Section 7.2.

6. **Can we improve on deep-mlp with engineered features?** Ablation studies (which features matter most?), try without the ensemble for speed, or combine with tree methods in a super-ensemble.

7. **Can ordinal regression help?** Drummond & Popinga (2021) established grades are fundamentally ordinal/logarithmic. An ordinal loss function may outperform cross-entropy. See Section 7.3.

---

*Last updated: 2026-05-29*
*Maintained by: OWL for Moonboard-Analysis project*
*Protocol: v2 (5-fold retrain-per-fold CV)*

---

## 12. Session Results (2026-06-06): Systematic F1 Optimization

### 12.1 Critical Data Audit

**GRADE_ORDER has 13 classes but only 10 have data.** Classes 6A, 6A+, 6B have ZERO samples.

**Actual distribution (after dedup, 25,738 sequences):**
| Grade | Count | Pct |
|-------|-------|-----|
| 6B+ | 8,369 | 32.5% |
| 6C+ | 3,838 | 14.9% |
| 7A | 3,564 | 13.8% |
| 7A+ | 2,765 | 10.7% |
| 7B+ | 1,541 | 6.0% |
| 6C | 2,605 | 10.1% |
| 7B | 1,213 | 4.7% |
| 7C | 1,211 | 4.7% |
| 7C+ | 434 | 1.7% |
| 8A | 198 | 0.8% |

**NOT a bell curve centered on 7A-7B+.** Heavily right-skewed with 6B+ as mode.

### 12.2 Protocol v1 vs Protocol v2

The 19.48% macro-F1 cited in earlier sections was from **Protocol v1** (data leakage via hold-swap augmentation across CV folds). Not reproducible on current benchmark harness (Protocol v2). Real best before this session: **17.15%** (ordinal regression).

### 12.3 Results Summary

| Approach | F1 | Delta | Notes |
|----------|-----|-------|-------|
| Single ordinal (flat 198-dim) | 17.15% | baseline | Leaderboard |
| Dual-ordinal ensemble (logit avg) | 17.49% | +0.34pp | Best BCE |
| **Focal ordinal ensemble (γ=2.0)** | **17.77%** | **+0.62pp** | **New best** |
| Hierarchical (3 groups) | 16.33% | -0.82pp | Error propagation |
| Gradient boosting (balanced) | 16.50% | -0.65pp | Worse within-2 |
| Targeted rare augmentation | 17.43% | +0.28pp | No improvement |
| Class-balanced ordinal | 17.08% | -0.07pp | No improvement |

### 12.4 Key Findings

1. **Focal loss helps** — γ=2.0 on ordinal thresholds gives +0.62pp over single ordinal
2. **Logit averaging >> probability averaging** for ordinal ensembles
3. **Architectural diversity > seed diversity** for ensemble diversity
4. **Augmentation doesn't help** — hold-swap variants are too similar to originals
5. **Hierarchical classification hurts** — group-level errors propagate

### 12.5 Current Best

`submissions/coral-engineered/` — Focal ordinal ensemble (γ=2.0), 17.77% F1 on 2016.
Triple ensemble (focal + BCE + class-balanced CE) benchmark is currently running.

*Last updated: 2026-06-06*
