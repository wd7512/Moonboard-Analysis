# Plan: CORAL-DeepMLP Ensemble — Beat Moonboard Leaderboard

## Goal
Maximize macro-F1 on the 2016 benchmark while staying within the 10-minute compute budget.

## Current Best (2016)
- Exact: 40.52% (deep-mlp-baseline, 656-dim features, 5-model CE ensemble)
- Macro-F1: 22.29% (ordinal-regression, 198-dim flat, CORAL+BCE)

## Key Insight
DeepMLP has the best features (656-dim: section-separated + bigram + meta) but uses cross-entropy loss.
Ordinal-regression has the best loss (CORAL = ordinal) but uses weak features (198-dim flat).

**The winning combo: CORAL ordinal loss + DeepMLP's 656-dim features + focal loss + ensemble.**

This is explicitly called out as the #1 open research direction in EXPERIMENTS.md Section 11 Q5.

## Architecture

### Feature Extraction (reuse from deep-mlp-baseline)
Same 656-dim feature vector:
- [0:198] start holds binary
- [198:396] middle holds binary
- [396:594] end holds binary
- [594:602] 8 engineered meta-features
- [602:652] 50-dim hold bigram hash features
- [652:655] 3 cross-section hold ratios
- [655] symmetry score
Per-fold standardization (zero mean, unit variance, fit on train only)

### Model: CORALNet
```text
Input(656) → Linear(512) → BatchNorm → LeakyReLU(0.1) → Dropout(0.15)
            → Linear(256) → BatchNorm → LeakyReLU(0.1) → Dropout(0.15)
            → Linear(128) → BatchNorm → LeakyReLU(0.1) → Dropout(0.15)
            → Linear(128→1, bias=False) + coral_bias(12)  # CORAL head
```
Kaiming normal initialization.

### CRITICAL: Bias Initialization
Initialize coral_bias from the empirical grade distribution of the training fold:
```text
For each threshold k: bias[k] = logit(P(grade > k))
where P(grade > k) = fraction of training samples with label > k
```
This gives the model a strong starting point. Without this, the model collapses to predicting a single grade.

### Loss: Focal BCE for Ordinal Thresholds
- Convert labels to ordinal: for grade k, thresholds 0..k-1 are 1, rest 0
- Focal loss: FL = -(1-p_t)^γ * BCE, γ=2.0

### Training
- AdamW, lr=0.002, weight_decay=1e-4
- Batch size: 256
- ReduceLROnPlateau(factor=0.5, patience=10)
- Early stopping patience=10
- Max epochs: 20
- Best-state checkpointing

### Ensemble
- 2-model ensemble with seeds: seed, seed+1
- Average ordinal logits (not probabilities) before converting to labels

### Prediction
- sum(sigmoid(logits) > 0.5) = predicted grade index

## Contract Requirements
- Function: `train_and_evaluate(sequences, grades, train_idx, test_idx, seed=42, **kwargs)`
- Returns: `{"exact_accuracy": float, "within_one_grade": float, "within_two_grades": float, "macro_f1": float}`
- Uses `set_seeds(seed)` from `moonboard_analysis.utils.reproducibility`
- Uses `evaluate_classification()` + `extract_required_metrics()` from `moonboard_analysis.training.metrics`
- main() with --data-path, --output-dir, --seed CLI args

## Submission Location
`submissions/coral-deepmlp-ensemble/main.py`

## Steps (use dev-workflow skill)
1. Read PLAN_CORAL_DEEPMLP.md for the full architecture spec
2. Reuse feature extraction from submissions/deep-mlp-baseline/main.py
3. Implement CORALNet with bias initialization from training data distribution
4. Implement FocalBCELoss(gamma=2.0)
5. 5-model ensemble with different seeds, average logits
6. Gate check: `uv run python submissions/check_experiment.py --submission-dir submissions/coral-deepmlp-ensemble`
7. Smoke test: `uv run moonboard-benchmark --submission-dir submissions/coral-deepmlp-ensemble --max-samples 1000 --output-json /tmp/smoke_test.json`
8. Report results

## Important Constraints
- NO tree-based methods (banned by gate checker)
- NO pre-trained weights
- NO data leakage: all preprocessing inside train_and_evaluate, per-fold standardization
- MPS-safe: no src_key_padding_mask, no Transformer
- Code must pass `uv run ruff check`
- The test_loader in `_extract_logits` should handle both (features, targets) and (features,) only tuples
