# Changelog

## [Unreleased]

### Added
- `submissions/` directory with model submission format and LSTM baseline reference (`submissions/lstm-baseline/main.py`)
- `CONTRIBUTING.md` -- contribution guidelines
- `CHANGELOG.md`
- Class-weighted cross-entropy loss in LSTM training (`train_lstm.py`) to handle grade imbalance
- Stratified train/test split in LSTM training to preserve grade distribution
- Stratified train/test split in LSTM evaluation (`evaluate_lstm.py`)

### Changed
- `evaluate_lstm.py`: `max_length` now loaded from checkpoint config instead of hardcoded 50
- `evaluate_lstm.py`: vocab from checkpoint is always used when available (prevents token-to-index mismatches)
- `README.md`: Fixed embedding dimension docs (16-dim, not 128-dim) to match actual config

### Fixed
- LSTM training now uses `stratify=encoded_grades` in train/test split (was unstratified)
- LSTM training now uses class-weighted loss matching the documented approach

## [0.1.0] - 2026-05-22

### Added
- Initial release
- Autoencoder route compression with PCA comparison
- LSTM grade classification with 5-fold CV benchmark harness
- CLI tools: `moonboard-train-ae`, `moonboard-compare-pca`, `moonboard-train-lstm`, `moonboard-evaluate-lstm`, `moonboard-benchmark`, `moonboard-viz-ae`
- MLflow experiment tracking
- GitHub Actions CI (lint, typecheck, test)
- 99 unit tests with pytest
