# Moonboard Analysis

Machine learning analysis of Moonboard climbing route data — route compression via autoencoders and grade classification via LSTMs.

## Project Structure

```
src/moonboard_analysis/     # Core Python package
  config.py                 # Hyperparameter dataclasses
  data/                     # Data loading, preprocessing, PyTorch datasets
  models/                   # Autoencoder, LSTM, PCA wrapper
  training/                 # Training loops, evaluation metrics
  utils/                    # Reproducibility seeding, path helpers
notebooks/                  # Jupyter notebooks for exploration
archive/Legacy/             # Previous analysis notebooks (archived)
Raw/                        # Raw Moonboard API JSON data
scripts/                    # Entry-point scripts
```

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Components

### Autoencoder Route Compression

Compresses 164-dimensional binary hold vectors into a low-dimensional
bottleneck and reconstructs them. Compared against PCA across 7 compression
ratios.

```bash
moonboard-train-ae
moonboard-compare-pca
```

Key results (autoencoder vs PCA at 5% bottleneck):
| Metric | Autoencoder | PCA |
|--------|-------------|-----|
| Binary Accuracy | 97.6% | 95.1% |
| Exact Match | 2.8% | 0.06% |

### LSTM Grade Classification

Predicts route grade (6B+ through 8A) from the ordered sequence of holds
using a 3-layer LSTM with class-weighted loss.

```bash
moonboard-train-lstm
moonboard-evaluate-lstm
```

Key results (at epoch 199/500):
| Tolerance | Accuracy |
|-----------|----------|
| Exact | 82.2% |
| Within 1 grade | 90.4% |
| Within 2 grades | 95.5% |

## Benchmark

We evaluate grade classification performance using 5-fold cross-validation
on the Moonboard dataset. Each fold uses 80% of routes for training and
20% for evaluation, with stratified sampling to preserve grade distribution.

### Metrics

We report three tolerance-based metrics:

- **Exact Match**: Predicted grade matches true grade exactly
- **Within ±1 Grade**: Predicted grade within one step of true grade
  (e.g., 7A+ for true 7A is acceptable)
- **Within ±2 Grades**: Predicted grade within two steps of true grade
  (e.g., 7B for true 7A is acceptable)

Grade hierarchy: 6B+, 6C, 6C+, 7A, 7A+, 7B, 7B+, 7C, 7C+, 8A, 8A+, ...

### Leaderboard

| Model           | Exact (%) | Within ±1 (%) | Within ±2 (%) |
|-----------------|-----------|---------------|---------------|
| LSTM Baseline   | 82.2      | 90.4          | 95.5          |

The baseline uses a 3-layer LSTM with:
- 128-dim embeddings for 164 hold types
- 256-dim hidden state
- Class-weighted cross-entropy loss to handle grade imbalance
- Trained for 500 epochs with Adam optimizer (lr=0.001)

### Usage

Run the benchmark on a new model:

```bash
moonboard-evaluate-lstm --model-path path/to/model.pth --seed 42
```

For help and more options:

```bash
moonboard-evaluate-lstm --help
```

Example output shows exact and within-tolerance accuracies:

```
==================================================
Evaluation Results
==================================================
Test Loss: 0.5234
Exact Accuracy: 0.8220
Within-1 Accuracy: 0.9040
Within-2 Accuracy: 0.9550
```

### Contributing

**Want to beat our baseline?** We welcome new model architectures,
feature engineering, or training techniques. Submit a PR with:

1. Updated model code in `src/moonboard_analysis/models/`
2. Evaluation results using the same benchmark setup
3. A brief description of your approach

We'll update the leaderboard with top-performing models!

## Reproducibility

- All random seeds are set explicitly (`utils/reproducibility.py`)
- Dependencies are pinned with version ranges in `pyproject.toml`
- MLflow tracks every run's hyperparameters and metrics
- Saved model weights: `Autoencoder_Moonboard.pth`, `LSTM_Moonboard.pth`
