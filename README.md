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

Compresses 164-dimensional binary hold vectors into a low-dimensional bottleneck and reconstructs them. Compared against PCA across 7 compression ratios.

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

Predicts route grade (6B+ through 8A) from the ordered sequence of holds using a 3-layer LSTM with class-weighted loss.

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

## Reproducibility

- All random seeds are set explicitly (`utils/reproducibility.py`)
- Dependencies are pinned with version ranges in `pyproject.toml`
- MLflow tracks every run's hyperparameters and metrics
- Saved model weights: `Autoencoder_Moonboard.pth`, `LSTM_Moonboard.pth`
