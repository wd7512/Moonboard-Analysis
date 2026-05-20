# Moonboard Analysis — Repo Improvement Plan

## Current Problems

| Problem | Examples |
|---------|----------|
| **Massive DRY violations** | `Autoencoder` class duplicated in 3 files, `ClimbingRouteDataset` in 4, `evaluate_reconstruction` in 3 |
| **No project config** | `pyproject.toml` missing, stale `egg-info/`, no pinned deps |
| **Not reproducible** | No random seeds in LSTM notebook, no dependency locking, no experiment tracking |
| **Windows-only paths** | `test.ipynb` uses `"Raw\\moonboard_problems_setup_2016.json"` — breaks on macOS/Linux |
| **Monolithic scripts** | `compare_pca.py` (519 lines) bundles model, data, training, PCA, plotting, reporting |
| **No tests, no lint config, no type hints** | Zero test coverage, no mypy/ruff/black config |
| **Generated artifacts tracked** | `.pth`, `.png`, `.csv`, `.npy` files all in git |
| **No evaluation script for LSTM** | All eval logic is inside the notebook |

---

## Phase 1: Foundation

### 1.1 Restore `pyproject.toml`

Project metadata, pinned dependencies, and tool config (ruff, mypy, pytest).

Dependencies to pin:
- numpy, matplotlib, torch, pandas, scikit-learn, seaborn, adjusttext
- mlflow (experiment tracking)
- pyyaml
- pytest (dev)

### 1.2 Create root `README.md`

Sections:
- Project overview / goal
- Installation instructions (clone, venv, pip install -e .)
- Project components: autoencoder, LSTM grade predictor, PCA comparison
- How to run each component
- Results summary (key metrics)

### 1.3 Fix cross-platform paths

`test.ipynb` line 75: `"Raw\\moonboard_problems_setup_2016.json"` → use `pathlib.Path`

### 1.4 Update `.gitignore`

Add entries for:
- `*.pth`, `*.pt` (model weights)
- `*.png`, `*.jpg` (generated figures)
- `*.csv` (generated results)
- `results/`, `outputs/` (generated output directories)
- `mlruns/` (MLflow logs)
- `__pycache__/`, `*.pyc`
- `.mypy_cache/`, `.ruff_cache/`, `.pytest_cache/`

### 1.5 Archive Legacy

Move `Legacy/` → `archive/Legacy/` to declutter the root. Update any internal paths if needed.

---

## Phase 2: DRY + SOLID Refactor

### 2.1 Create `src/moonboard_analysis/` Package

```
src/moonboard_analysis/
  __init__.py
  config.py                  # AutoencoderConfig, LSTMConfig dataclasses
  data/
    __init__.py
    loader.py                # load_2016_data(), load_2017_data()
    preprocessing.py         # preprocess_grades(), tokenize_moves(), clean_data()
    dataset.py               # ClimbingRouteDataset (single canonical version)
  models/
    __init__.py
    autoencoder.py           # Autoencoder nn.Module (single canonical version)
    lstm.py                  # ClimbingGradePredictor nn.Module
  training/
    __init__.py
    trainer.py               # generic train_autoencoder(), train_lstm()
    metrics.py               # evaluate_reconstruction(), evaluate_classification()
  utils/
    __init__.py
    reproducibility.py       # set_seeds() — numpy, torch, random, cudnn
    paths.py                 # project root finder, data dir helpers
```

### 2.2 Deduplicate

- Remove `autoencoder/autoencoder.py` and `autoencoder/compare_pca.py` monolithic scripts
- Refactor `compare_pca.py` logic into `scripts/compare_pca.py` that imports from the package
- All model definitions live in one place (`models/autoencoder.py`, `models/lstm.py`)
- All dataset classes live in one place (`data/dataset.py`)
- All evaluation functions live in one place (`training/metrics.py`)

### 2.3 Config Dataclasses

```python
@dataclass
class AutoencoderConfig:
    input_dim: int = 164
    bottleneck_dim: int = 8
    hidden_dim: int = 64
    epochs: int = 100
    batch_size: int = 64
    learning_rate: float = 0.001
    weight_decay: float = 1e-5
    seed: int = 42

@dataclass
class LSTMConfig:
    embed_dim: int = 16
    hidden_dim: int = 128
    num_layers: int = 3
    num_epochs: int = 500
    batch_size: int = 32
    learning_rate: float = 0.001
    seed: int = 42
    max_length: int | None = None  # auto-computed from data
```

---

## Phase 3: Reproducibility

### 3.1 Reproducibility Module

`utils/reproducibility.py`:
```python
def set_seeds(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
```

Called at the top of every entry-point script.

### 3.2 MLflow Experiment Tracking

Every training/evaluation script:
- `mlflow.set_experiment("autoencoder")` / `mlflow.set_experiment("lstm")`
- `mlflow.log_params(config.__dict__)` at start
- `mlflow.log_metric()` per epoch (loss, accuracy)
- `mlflow.log_artifact()` for figures, confusion matrices, saved models
- `mlflow.log_dict()` for results JSON

### 3.3 CLI Entry Points

In `pyproject.toml`:
```
[project.scripts]
moonboard-train-ae = "moonboard_analysis.scripts.train_autoencoder:main"
moonboard-compare-pca = "moonboard_analysis.scripts.compare_pca:main"
moonboard-train-lstm = "moonboard_analysis.scripts.train_lstm:main"
moonboard-evaluate-lstm = "moonboard_analysis.scripts.evaluate_lstm:main"
```

---

## Phase 4: LSTM Evaluation Pipeline

### 4.1 `scripts/evaluate_lstm.py`

Standalone evaluation script:

```
usage: moonboard-evaluate-lstm [--data PATH] [--checkpoint PATH] [--output-dir DIR]

Loads LSTM_Moonboard.pth, runs on held-out test set, reports:
  - Confusion matrix (saved as PNG + CSV)
  - Exact accuracy
  - Within-1 / within-2 / within-3 / within-4 grade accuracy
  - Per-class precision, recall, F1-score
  - Majority-class baseline comparison
  - All metrics logged to MLflow
```

### 4.2 `scripts/train_lstm.py`

Proper training script that completes the full 500 epochs (or configurable) with:
- Reproducible seed
- MLflow logging
- Configurable hyperparameters via dataclass
- Best-model checkpointing
- Training/validation loss curves

---

## Phase 5: Quality of Life

### 5.1 Type Hints

Add complete type hints to all new module functions and method signatures.

### 5.2 Basic Tests

```
tests/
  conftest.py               # pytest fixtures (sample data, models)
  test_models.py            # forward pass shape checks
  test_metrics.py           # evaluation metrics with known values
  test_data.py              # data loading and preprocessing
```

### 5.3 Makefile

```makefile
.PHONY: install lint typecheck test train-ae train-lstm evaluate-lstm

install:
    pip install -e ".[dev]"

typecheck:
    mypy src/

test:
    pytest tests/

train-ae:
    moonboard-train-ae

evaluate-lstm:
    moonboard-evaluate-lstm
```

---

## Out of Scope (for now)

- Pre-commit hooks (ruff, mypy, isort)
- Python logging module (stick with print for simplicity)
- GitHub Actions CI
- Docker dev environment
- DVC data versioning
- Hydra/OmegaConf (using simple dataclasses instead)
