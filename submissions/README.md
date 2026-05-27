# Submissions

This directory holds self-contained model submissions for the Moonboard Analysis benchmark leaderboard.

## Format

Each submission lives in its own folder under `submissions/<model-name>/` and must follow this structure:

```
submissions/<model-name>/
├── main.py            # Required entry point (see template below)
├── requirements.txt   # Optional extra dependencies (beyond project deps)
├── config.json        # Optional hyperparameters or metadata
└── ...                # Additional helper modules, weights, etc.
```

## `main.py` Contract

Every `main.py` must expose a `main()` function and be runnable as a standalone script via:

```bash
uv run python submissions/<model-name>/main.py [ARGS]
```

### Required CLI Arguments

| Argument | Description | Example |
|----------|-------------|---------|
| `--data-path` | Path to raw Moonboard JSON data | `Raw/moonboard_problems_setup_2016.json` |
| `--output-dir` | Directory to save model weights and results | `results/` |
| `--seed` | Random seed for reproducibility | `42` |

### Expected Behavior

1. **Load** data from `--data-path` (defaults to `Raw/moonboard_problems_setup_2016.json`).
2. **Preprocess** using the project's preprocessing utilities (`moonboard_analysis.data.preprocessing`).
3. **Train** the model on the training split (80/20 stratified split, preserving grade distribution).
4. **Save** the trained model weights to `--output-dir`.
5. **Print** evaluation metrics to stdout, including:
   - Exact accuracy
   - Within ±1 grade accuracy
   - Within ±2 grade accuracy

### Template

Replace `<model-name>` with your submission name (use lowercase-with-hyphens):

```python
"""Submissions/<model-name> — brief description of your approach."""

import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

# Import from the project package — do NOT duplicate src/ code
from moonboard_analysis.config import GRADE_ORDER
from moonboard_analysis.data.dataset import LSTMSequenceDataset
from moonboard_analysis.data.loader import load_lstm_data
from moonboard_analysis.data.preprocessing import (
    drop_duplicate_sequences,
    preprocess_lstm_data,
)
from moonboard_analysis.models.lstm import ClimbingGradePredictor
from moonboard_analysis.training.metrics import evaluate_classification
from moonboard_analysis.training.trainer import evaluate_lstm, train_lstm_epoch
from moonboard_analysis.utils.device import get_device
from moonboard_analysis.utils.reproducibility import set_seeds


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train <model-name> on Moonboard data"
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default="Raw/moonboard_problems_setup_2016.json",
    )
    parser.add_argument("--output-dir", type=str, default=".")
    parser.add_argument("--seed", type=int, default=42)
    # Add model-specific arguments below
    return parser.parse_args()


def build_vocab(sequences):
    tokens = set()
    for seq in sequences:
        tokens.update(seq)
    vocab = {token: i + 1 for i, token in enumerate(sorted(tokens))}
    vocab["<PAD>"] = 0
    return vocab


def main() -> None:
    args = parse_args()
    set_seeds(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # -- Load & preprocess --
    df = load_lstm_data(args.data_path)
    sequences = preprocess_lstm_data(df)
    sequences = drop_duplicate_sequences(sequences)

    route_sequences, route_grades = [], []
    for seq in sequences:
        grade = seq[-2]
        if grade in GRADE_ORDER:
            route_sequences.append(seq[:-2])
            route_grades.append(grade)

    grade_to_idx = {g: i for i, g in enumerate(GRADE_ORDER)}
    encoded_grades = [grade_to_idx[g] for g in route_grades]

    vocab = build_vocab(route_sequences)
    max_length = max(len(s) for s in route_sequences)

    train_seqs, test_seqs, train_grades, test_grades = train_test_split(
        route_sequences,
        encoded_grades,
        test_size=0.2,
        random_state=args.seed,
        stratify=encoded_grades,
    )

    train_ds = LSTMSequenceDataset(train_seqs, train_grades, vocab, max_length)
    test_ds = LSTMSequenceDataset(test_seqs, test_grades, vocab, max_length)
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=32)

    # -- Build model (replace with your architecture) --
    device = get_device()
    model = ClimbingGradePredictor(
        vocab_size=len(vocab),
        embed_dim=16,
        hidden_dim=128,
        num_layers=3,
        num_classes=len(GRADE_ORDER),
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # -- Train --
    for epoch in range(500):
        train_lstm_epoch(model, train_loader, criterion, optimizer, device)

    # -- Evaluate --
    all_preds, all_labels = [], []
    model.eval()
    with torch.no_grad():
        for seqs, grades in test_loader:
            seqs, grades = seqs.to(device), grades.to(device)
            outputs = model(seqs)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy().tolist())
            all_labels.extend(grades.cpu().numpy().tolist())

    metrics = evaluate_classification(all_labels, all_preds, len(GRADE_ORDER))

    print("=" * 50)
    print("Evaluation Results")
    print("=" * 50)
    print(f"Exact Accuracy:      {metrics['exact_accuracy']:.4f}")
    print(f"Within-1 Accuracy:   {metrics['within_1_accuracy']:.4f}")
    print(f"Within-2 Accuracy:   {metrics['within_2_accuracy']:.4f}")

    # -- Save model --
    save_path = output_dir / "model.pth"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": {
                "vocab_size": len(vocab),
                "embed_dim": 16,
                "hidden_dim": 128,
                "num_layers": 3,
                "num_classes": len(GRADE_ORDER),
                "max_length": max_length,
            },
            "vocab": vocab,
        },
        save_path,
    )
    print(f"Model saved to: {save_path}")


if __name__ == "__main__":
    main()
```

## Reference Submissions

- **`lstm-baseline/`** — the existing LSTM model ported as a submission. A
  good starting point for new submissions that want to experiment with
  architecture changes, different optimizers, or new training strategies.
