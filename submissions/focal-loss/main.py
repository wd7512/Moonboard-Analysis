"""submissions/focal-loss — Fast MLP with Focal Loss for Moonboard grade prediction.

Based on fast-mlp. Swaps CrossEntropyLoss for FocalLoss(γ=2.0) to address
grade class imbalance. Everything else identical to fast-mlp:
  - 198-dim binary hold vector with per-fold standardization
  - 2-layer MLP (256→128→12)
  - Adam with ReduceLROnPlateau scheduling
  - Best-model checkpointing

Usage:
    uv run python submissions/focal-loss/main.py --help
    uv run python submissions/focal-loss/main.py \
        --data-path Raw/moonboard_problems_setup_2016.json
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from moonboard_analysis.config import GRADE_ORDER
from moonboard_analysis.data.loader import load_lstm_data
from moonboard_analysis.data.preprocessing import (
    drop_duplicate_sequences,
    preprocess_lstm_data,
)
from moonboard_analysis.training.metrics import evaluate_classification, extract_required_metrics
from moonboard_analysis.utils.device import get_device
from moonboard_analysis.utils.reproducibility import set_seeds

NUM_COLS = 11
NUM_ROWS = 18
HOLD_VECTOR_DIM = NUM_COLS * NUM_ROWS  # 198
NUM_CLASSES = len(GRADE_ORDER)  # 12
GRADE_LABELS = frozenset(GRADE_ORDER)


def _hold_to_index(hold_name: str) -> int:
    if len(hold_name) < 2:
        return -1
    col_char = hold_name[0]
    if col_char < "A" or col_char > "K":
        return -1
    row_part = hold_name[1:]
    if not row_part.isdigit():
        return -1
    row = int(row_part)
    if row < 1 or row > 18:
        return -1
    col = ord(col_char) - ord("A")
    return (row - 1) * NUM_COLS + col


def _sequences_to_vectors(sequences: list[list[str]]) -> np.ndarray:
    """Flatten each route to a 198-dim binary hold vector.

    Extracts only valid hold tokens (skipping section delimiters and grade labels).
    This matches the Perceptron baseline's feature representation.
    """
    vectors = np.zeros((len(sequences), HOLD_VECTOR_DIM), dtype=np.float32)
    skip_tokens = GRADE_LABELS | {"GRADE_END", "START_END", "MIDDLE_END", "END_ROUTE"}
    for i, seq in enumerate(sequences):
        for token in seq:
            if token in skip_tokens:
                continue
            idx = _hold_to_index(token)
            if 0 <= idx < HOLD_VECTOR_DIM:
                vectors[i, idx] = 1.0
    return vectors


class FastMLP(nn.Module):
    """2-layer MLP: input -> 256 -> 128 -> num_classes."""

    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class FocalLoss(nn.Module):
    """Multi-class Focal Loss.

    FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)

    Where p_t is the softmax probability of the true class.
    Default γ=2.0 focuses training on hard misclassified examples.
    """

    def __init__(
        self, gamma: float = 2.0, alpha: torch.Tensor | None = None, reduction: str = "mean"
    ):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(inputs, targets, reduction="none")
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        if self.alpha is not None:
            alpha_t = self.alpha.to(inputs.device).gather(0, targets)
            focal_loss = alpha_t * focal_loss
        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        return focal_loss


# Global device — set in main()
device = torch.device("cpu")


def parse_args():
    parser = argparse.ArgumentParser(description="Fast MLP for Moonboard grade classification")
    parser.add_argument("--data-path", type=str, default="Raw/moonboard_problems_setup_2016.json")
    parser.add_argument("--output-dir", type=str, default=".")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--dropout", type=float, default=0.3)
    return parser.parse_args()


def train_and_evaluate(
    sequences: list[list[str]],
    grades: list[int],
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    seed: int = 42,
    hidden_dim: int = 256,
    epochs: int = 100,
    batch_size: int = 256,
    learning_rate: float = 0.001,
    dropout: float = 0.3,
    patience: int = 15,
) -> dict[str, float]:
    """Train a fresh FastMLP on train_idx, evaluate on test_idx.

    Key improvements over perceptron-baseline:
      - Per-fold feature standardization (train stats applied to both splits)
      - Label smoothing 0.05 for better generalization
      - Hidden dim 256 vs 128 (more capacity for same depth)
      - Larger batch size 256 vs 32 (faster training, more stable grads)
    """
    set_seeds(seed)

    train_seqs = [sequences[i] for i in train_idx]
    test_seqs = [sequences[i] for i in test_idx]
    y_train = np.array([grades[i] for i in train_idx], dtype=np.int64)
    y_test = np.array([grades[i] for i in test_idx], dtype=np.int64)

    # Feature extraction: 198-dim binary hold vectors
    X_train = _sequences_to_vectors(train_seqs)
    X_test = _sequences_to_vectors(test_seqs)

    # Per-fold standardization — fit on train, apply to both
    mu = X_train.mean(axis=0)
    sd = X_train.std(axis=0) + 1e-8
    X_train = (X_train - mu) / sd
    X_test = (X_test - mu) / sd

    train_ds = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long),
    )
    test_ds = TensorDataset(
        torch.tensor(X_test, dtype=torch.float32),
        torch.tensor(y_test, dtype=torch.long),
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size * 2)

    dev = get_device()
    model = FastMLP(HOLD_VECTOR_DIM, hidden_dim, NUM_CLASSES, dropout).to(dev)
    criterion = FocalLoss(gamma=2.0)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=10
    )

    best_loss = float("inf")
    best_state = None
    best_epoch = 0

    for epoch in range(epochs):
        # Train
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(dev), yb.to(dev)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()

        # Evaluate
        model.eval()
        test_loss = 0.0
        n_batches = 0
        with torch.no_grad():
            for xb, yb in test_loader:
                xb, yb = xb.to(dev), yb.to(dev)
                test_loss += criterion(model(xb), yb).item()
                n_batches += 1
        test_loss /= max(n_batches, 1)
        scheduler.step(test_loss)

        if test_loss < best_loss:
            best_loss = test_loss
            best_epoch = epoch
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if epoch - best_epoch >= patience:
            break

    # Load best checkpoint
    if best_state is not None:
        model.load_state_dict(best_state)
        model.to(dev)

    # Extract predictions
    all_preds, all_labels = [], []
    model.eval()
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(dev)
            preds = torch.argmax(model(xb), 1)
            all_preds.extend(preds.cpu().numpy().tolist())
            all_labels.extend(yb.numpy().tolist())

    metrics = evaluate_classification(all_labels, all_preds, NUM_CLASSES)
    return extract_required_metrics(metrics)


def main():
    global device
    args = parse_args()
    set_seeds(args.seed)
    device = get_device()

    data_path = args.data_path
    if not Path(data_path).exists():
        print(f"Error: Data file not found at '{data_path}'")
        sys.exit(1)

    print(f"Loading data from {data_path}")
    df = load_lstm_data(data_path)
    print(f"Raw data: {len(df)} routes")

    all_sequences = preprocess_lstm_data(df)
    all_sequences = drop_duplicate_sequences(all_sequences)
    print(f"After preprocessing: {len(all_sequences)} unique sequences")

    from sklearn.model_selection import train_test_split
    grade_to_idx = {g: i for i, g in enumerate(GRADE_ORDER)}
    valid_seqs, valid_grades = [], []
    for seq in all_sequences:
        g = seq[-2]
        if g in grade_to_idx:
            valid_seqs.append(seq)
            valid_grades.append(grade_to_idx[g])

    train_idx, test_idx = train_test_split(
        np.arange(len(valid_seqs)), test_size=0.2,
        random_state=args.seed, stratify=valid_grades,
    )

    print(f"Training on device: {device}")
    t0 = time.time()
    results = train_and_evaluate(
        valid_seqs, valid_grades, train_idx, test_idx,
        seed=args.seed, hidden_dim=args.hidden_dim, epochs=args.epochs,
        batch_size=args.batch_size, learning_rate=args.learning_rate,
        dropout=args.dropout, patience=args.patience,
    )
    elapsed = time.time() - t0

    print(f"\n{'='*50}")
    print("Evaluation Results")
    print(f"{'='*50}")
    print(f"Exact Accuracy:     {results['exact_accuracy']:.4f}")
    print(f"Within-1 Accuracy:  {results['within_one_grade']:.4f}")
    print(f"Within-2 Accuracy:  {results['within_two_grades']:.4f}")
    print(f"Training time:      {elapsed:.1f}s")


if __name__ == "__main__":
    main()
