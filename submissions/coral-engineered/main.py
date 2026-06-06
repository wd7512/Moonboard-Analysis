"""submissions/coral-engineered — Focal ordinal ensemble with gamma sweep and longer training.

Tests multiple gamma values for focal loss and longer training.
Gamma=2.0 was best in previous tests, but let's verify.

Usage:
    uv run python submissions/coral-engineered/main.py --help
    uv run python submissions/coral-engineered/main.py \\
        --data-path Raw/moonboard_problems_setup_2016.json
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
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
NUM_CLASSES = len(GRADE_ORDER)  # 13
GRADE_LABELS = frozenset(GRADE_ORDER)
NUM_THRESHOLDS = NUM_CLASSES - 1  # 12


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


class CORALModel(nn.Module):
    def __init__(self, input_dim: int, hidden_layers: list[int], num_classes: int, dropout: float = 0.3):
        super().__init__()
        self.num_thresholds = num_classes - 1
        layers = []
        prev_dim = input_dim
        for h in hidden_layers:
            layers.extend([
                nn.Linear(prev_dim, h),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_dim = h
        self.net = nn.Sequential(*layers)
        self.coral_weight = nn.Linear(prev_dim, 1, bias=False)
        self.coral_bias = nn.Parameter(torch.zeros(self.num_thresholds))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.net(x)
        shared = self.coral_weight(h)
        return shared + self.coral_bias.unsqueeze(0)


class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0):
        super().__init__()
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        p_t = torch.exp(-bce)
        return ((1.0 - p_t) ** self.gamma * bce).mean()


def _labels_to_ordinal(labels: np.ndarray) -> np.ndarray:
    targets = np.zeros((len(labels), NUM_THRESHOLDS), dtype=np.float32)
    for i, g in enumerate(labels):
        targets[i, :g] = 1.0
    return targets


def _ordinal_to_label(logits: torch.Tensor) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    return (probs > 0.5).sum(dim=1)


def _train_ordinal(
    model: nn.Module, train_loader: DataLoader, eval_loader: DataLoader,
    lr: float, dev: torch.device, epochs: int, patience: int, gamma: float = 2.0,
) -> None:
    criterion = FocalLoss(gamma=gamma)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=10)

    best_loss = float("inf")
    best_state = None
    best_epoch = 0

    for epoch in range(epochs):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(dev), yb.to(dev)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()

        model.eval()
        eval_loss = 0.0
        n_batches = 0
        with torch.no_grad():
            for xb, yb in eval_loader:
                xb, yb = xb.to(dev), yb.to(dev)
                eval_loss += criterion(model(xb), yb).item()
                n_batches += 1
        eval_loss /= max(n_batches, 1)
        scheduler.step(eval_loss)

        if eval_loss < best_loss:
            best_loss = eval_loss
            best_epoch = epoch
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if epoch - best_epoch >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
        model.to(dev)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Focal ordinal ensemble")
    parser.add_argument("--data-path", type=str, default="Raw/moonboard_problems_setup_2016.json")
    parser.add_argument("--output-dir", type=str, default=".")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gamma", type=float, default=2.0)
    return parser.parse_args()


def train_and_evaluate(
    sequences: list[list[str]],
    grades: list[int],
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    seed: int = 42,
    gamma: float = 2.0,
) -> dict[str, float]:
    """Train focal ordinal ensemble with longer training."""
    set_seeds(seed)

    train_seqs = [sequences[i] for i in train_idx]
    test_seqs = [sequences[i] for i in test_idx]
    y_train = np.array([grades[i] for i in train_idx], dtype=np.int64)
    y_test = np.array([grades[i] for i in test_idx], dtype=np.int64)

    dev = get_device()

    X_train = _sequences_to_vectors(train_seqs)
    X_test = _sequences_to_vectors(test_seqs)

    mu = X_train.mean(axis=0)
    sd = X_train.std(axis=0) + 1e-8
    X_train = (X_train - mu) / sd
    X_test = (X_test - mu) / sd

    y_train_ord = _labels_to_ordinal(y_train)

    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_ord_t = torch.tensor(y_train_ord, dtype=torch.float32)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)

    train_loader = DataLoader(
        TensorDataset(X_train_t, y_train_ord_t), batch_size=256, shuffle=True,
    )
    eval_loader = DataLoader(
        TensorDataset(X_train_t, y_train_ord_t), batch_size=512,
    )

    # Model A: Standard
    set_seeds(seed)
    model_a = CORALModel(HOLD_VECTOR_DIM, [256, 128], NUM_CLASSES, dropout=0.3).to(dev)
    _train_ordinal(model_a, train_loader, eval_loader, 0.001, dev, 200, 30, gamma=gamma)

    # Model B: Wider
    set_seeds(seed + 1)
    model_b = CORALModel(HOLD_VECTOR_DIM, [512, 256, 128], NUM_CLASSES, dropout=0.15).to(dev)
    _train_ordinal(model_b, train_loader, eval_loader, 0.001, dev, 200, 30, gamma=gamma)

    # Ensemble: average logits
    model_a.eval()
    model_b.eval()
    with torch.no_grad():
        logits_a = model_a(X_test_t.to(dev))
        logits_b = model_b(X_test_t.to(dev))

    avg_logits = (logits_a + logits_b) / 2.0
    y_pred = _ordinal_to_label(avg_logits)
    all_preds = y_pred.cpu().numpy().tolist()
    all_labels = y_test.tolist()

    metrics = evaluate_classification(all_labels, all_preds, NUM_CLASSES)
    return extract_required_metrics(metrics)


def main() -> None:
    args = parse_args()
    set_seeds(args.seed)

    data_path = args.data_path
    if not Path(data_path).exists():
        print(f"Error: Data file not found at '{data_path}'")
        sys.exit(1)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading data from {data_path}, gamma={args.gamma}")
    df = load_lstm_data(data_path)
    print(f"Raw data: {len(df)} routes")

    all_sequences = preprocess_lstm_data(df)
    all_sequences = drop_duplicate_sequences(all_sequences)
    print(f"After preprocessing: {len(all_sequences)} unique sequences")

    from sklearn.model_selection import train_test_split

    grade_to_idx = {g: i for i, g in enumerate(GRADE_ORDER)}
    valid_seqs: list[list[str]] = []
    valid_grades: list[int] = []
    for seq in all_sequences:
        grade = seq[-2]
        if grade in grade_to_idx:
            valid_seqs.append(seq)
            valid_grades.append(grade_to_idx[grade])

    train_idx, test_idx = train_test_split(
        np.arange(len(valid_seqs)), test_size=0.2, random_state=args.seed, stratify=valid_grades,
    )

    import time
    t0 = time.time()
    results = train_and_evaluate(
        valid_seqs, valid_grades, train_idx, test_idx, seed=args.seed, gamma=args.gamma,
    )
    elapsed = time.time() - t0

    print()
    print("=" * 50)
    print("Evaluation Results")
    print("=" * 50)
    print(f"Exact Accuracy:     {results['exact_accuracy']:.4f}")
    print(f"Within-1 Accuracy:  {results['within_one_grade']:.4f}")
    print(f"Within-2 Accuracy:  {results['within_two_grades']:.4f}")
    print(f"Macro-F1:           {results['macro_f1']:.4f}")
    print(f"Training time:      {elapsed:.1f}s")


if __name__ == "__main__":
    main()
