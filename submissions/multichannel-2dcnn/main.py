"""submissions/multichannel-2dcnn — Multi-channel 2D CNN grade predictor.

3 input channels (start/middle/end) mapped to binary spatial grids (18×11 each).
Compact conv stack: Conv2d(3→16, 3×3) × 2 → pool → Conv2d(16→32) → pool → Linear.

Target: paper 42% (Petashvili & Rodda 2023), baseline 36.81% (single-channel).

Exposes train_and_evaluate() for use by the benchmark harness.

Usage:
    uv run python submissions/multichannel-2dcnn/main.py --help
    uv run python submissions/multichannel-2dcnn/main.py \
        --data-path Raw/moonboard_problems_setup_2016.json
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
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

GRADE_LABELS = frozenset(GRADE_ORDER)
SECTION_TOKENS = frozenset({"START_END", "MIDDLE_END", "END_ROUTE", "GRADE_END"})


def hold_to_row_col(hold_name: str) -> tuple[int, int] | None:
    """Convert hold token to (row, col) or None if invalid."""
    if len(hold_name) < 2:
        return None
    col_char = hold_name[0]
    if col_char < "A" or col_char > "K":
        return None
    row_part = hold_name[1:]
    if not row_part.isdigit():
        return None
    col = ord(col_char) - ord("A")
    row = int(row_part) - 1
    if 0 <= row < NUM_ROWS and 0 <= col < NUM_COLS:
        return (row, col)
    return None


def sequence_to_multichannel(seq: list[str]) -> np.ndarray:
    """Convert a token sequence to a 3-channel binary matrix.

    Channels: [0] = start holds, [1] = middle holds, [2] = end holds.
    Section boundaries are identified by START_END, MIDDLE_END, END_ROUTE tokens.

    Returns:
        Float32 array of shape (3, 18, 11).
    """
    matrix = np.zeros((3, NUM_ROWS, NUM_COLS), dtype=np.float32)

    # Find section boundaries
    section_boundaries = []
    for tok_i, token in enumerate(seq):
        if token == "START_END":
            section_boundaries.append(("start_end", tok_i))
        elif token == "MIDDLE_END":
            section_boundaries.append(("middle_end", tok_i))
        elif token == "END_ROUTE":
            section_boundaries.append(("end_route", tok_i))

    # Determine start/middle/end hold ranges in the sequence
    # Default: before START_END = start, START_END to MIDDLE_END = middle, after MIDDLE_END = end
    start_range = (0, 0)
    middle_range = (0, 0)
    end_range = (0, 0)

    if len(section_boundaries) >= 3:
        # START_END, MIDDLE_END, END_ROUTE present
        start_end_idx = section_boundaries[0][1]
        middle_end_idx = section_boundaries[1][1]
        end_route_idx = section_boundaries[2][1]

        start_range = (0, start_end_idx)
        middle_range = (start_end_idx + 1, middle_end_idx)
        end_range = (middle_end_idx + 1, end_route_idx)
    elif len(section_boundaries) == 2:
        # Try to infer
        if section_boundaries[0][0] == "start_end" and section_boundaries[1][0] == "end_route":
            start_range = (0, section_boundaries[0][1])
            middle_range = (section_boundaries[0][1] + 1, section_boundaries[1][1])
        elif section_boundaries[0][0] == "start_end" and section_boundaries[1][0] == "middle_end":
            start_range = (0, section_boundaries[0][1])
            middle_range = (section_boundaries[0][1] + 1, section_boundaries[1][1])
            end_range = (section_boundaries[1][1] + 1, len(seq) - 2)  # last 2 are grade, GRADE_END
    elif len(section_boundaries) == 1:
        start_range = (0, section_boundaries[0][1])
        middle_range = (section_boundaries[0][1] + 1, len(seq) - 2)
    else:
        # No section markers — put all in middle
        middle_range = (0, len(seq) - 2)

    # Populate channels
    for ch, (lo, hi) in enumerate([start_range, middle_range, end_range]):
        for tok_i in range(lo, hi):
            if tok_i >= len(seq):
                break
            token = seq[tok_i]
            if token in GRADE_LABELS or token in SECTION_TOKENS:
                continue
            rc = hold_to_row_col(token)
            if rc is not None:
                row, col = rc
                matrix[ch, row, col] = 1.0

    return matrix


def sequences_to_multichannel_tensor(
    sequences: list[list[str]],
) -> np.ndarray:
    """Convert a list of token sequences to a batch of 3-channel matrices.

    Returns:
        Float32 array of shape (N, 3, 18, 11).
    """
    n = len(sequences)
    result = np.zeros((n, 3, NUM_ROWS, NUM_COLS), dtype=np.float32)
    for i, seq in enumerate(sequences):
        result[i] = sequence_to_multichannel(seq)
    return result


class MultiChannelCNN2D(nn.Module):
    """Compact 2D CNN with 3 input channels (start/middle/end).

    Architecture:
        Conv2d(3→16, 3×3) → BN → ReLU
        Conv2d(16→16, 3×3) → BN → ReLU → MaxPool(2)
        Conv2d(16→32, 3×3) → BN → ReLU → MaxPool(2)
        AdaptiveAvgPool(1) → Linear(32→12)
    """

    def __init__(self, num_classes: int = 12):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(32, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def train_and_evaluate(
    sequences: list[list[str]],
    grades: list[int],
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    seed: int = 42,
    epochs: int = 50,
    batch_size: int = 64,
    learning_rate: float = 0.001,
    patience: int = 10,
) -> dict[str, float]:
    """Train a fresh Multi-channel 2D CNN on train_idx, evaluate on test_idx.

    Returns:
        Dict with exact_accuracy, within_one_grade, within_two_grades, macro_f1.
    """
    set_seeds(seed)

    train_seqs = [sequences[i] for i in train_idx]
    test_seqs = [sequences[i] for i in test_idx]
    y_train = np.array([grades[i] for i in train_idx], dtype=np.int64)
    y_test = np.array([grades[i] for i in test_idx], dtype=np.int64)

    # Convert to multi-channel spatial tensors
    X_train = sequences_to_multichannel_tensor(train_seqs)
    X_test = sequences_to_multichannel_tensor(test_seqs)

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

    device = get_device()
    model = MultiChannelCNN2D(num_classes=len(GRADE_ORDER)).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=patience
    )

    best_loss = float("inf")
    best_state = None
    best_epoch = 0

    for epoch in range(epochs):
        # Train
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
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
                xb, yb = xb.to(device), yb.to(device)
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
        model.to(device)

    # Predict
    all_preds, all_labels = [], []
    model.eval()
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device)
            preds = torch.argmax(model(xb), 1)
            all_preds.extend(preds.cpu().numpy().tolist())
            all_labels.extend(yb.numpy().tolist())

    metrics = evaluate_classification(all_labels, all_preds, len(GRADE_ORDER))
    return extract_required_metrics(metrics)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Multi-channel 2D CNN — train and evaluate on Moonboard data"
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default="Raw/moonboard_problems_setup_2016.json",
    )
    parser.add_argument("--output-dir", type=str, default=".")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--patience", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seeds(args.seed)

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

    grade_to_idx = {g: i for i, g in enumerate(GRADE_ORDER)}
    valid_seqs, valid_grades = [], []
    for seq in all_sequences:
        g = seq[-2]
        if g in grade_to_idx:
            valid_seqs.append(seq)
            valid_grades.append(grade_to_idx[g])

    train_idx, test_idx = train_test_split(
        np.arange(len(valid_seqs)),
        test_size=0.2,
        random_state=args.seed,
        stratify=valid_grades,
    )

    device = get_device()
    print(f"Training on device: {device}")
    t0 = time.time()
    results = train_and_evaluate(
        valid_seqs, valid_grades, train_idx, test_idx,
        seed=args.seed, epochs=args.epochs,
        batch_size=args.batch_size, learning_rate=args.learning_rate,
        patience=args.patience,
    )
    elapsed = time.time() - t0

    print(f"\n{'=' * 50}")
    print("Evaluation Results")
    print(f"{'=' * 50}")
    print(f"Exact Accuracy:     {results['exact_accuracy']:.4f}")
    print(f"Within-1 Accuracy:  {results['within_one_grade']:.4f}")
    print(f"Within-2 Accuracy:  {results['within_two_grades']:.4f}")
    print(f"Macro-F1:           {results['macro_f1']:.4f}")
    print(f"Training time:      {elapsed:.1f}s")


if __name__ == "__main__":
    main()
