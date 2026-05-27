"""submissions/perceptron-baseline — MLP (perceptron) grade predictor baseline.

Trains a 3-layer MLP classifier on Moonboard route data. Each route is
flattened to a fixed-size binary hold vector, then passed through fully
connected layers to predict the climbing grade.

Exposes train_and_evaluate() for use by the benchmark harness.

Usage:
    uv run python submissions/perceptron-baseline/main.py --help
    uv run python submissions/perceptron-baseline/main.py --data-path Raw/moonboard_problems_setup_2016.json
"""

import argparse
import sys
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
from moonboard_analysis.training.metrics import evaluate_classification
from moonboard_analysis.utils.device import get_device
from moonboard_analysis.utils.reproducibility import set_seeds

NUM_COLS = 11
NUM_ROWS = 18
HOLD_VECTOR_DIM = NUM_COLS * NUM_ROWS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Perceptron (MLP) baseline — train and evaluate on Moonboard data"
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default="Raw/moonboard_problems_setup_2016.json",
        help="Path to raw Moonboard JSON data (default: Raw/moonboard_problems_setup_2016.json)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=".",
        help="Directory to save trained model weights (default: current directory)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=128,
        help="Hidden layer dimension (default: 128)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=300,
        help="Number of training epochs (default: 300)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Training batch size (default: 32)",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.001,
        help="Adam learning rate (default: 0.001)",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.3,
        help="Dropout probability (default: 0.3)",
    )
    return parser.parse_args()


def hold_to_index(hold_name: str) -> int:
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


def sequences_to_vectors(
    route_sequences: list[list[str]],
) -> np.ndarray:
    vectors = np.zeros((len(route_sequences), HOLD_VECTOR_DIM), dtype=np.float32)
    for i, seq in enumerate(route_sequences):
        for token in seq:
            idx = hold_to_index(token)
            if 0 <= idx < HOLD_VECTOR_DIM:
                vectors[i, idx] = 1.0
    return vectors


class MLPClassifier(nn.Module):
    def __init__(
        self, input_dim: int, hidden_dim: int, num_classes: int, dropout: float = 0.3
    ):
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


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    n_batches = 0
    for features, labels in loader:
        features, labels = features.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(features)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        n_batches += 1
    return total_loss / max(n_batches, 1)


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    n_batches = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for features, labels in loader:
            features, labels = features.to(device), labels.to(device)
            outputs = model(features)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
            n_batches += 1
    return total_loss / max(n_batches, 1), correct / total


def train_and_evaluate(
    sequences: list[list[str]],
    grades: list[int],
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    seed: int = 42,
    hidden_dim: int = 128,
    epochs: int = 300,
    batch_size: int = 32,
    learning_rate: float = 0.001,
    dropout: float = 0.3,
) -> dict[str, float]:
    """Train a fresh MLP on the training fold and evaluate on test fold.

    Args:
        sequences: Preprocessed route sequences (list of token lists).
        grades: Encoded grade labels.
        train_idx: Indices for the training fold.
        test_idx: Indices for the test fold.
        seed: Random seed for reproducibility.

    Returns:
        Dict with exact_accuracy, within_one_grade, within_two_grades.
    """
    set_seeds(seed)

    train_seqs = [sequences[i] for i in train_idx]
    test_seqs = [sequences[i] for i in test_idx]
    train_grades = [grades[i] for i in train_idx]
    test_grades = [grades[i] for i in test_idx]

    train_holds: list[list[str]] = []
    test_holds: list[list[str]] = []
    for seq in train_seqs:
        holds = [t for t in seq if hold_to_index(t) >= 0]
        train_holds.append(holds)
    for seq in test_seqs:
        holds = [t for t in seq if hold_to_index(t) >= 0]
        test_holds.append(holds)

    X_train = sequences_to_vectors(train_holds)
    X_test = sequences_to_vectors(test_holds)
    y_train = np.array(train_grades, dtype=np.int64)
    y_test = np.array(test_grades, dtype=np.int64)

    num_classes = len(GRADE_ORDER)

    train_ds = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long),
    )
    test_ds = TensorDataset(
        torch.tensor(X_test, dtype=torch.float32),
        torch.tensor(y_test, dtype=torch.long),
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size)

    device = get_device()
    model = MLPClassifier(
        input_dim=HOLD_VECTOR_DIM,
        hidden_dim=hidden_dim,
        num_classes=num_classes,
        dropout=dropout,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=15
    )

    best_test_loss = float("inf")
    for epoch in range(epochs):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        scheduler.step(test_loss)
        if test_loss < best_test_loss:
            best_test_loss = test_loss

    all_preds: list[int] = []
    all_labels: list[int] = []
    model.eval()
    with torch.no_grad():
        for features, lbls in test_loader:
            features = features.to(device)
            outputs = model(features)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy().tolist())
            all_labels.extend(lbls.numpy().tolist())

    metrics = evaluate_classification(all_labels, all_preds, num_classes)
    return {
        "exact_accuracy": metrics["exact_accuracy"],
        "within_one_grade": metrics["within_1_accuracy"],
        "within_two_grades": metrics["within_2_accuracy"],
    }


def main() -> None:
    args = parse_args()
    set_seeds(args.seed)

    data_path = args.data_path
    if not Path(data_path).exists():
        print(f"Error: Data file not found at '{data_path}'")
        print("Provide a valid path with --data-path")
        sys.exit(1)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading data from {data_path}")
    df = load_lstm_data(data_path)
    print(f"Raw data: {len(df)} routes")

    all_sequences = preprocess_lstm_data(df)
    all_sequences = drop_duplicate_sequences(all_sequences)
    print(f"After preprocessing: {len(all_sequences)} unique sequences")

    route_sequences: list[list[str]] = []
    route_grades: list[str] = []
    for seq in all_sequences:
        grade = seq[-2]
        if grade in GRADE_ORDER:
            holds = [t for t in seq[:-2] if hold_to_index(t) >= 0]
            route_sequences.append(holds)
            route_grades.append(grade)

    grade_to_idx = {g: i for i, g in enumerate(GRADE_ORDER)}
    encoded_grades = [grade_to_idx[g] for g in route_grades]
    num_classes = len(GRADE_ORDER)

    print(f"Valid routes: {len(route_sequences)}")
    print(f"Hold vector dim: {HOLD_VECTOR_DIM}")
    print(f"Number of classes: {num_classes}")

    X = sequences_to_vectors(route_sequences)
    y = np.array(encoded_grades, dtype=np.int64)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=args.seed,
        stratify=y,
    )
    print(f"Train: {len(X_train)}  Test: {len(X_test)}")

    train_ds = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long),
    )
    test_ds = TensorDataset(
        torch.tensor(X_test, dtype=torch.float32),
        torch.tensor(y_test, dtype=torch.long),
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size)

    device = get_device()
    print(f"Training on device: {device}")

    model = MLPClassifier(
        input_dim=HOLD_VECTOR_DIM,
        hidden_dim=args.hidden_dim,
        num_classes=num_classes,
        dropout=args.dropout,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=15
    )

    best_test_loss = float("inf")
    for epoch in range(args.epochs):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        scheduler.step(test_loss)

        if epoch % 50 == 0 or epoch == args.epochs - 1:
            print(
                f"Epoch {epoch:3d} | Train Loss: {train_loss:.4f} | "
                f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}"
            )

        if test_loss < best_test_loss:
            best_test_loss = test_loss

    all_preds: list[int] = []
    all_labels: list[int] = []
    model.eval()
    with torch.no_grad():
        for features, lbls in test_loader:
            features = features.to(device)
            outputs = model(features)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy().tolist())
            all_labels.extend(lbls.numpy().tolist())

    metrics = evaluate_classification(all_labels, all_preds, num_classes)

    print()
    print("=" * 50)
    print("Evaluation Results")
    print("=" * 50)
    print(f"Exact Accuracy:      {metrics['exact_accuracy']:.4f}")
    print(f"Within-1 Accuracy:   {metrics['within_1_accuracy']:.4f}")
    print(f"Within-2 Accuracy:   {metrics['within_2_accuracy']:.4f}")

    save_path = output_dir / "Perceptron_Moonboard.pth"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": {
                "input_dim": HOLD_VECTOR_DIM,
                "hidden_dim": args.hidden_dim,
                "num_classes": num_classes,
                "dropout": args.dropout,
            },
        },
        save_path,
    )
    print(f"Model saved to: {save_path}")


if __name__ == "__main__":
    main()
