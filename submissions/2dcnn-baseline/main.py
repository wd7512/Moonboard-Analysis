"""submissions/2dcnn-baseline — 2D CNN grade predictor reference submission.

Trains a 4-layer 2D CNN classifier on Moonboard route binary hold matrices
and evaluates using exact, within-1, and within-2 grade accuracy metrics.

Architecture follows Petashvili & Rodda (2023): 4 conv layers with 3×3
kernels, batch norm, ReLU, then fully connected layers.

Exposes train_and_evaluate() for use by the benchmark harness.

Usage:
    uv run python submissions/2dcnn-baseline/main.py --help
    uv run python submissions/2dcnn-baseline/main.py \\
        --data-path Raw/moonboard_problems_setup_2016.json
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="2D CNN baseline — train and evaluate on Moonboard data"
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
        "--epochs",
        type=int,
        default=50,
        help="Number of training epochs (default: 50)",
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
        "--patience",
        type=int,
        default=20,
        help="Early stopping patience (default: 20)",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.5,
        help="Dropout probability for FC layers (default: 0.5)",
    )
    return parser.parse_args()


def hold_to_matrix(holds: list[str]) -> np.ndarray:
    """Convert a list of hold tokens to an 18×11 binary matrix.

    Args:
        holds: Hold tokens like ["A1", "B5", "K18"].

    Returns:
        Float32 array of shape (18, 11) with 1.0 at hold positions.
    """
    matrix = np.zeros((NUM_ROWS, NUM_COLS), dtype=np.float32)
    for token in holds:
        if len(token) < 2:
            continue
        col_char = token[0]
        if col_char < "A" or col_char > "K":
            continue
        row_part = token[1:]
        if not row_part.isdigit():
            continue
        col = ord(col_char) - ord("A")
        row = int(row_part) - 1
        if 0 <= row < NUM_ROWS and 0 <= col < NUM_COLS:
            matrix[row, col] = 1.0
    return matrix


def sequences_to_matrices(route_sequences: list[list[str]]) -> np.ndarray:
    """Convert hold token sequences to an array of binary matrices.

    Args:
        route_sequences: List of hold token lists per route.

    Returns:
        Float32 array of shape (n_routes, 18, 11).
    """
    matrices = np.zeros((len(route_sequences), NUM_ROWS, NUM_COLS), dtype=np.float32)
    for i, holds in enumerate(route_sequences):
        matrices[i] = hold_to_matrix(holds)
    return matrices


class CNN2DGradePredictor(nn.Module):
    """4-layer 2D CNN for Moonboard grade prediction.

    Architecture (Petashvili & Rodda 2023):
        - 4 convolutional layers with 3×3 kernels, batch norm, ReLU
        - Adaptive average pooling
        - Two fully connected layers with dropout
    """

    def __init__(
        self,
        num_classes: int,
        channels: int = 1,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_layers(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


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
    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
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
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
            n_batches += 1
    return total_loss / max(n_batches, 1), correct / max(total, 1)


def train_and_evaluate(
    sequences: list[list[str]],
    grades: list[int],
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    seed: int = 42,
    epochs: int = 100,
    batch_size: int = 32,
    learning_rate: float = 0.001,
    patience: int = 20,
    dropout: float = 0.5,
) -> dict[str, float]:
    """Train a fresh 2D CNN on the training fold and evaluate on test fold.

    Args:
        sequences: Preprocessed route sequences (list of token lists).
        grades: Encoded grade labels (parallel to sequences).
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
        holds = [t for t in seq[:-2] if hold_to_matrix([t]).sum() > 0]
        train_holds.append(holds)
    for seq in test_seqs:
        holds = [t for t in seq[:-2] if hold_to_matrix([t]).sum() > 0]
        test_holds.append(holds)

    X_train = sequences_to_matrices(train_holds)
    X_test = sequences_to_matrices(test_holds)
    y_train = np.array(train_grades, dtype=np.int64)
    y_test = np.array(test_grades, dtype=np.int64)

    # Add channel dimension: (N, 18, 11) -> (N, 1, 18, 11)
    X_train = X_train[:, np.newaxis, :, :]
    X_test = X_test[:, np.newaxis, :, :]

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
    model = CNN2DGradePredictor(
        num_classes=num_classes,
        dropout=dropout,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=patience
    )

    best_test_loss = float("inf")
    best_epoch = 0
    for epoch in range(epochs):
        _ = train_epoch(model, train_loader, criterion, optimizer, device)
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        scheduler.step(test_loss)
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            best_epoch = epoch
        if epoch - best_epoch >= patience:
            print(f"Early stopping at epoch {epoch} (no improvement for {patience} epochs)")
            break

    all_preds: list[int] = []
    all_labels: list[int] = []
    model.eval()
    with torch.no_grad():
        for inputs, lbls in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
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
            holds = [t for t in seq[:-2] if hold_to_matrix([t]).sum() > 0]
            route_sequences.append(holds)
            route_grades.append(grade)

    grade_to_idx = {g: i for i, g in enumerate(GRADE_ORDER)}
    encoded_grades = [grade_to_idx[g] for g in route_grades]
    num_classes = len(GRADE_ORDER)

    print(f"Valid routes: {len(route_sequences)}")
    print(f"Number of classes: {num_classes}")

    X = sequences_to_matrices(route_sequences)
    y = np.array(encoded_grades, dtype=np.int64)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=args.seed,
        stratify=y,
    )
    print(f"Train: {len(X_train)}  Test: {len(X_test)}")

    # Add channel dimension: (N, 18, 11) -> (N, 1, 18, 11)
    X_train = X_train[:, np.newaxis, :, :]
    X_test = X_test[:, np.newaxis, :, :]

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

    model = CNN2DGradePredictor(
        num_classes=num_classes,
        dropout=args.dropout,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=args.patience
    )

    best_test_loss = float("inf")
    best_epoch = 0
    for epoch in range(args.epochs):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        scheduler.step(test_loss)

        if epoch % 50 == 0 or epoch == args.epochs - 1 or test_loss < best_test_loss:
            print(
                f"Epoch {epoch:3d} | Train Loss: {train_loss:.4f} | "
                f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}"
            )

        if test_loss < best_test_loss:
            best_test_loss = test_loss
            best_epoch = epoch

        if epoch - best_epoch >= args.patience:
            print(f"Early stopping at epoch {epoch} (no improvement for {args.patience} epochs)")
            break

    all_preds: list[int] = []
    all_labels: list[int] = []
    model.eval()
    with torch.no_grad():
        for inputs, lbls in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
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

    save_path = output_dir / "2DCNN_Moonboard.pth"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": {
                "num_classes": num_classes,
                "dropout": args.dropout,
            },
        },
        save_path,
    )
    print(f"Model saved to: {save_path}")


if __name__ == "__main__":
    main()
