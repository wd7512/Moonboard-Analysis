"""submissions/lstm-baseline — LSTM grade predictor reference submission.

Trains a 3-layer LSTM classifier on Moonboard route sequences and evaluates
using exact, within-1, and within-2 grade accuracy metrics.

Usage:
    uv run python submissions/lstm-baseline/main.py --help
    uv run python submissions/lstm-baseline/main.py --data-path Raw/moonboard_problems_setup_2016.json
"""

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

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
        description="LSTM baseline — train and evaluate on Moonboard data"
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
        "--embed-dim",
        type=int,
        default=16,
        help="Hold embedding dimension (default: 16)",
    )
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=128,
        help="LSTM hidden dimension (default: 128)",
    )
    parser.add_argument(
        "--num-layers",
        type=int,
        default=3,
        help="Number of LSTM layers (default: 3)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=500,
        help="Number of training epochs (default: 500)",
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
    return parser.parse_args()


def build_vocab(sequences: list[list[str]]) -> dict[str, int]:
    """Build token vocabulary from route sequences (0 reserved for padding)."""
    tokens: set[str] = set()
    for seq in sequences:
        tokens.update(seq)
    vocab: dict[str, int] = {token: i + 1 for i, token in enumerate(sorted(tokens))}
    vocab["<PAD>"] = 0
    return vocab


def main() -> None:
    args = parse_args()
    set_seeds(args.seed)

    # -- Resolve paths --
    data_path = args.data_path
    if not Path(data_path).exists():
        print(f"Error: Data file not found at '{data_path}'")
        print("Provide a valid path with --data-path")
        sys.exit(1)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # -- Load & preprocess --
    print(f"Loading data from {data_path}")
    df = load_lstm_data(data_path)
    print(f"Raw data: {len(df)} routes")

    sequences = preprocess_lstm_data(df)
    sequences = drop_duplicate_sequences(sequences)
    print(f"After preprocessing: {len(sequences)} unique sequences")

    route_sequences: list[list[str]] = []
    route_grades: list[str] = []
    for seq in sequences:
        grade = seq[-2]
        if grade in GRADE_ORDER:
            route_sequences.append(seq[:-2])
            route_grades.append(grade)

    grade_to_idx = {g: i for i, g in enumerate(GRADE_ORDER)}
    encoded_grades = [grade_to_idx[g] for g in route_grades]

    vocab = build_vocab(route_sequences)
    max_length = max(len(s) for s in route_sequences)
    num_classes = len(GRADE_ORDER)
    vocab_size = len(vocab)

    print(f"Vocab size: {vocab_size}")
    print(f"Max sequence length: {max_length}")
    print(f"Number of classes: {num_classes}")

    # -- Train/test split --
    train_seqs, test_seqs, train_grades, test_grades = train_test_split(
        route_sequences,
        encoded_grades,
        test_size=0.2,
        random_state=args.seed,
        stratify=encoded_grades,
    )
    print(f"Train: {len(train_seqs)}  Test: {len(test_seqs)}")

    train_ds = LSTMSequenceDataset(train_seqs, train_grades, vocab, max_length)
    test_ds = LSTMSequenceDataset(test_seqs, test_grades, vocab, max_length)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size)

    # -- Build model --
    device = get_device()
    print(f"Training on device: {device}")

    model = ClimbingGradePredictor(
        vocab_size=vocab_size,
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_classes=num_classes,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=20
    )

    # -- Training loop --
    best_test_loss = float("inf")
    for epoch in range(args.epochs):
        train_loss = train_lstm_epoch(model, train_loader, criterion, optimizer, device)
        test_loss, test_acc = evaluate_lstm(model, test_loader, criterion, device)
        scheduler.step(test_loss)

        if epoch % 50 == 0 or epoch == args.epochs - 1:
            print(
                f"Epoch {epoch:3d} | Train Loss: {train_loss:.4f} | "
                f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}"
            )

        if test_loss < best_test_loss:
            best_test_loss = test_loss

    # -- Final evaluation --
    all_preds: list[int] = []
    all_labels: list[int] = []
    model.eval()
    with torch.no_grad():
        for seqs, grades in test_loader:
            seqs, grades = seqs.to(device), grades.to(device)
            outputs = model(seqs)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy().tolist())
            all_labels.extend(grades.cpu().numpy().tolist())

    metrics = evaluate_classification(all_labels, all_preds, num_classes)

    print()
    print("=" * 50)
    print("Evaluation Results")
    print("=" * 50)
    print(f"Exact Accuracy:      {metrics['exact_accuracy']:.4f}")
    print(f"Within-1 Accuracy:   {metrics['within_1_accuracy']:.4f}")
    print(f"Within-2 Accuracy:   {metrics['within_2_accuracy']:.4f}")

    # -- Save model --
    save_path = output_dir / "LSTM_Moonboard.pth"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": {
                "vocab_size": vocab_size,
                "embed_dim": args.embed_dim,
                "hidden_dim": args.hidden_dim,
                "num_layers": args.num_layers,
                "num_classes": num_classes,
                "max_length": max_length,
            },
            "vocab": vocab,
        },
        save_path,
    )
    print(f"Model saved to: {save_path}")


if __name__ == "__main__":
    main()
