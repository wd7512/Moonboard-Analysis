import argparse
import sys
from pathlib import Path

import mlflow
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader

from moonboard_analysis.config import GRADE_ORDER, LSTMConfig
from moonboard_analysis.data.dataset import LSTMSequenceDataset
from moonboard_analysis.data.loader import load_lstm_data
from moonboard_analysis.data.preprocessing import (
    drop_duplicate_sequences,
    preprocess_lstm_data,
)
from moonboard_analysis.models.lstm import ClimbingGradePredictor
from moonboard_analysis.training.trainer import evaluate_lstm, train_lstm_epoch
from moonboard_analysis.utils.device import get_device
from moonboard_analysis.utils.reproducibility import set_seeds


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train LSTM grade predictor on Moonboard data")
    parser.add_argument("--data-path", type=str, default=None, help="Path to raw JSON data file")
    parser.add_argument("--output-dir", type=str, default="models", help="Directory to save model")
    parser.add_argument("--embed-dim", type=int, default=16, help="Hold embedding dimension")
    parser.add_argument("--hidden-dim", type=int, default=128, help="LSTM hidden dimension")
    parser.add_argument("--num-layers", type=int, default=3, help="Number of LSTM layers")
    parser.add_argument("--epochs", type=int, default=500, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--learning-rate", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--test-size", type=float, default=0.2, help="Fraction for test split")
    parser.add_argument("--max-length", type=int, default=None, help="Max sequence length")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


def build_vocab(sequences: list[list[str]]) -> dict[str, int]:
    tokens = set()
    for seq in sequences:
        tokens.update(seq)
    vocab = {token: i + 1 for i, token in enumerate(sorted(tokens))}
    vocab["<PAD>"] = 0
    return vocab


def encode_grades(grades: list[str], grade_order: list[str]) -> list[int]:
    grade_to_idx = {g: i for i, g in enumerate(grade_order)}
    encoded = []
    for g in grades:
        if g not in grade_to_idx:
            print(f"Warning: Unknown grade '{g}', skipping")
            continue
        encoded.append(grade_to_idx[g])
    return encoded


def main() -> None:
    args = parse_args()
    set_seeds(args.seed)

    data_path = args.data_path or "Raw/moonboard_problems_setup_2016.json"
    if not Path(data_path).exists():
        print(f"Error: Data file not found at '{data_path}'")
        print("Please provide a valid path with --data-path")
        sys.exit(1)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading data from {data_path}")
    df = load_lstm_data(data_path)
    print(f"Raw data: {len(df)} routes")

    sequences = preprocess_lstm_data(df)
    sequences = drop_duplicate_sequences(sequences)
    print(f"After preprocessing: {len(sequences)} sequences")

    route_sequences = []
    route_grades = []
    for seq in sequences:
        grade = seq[-2]
        if grade in GRADE_ORDER:
            route_sequences.append(seq[:-2])
            route_grades.append(grade)

    vocab = build_vocab(route_sequences)
    encoded_grades = encode_grades(route_grades, GRADE_ORDER)

    if len(encoded_grades) == 0:
        print("Error: No valid grades found in data")
        sys.exit(1)

    num_classes = len(GRADE_ORDER)
    vocab_size = len(vocab)

    if args.max_length is not None:
        max_length = args.max_length
    else:
        max_length = max(len(seq) for seq in route_sequences)

    print(f"Vocab size: {vocab_size}")
    print(f"Max sequence length: {max_length}")
    print(f"Number of classes: {num_classes}")

    train_seqs, test_seqs, train_grades, test_grades = train_test_split(
        route_sequences, encoded_grades, test_size=args.test_size, random_state=args.seed,
        stratify=encoded_grades,
    )

    train_dataset = LSTMSequenceDataset(train_seqs, train_grades, vocab, max_length)
    test_dataset = LSTMSequenceDataset(test_seqs, test_grades, vocab, max_length)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)

    device = get_device()
    print(f"Training on device: {device}")

    model = ClimbingGradePredictor(
        vocab_size=vocab_size,
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_classes=num_classes,
    ).to(device)

    # Compute class weights only for classes present in the data
    unique_classes = np.unique(encoded_grades)
    class_weights_present = compute_class_weight(
        class_weight="balanced",
        classes=unique_classes,
        y=encoded_grades,
    )
    # Build full weight tensor for all classes in GRADE_ORDER
    # Missing classes get weight 1.0 (no up/down weighting)
    class_weights_full = np.ones(len(GRADE_ORDER), dtype=np.float32)
    for cls, w in zip(unique_classes, class_weights_present):
        class_weights_full[cls] = w
    class_weights_tensor = torch.tensor(class_weights_full, dtype=torch.float32, device=device)
    print(f"Class weights: {dict(zip(unique_classes.tolist(), class_weights_present.round(4).tolist()))}")

    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=20
    )

    config = LSTMConfig(
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        test_size=args.test_size,
        seed=args.seed,
        data_path=data_path,
        model_save_path=str(output_dir / "LSTM_Moonboard.pth"),
        max_length=max_length,
    )

    mlflow.set_experiment("LSTM Grade Prediction Training")

    best_test_loss = float("inf")
    best_test_acc = 0.0

    with mlflow.start_run() as run:
        mlflow.log_params(
            {
                "vocab_size": vocab_size,
                "embed_dim": config.embed_dim,
                "hidden_dim": config.hidden_dim,
                "num_layers": config.num_layers,
                "num_epochs": config.num_epochs,
                "batch_size": config.batch_size,
                "learning_rate": config.learning_rate,
                "max_length": max_length,
                "num_classes": num_classes,
                "train_size": len(train_seqs),
                "test_size": len(test_seqs),
                "seed": config.seed,
            }
        )

        checkpoint_path = output_dir / "lstm_checkpoint.pth"

        for epoch in range(config.num_epochs):
            train_loss = train_lstm_epoch(model, train_loader, criterion, optimizer, device)
            test_loss, test_acc = evaluate_lstm(model, test_loader, criterion, device)
            scheduler.step(test_loss)

            if epoch % 10 == 0 or epoch == config.num_epochs - 1:
                print(
                    f"Epoch {epoch:3d} | Train Loss: {train_loss:.4f} | "
                    f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}"
                )

            mlflow.log_metrics(
                {"train_loss": train_loss, "test_loss": test_loss, "test_acc": test_acc},
                step=epoch,
            )

            if test_loss < best_test_loss:
                best_test_loss = test_loss
                best_test_acc = test_acc
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "config": {
                            "vocab_size": vocab_size,
                            "embed_dim": config.embed_dim,
                            "hidden_dim": config.hidden_dim,
                            "num_layers": config.num_layers,
                            "num_classes": num_classes,
                            "max_length": max_length,
                        },
                        "vocab": vocab,
                        "test_loss": test_loss,
                        "test_acc": test_acc,
                    },
                    checkpoint_path,
                )

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "config": {
                    "vocab_size": vocab_size,
                    "embed_dim": config.embed_dim,
                    "hidden_dim": config.hidden_dim,
                    "num_layers": config.num_layers,
                    "num_classes": num_classes,
                    "max_length": max_length,
                },
                "vocab": vocab,
            },
            config.model_save_path,
        )
        mlflow.log_artifact(config.model_save_path)
        mlflow.log_artifact(str(checkpoint_path))

        mlflow.log_metrics(
            {
                "best_test_loss": best_test_loss,
                "best_test_acc": best_test_acc,
            }
        )

        print(f"\nRun ID: {run.info.run_id}")
        print(f"Best test loss: {best_test_loss:.4f}")
        print(f"Best test accuracy: {best_test_acc:.4f}")
        print(f"Model saved to: {config.model_save_path}")
        print(f"Checkpoint saved to: {checkpoint_path}")


if __name__ == "__main__":
    main()
