import argparse
import sys
from pathlib import Path

import mlflow
import numpy as np
import torch
import torch.nn as nn
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
from moonboard_analysis.utils.device import get_device
from moonboard_analysis.utils.reproducibility import set_seeds


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate LSTM grade predictor on test set")
    parser.add_argument("--data-path", type=str, default=None, help="Path to raw JSON data file")
    parser.add_argument(
        "--model-path", type=str, default="models/LSTM_Moonboard.pth",
        help="Path to trained LSTM checkpoint",
    )
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for evaluation")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


def build_vocab(sequences: list[list[str]]) -> dict[str, int]:
    """Build vocabulary from sequences (0 is reserved for padding).

    Args:
        sequences: List of token sequences.

    Returns:
        Dict mapping token to index, with 0 reserved for PAD.
    """
    tokens = set()
    for seq in sequences:
        tokens.update(seq)
    vocab = {token: i + 1 for i, token in enumerate(sorted(tokens))}
    vocab["<PAD>"] = 0
    return vocab


def load_model_and_vocab(
    model_path: str, device: torch.device
) -> tuple[ClimbingGradePredictor, dict, dict]:
    if not Path(model_path).exists():
        print(f"Error: Model file not found at '{model_path}'")
        print("Train a model first with moonboard-train-lstm")
        sys.exit(1)

    checkpoint = torch.load(model_path, map_location=device, weights_only=True)

    # Handle both new and old checkpoint formats
    if "config" in checkpoint:
        config = checkpoint["config"]
        vocab = checkpoint.get("vocab", {})
    else:
        # Old format fallback - infer config from state dict
        print("Warning: Old checkpoint format detected. Inferring from state_dict.")
        if isinstance(checkpoint, dict) and "embedding.weight" in checkpoint:
            state_dict = checkpoint
        elif "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        else:
            state_dict = checkpoint

        # Infer dimensions from state dict
        vocab_size = state_dict["embedding.weight"].shape[0]
        embed_dim = state_dict["embedding.weight"].shape[1]
        # lstm layer hidden size is the second dimension of lstm.weight_ih_l0
        hidden_dim = state_dict["lstm.weight_ih_l0"].shape[0] // 4  # 4 gates in LSTM
        layer_keys = [k for k in state_dict if "lstm.weight_ih_l" in k]
        num_layers = max((int(k.split("_l")[-1]) for k in layer_keys), default=0) + 1
        num_classes = state_dict["fc.bias"].shape[0]

        config = {
            "vocab_size": vocab_size,
            "embed_dim": embed_dim,
            "hidden_dim": hidden_dim,
            "num_layers": num_layers,
            "num_classes": num_classes,
            "max_length": 50,
        }
        vocab = {}

    model = ClimbingGradePredictor(
        vocab_size=config["vocab_size"],
        embed_dim=config["embed_dim"],
        hidden_dim=config["hidden_dim"],
        num_layers=config["num_layers"],
        num_classes=config["num_classes"],
    )

    # Handle both new and old checkpoint state dict keys
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        # Try loading directly as state dict
        model.load_state_dict(checkpoint)

    model.to(device)
    model.eval()

    return model, vocab, config


def main() -> None:
    args = parse_args()
    set_seeds(args.seed)

    data_path = args.data_path or "Raw/moonboard_problems_setup_2016.json"
    if not Path(data_path).exists():
        print(f"Error: Data file not found at '{data_path}'")
        print("Please provide a valid path with --data-path")
        sys.exit(1)

    print(f"Loading data from {data_path}")
    df = load_lstm_data(data_path)

    sequences = preprocess_lstm_data(df)
    sequences = drop_duplicate_sequences(sequences)

    route_sequences = []
    route_grades = []
    for seq in sequences:
        grade = seq[-2]
        if grade in GRADE_ORDER:
            route_sequences.append(seq[:-2])
            route_grades.append(grade)

    grade_to_idx = {g: i for i, g in enumerate(GRADE_ORDER)}
    encoded_grades = [grade_to_idx[g] for g in route_grades]

    max_length = 50
    train_seqs, test_seqs, train_grades, test_grades = train_test_split(
        route_sequences, encoded_grades, test_size=0.2, random_state=args.seed,
        stratify=encoded_grades,
    )

    # Build vocab from training sequences (before loading model)
    vocab = build_vocab(train_seqs)

    device = get_device()
    print(f"Loading model from {args.model_path}")
    model, loaded_vocab, model_config = load_model_and_vocab(args.model_path, device)
    print(f"Model loaded on device: {device}")

    # Use loaded vocab if available, otherwise use built vocab
    if loaded_vocab:
        vocab = loaded_vocab

    max_length = model_config["max_length"]
    num_classes = model_config["num_classes"]

    test_dataset = LSTMSequenceDataset(test_seqs, test_grades, vocab, max_length)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)

    criterion = nn.CrossEntropyLoss()

    mlflow.set_experiment("LSTM Grade Prediction Evaluation")

    with mlflow.start_run() as run:
        mlflow.log_params(
            {
                "model_path": args.model_path,
                "data_path": data_path,
                "batch_size": args.batch_size,
                "seed": args.seed,
                "max_length": max_length,
                "num_classes": num_classes,
                "test_size": len(test_seqs),
            }
        )

        model.eval()
        all_preds = []
        all_labels = []
        total_loss = 0.0

        with torch.no_grad():
            for sequences_batch, grades_batch in test_loader:
                sequences_batch = sequences_batch.to(device)
                grades_batch = grades_batch.to(device)
                outputs = model(sequences_batch)
                loss = criterion(outputs, grades_batch)
                total_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                all_preds.extend(predicted.cpu().numpy().tolist())
                all_labels.extend(grades_batch.cpu().numpy().tolist())

        avg_loss = total_loss / len(test_loader)
        metrics = evaluate_classification(all_labels, all_preds, num_classes)

        print(f"\n{'=' * 50}")
        print("Evaluation Results")
        print(f"{'=' * 50}")
        print(f"Test Loss: {avg_loss:.4f}")
        print(f"Exact Accuracy: {metrics['exact_accuracy']:.4f}")
        print(f"Within-1 Accuracy: {metrics['within_1_accuracy']:.4f}")
        print(f"Within-2 Accuracy: {metrics['within_2_accuracy']:.4f}")
        print(f"Within-3 Accuracy: {metrics['within_3_accuracy']:.4f}")
        print(f"Within-4 Accuracy: {metrics['within_4_accuracy']:.4f}")

        print("\nConfusion Matrix:")
        print(metrics["confusion_matrix"])

        print("\nPer-Class Metrics:")
        print(f"{'Grade':<8} {'Precision':<12} {'Recall':<12} {'F1':<12}")
        print("-" * 44)
        num_classes_actual = len(metrics["per_class_precision"])
        for i in range(min(len(GRADE_ORDER), num_classes_actual)):
            grade = GRADE_ORDER[i]
            prec = metrics["per_class_precision"][i]
            rec = metrics["per_class_recall"][i]
            f1 = metrics["per_class_f1"][i]
            print(f"{grade:<8} {prec:<12.4f} {rec:<12.4f} {f1:<12.4f}")

        mlflow.log_metrics(
            {
                "test_loss": avg_loss,
                "exact_accuracy": metrics["exact_accuracy"],
                "within_1_accuracy": metrics["within_1_accuracy"],
                "within_2_accuracy": metrics["within_2_accuracy"],
                "within_3_accuracy": metrics["within_3_accuracy"],
                "within_4_accuracy": metrics["within_4_accuracy"],
            }
        )

        for i in range(min(len(GRADE_ORDER), num_classes_actual)):
            grade = GRADE_ORDER[i]
            mlflow.log_metrics(
                {
                    f"{grade}_precision": metrics["per_class_precision"][i],
                    f"{grade}_recall": metrics["per_class_recall"][i],
                    f"{grade}_f1": metrics["per_class_f1"][i],
                }
            )

        conf_matrix = metrics["confusion_matrix"]
        np.save("confusion_matrix.npy", conf_matrix)
        mlflow.log_artifact("confusion_matrix.npy")

        print(f"\nRun ID: {run.info.run_id}")


if __name__ == "__main__":
    main()
