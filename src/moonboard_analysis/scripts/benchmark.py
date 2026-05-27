"""CLI script for benchmarking trained LSTM models.

This script loads a trained LSTM checkpoint, evaluates it on test data,
computes comprehensive metrics, and generates results in JSON and Markdown format.

Usage:
    moonboard-benchmark --model-path models/LSTM_Moonboard.pth \\
        --data-path Raw/moonboard_problems_setup_2016.json \\
        --output-json results.json \\
        --output-markdown results.md
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import mlflow
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
    """Parse command-line arguments for benchmark script.

    Returns:
        Parsed arguments with model_path, data_path, output_json, output_markdown.
    """
    parser = argparse.ArgumentParser(
        description="Benchmark trained LSTM model and generate results"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="models/LSTM_Moonboard.pth",
        help="Path to trained LSTM checkpoint",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default=None,
        help=(
            "Path to raw JSON data file (optional, defaults to "
            "Raw/moonboard_problems_setup_2016.json"
        ),
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default="results.json",
        help="Path to write results JSON file",
    )
    parser.add_argument(
        "--output-markdown",
        type=str,
        default="results.md",
        help="Path to write results Markdown file",
    )
    parser.add_argument(
        "--batch-size", type=int, default=32, help="Batch size for evaluation"
    )
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
    """Load trained LSTM model and vocabulary from checkpoint.

    Args:
        model_path: Path to the saved model checkpoint.
        device: torch device for loading.

    Returns:
        Tuple of (model, vocab, config).

    Raises:
        SystemExit: If model file not found.
    """
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


def format_leaderboard_summary(metrics: dict, grade_order: list[str]) -> str:
    """Format a leaderboard-style summary of metrics.

    Args:
        metrics: Dictionary of computed metrics.
        grade_order: List of grade names in order.

    Returns:
        Formatted string for console display.
    """
    lines = []
    lines.append("\n" + "=" * 60)
    lines.append("BENCHMARK RESULTS SUMMARY")
    lines.append("=" * 60)

    lines.append("\nOverall Accuracy Metrics:")
    lines.append(f"  Exact Match:       {metrics['exact_accuracy']:.4f}")
    lines.append(f"  Within ±1 Grade:   {metrics['within_1_accuracy']:.4f}")
    lines.append(f"  Within ±2 Grades:  {metrics['within_2_accuracy']:.4f}")
    lines.append(f"  Within ±3 Grades:  {metrics['within_3_accuracy']:.4f}")
    lines.append(f"  Within ±4 Grades:  {metrics['within_4_accuracy']:.4f}")

    lines.append("\nPer-Grade Performance:")
    lines.append(f"{'Grade':<8} {'Precision':<12} {'Recall':<12} {'F1':<12}")
    lines.append("-" * 44)

    num_actual = len(metrics["per_class_precision"])
    for i, grade in enumerate(grade_order):
        if i >= num_actual:
            break
        prec = metrics["per_class_precision"][i]
        rec = metrics["per_class_recall"][i]
        f1 = metrics["per_class_f1"][i]
        lines.append(f"{grade:<8} {prec:<12.4f} {rec:<12.4f} {f1:<12.4f}")

    lines.append("=" * 60)
    return "\n".join(lines)


def generate_markdown_report(
    results: dict,
    model_path: str,
    data_path: str,
    metrics: dict,
    num_classes: int,
) -> str:
    """Generate a comprehensive Markdown report of benchmark results.

    Args:
        results: Dictionary with loss, metrics, summary from benchmark.
        model_path: Path to evaluated model.
        data_path: Path to evaluation data.
        metrics: Dictionary of computed metrics.
        num_classes: Number of classes in the model.

    Returns:
        Formatted Markdown string.
    """
    lines = []
    lines.append("# Moonboard LSTM Benchmark Results")
    lines.append("")

    lines.append("## Metadata")
    lines.append(f"- **Model**: {model_path}")
    lines.append(f"- **Data**: {data_path}")
    lines.append(f"- **Timestamp**: {datetime.now().isoformat()}")
    lines.append("")

    lines.append("## Overall Metrics")
    lines.append(
        "| Metric | Value |"
    )
    lines.append("|--------|-------|")
    lines.append(f"| Test Loss | {results['loss']:.6f} |")
    lines.append(
        f"| Exact Accuracy | {metrics['exact_accuracy']:.4f} |"
    )
    lines.append(
        f"| Within ±1 Grade | {metrics['within_1_accuracy']:.4f} |"
    )
    lines.append(
        f"| Within ±2 Grades | {metrics['within_2_accuracy']:.4f} |"
    )
    lines.append(
        f"| Within ±3 Grades | {metrics['within_3_accuracy']:.4f} |"
    )
    lines.append(
        f"| Within ±4 Grades | {metrics['within_4_accuracy']:.4f} |"
    )
    lines.append("")

    lines.append("## Per-Grade Performance")
    lines.append("| Grade | Precision | Recall | F1 |")
    lines.append("|-------|-----------|--------|-----|")

    num_actual = len(metrics["per_class_precision"])
    grade_order = GRADE_ORDER[:num_classes]
    for i in range(min(num_classes, num_actual)):
        grade = grade_order[i] if i < len(grade_order) else f"Class {i}"
        prec = metrics["per_class_precision"][i]
        rec = metrics["per_class_recall"][i]
        f1 = metrics["per_class_f1"][i]
        lines.append(f"| {grade} | {prec:.4f} | {rec:.4f} | {f1:.4f} |")
    lines.append("")

    return "\n".join(lines)


def main() -> None:
    """Main entry point for benchmark CLI.

    Loads model, evaluates on test data, computes metrics, writes results.
    """
    args = parse_args()
    set_seeds(args.seed)

    # Determine data path
    data_path = args.data_path or "Raw/moonboard_problems_setup_2016.json"
    if not Path(data_path).exists():
        print(f"Error: Data file not found at '{data_path}'")
        print("Please provide a valid path with --data-path")
        sys.exit(1)

    device = get_device()

    # Load and preprocess data FIRST to build vocab
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

    # Encode grades BEFORE split (so split uses encoded labels)
    grade_to_idx = {g: i for i, g in enumerate(GRADE_ORDER)}
    encoded_grades = [grade_to_idx[g] for g in route_grades]

    # Split FIRST (before vocab building to prevent data leakage)
    train_seqs, test_seqs, train_grades, test_grades = train_test_split(
        route_sequences, encoded_grades, test_size=0.2, random_state=args.seed,
        stratify=encoded_grades,
    )

    # Build vocabulary from training sequences ONLY (no data leakage)
    vocab = build_vocab(train_seqs)
    print(f"Built vocabulary with {len(vocab)} tokens")

    # Now load model
    print(f"Loading model from {args.model_path}")
    model, loaded_vocab, model_config = load_model_and_vocab(args.model_path, device)
    print(f"Model loaded on device: {device}")

    # Use model's vocab if available, otherwise use built vocab
    if loaded_vocab:
        vocab = loaded_vocab

    max_length = model_config["max_length"]
    num_classes = model_config["num_classes"]

    test_dataset = LSTMSequenceDataset(test_seqs, test_grades, vocab, max_length)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)

    criterion = nn.CrossEntropyLoss()

    mlflow.set_experiment("LSTM Grade Prediction Benchmark")

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

        # Run benchmark
        print("\nRunning benchmark evaluation...")
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

        # Build results
        results = {
            "loss": float(avg_loss),
            "metrics": {
                "test_loss": float(avg_loss),
                "exact_accuracy": float(metrics["exact_accuracy"]),
                "within_1_accuracy": float(metrics["within_1_accuracy"]),
                "within_2_accuracy": float(metrics["within_2_accuracy"]),
                "within_3_accuracy": float(metrics["within_3_accuracy"]),
                "within_4_accuracy": float(metrics["within_4_accuracy"]),
                "per_class_precision": [float(x) for x in metrics["per_class_precision"]],
                "per_class_recall": [float(x) for x in metrics["per_class_recall"]],
                "per_class_f1": [float(x) for x in metrics["per_class_f1"]],
                "confusion_matrix": metrics["confusion_matrix"].tolist(),
            },
            "summary": {
                "test_loss": float(avg_loss),
                "exact_accuracy": float(metrics["exact_accuracy"]),
                "within_1_accuracy": float(metrics["within_1_accuracy"]),
                "within_2_accuracy": float(metrics["within_2_accuracy"]),
                "within_3_accuracy": float(metrics["within_3_accuracy"]),
                "within_4_accuracy": float(metrics["within_4_accuracy"]),
            },
        }

        # Write JSON results
        output_json = Path(args.output_json)
        output_json.parent.mkdir(parents=True, exist_ok=True)
        with open(output_json, "w") as f:
            json.dump(results, f, indent=2)
        print(f"✓ JSON results written to {output_json}")
        mlflow.log_artifact(str(output_json))

        # Write Markdown results
        output_markdown = Path(args.output_markdown)
        output_markdown.parent.mkdir(parents=True, exist_ok=True)
        markdown_report = generate_markdown_report(
            results, args.model_path, data_path, metrics, num_classes
        )
        with open(output_markdown, "w") as f:
            f.write(markdown_report)
        print(f"✓ Markdown report written to {output_markdown}")
        mlflow.log_artifact(str(output_markdown))

        # Log metrics to MLflow
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

        num_actual = len(metrics["per_class_precision"])
        for i, grade in enumerate(GRADE_ORDER[:num_classes]):
            if i >= num_actual:
                break
            safe_grade = grade.replace("+", "").replace("/", "")
            mlflow.log_metrics(
                {
                    f"grade_{safe_grade}_precision": metrics["per_class_precision"][i],
                    f"grade_{safe_grade}_recall": metrics["per_class_recall"][i],
                    f"grade_{safe_grade}_f1": metrics["per_class_f1"][i],
                }
            )

        # Print leaderboard summary to stdout
        print(format_leaderboard_summary(metrics, GRADE_ORDER[:num_classes]))

        print(f"\nRun ID: {run.info.run_id}")


if __name__ == "__main__":
    main()
