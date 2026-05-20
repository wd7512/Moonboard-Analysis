import argparse
import sys
from pathlib import Path

import mlflow
import numpy as np
import torch
from sklearn.model_selection import train_test_split

from moonboard_analysis.config import AutoencoderConfig
from moonboard_analysis.training.metrics import evaluate_reconstruction
from moonboard_analysis.training.trainer import train_autoencoder
from moonboard_analysis.utils.device import get_device
from moonboard_analysis.utils.reproducibility import set_seeds


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train autoencoder on Moonboard hold data")
    parser.add_argument("--data-path", type=str, default=None, help="Path to .npy feature file")
    parser.add_argument("--output-dir", type=str, default="models", help="Directory to save model")
    parser.add_argument("--bottleneck-dim", type=int, default=8, help="Bottleneck dimension")
    parser.add_argument("--hidden-dim", type=int, default=64, help="Hidden layer dimension")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--learning-rate", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-5, help="Weight decay")
    parser.add_argument("--test-size", type=float, default=0.2, help="Fraction for test split")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seeds(args.seed)

    data_path = args.data_path or "archive/Legacy/2016TrainingData164.npy"
    if not Path(data_path).exists():
        print(f"Error: Data file not found at '{data_path}'")
        print("Please provide a valid path with --data-path")
        sys.exit(1)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading data from {data_path}")
    data = np.load(data_path, allow_pickle=True)
    grades = np.array([row[0] for row in data], dtype=float)
    features = np.stack([row[1] for row in data]).astype(np.float32)
    print(f"Data shape: {features.shape}")

    train_features, test_features, train_grades, test_grades = train_test_split(
        features, grades, test_size=args.test_size, random_state=args.seed
    )

    config = AutoencoderConfig(
        input_dim=features.shape[1],
        bottleneck_dim=args.bottleneck_dim,
        hidden_dim=args.hidden_dim,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        seed=args.seed,
        bounded=True,
        data_path=data_path,
        model_save_path=str(output_dir / "Autoencoder_Moonboard.pth"),
    )

    device = get_device()
    print(f"Training on device: {device}")

    mlflow.set_experiment("Autoencoder Training")

    with mlflow.start_run() as run:
        mlflow.log_params(
            {
                "input_dim": config.input_dim,
                "bottleneck_dim": config.bottleneck_dim,
                "hidden_dim": config.hidden_dim,
                "epochs": config.epochs,
                "batch_size": config.batch_size,
                "learning_rate": config.learning_rate,
                "weight_decay": config.weight_decay,
                "seed": config.seed,
                "train_size": len(train_features),
                "test_size": len(test_features),
            }
        )

        train_tensor = torch.tensor(train_features, dtype=torch.float32)
        test_tensor = torch.tensor(test_features, dtype=torch.float32)

        model, device = train_autoencoder(train_tensor, test_tensor, config, device)

        train_metrics = evaluate_reconstruction(model, train_features, device)
        test_metrics = evaluate_reconstruction(model, test_features, device)

        mlflow.log_metrics(
            {
                "train_mse": train_metrics["mse"],
                "train_binary_accuracy": train_metrics["binary_accuracy"],
                "train_exact_match": train_metrics["exact_match"],
                "test_mse": test_metrics["mse"],
                "test_binary_accuracy": test_metrics["binary_accuracy"],
                "test_exact_match": test_metrics["exact_match"],
            }
        )

        model_path = config.model_save_path
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "config": {
                    "input_dim": config.input_dim,
                    "bottleneck_dim": config.bottleneck_dim,
                    "hidden_dim": config.hidden_dim,
                    "bounded": config.bounded,
                },
            },
            model_path,
        )
        mlflow.log_artifact(model_path)

        print(f"\nRun ID: {run.info.run_id}")
        print(f"Model saved to: {model_path}")
        print(f"Test MSE: {test_metrics['mse']:.6f}")
        print(f"Test Binary Accuracy: {test_metrics['binary_accuracy']:.4f}")
        print(f"Test Exact Match: {test_metrics['exact_match']:.4f}")


if __name__ == "__main__":
    main()
