import argparse
import sys
from pathlib import Path

import mlflow
import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

from moonboard_analysis.config import AutoencoderConfig
from moonboard_analysis.models.autoencoder import Autoencoder
from moonboard_analysis.training.metrics import evaluate_reconstruction
from moonboard_analysis.training.trainer import train_autoencoder
from moonboard_analysis.utils.reproducibility import set_seeds


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare autoencoder vs PCA reconstruction quality"
    )
    parser.add_argument("--data-path", type=str, default=None, help="Path to .npy feature file")
    parser.add_argument(
        "--ae-model-path", type=str, default=None, help="Path to trained autoencoder checkpoint"
    )
    parser.add_argument("--bottleneck-dim", type=int, default=8, help="Bottleneck / PCA components")
    parser.add_argument("--epochs", type=int, default=100, help="Autoencoder training epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--learning-rate", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


def train_and_evaluate_autoencoder(
    train_features: np.ndarray,
    test_features: np.ndarray,
    bottleneck_dim: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
) -> dict[str, float]:
    config = AutoencoderConfig(
        input_dim=train_features.shape[1],
        bottleneck_dim=bottleneck_dim,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        seed=seed,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_tensor = torch.tensor(train_features, dtype=torch.float32)
    test_tensor = torch.tensor(test_features, dtype=torch.float32)

    model, device = train_autoencoder(train_tensor, test_tensor, config, device)
    return evaluate_reconstruction(model, test_features, device)


def load_autoencoder(
    model_path: str, input_dim: int, bottleneck_dim: int, device: torch.device
) -> Autoencoder:
    if not Path(model_path).exists():
        print(f"Error: Autoencoder checkpoint not found at '{model_path}'")
        sys.exit(1)
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model = Autoencoder(input_dim, bottleneck_dim)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model


def evaluate_pca(
    train_features: np.ndarray, test_features: np.ndarray, n_components: int
) -> dict[str, float]:
    pca = PCA(n_components=n_components)
    pca.fit(train_features)
    reconstructed = pca.inverse_transform(pca.transform(test_features))

    test_tensor = torch.tensor(test_features, dtype=torch.float32)
    recon_tensor = torch.tensor(reconstructed, dtype=torch.float32)

    mse = torch.nn.MSELoss()(recon_tensor, test_tensor).item()
    binary_original = (test_tensor > 0.5).float()
    binary_reconstructed = (recon_tensor > 0.5).float()
    binary_accuracy = (binary_original == binary_reconstructed).float().mean().item()
    exact_match = (binary_original == binary_reconstructed).all(dim=1).float().mean().item()

    return {"mse": mse, "binary_accuracy": binary_accuracy, "exact_match": exact_match}


def main() -> None:
    args = parse_args()
    set_seeds(args.seed)

    data_path = args.data_path or "archive/Legacy/2016TrainingData164.npy"
    if not Path(data_path).exists():
        print(f"Error: Data file not found at '{data_path}'")
        print("Please provide a valid path with --data-path")
        sys.exit(1)

    print(f"Loading data from {data_path}")
    data = np.load(data_path, allow_pickle=True)
    features = data[:, 1:].astype(np.float32)
    print(f"Data shape: {features.shape}")

    train_features, test_features = train_test_split(
        features, test_size=0.2, random_state=args.seed
    )

    mlflow.set_experiment("Autoencoder vs PCA Comparison")

    with mlflow.start_run() as run:
        mlflow.log_params(
            {
                "data_path": data_path,
                "bottleneck_dim": args.bottleneck_dim,
                "ae_epochs": args.epochs,
                "ae_batch_size": args.batch_size,
                "ae_learning_rate": args.learning_rate,
                "seed": args.seed,
                "n_samples": len(features),
                "n_features": features.shape[1],
            }
        )

        print("\n--- Training Autoencoder ---")
        ae_metrics = train_and_evaluate_autoencoder(
            train_features,
            test_features,
            args.bottleneck_dim,
            args.epochs,
            args.batch_size,
            args.learning_rate,
            args.seed,
        )
        print(f"AE MSE: {ae_metrics['mse']:.6f}")
        print(f"AE Binary Accuracy: {ae_metrics['binary_accuracy']:.4f}")
        print(f"AE Exact Match: {ae_metrics['exact_match']:.4f}")

        print("\n--- Evaluating PCA ---")
        pca_metrics = evaluate_pca(train_features, test_features, args.bottleneck_dim)
        print(f"PCA MSE: {pca_metrics['mse']:.6f}")
        print(f"PCA Binary Accuracy: {pca_metrics['binary_accuracy']:.4f}")
        print(f"PCA Exact Match: {pca_metrics['exact_match']:.4f}")

        mlflow.log_metrics(
            {
                "ae_mse": ae_metrics["mse"],
                "ae_binary_accuracy": ae_metrics["binary_accuracy"],
                "ae_exact_match": ae_metrics["exact_match"],
                "pca_mse": pca_metrics["mse"],
                "pca_binary_accuracy": pca_metrics["binary_accuracy"],
                "pca_exact_match": pca_metrics["exact_match"],
                "mse_improvement": pca_metrics["mse"] - ae_metrics["mse"],
                "binary_accuracy_improvement": ae_metrics["binary_accuracy"]
                - pca_metrics["binary_accuracy"],
            }
        )

        print("\n--- Comparison Summary ---")
        print(f"MSE improvement (AE - PCA): {pca_metrics['mse'] - ae_metrics['mse']:.6f}")
        print(
            f"Binary accuracy improvement: "
            f"{ae_metrics['binary_accuracy'] - pca_metrics['binary_accuracy']:.4f}"
        )
        print(f"\nRun ID: {run.info.run_id}")


if __name__ == "__main__":
    main()
