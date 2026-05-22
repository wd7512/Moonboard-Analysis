import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless-friendly
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from typing import Any

from moonboard_analysis.config import AutoencoderConfig
from moonboard_analysis.models.autoencoder import Autoencoder
from moonboard_analysis.training.metrics import evaluate_reconstruction
from moonboard_analysis.training.trainer import train_autoencoder
from moonboard_analysis.utils.device import get_device
from moonboard_analysis.utils.reproducibility import set_seeds


# ---------------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------------

def load_feature_matrix(data_path: str) -> np.ndarray:
    """Return float32 array of shape (n_samples, n_features).

    The legacy .npy files store rows as ``[grade, vector]`` pairs with dtype
    *object*.  We detect that layout and stack just the vectors.
    """
    raw = np.load(data_path, allow_pickle=True)

    # Case A – normal numeric array, second column onward is the feature matrix
    if raw.dtype != object:
        return raw[:, 1:].astype(np.float32)

    # Case B – object array where each row is [grade (int), vector (array)]
    if len(raw.shape) == 2 and raw.shape[1] == 2:
        n = raw.shape[0]
        vectors = np.array([raw[i, 1] for i in range(n)], dtype=np.float32)
        return vectors

    raise ValueError(f"Unexpected data shape/dtype: shape={raw.shape}, dtype={raw.dtype}")


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare autoencoder vs PCA reconstruction quality across bottleneck dims"
    )
    parser.add_argument("--data-path", type=str, default=None,
                        help="Path to .npy feature file")
    parser.add_argument("--dims", type=int, nargs="+",
                        default=[2, 4, 8, 16, 32, 64],
                        help="Bottleneck / PCA component sizes to sweep")
    parser.add_argument("--epochs", type=int, default=100,
                        help="Autoencoder training epochs per dim")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Batch size")
    parser.add_argument("--learning-rate", type=float, default=0.001,
                        help="Learning rate")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--plot-path", type=str,
                        default="models/comparison_sweep.png",
                        help="Output path for the comparison chart")
    parser.add_argument("--no-plot", action="store_true",
                        help="Skip generating the plot")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Core evaluation helpers
# ---------------------------------------------------------------------------

def evaluate_pca(
    train_features: np.ndarray,
    test_features:  np.ndarray,
    n_components: int,
) -> dict[str, float]:
    pca = PCA(n_components=n_components)
    pca.fit(train_features)
    reconstructed = pca.inverse_transform(pca.transform(test_features))

    test_tensor  = torch.tensor(test_features,  dtype=torch.float32)
    recon_tensor = torch.tensor(reconstructed, dtype=torch.float32)

    mse = torch.nn.MSELoss()(recon_tensor, test_tensor).item()
    binary_original      = (test_tensor > 0.5).float()
    binary_reconstructed = (recon_tensor > 0.5).float()
    binary_accuracy = (binary_original == binary_reconstructed).float().mean().item()
    exact_match     = (binary_original == binary_reconstructed).all(dim=1).float().mean().item()

    return {"mse": mse, "binary_accuracy": binary_accuracy, "exact_match": exact_match}


def train_and_evaluate_autoencoder(
    train_features: np.ndarray,
    test_features:  np.ndarray,
    bottleneck_dim: int,
    epochs:         int,
    batch_size:     int,
    learning_rate:  float,
    seed:           int,
) -> dict[str, float]:
    config = AutoencoderConfig(
        input_dim      = train_features.shape[1],
        bottleneck_dim = bottleneck_dim,
        epochs         = epochs,
        batch_size     = batch_size,
        learning_rate  = learning_rate,
        seed           = seed,
    )
    device = get_device()
    train_tensor = torch.tensor(train_features, dtype=torch.float32)
    test_tensor  = torch.tensor(test_features,  dtype=torch.float32)
    model, device = train_autoencoder(train_tensor, test_tensor, config, device)
    return evaluate_reconstruction(model, test_features, device)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def make_plot(
    dims:        list[int],
    pca_results: list[dict[str, float]],
    ae_results:  list[dict[str, float]],
    plot_path:   str,
) -> None:
    pca_bin   = [r["binary_accuracy"] for r in pca_results]
    ae_bin    = [r["binary_accuracy"] for r in ae_results]
    pca_exact = [r["exact_match"]      for r in pca_results]
    ae_exact  = [r["exact_match"]      for r in ae_results]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True)
    fig.suptitle("PCA vs Autoencoder: Reconstruction by Bottleneck Dimension",
                 fontsize=13)

    for ax, pca_vals, ae_vals, title in [
        (axes[0], pca_bin,   ae_bin,   "Binary Accuracy  (thresh 0.5)"),
        (axes[1], pca_exact, ae_exact, "Exact Match Rate"),
    ]:
        ax.plot(dims, pca_vals, "o-", color="#4C72B0", label="PCA",
                linewidth=2, markersize=7)
        ax.plot(dims, ae_vals,  "s-", color="#DD8452", label="Autoencoder",
                linewidth=2, markersize=7)
        ax.set_xscale("log", base=2)
        ax.set_xticks(dims)
        ax.set_xticklabels([str(d) for d in dims])
        ax.set_title(title)
        ax.set_ylim(0.0, 1.05)
        ax.legend()
        ax.grid(True, linestyle="--", alpha=0.5)

    axes[0].set_ylabel("Binary Accuracy")
    axes[1].set_ylabel("Exact Match Rate")
    for ax in axes:
        ax.set_xlabel("Bottleneck Dimension (log₂)")

    plt.tight_layout()
    Path(plot_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n📊  Plot saved → {plot_path}")


# ---------------------------------------------------------------------------
# Main sweep
# ---------------------------------------------------------------------------

def main() -> None:
    args   = parse_args()
    set_seeds(args.seed)

    data_path = args.data_path or "archive/Legacy/2016TrainingData164.npy"
    if not Path(data_path).exists():
        print(f"Error: Data file not found at '{data_path}'")
        print("Provide a valid path with --data-path")
        sys.exit(1)

    print(f"Loading data from {data_path}")
    features = load_feature_matrix(data_path)
    print(f"Data shape: {features.shape}")

    train_features, test_features = train_test_split(
        features, test_size=0.2, random_state=args.seed
    )

    mlflow.set_experiment("Autoencoder vs PCA Comparison")
    all_results: list[dict[str, Any]] = []

    with mlflow.start_run() as run:
        mlflow.log_params(
            {
                "data_path":    data_path,
                "dims":         str(args.dims),
                "ae_epochs":    args.epochs,
                "ae_batch_size": args.batch_size,
                "ae_lr":        args.learning_rate,
                "seed":         args.seed,
                "n_samples":    len(features),
                "n_features":   features.shape[1],
                "mode":         "sweep",
            }
        )

        for dim in args.dims:
            print(f"\n{'='*60}")
            print(f"  Bottleneck dimension: {dim}")
            print(f"{'='*60}")

            # ── PCA ──────────────────────────────────────────────────────────
            print("[PCA]")
            pca_res = evaluate_pca(train_features, test_features, dim)
            for k, v in pca_res.items():
                mlflow.log_metric(f"pca_{k}", v, step=dim)
            print(f"  MSE:           {pca_res['mse']:.6f}")
            print(f"  Binary Acc:    {pca_res['binary_accuracy']:.4f}")
            print(f"  Exact Match:   {pca_res['exact_match']:.4f}")

            # ── Autoencoder ─────────────────────────────────────────────────
            print(f"[Autoencoder] training {args.epochs} epochs …")
            ae_res = train_and_evaluate_autoencoder(
                train_features, test_features,
                dim, args.epochs, args.batch_size, args.learning_rate, args.seed,
            )
            for k, v in ae_res.items():
                mlflow.log_metric(f"ae_{k}", v, step=dim)
            print(f"  MSE:           {ae_res['mse']:.6f}")
            print(f"  Binary Acc:    {ae_res['binary_accuracy']:.4f}")
            print(f"  Exact Match:   {ae_res['exact_match']:.4f}")

            all_results.append({"dim": dim, "pca": pca_res, "autoenc": ae_res})

        # ── Summary table ───────────────────────────────────────────────────
        print("\n\n=== SWEEP RESULTS SUMMARY ===")
        print(f"{'Dim':>6}  {'PCA BinAcc':>10}  {'AE BinAcc':>10}  "
              f"{'PCA ExMatch':>11}  {'AE ExMatch':>11}")
        print("-" * 60)
        for row in all_results:
            print(
                f"{row['dim']:>6}  "
                f"{row['pca']['binary_accuracy']:>10.4f}  "
                f"{row['autoenc']['binary_accuracy']:>10.4f}  "
                f"{row['pca']['exact_match']:>11.4f}  "
                f"{row['autoenc']['exact_match']:>11.4f}"
            )

        # ── Plot ────────────────────────────────────────────────────────────
        if not args.no_plot:
            dims  = [r["dim"]     for r in all_results]
            pca_r = [r["pca"]     for r in all_results]
            ae_r  = [r["autoenc"] for r in all_results]
            make_plot(dims, pca_r, ae_r, args.plot_path)

    print(f"\nRun ID: {run.info.run_id}")


if __name__ == "__main__":
    main()
