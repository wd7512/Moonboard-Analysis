import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless-friendly
from typing import Any

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold

from moonboard_analysis.config import AutoencoderConfig
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
    parser.add_argument("--folds", type=int, default=5,
                        help="Number of cross-validation folds (default: 5)")
    parser.add_argument("--plot-path", type=str,
                        default="models/comparison_sweep.png",
                        help="Output path for the comparison chart")
    parser.add_argument("--cv-plot-path", type=str,
                        default="models/comparison_sweep_cv.png",
                        help="Output path for the cross-validation plot")
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
# Aggregation helper
# ---------------------------------------------------------------------------

def agg(series: list[float]) -> dict[str, Any]:
    """Compute mean, std, n, and raw values from a list of fold scores."""
    arr = np.array(series)
    return {
        "mean":   arr.mean().item(),
        "std":    arr.std().item(),
        "n":      len(series),
        "values": series,
    }


def _clean(d: dict[str, Any]) -> dict[str, Any]:
    """Recursively convert numpy scalars to plain Python types for JSON."""
    out: dict[str, Any] = {}
    for k2, v2 in d.items():
        if isinstance(v2, dict):
            out[k2] = {
                k3: float(v3)
                if isinstance(v3, (np.floating, np.integer))
                else v3
                for k3, v3 in v2.items()
            }
        else:
            out[k2] = v2
    return out


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


def make_cv_plot(
    dims:          list[int],
    cv_results:    list[dict[str, Any]],
    cv_plot_path:  str,
    n_folds:       int,
) -> None:
    """Plot CV means with ±1-std shaded bands for binary accuracy and exact match."""
    # Build per-metric arrays shaped (n_dims, n_folds)
    metric_keys = ["binary_accuracy", "exact_match"]
    titles      = ["Binary Accuracy  (thresh 0.5)", "Exact Match Rate"]

    fig, axes = plt.subplots(1, len(metric_keys), figsize=(12, 5), sharex=True)
    fig.suptitle(
        f"PCA vs Autoencoder: {n_folds}-Fold CV Results (mean ± 1 std)",
        fontsize=13,
    )

    for ax, key, title in zip(axes, metric_keys, titles):
        pca_means  = [r["pca"][key]["mean"]  for r in cv_results]
        pca_stds   = [r["pca"][key]["std"]   for r in cv_results]
        ae_means   = [r["autoenc"][key]["mean"] for r in cv_results]
        ae_stds   = [r["autoenc"][key]["std"]  for r in cv_results]

        # Plot means
        ax.plot(dims, pca_means, "o-", color="#4C72B0", label="PCA",
                linewidth=2, markersize=7)
        ax.plot(dims, ae_means,  "s-", color="#DD8452", label="Autoencoder",
                linewidth=2, markersize=7)

        # Shade ±1 std band
        ax.fill_between(dims,
                        [m - s for m, s in zip(pca_means, pca_stds)],
                        [m + s for m, s in zip(pca_means, pca_stds)],
                        color="#4C72B0", alpha=0.15)
        ax.fill_between(dims,
                        [m - s for m, s in zip(ae_means, ae_stds)],
                        [m + s for m, s in zip(ae_means, ae_stds)],
                        color="#DD8452", alpha=0.15)

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
    Path(cv_plot_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(cv_plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n📊  CV plot saved → {cv_plot_path}")


# ---------------------------------------------------------------------------
# Main sweep with KFold cross-validation
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

    mlflow.set_experiment("Autoencoder vs PCA Comparison")

    # ── KFold setup ─────────────────────────────────────────────────────────
    kf = KFold(n_splits=args.folds, shuffle=True, random_state=args.seed)

    # fold_metrics[dim] = {"pca": [fold_result, ...], "autoenc": [fold_result, ...]}
    fold_metrics: dict[int, dict[str, list[dict[str, float]]]] = {}
    # agg_metrics[dim] = {"pca": {metric: {mean, std, n, values}}, "autoenc": ...}
    agg_metrics:  dict[int, dict[str, dict[str, Any]]]          = {}

    with mlflow.start_run() as run:
        mlflow.log_params(
            {
                "data_path":        data_path,
                "dims":             str(args.dims),
                "ae_epochs":        args.epochs,
                "ae_batch_size":    args.batch_size,
                "ae_lr":            args.learning_rate,
                "seed":             args.seed,
                "n_samples":        len(features),
                "n_features":       features.shape[1],
                "mode":             "cv_sweep",
                "n_folds":          args.folds,
            }
        )

        for dim in args.dims:
            print(f"\n{'='*60}")
            print(f"  Bottleneck dimension: {dim}  ({args.folds} folds)")
            print(f"{'='*60}")

            fold_pca:   list[dict[str, float]] = []
            fold_autoenc: list[dict[str, float]] = []

            for fold_idx, (train_idx, test_idx) in enumerate(kf.split(features)):
                train_fold = features[train_idx]
                test_fold  = features[test_idx]
                print(
                    f"  Fold {fold_idx + 1}/{args.folds}  "
                    f"(train={len(train_fold)}, test={len(test_fold)})"
                )

                # ── PCA for this fold ─────────────────────────────────────────
                pca_res = evaluate_pca(train_fold, test_fold, dim)
                fold_pca.append(pca_res)

                # ── Autoencoder for this fold ─────────────────────────────────
                ae_res = train_and_evaluate_autoencoder(
                    train_fold, test_fold,
                    dim, args.epochs, args.batch_size, args.learning_rate, args.seed,
                )
                fold_autoenc.append(ae_res)

            fold_metrics[dim] = {"pca": fold_pca, "autoenc": fold_autoenc}

            # ── Compute normalisation (mean per-feature variance across all folds) ─
            # Test-fold variance acts as the input variance proxy for rel_mse.
            fold_test_vars: list[float] = []
            for _, test_idx in kf.split(features):
                fold_test_vars.append(float(np.var(features[test_idx], axis=0).mean()))
            input_variance = float(np.mean(fold_test_vars))

            # ── Aggregate across folds ───────────────────────────────────────
            pca_agg = {k: agg([f[k] for f in fold_pca])   for k in fold_pca[0]}
            ae_agg  = {k: agg([f[k] for f in fold_autoenc]) for k in fold_autoenc[0]}

            # recon_loss (alias for mse for readability) ─────────────────────
            pca_agg["recon_loss"] = pca_agg["mse"]
            ae_agg["recon_loss"]  = ae_agg["mse"]

            # rel_mse = mse / mean_per_feature_input_variance ───────────────
            pca_agg["rel_mse"] = {"mean": pca_agg["mse"]["mean"] / input_variance,
                                  "std":   0.0,
                                  "n":     1,
                                  "values": [pca_agg["mse"]["mean"] / input_variance]}
            ae_agg["rel_mse"]  = {"mean": ae_agg["mse"]["mean"] /  input_variance,
                                  "std":   0.0,
                                  "n":     1,
                                  "values": [ae_agg["mse"]["mean"] /  input_variance]}

            agg_metrics[dim] = {"pca": pca_agg, "autoenc": ae_agg}

            # Log CV summary metrics (mean only) to MLflow
            for k, v in pca_agg.items():
                mlflow.log_metric(f"pca_cv_{k}_mean", v["mean"], step=dim)
            for k, v in ae_agg.items():
                mlflow.log_metric(f"ae_cv_{k}_mean",  v["mean"], step=dim)

            print(f"\n  CV summary (dim={dim}):")
            print(f"  {'PCA BinAcc':>12}: "
                  f"{pca_agg['binary_accuracy']['mean']:.4f}"
                  f" ± {pca_agg['binary_accuracy']['std']:.4f}")
            print(f"  {'AE  BinAcc':>12}: "
                  f"{ae_agg['binary_accuracy']['mean']:.4f}"
                  f" ± {ae_agg['binary_accuracy']['std']:.4f}")
            print(f"  {'PCA ExMatch':>12}: "
                  f"{pca_agg['exact_match']['mean']:.4f}"
                  f" ± {pca_agg['exact_match']['std']:.4f}")
            print(f"  {'AE  ExMatch':>12}: "
                  f"{ae_agg['exact_match']['mean']:.4f}"
                  f" ± {ae_agg['exact_match']['std']:.4f}")

        # ── Cross-validation summary table ───────────────────────────────────
        print("\n\n=== CROSS-VALIDATION RESULTS (mean ± std) ===")
        print(f"{'Dim':>6}  "
              f"{'PCA BinAcc':>15}  {'AE BinAcc':>15}  "
              f"{'PCA ExMatch':>15}  {'AE ExMatch':>15}")
        print("-" * 78)

        all_cv_results: list[dict[str, Any]] = []
        for dim in args.dims:
            fm = fold_metrics[dim]
            pca_agg = {k: agg([f[k] for f in fm["pca"]])   for k in fm["pca"][0]}
            ae_agg  = {k: agg([f[k] for f in fm["autoenc"]]) for k in fm["autoenc"][0]}
            all_cv_results.append({"dim": dim, "pca": pca_agg, "autoenc": ae_agg})

            print(
                f"{dim:>6}  "
                f"{pca_agg['binary_accuracy']['mean']:>8.4f}"
                f"±{pca_agg['binary_accuracy']['std']:<5.4f}  "
                f"{ae_agg['binary_accuracy']['mean']:>8.4f}"
                f"±{ae_agg['binary_accuracy']['std']:<5.4f}  "
                f"{pca_agg['exact_match']['mean']:>8.4f}"
                f"±{pca_agg['exact_match']['std']:<5.4f}  "
                f"{ae_agg['exact_match']['mean']:>8.4f}"
                f"±{ae_agg['exact_match']['std']:<5.4f}"
            )

        # ── CV plot ──────────────────────────────────────────────────────────
        if not args.no_plot:
            make_cv_plot(args.dims, all_cv_results, args.cv_plot_path, args.folds)

        # ── Save CV results JSON ─────────────────────────────────────────────
        json_path = "models/sweep_cv_results.json"
        Path(json_path).parent.mkdir(parents=True, exist_ok=True)
        # Make JSON-serialisable: convert Any dicts losing Non-float values
        json_out = {
            "dims": args.dims,
            "results": [],
            "meta": {
                "script":        "compare_pca.py",
                "data_path":     data_path,
                "n_folds":       args.folds,
                "ae_epochs":     args.epochs,
                "ae_batch_size": args.batch_size,
                "ae_learning_rate": args.learning_rate,
                "seed":          args.seed,
                "n_samples":     len(features),
                "n_features":    features.shape[1],
                "mode":          "cv_sweep",
                "metrics":       ["binary_accuracy", "exact_match", "mse", "recon_loss", "rel_mse"],
            },
        }
        for dim in args.dims:
            pa = agg_metrics[dim]["pca"]
            aa = agg_metrics[dim]["autoenc"]
            json_out["results"].append(
                {"dim": dim, "pca": _clean(pa), "autoenc": _clean(aa)}
            )

        with open(json_path, "w") as f:
            json.dump(json_out, f, indent=2)
        print(f"\n💾  CV results saved → {json_path}")

    print(f"\nRun ID: {run.info.run_id}")


if __name__ == "__main__":
    main()
