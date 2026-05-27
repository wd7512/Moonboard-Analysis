import numpy as np
import torch
import torch.nn as nn

from moonboard_analysis.models.autoencoder import Autoencoder


def evaluate_reconstruction(
    model: Autoencoder, features: np.ndarray, device: torch.device
) -> dict[str, float]:
    """Evaluate autoencoder reconstruction quality.

    Returns dict with keys: mse, binary_accuracy, exact_match.
    """
    model.eval()
    with torch.no_grad():
        features_tensor = torch.tensor(features, dtype=torch.float32).to(device)
        reconstructed = model(features_tensor)

        mse = nn.MSELoss()(reconstructed, features_tensor).item()

        binary_original = (features_tensor > 0.5).float()
        binary_reconstructed = (reconstructed > 0.5).float()
        binary_accuracy = (
            (binary_original == binary_reconstructed).float().mean().item()
        )

        exact_match = (
            (binary_original == binary_reconstructed).all(dim=1).float().mean().item()
        )

    return {
        "mse": mse,
        "binary_accuracy": binary_accuracy,
        "exact_match": exact_match,
    }


def evaluate_classification(
    y_true: list[int], y_pred: list[int], num_classes: int
) -> dict:
    """Compute per-class and overall classification metrics."""
    from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

    conf_matrix = confusion_matrix(y_true, y_pred, labels=range(num_classes))
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=range(num_classes), zero_division=0
    )

    within_1 = _accuracy_within_diagonal(conf_matrix, width=1)
    within_2 = _accuracy_within_diagonal(conf_matrix, width=2)
    within_3 = _accuracy_within_diagonal(conf_matrix, width=3)
    within_4 = _accuracy_within_diagonal(conf_matrix, width=4)

    total_correct = sum(conf_matrix[i][i] for i in range(num_classes))
    exact_accuracy = total_correct / conf_matrix.sum()

    return {
        "confusion_matrix": conf_matrix,
        "exact_accuracy": float(exact_accuracy),
        "within_1_accuracy": float(within_1),
        "within_2_accuracy": float(within_2),
        "within_3_accuracy": float(within_3),
        "within_4_accuracy": float(within_4),
        "per_class_precision": precision.tolist(),
        "per_class_recall": recall.tolist(),
        "per_class_f1": f1.tolist(),
    }


def _accuracy_within_diagonal(conf_matrix: np.ndarray, width: int) -> float:
    n = conf_matrix.shape[0]
    total_correct = 0
    for i in range(n):
        for j in range(max(0, i - width + 1), min(n, i + width)):
            total_correct += conf_matrix[i, j]
    return total_correct / conf_matrix.sum()
