"""5-fold cross-validation benchmark harness for Moonboard climbing grade prediction.

This module implements a SOLID-based benchmark harness for evaluating
**pre-trained** LSTM models using multiple metrics across n-fold
cross-validation.

IMPORTANT: This harness is designed for PRE-TRAINED models only.  It evaluates
the same static model on each test fold **without** fold-wise retraining.  If
you need to train a fresh model per fold (e.g. to measure generalisation of a
training procedure), this harness is not appropriate.
"""

import json
import logging
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import KFold

logger = logging.getLogger(__name__)

# Module constants
DEFAULT_N_SPLITS = 5
DEFAULT_TEST_SIZE = 0.2
MIN_N_SPLITS = 1
MAX_N_SPLITS = 10


class MetricComputer(ABC):
    """Abstract base class for benchmark metrics (SOLID: Interface Segregation).

    Each metric should compute a single, well-defined evaluation criterion.
    """

    @abstractmethod
    def compute(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute the metric value.

        Args:
            y_true: Ground truth labels of shape (n_samples,)
            y_pred: Predicted labels of shape (n_samples,)

        Returns:
            Metric score as a float in [0, 1].
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the metric name identifier."""
        pass


class ExactAccuracy(MetricComputer):
    """Metric: exact match accuracy (grade match exactly)."""

    def compute(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute exact accuracy.

        Args:
            y_true: Ground truth labels
            y_pred: Predicted labels

        Returns:
            Proportion of exact matches in [0, 1]
        """
        if len(y_true) == 0:
            return 0.0
        matches = (y_true == y_pred).sum()
        return float(matches) / len(y_true)

    @property
    def name(self) -> str:
        """Return metric name."""
        return "exact_accuracy"


class WithinOneGrade(MetricComputer):
    """Metric: accuracy within one grade level."""

    def compute(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute within-one-grade accuracy.

        Args:
            y_true: Ground truth labels
            y_pred: Predicted labels

        Returns:
            Proportion of predictions within 1 grade level in [0, 1]
        """
        if len(y_true) == 0:
            return 0.0
        within_one = (np.abs(y_true - y_pred) <= 1).sum()
        return float(within_one) / len(y_true)

    @property
    def name(self) -> str:
        """Return metric name."""
        return "within_one_grade"


class WithinTwoGrades(MetricComputer):
    """Metric: accuracy within two grade levels."""

    def compute(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute within-two-grades accuracy.

        Args:
            y_true: Ground truth labels
            y_pred: Predicted labels

        Returns:
            Proportion of predictions within 2 grade levels in [0, 1]
        """
        if len(y_true) == 0:
            return 0.0
        within_two = (np.abs(y_true - y_pred) <= 2).sum()
        return float(within_two) / len(y_true)

    @property
    def name(self) -> str:
        """Return metric name."""
        return "within_two_grades"


@dataclass
class BenchmarkResults:
    """Results container for 5-fold CV benchmark (SOLID: Single Responsibility).

    Holds fold-wise results and provides aggregation + serialization.
    """

    fold_results: list[dict[str, float]]
    """List of metric dictionaries, one per fold."""

    def mean_scores(self) -> dict[str, float]:
        """Compute mean metric scores across folds.

        Returns:
            Dict mapping metric names to mean values.
        """
        if not self.fold_results:
            return {}

        metric_names = self.fold_results[0].keys()
        means = {}
        for metric_name in metric_names:
            values = [fold[metric_name] for fold in self.fold_results]
            means[metric_name] = float(np.mean(values))
        return means

    def std_scores(self) -> dict[str, float]:
        """Compute std dev of metric scores across folds.

        Returns:
            Dict mapping metric names to std dev values.
        """
        if not self.fold_results:
            return {}

        metric_names = self.fold_results[0].keys()
        stds = {}
        for metric_name in metric_names:
            values = [fold[metric_name] for fold in self.fold_results]
            stds[metric_name] = float(np.std(values))
        return stds

    def to_json(self) -> str:
        """Serialize results to JSON string.

        Returns:
            JSON string containing fold_results, mean_scores, and std_scores.
        """
        data = {
            "fold_results": self.fold_results,
            "mean_scores": self.mean_scores(),
            "std_scores": self.std_scores(),
        }
        return json.dumps(data, indent=2)

    def to_markdown_table(self) -> str:
        """Generate markdown table representation of results.

        Returns:
            Markdown-formatted table with folds, means, and stds.
        """
        if not self.fold_results:
            return ""

        metric_names = list(self.fold_results[0].keys())
        means = self.mean_scores()
        stds = self.std_scores()

        # Build header
        header = "| Fold | " + " | ".join(metric_names) + " |"
        separator = (
            "|" + "|".join(["-" * 6] * (len(metric_names) + 1)) + "|"
        )

        lines = [header, separator]

        # Add fold rows
        for fold_idx, fold_result in enumerate(self.fold_results):
            values = [f"{fold_result[m]:.4f}" for m in metric_names]
            row = f"| {fold_idx} | " + " | ".join(values) + " |"
            lines.append(row)

        # Add separator and stats rows
        lines.append(separator)
        mean_values = [f"{means[m]:.4f}" for m in metric_names]
        mean_row = "| Mean | " + " | ".join(mean_values) + " |"
        lines.append(mean_row)

        std_values = [f"{stds[m]:.4f}" for m in metric_names]
        std_row = "| Std | " + " | ".join(std_values) + " |"
        lines.append(std_row)

        return "\n".join(lines)


class BenchmarkHarness:
    """Orchestrates n-fold CV benchmarking for **pre-trained** models only.

    Accepts pluggable metrics (abstract MetricComputer), runs cross-validation
    by evaluating the same static model on each test fold, and aggregates
    results.

    .. caution::

       This harness does **not** retrain the model per fold.  It is intended
       solely for evaluating a **pre-trained** model's performance on different
       data splits.  For measuring the generalisation of a training procedure,
       you would need fold-wise retraining (not provided here).
    """

    def __init__(
        self,
        model: nn.Module,
        metrics: Sequence[MetricComputer],
        device: Any | None = None,
    ) -> None:
        """Initialize benchmark harness.

        Args:
            model: Pre-trained PyTorch model to benchmark.
            metrics: List of MetricComputer instances (pluggable, SOLID).
            device: Torch device (cpu/cuda). Auto-detected if None.
        """
        self.model = model
        self.metrics = metrics
        self.device = device or (
            torch.device("cuda") if torch.cuda.is_available()
            else torch.device("cpu")
        )

    @staticmethod
    def _param_mean_abs(model: nn.Module) -> float:
        """Compute mean absolute value across all model parameters."""
        total = 0.0
        count = 0
        for p in model.parameters():
            total += p.data.abs().sum().item()
            count += p.data.numel()
        return total / count if count > 0 else 0.0

    def _warn_if_untrained(self) -> None:
        """Log a warning when model weights resemble a random initialisation.

        Freshly initialised PyTorch linear layers (Kaiming Uniform) produce
        weights around ±0.09 for typical hidden sizes.  If the mean absolute
        weight is orders of magnitude smaller the model is probably untrained.
        """
        mean_abs = self._param_mean_abs(self.model)
        if mean_abs < 1e-6:
            logger.warning(
                "Model weights appear to be near-zero "
                "(mean |param| = %g). BenchmarkHarness is designed for "
                "PRE-TRAINED models only. If this model has not been "
                "trained, results will be meaningless.",
                mean_abs,
            )

    def run_benchmark(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        n_splits: int = DEFAULT_N_SPLITS,
    ) -> BenchmarkResults:
        """Run n-fold cross-validation benchmark on a **pre-trained** model.

        .. note::

           The model is **not** retrained per fold.  This method is suitable
           only for pre-trained models that you want to evaluate on multiple
           data splits.

        Args:
            features: Feature matrix of shape (n_samples, n_features).
            labels: Label vector of shape (n_samples,).
            n_splits: Number of CV folds (default: 5).

        Returns:
            BenchmarkResults containing per-fold metric scores.

        Raises:
            ValueError: If n_splits is not in range [1, 10].
        """
        self._warn_if_untrained()

        # Validate inputs
        if len(features) != len(labels):
            raise ValueError(
                f"Features and labels must have the same length, "
                f"got {len(features)} and {len(labels)}"
            )

        # Validate n_splits
        if not isinstance(n_splits, int):
            raise TypeError(f"n_splits must be an integer, got {type(n_splits).__name__}")
        if n_splits < MIN_N_SPLITS or n_splits > MAX_N_SPLITS:
            raise ValueError(
                f"n_splits must be between {MIN_N_SPLITS} and {MAX_N_SPLITS}, got {n_splits}"
            )

        kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        fold_results: list[dict[str, float]] = []

        for train_idx, test_idx in kfold.split(features):
            # Split data
            X_train, X_test = features[train_idx], features[test_idx]
            y_train, y_test = labels[train_idx], labels[test_idx]

            # Simple training loop (for mock model)
            fold_result = self._evaluate_fold(
                self.model, X_train, y_train, X_test, y_test
            )
            fold_results.append(fold_result)

        return BenchmarkResults(fold_results=fold_results)

    def _evaluate_fold(
        self,
        model: nn.Module,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
    ) -> dict[str, float]:
        """Evaluate model on a single fold and compute all metrics.

        Args:
            model: Model to evaluate.
            X_train: Training features.
            y_train: Training labels.
            X_test: Test features.
            y_test: Test labels.

        Returns:
            Dict mapping metric names to scores for this fold.
        """
        # Convert to tensors
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(
            self.device
        )

        # Forward pass
        model.to(self.device)
        model.eval()
        with torch.no_grad():
            outputs = model(X_test_tensor)
            y_pred = torch.argmax(outputs, dim=1).cpu().numpy()

        # Compute metrics
        fold_scores = {}
        for metric in self.metrics:
            score = metric.compute(y_test, y_pred)
            fold_scores[metric.name] = score

        return fold_scores
