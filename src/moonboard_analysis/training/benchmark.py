"""n-fold cross-validation benchmark harness for Moonboard climbing grade prediction.

This module implements a SOLID-based benchmark harness for evaluating
model training procedures via proper retrain-per-fold cross-validation.
"""

import json
import logging
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import torch
from sklearn.model_selection import KFold

logger = logging.getLogger(__name__)

DEFAULT_N_SPLITS = 5
DEFAULT_TEST_SIZE = 0.2
MIN_N_SPLITS = 1
MAX_N_SPLITS = 10


class MetricComputer(ABC):
    @abstractmethod
    def compute(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        pass


class ExactAccuracy(MetricComputer):
    def compute(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        if len(y_true) == 0:
            return 0.0
        matches = (y_true == y_pred).sum()
        return float(matches) / len(y_true)

    @property
    def name(self) -> str:
        return "exact_accuracy"


class WithinOneGrade(MetricComputer):
    def compute(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        if len(y_true) == 0:
            return 0.0
        within_one = (np.abs(y_true - y_pred) <= 1).sum()
        return float(within_one) / len(y_true)

    @property
    def name(self) -> str:
        return "within_one_grade"


class WithinTwoGrades(MetricComputer):
    def compute(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        if len(y_true) == 0:
            return 0.0
        within_two = (np.abs(y_true - y_pred) <= 2).sum()
        return float(within_two) / len(y_true)

    @property
    def name(self) -> str:
        return "within_two_grades"


REQUIRED_METRIC_KEYS = {"exact_accuracy", "within_one_grade", "within_two_grades", "macro_f1"}


@dataclass
class BenchmarkResults:
    fold_results: list[dict[str, float]]

    def __post_init__(self):
        if not self.fold_results:
            return
        for i, fold in enumerate(self.fold_results):
            missing = REQUIRED_METRIC_KEYS - set(fold.keys())
            if missing:
                raise ValueError(
                    f"Fold {i} result missing required metric keys: {missing}. "
                    f"Got keys: {list(fold.keys())}"
                )

    def mean_scores(self) -> dict[str, float]:
        if not self.fold_results:
            return {}
        metric_names = self.fold_results[0].keys()
        means = {}
        for metric_name in metric_names:
            values = [fold[metric_name] for fold in self.fold_results]
            means[metric_name] = float(np.mean(values))
        return means

    def std_scores(self) -> dict[str, float]:
        if not self.fold_results:
            return {}
        metric_names = self.fold_results[0].keys()
        stds = {}
        for metric_name in metric_names:
            values = [fold[metric_name] for fold in self.fold_results]
            stds[metric_name] = float(np.std(values))
        return stds

    def to_json(self) -> str:
        data = {
            "fold_results": self.fold_results,
            "mean_scores": self.mean_scores(),
            "std_scores": self.std_scores(),
        }
        return json.dumps(data, indent=2)

    def to_markdown_table(self) -> str:
        if not self.fold_results:
            return ""
        metric_names = list(self.fold_results[0].keys())
        means = self.mean_scores()
        stds = self.std_scores()

        header = "| Fold | " + " | ".join(metric_names) + " |"
        separator = "|" + "|".join(["-" * 6] * (len(metric_names) + 1)) + "|"
        lines = [header, separator]

        for fold_idx, fold_result in enumerate(self.fold_results):
            values = [f"{fold_result[m]:.4f}" for m in metric_names]
            row = f"| {fold_idx} | " + " | ".join(values) + " |"
            lines.append(row)

        lines.append(separator)
        mean_values = [f"{means[m]:.4f}" for m in metric_names]
        mean_row = "| Mean | " + " | ".join(mean_values) + " |"
        lines.append(mean_row)

        std_values = [f"{stds[m]:.4f}" for m in metric_names]
        std_row = "| Std | " + " | ".join(std_values) + " |"
        lines.append(std_row)

        return "\n".join(lines)


class BenchmarkHarness:
    """Orchestrates n-fold CV benchmarking with retrain-per-fold.

    Accepts a model_factory (callable that creates a fresh model each time),
    pluggable metrics, and optional train/predict functions to support
    any model type (PyTorch, sklearn, etc.).
    """

    def __init__(
        self,
        model_factory: Callable[[], Any],
        metrics: Sequence[MetricComputer],
        device: Any | None = None,
    ) -> None:
        self.model_factory = model_factory
        self.metrics = metrics
        self.device = device or (
            torch.device("cuda") if torch.cuda.is_available()
            else torch.device("cpu")
        )

    @staticmethod
    def _default_predict(model: Any, X: np.ndarray, device: torch.device) -> np.ndarray:
        model.to(device)
        model.eval()
        X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
        with torch.no_grad():
            outputs = model(X_tensor)
            return torch.argmax(outputs, dim=1).cpu().numpy()

    @staticmethod
    def _default_train(model: Any, X_train: np.ndarray, y_train: np.ndarray) -> None:
        pass

    def run_benchmark(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        n_splits: int = DEFAULT_N_SPLITS,
        train_fn: Callable[[Any, np.ndarray, np.ndarray], None] | None = None,
        predict_fn: Callable[[Any, np.ndarray], np.ndarray] | None = None,
    ) -> BenchmarkResults:
        """Run n-fold cross-validation, training a fresh model per fold.

        Args:
            features: Feature matrix of shape (n_samples, n_features).
            labels: Label vector of shape (n_samples,).
            n_splits: Number of CV folds (default: 5).
            train_fn: Callable(model, X_train, y_train) that trains model in-place.
                If None, training is skipped (for testing).
            predict_fn: Callable(model, X_test) returning prediction array.
                If None, defaults to PyTorch forward + argmax.

        Returns:
            BenchmarkResults containing per-fold metric scores.
        """
        if len(features) != len(labels):
            raise ValueError(
                f"Features and labels must have the same length, "
                f"got {len(features)} and {len(labels)}"
            )

        if not isinstance(n_splits, int):
            raise TypeError(f"n_splits must be an integer, got {type(n_splits).__name__}")
        if n_splits < MIN_N_SPLITS or n_splits > MAX_N_SPLITS:
            raise ValueError(
                f"n_splits must be between {MIN_N_SPLITS} and {MAX_N_SPLITS}, got {n_splits}"
            )

        kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        fold_results: list[dict[str, float]] = []
        _train_fn = train_fn or self._default_train
        _predict_fn = predict_fn or (lambda m, x: self._default_predict(m, x, self.device))

        for train_idx, test_idx in kfold.split(features):
            X_train, X_test = features[train_idx], features[test_idx]
            y_train, y_test = labels[train_idx], labels[test_idx]

            model = self.model_factory()
            _train_fn(model, X_train, y_train)
            y_pred = _predict_fn(model, X_test)

            fold_scores = {}
            for metric in self.metrics:
                fold_scores[metric.name] = metric.compute(y_test, y_pred)
            fold_results.append(fold_scores)

        return BenchmarkResults(fold_results=fold_results)
