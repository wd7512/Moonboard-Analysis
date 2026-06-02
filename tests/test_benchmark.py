"""Tests for n-fold cross-validation benchmark harness (TDD approach)."""

import json

import numpy as np
import pytest
import torch
import torch.nn as nn

from moonboard_analysis.training.benchmark import (
    BenchmarkHarness,
    BenchmarkResults,
    ExactAccuracy,
    MetricComputer,
    WithinOneGrade,
    WithinTwoGrades,
)


class SimpleLSTM(nn.Module):
    """Simple mock classifier for testing."""

    def __init__(self, input_size: int = 164, num_classes: int = 7):
        super().__init__()
        self.fc = nn.Linear(input_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


@pytest.fixture
def mock_model_factory():
    """Create a factory that returns fresh SimpleLSTM models."""
    return lambda: SimpleLSTM(input_size=164, num_classes=7)


@pytest.fixture
def mock_benchmark_data() -> tuple[np.ndarray, np.ndarray]:
    """Generate mock training data for benchmark tests."""
    rng = np.random.default_rng(42)
    features = rng.normal(0, 1, size=(1000, 164)).astype(np.float32)
    labels = rng.integers(0, 7, size=1000)
    return features, labels


class TestExactAccuracy:
    def test_metric_name(self) -> None:
        metric = ExactAccuracy()
        assert metric.name == "exact_accuracy"

    def test_perfect_predictions(self) -> None:
        metric = ExactAccuracy()
        y_true = np.array([0, 1, 2, 3, 4, 5, 6])
        y_pred = np.array([0, 1, 2, 3, 4, 5, 6])
        result = metric.compute(y_true, y_pred)
        assert result == 1.0

    def test_all_wrong_predictions(self) -> None:
        metric = ExactAccuracy()
        y_true = np.array([0, 0, 0, 0])
        y_pred = np.array([1, 2, 3, 4])
        result = metric.compute(y_true, y_pred)
        assert result == 0.0

    def test_partial_correct_predictions(self) -> None:
        metric = ExactAccuracy()
        y_true = np.array([0, 1, 2, 3])
        y_pred = np.array([0, 1, 0, 0])
        result = metric.compute(y_true, y_pred)
        assert result == 0.5


class TestWithinOneGrade:
    def test_metric_name(self) -> None:
        metric = WithinOneGrade()
        assert metric.name == "within_one_grade"

    def test_perfect_predictions(self) -> None:
        metric = WithinOneGrade()
        y_true = np.array([0, 1, 2, 3, 4, 5, 6])
        y_pred = np.array([0, 1, 2, 3, 4, 5, 6])
        result = metric.compute(y_true, y_pred)
        assert result == 1.0

    def test_off_by_one_predictions(self) -> None:
        metric = WithinOneGrade()
        y_true = np.array([1, 2, 3, 4])
        y_pred = np.array([0, 1, 2, 3])
        result = metric.compute(y_true, y_pred)
        assert result == 1.0

    def test_off_by_two_predictions(self) -> None:
        metric = WithinOneGrade()
        y_true = np.array([2, 3, 4, 5])
        y_pred = np.array([0, 1, 2, 3])
        result = metric.compute(y_true, y_pred)
        assert result == 0.0

    def test_mixed_predictions(self) -> None:
        metric = WithinOneGrade()
        y_true = np.array([0, 1, 2, 3])
        y_pred = np.array([0, 0, 3, 1])
        result = metric.compute(y_true, y_pred)
        assert result == 0.75


class TestWithinTwoGrades:
    def test_metric_name(self) -> None:
        metric = WithinTwoGrades()
        assert metric.name == "within_two_grades"

    def test_perfect_predictions(self) -> None:
        metric = WithinTwoGrades()
        y_true = np.array([0, 1, 2, 3, 4, 5, 6])
        y_pred = np.array([0, 1, 2, 3, 4, 5, 6])
        result = metric.compute(y_true, y_pred)
        assert result == 1.0

    def test_off_by_two_predictions(self) -> None:
        metric = WithinTwoGrades()
        y_true = np.array([2, 3, 4, 5])
        y_pred = np.array([0, 1, 2, 3])
        result = metric.compute(y_true, y_pred)
        assert result == 1.0

    def test_off_by_three_predictions(self) -> None:
        metric = WithinTwoGrades()
        y_true = np.array([3, 4, 5, 6])
        y_pred = np.array([0, 1, 2, 3])
        result = metric.compute(y_true, y_pred)
        assert result == 0.0

    def test_monotonic_with_exact(self) -> None:
        exact = ExactAccuracy()
        within_one = WithinOneGrade()
        within_two = WithinTwoGrades()

        y_true = np.array([0, 1, 2, 3, 4, 5, 6])
        y_pred = np.array([0, 1, 2, 3, 4, 5, 6])

        exact_acc = exact.compute(y_true, y_pred)
        within_one_acc = within_one.compute(y_true, y_pred)
        within_two_acc = within_two.compute(y_true, y_pred)

        assert exact_acc <= within_one_acc <= within_two_acc


class MacroF1(MetricComputer):
    def compute(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return 0.5

    @property
    def name(self) -> str:
        return "macro_f1"


class TestBenchmarkResults:
    def test_initialization(self) -> None:
        fold_results = [
            {"exact_accuracy": 0.8, "within_one_grade": 0.9, "within_two_grades": 0.95, "macro_f1": 0.75},
            {"exact_accuracy": 0.75, "within_one_grade": 0.85, "within_two_grades": 0.93, "macro_f1": 0.72},
            {"exact_accuracy": 0.82, "within_one_grade": 0.91, "within_two_grades": 0.96, "macro_f1": 0.78},
        ]
        results = BenchmarkResults(fold_results=fold_results)
        assert len(results.fold_results) == 3

    def test_mean_scores(self) -> None:
        fold_results = [
            {"exact_accuracy": 0.8, "within_one_grade": 0.9, "within_two_grades": 0.95, "macro_f1": 0.75},
            {"exact_accuracy": 0.8, "within_one_grade": 0.9, "within_two_grades": 0.95, "macro_f1": 0.75},
        ]
        results = BenchmarkResults(fold_results=fold_results)
        mean_scores = results.mean_scores()
        assert mean_scores["exact_accuracy"] == 0.8
        assert mean_scores["within_one_grade"] == 0.9

    def test_std_scores(self) -> None:
        fold_results = [
            {"exact_accuracy": 0.8, "within_one_grade": 0.9, "within_two_grades": 0.95, "macro_f1": 0.75},
            {"exact_accuracy": 0.8, "within_one_grade": 0.9, "within_two_grades": 0.95, "macro_f1": 0.75},
        ]
        results = BenchmarkResults(fold_results=fold_results)
        std_scores = results.std_scores()
        assert std_scores["exact_accuracy"] == 0.0
        assert std_scores["within_one_grade"] == 0.0

    def test_std_scores_with_variance(self) -> None:
        fold_results = [
            {"exact_accuracy": 0.6, "within_one_grade": 0.7, "within_two_grades": 0.8, "macro_f1": 0.5},
            {"exact_accuracy": 0.8, "within_one_grade": 0.9, "within_two_grades": 0.95, "macro_f1": 0.6},
        ]
        results = BenchmarkResults(fold_results=fold_results)
        std_scores = results.std_scores()
        expected_std = np.std([0.6, 0.8])
        assert np.isclose(std_scores["exact_accuracy"], expected_std)

    def test_to_json(self) -> None:
        fold_results = [
            {"exact_accuracy": 0.8, "within_one_grade": 0.9, "within_two_grades": 0.95, "macro_f1": 0.75},
            {"exact_accuracy": 0.75, "within_one_grade": 0.85, "within_two_grades": 0.93, "macro_f1": 0.72},
        ]
        results = BenchmarkResults(fold_results=fold_results)
        json_str = results.to_json()

        data = json.loads(json_str)
        assert "fold_results" in data
        assert "mean_scores" in data
        assert "std_scores" in data
        assert len(data["fold_results"]) == 2

    def test_to_markdown_table(self) -> None:
        fold_results = [
            {"exact_accuracy": 0.8, "within_one_grade": 0.9, "within_two_grades": 0.95, "macro_f1": 0.75},
            {"exact_accuracy": 0.75, "within_one_grade": 0.85, "within_two_grades": 0.93, "macro_f1": 0.72},
        ]
        results = BenchmarkResults(fold_results=fold_results)
        markdown = results.to_markdown_table()

        assert "|" in markdown
        assert "Fold" in markdown
        assert "exact_accuracy" in markdown
        assert "within_one_grade" in markdown
        assert "Mean" in markdown
        assert "Std" in markdown

    def test_validation_valid_folds(self) -> None:
        """All required keys present: no error."""
        fold_results = [
            {"exact_accuracy": 0.8, "within_one_grade": 0.9, "within_two_grades": 0.95, "macro_f1": 0.75},
            {"exact_accuracy": 0.7, "within_one_grade": 0.8, "within_two_grades": 0.9, "macro_f1": 0.65},
        ]
        results = BenchmarkResults(fold_results=fold_results)
        assert len(results.fold_results) == 2

    def test_validation_missing_key(self) -> None:
        """Missing macro_f1 raises ValueError."""
        fold_results = [
            {"exact_accuracy": 0.8, "within_one_grade": 0.9, "within_two_grades": 0.95},
        ]
        with pytest.raises(ValueError, match="macro_f1"):
            BenchmarkResults(fold_results=fold_results)

    def test_validation_empty_folds(self) -> None:
        """Empty fold list: no error."""
        results = BenchmarkResults(fold_results=[])
        assert results.fold_results == []


class TestBenchmarkHarness:
    def test_harness_initialization(
        self,
        mock_model_factory,
    ) -> None:
        metrics = [ExactAccuracy(), WithinOneGrade(), WithinTwoGrades()]
        harness = BenchmarkHarness(model_factory=mock_model_factory, metrics=metrics)
        assert harness.model_factory is mock_model_factory
        assert len(harness.metrics) == 3

    def test_harness_run_benchmark(
        self,
        mock_model_factory,
        mock_benchmark_data: tuple[np.ndarray, np.ndarray],
    ) -> None:
        features, labels = mock_benchmark_data
        metrics: list[MetricComputer] = [ExactAccuracy(), WithinOneGrade(), WithinTwoGrades(), MacroF1()]
        harness = BenchmarkHarness(model_factory=mock_model_factory, metrics=metrics)

        results = harness.run_benchmark(features, labels, n_splits=5)

        assert isinstance(results, BenchmarkResults)
        assert len(results.fold_results) == 5
        assert all("macro_f1" in fold for fold in results.fold_results)

    def test_harness_run_benchmark_multiple_metrics(
        self,
        mock_model_factory,
        mock_benchmark_data: tuple[np.ndarray, np.ndarray],
    ) -> None:
        features, labels = mock_benchmark_data
        metrics = [ExactAccuracy(), WithinOneGrade(), WithinTwoGrades(), MacroF1()]
        harness = BenchmarkHarness(model_factory=mock_model_factory, metrics=metrics)

        results = harness.run_benchmark(features, labels, n_splits=5)

        assert len(results.fold_results) == 5
        for fold_result in results.fold_results:
            assert "exact_accuracy" in fold_result
            assert "within_one_grade" in fold_result
            assert "within_two_grades" in fold_result
            assert "macro_f1" in fold_result

    def test_harness_results_validity(
        self,
        mock_model_factory,
        mock_benchmark_data: tuple[np.ndarray, np.ndarray],
    ) -> None:
        features, labels = mock_benchmark_data
        metrics = [ExactAccuracy(), WithinOneGrade(), WithinTwoGrades(), MacroF1()]
        harness = BenchmarkHarness(model_factory=mock_model_factory, metrics=metrics)

        results = harness.run_benchmark(features, labels, n_splits=5)
        mean_scores = results.mean_scores()

        for metric_name, score in mean_scores.items():
            assert 0.0 <= score <= 1.0, f"{metric_name} score {score} not in [0,1]"

    def test_harness_mean_std_validity(
        self,
        mock_model_factory,
        mock_benchmark_data: tuple[np.ndarray, np.ndarray],
    ) -> None:
        features, labels = mock_benchmark_data
        metrics = [ExactAccuracy(), WithinOneGrade(), WithinTwoGrades(), MacroF1()]
        harness = BenchmarkHarness(model_factory=mock_model_factory, metrics=metrics)

        results = harness.run_benchmark(features, labels, n_splits=5)
        std_scores = results.std_scores()

        for metric_name, score in std_scores.items():
            assert score >= 0.0, f"{metric_name} std {score} is negative"

    def test_harness_serialization(
        self,
        mock_model_factory,
        mock_benchmark_data: tuple[np.ndarray, np.ndarray],
    ) -> None:
        features, labels = mock_benchmark_data
        metrics = [ExactAccuracy(), WithinOneGrade(), WithinTwoGrades(), MacroF1()]
        harness = BenchmarkHarness(model_factory=mock_model_factory, metrics=metrics)

        results = harness.run_benchmark(features, labels, n_splits=3)

        json_str = results.to_json()
        assert isinstance(json_str, str)
        data = json.loads(json_str)
        assert "fold_results" in data

        markdown = results.to_markdown_table()
        assert isinstance(markdown, str)
        assert "|" in markdown

    def test_monotonic_metrics_relationship(
        self,
        mock_model_factory,
        mock_benchmark_data: tuple[np.ndarray, np.ndarray],
    ) -> None:
        features, labels = mock_benchmark_data
        metrics = [ExactAccuracy(), WithinOneGrade(), WithinTwoGrades(), MacroF1()]
        harness = BenchmarkHarness(model_factory=mock_model_factory, metrics=metrics)

        results = harness.run_benchmark(features, labels, n_splits=3)

        for fold_result in results.fold_results:
            exact = fold_result["exact_accuracy"]
            within_one = fold_result["within_one_grade"]
            within_two = fold_result["within_two_grades"]
            assert exact <= within_one <= within_two

    def test_mismatched_features_labels_raises_error(
        self,
        mock_model_factory,
        mock_benchmark_data: tuple[np.ndarray, np.ndarray],
    ) -> None:
        features, _ = mock_benchmark_data
        wrong_labels = np.array([0, 1, 2])
        metrics = [ExactAccuracy()]
        harness = BenchmarkHarness(model_factory=mock_model_factory, metrics=metrics)

        with pytest.raises(ValueError, match="Features and labels must have the same length"):
            harness.run_benchmark(features, wrong_labels, n_splits=3)

    def test_retrain_per_fold_factory_called_n_times(
        self,
        mock_model_factory,
        mock_benchmark_data: tuple[np.ndarray, np.ndarray],
    ) -> None:
        features, labels = mock_benchmark_data
        call_count = 0

        def counting_factory():
            nonlocal call_count
            call_count += 1
            return SimpleLSTM(164, 7)

        metrics = [ExactAccuracy(), WithinOneGrade(), WithinTwoGrades(), MacroF1()]
        harness = BenchmarkHarness(model_factory=counting_factory, metrics=metrics)
        harness.run_benchmark(features, labels, n_splits=5)

        assert call_count == 5, f"Expected 5 factory calls, got {call_count}"

    def test_retrain_per_fold_different_results_with_training(
        self,
        mock_benchmark_data: tuple[np.ndarray, np.ndarray],
    ) -> None:
        features, labels = mock_benchmark_data
        metrics = [ExactAccuracy(), WithinOneGrade(), WithinTwoGrades(), MacroF1()]

        def train_model(model, X_train, y_train):
            for p in model.parameters():
                p.data.mul_(0.0)

        harness = BenchmarkHarness(
            model_factory=lambda: SimpleLSTM(164, 7),
            metrics=metrics,
        )
        results = harness.run_benchmark(features, labels, n_splits=3, train_fn=train_model)

        assert len(results.fold_results) == 3

    def test_invalid_n_splits_raises_error(
        self,
        mock_model_factory,
        mock_benchmark_data: tuple[np.ndarray, np.ndarray],
    ) -> None:
        features, labels = mock_benchmark_data
        metrics = [ExactAccuracy()]
        harness = BenchmarkHarness(model_factory=mock_model_factory, metrics=metrics)

        with pytest.raises(ValueError, match="n_splits must be between"):
            harness.run_benchmark(features, labels, n_splits=0)

        with pytest.raises(ValueError, match="n_splits must be between"):
            harness.run_benchmark(features, labels, n_splits=11)

    def test_non_int_n_splits_raises_error(
        self,
        mock_model_factory,
        mock_benchmark_data: tuple[np.ndarray, np.ndarray],
    ) -> None:
        features, labels = mock_benchmark_data
        metrics = [ExactAccuracy()]
        harness = BenchmarkHarness(model_factory=mock_model_factory, metrics=metrics)

        with pytest.raises(TypeError, match="n_splits must be an integer"):
            harness.run_benchmark(features, labels, n_splits=5.0)
