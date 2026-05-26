"""Tests for 5-fold cross-validation benchmark harness (TDD approach)."""

import json

import numpy as np
import pytest
import torch
import torch.nn as nn

from moonboard_analysis.training.benchmark import (
    BenchmarkHarness,
    BenchmarkResults,
    ExactAccuracy,
    WithinOneGrade,
    WithinTwoGrades,
)


@pytest.fixture
def mock_lstm_classifier() -> nn.Module:
    """Create a mock LSTM classifier for testing.

    Returns a simple neural network that always predicts random grades.
    """

    class SimpleLSTM(nn.Module):
        """Simple mock classifier."""

        def __init__(self, input_size: int = 164, num_classes: int = 7):
            super().__init__()
            self.fc = nn.Linear(input_size, num_classes)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Forward pass."""
            return self.fc(x)

    return SimpleLSTM(input_size=164, num_classes=7)


@pytest.fixture
def mock_benchmark_data() -> tuple[np.ndarray, np.ndarray]:
    """Generate mock training data for benchmark tests.

    Returns:
        Tuple of (features, labels) where:
        - features: np.ndarray of shape (1000, 164) with float values
        - labels: np.ndarray of shape (1000,) with int grades 0-6
    """
    rng = np.random.default_rng(42)
    features = rng.normal(0, 1, size=(1000, 164)).astype(np.float32)
    labels = rng.integers(0, 7, size=1000)
    return features, labels


class TestExactAccuracy:
    """Test ExactAccuracy metric computation."""

    def test_metric_name(self) -> None:
        """Verify metric has correct name."""
        metric = ExactAccuracy()
        assert metric.name == "exact_accuracy"

    def test_perfect_predictions(self) -> None:
        """Verify perfect predictions yield 1.0 accuracy."""
        metric = ExactAccuracy()
        y_true = np.array([0, 1, 2, 3, 4, 5, 6])
        y_pred = np.array([0, 1, 2, 3, 4, 5, 6])
        result = metric.compute(y_true, y_pred)
        assert result == 1.0

    def test_all_wrong_predictions(self) -> None:
        """Verify all wrong predictions yield 0.0 accuracy."""
        metric = ExactAccuracy()
        y_true = np.array([0, 0, 0, 0])
        y_pred = np.array([1, 2, 3, 4])
        result = metric.compute(y_true, y_pred)
        assert result == 0.0

    def test_partial_correct_predictions(self) -> None:
        """Verify partial correct predictions yield intermediate accuracy."""
        metric = ExactAccuracy()
        y_true = np.array([0, 1, 2, 3])
        y_pred = np.array([0, 1, 0, 0])
        result = metric.compute(y_true, y_pred)
        assert result == 0.5


class TestWithinOneGrade:
    """Test WithinOneGrade metric computation."""

    def test_metric_name(self) -> None:
        """Verify metric has correct name."""
        metric = WithinOneGrade()
        assert metric.name == "within_one_grade"

    def test_perfect_predictions(self) -> None:
        """Verify perfect predictions yield 1.0 accuracy."""
        metric = WithinOneGrade()
        y_true = np.array([0, 1, 2, 3, 4, 5, 6])
        y_pred = np.array([0, 1, 2, 3, 4, 5, 6])
        result = metric.compute(y_true, y_pred)
        assert result == 1.0

    def test_off_by_one_predictions(self) -> None:
        """Verify off-by-one predictions yield 1.0 accuracy."""
        metric = WithinOneGrade()
        y_true = np.array([1, 2, 3, 4])
        y_pred = np.array([0, 1, 2, 3])  # All off by 1
        result = metric.compute(y_true, y_pred)
        assert result == 1.0

    def test_off_by_two_predictions(self) -> None:
        """Verify off-by-two predictions yield < 1.0 accuracy."""
        metric = WithinOneGrade()
        y_true = np.array([2, 3, 4, 5])
        y_pred = np.array([0, 1, 2, 3])  # All off by 2
        result = metric.compute(y_true, y_pred)
        assert result == 0.0

    def test_mixed_predictions(self) -> None:
        """Verify mixed predictions yield expected accuracy."""
        metric = WithinOneGrade()
        y_true = np.array([0, 1, 2, 3])
        y_pred = np.array([0, 0, 3, 1])  # 0: exact, 1: off-by-1, 2: off-by-1, 3: off-by-2
        result = metric.compute(y_true, y_pred)
        assert result == 0.75


class TestWithinTwoGrades:
    """Test WithinTwoGrades metric computation."""

    def test_metric_name(self) -> None:
        """Verify metric has correct name."""
        metric = WithinTwoGrades()
        assert metric.name == "within_two_grades"

    def test_perfect_predictions(self) -> None:
        """Verify perfect predictions yield 1.0 accuracy."""
        metric = WithinTwoGrades()
        y_true = np.array([0, 1, 2, 3, 4, 5, 6])
        y_pred = np.array([0, 1, 2, 3, 4, 5, 6])
        result = metric.compute(y_true, y_pred)
        assert result == 1.0

    def test_off_by_two_predictions(self) -> None:
        """Verify off-by-two predictions yield 1.0 accuracy."""
        metric = WithinTwoGrades()
        y_true = np.array([2, 3, 4, 5])
        y_pred = np.array([0, 1, 2, 3])  # All off by 2
        result = metric.compute(y_true, y_pred)
        assert result == 1.0

    def test_off_by_three_predictions(self) -> None:
        """Verify off-by-three predictions yield < 1.0 accuracy."""
        metric = WithinTwoGrades()
        y_true = np.array([3, 4, 5, 6])
        y_pred = np.array([0, 1, 2, 3])  # All off by 3
        result = metric.compute(y_true, y_pred)
        assert result == 0.0

    def test_monotonic_with_exact(self) -> None:
        """Verify within-two >= within-one for same predictions."""
        exact = ExactAccuracy()
        within_one = WithinOneGrade()
        within_two = WithinTwoGrades()

        y_true = np.array([0, 1, 2, 3, 4, 5, 6])
        y_pred = np.array([0, 1, 2, 3, 4, 5, 6])

        exact_acc = exact.compute(y_true, y_pred)
        within_one_acc = within_one.compute(y_true, y_pred)
        within_two_acc = within_two.compute(y_true, y_pred)

        assert exact_acc <= within_one_acc <= within_two_acc


class TestBenchmarkResults:
    """Test BenchmarkResults dataclass."""

    def test_initialization(self) -> None:
        """Verify BenchmarkResults can be initialized with fold results."""
        fold_results = [
            {"exact_accuracy": 0.8, "within_one_grade": 0.9},
            {"exact_accuracy": 0.75, "within_one_grade": 0.85},
            {"exact_accuracy": 0.82, "within_one_grade": 0.91},
        ]
        results = BenchmarkResults(fold_results=fold_results)
        assert len(results.fold_results) == 3

    def test_mean_scores(self) -> None:
        """Verify mean score computation."""
        fold_results = [
            {"exact_accuracy": 0.8, "within_one_grade": 0.9},
            {"exact_accuracy": 0.8, "within_one_grade": 0.9},
        ]
        results = BenchmarkResults(fold_results=fold_results)
        mean_scores = results.mean_scores()
        assert mean_scores["exact_accuracy"] == 0.8
        assert mean_scores["within_one_grade"] == 0.9

    def test_std_scores(self) -> None:
        """Verify std dev score computation."""
        fold_results = [
            {"exact_accuracy": 0.8, "within_one_grade": 0.9},
            {"exact_accuracy": 0.8, "within_one_grade": 0.9},
        ]
        results = BenchmarkResults(fold_results=fold_results)
        std_scores = results.std_scores()
        assert std_scores["exact_accuracy"] == 0.0
        assert std_scores["within_one_grade"] == 0.0

    def test_std_scores_with_variance(self) -> None:
        """Verify std dev computation with actual variance."""
        fold_results = [
            {"exact_accuracy": 0.6},
            {"exact_accuracy": 0.8},
        ]
        results = BenchmarkResults(fold_results=fold_results)
        std_scores = results.std_scores()
        expected_std = np.std([0.6, 0.8])
        assert np.isclose(std_scores["exact_accuracy"], expected_std)

    def test_to_json(self) -> None:
        """Verify JSON serialization."""
        fold_results = [
            {"exact_accuracy": 0.8, "within_one_grade": 0.9},
            {"exact_accuracy": 0.75, "within_one_grade": 0.85},
        ]
        results = BenchmarkResults(fold_results=fold_results)
        json_str = results.to_json()

        # Verify it's valid JSON
        data = json.loads(json_str)
        assert "fold_results" in data
        assert "mean_scores" in data
        assert "std_scores" in data
        assert len(data["fold_results"]) == 2

    def test_to_markdown_table(self) -> None:
        """Verify markdown table generation."""
        fold_results = [
            {"exact_accuracy": 0.8, "within_one_grade": 0.9},
            {"exact_accuracy": 0.75, "within_one_grade": 0.85},
        ]
        results = BenchmarkResults(fold_results=fold_results)
        markdown = results.to_markdown_table()

        # Verify markdown structure
        assert "|" in markdown
        assert "Fold" in markdown
        assert "exact_accuracy" in markdown
        assert "within_one_grade" in markdown
        assert "Mean" in markdown
        assert "Std" in markdown


class TestBenchmarkHarness:
    """Test BenchmarkHarness integration."""

    def test_harness_initialization(
        self,
        mock_lstm_classifier: nn.Module,
    ) -> None:
        """Verify BenchmarkHarness can be initialized."""
        metrics = [ExactAccuracy(), WithinOneGrade(), WithinTwoGrades()]
        harness = BenchmarkHarness(model=mock_lstm_classifier, metrics=metrics)
        assert harness.model is mock_lstm_classifier
        assert len(harness.metrics) == 3

    def test_harness_run_benchmark(
        self,
        mock_lstm_classifier: nn.Module,
        mock_benchmark_data: tuple[np.ndarray, np.ndarray],
    ) -> None:
        """Verify BenchmarkHarness runs 5-fold CV successfully."""
        features, labels = mock_benchmark_data
        metrics = [ExactAccuracy()]
        harness = BenchmarkHarness(model=mock_lstm_classifier, metrics=metrics)

        results = harness.run_benchmark(features, labels, n_splits=5)

        assert isinstance(results, BenchmarkResults)
        assert len(results.fold_results) == 5
        assert all("exact_accuracy" in fold for fold in results.fold_results)

    def test_harness_run_benchmark_multiple_metrics(
        self,
        mock_lstm_classifier: nn.Module,
        mock_benchmark_data: tuple[np.ndarray, np.ndarray],
    ) -> None:
        """Verify BenchmarkHarness computes all metrics per fold."""
        features, labels = mock_benchmark_data
        metrics = [ExactAccuracy(), WithinOneGrade(), WithinTwoGrades()]
        harness = BenchmarkHarness(model=mock_lstm_classifier, metrics=metrics)

        results = harness.run_benchmark(features, labels, n_splits=5)

        assert len(results.fold_results) == 5
        for fold_result in results.fold_results:
            assert "exact_accuracy" in fold_result
            assert "within_one_grade" in fold_result
            assert "within_two_grades" in fold_result

    def test_harness_results_validity(
        self,
        mock_lstm_classifier: nn.Module,
        mock_benchmark_data: tuple[np.ndarray, np.ndarray],
    ) -> None:
        """Verify all metric scores are in [0, 1] range."""
        features, labels = mock_benchmark_data
        metrics = [ExactAccuracy(), WithinOneGrade(), WithinTwoGrades()]
        harness = BenchmarkHarness(model=mock_lstm_classifier, metrics=metrics)

        results = harness.run_benchmark(features, labels, n_splits=5)
        mean_scores = results.mean_scores()

        for metric_name, score in mean_scores.items():
            assert 0.0 <= score <= 1.0, f"{metric_name} score {score} not in [0,1]"

    def test_harness_mean_std_validity(
        self,
        mock_lstm_classifier: nn.Module,
        mock_benchmark_data: tuple[np.ndarray, np.ndarray],
    ) -> None:
        """Verify std dev scores are non-negative."""
        features, labels = mock_benchmark_data
        metrics = [ExactAccuracy(), WithinOneGrade(), WithinTwoGrades()]
        harness = BenchmarkHarness(model=mock_lstm_classifier, metrics=metrics)

        results = harness.run_benchmark(features, labels, n_splits=5)
        std_scores = results.std_scores()

        for metric_name, score in std_scores.items():
            assert score >= 0.0, f"{metric_name} std {score} is negative"

    def test_harness_serialization(
        self,
        mock_lstm_classifier: nn.Module,
        mock_benchmark_data: tuple[np.ndarray, np.ndarray],
    ) -> None:
        """Verify results can be serialized to JSON and markdown."""
        features, labels = mock_benchmark_data
        metrics = [ExactAccuracy(), WithinOneGrade()]
        harness = BenchmarkHarness(model=mock_lstm_classifier, metrics=metrics)

        results = harness.run_benchmark(features, labels, n_splits=3)

        # Test JSON
        json_str = results.to_json()
        assert isinstance(json_str, str)
        data = json.loads(json_str)
        assert "fold_results" in data

        # Test Markdown
        markdown = results.to_markdown_table()
        assert isinstance(markdown, str)
        assert "|" in markdown

    def test_monotonic_metrics_relationship(
        self,
        mock_lstm_classifier: nn.Module,
        mock_benchmark_data: tuple[np.ndarray, np.ndarray],
    ) -> None:
        """Verify exact_acc <= within_one <= within_two for all folds."""
        features, labels = mock_benchmark_data
        metrics = [ExactAccuracy(), WithinOneGrade(), WithinTwoGrades()]
        harness = BenchmarkHarness(model=mock_lstm_classifier, metrics=metrics)

        results = harness.run_benchmark(features, labels, n_splits=3)

        for fold_result in results.fold_results:
            exact = fold_result["exact_accuracy"]
            within_one = fold_result["within_one_grade"]
            within_two = fold_result["within_two_grades"]
            assert exact <= within_one <= within_two
