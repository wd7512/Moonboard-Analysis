"""Tests for evaluation metrics with known expected values."""

import numpy as np

from moonboard_analysis.training.metrics import (
    _accuracy_within_diagonal,
    evaluate_classification,
)


class TestEvaluateClassification:
    """Test classification metric computation."""

    def test_perfect_classification(self) -> None:
        """Verify metrics for perfect predictions yield 1.0 accuracy."""
        y_true = [0, 1, 2, 0, 1]
        y_pred = [0, 1, 2, 0, 1]
        metrics = evaluate_classification(y_true, y_pred, num_classes=3)

        assert metrics["exact_accuracy"] == 1.0
        assert metrics["within_1_accuracy"] == 1.0
        assert metrics["within_2_accuracy"] == 1.0

    def test_all_wrong_classification(self) -> None:
        """Verify metrics for completely wrong predictions yield 0.0 accuracy."""
        y_true = [0, 0, 0, 0]
        y_pred = [1, 2, 1, 2]
        metrics = evaluate_classification(y_true, y_pred, num_classes=3)

        assert metrics["exact_accuracy"] == 0.0

    def test_within_k_accuracy_monotonic(self) -> None:
        """Verify within-k accuracy is non-decreasing as k increases."""
        y_true = [0, 1, 2, 3, 4]
        y_pred = [1, 2, 3, 4, 0]
        metrics = evaluate_classification(y_true, y_pred, num_classes=5)

        assert metrics["within_1_accuracy"] <= metrics["within_2_accuracy"]
        assert metrics["within_2_accuracy"] <= metrics["within_3_accuracy"]
        assert metrics["within_3_accuracy"] <= metrics["within_4_accuracy"]

    def test_per_class_metrics_length(self) -> None:
        """Verify per-class metric lists match number of classes."""
        num_classes = 5
        y_true = [0, 1, 2, 3, 4]
        y_pred = [0, 1, 2, 3, 4]
        metrics = evaluate_classification(y_true, y_pred, num_classes=num_classes)

        assert len(metrics["per_class_precision"]) == num_classes
        assert len(metrics["per_class_recall"]) == num_classes
        assert len(metrics["per_class_f1"]) == num_classes

    def test_confusion_matrix_shape(self) -> None:
        """Verify confusion matrix is square with correct dimensions."""
        num_classes = 4
        y_true = [0, 1, 2, 3]
        y_pred = [0, 1, 2, 3]
        metrics = evaluate_classification(y_true, y_pred, num_classes=num_classes)

        conf_matrix = metrics["confusion_matrix"]
        assert conf_matrix.shape == (num_classes, num_classes)


class TestAccuracyWithinDiagonal:
    """Test the diagonal accuracy helper function."""

    def test_width_1_perfect(self) -> None:
        """Verify width=1 on identity matrix returns 1.0."""
        conf_matrix = np.eye(3, dtype=int)
        result = _accuracy_within_diagonal(conf_matrix, width=1)
        assert result == 1.0

    def test_width_1_all_off_diagonal(self) -> None:
        """Verify width=1 on off-diagonal matrix returns 0.0."""
        conf_matrix = np.array([[0, 1], [1, 0]], dtype=int)
        result = _accuracy_within_diagonal(conf_matrix, width=1)
        assert result == 0.0

    def test_width_2_captures_adjacent(self) -> None:
        """Verify width=2 captures adjacent diagonal elements."""
        conf_matrix = np.array(
            [[0, 1, 0],
             [1, 0, 1],
             [0, 1, 0]],
            dtype=int
        )
        result = _accuracy_within_diagonal(conf_matrix, width=2)
        expected = 4.0 / 4.0
        assert result == expected

    def test_uniform_matrix(self) -> None:
        """Verify uniform confusion matrix yields predictable result."""
        conf_matrix = np.ones((3, 3), dtype=int)
        result = _accuracy_within_diagonal(conf_matrix, width=1)
        assert result == 3.0 / 9.0
