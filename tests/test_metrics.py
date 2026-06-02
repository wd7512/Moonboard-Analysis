"""Tests for evaluation metrics with known expected values."""

import numpy as np
import pytest

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

    @pytest.mark.parametrize("width", [1, 2, 3, 4])
    def test_identity_all_within_k(self, width: int) -> None:
        """Any width on identity matrix returns 1.0."""
        conf_matrix = np.eye(4, dtype=int)
        assert _accuracy_within_diagonal(conf_matrix, width) == 1.0

    def test_width_0_diagonal_only(self) -> None:
        """width=0 should only capture the diagonal."""
        conf_matrix = np.array([[1, 2], [3, 4]], dtype=int)
        assert _accuracy_within_diagonal(conf_matrix, width=0) == 5.0 / 10.0

    def test_anti_diagonal_4x4(self) -> None:
        """Anti-diagonal 4x4: check that within-K captures correct off-diagonals."""
        conf_matrix = np.array(
            [[0, 0, 0, 1],
             [0, 0, 1, 0],
             [0, 1, 0, 0],
             [1, 0, 0, 0]],
            dtype=int,
        )
        # width=1: captures i-1..i+1 → total = 2/4 = 0.5
        assert _accuracy_within_diagonal(conf_matrix, width=1) == 0.5
        # width=2: captures i-2..i+2 → total = 2/4 = 0.5
        assert _accuracy_within_diagonal(conf_matrix, width=2) == 0.5
        # width=3: ±3 on 4-element array covers full range → total = 4/4 = 1.0
        assert _accuracy_within_diagonal(conf_matrix, width=3) == 1.0

    def test_synthetic_4x4(self) -> None:
        """Hand-computed within-K accuracies for a synthetic 4x4 matrix."""
        conf_matrix = np.array(
            [[5, 2, 1, 0],
             [1, 6, 2, 1],
             [0, 1, 7, 2],
             [1, 0, 1, 6]],
            dtype=int,
        )
        # Total sum = 36
        # width=0 (diagonal): 5+6+7+6 = 24 → 24/36 = 2/3
        assert _accuracy_within_diagonal(conf_matrix, width=0) == pytest.approx(24 / 36)
        # width=1 (±1): diag + ±1 band
        #   (0,0)=5, (0,1)=2
        #   (1,0)=1, (1,1)=6, (1,2)=2
        #   (2,1)=1, (2,2)=7, (2,3)=2
        #   (3,2)=1, (3,3)=6
        #   total = 33 → 33/36
        assert _accuracy_within_diagonal(conf_matrix, width=1) == pytest.approx(33 / 36)
        # width=2 (±2):
        #   (0,0)=5, (0,1)=2, (0,2)=1
        #   (1,0)=1, (1,1)=6, (1,2)=2, (1,3)=1
        #   (2,0)=0, (2,1)=1, (2,2)=7, (2,3)=2
        #   (3,1)=0, (3,2)=1, (3,3)=6
        #   total = 35 → 35/36
        assert _accuracy_within_diagonal(conf_matrix, width=2) == pytest.approx(35 / 36)
        # width=3 (±3): captures entire matrix → 1.0
        assert _accuracy_within_diagonal(conf_matrix, width=3) == pytest.approx(1.0)

    def test_edge_rows_first_and_last(self) -> None:
        """First and last rows should not underflow/overflow indices."""
        conf_matrix = np.ones((4, 4), dtype=int)
        # width=1 on first row: range(max(0,0-1), min(4,0+2)) = range(0,2) → 2 elements
        # width=1 on last  row: range(max(0,3-1), min(4,3+2)) = range(2,4) → 2 elements
        # interior rows: 3 elements each
        # total = 2 + 3 + 3 + 2 = 10, sum = 16
        assert _accuracy_within_diagonal(conf_matrix, width=1) == 10.0 / 16.0

    def test_width_beyond_matrix(self) -> None:
        """width >= n-1 captures the entire matrix."""
        conf_matrix = np.array(
            [[1, 2],
             [3, 4]],
            dtype=int,
        )
        assert _accuracy_within_diagonal(conf_matrix, width=5) == 1.0
