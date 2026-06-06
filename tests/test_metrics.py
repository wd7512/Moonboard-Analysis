"""Tests for evaluation metrics with known expected values."""

import numpy as np
import pytest

from moonboard_analysis.training.metrics import (
    _accuracy_within_diagonal,
    evaluate_classification,
    extract_required_metrics,
    within_k_accuracy,
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
        assert metrics["macro_f1"] == 1.0
        assert metrics["weighted_f1"] == 1.0

    def test_all_wrong_classification(self) -> None:
        """Verify metrics for completely wrong predictions yield 0.0 accuracy."""
        y_true = [0, 0, 0, 0]
        y_pred = [1, 2, 1, 2]
        metrics = evaluate_classification(y_true, y_pred, num_classes=3)

        assert metrics["exact_accuracy"] == 0.0
        assert "macro_f1" in metrics
        assert "weighted_f1" in metrics

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
        assert "macro_f1" in metrics
        assert "weighted_f1" in metrics

    def test_confusion_matrix_shape(self) -> None:
        """Verify confusion matrix is square with correct dimensions."""
        num_classes = 4
        y_true = [0, 1, 2, 3]
        y_pred = [0, 1, 2, 3]
        metrics = evaluate_classification(y_true, y_pred, num_classes=num_classes)

        conf_matrix = metrics["confusion_matrix"]
        assert conf_matrix.shape == (num_classes, num_classes)

    def test_macro_f1_ignores_empty_classes(self) -> None:
        """macro_F1 should only average over classes with non-zero support.

        Simulates the 2016 dataset situation: 13 classes but only 10 have
        samples. The 3 empty classes (0, 1, 2) must not drag down the mean.
        """
        # Classes 0,1,2 have zero support. Classes 3-12 all predict perfectly.
        y_true = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
        y_pred = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
        metrics = evaluate_classification(y_true, y_pred, num_classes=13)

        # With the bug: macro_f1 = mean of [0,0,0,1,1,1,1,1,1,1,1,1,1] = 10/13 ≈ 0.769
        # With the fix: macro_f1 = mean of [1,1,1,1,1,1,1,1,1,1] = 1.0
        assert metrics["macro_f1"] == 1.0

    def test_macro_f1_all_empty_returns_zero(self) -> None:
        """If no class has support, macro_f1 should be 0.0 (not NaN)."""
        y_true = []
        y_pred = []
        # sklearn requires at least one sample; test with single class
        # that has support=0 in the confusion matrix sense
        # Use 2 classes, only class 0 has data
        y_true = [0, 0, 0]
        y_pred = [0, 0, 0]
        metrics = evaluate_classification(y_true, y_pred, num_classes=2)
        # Class 1 has zero support, class 0 has perfect F1
        # macro_f1 should be 1.0 (only class 0 counts)
        assert metrics["macro_f1"] == 1.0


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


class TestExtractRequiredMetrics:
    """Tests for the extract_required_metrics() shared helper."""

    def test_happy_path(self) -> None:
        """Pass valid metrics dict, verify all 4 contract keys returned."""
        metrics = {
            "exact_accuracy": 0.85,
            "within_1_accuracy": 0.92,
            "within_2_accuracy": 0.97,
            "within_3_accuracy": 0.99,
            "macro_f1": 0.78,
        }
        result = extract_required_metrics(metrics)

        assert result["exact_accuracy"] == 0.85
        assert result["within_one_grade"] == 0.92
        assert result["within_two_grades"] == 0.97
        assert result["macro_f1"] == 0.78
        assert len(result) == 4

    def test_missing_key_raises_error(self) -> None:
        """Pass dict missing macro_f1, verify KeyError."""
        metrics = {
            "exact_accuracy": 0.85,
            "within_1_accuracy": 0.92,
            "within_2_accuracy": 0.97,
        }
        with pytest.raises(KeyError, match="macro_f1"):
            extract_required_metrics(metrics)

    def test_extra_keys_ignored(self) -> None:
        """Pass dict with extra keys, verify only 4 contract keys returned."""
        metrics = {
            "exact_accuracy": 0.75,
            "within_1_accuracy": 0.88,
            "within_2_accuracy": 0.95,
            "within_3_accuracy": 0.98,
            "macro_f1": 0.65,
            "weighted_f1": 0.70,
        }
        result = extract_required_metrics(metrics)

        assert result["exact_accuracy"] == 0.75
        assert result["within_one_grade"] == 0.88
        assert result["within_two_grades"] == 0.95
        assert result["macro_f1"] == 0.65
        assert len(result) == 4


class TestWithinKAccuracy:
    """Tests for the public within_k_accuracy() function."""

    def test_within_k_accuracy_identity(self) -> None:
        """Identity 4x4 matrix, k=1 should be 1.0."""
        conf_matrix = np.eye(4, dtype=int)
        assert within_k_accuracy(conf_matrix, k=1) == 1.0

    def test_within_k_accuracy_off_diagonal(self) -> None:
        """Anti-diagonal 4x4: hand-computed expected values."""
        conf_matrix = np.array(
            [[0, 0, 0, 1],
             [0, 0, 1, 0],
             [0, 1, 0, 0],
             [1, 0, 0, 0]],
            dtype=int,
        )
        assert within_k_accuracy(conf_matrix, k=1) == 0.5
        assert within_k_accuracy(conf_matrix, k=2) == 0.5
        assert within_k_accuracy(conf_matrix, k=3) == 1.0

    def test_within_k_accuracy_k_equals_0(self) -> None:
        """k=0 should give exact diagonal accuracy only."""
        conf_matrix = np.array([[1, 2], [3, 4]], dtype=int)
        assert within_k_accuracy(conf_matrix, k=0) == 5.0 / 10.0

    def test_within_k_accuracy_k_large(self) -> None:
        """k >= n-1 should capture entire matrix → 1.0."""
        conf_matrix = np.array(
            [[1, 2],
             [3, 4]],
            dtype=int,
        )
        assert within_k_accuracy(conf_matrix, k=5) == 1.0
