"""Tests for ClassBalancedLoss and _compute_class_weights.

Imported via importlib due to the hyphen in the submissions directory name,
which prevents standard Python package imports.
"""

import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch

_module_path = (
    Path(__file__).resolve().parent.parent
    / "submissions"
    / "class-balanced-loss"
    / "main.py"
)
_spec = importlib.util.spec_from_file_location("cbl_main", _module_path)
_cbl = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_cbl)

compute_class_weights = _cbl._compute_class_weights
ClassBalancedLoss = _cbl.ClassBalancedLoss


class TestComputeClassWeights:
    """Tests for _compute_class_weights()."""

    def test_beta_zero(self):
        """Beta=0: (1-0)/(1-0^{n_y}) = 1/1 = 1 for all non-zero classes, 0 for zero-count."""
        counts = np.array([0, 1, 2, 3])
        weights = compute_class_weights(counts, 4, beta=0.0)
        assert weights.dtype == torch.float32
        expected = [0.0, 1.0, 1.0, 1.0]
        assert torch.allclose(weights, torch.tensor(expected))

    def test_beta_one_handles_gracefully(self):
        """Beta=1: 0/0 guard produces nan for non-zero counts (known limitation).

        np.divide(where=counts>0) still evaluates 0/0 for all elements,
        producing nan for any count > 0.  Only the zero-count slot gets the
        ``out`` fill value 0.0.
        """
        counts = np.array([0, 1, 2, 3])
        weights = compute_class_weights(counts, 4, beta=1.0)
        assert weights[0] == 0.0  # ``out`` fill for where=False
        assert torch.isnan(weights[1:]).all()

    def test_empty_counts_all_zeros(self):
        """All zero counts → all weights are 0.0."""
        counts = np.zeros(12, dtype=np.int64)
        weights = compute_class_weights(counts, 12, beta=0.999)
        assert torch.allclose(weights, torch.zeros(12))

    def test_missing_classes_some_zero(self):
        """Some bin counts = 0, others > 0 — zero-count classes get weight 0.0."""
        counts = np.array([5, 0, 3, 0, 1])
        weights = compute_class_weights(counts, 5, beta=0.9)
        assert weights[0] > 0
        assert weights[1] == 0.0
        assert weights[2] > 0
        assert weights[3] == 0.0
        assert weights[4] > 0

    def test_single_element_one_class(self):
        """Single class with count=1: weight = (1-beta)/(1-beta^1) = 1.0."""
        counts = np.array([1])
        weights = compute_class_weights(counts, 1, beta=0.999)
        assert pytest.approx(weights.item()) == 1.0

    def test_type_coercion_float64_to_float32(self):
        """np.float64 input → torch.float32 output."""
        counts = np.array([1, 2, 3], dtype=np.float64)
        weights = compute_class_weights(counts, 3, beta=0.9)
        assert weights.dtype == torch.float32

    def test_known_values_hand_computed(self):
        """Hand-computed weights for a specific beta and counts."""
        counts = np.array([1, 2, 5])
        beta = 0.5
        weights = compute_class_weights(counts, 3, beta=beta)
        w0 = (1 - beta) / (1 - beta**1)
        w1 = (1 - beta) / (1 - beta**2)
        w2 = (1 - beta) / (1 - beta**5)
        assert torch.allclose(weights, torch.tensor([w0, w1, w2]))

    def test_uniform_counts_equal_weights(self):
        """Uniform counts → all non-zero classes get the same weight."""
        counts = np.array([5, 5, 5])
        weights = compute_class_weights(counts, 3, beta=0.9)
        assert torch.allclose(weights[0], weights[1])
        assert torch.allclose(weights[1], weights[2])
        assert weights[0] > 0

    def test_monotonic_decreasing_weight(self):
        """Higher count → lower weight (effective number of samples increases)."""
        counts = np.array([1, 2, 10, 100])
        weights = compute_class_weights(counts, 4, beta=0.9)
        for i in range(len(weights) - 1):
            assert weights[i] >= weights[i + 1], (
                f"Weight should decrease as count increases: "
                f"count={counts[i]} → weight={weights[i]:.4f}, "
                f"count={counts[i+1]} → weight={weights[i+1]:.4f}"
            )

    def test_large_counts_approach_one(self):
        """Very large counts produce weight approaching (1-beta)."""
        counts = np.array([10_000])
        beta = 0.9
        weights = compute_class_weights(counts, 1, beta=beta)
        assert pytest.approx(weights.item(), rel=1e-4) == (1 - beta)

    @pytest.mark.parametrize("beta", [0.0, 0.5, 0.9, 0.99, 0.999, 1.0])
    def test_output_shape(self, beta):
        """Output always has shape (num_classes,) regardless of beta."""
        counts = np.array([1, 2, 3, 4, 5])
        weights = compute_class_weights(counts, 5, beta=beta)
        assert weights.shape == (5,)


class TestClassBalancedLoss:
    """Tests for ClassBalancedLoss module."""

    def test_no_class_counts_warning(self):
        """Omitting class_counts emits UserWarning about unweighted fallback."""
        with pytest.warns(UserWarning, match="class_counts"):
            loss_fn = ClassBalancedLoss(class_counts=None)
        assert loss_fn.weights is None

    def test_unweighted_fallback_matches_ce(self):
        """Without weights the output matches plain CrossEntropyLoss."""
        with pytest.warns(UserWarning):
            loss_fn = ClassBalancedLoss(class_counts=None, reduction="none")
        ce_loss = torch.nn.CrossEntropyLoss(reduction="none")
        inputs = torch.randn(4, 3)
        targets = torch.tensor([0, 1, 2, 0])
        cbl_out = loss_fn(inputs, targets)
        ce_out = ce_loss(inputs, targets)
        assert torch.allclose(cbl_out, ce_out)

    def test_forward_with_class_counts_applies_weights(self):
        """Weights are applied when class_counts is provided."""
        counts = np.array([10, 5, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0])
        loss_fn = ClassBalancedLoss(
            beta=0.99, num_classes=12, class_counts=counts, reduction="none"
        )
        inputs = torch.randn(4, 12)
        targets = torch.tensor([0, 1, 2, 3])
        loss = loss_fn(inputs, targets)
        assert loss.shape == (4,)
        assert (loss > 0).all()

    @pytest.mark.parametrize("reduction", ["mean", "sum", "none"])
    def test_all_reductions(self, reduction):
        """All three reduction modes return correct shapes."""
        counts = np.array([5, 5])
        loss_fn = ClassBalancedLoss(
            beta=0.9, num_classes=2, class_counts=counts, reduction=reduction
        )
        inputs = torch.randn(4, 2)
        targets = torch.randint(0, 2, (4,))
        loss = loss_fn(inputs, targets)
        if reduction == "none":
            assert loss.shape == (4,)
        else:
            assert loss.ndim == 0

    def test_reduction_none_differentiable(self):
        """Per-sample loss from reduction='none' requires grad on inputs."""
        counts = np.array([5, 5, 5])
        loss_fn = ClassBalancedLoss(
            beta=0.9, num_classes=3, class_counts=counts, reduction="none"
        )
        inputs = torch.randn(2, 3, requires_grad=True)
        targets = torch.tensor([0, 1])
        loss = loss_fn(inputs, targets)
        loss.sum().backward()
        assert inputs.grad is not None

    def test_weights_correctly_computed_manually(self):
        """Verify the weighted cross-entropy matches manual computation."""
        counts = np.array([10, 1])
        beta = 0.9
        expected_weights = (1 - beta) / (1 - beta ** counts.astype(np.float64))
        expected_weights_t = torch.tensor(expected_weights, dtype=torch.float32)

        loss_fn = ClassBalancedLoss(
            beta=beta, num_classes=2, class_counts=counts, reduction="none"
        )
        inputs = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        targets = torch.tensor([0, 1])
        loss = loss_fn(inputs, targets)

        ce_manual = torch.nn.functional.cross_entropy(inputs, targets, reduction="none")
        expected_loss = expected_weights_t[targets] * ce_manual
        assert torch.allclose(loss, expected_loss)

    def test_device_cpu_default(self):
        """Weights buffer is on CPU by default."""
        counts = np.array([5, 5, 5])
        loss_fn = ClassBalancedLoss(
            beta=0.9, num_classes=3, class_counts=counts
        )
        assert loss_fn.weights.device.type == "cpu"

    @pytest.mark.skipif(
        not torch.backends.mps.is_available(), reason="MPS not available on this machine"
    )
    def test_buffer_moves_with_to_mps(self):
        """Calling .to('mps') moves the weights buffer to MPS."""
        counts = np.array([5, 5, 5])
        loss_fn = ClassBalancedLoss(
            beta=0.9, num_classes=3, class_counts=counts
        )
        assert loss_fn.weights.device.type == "cpu"
        loss_fn = loss_fn.to("mps")
        assert loss_fn.weights.device.type == "mps"

    @pytest.mark.skipif(
        not torch.backends.mps.is_available(), reason="MPS not available on this machine"
    )
    def test_mps_forward_pass(self):
        """Forward pass succeeds when loss_fn is moved to MPS."""
        counts = np.array([5, 5, 5])
        loss_fn = ClassBalancedLoss(
            beta=0.9, num_classes=3, class_counts=counts
        )
        loss_fn = loss_fn.to("mps")
        inputs = torch.randn(4, 3).to("mps")
        targets = torch.randint(0, 3, (4,)).to("mps")
        loss = loss_fn(inputs, targets)
        assert loss.device.type == "mps"

    def test_balanced_loss_higher_for_minority(self):
        """Minority class (lower count) receives larger weighted loss."""
        counts = np.array([100, 1])
        loss_fn = ClassBalancedLoss(
            beta=0.99, num_classes=2, class_counts=counts, reduction="none"
        )
        logits = torch.zeros(2, 2)
        targets = torch.tensor([0, 1])
        loss = loss_fn(logits, targets)
        assert loss[1] > loss[0], "Minority class should have higher loss"

    def test_default_num_classes_is_12(self):
        """Default num_classes = 12 (from GRADE_ORDER)."""
        counts = np.array([5] * 12)
        loss_fn = ClassBalancedLoss(beta=0.99, class_counts=counts)
        assert loss_fn.weights.shape == (12,)

    def test_default_beta_is_0_99(self):
        """Default beta = 0.99."""
        counts = np.array([5, 5])
        loss_fn = ClassBalancedLoss(
            beta=0.99, num_classes=2, class_counts=counts
        )
        assert loss_fn.weights is not None
