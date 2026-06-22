"""Shared training loop and loss functions for submission training."""

import warnings
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from moonboard_analysis.training.metrics import (
    evaluate_classification,
    extract_required_metrics,
)

LossType = Literal["ce", "focal", "class_balanced"]


@dataclass
class TrainingConfig:
    """Hyperparameter configuration for shared training loops."""

    epochs: int = 100
    lr: float = 0.001
    batch_size: int = 256
    weight_decay: float = 0.0
    label_smoothing: float = 0.0
    patience: int = 15
    scheduler_type: str = "reduce_on_plateau"
    scheduler_params: dict[str, Any] = field(
        default_factory=lambda: {"factor": 0.5, "patience": 10}
    )
    loss_type: LossType = "ce"
    loss_params: dict[str, Any] = field(default_factory=dict)


class FocalLoss(nn.Module):
    """Multi-class Focal Loss.

    FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)
    """

    def __init__(
        self, gamma: float = 2.0, alpha: torch.Tensor | None = None, reduction: str = "mean"
    ):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(inputs, targets, reduction="none")
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        if self.alpha is not None:
            alpha_t = self.alpha.to(inputs.device).gather(0, targets)
            focal_loss = alpha_t * focal_loss
        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        return focal_loss


def compute_class_weights(counts: np.ndarray, num_classes: int, beta: float) -> torch.Tensor:
    """Compute class-balanced weights from effective number of samples.

    w_y = (1 - β) / (1 - β^{n_y})
    """
    counts = counts.astype(np.float64)
    if len(counts) < num_classes:
        counts = np.pad(counts, (0, num_classes - len(counts)), constant_values=0)
    elif len(counts) > num_classes:
        counts = counts[:num_classes]
    if beta == 1.0:
        weights = np.divide(
            1.0,
            counts,
            where=counts > 0,
            out=np.zeros_like(counts, dtype=np.float64),
        )
    else:
        weights = np.divide(
            1.0 - beta,
            1.0 - np.power(beta, counts),
            where=counts > 0,
            out=np.full_like(counts, 0.0, dtype=np.float64),
        )
    return torch.tensor(weights, dtype=torch.float32)


class ClassBalancedLoss(nn.Module):
    """Class-Balanced Loss based on Effective Number of Samples.

    Cui et al., CVPR 2019.
    """

    def __init__(
        self,
        beta: float = 0.99,
        num_classes: int = 12,
        reduction: str = "mean",
        class_counts: np.ndarray | None = None,
    ):
        super().__init__()
        self.reduction = reduction
        if class_counts is not None:
            weights = compute_class_weights(class_counts, num_classes, beta)
            self.register_buffer("weights", weights)
        else:
            warnings.warn(
                "ClassBalancedLoss created without class_counts — "
                "falling back to unweighted cross-entropy.",
                stacklevel=2,
            )
            self.weights = None

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(inputs, targets, reduction="none")
        if self.weights is not None:
            weight_per_sample = self.weights[targets]
            ce_loss = weight_per_sample * ce_loss
        if self.reduction == "mean":
            return ce_loss.mean()
        elif self.reduction == "sum":
            return ce_loss.sum()
        return ce_loss


def build_criterion(
    config: TrainingConfig,
    num_classes: int,
    y_train: np.ndarray | None = None,
    device: torch.device | None = None,
) -> nn.Module:
    """Build loss function from TrainingConfig.

    Args:
        config: Training configuration specifying loss type and parameters.
        num_classes: Number of output classes.
        y_train: Training labels (needed for class-balanced loss).
        device: Target device.

    Returns:
        Loss module.
    """
    if config.loss_type == "ce":
        return nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)
    elif config.loss_type == "focal":
        gamma = config.loss_params.get("gamma", 2.0)
        return FocalLoss(gamma=gamma)
    elif config.loss_type == "class_balanced":
        beta = config.loss_params.get("beta", 0.999)
        class_counts = None
        if y_train is not None:
            class_counts = np.bincount(y_train, minlength=num_classes)
        return ClassBalancedLoss(beta=beta, num_classes=num_classes, class_counts=class_counts)
    else:
        raise ValueError(f"Unknown loss type: {config.loss_type}")


def build_optimizer(model: nn.Module, config: TrainingConfig) -> optim.Optimizer:
    """Build optimizer from TrainingConfig."""
    return optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)


def build_scheduler(optimizer: optim.Optimizer, config: TrainingConfig) -> object:
    """Build learning rate scheduler from TrainingConfig."""
    if config.scheduler_type == "reduce_on_plateau":
        return optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", **config.scheduler_params
        )
    elif config.scheduler_type == "cosine":
        return optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config.epochs, **config.scheduler_params
        )
    else:
        raise ValueError(f"Unknown scheduler type: {config.scheduler_type}")


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> float:
    """Train for one epoch, return average loss."""
    model.train()
    total_loss = 0.0
    n_batches = 0
    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        loss = criterion(model(inputs), labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        n_batches += 1
    return total_loss / max(n_batches, 1)


@torch.no_grad()
def evaluate_loss(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """Evaluate loss on a DataLoader."""
    model.eval()
    total_loss = 0.0
    n_batches = 0
    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        loss = criterion(model(inputs), labels)
        total_loss += loss.item()
        n_batches += 1
    return total_loss / max(n_batches, 1)


@torch.no_grad()
def extract_predictions(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[list[int], list[int]]:
    """Extract predictions and labels from a DataLoader."""
    all_preds: list[int] = []
    all_labels: list[int] = []
    model.eval()
    for inputs, labels in loader:
        inputs = inputs.to(device)
        outputs = model(inputs)
        preds = torch.argmax(outputs, dim=1)
        all_preds.extend(preds.cpu().numpy().tolist())
        all_labels.extend(labels.numpy().tolist())
    return all_preds, all_labels


def train_standard(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    config: TrainingConfig,
    num_classes: int,
    y_train: np.ndarray | None = None,
    device: torch.device | None = None,
) -> dict[str, float]:
    """Standard training loop with early stopping, LR scheduling, and best-state checkpointing.

    Args:
        model: The model to train.
        train_loader: Training data loader.
        test_loader: Validation/test data loader.
        config: Training hyperparameters.
        num_classes: Number of output classes.
        y_train: Training labels (needed for class-balanced loss).
        device: Target device (default: auto-detect).

    Returns:
        Dict with metrics from evaluate_classification.
    """
    if device is None:
        from moonboard_analysis.utils.device import get_device

        device = get_device()

    model = model.to(device)
    criterion = build_criterion(config, num_classes, y_train, device).to(device)
    optimizer = build_optimizer(model, config)
    scheduler = build_scheduler(optimizer, config)

    best_loss = float("inf")
    best_state = None
    best_epoch = 0

    for epoch in range(config.epochs):
        train_one_epoch(model, train_loader, criterion, optimizer, device)
        test_loss = evaluate_loss(model, test_loader, criterion, device)

        if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(test_loss)

        if test_loss < best_loss:
            best_loss = test_loss
            best_epoch = epoch
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if epoch - best_epoch >= config.patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
        model.to(device)

    all_preds, all_labels = extract_predictions(model, test_loader, device)
    metrics = evaluate_classification(all_labels, all_preds, num_classes)
    return extract_required_metrics(metrics)
