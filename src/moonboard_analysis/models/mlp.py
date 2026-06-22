"""Shared MLP model definitions and factory."""

from typing import Literal

import torch
import torch.nn as nn

ArchitectureName = Literal["fast-mlp", "deep-mlp", "perceptron"]

ARCHITECTURES: dict[ArchitectureName, list[int]] = {
    "fast-mlp": [198, 256, 128, 12],
    "deep-mlp": [656, 512, 256, 128, 12],
    "perceptron": [198, 128, 64, 12],
}


class MLP(nn.Module):
    """Configurable MLP with BatchNorm, dropout, and activation."""

    def __init__(
        self,
        layer_sizes: list[int],
        dropout: float = 0.3,
        activation: nn.Module = nn.ReLU(),
        use_batchnorm: bool = True,
    ):
        super().__init__()
        layers: list[nn.Module] = []
        for i in range(len(layer_sizes) - 1):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
            if i < len(layer_sizes) - 2:
                if use_batchnorm:
                    layers.append(nn.BatchNorm1d(layer_sizes[i + 1]))
                layers.append(activation)
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def create_mlp(
    architecture: ArchitectureName = "fast-mlp",
    dropout: float = 0.3,
    activation: nn.Module | None = None,
    use_batchnorm: bool = True,
    num_classes: int | None = None,
) -> MLP:
    """Factory: build an MLP from a named architecture.

    Args:
        architecture: One of 'fast-mlp', 'deep-mlp', 'perceptron'.
        dropout: Dropout probability between hidden layers.
        activation: Activation function (default: ReLU).
        use_batchnorm: Whether to insert BatchNorm1d after each Linear.
        num_classes: Override the output dimension if provided.

    Returns:
        Configured MLP instance.
    """
    layer_sizes = list(ARCHITECTURES[architecture])
    if num_classes is not None:
        layer_sizes[-1] = num_classes
    act = activation or nn.ReLU()
    return MLP(layer_sizes, dropout=dropout, activation=act, use_batchnorm=use_batchnorm)
