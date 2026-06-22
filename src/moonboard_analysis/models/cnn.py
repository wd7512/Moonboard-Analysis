"""Shared 2D CNN model definitions and factory."""

from typing import Literal

import torch
import torch.nn as nn

CNNArchitectureName = Literal["2dcnn-baseline", "multichannel-2dcnn"]


class CNN2DGradePredictor(nn.Module):
    """Configurable 2D CNN for spatial hold data.

    Supports both single-channel (18x11) and multi-channel (3x18x11) variants.
    """

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 12,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_layers(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class MultiChannelCNN2D(nn.Module):
    """Compact 2D CNN with configurable input channels.

    Suitable for multi-channel spatial data (e.g. start/middle/end layers).
    """

    def __init__(self, in_channels: int = 3, num_classes: int = 12):
        super().__init__()
        self.in_channels = in_channels
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(32, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def create_cnn(
    architecture: CNNArchitectureName = "2dcnn-baseline",
    num_classes: int = 12,
    dropout: float = 0.5,
) -> nn.Module:
    """Factory: build a 2D CNN from a named architecture.

    Args:
        architecture: '2dcnn-baseline' (single-channel) or 'multichannel-2dcnn'.
        num_classes: Number of output classes.
        dropout: Dropout probability (only used by '2dcnn-baseline').

    Returns:
        Configured CNN module.
    """
    if architecture == "2dcnn-baseline":
        return CNN2DGradePredictor(in_channels=1, num_classes=num_classes, dropout=dropout)
    elif architecture == "multichannel-2dcnn":
        return MultiChannelCNN2D(in_channels=3, num_classes=num_classes)
    else:
        raise ValueError(f"Unknown CNN architecture: {architecture}")
