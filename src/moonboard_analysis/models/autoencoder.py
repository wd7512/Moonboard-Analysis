import torch
import torch.nn as nn


class Autoencoder(nn.Module):
    """Autoencoder for Moonboard route compression.

    Encoder: input_dim -> 64 -> ReLU -> BatchNorm -> bottleneck_dim -> activation
    Decoder: bottleneck_dim -> 64 -> ReLU -> BatchNorm -> input_dim -> Sigmoid

    When bounded=True, the bottleneck uses tanh to constrain latent values
    to [-1, 1] range, enabling fixed slider ranges in the visualization app.
    """

    def __init__(self, input_dim: int, bottleneck_dim: int, bounded: bool = False):
        super().__init__()
        self.input_dim = input_dim
        self.bottleneck_dim = bottleneck_dim
        self.bounded = bounded

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Linear(64, bottleneck_dim),
            nn.Tanh() if bounded else nn.ReLU(),
        )

        self.decoder = nn.Sequential(
            nn.Linear(bottleneck_dim, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Linear(64, input_dim),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(x)
