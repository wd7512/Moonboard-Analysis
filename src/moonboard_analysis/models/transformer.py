"""Shared Transformer model definition and factory."""

import math
from typing import cast

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for Transformer."""

    def __init__(self, d_model: int, max_len: int = 256):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pe = cast(torch.Tensor, self.pe)
        return x + pe[:, : x.size(1), :]


class TransformerGradePredictor(nn.Module):
    """Configurable TransformerEncoder for Moonboard grade prediction.

    Architecture:
        Embedding(vocab_size -> d_model) -> PositionalEncoding ->
        TransformerEncoder(n_layers, n_heads) -> MeanPool -> Linear
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 64,
        nhead: int = 2,
        num_layers: int = 2,
        dim_feedforward: int = 128,
        num_classes: int = 12,
        max_len: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_encoder = PositionalEncoding(d_model, max_len)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="relu",
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        x = self.embedding(x)
        x = self.pos_encoder(x)
        x = self.transformer(x)
        if mask is not None:
            active = 1.0 - mask.unsqueeze(-1).float()
            x = x * active
            lengths = active.sum(dim=1)
            x = x.sum(dim=1) / lengths.clamp(min=1)
        else:
            x = x.mean(dim=1)
        return self.classifier(x)


def create_transformer(
    vocab_size: int,
    d_model: int = 64,
    nhead: int = 2,
    num_layers: int = 2,
    dim_feedforward: int = 128,
    num_classes: int = 12,
    max_len: int = 256,
    dropout: float = 0.1,
) -> TransformerGradePredictor:
    """Factory: build a TransformerGradePredictor."""
    return TransformerGradePredictor(
        vocab_size=vocab_size,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        dim_feedforward=dim_feedforward,
        num_classes=num_classes,
        max_len=max_len,
        dropout=dropout,
    )
