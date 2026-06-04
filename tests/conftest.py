"""Pytest fixtures for Moonboard analysis tests."""

import numpy as np
import pandas as pd
import pytest

from moonboard_analysis.config import GRADE_ORDER, AutoencoderConfig, LSTMConfig
from moonboard_analysis.models.autoencoder import Autoencoder
from moonboard_analysis.models.lstm import ClimbingGradePredictor


@pytest.fixture
def autoencoder_config() -> AutoencoderConfig:
    """Return a default autoencoder configuration."""
    return AutoencoderConfig()


@pytest.fixture
def lstm_config() -> LSTMConfig:
    """Return a default LSTM configuration."""
    return LSTMConfig()


@pytest.fixture
def sample_autoencoder_features() -> np.ndarray:
    """Generate sample binary hold feature vectors for autoencoder tests.

    Returns:
        np.ndarray of shape (32, 164) with binary values.
    """
    rng = np.random.default_rng(42)
    return rng.integers(0, 2, size=(32, 164)).astype(np.float32)


@pytest.fixture
def sample_lstm_sequences() -> list[list[str]]:
    """Generate sample tokenised route sequences for LSTM tests.

    Returns:
        List of 32 sequences, each containing hold token strings.
    """
    holds = ["H1", "H2", "H3", "H4", "H5", "START_END", "MIDDLE_END", "END_ROUTE", "GRADE_END"]
    rng = np.random.default_rng(42)
    sequences: list[list[str]] = []
    for _ in range(32):
        length = rng.integers(5, 15)
        seq = [rng.choice(holds).item() for _ in range(length)]
        sequences.append(seq)
    return sequences


@pytest.fixture
def sample_grades() -> list[int]:
    """Generate sample grade labels for LSTM tests.

    Returns:
        List of 32 integer grade labels in range [0, 11].
    """
    rng = np.random.default_rng(42)
    return rng.integers(0, 12, size=32).tolist()


@pytest.fixture
def sample_vocab() -> dict[str, int]:
    """Return a sample vocabulary mapping for LSTM tests."""
    return {
        "H1": 1,
        "H2": 2,
        "H3": 3,
        "H4": 4,
        "H5": 5,
        "START_END": 6,
        "MIDDLE_END": 7,
        "END_ROUTE": 8,
        "GRADE_END": 9,
    }


@pytest.fixture
def mock_autoencoder(autoencoder_config: AutoencoderConfig) -> Autoencoder:
    """Return an untrained autoencoder model."""
    return Autoencoder(
        input_dim=autoencoder_config.input_dim,
        bottleneck_dim=autoencoder_config.bottleneck_dim,
    )


@pytest.fixture
def mock_lstm(lstm_config: LSTMConfig) -> ClimbingGradePredictor:
    """Return an untrained LSTM grade predictor model."""
    return ClimbingGradePredictor(
        vocab_size=100,
        embed_dim=lstm_config.embed_dim,
        hidden_dim=lstm_config.hidden_dim,
        num_layers=lstm_config.num_layers,
        num_classes=len(GRADE_ORDER),
    )


@pytest.fixture
def sample_raw_dataframe() -> pd.DataFrame:
    """Generate a sample raw DataFrame mimicking Moonboard JSON data.

    Returns:
        pd.DataFrame with columns: Method, Grade, Name, Rating, Repeats, Moves.
    """
    data = {
        "Method": ["Flash", "Redpoint", "Flash"],
        "Grade": ["6A", "6B+", "7A"],
        "Name": ["Route1", "Route2", "Route3"],
        "Rating": [100, 200, 150],
        "Repeats": [50, 30, 40],
        "Moves": [
            [
                {"Description": "H1", "IsStart": True, "IsEnd": False},
                {"Description": "H2", "IsStart": False, "IsEnd": False},
                {"Description": "H3", "IsStart": False, "IsEnd": True},
            ],
            [
                {"Description": "H4", "IsStart": True, "IsEnd": False},
                {"Description": "H5", "IsStart": False, "IsEnd": False},
                {"Description": "H1", "IsStart": False, "IsEnd": True},
            ],
            [
                {"Description": "H2", "IsStart": True, "IsEnd": False},
                {"Description": "H3", "IsStart": False, "IsEnd": False},
                {"Description": "H4", "IsStart": False, "IsEnd": True},
            ],
        ],
    }
    return pd.DataFrame(data)
