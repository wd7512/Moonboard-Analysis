"""Tests for model forward pass shapes and basic functionality."""

import torch

from moonboard_analysis.config import GRADE_ORDER
from moonboard_analysis.models.autoencoder import Autoencoder
from moonboard_analysis.models.lstm import ClimbingGradePredictor


class TestAutoencoderShapes:
    """Test autoencoder forward pass output shapes."""

    def test_forward_output_shape(
        self, mock_autoencoder: Autoencoder, sample_autoencoder_features: torch.Tensor
    ) -> None:
        """Verify forward pass returns tensor with matching input shape."""
        features = torch.tensor(sample_autoencoder_features, dtype=torch.float32)
        output = mock_autoencoder(features)
        assert output.shape == features.shape

    def test_encode_output_shape(
        self, mock_autoencoder: Autoencoder, sample_autoencoder_features: torch.Tensor
    ) -> None:
        """Verify encode method returns tensor with bottleneck dimension."""
        features = torch.tensor(sample_autoencoder_features, dtype=torch.float32)
        encoded = mock_autoencoder.encode(features)
        assert encoded.shape == (features.shape[0], mock_autoencoder.bottleneck_dim)

    def test_decode_output_shape(
        self, mock_autoencoder: Autoencoder, sample_autoencoder_features: torch.Tensor
    ) -> None:
        """Verify decode method returns tensor with input dimension."""
        batch_size = sample_autoencoder_features.shape[0]
        bottleneck_input = torch.randn(
            batch_size, mock_autoencoder.bottleneck_dim, dtype=torch.float32
        )
        decoded = mock_autoencoder.decode(bottleneck_input)
        assert decoded.shape == (batch_size, mock_autoencoder.input_dim)

    def test_forward_output_range(
        self, mock_autoencoder: Autoencoder, sample_autoencoder_features: torch.Tensor
    ) -> None:
        """Verify forward pass output is in [0, 1] range due to Sigmoid."""
        features = torch.tensor(sample_autoencoder_features, dtype=torch.float32)
        output = mock_autoencoder(features)
        assert output.min() >= 0.0
        assert output.max() <= 1.0

    def test_different_batch_sizes(
        self, mock_autoencoder: Autoencoder
    ) -> None:
        """Verify forward pass works with various batch sizes."""
        mock_autoencoder.eval()
        for batch_size in [1, 8, 64, 128]:
            x = torch.randn(batch_size, mock_autoencoder.input_dim, dtype=torch.float32)
            output = mock_autoencoder(x)
            assert output.shape == (batch_size, mock_autoencoder.input_dim)


class TestLSTMShapes:
    """Test LSTM grade predictor forward pass output shapes."""

    def test_forward_output_shape(
        self,
        mock_lstm: ClimbingGradePredictor,
        sample_lstm_sequences: list[list[str]],
        sample_vocab: dict[str, int],
    ) -> None:
        """Verify forward pass returns logits with shape (batch_size, num_classes)."""
        max_length = 20
        batch_size = len(sample_lstm_sequences)
        seq_indices = [
            [sample_vocab.get(h, 0) for h in seq[:max_length]]
            + [0] * (max_length - min(len(seq), max_length))
            for seq in sample_lstm_sequences
        ]
        x = torch.tensor(seq_indices, dtype=torch.long)
        output = mock_lstm(x)
        assert output.shape == (batch_size, len(GRADE_ORDER))

    def test_forward_output_is_logits(
        self, mock_lstm: ClimbingGradePredictor
    ) -> None:
        """Verify forward pass outputs are unnormalized logits (not probabilities)."""
        batch_size = 4
        seq_length = 10
        x = torch.randint(0, 100, (batch_size, seq_length), dtype=torch.long)
        output = mock_lstm(x)
        assert output.min() < 0 or output.max() > 1

    def test_single_sequence(
        self, mock_lstm: ClimbingGradePredictor
    ) -> None:
        """Verify forward pass works with a single sequence."""
        x = torch.randint(0, 100, (1, 5), dtype=torch.long)
        output = mock_lstm(x)
        assert output.shape == (1, len(GRADE_ORDER))

    def test_variable_sequence_lengths(
        self, mock_lstm: ClimbingGradePredictor
    ) -> None:
        """Verify forward pass works with different sequence lengths."""
        for seq_length in [1, 5, 20, 50]:
            x = torch.randint(0, 100, (4, seq_length), dtype=torch.long)
            output = mock_lstm(x)
            assert output.shape == (4, len(GRADE_ORDER))
