"""Tests for data loading and preprocessing functionality."""

import numpy as np
import pandas as pd
import torch

from moonboard_analysis.data.dataset import AutoencoderDataset, LSTMSequenceDataset
from moonboard_analysis.data.preprocessing import (
    drop_duplicate_sequences,
    get_sorted_descriptions_from_dict,
    preprocess_lstm_data,
)


class TestAutoencoderDataset:
    """Test autoencoder dataset class."""

    def test_dataset_length(self, sample_autoencoder_features: np.ndarray) -> None:
        """Verify dataset length matches number of feature vectors."""
        features = torch.tensor(sample_autoencoder_features, dtype=torch.float32)
        dataset = AutoencoderDataset(features)
        assert len(dataset) == sample_autoencoder_features.shape[0]

    def test_dataset_item_shape(self, sample_autoencoder_features: np.ndarray) -> None:
        """Verify individual item has correct feature dimension."""
        features = torch.tensor(sample_autoencoder_features, dtype=torch.float32)
        dataset = AutoencoderDataset(features)
        item = dataset[0]
        assert item.shape == (sample_autoencoder_features.shape[1],)

    def test_dataset_indexing(self, sample_autoencoder_features: np.ndarray) -> None:
        """Verify dataset returns correct items for given indices."""
        features = torch.tensor(sample_autoencoder_features, dtype=torch.float32)
        dataset = AutoencoderDataset(features)
        assert torch.equal(dataset[0], features[0])
        assert torch.equal(dataset[-1], features[-1])


class TestLSTMSequenceDataset:
    """Test LSTM sequence dataset class."""

    def test_dataset_length(
        self,
        sample_lstm_sequences: list[list[str]],
        sample_grades: list[int],
        sample_vocab: dict[str, int],
    ) -> None:
        """Verify dataset length matches number of sequences."""
        max_length = 20
        dataset = LSTMSequenceDataset(
            sample_lstm_sequences, sample_grades, sample_vocab, max_length
        )
        assert len(dataset) == len(sample_lstm_sequences)

    def test_dataset_item_types(
        self,
        sample_lstm_sequences: list[list[str]],
        sample_grades: list[int],
        sample_vocab: dict[str, int],
    ) -> None:
        """Verify dataset items are tensors with correct dtypes."""
        max_length = 20
        dataset = LSTMSequenceDataset(
            sample_lstm_sequences, sample_grades, sample_vocab, max_length
        )
        seq, grade = dataset[0]
        assert seq.dtype == torch.long
        assert grade.dtype == torch.long

    def test_sequence_padding(
        self,
        sample_vocab: dict[str, int],
    ) -> None:
        """Verify sequences are padded to max_length with zeros."""
        max_length = 10
        sequences = [["H1", "H2"], ["H3"]]
        grades = [0, 1]
        dataset = LSTMSequenceDataset(sequences, grades, sample_vocab, max_length)

        seq, _ = dataset[0]
        assert len(seq) == max_length
        assert seq[-1] == 0

    def test_sequence_truncation(
        self,
        sample_vocab: dict[str, int],
    ) -> None:
        """Verify sequences longer than max_length are truncated."""
        max_length = 3
        sequences = [["H1", "H2", "H3", "H4", "H5"]]
        grades = [0]
        dataset = LSTMSequenceDataset(sequences, grades, sample_vocab, max_length)

        seq, _ = dataset[0]
        assert len(seq) == max_length


class TestPreprocessing:
    """Test data preprocessing functions."""

    def test_get_sorted_descriptions_structure(
        self, sample_raw_dataframe: pd.DataFrame
    ) -> None:
        """Verify output contains expected marker tokens."""
        moves = sample_raw_dataframe["Moves"].iloc[0]
        grade = sample_raw_dataframe["Grade"].iloc[0]
        result = get_sorted_descriptions_from_dict(moves, grade)

        assert isinstance(result, list)
        assert len(result) >= 1
        assert "START_END" in result[0]
        assert "END_ROUTE" in result[0]
        assert "GRADE_END" in result[0]

    def test_preprocess_lstm_data_columns(
        self, sample_raw_dataframe: pd.DataFrame
    ) -> None:
        """Verify preprocessing returns non-empty list of sequences."""
        result = preprocess_lstm_data(sample_raw_dataframe)
        assert isinstance(result, list)
        assert len(result) > 0

    def test_preprocess_drops_grades(
        self, sample_raw_dataframe: pd.DataFrame
    ) -> None:
        """Verify grades in GRADES_TO_DROP are excluded."""
        df = sample_raw_dataframe.copy()
        df = pd.concat(
            [
                df,
                pd.DataFrame(
                    {
                        "Method": ["Flash"],
                        "Grade": ["8B"],
                        "Name": ["DropRoute"],
                        "Rating": [50],
                        "Repeats": [10],
                        "Moves": [
                            [
                                {"Description": "H1", "IsStart": True, "IsEnd": False},
                                {"Description": "H2", "IsStart": False, "IsEnd": True},
                            ]
                        ],
                    }
                ),
            ],
            ignore_index=True,
        )
        result = preprocess_lstm_data(df)
        for seq in result:
            assert "8B" not in seq

    def test_drop_duplicate_sequences(self) -> None:
        """Verify duplicate sequences are removed while preserving order."""
        sequences = [["A", "B"], ["A", "B"], ["C", "D"], ["A", "B"]]
        result = drop_duplicate_sequences(sequences)

        assert len(result) == 2
        assert result[0] == ["A", "B"]
        assert result[1] == ["C", "D"]

    def test_drop_duplicate_sequences_no_duplicates(self) -> None:
        """Verify function returns unchanged list when no duplicates exist."""
        sequences = [["A"], ["B"], ["C"]]
        result = drop_duplicate_sequences(sequences)
        assert result == sequences
