import torch
from torch.utils.data import Dataset


class AutoencoderDataset(Dataset):
    """Dataset for autoencoder training on binary hold feature vectors."""

    def __init__(self, features: torch.Tensor):
        self.features = features

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.features[idx]


class LSTMSequenceDataset(Dataset):
    """Dataset for LSTM grade classification on tokenised sequences."""

    def __init__(
        self,
        sequences: list[list[str]],
        grades: list[int],
        vocab: dict[str, int],
        max_length: int,
    ):
        self.sequences = sequences
        self.grades = grades
        self.vocab = vocab
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        sequence = self.sequences[idx]
        seq_indices = [self.vocab.get(hold, 0) for hold in sequence]
        seq_indices = seq_indices[: self.max_length] + [0] * (
            self.max_length - len(seq_indices)
        )
        return torch.tensor(seq_indices, dtype=torch.long), torch.tensor(
            self.grades[idx], dtype=torch.long
        )
