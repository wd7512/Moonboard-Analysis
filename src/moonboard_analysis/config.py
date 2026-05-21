from dataclasses import dataclass

GRADE_ORDER = ["6A", "6A+", "6B+", "6C", "6C+", "7A", "7A+", "7B", "7B+", "7C", "7C+", "8A"]


@dataclass
class AutoencoderConfig:
    input_dim: int = 164
    bottleneck_dim: int = 8
    hidden_dim: int = 64
    epochs: int = 100
    batch_size: int = 64
    learning_rate: float = 0.001
    weight_decay: float = 1e-5
    seed: int = 42
    bounded: bool = True
    data_path: str = "archive/Legacy/2016TrainingData164.npy"
    model_save_path: str = "Autoencoder_Moonboard.pth"


@dataclass
class LSTMConfig:
    embed_dim: int = 16
    hidden_dim: int = 128
    num_layers: int = 3
    num_epochs: int = 500
    batch_size: int = 32
    learning_rate: float = 0.001
    test_size: float = 0.2
    seed: int = 42
    data_path: str = "Raw/moonboard_problems_setup_2016.json"
    model_save_path: str = "LSTM_Moonboard.pth"
    max_length: int | None = None
