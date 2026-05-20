import numpy as np
import pandas as pd


def load_autoencoder_data(path: str) -> np.ndarray:
    """Load preprocessed npy file with shape (n_routes, 165) where
    column 0 is grade label and columns 1..164 are binary hold features."""
    data = np.load(path, allow_pickle=True)
    return data


def load_lstm_data(path: str) -> pd.DataFrame:
    """Load raw Moonboard JSON data into a DataFrame."""
    df = pd.read_json(path).T
    return df
