import random

import numpy as np
import torch


def set_seeds(seed: int = 42) -> None:
    """Set random seeds for reproducibility across numpy, torch, and Python random."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
