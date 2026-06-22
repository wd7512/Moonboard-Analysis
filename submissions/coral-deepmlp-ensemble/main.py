"""submissions/coral-deepmlp-ensemble — CORAL ordinal head + DeepMLP features + focal loss.

Combines the strongest pieces of three prior submissions:
  - deep-mlp-baseline: 656-dim feature engineering (section-separated hold
    vectors + engineered meta-features + hold bigram hashes + symmetry)
  - ordinal-regression: CORAL ordinal regression head (shared linear weight
    + K-1 cumulative biases) with BCE over thresholds
  - focal-loss: focal loss weighting (1-p_t)^gamma for class imbalance

Plus the 5-model variance-reduction ensemble that made deep-mlp #1.

Architecture:
  Input(656) -> Linear(512) -> BN -> LeakyReLU(0.1) -> Dropout(0.15)
            -> Linear(256) -> BN -> LeakyReLU(0.1) -> Dropout(0.15)
            -> Linear(128) -> BN -> LeakyReLU(0.1) -> Dropout(0.15)
            -> Linear(1, bias=False) + coral_bias(12)   # CORAL head

Key innovation: bias initialization from empirical grade distribution.
For each threshold k: bias[k] = logit(P(grade > k)) from training data.
Without this, the CORAL model collapses to predicting a single grade.
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from moonboard_analysis.config import GRADE_ORDER
from moonboard_analysis.data.loader import load_lstm_data
from moonboard_analysis.data.preprocessing import (
    drop_duplicate_sequences,
    preprocess_lstm_data,
)
from moonboard_analysis.training.metrics import evaluate_classification, extract_required_metrics
from moonboard_analysis.utils.device import get_device
from moonboard_analysis.utils.reproducibility import set_seeds

NUM_COLS = 11
NUM_ROWS = 18
HOLD_VECTOR_DIM = NUM_COLS * NUM_ROWS  # 198
EXTRA_FEATURES = 62  # 8 meta + 50 bigram + 3 ratios + 1 symmetry
INPUT_DIM = HOLD_VECTOR_DIM * 3 + EXTRA_FEATURES  # 656
NUM_CLASSES = len(GRADE_ORDER)  # 13
NUM_THRESHOLDS = NUM_CLASSES - 1  # 12
GRADE_LABELS = frozenset(GRADE_ORDER)


def _hold_to_index(hold_name: str) -> int:
    if len(hold_name) < 2:
        return -1
    col_char = hold_name[0]
    if col_char < "A" or col_char > "K":
        return -1
    row_part = hold_name[1:]
    if not row_part.isdigit():
        return -1
    row = int(row_part)
    if row < 1 or row > 18:
        return -1
    col = ord(col_char) - ord("A")
    return (row - 1) * NUM_COLS + col


def _extract_sections(seq: list[str]) -> tuple[list[str], list[str], list[str]]:
    start: list[str] = []
    middle: list[str] = []
    end: list[str] = []
    section = "start"
    for token in seq:
        if token in GRADE_LABELS or token == "GRADE_END":
            break
        if token == "START_END":
            section = "middle"
        elif token == "MIDDLE_END":
            section = "end"
        elif token == "END_ROUTE":
            pass
        elif section == "start":
            start.append(token)
        elif section == "middle":
            middle.append(token)
        elif section == "end":
            end.append(token)
    return start, middle, end


def _section_to_vector(holds: list[str]) -> np.ndarray:
    vec = np.zeros(HOLD_VECTOR_DIM, dtype=np.float32)
    for h in holds:
        idx = _hold_to_index(h)
        if 0 <= idx < HOLD_VECTOR_DIM:
            vec[idx] = 1.0
    return vec


HASH_BINS = 50


def _pair_hash(i: int, j: int, n_bins: int = HASH_BINS) -> int:
    return (i * 100003 + j * 7 + 13) % n_bins


def _compute_hold_bigram_features(holds: list[str]) -> np.ndarray:
    vec = np.zeros(HASH_BINS, dtype=np.float32)
    idxs = [_hold_to_index(h) for h in holds]
    idxs = sorted(set(idx for idx in idxs if 0 <= idx < HOLD_VECTOR_DIM))
    for pos_i in range(len(idxs)):
        for pos_j in range(pos_i + 1, len(idxs)):
            bin_idx = _pair_hash(idxs[pos_i], idxs[pos_j])
            vec[bin_idx] += 1.0
    return vec


def _sequences_to_features(sequences: list[list[str]]) -> np.ndarray:
    """Convert sequences to (N, INPUT_DIM) feature matrix."""
    base = 3 * HOLD_VECTOR_DIM
    n = len(sequences)
    features = np.zeros((n, INPUT_DIM), dtype=np.float32)
    for i, seq in enumerate(sequences):
        start, middle, end = _extract_sections(seq)
        features[i, 0:HOLD_VECTOR_DIM] = _section_to_vector(start)
        features[i, HOLD_VECTOR_DIM:2 * HOLD_VECTOR_DIM] = _section_to_vector(middle)
        features[i, 2 * HOLD_VECTOR_DIM:base] = _section_to_vector(end)
        all_holds = start + middle + end
        n_s, n_m, n_e = len(start), len(middle), len(end)
        n_total = n_s + n_m + n_e
        features[i, base + 0] = np.log1p(n_s)
        features[i, base + 1] = np.log1p(n_m)
        features[i, base + 2] = np.log1p(n_e)
        features[i, base + 3] = np.log1p(n_total)
        if all_holds:
            idxs = [_hold_to_index(h) for h in all_holds]
            idxs = [idx for idx in idxs if 0 <= idx < HOLD_VECTOR_DIM]
            rows = [idx // NUM_COLS for idx in idxs]
            cols = [idx % NUM_COLS for idx in idxs]
            features[i, base + 4] = (max(rows) - min(rows) + 1) / NUM_ROWS
            features[i, base + 5] = np.mean(rows) / NUM_ROWS
            features[i, base + 6] = np.mean(cols) / NUM_COLS
        else:
            features[i, base + 4] = 0.0
            features[i, base + 5] = 0.0
            features[i, base + 6] = 0.0
        n_sections = sum(1 for s in [n_s, n_m, n_e] if s > 0)
        features[i, base + 7] = n_sections / 3.0
        bigram_feats = _compute_hold_bigram_features(all_holds)
        features[i, base + 8:base + 8 + HASH_BINS] = bigram_feats
        eps = 1e-8
        features[i, base + 8 + HASH_BINS + 0] = n_s / (n_total + eps)
        features[i, base + 8 + HASH_BINS + 1] = n_m / (n_total + eps)
        features[i, base + 8 + HASH_BINS + 2] = n_e / (n_total + eps)
        if all_holds:
            idxs = [_hold_to_index(h) for h in all_holds]
            idxs = [idx for idx in idxs if 0 <= idx < HOLD_VECTOR_DIM]
            cols_i = [idx % NUM_COLS for idx in idxs]
            n_left = sum(1 for c in cols_i if c < 5)
            n_right = sum(1 for c in cols_i if c > 5)
            features[i, base + 8 + HASH_BINS + 3] = (
                1.0 - abs(n_left - n_right) / (n_left + n_right + eps)
            )
        else:
            features[i, base + 8 + HASH_BINS + 3] = 0.0
    return features


def _init_weights(module: nn.Module) -> None:
    if isinstance(module, nn.Linear):
        nn.init.kaiming_normal_(module.weight, mode="fan_in", nonlinearity="leaky_relu", a=0.1)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)


def _compute_bias_init(labels: np.ndarray, num_thresholds: int) -> np.ndarray:
    """Initialize CORAL biases from empirical grade distribution."""
    biases = np.zeros(num_thresholds, dtype=np.float64)
    for k in range(num_thresholds):
        p = np.mean(labels > k)
        p = np.clip(p, 1e-4, 1 - 1e-4)
        biases[k] = np.log(p / (1 - p))
    return biases


class CORALNet(nn.Module):
    def __init__(self, input_dim: int, num_thresholds: int, dropout: float = 0.15,
                 bias_init: np.ndarray | None = None):
        super().__init__()
        self.num_thresholds = num_thresholds
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout),
        )
        self.coral_weight = nn.Linear(128, 1, bias=False)
        self.coral_bias = nn.Parameter(torch.zeros(num_thresholds))
        self.apply(_init_weights)
        if bias_init is not None:
            with torch.no_grad():
                self.coral_bias.copy_(torch.tensor(bias_init, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.net(x)
        shared = self.coral_weight(h)
        logits = shared + self.coral_bias.unsqueeze(0)
        return logits


class FocalBCELoss(nn.Module):
    def __init__(self, gamma: float = 2.0):
        super().__init__()
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        p = torch.sigmoid(logits)
        pt = p * targets + (1 - p) * (1 - targets)
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        focal = (1.0 - pt) ** self.gamma * bce
        return focal.mean()


def _labels_to_ordinal(labels: np.ndarray, num_thresholds: int) -> np.ndarray:
    targets = np.zeros((len(labels), num_thresholds), dtype=np.float32)
    for i, g in enumerate(labels):
        if g > 0:
            targets[i, :g] = 1.0
    return targets


def _ordinal_to_label(logits: torch.Tensor) -> torch.Tensor:
    return (torch.sigmoid(logits) > 0.5).sum(dim=1)


def _train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    n_batches = 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        loss = criterion(model(xb), yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        n_batches += 1
    return total_loss / max(n_batches, 1)


def _eval_loss(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    n_batches = 0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            total_loss += criterion(model(xb), yb).item()
            n_batches += 1
    return total_loss / max(n_batches, 1)


def _extract_logits(model, loader, device):
    all_logits = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            features = batch[0].to(device)
            logits = model(features)
            all_logits.append(logits.cpu().numpy())
    return np.concatenate(all_logits, axis=0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="CORAL-DeepMLP ensemble — ordinal head + focal loss on 656-dim features"
    )
    parser.add_argument("--data-path", type=str, default="Raw/moonboard_problems_setup_2016.json")
    parser.add_argument("--output-dir", type=str, default=".")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    return parser.parse_args()


def train_and_evaluate(
    sequences: list[list[str]],
    grades: list[int],
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    seed: int = 42,
    epochs: int = 200,
    batch_size: int = 256,
    learning_rate: float = 0.001,
    dropout: float = 0.15,
    patience: int = 25,
    focal_gamma: float = 2.0,
) -> dict[str, float]:
    train_seqs = [sequences[i] for i in train_idx]
    test_seqs = [sequences[i] for i in test_idx]
    y_train = np.array([grades[i] for i in train_idx], dtype=np.int64)
    y_test = np.array([grades[i] for i in test_idx], dtype=np.int64)

    X_train = _sequences_to_features(train_seqs)
    X_test = _sequences_to_features(test_seqs)

    feat_mean = X_train.mean(axis=0)
    feat_std = X_train.std(axis=0) + 1e-8
    X_train = (X_train - feat_mean) / feat_std
    X_test = (X_test - feat_mean) / feat_std

    y_train_ord = _labels_to_ordinal(y_train, NUM_THRESHOLDS)
    bias_init = _compute_bias_init(y_train, NUM_THRESHOLDS)

    device = get_device()

    train_ds = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train_ord, dtype=torch.float32),
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    eval_loader = DataLoader(train_ds, batch_size=batch_size * 2, shuffle=False)
    test_loader = DataLoader(
        TensorDataset(torch.tensor(X_test, dtype=torch.float32)),
        batch_size=batch_size * 2,
    )

    criterion = FocalBCELoss(gamma=focal_gamma)

    all_logits: list[np.ndarray] = []
    for ensemble_seed in (seed, seed + 1, seed + 2):
        set_seeds(ensemble_seed)
        model = CORALNet(
            input_dim=INPUT_DIM,
            num_thresholds=NUM_THRESHOLDS,
            dropout=dropout,
            bias_init=bias_init,
        ).to(device)
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=10
        )
        best_loss = float("inf")
        best_state = None
        best_epoch = 0
        for epoch in range(epochs):
            _ = _train_epoch(model, train_loader, criterion, optimizer, device)
            val_loss = _eval_loss(model, eval_loader, criterion, device)
            scheduler.step(val_loss)
            if val_loss < best_loss:
                best_loss = val_loss
                best_epoch = epoch
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            if epoch - best_epoch >= patience:
                break
        if best_state is not None:
            model.load_state_dict(best_state)
            model.to(device)
        logits = _extract_logits(model, test_loader, device)
        all_logits.append(logits)

    avg_logits = np.mean(all_logits, axis=0)
    avg_logits_t = torch.tensor(avg_logits, dtype=torch.float32)
    y_pred = _ordinal_to_label(avg_logits_t).cpu().numpy().tolist()

    metrics = evaluate_classification(y_test.tolist(), y_pred, NUM_CLASSES)
    return extract_required_metrics(metrics)


def main() -> None:
    args = parse_args()
    set_seeds(args.seed)
    data_path = args.data_path
    if not Path(data_path).exists():
        print(f"Error: Data file not found at '{data_path}'")
        sys.exit(1)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Loading data from {data_path}")
    df = load_lstm_data(data_path)
    print(f"Raw data: {len(df)} routes")
    df = df.drop_duplicates(subset=["Name"])
    all_sequences = preprocess_lstm_data(df, augment=False)
    all_sequences = drop_duplicate_sequences(all_sequences)
    print(f"After preprocessing: {len(all_sequences)} unique sequences")
    grade_to_idx = {g: i for i, g in enumerate(GRADE_ORDER)}
    valid_seqs, valid_grades = [], []
    for seq in all_sequences:
        grade = seq[-2]
        if grade in grade_to_idx:
            valid_seqs.append(seq)
            valid_grades.append(grade_to_idx[grade])
    from sklearn.model_selection import train_test_split
    train_idx, test_idx = train_test_split(
        np.arange(len(valid_seqs)), test_size=0.2, random_state=args.seed, stratify=valid_grades,
    )
    t0 = time.time()
    results = train_and_evaluate(
        valid_seqs, valid_grades, train_idx, test_idx,
        seed=args.seed, epochs=args.epochs, batch_size=args.batch_size,
        learning_rate=args.learning_rate, dropout=args.dropout,
        patience=args.patience, focal_gamma=args.focal_gamma,
    )
    elapsed = time.time() - t0
    print(f"\n{'=' * 50}\nEvaluation Results\n{'=' * 50}")
    print(f"Exact Accuracy:     {results['exact_accuracy']:.4f}")
    print(f"Within-1 Accuracy:  {results['within_one_grade']:.4f}")
    print(f"Within-2 Accuracy:  {results['within_two_grades']:.4f}")
    print(f"Macro-F1:           {results['macro_f1']:.4f}")
    print(f"Training time:      {elapsed:.1f}s")


if __name__ == "__main__":
    main()
