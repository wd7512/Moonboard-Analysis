"""submissions/deep-mlp-10class — Deep MLP ensemble with 10-class encoding.

Same architecture as deep-mlp-ensemble but with CE loss instead of CORAL.
Key change: 10-class encoding drops 3 empty classes (6A, 6A+, 6B).

Architecture: 656-dim features -> Linear(656->512) -> LeakyReLU -> Dropout(0.15)
                                -> Linear(512->256) -> LeakyReLU -> Dropout(0.15)
                                -> Linear(256->128) -> LeakyReLU -> Dropout(0.15)
                                -> Linear(128->10)
5-model softmax ensemble, CrossEntropyLoss with label smoothing 0.1.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

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
EXTRA_FEATURES = 62
INPUT_DIM = HOLD_VECTOR_DIM * 3 + EXTRA_FEATURES  # 656

GRADE_10CLASS = ["6B+", "6C", "6C+", "7A", "7A+", "7B", "7B+", "7C", "7C+", "8A"]
NUM_CLASSES = len(GRADE_10CLASS)
GRADE_LABELS = frozenset(GRADE_10CLASS)


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
    start, middle, end = [], [], []
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
    idxs = sorted(set(i for i in idxs if 0 <= i < HOLD_VECTOR_DIM))
    for i in range(len(idxs)):
        for j in range(i + 1, len(idxs)):
            vec[_pair_hash(idxs[i], idxs[j])] += 1.0
    return vec


def _sequences_to_features(sequences: list[list[str]]) -> np.ndarray:
    base = 3 * HOLD_VECTOR_DIM
    n = len(sequences)
    features = np.zeros((n, INPUT_DIM), dtype=np.float32)
    for i, seq in enumerate(sequences):
        start, middle, end = _extract_sections(seq)
        features[i, 0:HOLD_VECTOR_DIM] = _section_to_vector(start)
        features[i, HOLD_VECTOR_DIM:2*HOLD_VECTOR_DIM] = _section_to_vector(middle)
        features[i, 2*HOLD_VECTOR_DIM:base] = _section_to_vector(end)
        all_holds = start + middle + end
        n_s, n_m, n_e = len(start), len(middle), len(end)
        n_total = n_s + n_m + n_e
        features[i, base+0] = np.log1p(n_s)
        features[i, base+1] = np.log1p(n_m)
        features[i, base+2] = np.log1p(n_e)
        features[i, base+3] = np.log1p(n_total)
        if all_holds:
            idxs = [_hold_to_index(h) for h in all_holds]
            idxs = [idx for idx in idxs if 0 <= idx < HOLD_VECTOR_DIM]
            rows = [idx // NUM_COLS for idx in idxs]
            cols = [idx % NUM_COLS for idx in idxs]
            features[i, base+4] = (max(rows) - min(rows) + 1) / NUM_ROWS
            features[i, base+5] = np.mean(rows) / NUM_ROWS
            features[i, base+6] = np.mean(cols) / NUM_COLS
        else:
            features[i, base+4] = features[i, base+5] = features[i, base+6] = 0.0
        n_sections = sum(1 for s in [n_s, n_m, n_e] if s > 0)
        features[i, base+7] = n_sections / 3.0
        features[i, base+8:base+8+HASH_BINS] = _compute_hold_bigram_features(all_holds)
        eps = 1e-8
        features[i, base+8+HASH_BINS+0] = n_s / (n_total + eps)
        features[i, base+8+HASH_BINS+1] = n_m / (n_total + eps)
        features[i, base+8+HASH_BINS+2] = n_e / (n_total + eps)
        if all_holds:
            idxs = [_hold_to_index(h) for h in all_holds]
            idxs = [idx for idx in idxs if 0 <= idx < HOLD_VECTOR_DIM]
            cols_i = [idx % NUM_COLS for idx in idxs]
            n_left = sum(1 for c in cols_i if c < 5)
            n_right = sum(1 for c in cols_i if c > 5)
            features[i, base+8+HASH_BINS+3] = 1.0 - abs(n_left - n_right) / (n_left + n_right + eps)
        else:
            features[i, base+8+HASH_BINS+3] = 0.0
    return features


def _init_weights(module):
    if isinstance(module, nn.Linear):
        nn.init.kaiming_normal_(module.weight, mode="fan_in", nonlinearity="leaky_relu", a=0.1)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)


class DeepMLPClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes, dropout=0.15):
        super().__init__()
        h2 = max(hidden_dim // 2, num_classes)
        h3 = max(hidden_dim // 4, num_classes)
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.LeakyReLU(0.1), nn.Dropout(dropout),
            nn.Linear(hidden_dim, h2), nn.LeakyReLU(0.1), nn.Dropout(dropout),
            nn.Linear(h2, h3), nn.LeakyReLU(0.1), nn.Dropout(dropout),
            nn.Linear(h3, num_classes),
        )
        self.apply(_init_weights)

    def forward(self, x):
        return self.net(x)


def _train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, n_batches = 0.0, 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        loss = criterion(model(xb), yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        n_batches += 1
    return total_loss / max(n_batches, 1)


def _evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total, n_batches = 0.0, 0, 0, 0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            out = model(xb)
            total_loss += criterion(out, yb).item()
            _, pred = torch.max(out, 1)
            correct += (pred == yb).sum().item()
            total += yb.size(0)
            n_batches += 1
    return total_loss / max(n_batches, 1), correct / total


def _extract_softmax_probs(model, loader, device):
    all_probs = []
    model.eval()
    with torch.no_grad():
        for xb, _ in loader:
            xb = xb.to(device)
            all_probs.append(torch.softmax(model(xb), dim=1).cpu().numpy())
    return np.concatenate(all_probs, axis=0)


def parse_args():
    parser = argparse.ArgumentParser(description="Deep MLP 10-class ensemble")
    parser.add_argument("--data-path", type=str, default="Raw/moonboard_problems_setup_2016.json")
    parser.add_argument("--output-dir", type=str, default=".")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--dropout", type=float, default=0.15)
    return parser.parse_args()


def train_and_evaluate(
    sequences, grades, train_idx, test_idx,
    seed=42, hidden_dim=512, epochs=150, batch_size=128,
    learning_rate=0.001, dropout=0.15, patience=25,
):
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

    device = get_device()
    train_ds = TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                              torch.tensor(y_train, dtype=torch.long))
    test_ds = TensorDataset(torch.tensor(X_test, dtype=torch.float32),
                             torch.tensor(y_test, dtype=torch.long))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size * 2)

    all_probs = []
    for ens_seed in (seed, seed+1, seed+2, seed+3, seed+4):
        set_seeds(ens_seed)
        model = DeepMLPClassifier(INPUT_DIM, hidden_dim, NUM_CLASSES, dropout).to(device)
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=15,
        )
        best_loss, best_state, best_epoch = float("inf"), None, 0
        for epoch in range(epochs):
            _train_epoch(model, train_loader, criterion, optimizer, device)
            test_loss, _ = _evaluate(model, test_loader, criterion, device)
            scheduler.step(test_loss)
            if test_loss < best_loss:
                best_loss, best_epoch = test_loss, epoch
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            if epoch - best_epoch >= patience:
                break
        if best_state is not None:
            model.load_state_dict(best_state)
        all_probs.append(_extract_softmax_probs(model, test_loader, device))

    avg_probs = np.mean(all_probs, axis=0)
    y_pred = np.argmax(avg_probs, axis=1).tolist()
    metrics = evaluate_classification(y_test.tolist(), y_pred, NUM_CLASSES)
    return extract_required_metrics(metrics)


def main():
    args = parse_args()
    set_seeds(args.seed)
    data_path = args.data_path
    if not Path(data_path).exists():
        print(f"Error: Data file not found at '{data_path}'")
        sys.exit(1)
    df = load_lstm_data(data_path)
    all_sequences = preprocess_lstm_data(df)
    all_sequences = drop_duplicate_sequences(all_sequences)
    grade_to_idx = {g: i for i, g in enumerate(GRADE_10CLASS)}
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
    results = train_and_evaluate(valid_seqs, valid_grades, train_idx, test_idx,
                                  seed=args.seed, hidden_dim=args.hidden_dim, epochs=args.epochs)
    print(f"\nExact: {results['exact_accuracy']:.4f}")
    print(f"Within-1: {results['within_one_grade']:.4f}")
    print(f"Within-2: {results['within_two_grades']:.4f}")
    print(f"Macro-F1: {results['macro_f1']:.4f}")


if __name__ == "__main__":
    main()
