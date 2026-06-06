"""submissions/coral-engineered-f1loss — Direct macro-F1 optimization with soft-F1 loss."""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
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
HOLD_VECTOR_DIM = NUM_COLS * NUM_ROWS
NUM_CLASSES = len(GRADE_ORDER)
GRADE_LABELS = frozenset(GRADE_ORDER)


def _hold_to_index(hold_name):
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
    return (row - 1) * NUM_COLS + (ord(col_char) - ord("A"))


def _sequences_to_vectors(sequences):
    vectors = np.zeros((len(sequences), HOLD_VECTOR_DIM), dtype=np.float32)
    skip = GRADE_LABELS | {"GRADE_END", "START_END", "MIDDLE_END", "END_ROUTE"}
    for i, seq in enumerate(sequences):
        for token in seq:
            if token in skip:
                continue
            idx = _hold_to_index(token)
            if 0 <= idx < HOLD_VECTOR_DIM:
                vectors[i, idx] = 1.0
    return vectors


class CORALModel(nn.Module):
    def __init__(self, input_dim, num_classes, dropout=0.3):
        super().__init__()
        self.num_thresholds = num_classes - 1
        h = 256
        self.net = nn.Sequential(
            nn.Linear(input_dim, h), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(h, h // 2), nn.ReLU(), nn.Dropout(dropout),
        )
        self.coral_weight = nn.Linear(h // 2, 1, bias=False)
        self.coral_bias = nn.Parameter(torch.zeros(self.num_thresholds))

    def forward(self, x):
        return self.coral_weight(self.net(x)) + self.coral_bias.unsqueeze(0)


class SoftF1Loss(nn.Module):
    def __init__(self, num_classes, epsilon=1e-6):
        super().__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon

    def forward(self, logits, targets_int):
        probs = torch.sigmoid(logits)
        batch_size = probs.shape[0]
        class_probs = torch.zeros(batch_size, self.num_classes, device=logits.device)
        class_probs[:, 0] = 1.0 - probs[:, 0]
        for k in range(1, self.num_classes - 1):
            class_probs[:, k] = probs[:, k - 1] * (1.0 - probs[:, k])
        class_probs[:, -1] = probs[:, -1]
        class_probs = class_probs / (class_probs.sum(dim=1, keepdim=True) + self.epsilon)
        targets_onehot = torch.zeros_like(class_probs)
        targets_onehot.scatter_(1, targets_int.unsqueeze(1), 1.0)
        soft_tp = (class_probs * targets_onehot).sum(dim=0)
        soft_fp = (class_probs * (1 - targets_onehot)).sum(dim=0)
        soft_fn = ((1 - class_probs) * targets_onehot).sum(dim=0)
        precision = soft_tp / (soft_tp + soft_fp + self.epsilon)
        recall = soft_tp / (soft_tp + soft_fn + self.epsilon)
        f1 = 2 * precision * recall / (precision + recall + self.epsilon)
        return 1.0 - f1.mean()


def _train(model, train_loader, eval_loader, lr, dev, epochs, patience):
    criterion = SoftF1Loss(NUM_CLASSES)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=10)
    best_loss = float("inf")
    best_state = None
    best_epoch = 0
    for epoch in range(epochs):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(dev), yb.to(dev)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
        model.eval()
        eval_loss = 0.0
        n = 0
        with torch.no_grad():
            for xb, yb in eval_loader:
                xb, yb = xb.to(dev), yb.to(dev)
                eval_loss += criterion(model(xb), yb).item()
                n += 1
        eval_loss /= max(n, 1)
        scheduler.step(eval_loss)
        if eval_loss < best_loss:
            best_loss = eval_loss
            best_epoch = epoch
            best_state = {
                k: v.cpu().clone() for k, v in model.state_dict().items()
            }
        if epoch - best_epoch >= patience:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
        model.to(dev)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, default="Raw/moonboard_problems_setup_2016.json")
    parser.add_argument("--output-dir", type=str, default=".")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def train_and_evaluate(sequences, grades, train_idx, test_idx, seed=42):
    set_seeds(seed)
    train_seqs = [sequences[i] for i in train_idx]
    test_seqs = [sequences[i] for i in test_idx]
    y_train = np.array([grades[i] for i in train_idx], dtype=np.int64)
    y_test = np.array([grades[i] for i in test_idx], dtype=np.int64)
    dev = get_device()
    X_train = _sequences_to_vectors(train_seqs)
    X_test = _sequences_to_vectors(test_seqs)
    mu = X_train.mean(axis=0)
    sd = X_train.std(axis=0) + 1e-8
    X_train = (X_train - mu) / sd
    X_test = (X_test - mu) / sd
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.long)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    train_loader = DataLoader(
        TensorDataset(X_train_t, y_train_t), batch_size=256, shuffle=True,
    )
    eval_loader = DataLoader(
        TensorDataset(X_train_t, y_train_t), batch_size=512,
    )
    set_seeds(seed)
    model = CORALModel(HOLD_VECTOR_DIM, NUM_CLASSES, dropout=0.5).to(dev)
    _train(model, train_loader, eval_loader, 0.001, dev, 200, 30)
    model.eval()
    with torch.no_grad():
        logits = model(X_test_t.to(dev))
        probs = torch.sigmoid(logits)
        class_probs = torch.zeros(len(y_test), NUM_CLASSES, device=dev)
        class_probs[:, 0] = 1.0 - probs[:, 0]
        for k in range(1, NUM_CLASSES - 1):
            class_probs[:, k] = probs[:, k - 1] * (1.0 - probs[:, k])
        class_probs[:, -1] = probs[:, -1]
        class_probs /= class_probs.sum(dim=1, keepdim=True) + 1e-8
        y_pred = torch.argmax(class_probs, dim=1).cpu().numpy().tolist()
    all_labels = y_test.tolist()
    metrics = evaluate_classification(all_labels, y_pred, NUM_CLASSES)
    return extract_required_metrics(metrics)


def main():
    args = parse_args()
    set_seeds(args.seed)
    data_path = args.data_path
    if not Path(data_path).exists():
        print("Error")
        sys.exit(1)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    df = load_lstm_data(data_path)
    all_sequences = preprocess_lstm_data(df)
    all_sequences = drop_duplicate_sequences(all_sequences)
    grade_to_idx = {g: i for i, g in enumerate(GRADE_ORDER)}
    valid_seqs, valid_grades = [], []
    for seq in all_sequences:
        g = seq[-2]
        if g in grade_to_idx:
            valid_seqs.append(seq)
            valid_grades.append(grade_to_idx[g])
    from sklearn.model_selection import train_test_split
    train_idx, test_idx = train_test_split(
        np.arange(len(valid_seqs)),
        test_size=0.2,
        random_state=args.seed,
        stratify=valid_grades,
    )
    import time
    t0 = time.time()
    results = train_and_evaluate(valid_seqs, valid_grades, train_idx, test_idx, seed=args.seed)
    elapsed = time.time() - t0
    print(
        f"Macro-F1: {results['macro_f1']:.4f}  "
        f"Exact: {results['exact_accuracy']:.4f}  "
        f"Time: {elapsed:.1f}s"
    )


if __name__ == "__main__":
    main()
