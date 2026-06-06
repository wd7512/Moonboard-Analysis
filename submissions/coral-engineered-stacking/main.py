"""submissions/coral-engineered-stacking — Stacking meta-learner ensemble.

Level 0: Three diverse models make predictions on training data (out-of-fold).
Level 1: A meta-learner (lightweight MLP) learns to combine Level 0 predictions.

Level 0 models:
  A: Focal ordinal (γ=2.0), 256→128
  B: BCE ordinal, 512→256→128
  C: Class-balanced CE, 256→128

Level 1 meta-learner: Takes Level 0 class probabilities as input, outputs final prediction.
Trained on out-of-fold predictions from Level 0 (no leakage).

Usage:
    uv run python submissions/coral-engineered-stacking/main.py --help
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold

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
NUM_CLASSES = len(GRADE_ORDER)  # 13
GRADE_LABELS = frozenset(GRADE_ORDER)
NUM_THRESHOLDS = NUM_CLASSES - 1  # 12


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


def _sequences_to_vectors(sequences: list[list[str]]) -> np.ndarray:
    vectors = np.zeros((len(sequences), HOLD_VECTOR_DIM), dtype=np.float32)
    skip_tokens = GRADE_LABELS | {"GRADE_END", "START_END", "MIDDLE_END", "END_ROUTE"}
    for i, seq in enumerate(sequences):
        for token in seq:
            if token in skip_tokens:
                continue
            idx = _hold_to_index(token)
            if 0 <= idx < HOLD_VECTOR_DIM:
                vectors[i, idx] = 1.0
    return vectors


class CORALModel(nn.Module):
    def __init__(self, input_dim: int, hidden_layers: list[int], num_classes: int, dropout: float = 0.3):
        super().__init__()
        self.num_thresholds = num_classes - 1
        layers = []
        prev_dim = input_dim
        for h in hidden_layers:
            layers.extend([
                nn.Linear(prev_dim, h),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_dim = h
        self.net = nn.Sequential(*layers)
        self.coral_weight = nn.Linear(prev_dim, 1, bias=False)
        self.coral_bias = nn.Parameter(torch.zeros(self.num_thresholds))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.net(x)
        shared = self.coral_weight(h)
        return shared + self.coral_bias.unsqueeze(0)


class ClassifierModel(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MetaLearner(nn.Module):
    """Takes Level 0 class probabilities as input, outputs final prediction."""
    def __init__(self, input_dim: int, num_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0):
        super().__init__()
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        p_t = torch.exp(-bce)
        return ((1.0 - p_t) ** self.gamma * bce).mean()


class ClassBalancedLoss(nn.Module):
    def __init__(self, class_counts: np.ndarray, beta: float = 0.99):
        super().__init__()
        counts = class_counts.astype(np.float64)
        weights = np.divide(
            1.0 - beta,
            1.0 - np.power(beta, counts),
            where=counts > 0,
            out=np.full_like(counts, 0.0, dtype=np.float64),
        )
        self.register_buffer("weights", torch.tensor(weights, dtype=torch.float32))

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        weights = self.weights.to(logits.device)
        return F.cross_entropy(logits, targets, weight=weights)


def _labels_to_ordinal(labels: np.ndarray) -> np.ndarray:
    targets = np.zeros((len(labels), NUM_THRESHOLDS), dtype=np.float32)
    for i, g in enumerate(labels):
        targets[i, :g] = 1.0
    return targets


def _train_level0_ordinal(model, train_loader, eval_loader, lr, dev, epochs, patience, gamma):
    criterion = FocalLoss(gamma=gamma) if gamma > 0 else nn.BCEWithLogitsLoss()
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
        n_batches = 0
        with torch.no_grad():
            for xb, yb in eval_loader:
                xb, yb = xb.to(dev), yb.to(dev)
                eval_loss += criterion(model(xb), yb).item()
                n_batches += 1
        eval_loss /= max(n_batches, 1)
        scheduler.step(eval_loss)
        if eval_loss < best_loss:
            best_loss = eval_loss
            best_epoch = epoch
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        if epoch - best_epoch >= patience:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
        model.to(dev)


def _train_level0_classifier(model, train_loader, eval_loader, lr, dev, epochs, patience, y_train):
    class_counts = np.bincount(y_train, minlength=NUM_CLASSES).astype(np.float64)
    criterion = ClassBalancedLoss(class_counts, beta=0.99)
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
            loss = criterion(model(xb), yb.long())
            loss.backward()
            optimizer.step()
        model.eval()
        eval_loss = 0.0
        n_batches = 0
        with torch.no_grad():
            for xb, yb in eval_loader:
                xb, yb = xb.to(dev), yb.to(dev)
                eval_loss += criterion(model(xb), yb.long()).item()
                n_batches += 1
        eval_loss /= max(n_batches, 1)
        scheduler.step(eval_loss)
        if eval_loss < best_loss:
            best_loss = eval_loss
            best_epoch = epoch
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        if epoch - best_epoch >= patience:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
        model.to(dev)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stacking meta-learner ensemble")
    parser.add_argument("--data-path", type=str, default="Raw/moonboard_problems_setup_2016.json")
    parser.add_argument("--output-dir", type=str, default=".")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def train_and_evaluate(
    sequences: list[list[str]],
    grades: list[int],
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    seed: int = 42,
) -> dict[str, float]:
    """Train stacking ensemble with out-of-fold meta-learner."""
    set_seeds(seed)

    train_seqs = [sequences[i] for i in train_idx]
    test_seqs = [sequences[i] for i in test_idx]
    y_train_full = np.array([grades[i] for i in train_idx], dtype=np.int64)
    y_test = np.array([grades[i] for i in test_idx], dtype=np.int64)

    dev = get_device()

    X_train_full = _sequences_to_vectors(train_seqs)
    X_test = _sequences_to_vectors(test_seqs)

    mu = X_train_full.mean(axis=0)
    sd = X_train_full.std(axis=0) + 1e-8
    X_train_full = (X_train_full - mu) / sd
    X_test = (X_test - mu) / sd

    n_train = len(train_idx)
    n_test = len(test_idx)

    # === Level 0: Generate out-of-fold predictions ===
    n_folds = 5
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)

    # Out-of-fold probability predictions for meta-learner training
    oof_probs_a = np.zeros((n_train, NUM_CLASSES), dtype=np.float32)
    oof_probs_b = np.zeros((n_train, NUM_CLASSES), dtype=np.float32)
    oof_probs_c = np.zeros((n_train, NUM_CLASSES), dtype=np.float32)

    # Test predictions from each fold
    test_probs_a_list = []
    test_probs_b_list = []
    test_probs_c_list = []

    for fold_idx, (fold_train_idx, fold_val_idx) in enumerate(skf.split(np.zeros(n_train), y_train_full)):
        print(f"  Level 0 Fold {fold_idx+1}/{n_folds}...")

        X_fold_train = X_train_full[fold_train_idx]
        X_fold_val = X_train_full[fold_val_idx]
        y_fold_train = y_train_full[fold_train_idx]
        y_fold_val = y_train_full[fold_val_idx]

        X_ft = torch.tensor(X_fold_train, dtype=torch.float32)
        X_fv = torch.tensor(X_fold_val, dtype=torch.float32)
        X_test_t = torch.tensor(X_test, dtype=torch.float32)

        y_ft_ord = _labels_to_ordinal(y_fold_train)

        train_ord_loader = DataLoader(TensorDataset(X_ft, torch.tensor(y_ft_ord, dtype=torch.float32)), batch_size=256, shuffle=True)
        eval_ord_loader = DataLoader(TensorDataset(X_fv, torch.tensor(_labels_to_ordinal(y_fold_val), dtype=torch.float32)), batch_size=512)

        train_cls_loader = DataLoader(TensorDataset(X_ft, torch.tensor(y_fold_train, dtype=torch.long)), batch_size=256, shuffle=True)
        eval_cls_loader = DataLoader(TensorDataset(X_fv, torch.tensor(y_fold_val, dtype=torch.long)), batch_size=512)

        # Model A: Focal ordinal
        set_seeds(seed + fold_idx * 10)
        model_a = CORALModel(HOLD_VECTOR_DIM, [256, 128], NUM_CLASSES, dropout=0.3).to(dev)
        _train_level0_ordinal(model_a, train_ord_loader, eval_ord_loader, 0.001, dev, 100, 15, gamma=2.0)

        # Model BCE ordinal
        set_seeds(seed + fold_idx * 10 + 1)
        model_b = CORALModel(HOLD_VECTOR_DIM, [512, 256, 128], NUM_CLASSES, dropout=0.15).to(dev)
        _train_level0_ordinal(model_b, train_ord_loader, eval_ord_loader, 0.001, dev, 100, 15, gamma=0.0)

        # Model C: Class-balanced CE
        set_seeds(seed + fold_idx * 10 + 2)
        model_c = ClassifierModel(HOLD_VECTOR_DIM, 256, NUM_CLASSES, dropout=0.3).to(dev)
        _train_level0_classifier(model_c, train_cls_loader, eval_cls_loader, 0.001, dev, 100, 15, y_fold_train)

        # Generate OOF predictions
        model_a.eval()
        model_b.eval()
        model_c.eval()
        with torch.no_grad():
            # Ordinal models
            logits_a = model_a(X_fv.to(dev))
            probs_a = torch.sigmoid(logits_a).cpu().numpy()
            class_probs_a = np.zeros((len(fold_val_idx), NUM_CLASSES), dtype=np.float32)
            class_probs_a[:, 0] = 1.0 - probs_a[:, 0]
            for k in range(1, NUM_CLASSES - 1):
                class_probs_a[:, k] = probs_a[:, k - 1] * (1.0 - probs_a[:, k])
            class_probs_a[:, -1] = probs_a[:, -1]
            class_probs_a /= class_probs_a.sum(axis=1, keepdims=True) + 1e-8
            oof_probs_a[fold_val_idx] = class_probs_a

            logits_b = model_b(X_fv.to(dev))
            probs_b = torch.sigmoid(logits_b).cpu().numpy()
            class_probs_b = np.zeros((len(fold_val_idx), NUM_CLASSES), dtype=np.float32)
            class_probs_b[:, 0] = 1.0 - probs_b[:, 0]
            for k in range(1, NUM_CLASSES - 1):
                class_probs_b[:, k] = probs_b[:, k - 1] * (1.0 - probs_b[:, k])
            class_probs_b[:, -1] = probs_b[:, -1]
            class_probs_b /= class_probs_b.sum(axis=1, keepdims=True) + 1e-8
            oof_probs_b[fold_val_idx] = class_probs_b

            # Classifier model
            logits_c = model_c(X_fv.to(dev))
            oof_probs_c[fold_val_idx] = F.softmax(logits_c, dim=1).cpu().numpy()

            # Test predictions
            test_logits_a = model_a(X_test_t.to(dev))
            test_probs_a = torch.sigmoid(test_logits_a).cpu().numpy()
            test_class_probs_a = np.zeros((n_test, NUM_CLASSES), dtype=np.float32)
            test_class_probs_a[:, 0] = 1.0 - test_probs_a[:, 0]
            for k in range(1, NUM_CLASSES - 1):
                test_class_probs_a[:, k] = test_probs_a[:, k - 1] * (1.0 - test_probs_a[:, k])
            test_class_probs_a[:, -1] = test_probs_a[:, -1]
            test_class_probs_a /= test_class_probs_a.sum(axis=1, keepdims=True) + 1e-8
            test_probs_a_list.append(test_class_probs_a)

            test_logits_b = model_b(X_test_t.to(dev))
            test_probs_b = torch.sigmoid(test_logits_b).cpu().numpy()
            test_class_probs_b = np.zeros((n_test, NUM_CLASSES), dtype=np.float32)
            test_class_probs_b[:, 0] = 1.0 - test_probs_b[:, 0]
            for k in range(1, NUM_CLASSES - 1):
                test_class_probs_b[:, k] = test_probs_b[:, k - 1] * (1.0 - test_probs_b[:, k])
            test_class_probs_b[:, -1] = test_probs_b[:, -1]
            test_class_probs_b /= test_class_probs_b.sum(axis=1, keepdims=True) + 1e-8
            test_probs_b_list.append(test_class_probs_b)

            test_logits_c = model_c(X_test_t.to(dev))
            test_probs_c_list.append(F.softmax(test_logits_c, dim=1).cpu().numpy())

    # === Level 1: Train meta-learner on OOF predictions ===
    print("  Training meta-learner...")
    meta_input = np.concatenate([oof_probs_a, oof_probs_b, oof_probs_c], axis=1)

    set_seeds(seed + 100)
    meta_model = MetaLearner(meta_input.shape[1], NUM_CLASSES).to(dev)

    meta_X = torch.tensor(meta_input, dtype=torch.float32)
    meta_y = torch.tensor(y_train_full, dtype=torch.long)
    meta_dataset = TensorDataset(meta_X, meta_y)
    meta_loader = DataLoader(meta_dataset, batch_size=256, shuffle=True)

    meta_criterion = ClassBalancedLoss(np.bincount(y_train_full, minlength=NUM_CLASSES), beta=0.99)
    meta_optimizer = optim.Adam(meta_model.parameters(), lr=0.001)
    for epoch in range(200):
        meta_model.train()
        for xb, yb in meta_loader:
            xb, yb = xb.to(dev), yb.to(dev)
            meta_optimizer.zero_grad()
            loss = meta_criterion(meta_model(xb), yb)
            loss.backward()
            meta_optimizer.step()

    # === Predict ===
    meta_model.eval()
    test_meta_input = np.concatenate([
        np.mean(test_probs_a_list, axis=0),
        np.mean(test_probs_b_list, axis=0),
        np.mean(test_probs_c_list, axis=0),
    ], axis=1)
    with torch.no_grad():
        meta_logits = meta_model(torch.tensor(test_meta_input, dtype=torch.float32).to(dev))
        y_pred = torch.argmax(meta_logits, dim=1).cpu().numpy().tolist()

    all_labels = y_test.tolist()
    metrics = evaluate_classification(all_labels, y_pred, NUM_CLASSES)
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

    all_sequences = preprocess_lstm_data(df)
    all_sequences = drop_duplicate_sequences(all_sequences)
    print(f"After preprocessing: {len(all_sequences)} unique sequences")

    from sklearn.model_selection import train_test_split

    grade_to_idx = {g: i for i, g in enumerate(GRADE_ORDER)}
    valid_seqs: list[list[str]] = []
    valid_grades: list[int] = []
    for seq in all_sequences:
        grade = seq[-2]
        if grade in grade_to_idx:
            valid_seqs.append(seq)
            valid_grades.append(grade_to_idx[grade])

    train_idx, test_idx = train_test_split(
        np.arange(len(valid_seqs)), test_size=0.2, random_state=args.seed, stratify=valid_grades,
    )

    import time
    t0 = time.time()
    results = train_and_evaluate(valid_seqs, valid_grades, train_idx, test_idx, seed=args.seed)
    elapsed = time.time() - t0

    print()
    print("=" * 50)
    print("Evaluation Results")
    print("=" * 50)
    print(f"Exact Accuracy:     {results['exact_accuracy']:.4f}")
    print(f"Within-1 Accuracy:  {results['within_one_grade']:.4f}")
    print(f"Within-2 Accuracy:  {results['within_two_grades']:.4f}")
    print(f"Macro-F1:           {results['macro_f1']:.4f}")
    print(f"Training time:      {elapsed:.1f}s")


if __name__ == "__main__":
    main()
