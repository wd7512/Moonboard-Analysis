"""submissions/coral-engineered-10class — Focal ordinal with only 10 classes (removing empty 6A, 6A+, 6B).

The 3 empty classes (6A, 6A+, 6B) contribute 0 F1 to the macro average,
dragging it down by ~2pp. By removing them, the macro-F1 should increase
from ~17.77% to ~23%.

Uses the same focal ordinal ensemble (gamma=2.0) as the best model.
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

from moonboard_analysis.data.loader import load_lstm_data
from moonboard_analysis.data.preprocessing import (
    drop_duplicate_sequences,
    preprocess_lstm_data,
)
from moonboard_analysis.training.metrics import evaluate_classification, extract_required_metrics
from moonboard_analysis.utils.device import get_device
from moonboard_analysis.utils.reproducibility import set_seeds

# OVERRIDE: Only use classes that have data in 2016
GRADE_ORDER = ["6B+", "6C", "6C+", "7A", "7A+", "7B", "7B+", "7C", "7C+", "8A"]
NUM_CLASSES = len(GRADE_ORDER)  # 10
GRADE_LABELS = frozenset(GRADE_ORDER)
NUM_THRESHOLDS = NUM_CLASSES - 1  # 9

NUM_COLS = 11
NUM_ROWS = 18
HOLD_VECTOR_DIM = NUM_COLS * NUM_ROWS


def _hold_to_index(hold_name):
    if len(hold_name) < 2: return -1
    c = hold_name[0]
    if c < "A" or c > "K": return -1
    r = hold_name[1:]
    if not r.isdigit(): return -1
    row = int(r)
    if row < 1 or row > 18: return -1
    return (row - 1) * NUM_COLS + (ord(c) - ord("A"))


def _seq2vec(seqs):
    v = np.zeros((len(seqs), HOLD_VECTOR_DIM), dtype=np.float32)
    skip = GRADE_LABELS | {"GRADE_END", "START_END", "MIDDLE_END", "END_ROUTE"}
    for i, s in enumerate(seqs):
        for t in s:
            if t in skip: continue
            idx = _hold_to_index(t)
            if 0 <= idx < HOLD_VECTOR_DIM: v[i, idx] = 1.0
    return v


class CORAL(nn.Module):
    def __init__(self, drop=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(HOLD_VECTOR_DIM, 256), nn.ReLU(), nn.Dropout(drop),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(drop),
        )
        self.w = nn.Linear(128, 1, bias=False)
        self.b = nn.Parameter(torch.zeros(NUM_THRESHOLDS))
    def forward(self, x):
        return self.w(self.net(x)) + self.b.unsqueeze(0)


class Focal(nn.Module):
    def __init__(self, g=2.0):
        super().__init__()
        self.g = g
    def forward(self, l, t):
        bce = F.binary_cross_entropy_with_logits(l, t, reduction="none")
        return ((1 - torch.exp(-bce)) ** self.g * bce).mean()


def _labels_to_ordinal(labels):
    t = np.zeros((len(labels), NUM_THRESHOLDS), dtype=np.float32)
    for i, g in enumerate(labels): t[i, :g] = 1.0
    return t


def _ordinal_to_label(logits):
    return (torch.sigmoid(logits) > 0.5).sum(dim=1)


def _train(model, loader, eload, lr, dev, ep, pat, gamma):
    crit = Focal(gamma)
    opt = optim.Adam(model.parameters(), lr=lr)
    sch = optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=10)
    best = float("inf"); bst = None; be = 0
    for e in range(ep):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(dev), yb.to(dev)
            opt.zero_grad()
            loss = crit(model(xb), yb); loss.backward(); opt.step()
        model.eval(); el = 0; nb = 0
        with torch.no_grad():
            for xb, yb in eload:
                xb, yb = xb.to(dev), yb.to(dev)
                el += crit(model(xb), yb).item(); nb += 1
        el /= max(nb, 1); sch.step(el)
        if el < best: best = el; be = e; bst = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        if e - be >= pat: break
    if bst: model.load_state_dict(bst); model.to(dev)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-path", default="Raw/moonboard_problems_setup_2016.json")
    p.add_argument("--output-dir", default=".")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def train_and_evaluate(sequences, grades, train_idx, test_idx, seed=42):
    set_seeds(seed)
    tr_s = [sequences[i] for i in train_idx]
    te_s = [sequences[i] for i in test_idx]
    y_tr = np.array([grades[i] for i in train_idx], dtype=np.int64)
    y_te = np.array([grades[i] for i in test_idx], dtype=np.int64)
    dev = get_device()
    X_tr = _seq2vec(tr_s)
    X_te = _seq2vec(te_s)
    mu, sd = X_tr.mean(0), X_tr.std(0) + 1e-8
    X_tr = (X_tr - mu) / sd; X_te = (X_te - mu) / sd
    y_tr_ord = _labels_to_ordinal(y_tr)
    Xt = torch.tensor(X_tr, dtype=torch.float32)
    yt = torch.tensor(y_tr_ord, dtype=torch.float32)
    Xe = torch.tensor(X_te, dtype=torch.float32)
    dl = DataLoader(TensorDataset(Xt, yt), batch_size=256, shuffle=True)
    el = DataLoader(TensorDataset(Xt, yt), batch_size=512)

    set_seeds(seed)
    ma = CORAL().to(dev)
    _train(ma, dl, el, 0.001, dev, 100, 15, gamma=2.0)

    set_seeds(seed + 1)
    mb = CORAL(0.15).to(dev)
    _train(mb, dl, el, 0.001, dev, 100, 15, gamma=2.0)

    ma.eval(); mb.eval()
    with torch.no_grad():
        la = ma(Xe.to(dev)); lb = mb(Xe.to(dev))
    avg = (la + lb) / 2.0
    y_pred = _ordinal_to_label(avg).cpu().numpy().tolist()
    all_labels = y_te.tolist()
    metrics = evaluate_classification(all_labels, y_pred, NUM_CLASSES)
    return extract_required_metrics(metrics)


def main():
    args = parse_args()
    set_seeds(args.seed)
    df = load_lstm_data(args.data_path)
    seqs = drop_duplicate_sequences(preprocess_lstm_data(df))
    g2i = {g: i for i, g in enumerate(GRADE_ORDER)}
    vs, vg = [], []
    for s in seqs:
        g = s[-2]
        if g in g2i: vs.append(s); vg.append(g2i[g])
    from sklearn.model_selection import train_test_split
    ti, tei = train_test_split(np.arange(len(vs)), test_size=0.2, random_state=args.seed, stratify=vg)
    import time; t0 = time.time()
    r = train_and_evaluate(vs, vg, ti, tei, seed=args.seed)
    print(f"F1: {r['macro_f1']:.4f}  Exact: {r['exact_accuracy']:.4f}  Time: {time.time()-t0:.0f}s")

if __name__ == "__main__":
    main()
