"""submissions/coral-engineered-f1ens — Ensemble of F1-loss + focal ordinal.

Two models with different objectives:
  Model A: Direct soft-F1 loss optimization (18.04% alone)
  Model B: Focal ordinal γ=2.0 (17.77% alone)

Ensemble via class probability averaging.
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
        self.b = nn.Parameter(torch.zeros(NUM_CLASSES - 1))
    def forward(self, x):
        return self.w(self.net(x)) + self.b.unsqueeze(0)


class SoftF1(nn.Module):
    def __init__(self):
        super().__init__()
        self.eps = 1e-6
    def forward(self, logits, y):
        p = torch.sigmoid(logits)
        n = p.shape[0]
        cp = torch.zeros(n, NUM_CLASSES, device=logits.device)
        cp[:, 0] = 1 - p[:, 0]
        for k in range(1, NUM_CLASSES - 1):
            cp[:, k] = p[:, k-1] * (1 - p[:, k])
        cp[:, -1] = p[:, -1]
        cp = cp / (cp.sum(1, keepdim=True) + self.eps)
        oh = torch.zeros_like(cp)
        oh.scatter_(1, y.unsqueeze(1), 1)
        tp = (cp * oh).sum(0)
        fp = (cp * (1 - oh)).sum(0)
        fn = ((1 - cp) * oh).sum(0)
        prec = tp / (tp + fp + self.eps)
        rec = tp / (tp + fn + self.eps)
        return 1 - (2 * prec * rec / (prec + rec + self.eps)).mean()


class Focal(nn.Module):
    def __init__(self, g=2.0):
        super().__init__()
        self.g = g
    def forward(self, l, t):
        bce = F.binary_cross_entropy_with_logits(l, t, reduction="none")
        return ((1 - torch.exp(-bce)) ** self.g * bce).mean()


def _train(model, loader, eload, lr, dev, ep, pat, y=None, mode="f1"):
    crit = SoftF1() if mode == "f1" else Focal(2.0)
    opt = optim.Adam(model.parameters(), lr=lr)
    sch = optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=10)
    best = float("inf")
    bst = None
    be = 0
    for e in range(ep):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(dev), yb.to(dev)
            opt.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward()
            opt.step()
        model.eval()
        el = 0
        nb = 0
        with torch.no_grad():
            for xb, yb in eload:
                xb, yb = xb.to(dev), yb.to(dev)
                el += crit(model(xb), yb).item()
                nb += 1
        el /= max(nb, 1)
        sch.step(el)
        if el < best:
            best = el
            be = e
            bst = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        if e - be >= pat: break
    if bst:
        model.load_state_dict(bst)
        model.to(dev)


def _to_cprobs(logits, dev):
    p = torch.sigmoid(logits).cpu().numpy()
    n = p.shape[0]
    cp = np.zeros((n, NUM_CLASSES), dtype=np.float32)
    cp[:, 0] = 1 - p[:, 0]
    for k in range(1, NUM_CLASSES - 1):
        cp[:, k] = p[:, k-1] * (1 - p[:, k])
    cp[:, -1] = p[:, -1]
    return cp / (cp.sum(1, keepdims=True) + 1e-8)


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
    X_tr = (X_tr - mu) / sd
    X_te = (X_te - mu) / sd
    Xt = torch.tensor(X_tr, dtype=torch.float32)
    yt = torch.tensor(y_tr, dtype=torch.long)
    Xe = torch.tensor(X_te, dtype=torch.float32)
    dl = DataLoader(TensorDataset(Xt, yt), batch_size=256, shuffle=True)
    el = DataLoader(TensorDataset(Xt, yt), batch_size=512)

    set_seeds(seed)
    ma = CORAL().to(dev)
    _train(ma, dl, el, 0.001, dev, 200, 30, mode="f1")

    set_seeds(seed + 1)
    mb = CORAL(0.15).to(dev)
    _train(mb, dl, el, 0.001, dev, 100, 15, mode="focal")

    ma.eval()
    mb.eval()
    with torch.no_grad():
        pa = _to_cprobs(ma(Xe.to(dev)), dev)
        pb = _to_cprobs(mb(Xe.to(dev)), dev)
    y_pred = np.argmax((pa + pb) / 2, axis=1).tolist()
    return evaluate_classification(y_te.tolist(), y_pred, NUM_CLASSES)


def main():
    args = parse_args()
    set_seeds(args.seed)
    df = load_lstm_data(args.data_path)
    seqs = drop_duplicate_sequences(preprocess_lstm_data(df))
    g2i = {g: i for i, g in enumerate(GRADE_ORDER)}
    vs, vg = [], []
    for s in seqs:
        g = s[-2]
        if g in g2i:
            vs.append(s)
            vg.append(g2i[g])
    from sklearn.model_selection import train_test_split
    ti, tei = train_test_split(np.arange(len(vs)), test_size=0.2, random_state=args.seed, stratify=vg)
    import time
    t0 = time.time()
    r = train_and_evaluate(vs, vg, ti, tei, seed=args.seed)
    print(f"F1: {r['macro_f1']:.4f}  Exact: {r['exact_accuracy']:.4f}  Time: {time.time()-t0:.0f}s")

if __name__ == "__main__":
    main()
