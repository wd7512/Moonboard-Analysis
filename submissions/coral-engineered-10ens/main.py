"""submissions/coral-engineered-10ens — Ensemble of 10-class focal + F1-loss."""

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

GRADE_ORDER = ["6B+", "6C", "6C+", "7A", "7A+", "7B", "7B+", "7C", "7C+", "8A"]
NUM_CLASSES = len(GRADE_ORDER)
GRADE_LABELS = frozenset(GRADE_ORDER)
NUM_THRESHOLDS = NUM_CLASSES - 1
HOLD_VECTOR_DIM = 198


def _hold_to_index(h):
    if len(h) < 2: return -1
    c = h[0]
    if c < "A" or c > "K": return -1
    r = h[1:]
    if not r.isdigit(): return -1
    row = int(r)
    if row < 1 or row > 18: return -1
    return (row - 1) * 11 + (ord(c) - ord("A"))


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
        pr = tp / (tp + fp + self.eps)
        re = tp / (tp + fn + self.eps)
        return 1 - (2 * pr * re / (pr + re + self.eps)).mean()


def _train(model, dl, el, lr, dev, ep, pat, gamma=2.0, mode="focal", y_tr=None):
    crit = Focal(gamma) if mode == "focal" else SoftF1()
    opt = optim.Adam(model.parameters(), lr=lr)
    sch = optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=10)
    best = float("inf"); bst = None; be = 0
    for e in range(ep):
        model.train()
        for xb, yb in dl:
            xb, yb = xb.to(dev), yb.to(dev)
            opt.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward()
            opt.step()
        model.eval()
        eloss = 0.0
        nb = 0
        with torch.no_grad():
            for xb, yb in el:
                xb, yb = xb.to(dev), yb.to(dev)
                eloss += crit(model(xb), yb).item()
                nb += 1
        eloss /= max(nb, 1)
        sch.step(eloss)
        if eloss < best:
            best = eloss
            be = e
            bst = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        if e - be >= pat:
            break
    if bst: model.load_state_dict(bst); model.to(dev)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-path", default="Raw/moonboard_problems_setup_2016.json")
    p.add_argument("--output-dir", default=".")
    return p.parse_args()


def train_and_evaluate(sequences, grades, train_idx, test_idx, seed=42):
    set_seeds(seed)
    dev = get_device()
    Xt = torch.tensor(_seq2vec([sequences[i] for i in train_idx]), dtype=torch.float32)
    yt = torch.tensor([grades[i] for i in train_idx], dtype=torch.long)
    Xe = torch.tensor(_seq2vec([sequences[i] for i in test_idx]), dtype=torch.float32)
    yte = [grades[i] for i in test_idx]
    g2o = _labels_to_ordinal(yt.cpu().numpy())
    dl_f = DataLoader(
        TensorDataset(Xt, torch.tensor(g2o, dtype=torch.float32)),
        batch_size=256, shuffle=True,
    )
    el_f = DataLoader(
        TensorDataset(Xt, torch.tensor(g2o, dtype=torch.float32)),
        batch_size=512,
    )
    dl_c = DataLoader(
        TensorDataset(Xt, yt.long()),
        batch_size=256, shuffle=True,
    )
    el_c = DataLoader(
        TensorDataset(Xt, yt.long()),
        batch_size=512,
    )

    set_seeds(seed)
    ma = CORAL().to(dev)
    _train(ma, dl_f, el_f, 0.001, dev, 100, 15, gamma=2.0, mode="focal")

    set_seeds(seed + 1)
    mb = CORAL(0.15).to(dev)
    _train(mb, dl_f, el_f, 0.001, dev, 100, 15, gamma=2.0, mode="focal")

    set_seeds(seed + 2)
    mc = CORAL(0.5).to(dev)
    _train(mc, dl_c, el_c, 0.001, dev, 200, 30, mode="f1")

    ma.eval(); mb.eval(); mc.eval()
    with torch.no_grad():
        pa = _to_cprobs_focal(ma(Xe.to(dev)))
        pb = _to_cprobs_focal(mb(Xe.to(dev)))
        pc = _to_cprobs_f1(mc(Xe.to(dev)))
    avg = (pa + pb + pc) / 3.0
    y_pred = np.argmax(avg, axis=1).tolist()
    return extract_required_metrics(evaluate_classification(yte, y_pred, NUM_CLASSES))


def _to_cprobs_focal(logits):
    p = torch.sigmoid(logits).cpu().numpy()
    n = p.shape[0]
    cp = np.zeros((n, NUM_CLASSES), dtype=np.float32)
    cp[:, 0] = 1 - p[:, 0]
    for k in range(1, NUM_CLASSES - 1):
        cp[:, k] = p[:, k-1] * (1 - p[:, k])
    cp[:, -1] = p[:, -1]
    return cp / (cp.sum(1, keepdims=True) + 1e-8)


def _to_cprobs_f1(logits):
    p = torch.sigmoid(logits).cpu().numpy()
    n = p.shape[0]
    cp = np.zeros((n, NUM_CLASSES), dtype=np.float32)
    cp[:, 0] = 1 - p[:, 0]
    for k in range(1, NUM_CLASSES - 1):
        cp[:, k] = p[:, k-1] * (1 - p[:, k])
    cp[:, -1] = p[:, -1]
    return cp / (cp.sum(1, keepdims=True) + 1e-8)


def _labels_to_ordinal(labels):
    t = np.zeros((len(labels), NUM_THRESHOLDS), dtype=np.float32)
    for i, g in enumerate(labels): t[i, :g] = 1.0
    return t


def main():
    args = parse_args()
    set_seeds(42)
    df = load_lstm_data(args.data_path)
    seqs = drop_duplicate_sequences(preprocess_lstm_data(df))
    g2i = {g: i for i, g in enumerate(GRADE_ORDER)}
    vs, vg = [], []
    for s in seqs:
        g = s[-2]
        if g in g2i: vs.append(s); vg.append(g2i[g])
    from sklearn.model_selection import train_test_split
    ti, tei = train_test_split(np.arange(len(vs)), test_size=0.2, random_state=42, stratify=vg)
    import time; t0 = time.time()
    r = train_and_evaluate(vs, vg, ti, tei)
    print(f"F1: {r['macro_f1']:.4f}  Exact: {r['exact_accuracy']:.4f}  Time: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
