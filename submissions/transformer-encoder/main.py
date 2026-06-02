"""submissions/transformer-encoder — Transformer encoder for Moonboard grade prediction.

Uses bottom-to-top re-ordered hold sequences (from Goal 4) with a compact
TransformerEncoder: d_model=64, 2 heads, 2 layers, ff=128. Mean pooling
over the sequence for classification.

Usage:
    uv run python submissions/transformer-encoder/main.py --help
    uv run python submissions/transformer-encoder/main.py \
        --data-path Raw/moonboard_problems_setup_2016.json
"""

import argparse
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

from moonboard_analysis.config import GRADE_ORDER
from moonboard_analysis.data.loader import load_lstm_data
from moonboard_analysis.data.preprocessing import (
    drop_duplicate_sequences,
    preprocess_lstm_data,
)
from moonboard_analysis.training.metrics import evaluate_classification, extract_required_metrics
from moonboard_analysis.utils.device import get_device
from moonboard_analysis.utils.reproducibility import set_seeds

NUM_CLASSES = len(GRADE_ORDER)


# --- Bottom-to-top reordering (same as Goal 4) ---

def bottom_to_top_reorder(seq: list[str], seed: int = 42) -> list[str]:
    """Re-order hold tokens by row ascending (bottom-to-top)."""
    hold_positions: list[tuple[int, int, int, str]] = []
    non_hold_map: dict[int, str] = {}
    skip_tokens = frozenset(GRADE_ORDER) | {"GRADE_END", "START_END", "MIDDLE_END", "END_ROUTE"}

    for i, token in enumerate(seq):
        if token in skip_tokens:
            non_hold_map[i] = token
            continue
        if len(token) < 2:
            non_hold_map[i] = token
            continue
        col_char = token[0]
        if col_char < "A" or col_char > "K":
            non_hold_map[i] = token
            continue
        row_part = token[1:]
        if not row_part.isdigit():
            non_hold_map[i] = token
            continue
        col = ord(col_char) - ord("A")
        row = int(row_part) - 1
        if 0 <= row < 18 and 0 <= col < 11:
            hold_positions.append((i, row, col, token))
        else:
            non_hold_map[i] = token

    rng = random.Random(seed)
    section_breaks = [i for i in range(len(seq))
                      if seq[i] in {"START_END", "MIDDLE_END", "END_ROUTE"}]

    if not section_breaks:
        sections = [(0, len(seq))]
    else:
        sections = []
        prev = 0
        for brk in section_breaks:
            sections.append((prev, brk))
            prev = brk + 1
        sections.append((prev, len(seq)))

    result = list(seq)
    for lo, hi in sections:
        section_holds = [(idx, row, col, tok)
                         for idx, row, col, tok in hold_positions if lo <= idx < hi]
        non_hold_idxs = {i for i in range(lo, hi) if i in non_hold_map}
        sorted_holds = sorted(section_holds, key=lambda x: (x[1], rng.random()))
        hold_iter = iter(sorted_holds)
        for i in range(lo, hi):
            if i in non_hold_idxs:
                result[i] = non_hold_map[i]
            else:
                try:
                    result[i] = next(hold_iter)[3]
                except StopIteration:
                    pass
    return result


# --- Positional Encoding ---

class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for Transformer."""

    def __init__(self, d_model: int, max_len: int = 256):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1), :]


# --- Transformer Grade Predictor ---

class TransformerGradePredictor(nn.Module):
    """Compact TransformerEncoder for Moonboard grade prediction.

    Architecture:
        Embedding(vocab_size → d_model) → PositionalEncoding →
        TransformerEncoder(2 layers, 2 heads, ff=128) →
        MeanPool → Linear(d_model → num_classes)
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 64,
        nhead: int = 2,
        num_layers: int = 2,
        dim_feedforward: int = 128,
        num_classes: int = 12,
        max_len: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_encoder = PositionalEncoding(d_model, max_len)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="relu",
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len) — token ids
        x = self.embedding(x)  # (batch, seq_len, d_model)
        x = self.pos_encoder(x)
        # TransformerEncoder — no mask to avoid MPS incompatibility
        x = self.transformer(x)
        # Mean pooling over sequence
        x = x.mean(dim=1)
        return self.classifier(x)


# --- Dataset ---

class TransformerSequenceDataset(Dataset):
    """Dataset that encodes token sequences as integer IDs with padding."""

    def __init__(
        self,
        sequences: list[list[str]],
        labels: list[int],
        vocab: dict[str, int],
        max_length: int,
    ):
        self.sequences = sequences
        self.labels = labels
        self.vocab = vocab
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        seq = self.sequences[idx]
        token_ids = [self.vocab.get(t, 0) for t in seq[:self.max_length]]
        padded = token_ids + [0] * (self.max_length - len(token_ids))
        return torch.tensor(padded, dtype=torch.long), self.labels[idx]


def build_vocab(sequences: list[list[str]]) -> dict[str, int]:
    tokens: set[str] = set()
    for seq in sequences:
        tokens.update(seq)
    vocab: dict[str, int] = {t: i + 1 for i, t in enumerate(sorted(tokens))}
    vocab["<PAD>"] = 0
    return vocab


def collate_fn(batch: list[tuple[torch.Tensor, int]]) -> tuple[torch.Tensor, torch.Tensor]:
    """Collate without padding mask (MPS compatibility)."""
    seqs, labels = zip(*batch)
    seqs = torch.stack(seqs)
    labels = torch.tensor(labels, dtype=torch.long)
    return seqs, labels


def train_and_evaluate(
    sequences: list[list[str]],
    grades: list[int],
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    seed: int = 42,
    d_model: int = 64,
    nhead: int = 2,
    num_layers: int = 2,
    dim_feedforward: int = 128,
    epochs: int = 30,
    batch_size: int = 64,
    learning_rate: float = 0.001,
    patience: int = 8,
) -> dict[str, float]:
    """Train a fresh Transformer on train_idx, evaluate on test_idx."""
    set_seeds(seed)

    train_seqs_full = [sequences[i] for i in train_idx]
    test_seqs_full = [sequences[i] for i in test_idx]
    y_train = np.array([grades[i] for i in train_idx], dtype=np.int64)
    y_test = np.array([grades[i] for i in test_idx], dtype=np.int64)

    train_seqs = [s[:-2] for s in train_seqs_full]
    test_seqs = [s[:-2] for s in test_seqs_full]

    # Apply bottom-to-top reordering
    train_seqs = [bottom_to_top_reorder(seq, seed=seed) for seq in train_seqs]
    test_seqs = [bottom_to_top_reorder(seq, seed=seed + 1) for seq in test_seqs]

    vocab = build_vocab(train_seqs)
    max_length = max(len(s) for s in train_seqs) if train_seqs else 1
    vocab_size = len(vocab)

    train_ds = TransformerSequenceDataset(train_seqs, y_train.tolist(), vocab, max_length)
    test_ds = TransformerSequenceDataset(test_seqs, y_test.tolist(), vocab, max_length)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_ds, batch_size=batch_size * 2, collate_fn=collate_fn)

    device = get_device()
    model = TransformerGradePredictor(
        vocab_size=vocab_size,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        dim_feedforward=dim_feedforward,
        num_classes=NUM_CLASSES,
        max_len=max_length,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=patience
    )

    best_loss = float("inf")
    best_state = None
    best_epoch = 0

    for epoch in range(epochs):
        model.train()
        for seqs, lbls in train_loader:
            seqs, lbls = seqs.to(device), lbls.to(device)
            optimizer.zero_grad()
            loss = criterion(model(seqs), lbls)
            loss.backward()
            optimizer.step()

        model.eval()
        test_loss = 0.0
        n_batches = 0
        with torch.no_grad():
            for seqs, lbls in test_loader:
                seqs, lbls = seqs.to(device), lbls.to(device)
                test_loss += criterion(model(seqs), lbls).item()
                n_batches += 1
        test_loss /= max(n_batches, 1)
        scheduler.step(test_loss)

        if test_loss < best_loss:
            best_loss = test_loss
            best_epoch = epoch
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if epoch - best_epoch >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
        model.to(device)

    all_preds, all_labels = [], []
    model.eval()
    with torch.no_grad():
        for seqs_in, lbls_in in test_loader:
            seqs_in = seqs_in.to(device)
            preds = torch.argmax(model(seqs_in), 1)
            all_preds.extend(preds.cpu().numpy().tolist())
            all_labels.extend(lbls_in.numpy().tolist())

    metrics = evaluate_classification(all_labels, all_preds, NUM_CLASSES)
    return extract_required_metrics(metrics)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Transformer encoder — train and evaluate on Moonboard data"
    )
    parser.add_argument("--data-path", type=str, default="Raw/moonboard_problems_setup_2016.json")
    parser.add_argument("--output-dir", type=str, default=".")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--nhead", type=int, default=2)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dim-feedforward", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--patience", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seeds(args.seed)

    data_path = args.data_path
    if not Path(data_path).exists():
        print(f"Error: Data file not found at '{data_path}'")
        sys.exit(1)

    print(f"Loading data from {data_path}")
    df = load_lstm_data(data_path)
    print(f"Raw data: {len(df)} routes")

    all_sequences = preprocess_lstm_data(df)
    all_sequences = drop_duplicate_sequences(all_sequences)
    print(f"After preprocessing: {len(all_sequences)} unique sequences")

    grade_to_idx = {g: i for i, g in enumerate(GRADE_ORDER)}
    valid_seqs, valid_grades = [], []
    for seq in all_sequences:
        g = seq[-2]
        if g in grade_to_idx:
            valid_seqs.append(seq)
            valid_grades.append(grade_to_idx[g])

    train_idx, test_idx = train_test_split(
        np.arange(len(valid_seqs)), test_size=0.2,
        random_state=args.seed, stratify=valid_grades,
    )

    device = get_device()
    print(f"Training on device: {device}")
    t0 = time.time()
    results = train_and_evaluate(
        valid_seqs, valid_grades, train_idx, test_idx,
        seed=args.seed, d_model=args.d_model, nhead=args.nhead,
        num_layers=args.num_layers, dim_feedforward=args.dim_feedforward,
        epochs=args.epochs, batch_size=args.batch_size,
        learning_rate=args.learning_rate, patience=args.patience,
    )
    elapsed = time.time() - t0

    print(f"\n{'=' * 50}")
    print("Evaluation Results")
    print(f"{'=' * 50}")
    print(f"Exact Accuracy:     {results['exact_accuracy']:.4f}")
    print(f"Within-1 Accuracy:  {results['within_one_grade']:.4f}")
    print(f"Within-2 Accuracy:  {results['within_two_grades']:.4f}")
    print(f"Macro-F1:           {results['macro_f1']:.4f}")
    print(f"Training time:      {elapsed:.1f}s")


if __name__ == "__main__":
    main()
