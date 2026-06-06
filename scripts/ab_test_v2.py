#!/usr/bin/env python3
"""Quick A/B test: class-balanced CE on flat features vs ensemble."""

import sys
import time
sys.path.insert(0, "submissions/coral-engineered")

from main import (
    MLP, NUM_CLASSES, HOLD_VECTOR_DIM, ENGINEERED_DIM,
    _sequences_to_flat, _sequences_to_engineered,
    _compute_class_weights, ClassBalancedLoss,
    _train_model, _extract_probs,
)
from moonboard_analysis.config import GRADE_ORDER
from moonboard_analysis.data.loader import load_lstm_data
from moonboard_analysis.data.preprocessing import drop_duplicate_sequences, preprocess_lstm_data
from moonboard_analysis.training.metrics import evaluate_classification, extract_required_metrics
from moonboard_analysis.utils.device import get_device
from moonboard_analysis.utils.reproducibility import set_seeds

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split


def run_experiment(name, fn):
    """Run a single experiment and print results."""
    print(f"\n{'='*60}")
    print(f"Experiment: {name}")
    print(f"{'='*60}")
    set_seeds(42)
    df = load_lstm_data("Raw/moonboard_problems_setup_2016.json")
    all_sequences = preprocess_lstm_data(df)
    all_sequences = drop_duplicate_sequences(all_sequences)
    grade_to_idx = {g: i for i, g in enumerate(GRADE_ORDER)}
    valid_seqs, valid_grades = [], []
    for seq in all_sequences:
        grade = seq[-2]
        if grade in grade_to_idx:
            valid_seqs.append(seq)
            valid_grades.append(grade_to_idx[grade])
    train_idx, test_idx = train_test_split(
        np.arange(len(valid_seqs)), test_size=0.2,
        random_state=42, stratify=valid_grades,
    )
    t0 = time.time()
    results = fn(valid_seqs, valid_grades, train_idx, test_idx)
    elapsed = time.time() - t0
    print(f"Exact:     {results['exact_accuracy']:.4f}")
    print(f"Within-1:  {results['within_one_grade']:.4f}")
    print(f"Within-2:  {results['within_two_grades']:.4f}")
    print(f"Macro-F1:  {results['macro_f1']:.4f}")
    print(f"Time:      {elapsed:.1f}s")
    return results


def single_cb_ce_flat(sequences, grades, train_idx, test_idx):
    """Single class-balanced CE on flat features."""
    set_seeds(42)
    dev = get_device()
    train_seqs = [sequences[i] for i in train_idx]
    test_seqs = [sequences[i] for i in test_idx]
    y_train = np.array([grades[i] for i in train_idx], dtype=np.int64)
    y_test = np.array([grades[i] for i in test_idx], dtype=np.int64)
    X_train = _sequences_to_flat(train_seqs)
    X_test = _sequences_to_flat(test_seqs)
    mu = X_train.mean(axis=0)
    sd = X_train.std(axis=0) + 1e-8
    X_train = (X_train - mu) / sd
    X_test = (X_test - mu) / sd
    train_loader = DataLoader(
        TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long)),
        batch_size=256, shuffle=True,
    )
    test_loader = DataLoader(
        TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.long)),
        batch_size=512,
    )
    class_counts = np.bincount(y_train, minlength=NUM_CLASSES)
    class_weights = _compute_class_weights(class_counts, NUM_CLASSES, beta=0.99).to(dev)
    model = MLP(HOLD_VECTOR_DIM, 256, NUM_CLASSES, dropout=0.3).to(dev)
    criterion = ClassBalancedLoss(class_weights=class_weights)
    _train_model(model, train_loader, test_loader, criterion, 0.001, dev, 100, 15)
    probs = _extract_probs(model, test_loader, dev)
    y_pred = np.argmax(probs, axis=1).tolist()
    metrics = evaluate_classification(y_test.tolist(), y_pred, NUM_CLASSES)
    return extract_required_metrics(metrics)


def ensemble_cb_ce(sequences, grades, train_idx, test_idx):
    """Ensemble 80/20: class-balanced CE flat + standard engineered."""
    set_seeds(42)
    dev = get_device()
    train_seqs = [sequences[i] for i in train_idx]
    test_seqs = [sequences[i] for i in test_idx]
    y_train = np.array([grades[i] for i in train_idx], dtype=np.int64)
    y_test = np.array([grades[i] for i in test_idx], dtype=np.int64)
    # Model A: class-balanced CE on flat features
    X_train_flat = _sequences_to_flat(train_seqs)
    X_test_flat = _sequences_to_flat(test_seqs)
    mu_a = X_train_flat.mean(axis=0)
    sd_a = X_train_flat.std(axis=0) + 1e-8
    X_train_flat = (X_train_flat - mu_a) / sd_a
    X_test_flat = (X_test_flat - mu_a) / sd_a
    train_loader_a = DataLoader(
        TensorDataset(torch.tensor(X_train_flat, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long)),
        batch_size=256, shuffle=True,
    )
    test_loader_a = DataLoader(
        TensorDataset(torch.tensor(X_test_flat, dtype=torch.float32), torch.tensor(y_test, dtype=torch.long)),
        batch_size=512,
    )
    class_counts = np.bincount(y_train, minlength=NUM_CLASSES)
    class_weights = _compute_class_weights(class_counts, NUM_CLASSES, beta=0.99).to(dev)
    model_a = MLP(HOLD_VECTOR_DIM, 256, NUM_CLASSES, dropout=0.3).to(dev)
    criterion_a = ClassBalancedLoss(class_weights=class_weights)
    _train_model(model_a, train_loader_a, test_loader_a, criterion_a, 0.001, dev, 100, 15)
    probs_a = _extract_probs(model_a, test_loader_a, dev)
    # Model B: standard MLP on engineered features
    X_train_eng = _sequences_to_engineered(train_seqs)
    X_test_eng = _sequences_to_engineered(test_seqs)
    mu_b = X_train_eng.mean(axis=0)
    sd_b = X_train_eng.std(axis=0) + 1e-8
    X_train_eng = (X_train_eng - mu_b) / sd_b
    X_test_eng = (X_test_eng - mu_b) / sd_b
    train_loader_b = DataLoader(
        TensorDataset(torch.tensor(X_train_eng, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long)),
        batch_size=256, shuffle=True,
    )
    test_loader_b = DataLoader(
        TensorDataset(torch.tensor(X_test_eng, dtype=torch.float32), torch.tensor(y_test, dtype=torch.long)),
        batch_size=512,
    )
    model_b = MLP(ENGINEERED_DIM, 512, NUM_CLASSES, dropout=0.15).to(dev)
    criterion_b = nn.CrossEntropyLoss(label_smoothing=0.05)
    _train_model(model_b, train_loader_b, test_loader_b, criterion_b, 0.001, dev, 100, 15)
    probs_b = _extract_probs(model_b, test_loader_b, dev)
    avg_probs = 0.8 * probs_a + 0.2 * probs_b
    y_pred = np.argmax(avg_probs, axis=1).tolist()
    metrics = evaluate_classification(y_test.tolist(), y_pred, NUM_CLASSES)
    return extract_required_metrics(metrics)


if __name__ == "__main__":
    r1 = run_experiment("Single class-balanced CE on flat features", single_cb_ce_flat)
    r2 = run_experiment("Ensemble 80/20 (CB-CE flat + standard engineered)", ensemble_cb_ce)

    print("\n" + "="*60)
    print("COMPARISON (full 26K dataset)")
    print("="*60)
    print(f"{'Metric':<20} {'Single CB-CE':<15} {'Ensemble 80/20':<15}")
    print("-"*50)
    print(f"{'exact_accuracy':<20} {r1['exact_accuracy']:<15.4f} {r2['exact_accuracy']:<15.4f}")
    print(f"{'macro_f1':<20} {r1['macro_f1']:<15.4f} {r2['macro_f1']:<15.4f}")
    print(f"{'within_one_grade':<20} {r1['within_one_grade']:<15.4f} {r2['within_one_grade']:<15.4f}")
    print(f"{'within_two_grades':<20} {r1['within_two_grades']:<15.4f} {r2['within_two_grades']:<15.4f}")
