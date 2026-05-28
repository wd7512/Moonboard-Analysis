"""submissions/gradient-boost-baseline — Voting Ensemble grade predictor.

Combines RandomForest (300 trees) and HistGradientBoosting (500 iterations)
via soft-voting ensemble on 164-dim grid features plus 8 engineered meta-features.

Outperforms any single model by combining bagging + boosting strengths.

Exposes train_and_evaluate() for use by the benchmark harness.

Usage:
    uv run python submissions/gradient-boost-baseline/main.py --help
    uv run python submissions/gradient-boost-baseline/main.py \
        --data-path Raw/moonboard_problems_setup_2016.json
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    RandomForestClassifier,
    VotingClassifier,
)
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from moonboard_analysis.config import GRADE_ORDER
from moonboard_analysis.data.grid_mapping import GridMapper
from moonboard_analysis.data.loader import load_lstm_data
from moonboard_analysis.data.preprocessing import (
    drop_duplicate_sequences,
    preprocess_lstm_data,
)
from moonboard_analysis.utils.reproducibility import set_seeds

NUM_COLS = 11
NUM_ROWS = 18


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Voting Ensemble (RF + HistGB) grade classifier"
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default="Raw/moonboard_problems_setup_2016.json",
        help="Path to raw Moonboard JSON data",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    return parser.parse_args()


def hold_to_index(hold_name: str) -> int:
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


def compute_additional_features(sequences: list[list[str]]) -> np.ndarray:
    """Compute 8 engineered meta-features per route."""
    n = len(sequences)
    feats = np.zeros((n, 8), dtype=np.float32)
    for i, seq in enumerate(sequences):
        tokens = seq[:-2]
        start_holds: list[str] = []
        middle_holds: list[str] = []
        end_holds: list[str] = []

        section = "start"
        for token in tokens:
            if token == "START_END":
                section = "middle"
            elif token == "MIDDLE_END":
                section = "end"
            elif token == "END_ROUTE":
                pass
            elif section == "start":
                start_holds.append(token)
            elif section == "middle":
                middle_holds.append(token)
            elif section == "end":
                end_holds.append(token)

        n_start = len(start_holds)
        n_mid = len(middle_holds)
        n_end = len(end_holds)
        total = n_start + n_mid + n_end

        feats[i, 0] = n_start
        feats[i, 1] = n_mid
        feats[i, 2] = n_end
        feats[i, 3] = total
        feats[i, 4] = n_start / max(total, 1)
        feats[i, 5] = n_mid / max(total, 1)
        feats[i, 6] = n_end / max(total, 1)

        start_rows = []
        for h in start_holds:
            idx = hold_to_index(h)
            if idx >= 0:
                start_rows.append(idx // NUM_COLS)
        feats[i, 7] = np.mean(start_rows) if start_rows else 0.0
    return feats


def build_feature_matrix(sequences: list[list[str]]) -> np.ndarray:
    """Build 164-dim grid-based hold features for each route."""
    mapper = GridMapper()
    all_vecs: list[np.ndarray] = []

    for seq in sequences:
        tokens = seq[:-2]

        start_holds: list[str] = []
        middle_holds: list[str] = []
        end_holds: list[str] = []

        section = "start"
        for token in tokens:
            if token == "START_END":
                section = "middle"
            elif token == "MIDDLE_END":
                section = "end"
            elif token == "END_ROUTE":
                pass
            elif section == "start":
                start_holds.append(token)
            elif section == "middle":
                middle_holds.append(token)
            elif section == "end":
                end_holds.append(token)

        grid = np.zeros((3, 18, 11), dtype=np.float32)
        for hold in start_holds:
            row, col = GridMapper._convert_key(hold)
            grid[0, row, col] = 1
        for hold in middle_holds:
            row, col = GridMapper._convert_key(hold)
            grid[1, row, col] = 1
        for hold in end_holds:
            row, col = GridMapper._convert_key(hold)
            grid[2, row, col] = 1

        vec = mapper.grid_to_vector(grid)
        all_vecs.append(vec)

    return np.array(all_vecs, dtype=np.float32)


def extract_labels(sequences: list[list[str]]) -> list[int]:
    """Extract grade labels from sequences."""
    grade_to_idx = {g: i for i, g in enumerate(GRADE_ORDER)}
    return [grade_to_idx[seq[-2]] for seq in sequences]


def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> dict[str, float]:
    """Compute exact, within-1, within-2 accuracy metrics."""
    conf_matrix = confusion_matrix(y_true, y_pred, labels=range(n_classes))

    total_correct = sum(conf_matrix[i][i] for i in range(n_classes))
    exact = total_correct / conf_matrix.sum()

    def _within_k(k: int) -> float:
        correct = 0
        for i in range(n_classes):
            for j in range(max(0, i - k), min(n_classes, i + k + 1)):
                correct += conf_matrix[i, j]
        return correct / conf_matrix.sum()

    return {
        "exact_accuracy": exact,
        "within_one_grade": _within_k(1),
        "within_two_grades": _within_k(2),
    }


def train_and_evaluate(
    sequences: list[list[str]],
    grades: list[int],
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    seed: int = 42,
) -> dict[str, float]:
    """Train Voting Ensemble on fold and evaluate.

    Uses RF + HistGradientBoosting via soft voting. Handles missing classes
    in folds via LabelEncoder remapping.
    """
    set_seeds(seed)

    X = np.concatenate([
        build_feature_matrix(sequences),
        compute_additional_features(sequences),
    ], axis=1)
    y = np.array(grades, dtype=np.int64)
    n_classes = len(GRADE_ORDER)

    X_train, X_test = X[train_idx], X[test_idx]
    y_train_raw, y_test_raw = y[train_idx], y[test_idx]

    # Remap labels to contiguous 0-based for cases where a fold
    # doesn't contain all 12 grades.
    le = LabelEncoder()
    le.fit(y_train_raw)
    y_train = le.transform(y_train_raw)

    rf = RandomForestClassifier(n_estimators=300, random_state=seed)
    hist = HistGradientBoostingClassifier(
        max_iter=500, random_state=seed,
    )

    ensemble = VotingClassifier(
        estimators=[("rf", rf), ("hist", hist)],
        voting="soft",
    )
    ensemble.fit(X_train, y_train)
    y_pred_remapped = ensemble.predict(X_test)
    y_pred = le.inverse_transform(y_pred_remapped.astype(np.int64))

    return _compute_metrics(y_test_raw, y_pred, n_classes)


def main() -> None:
    args = parse_args()
    set_seeds(args.seed)

    data_path = args.data_path
    if not Path(data_path).exists():
        print(f"Error: Data file not found at '{data_path}'")
        print("Provide a valid path with --data-path")
        sys.exit(1)

    output_dir = Path(".")
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading data from {data_path}")
    df = load_lstm_data(data_path)
    print(f"Raw data: {len(df)} routes")

    sequences = preprocess_lstm_data(df)
    sequences = drop_duplicate_sequences(sequences)
    print(f"After preprocessing: {len(sequences)} unique sequences")

    print("Building feature matrix...")
    X_grid = build_feature_matrix(sequences)
    X_extra = compute_additional_features(sequences)
    X = np.concatenate([X_grid, X_extra], axis=1)
    y = np.array(extract_labels(sequences), dtype=np.int64)

    n_classes = len(GRADE_ORDER)
    print(f"Feature matrix: {X.shape}")
    print(f"Number of classes: {n_classes}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=args.seed, stratify=y,
    )
    print(f"Train: {X_train.shape[0]}  Test: {X_test.shape[0]}")

    print("Training Voting Ensemble (RF + HistGB)...")
    rf = RandomForestClassifier(n_estimators=300, random_state=args.seed)
    hist = HistGradientBoostingClassifier(
        max_iter=500, random_state=args.seed,
    )
    ensemble = VotingClassifier(
        estimators=[("rf", rf), ("hist", hist)], voting="soft",
    )
    ensemble.fit(X_train, y_train)
    print("Training complete.")

    y_pred = ensemble.predict(X_test).tolist()
    y_test_list = y_test.tolist()

    conf = confusion_matrix(y_test_list, y_pred, labels=range(n_classes))
    total_correct = sum(conf[i][i] for i in range(n_classes))
    exact = total_correct / conf.sum()

    def _within_k(k: int) -> float:
        correct = 0
        for i in range(n_classes):
            for j in range(max(0, i - k), min(n_classes, i + k + 1)):
                correct += conf[i, j]
        return correct / conf.sum()

    print()
    print("=" * 50)
    print("Evaluation Results")
    print("=" * 50)
    print(f"Exact Accuracy:      {exact:.4f}")
    print(f"Within-1 Accuracy:   {_within_k(1):.4f}")
    print(f"Within-2 Accuracy:   {_within_k(2):.4f}")


if __name__ == "__main__":
    main()
