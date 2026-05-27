"""submissions/ridge-baseline — Ridge regression grade predictor baseline.

Trains a Ridge regression classifier on Moonboard route data. Each route is
converted to a 164-dimensional binary hold vector (3 layers: start/middle/end),
then a Ridge regression model is trained to predict the climbing grade.

The model predicts continuous values which are rounded to the nearest integer
grade index for classification metrics.

Exposes train_and_evaluate() for use by the benchmark harness.

Usage:
    uv run python submissions/ridge-baseline/main.py --help
    uv run python submissions/ridge-baseline/main.py \\
        --data-path Raw/moonboard_problems_setup_2016.json
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split

from moonboard_analysis.config import GRADE_ORDER
from moonboard_analysis.data.grid_mapping import GridMapper
from moonboard_analysis.data.loader import load_lstm_data
from moonboard_analysis.data.preprocessing import (
    drop_duplicate_sequences,
    preprocess_lstm_data,
)
from moonboard_analysis.utils.reproducibility import set_seeds


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ridge regression baseline — train and evaluate on Moonboard data"
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default="Raw/moonboard_problems_setup_2016.json",
        help="Path to raw Moonboard JSON data (default: Raw/moonboard_problems_setup_2016.json)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=".",
        help="Directory to save trained model (default: current directory)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=1.0,
        help="Ridge regularization strength (default: 1.0)",
    )
    return parser.parse_args()


def sequences_to_grids(
    sequences: list[list[str]],
) -> tuple[list[np.ndarray], list[int]]:
    mapper = GridMapper()
    features: list[np.ndarray] = []
    labels: list[int] = []

    grade_to_idx = {g: i for i, g in enumerate(GRADE_ORDER)}

    for seq in sequences:
        grade = seq[-2]
        if grade not in grade_to_idx:
            continue

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
        features.append(vec)
        labels.append(grade_to_idx[grade])

    return features, labels


def train_and_evaluate(
    sequences: list[list[str]],
    grades: list[int],
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    seed: int = 42,
    alpha: float = 1.0,
) -> dict[str, float]:
    """Train a fresh Ridge regression model on the training fold and evaluate on test fold.

    Args:
        sequences: Full preprocessed route sequences. Each sequence includes
            hold tokens followed by the grade string at index [-2] and
            'GRADE_END' at index [-1].
        grades: Integer-encoded grade labels (parallel to sequences).
        train_idx: Indices into sequences/grades for the training fold.
        test_idx: Indices into sequences/grades for the test fold.
        seed: Random seed for reproducibility.
        alpha: Ridge regularization strength.

    Returns:
        Dict with exact_accuracy, within_one_grade, within_two_grades.
    """
    set_seeds(seed)

    train_seqs = [sequences[i] for i in train_idx]
    test_seqs = [sequences[i] for i in test_idx]

    features_list_train, labels_train = sequences_to_grids(train_seqs)
    features_list_test, labels_test = sequences_to_grids(test_seqs)

    X_train = np.array(features_list_train, dtype=np.float32)
    y_train = np.array(labels_train, dtype=np.float32)
    X_test = np.array(features_list_test, dtype=np.float32)
    y_test = np.array(labels_test, dtype=np.int64)

    num_classes = len(GRADE_ORDER)

    model = Ridge(alpha=alpha, random_state=seed)
    model.fit(X_train, y_train)

    # Round predictions to nearest valid grade index
    y_pred_raw = model.predict(X_test)
    y_pred = np.clip(np.round(y_pred_raw), 0, num_classes - 1).astype(int)

    # Compute metrics via confusion matrix
    from sklearn.metrics import confusion_matrix

    conf_matrix = confusion_matrix(
        y_test, y_pred, labels=range(num_classes)
    )

    total_correct = sum(conf_matrix[i][i] for i in range(num_classes))
    exact_accuracy = total_correct / conf_matrix.sum()

    def _within_k(k: int) -> float:
        correct = 0
        for i in range(num_classes):
            for j in range(max(0, i - k), min(num_classes, i + k + 1)):
                correct += conf_matrix[i, j]
        return correct / conf_matrix.sum()

    within_1 = _within_k(1)
    within_2 = _within_k(2)

    return {
        "exact_accuracy": exact_accuracy,
        "within_one_grade": within_1,
        "within_two_grades": within_2,
    }


def main() -> None:
    args = parse_args()
    set_seeds(args.seed)

    data_path = args.data_path
    if not Path(data_path).exists():
        print(f"Error: Data file not found at '{data_path}'")
        print("Provide a valid path with --data-path")
        sys.exit(1)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading data from {data_path}")
    df = load_lstm_data(data_path)
    print(f"Raw data: {len(df)} routes")

    sequences = preprocess_lstm_data(df)
    sequences = drop_duplicate_sequences(sequences)
    print(f"After preprocessing: {len(sequences)} unique sequences")

    print("Converting sequences to 164-dim binary hold vectors...")
    features_list, labels = sequences_to_grids(sequences)
    X = np.array(features_list, dtype=np.float32)
    y = np.array(labels, dtype=np.int64)

    num_classes = len(GRADE_ORDER)
    print(f"Feature matrix: {X.shape}")
    print(f"Number of classes: {num_classes}")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=args.seed,
        stratify=y,
    )
    print(f"Train: {X_train.shape[0]}  Test: {X_test.shape[0]}")

    print(f"Ridge regression (alpha={args.alpha})...")
    model = Ridge(alpha=args.alpha, random_state=args.seed)
    model.fit(X_train, y_train.astype(np.float32))
    print("Training complete.")

    y_pred_raw = model.predict(X_test)
    y_pred = np.clip(np.round(y_pred_raw), 0, num_classes - 1).astype(int)

    from sklearn.metrics import confusion_matrix

    conf_matrix = confusion_matrix(
        y_test, y_pred, labels=range(num_classes)
    )

    total_correct = sum(conf_matrix[i][i] for i in range(num_classes))
    exact_accuracy = total_correct / conf_matrix.sum()

    def _within_k(k: int) -> float:
        correct = 0
        for i in range(num_classes):
            for j in range(max(0, i - k), min(num_classes, i + k + 1)):
                correct += conf_matrix[i, j]
        return correct / conf_matrix.sum()

    within_1 = _within_k(1)
    within_2 = _within_k(2)

    print()
    print("=" * 50)
    print("Evaluation Results")
    print("=" * 50)
    print(f"Exact Accuracy:      {exact_accuracy:.4f}")
    print(f"Within-1 Accuracy:   {within_1:.4f}")
    print(f"Within-2 Accuracy:   {within_2:.4f}")

    import joblib

    save_path = output_dir / "ridge_model.joblib"
    joblib.dump(model, save_path)
    print(f"Model saved to: {save_path}")


if __name__ == "__main__":
    main()
