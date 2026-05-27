"""submissions/tree-baseline — Random Forest grade predictor baseline.

Trains a Random Forest classifier on 164-dimensional binary hold vectors
and evaluates using exact, within-1, and within-2 grade accuracy metrics.

Usage:
    uv run python submissions/tree-baseline/main.py --help
    uv run python submissions/tree-baseline/main.py --data-path Raw/moonboard_problems_setup_2016.json
"""

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
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
        description="Tree baseline — Random Forest grade classifier on binary hold vectors"
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
        "--n-estimators",
        type=int,
        default=200,
        help="Number of trees in the Random Forest (default: 200)",
    )
    return parser.parse_args()


def sequences_to_grids(
    sequences: list[list[str]],
) -> tuple[list[np.ndarray], list[int]]:
    """Convert route sequences to 164-dim binary feature vectors.

    Each sequence is a flat token list: [start_holds..., 'START_END',
    middle_holds..., 'MIDDLE_END', end_holds..., 'END_ROUTE', grade, 'GRADE_END'].

    Returns:
        features: List of 164-dim binary numpy arrays.
        labels: List of integer grade indices.
    """
    mapper = GridMapper()
    features: list[np.ndarray] = []
    labels: list[int] = []

    grade_to_idx = {g: i for i, g in enumerate(GRADE_ORDER)}

    for seq in sequences:
        grade = seq[-2]
        if grade not in grade_to_idx:
            continue

        tokens = seq[:-2]  # remove grade and 'GRADE_END'

        # Split into start / middle / end sections
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
                pass  # sentinel, already in end
            elif section == "start":
                start_holds.append(token)
            elif section == "middle":
                middle_holds.append(token)
            elif section == "end":
                end_holds.append(token)

        # Build 3x18x11 grid
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


def main() -> None:
    args = parse_args()
    set_seeds(args.seed)

    # -- Resolve paths --
    data_path = args.data_path
    if not Path(data_path).exists():
        print(f"Error: Data file not found at '{data_path}'")
        print("Provide a valid path with --data-path")
        sys.exit(1)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # -- Load & preprocess --
    print(f"Loading data from {data_path}")
    df = load_lstm_data(data_path)
    print(f"Raw data: {len(df)} routes")

    sequences = preprocess_lstm_data(df)
    sequences = drop_duplicate_sequences(sequences)
    print(f"After preprocessing: {len(sequences)} unique sequences")

    # -- Convert to binary feature vectors --
    print("Converting sequences to 164-dim binary hold vectors...")
    features_list, labels = sequences_to_grids(sequences)
    X = np.array(features_list, dtype=np.float32)
    y = np.array(labels, dtype=np.int64)

    num_classes = len(GRADE_ORDER)
    print(f"Feature matrix: {X.shape}")
    print(f"Number of classes: {num_classes}")

    # -- Train/test split --
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=args.seed,
        stratify=y,
    )
    print(f"Train: {X_train.shape[0]}  Test: {X_test.shape[0]}")

    # -- Train Random Forest --
    print(f"Training Random Forest (n_estimators={args.n_estimators})...")
    clf = RandomForestClassifier(
        n_estimators=args.n_estimators,
        random_state=args.seed,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)
    print("Training complete.")

    # -- Evaluate --
    y_pred = clf.predict(X_test).tolist()
    y_test_list = y_test.tolist()

    # Build confusion matrix with all classes present (avoids dimension mismatch
    # when some classes have no true samples in the test split).
    from sklearn.metrics import confusion_matrix

    conf_matrix = confusion_matrix(
        y_test_list, y_pred, labels=range(num_classes)
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

    # -- Save model --
    save_path = output_dir / "tree_model.joblib"
    joblib.dump(clf, save_path)
    print(f"Model saved to: {save_path}")


if __name__ == "__main__":
    main()
