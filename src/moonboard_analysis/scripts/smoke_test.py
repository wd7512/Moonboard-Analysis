"""CLI smoke test — quick sanity check on a single submission using minimal data.

Runs one train/test split on a stratified subsample (default 1000 routes)
and prints key metrics. Intended to complete in <30s for any submission.

Usage:
    moonboard-smoke-test --submission-dir submissions/fast-mlp
    moonboard-smoke-test --submission-dir submissions/fast-mlp --samples 500
"""

import argparse
import importlib.util
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from moonboard_analysis.config import GRADE_ORDER
from moonboard_analysis.data.loader import load_lstm_data
from moonboard_analysis.data.preprocessing import (
    drop_duplicate_sequences,
    preprocess_lstm_data,
)
from moonboard_analysis.utils.reproducibility import set_seeds


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Quick smoke test for a Moonboard submission"
    )
    parser.add_argument(
        "--submission-dir",
        type=str,
        required=True,
        help="Path to submission directory containing main.py",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default=None,
        help="Path to raw Moonboard JSON data (default: Raw/moonboard_problems_setup_2016.json)",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=1000,
        help="Number of stratified samples to use (default: 1000)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


def load_submission(submission_dir: str):
    main_path = Path(submission_dir) / "main.py"
    if not main_path.exists():
        print(f"Error: No main.py found in '{submission_dir}'")
        sys.exit(1)

    spec = importlib.util.spec_from_file_location("submission_main", str(main_path))
    if spec is None or spec.loader is None:
        print(f"Error: Could not load '{main_path}'")
        sys.exit(1)

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    if not hasattr(module, "train_and_evaluate"):
        print(f"Error: 'main.py' in '{submission_dir}' must expose a train_and_evaluate function")
        sys.exit(1)

    return module.train_and_evaluate


def stratified_sample(
    sequences: list[list[str]],
    grades: list[int],
    n_samples: int,
    seed: int = 42,
) -> tuple[list[list[str]], list[int]]:
    """Stratified subsample to approximately n_samples while preserving grade distribution."""
    grade_indices: dict[int, list[int]] = defaultdict(list)
    for i, g in enumerate(grades):
        grade_indices[g].append(i)

    rng = np.random.default_rng(seed)
    sampled_indices: list[int] = []
    samples_per_grade = max(1, n_samples // len(GRADE_ORDER))

    for grade_idx in range(len(GRADE_ORDER)):
        indices = grade_indices.get(grade_idx, [])
        if len(indices) >= samples_per_grade:
            sampled = rng.choice(indices, samples_per_grade, replace=False)
        else:
            sampled = np.array(indices)
        sampled_indices.extend(sampled.tolist())

    rng.shuffle(sampled_indices)
    return (
        [sequences[i] for i in sampled_indices],
        [grades[i] for i in sampled_indices],
    )


def main() -> None:
    args = parse_args()
    set_seeds(args.seed)

    data_path = args.data_path or "Raw/moonboard_problems_setup_2016.json"
    if not Path(data_path).exists():
        print(f"Error: Data file not found at '{data_path}'")
        sys.exit(1)

    train_and_evaluate = load_submission(args.submission_dir)

    print(f"Loading data from {data_path}")
    df = load_lstm_data(data_path)
    df = df.drop_duplicates(subset=["Name"])
    sequences = preprocess_lstm_data(df, augment=False)
    sequences = drop_duplicate_sequences(sequences)

    route_sequences: list[list[str]] = []
    route_grades: list[int] = []
    grade_to_idx = {g: i for i, g in enumerate(GRADE_ORDER)}
    for seq in sequences:
        grade = seq[-2]
        if grade in grade_to_idx:
            route_sequences.append(seq)
            route_grades.append(grade_to_idx[grade])

    print(f"Total available routes: {len(route_sequences)}")

    if args.samples < len(route_sequences):
        route_sequences, route_grades = stratified_sample(
            route_sequences, route_grades, args.samples, seed=args.seed
        )
        print(f"Subsampled to: {len(route_sequences)} routes (stratified)")

    from sklearn.model_selection import train_test_split

    train_idx, test_idx = train_test_split(
        np.arange(len(route_sequences)),
        test_size=0.2,
        random_state=args.seed,
        stratify=route_grades,
    )

    print(f"Running single-fold smoke test on {args.submission_dir}...")
    t0 = time.time()
    try:
        results = train_and_evaluate(
            sequences=route_sequences,
            grades=route_grades,
            train_idx=train_idx,
            test_idx=test_idx,
            seed=args.seed,
        )
    except Exception as e:
        print(f"[FAIL] Smoke test failed with exception: {e}")
        sys.exit(1)

    elapsed = time.time() - t0

    print(f"\n{'=' * 50}")
    print("Smoke Test Results")
    print(f"{'=' * 50}")
    print(f"  exact_accuracy:     {results['exact_accuracy']:.4f}")
    print(f"  within_one_grade:   {results['within_one_grade']:.4f}")
    print(f"  within_two_grades:  {results['within_two_grades']:.4f}")
    print(f"  macro_f1:           {results.get('macro_f1', 0.0):.4f}")
    print(f"  time:               {elapsed:.1f}s")
    print(f"{'=' * 50}")

    sys.exit(0)


if __name__ == "__main__":
    main()
