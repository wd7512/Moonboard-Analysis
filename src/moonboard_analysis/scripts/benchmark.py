"""CLI script for running retrain-per-fold CV benchmark on submissions.

Loads data, runs KFold cross-validation, calling each submission's
train_and_evaluate() per fold, and outputs aggregated results.

Usage:
    moonboard-benchmark --submission-dir submissions/lstm-baseline \
        --data-path Raw/moonboard_problems_setup_2016.json \
        --output-json results.json \
        --output-markdown results.md
"""

import argparse
import importlib.util
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
from sklearn.model_selection import KFold

from moonboard_analysis.config import GRADE_ORDER
from moonboard_analysis.data.loader import load_lstm_data
from moonboard_analysis.data.preprocessing import (
    drop_duplicate_sequences,
    preprocess_lstm_data,
)
from moonboard_analysis.training.benchmark import BenchmarkResults
from moonboard_analysis.utils.reproducibility import set_seeds


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run retrain-per-fold CV benchmark on a submission"
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
        "--output-json",
        type=str,
        default="results.json",
        help="Path to write results JSON file",
    )
    parser.add_argument(
        "--output-markdown",
        type=str,
        default="results.md",
        help="Path to write results Markdown file",
    )
    parser.add_argument(
        "--n-splits",
        type=int,
        default=5,
        help="Number of CV folds (default: 5)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum samples (default: all). Prefer moonboard-smoke-test for fast checks.",
    )
    return parser.parse_args()


def load_submission(submission_dir: str):
    """Import train_and_evaluate from a submission's main.py.

    Args:
        submission_dir: Path to submission directory.

    Returns:
        The train_and_evaluate function.

    Raises:
        SystemExit: If main.py is missing or lacks train_and_evaluate.
    """
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


def format_leaderboard_summary(results: BenchmarkResults) -> str:
    """Format a leaderboard-style summary of CV results.

    Args:
        results: BenchmarkResults with fold data.

    Returns:
        Formatted string for console display.
    """
    means = results.mean_scores()
    stds = results.std_scores()

    lines = []
    lines.append("\n" + "=" * 60)
    lines.append("CROSS-VALIDATION BENCHMARK RESULTS")
    lines.append("=" * 60)
    lines.append(f"{'Metric':<25} {'Mean':<10} {'Std':<10}")
    lines.append("-" * 45)
    for metric_name in means:
        lines.append(f"{metric_name:<25} {means[metric_name]:<10.4f} {stds[metric_name]:<10.4f}")
    lines.append("=" * 60)
    return "\n".join(lines)


def generate_markdown_report(
    results: BenchmarkResults,
    submission_dir: str,
    data_path: str,
    n_splits: int,
) -> str:
    """Generate a comprehensive Markdown report of CV benchmark results.

    Args:
        results: BenchmarkResults with fold data.
        submission_dir: Path to evaluated submission.
        data_path: Path to evaluation data.
        n_splits: Number of CV folds.

    Returns:
        Formatted Markdown string.
    """
    means = results.mean_scores()
    stds = results.std_scores()

    lines = []
    lines.append("# Moonboard CV Benchmark Results")
    lines.append("")
    lines.append("## Metadata")
    lines.append(f"- **Submission**: {submission_dir}")
    lines.append(f"- **Data**: {data_path}")
    lines.append(f"- **Timestamp**: {datetime.now().isoformat()}")
    lines.append(f"- **CV Folds**: {n_splits}")
    lines.append("")

    lines.append("## Overall Metrics")
    lines.append("| Metric | Mean ± Std |")
    lines.append("|--------|------------|")
    for metric_name in means:
        lines.append(f"| {metric_name} | {means[metric_name]:.4f} ± {stds[metric_name]:.4f} |")
    lines.append("")

    lines.append("## Per-Fold Results")
    lines.append(results.to_markdown_table())
    lines.append("")

    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    set_seeds(args.seed)

    # Determine data path
    data_path = args.data_path or "Raw/moonboard_problems_setup_2016.json"
    if not Path(data_path).exists():
        print(f"Error: Data file not found at '{data_path}'")
        print("Please provide a valid path with --data-path")
        sys.exit(1)

    # Load submission
    print(f"Loading submission from {args.submission_dir}")
    train_and_evaluate = load_submission(args.submission_dir)

    # Load and preprocess data
    print(f"Loading data from {data_path}")
    df = load_lstm_data(data_path)
    print(f"Raw data: {len(df)} routes")

    # Deduplicate by route name to prevent the same route from appearing
    # in both train and test folds after CV split.
    df = df.drop_duplicates(subset=["Name"])
    print(f"After deduplication by route name: {len(df)} unique routes")

    # Augmentation disabled during CV benchmarking — hold-swap creates
    # multiple token sequences per route that could leak across folds.

    sequences = preprocess_lstm_data(df, augment=False)
    sequences = drop_duplicate_sequences(sequences)
    print(f"After preprocessing: {len(sequences)} unique sequences")

    route_sequences: list[list[str]] = []
    route_grades: list[str] = []
    for seq in sequences:
        grade = seq[-2]
        if grade in GRADE_ORDER:
            route_sequences.append(seq)
            route_grades.append(grade)

    grade_to_idx = {g: i for i, g in enumerate(GRADE_ORDER)}
    encoded_grades = [grade_to_idx[g] for g in route_grades]
    num_samples = len(route_sequences)
    print(f"Valid routes: {num_samples}")

    # Optionally cap samples for faster benchmarking
    if args.max_samples is not None and args.max_samples < num_samples:
        # Sample with stratification by grade to maintain class distribution
        from collections import defaultdict
        grade_indices = defaultdict(list)
        for i, g in enumerate(encoded_grades):
            grade_indices[g].append(i)

        # Sample proportionally from each grade
        np.random.seed(args.seed)
        sampled_indices = []
        samples_per_grade = args.max_samples // len(GRADE_ORDER)
        for grade_idx in range(len(GRADE_ORDER)):
            indices = grade_indices[grade_idx]
            if len(indices) >= samples_per_grade:
                sampled = np.random.choice(indices, samples_per_grade, replace=False)
            else:
                sampled = np.array(indices)
            sampled_indices.extend(sampled)

        np.random.shuffle(sampled_indices)
        route_sequences = [route_sequences[i] for i in sampled_indices]
        encoded_grades = [encoded_grades[i] for i in sampled_indices]
        num_samples = len(route_sequences)
        print(f"Capped to: {num_samples} samples (stratified)")

    # Run CV
    n_splits = args.n_splits
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=args.seed)
    fold_results: list[dict[str, float]] = []

    print(f"\nRunning {n_splits}-fold cross-validation...")
    all_indices = np.arange(num_samples)

    for fold_idx, (train_idx, test_idx) in enumerate(kfold.split(all_indices)):
        print(f"  Fold {fold_idx + 1}/{n_splits}...")
        metrics = train_and_evaluate(
            sequences=route_sequences,
            grades=encoded_grades,
            train_idx=train_idx,
            test_idx=test_idx,
            seed=args.seed,
        )
        print(f"    Results: {metrics}")
        fold_results.append(metrics)

    results = BenchmarkResults(fold_results=fold_results)
    print(f"\nMean scores: {results.mean_scores()}")
    print(f"Std scores:  {results.std_scores()}")

    # Write JSON output
    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    json_str = results.to_json()
    output_json.write_text(json_str)
    print(f"JSON results written to {output_json}")

    # Write Markdown output
    output_markdown = Path(args.output_markdown)
    output_markdown.parent.mkdir(parents=True, exist_ok=True)
    markdown = generate_markdown_report(results, args.submission_dir, data_path, n_splits)
    output_markdown.write_text(markdown)
    print(f"Markdown report written to {output_markdown}")

    # Print leaderboard to console
    print(format_leaderboard_summary(results))


if __name__ == "__main__":
    main()
