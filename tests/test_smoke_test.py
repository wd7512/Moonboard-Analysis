"""Tests for the smoke test CLI and sampling logic."""

from pathlib import Path

import numpy as np
import pytest

from moonboard_analysis.scripts.smoke_test import stratified_sample


class TestStratifiedSample:
    def test_preserves_total_samples(self):
        sequences = [[f"tok{i}"] for i in range(200)]
        grades = [i % 12 for i in range(200)]
        sampled_seqs, sampled_grades = stratified_sample(sequences, grades, 100, seed=42)
        assert len(sampled_seqs) == len(sampled_grades)
        assert abs(len(sampled_seqs) - 100) <= 12

    def test_preserves_grade_distribution(self):
        sequences = [[f"tok{i}"] for i in range(240)]
        grades = [i % 12 for i in range(240)]
        _, sampled_grades = stratified_sample(sequences, grades, 120, seed=42)
        unique, counts = np.unique(sampled_grades, return_counts=True)
        assert len(unique) == 12
        assert all(abs(c - 10) <= 1 for c in counts)

    def test_handles_small_dataset(self):
        sequences = [[f"tok{i}"] for i in range(6)]
        grades = [0, 0, 1, 1, 2, 2]
        sampled_seqs, sampled_grades = stratified_sample(sequences, grades, 100, seed=42)
        assert len(sampled_seqs) == 6
        assert len(sampled_grades) == 6

    def test_deterministic_with_seed(self):
        sequences = [[f"tok{i}"] for i in range(200)]
        grades = [i % 12 for i in range(200)]
        _, g1 = stratified_sample(sequences, grades, 100, seed=42)
        _, g2 = stratified_sample(sequences, grades, 100, seed=42)
        assert g1 == g2

    def test_different_seed_different_sample(self):
        sequences = [[f"tok{i}"] for i in range(200)]
        grades = [i % 12 for i in range(200)]
        _, g1 = stratified_sample(sequences, grades, 100, seed=42)
        _, g2 = stratified_sample(sequences, grades, 100, seed=99)
        assert g1 != g2


class TestSmokeTestCLI:
    def test_help_flag(self):
        import subprocess
        result = subprocess.run(
            ["uv", "run", "moonboard-smoke-test", "--help"],
            capture_output=True, text=True, timeout=30,
        )
        assert result.returncode == 0
        assert "usage:" in result.stdout.lower() or "usage:" in result.stderr.lower()

    def test_cli_missing_submission_dir(self):
        import subprocess
        result = subprocess.run(
            ["uv", "run", "moonboard-smoke-test"],
            capture_output=True, text=True, timeout=30,
        )
        assert result.returncode != 0
        assert "required" in result.stderr.lower() or "required" in result.stdout.lower()

    def test_cli_runs_on_fast_mlp(self):
        import subprocess
        sub_dir = Path(__file__).resolve().parent.parent / "submissions" / "fast-mlp"
        if not sub_dir.exists():
            pytest.skip("fast-mlp submission not found")
        result = subprocess.run(
            [
                "uv", "run", "moonboard-smoke-test",
                "--submission-dir", str(sub_dir),
                "--samples", "100",
            ],
            capture_output=True, text=True, timeout=120,
        )
        assert result.returncode == 0, f"stdout: {result.stdout}\nstderr: {result.stderr}"
        assert "exact_accuracy" in result.stdout
        assert "within_one_grade" in result.stdout
        assert "within_two_grades" in result.stdout
        assert "macro_f1" in result.stdout
