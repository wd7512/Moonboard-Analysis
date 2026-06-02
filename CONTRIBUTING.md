# Contributing to Moonboard Analysis

Thanks for wanting to contribute. This covers how to submit models, report issues, and improve the project.

## Quick Start

```bash
git clone https://github.com/wd7512/Moonboard-Analysis.git
cd Moonboard-Analysis
uv sync
```

Run the tests:

```bash
uv run pytest tests/ -x -q
```

Train and evaluate the baseline LSTM:

```bash
uv run moonboard-train-lstm
uv run moonboard-evaluate-lstm
```

## Submitting a Model

See [`submissions/README.md`](submissions/README.md) for the full format. The short version:

1. Create `submissions/<your-model-name>/` with a `main.py` that accepts `--data-path`, `--output-dir`, `--seed` arguments
2. Run the gate checker first:

```bash
uv run python submissions/check_experiment.py \
    --submission-dir submissions/<your-model-name>
```

The checker validates the interface contract, no pre-trained weights, tree-method policy, reproducibility, required metrics, feature redundancy, code quality, and training time. Fix any failures before submitting a PR.

3. Check [`EXPERIMENTS.md`](EXPERIMENTS.md) to make sure your experiment isn't redundant. If your model family + feature representation is listed as "Exhausted" or "Banned", you need a novel variation.

4. Run the benchmark:

```bash
uv run moonboard-benchmark --submission-dir submissions/<your-model-name>
```

5. Submit a PR with your code, benchmark results, and an EXPERIMENTS.md entry.

## Code Style

```bash
uv run ruff check src/
uv run mypy src/
uv run pytest tests/ -x -q
```

All tests must pass before a PR will be merged.

## Pull Request Process

1. Fork the repo and create a feature branch
2. Make changes with focused commits
3. Add or update tests for new functionality
4. Make sure CI passes (lint, typecheck, test)
5. In the PR, describe your changes, include benchmark results if applicable, and list any new dependencies

## Reporting Issues

Open a GitHub issue with:
- What the bug or feature request is
- Steps to reproduce (for bugs)
- Expected vs actual behavior
- Python version and OS
