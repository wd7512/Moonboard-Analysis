# Contributing to Moonboard Analysis

Thank you for your interest in contributing! This document outlines how to submit models, report issues, and improve the project.

## Quick Start

```bash
# Clone and set up
git clone https://github.com/wd7512/Moonboard-Analysis.git
cd Moonboard-Analysis
uv sync

# Run tests
uv run pytest tests/ -x -q

# Train the baseline LSTM
uv run moonboard-train-lstm

# Evaluate
uv run moonboard-evaluate-lstm
```

## Submitting a Model

See [`submissions/README.md`](submissions/README.md) for the full submission format. In short:

1. Create a folder under `submissions/<your-model-name>/`
2. Include a `main.py` with `--data-path`, `--output-dir`, `--seed` arguments
3. Train, evaluate, and save your model
4. Submit a PR with your code and benchmark results

## Code Style

- **Linting:** `uv run ruff check src/`
- **Type checking:** `uv run mypy src/`
- **Tests:** `uv run pytest tests/ -x -q`
- All tests must pass before submitting a PR

## Pull Request Process

1. Fork the repository and create a feature branch
2. Make your changes with clear, focused commits
3. Add/update tests for any new functionality
4. Ensure CI passes (lint, typecheck, test)
5. Submit a PR with:
   - Description of changes
   - Benchmark results (if applicable)
   - Any new dependencies

## Reporting Issues

Open a GitHub issue with:
- Description of the bug or feature request
- Steps to reproduce (for bugs)
- Expected vs actual behavior
- Python version and OS

## Code of Conduct

Be respectful and constructive. We welcome contributors of all experience levels.
