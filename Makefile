.PHONY: install typecheck test lint train-ae train-lstm evaluate-lstm compare-pca clean

install:
	uv pip install -e ".[dev]"

typecheck:
	uv run mypy src/

test:
	uv run pytest tests/

lint:
	uv run ruff check src/

train-ae:
	uv run moonboard-train-ae

train-lstm:
	uv run moonboard-train-lstm

evaluate-lstm:
	uv run moonboard-evaluate-lstm

compare-pca:
	uv run moonboard-compare-pca

clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache .ruff_cache .mypy_cache mlruns/
