# Moonboard Dual-Dataset Benchmark — Implementation Plan

> **Goal:** Add Masters 2017 hold-setup support to the Moonboard benchmark suite, re-run all 13 submissions on both 2016 and 2017 datasets, and produce a two-column leaderboard. Closes issues #36 and #32.

**Architecture:** Four phases — (1) grade config, (2) grid mapping, (3) overnight benchmarks, (4) leaderboard merge. Each phase ends with a gate. TDD throughout.

**Data isolation:** 2016 and 2017 benchmarks are independent runs. No data crosses between them. The leaderboard stores each dataset's results separately.

**Tech Stack:** Python, PyTorch, sklearn, numpy, pytest, ruff, GitHub Actions CI.

---

## Phase 1: GRADE_ORDER Support for Masters 2017

Add `"6B"` to `GRADE_ORDER` so the 1,766 Masters 2017 routes at that grade are included. This adds 1 class (12 → 13) to all models' output layers dynamically since they read `len(GRADE_ORDER)` at runtime.

### Phase Dependency Map

Phase 1 (grade fix) must come before everything else — benchmark results would be wrong without it.
Phase 2 (grid mapping) affects only 3 submissions; the other 10 can run without it.
Phase 3 (benchmarks) depends on Phase 1 + 2.
Phase 4 (leaderboard) depends on Phase 3.

### Gate 1: GRADE_ORDER + test suite passes, lint clean
- `GRADE_ORDER` includes "6B" at correct position
- All existing tests pass (some need updating for 13-class assumption)
- `ruff check src tests` passes
- `mypy src` passes

### Task 1.1: Update GRADE_ORDER in config.py

**Objective:** Insert "6B" after "6A+" in the grade list.

**Files:**
- Modify: `src/moonboard_analysis/config.py:3`

**Step 1: Understand current ordering**

The current order is:
```python
GRADE_ORDER = ["6A", "6A+", "6B+", "6C", "6C+", "7A", "7A+", "7B", "7B+", "7C", "7C+", "8A"]
```

"6B" is missing. It belongs between "6A+" and "6B+" (6B < 6B+).

**Step 2: Write failing test**

Add to a new test in `tests/test_config.py`:

```python
"""Tests for Moonboard configuration constants."""

from moonboard_analysis.config import GRADE_ORDER


class TestGradeOrder:
    """Grade ordering and completeness."""

    def test_grade_order_includes_6b(self) -> None:
        """6B should be in GRADE_ORDER for Masters 2017 support."""
        assert "6B" in GRADE_ORDER

    def test_grade_order_length(self) -> None:
        """After adding 6B, there should be 13 grades."""
        assert len(GRADE_ORDER) == 13

    def test_grade_order_sorted(self) -> None:
        """Grades should be in ascending difficulty order."""
        for i in range(len(GRADE_ORDER) - 1):
            assert GRADE_ORDER[i] < GRADE_ORDER[i + 1], (
                f"{GRADE_ORDER[i]} should come before {GRADE_ORDER[i + 1]}"
            )
```

Run: `uv run pytest tests/test_config.py -v`
Expected: FAIL — `GRADE_ORDER` doesn't include "6B", length is 12

**Step 3: Implement**

```python
GRADE_ORDER = ["6A", "6A+", "6B", "6B+", "6C", "6C+", "7A", "7A+", "7B", "7B+", "7C", "7C+", "8A"]
```

**Step 4: Verify**

Run: `uv run pytest tests/test_config.py -v`
Expected: PASS

### Task 1.2: Update tests that hardcode 12-class assumption

**Objective:** Find and fix test fixtures and assertions that assume `len(GRADE_ORDER) == 12`.

**Files:**
- Modify: `tests/test_benchmark.py`
- Modify: `tests/test_metrics.py`
- Modify: `tests/test_models.py`
- Modify: `tests/test_smoke_test.py`
- Modify: `tests/test_class_balanced_loss.py`

**Step 1: Find hardcoded 12 references**

Run: `grep -rn "12" tests/ --include="*.py" | grep -v "__pycache__" | grep -v ".pyc"`

Look for:
- `num_classes=12` or `NUM_CLASSES = 12`
- `len(GRADE_ORDER)` used to generate 12-element fixtures
- Test data generators that produce 12-class labels

**Step 2: Fix each occurrence**

For each hardcoded `12` that's a class count, replace with `len(GRADE_ORDER)` or `13`.

Example pattern in test fixtures:
```python
# Before
NUM_CLASSES = 12
# After
from moonboard_analysis.config import GRADE_ORDER
NUM_CLASSES = len(GRADE_ORDER)
```

**Step 3: Verify**

Run: `uv run pytest tests/ -q --tb=short`
Expected: All passing (or known pre-existing failures only)

### Task 1.3: Verify no other hardcoded 12s in submissions

**Objective:** Check submissions don't hardcode 12 anywhere.

**Files:** Check all `submissions/*/main.py`

**Verification:**
```bash
grep -rn "12\|NUM_CLASSES\s*=\|num_classes\s*=" submissions/*/main.py | grep -v "__pycache__"
```

All submissions should use `len(GRADE_ORDER)` or `from moonboard_analysis.config import GRADE_ORDER`. If any hardcode 12, fix them to import dynamically.

**Step 4: Gate checkpoint**

```bash
uv run pytest tests/ -q --tb=short && uv run ruff check src tests && uv run mypy src
```

All must pass.

### Task 1.5: Commit phase 1

```bash
git add -A && git commit -m "feat: add 6B to GRADE_ORDER for Masters 2017 support

- Insert 6B between 6A+ and 6B+ (13 classes total)
- Update all test fixtures to use len(GRADE_ORDER) dynamically
- 1,766 additional routes from Masters 2017 now included"
```

---

## Phase 2: Grid Mapping Support for Masters 2017

GridMapper compresses a 242-dim grid (3 layers × 18 rows × 11 cols) into 164-dim by removing 78 positions that are always zero (null holds) on the 2016 board. On the Masters 2017 board, ALL 198 hold positions are used — zero null holds — so the compressed representation would incorrectly zero out real data.

Design: Add a `setup` parameter to GridMapper that selects the appropriate null-hold set. This is the SOLID open/closed approach — new setups don't modify existing code.

### Gate 2: GridMapper supports both setups
- GridMapper accepts `setup="2016"` (default, backward-compatible) and `setup="master2017"`
- `vector_to_grid` and `grid_to_vector` work correctly for both setups
- All existing GridMapper tests still pass (backward compatible)
- New tests cover 2017 setup
- 3 GridMapper-dependent submissions pass on 2017 data

### Task 2.1: Write failing tests for dual-setup GridMapper

**Objective:** Test that GridMapper can handle both 2016 and 2017 hold setups.

**Files:**
- Modify: `tests/test_grid_mapping.py`

**Step 1: Understand the null holds for each setup**

From our analysis:
- 2016: 58 null hold descriptions → 78 insert indices → 164-dim vector (242 - 78)
- 2017: 0 null hold descriptions → 0 insert indices → 242-dim vector (full grid)

**Step 2: Write failing tests**

Add to `tests/test_grid_mapping.py`:

```python
class TestGridMapperSetup:
    """Test GridMapper with different hold setups."""

    def test_default_setup_is_2016(self) -> None:
        """Default constructor should use 2016 setup."""
        mapper = GridMapper()
        assert mapper.setup == "2016"

    def test_2016_setup_vector_dim(self) -> None:
        """2016 setup should produce 164-dim vectors."""
        mapper = GridMapper(setup="2016")
        vec = mapper.grid_to_vector(np.zeros((3, 18, 11)))
        assert vec.shape == (164,)

    def test_2017_setup_vector_dim(self) -> None:
        """Master 2017 setup should produce 242-dim vectors (no compression)."""
        mapper = GridMapper(setup="master2017")
        grid = np.zeros((3, 18, 11))
        vec = mapper.grid_to_vector(grid)
        assert vec.shape == (242,)

    def test_2017_null_holds_empty(self) -> None:
        """Master 2017 has no null holds — all positions used."""
        mapper = GridMapper(setup="master2017")
        assert len(mapper.NULL_HOLDS) == 0

    def test_2017_vector_roundtrip(self) -> None:
        """Round-trip should preserve full 242-dim vector."""
        mapper = GridMapper(setup="master2017")
        rng = np.random.default_rng(42)
        original = rng.random(242).astype(np.float32)
        grid = mapper.vector_to_grid(original)
        recovered = mapper.grid_to_vector(grid)
        np.testing.assert_array_almost_equal(original, recovered)

    def test_2017_grid_roundtrip(self) -> None:
        """Round-trip should preserve full 3x18x11 grid."""
        mapper = GridMapper(setup="master2017")
        rng = np.random.default_rng(42)
        original = rng.random((3, 18, 11)).astype(np.float32)
        vec = mapper.grid_to_vector(original)
        recovered = mapper.vector_to_grid(vec)
        np.testing.assert_array_almost_equal(original, recovered)
```

Run: `uv run pytest tests/test_grid_mapping.py::TestGridMapperSetup -v`
Expected: FAIL — GridMapper doesn't accept `setup` parameter yet

### Task 2.2: Implement dual-setup GridMapper

**Objective:** Add `setup` parameter to GridMapper with null holds for each setup.

**Files:**
- Modify: `src/moonboard_analysis/data/grid_mapping.py`

**Step 1: Design the approach**

```python
class GridMapper:
    """Maps between condensed vectors and 3x18x11 Moonboard grid.
    
    Supports multiple hold setups via the `setup` parameter:
    - "2016" (default): 164-dim vectors (58 null holds compressed)
    - "master2017": 242-dim vectors (no null holds — all positions used)
    """
    
    # Shared configuration
    NUM_ROWS = 18
    NUM_COLS = 11
    
    # Per-setup null holds
    _NULL_HOLDS: dict[str, list[str]] = {
        "2016": [
            "F18", "J18", "A17", "B17", "C17", "E17", "F17",
            "H17", "I17", "J17", "K17", "J15", "K15", "B14",
            "A8", "A7", "A6", "H6", "B5", "E5", "G5",
            "A4", "C4", "D4", "E4", "F4", "H4", "J4", "K4",
            "A3", "C3", "E3", "F3", "G3", "H3", "I3", "J3", "K3",
            "A2", "B2", "C2", "D2", "E2", "F2", "H2", "I2", "K2",
            "A1", "B1", "C1", "D1", "E1", "F1", "G1", "H1", "I1", "J1", "K1",
        ],
        "master2017": [],  # All positions used
    }
    
    _insert_indices: dict[str, list[int]] = {}
```

**Step 2: Implement**

Modified `__init__`:
```python
def __init__(self, setup: str = "2016") -> None:
    self.setup = setup
    if setup not in self._NULL_HOLDS:
        raise ValueError(f"Unknown setup: {setup!r}. Choose from: {list(self._NULL_HOLDS.keys())}")
    self.NULL_HOLDS = self._NULL_HOLDS[setup]
    if setup not in self._insert_indices:
        self._insert_indices[setup] = self._compute_insert_indices()
```

Modified `_compute_insert_indices` to use instance's NULL_HOLDS:
```python
def _compute_insert_indices(self) -> list[int]:
    null_positions: set[tuple[int, int]] = set()
    for hold in self.NULL_HOLDS:
        row, col = self._convert_key(hold)
        null_positions.add((row, col))
    # ... rest same but use self.NULL_HOLDS
```

**Step 3: Verify**

Run: `uv run pytest tests/test_grid_mapping.py -v`
Expected: All tests pass (existing + new)

### Task 2.3: Verify backward compatibility

**Objective:** Ensure all existing GridMapper consumers still work.

**Step 1: Check all GridMapper instantiations**

```bash
grep -rn "GridMapper()" src/ submissions/ tests/ --include="*.py"
```

Every existing `GridMapper()` should still work (defaults to "2016"). No changes needed to consumers that only use the 2016 dataset.

### Task 2.4: Update GridMapper-dependent submissions to support 2017

**Objective:** The 3 submissions (tree-baseline, ridge-baseline, gradient-boost-baseline) use GridMapper's 164-dim compressed representation. For 2017, they need to use the full 242-dim vector.

**Approach:** Modify these submissions to accept a `--data-path` derived setup parameter, or auto-detect setup from the data. The simplest TDD approach: pass the vector dimension through.

**Files:**
- Modify: `submissions/tree-baseline/main.py`
- Modify: `submissions/ridge-baseline/main.py`
- Modify: `submissions/gradient-boost-baseline/main.py`

**Step 1: Check how each submission uses GridMapper**

Read each submission to understand the GridMapper usage pattern. The typical pattern is:
```python
mapper = GridMapper()
# ... later ...
features = mapper.grid_to_vector(grid_data)
```

**Step 2: Write failing test for auto-detect**

The cleanest approach: update GridMapper to accept a `setup` inference from the data, or just let the benchmark pass the setup through.

Actually, looking at the benchmark pipeline: the benchmark CLI calls `train_and_evaluate()` which each submission defines. The submission reads the data itself via `preprocess_lstm_data()`. So the submission needs to know which dataset it's running on.

The simplest approach that doesn't require changes to the benchmarking script: pass the dataset name or setup as part of the submission's configuration. The benchmark already imports `GRADE_ORDER` dynamically — we can do the same for GridMapper setup.

**For the overnight plan:** Since the benchmark script loads and preprocesses data, then calls `train_and_evaluate(sequences, grades, ...)`, the submissions that use GridMapper also call `preprocess_lstm_data()` internally to re-parse sequences into vectors. The dataset is already implicitly available.

**Solution:** Update GridMapper instantiation in these 3 submissions to check dataset characteristics at runtime (number of unique holds, hold range patterns). If row 1 holds exist → 2017 setup. Otherwise → 2016.

Actually, that's fragile. Simpler: extract a function that detects setup from the sequence data:

```python
def _detect_grid_setup(sequences: list[list[str]]) -> str:
    """
    Detect Moonboard setup from hold tokens in sequences.
    
    Masters 2017 uses row 1 holds (A1-K1) which are always null
    on the 2016 board. If any row-1 hold appears, it's 2017.
    """
    for seq in sequences:
        for token in seq:
            if token and token[0].isalpha() and len(token) >= 2:
                try:
                    if int(token[1:]) == 1:
                        return "master2017"
                except ValueError:
                    continue
    return "2016"
```

Then in each submission's `train_and_evaluate`:
```python
setup = _detect_grid_setup(sequences)
mapper = GridMapper(setup=setup)
```

Write tests for this detection function first.

**Step 3: Modify submissions**

Each of the 3 submissions gets a `_detect_grid_setup` helper (or import it from a shared location) and passes it to GridMapper.

Most pragmatic: add `_detect_grid_setup` to `grid_mapping.py` as a module-level function, then import in submissions.

### Task 2.5: Write tests for setup detection

Add to `tests/test_grid_mapping.py`:

```python
from moonboard_analysis.data.grid_mapping import detect_grid_setup


class TestDetectGridSetup:
    """Test automatic setup detection from route sequences."""

    def test_2016_no_row1(self) -> None:
        """Sequences without row-1 holds should be 2016."""
        seqs = [["A18", "GRADE_END", "B10", "MIDDLE_END", "K18", "END_ROUTE", "6B+"]]
        assert detect_grid_setup(seqs) == "2016"

    def test_2017_has_row1(self) -> None:
        """Sequences with row-1 holds should be master2017."""
        seqs = [["A1", "GRADE_END", "B10", "MIDDLE_END", "K18", "END_ROUTE", "6B+"]]
        assert detect_grid_setup(seqs) == "master2017"

    def test_empty_sequences(self) -> None:
        """Empty sequence list should default to 2016."""
        assert detect_grid_setup([]) == "2016"
```

Run: `uv run pytest tests/test_grid_mapping.py::TestDetectGridSetup -v`
Expected: FAIL — `detect_grid_setup` not defined

### Task 2.6: Commit phase 2

```bash
git add -A && git commit -m "feat: support Masters 2017 hold setup in GridMapper

- Add 'setup' parameter to GridMapper ('2016' | 'master2017')
- master2017: empty null holds → 242-dim vectors (no compression)
- Add detect_grid_setup() for auto-detection from route sequences
- Add 3 GridMapper-dependent submissions for dual-setup support"
```

---

## Phase 3: Run Benchmarks

Run all 13 submissions on both datasets. Each dataset runs independently. Results are written to separate files (results/*-2016.* and results/*-master2017.*).

### Submission classification

**Token-sequence models (6)** — no changes needed:
- lstm-baseline
- bottom-top-lstm
- transformer-encoder
- 2dcnn-baseline
- multichannel-2dcnn
- deep-mlp-baseline (uses bigram/meta features, not grid-dependent)

**Hold-vector models (4)** — no changes needed:
- fast-mlp
- focal-loss
- class-balanced-loss
- perceptron-baseline
- ordinal-regression

**GridMapper-dependent (3)** — need Phase 2 fix:
- tree-baseline
- ridge-baseline
- gradient-boost-baseline

All 13 need the Phase 1 GRADE_ORDER fix (which they already have).

### Gate 3: All benchmarks complete with results files
- All 13 submissions have 2016 results
- All 13 submissions have master2017 results
- Results files exist in results/ with proper naming convention
- Each run completes 5-fold CV

### Task 3.1: Baseline 2016 re-run (with 13 classes)

Re-run all 13 submissions on the 2016 dataset now that GRADE_ORDER has 13 classes. Results will differ from the existing leaderboard because the output layer changed.

**Decision:** Run sequentially (takes ~2 hrs) or parallel (faster on multi-core). Use background processes.

**Run command pattern:**
```bash
cd ~/repos/Moonboard-Analysis
uv run python src/moonboard_analysis/scripts/benchmark.py \
    --submission-dir submissions/<name> \
    --data-path Raw/moonboard_problems_setup_2016.json \
    --output-json results/<name>-2016.json \
    --output-markdown results/<name>-2016.md
```

**Estimated time per submission:** ~5-10 min (5-fold CV, varies by model complexity)
**Total:** ~13 × 8 min ≈ 2 hrs for all 13 on one dataset
**Both datasets:** ~4 hrs total

**Parallel execution plan:**
- Run 3-4 submissions in parallel (CPU-bound, but some use PyTorch)
- Monitor progress via `process(action='poll')` every ~5 min
- Total wall time: ~1 hr per dataset (4 parallel workers)

### Task 3.2: Masters 2017 run (same submissions)

Same pattern, different data path:
```bash
uv run python src/moonboard_analysis/scripts/benchmark.py \
    --submission-dir submissions/<name> \
    --data-path Raw/moonboard_problems_setup_master2017.json \
    --output-json results/<name>-master2017.json \
    --output-markdown results/<name>-master2017.md
```

### Task 3.3: Monitor and verify benchmarks

**Progress check:**
```bash
# Check which results files exist
ls results/*-2016.json results/*-master2017.json 2>/dev/null
```

**Verification:** Each results JSON should have valid `fold_results` and `mean_scores`:
```bash
for f in results/*-2016.json; do
    name=$(basename "$f")
    python3 -c "import json; d=json.load(open('$f')); print(f'$name: {len(d[\"fold_results\"])} folds, {len(d[\"mean_scores\"])} metrics')"
done
```

### Task 3.4: Commit benchmark results

```bash
git add results/*-2016.json results/*-master2017.json
git add results/*-2016.md results/*-master2017.md
git commit -m "feat: benchmark results for 2016 and Masters 2017

- All 13 submissions re-run on 2016 with 13-class GRADE_ORDER
- All 13 submissions run on Masters 2017
- Each: 5-fold CV, 4 metrics (exact, ±1, ±2, macro-F1)"
```

Note: results files are NOT gitignored (they're committed as part of the repo for GitHub Pages deployment).

---

## Phase 4: Leaderboard Merge

Update the leaderboard schema and regenerate all embedded tables.

### Gate 4: Two-column leaderboard deployed
- `leaderboard.json` contains both datasets with entries sorted per dataset
- All 5 target files (README.md, EXPERIMENTS.md, docs/index.md, results/full_data_summary.md) updated
- `uv run python generate_leaderboard.py --check` exits 0

### Task 4.1: Update leaderboard schema

**Objective:** Change `leaderboard.json` from single-dataset to multi-dataset format.

**Files:**
- Modify: `results/leaderboard.json`
- Modify: `src/moonboard_analysis/scripts/generate_leaderboard.py`

**New schema:**
```json
{
  "version": "2.0",
  "datasets": {
    "2016": {
      "num_routes": 25738,
      "cv_folds": 5,
      "description": "2016 hold setup (25,738 unique routes)",
      "entries": [...]
    },
    "master2017": {
      "num_routes": 19363,
      "cv_folds": 5,
      "description": "Masters 2017 hold setup (19,363 unique routes after adding 6B)",
      "entries": [...]
    }
  }
}
```

**Step 1: Write failing tests for new schema**

Add to `tests/test_generate_leaderboard.py`:

```python
def test_leaderboard_has_two_datasets() -> None:
    """Leaderboard should contain both 2016 and master2017 datasets."""
    leaderboard = load_leaderboard("results/leaderboard.json")
    assert "datasets" in leaderboard
    assert "2016" in leaderboard["datasets"]
    assert "master2017" in leaderboard["datasets"]
```

**Step 2: Implement schema change**

Write the updated `leaderboard.json` via a script that aggregates the per-submission results files into the new schema.

Create `scripts/aggregate_leaderboard.py`:

```python
"""Aggregate per-submission benchmark results into leaderboard.json."""
import json
from pathlib import Path

RESULTS_DIR = Path("results")
SUBMISSIONS = [
    "deep-mlp-baseline", "focal-loss", "fast-mlp", "class-balanced-loss",
    "perceptron-baseline", "lstm-baseline", "bottom-top-lstm",
    "transformer-encoder", "2dcnn-baseline", "multichannel-2dcnn",
    "tree-baseline", "ridge-baseline", "gradient-boost-baseline", "ordinal-regression",
]


def load_results(submission: str, dataset: str) -> dict:
    path = RESULTS_DIR / f"{submission}-{dataset}.json"
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    return data["mean_scores"]


# Build entries per dataset
datasets = {}
for dataset in ("2016", "master2017"):
    entries = []
    for sub in SUBMISSIONS:
        scores = load_results(sub, dataset)
        if scores is None:
            continue
        entries.append({
            "submission": sub,
            "exact_accuracy": {"mean": scores["exact_accuracy"], "std": ...},
            ...
        })
    datasets[dataset] = {
        "num_routes": ...,  # Count from data
        "cv_folds": 5,
        "entries": sorted(entries, key=lambda e: e["exact_accuracy"]["mean"], reverse=True),
    }
```

**Step 3: Build the aggregation script**

Full implementation with TDD: write failing test for the aggregator, then implement.

### Task 4.2: Update generate_leaderboard.py for dual-dataset

**Objective:** Support generating tables that compare both datasets side by side.

The `generate_leaderboard.py` script currently reads `leaderboard.json` and generates markdown tables. Update it to handle the `"datasets"` key.

**Step 1: Write failing tests**

```python
def test_generate_two_column_table() -> None:
    """Should generate a table with both 2016 and 2017 metrics."""
    entries_2016 = [{"submission": "a", "exact_accuracy": {"mean": 0.5, "std": 0.1}, ...}]
    entries_2017 = [{"submission": "a", "exact_accuracy": {"mean": 0.4, "std": 0.2}, ...}]
    table = generate_comparison_table(entries_2016, entries_2017)
    assert "2016" in table
    assert "2017" in table
    assert "| a |" in table
```

**Step 2: Create comparison table format**

New table format:
```markdown
| Model | 2016 Exact (%) | 2017 Exact (%) | 2016 ±1 (%) | 2017 ±1 (%) |
|---|---|---|---|---|
```

Or more practically, two tables per file — one for each dataset — with an intro heading.

**Step 3: Update the `replace_between_markers` pipeline**

The current pipeline replaces between `<!-- LEADERBOARD-START -->` and `<!-- LEADERBOARD-END -->` markers. Keep this structure but extend the content to include both datasets' tables.

### Task 4.3: Regenerate all embedded tables

```bash
uv run python src/moonboard_analysis/scripts/generate_leaderboard.py --write
```

Verify:

```bash
uv run python src/moonboard_analysis/scripts/generate_leaderboard.py --check
```

Expected: exit code 0 (all files up to date)

### Task 4.4: Verify full regression suite

```bash
uv run ruff check src tests
uv run mypy src
uv run pytest -x -q --tb=short
```

All must pass.

### Task 4.5: Commit phase 4

```bash
git add -A && git commit -m "feat: dual-dataset leaderboard with 2016 and Masters 2017

- Two-column leaderboard.json with both datasets
- Comparison tables in README, EXPERIMENTS, docs
- Leaderboard aggregation script
- All 13 submissions benchmarked on both hold setups"
```

---

## Phase 5: PR + Gemini Review Cycle

### Task 5.1: Push branch and open PR

```bash
cd ~/repos/Moonboard-Analysis

# Create feature branch (if not already on one)
git checkout -b feat/36-32-dual-dataset-benchmark

# Push
git push -u origin feat/36-32-dual-dataset-benchmark

# Create PR
gh pr create \
  --title "feat: dual-dataset benchmark — 2016 + Masters 2017" \
  --body-file /tmp/pr-body.md \
  --label enhancement
```

### Task 5.2: Wait for CI

```bash
gh pr checks --watch
```

Expected: all checks green (lint, typecheck, test).

### Task 5.3: Request Gemini review

```bash
gh pr comment <N> --body "/gemini review"
```

### Task 5.4: Iterative review cycle

**Loop until no findings:**

1. Wait 30-60s for Gemini to post review
2. Read review body:
   ```bash
   gh pr view <N> --json reviews --jq '.reviews[-1].body'
   ```
3. For each finding:
   - Apply fix (patch/write_file)
   - Re-run the check Gemini cited (ruff, pytest, etc.)
   - Commit and push
4. After all fixes:
   ```bash
   git push
   gh pr comment <N> --body "/gemini review"
   ```

**Exit condition:** Gemini posts a review with no findings or "Looks good" / "No issues found".

### Task 5.5: Verify all threads resolved

```bash
# Check for unresolved threads
gh pr view <N> --json reviewThreads --jq '.reviewThreads[] | select(.isResolved == false) | .id'
```

If any exist, reply to each thread and resolve via GraphQL.

### Task 5.6: Final regression + merge

```bash
# Final checks
uv run ruff check src tests
uv run mypy src
uv run pytest -x -q --tb=short

# Merge
gh pr merge <N> --squash --delete-branch
```

---

## Execution Order

Phase 1 → Phase 2 → Phase 3 → Phase 4 → Phase 5 (PR + review)

Within Phase 3, run 2016 benchmarks first, then 2017. Within each dataset, run submissions in parallel (3-4 at a time). GridMapper-dependent submissions can run on 2016 immediately (default setup), but need Phase 2 complete before running on 2017.
