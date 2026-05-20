# Autoencoder Latent Space Visualizer — Plan

## Overview

Interactive Gradio web app for exploring the autoencoder's learned latent space.
Users select a route, manipulate its 8-dimensional encoded representation via
sliders, and see the reconstructed route rendered on a Moonboard grid in real time.

## Tech Stack

| Layer | Choice | Rationale |
|-------|--------|-----------|
| Framework | Gradio >=4.0 | Purpose-built ML demos, native Python, slider components built in |
| Backend | PyTorch (existing) | Load `Autoencoder_Moonboard.pth` directly |
| Visualization | Matplotlib | Grid rendering with hold-level control |
| Data | NumPy (existing) | Load `2016TrainingData164.npy` for route selection |

## New Files

```
src/moonboard_analysis/
  data/
    grid_mapping.py          # GridMapper class — 164-dim ↔ 18x11 grid
  visualization/
    __init__.py
    renderer.py              # GridRenderer class — grid → matplotlib figure
```

```
scripts/
  launch_autoencoder_viz.py  # Gradio app entry point
```

## `GridMapper` (`src/moonboard_analysis/data/grid_mapping.py`)

Single responsibility: bidirectional mapping between flattened 164-dim vectors
and the 18x11 Moonboard hold grid.

```python
class GridMapper:
    NULL_HOLDS: list[str]          # 78 holds that never appear in routes
    _insert_indices: list[int]     # positions in 242-dim where null holds go

    def __init__(self)
    def vector_to_grid(vec: np.ndarray) -> np.ndarray   # 164 → 18x11
    def grid_to_vector(grid: np.ndarray) -> np.ndarray  # 18x11 → 164
```

Extracts and cleans up logic from `archive/Legacy/Reverse.py`:
- No matplotlib dependency
- No global state
- Null hold positions defined once as a class constant
- `convert_key()` logic encapsulated internally

## `GridRenderer` (`src/moonboard_analysis/visualization/renderer.py`)

Single responsibility: grid data → matplotlib figures. No model or data logic.

```python
class GridRenderer:
    def __init__(self, mapper: GridMapper)
    def render_single(grid: np.ndarray, title: str) -> Figure
    def render_comparison(
        original: np.ndarray,
        reconstructed: np.ndarray,
        threshold: float = 0.5
    ) -> Figure
```

Rendering details:
- Dark background (Moonboard aesthetic)
- Holds as circles: active = bright, inactive = dim
- Column labels A-K, row labels 1-18
- Comparison view: side-by-side with diff highlighting

## Gradio App (`scripts/launch_autoencoder_viz.py`)

Thin orchestration layer. Dependencies injected, no business logic.

Startup:
1. Load `Autoencoder_Moonboard.pth` checkpoint
2. Load `2016TrainingData164.npy`
3. Compute latent space statistics (p5/p95 per dimension for slider ranges)
4. Initialize `GridMapper` and `GridRenderer`

UI components:
- **Route selector**: Dropdown showing "Route #N — Grade: X" (grade decoded from col 0)
- **8 sliders**: One per latent dimension, range = p5 to p95 of encoded training data
- **Reset button**: Sets sliders to actual encoded values for selected route
- **Randomize button**: Sets sliders to random values within range
- **Threshold slider**: 0.1-0.9, controls sigmoid→binary cutoff for visualization
- **Output**: Side-by-side Moonboard grids (original vs reconstructed) + MSE display

## Modified Files

**`pyproject.toml`** — add to dependencies:
```toml
gradio>=4.0,<5",
```

## Design Decisions

| Decision | Rationale |
|----------|-----------|
| Single grid, one color | 164-dim encoding collapsed start/middle/end — role info is lost |
| Threshold-based binarization | Decoder outputs sigmoid (0-1); 0.5 default, user-adjustable |
| Slider ranges from data p5/p95 | Covers meaningful latent space without extreme outliers |
| No route names in selector | .npy only stores grade index, not metadata |
| Legacy `Reverse.py` not imported | Extracted into clean `GridMapper`; archive stays archival |

## Execution Order

1. Add `gradio` to `pyproject.toml`
2. Create `GridMapper` with tests
3. Create `GridRenderer` with tests
4. Build `launch_autoencoder_viz.py`
5. Manual testing: verify grid rendering matches known routes
6. Run: `uv run python scripts/launch_autoencoder_viz.py`
