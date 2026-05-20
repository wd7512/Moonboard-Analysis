"""Tests for GridRenderer visualization."""

import matplotlib.pyplot as plt
import numpy as np
import pytest

from moonboard_analysis.data.grid_mapping import GridMapper
from moonboard_analysis.visualization.renderer import GridRenderer


@pytest.fixture
def mapper() -> GridMapper:
    """Return a GridMapper instance."""
    return GridMapper()


@pytest.fixture
def renderer(mapper: GridMapper) -> GridRenderer:
    """Return a GridRenderer instance."""
    return GridRenderer(mapper)


@pytest.fixture
def sample_grid() -> np.ndarray:
    """Return a sample 3x18x11 binary grid with a few active holds."""
    grid = np.zeros((3, 18, 11), dtype=np.float32)
    grid[0, 15, 5] = 1
    grid[1, 10, 3] = 1
    grid[1, 8, 7] = 1
    grid[2, 0, 5] = 1
    return grid


@pytest.fixture
def sample_18x11() -> np.ndarray:
    """Return a sample 18x11 binary array."""
    arr = np.zeros((18, 11), dtype=np.float32)
    arr[15, 5] = 1
    arr[10, 3] = 1
    return arr


@pytest.fixture
def sample_164() -> np.ndarray:
    """Return a sample 164-dim binary vector."""
    rng = np.random.default_rng(42)
    return rng.integers(0, 2, size=164).astype(np.float32)


class TestRenderSingle:
    """Test render_single method."""

    def test_returns_figure(self, renderer: GridRenderer, sample_grid: np.ndarray) -> None:
        """Verify render_single returns a matplotlib Figure."""
        fig = renderer.render_single(sample_grid)
        assert isinstance(fig, plt.Figure)

    def test_figure_has_axes(self, renderer: GridRenderer, sample_grid: np.ndarray) -> None:
        """Verify the figure contains at least one axis."""
        fig = renderer.render_single(sample_grid)
        assert len(fig.get_axes()) >= 1

    def test_custom_title(self, renderer: GridRenderer, sample_grid: np.ndarray) -> None:
        """Verify custom title is set on the figure."""
        fig = renderer.render_single(sample_grid, title="My Route")
        ax = fig.get_axes()[0]
        assert ax.get_title() == "My Route"

    def test_all_zero_grid(self, renderer: GridRenderer) -> None:
        """Verify rendering works with an empty grid."""
        grid = np.zeros((3, 18, 11), dtype=np.float32)
        fig = renderer.render_single(grid)
        assert isinstance(fig, plt.Figure)

    def test_all_ones_grid(self, renderer: GridRenderer) -> None:
        """Verify rendering works with all holds active."""
        grid = np.ones((3, 18, 11), dtype=np.float32)
        fig = renderer.render_single(grid)
        assert isinstance(fig, plt.Figure)

    def test_closes_figure(self, renderer: GridRenderer, sample_grid: np.ndarray) -> None:
        """Verify figure can be closed without errors."""
        fig = renderer.render_single(sample_grid)
        plt.close(fig)


class TestRenderComparison:
    """Test render_comparison method."""

    def test_returns_figure(
        self, renderer: GridRenderer, sample_grid: np.ndarray
    ) -> None:
        """Verify render_comparison returns a matplotlib Figure."""
        fig = renderer.render_comparison(sample_grid, sample_grid)
        assert isinstance(fig, plt.Figure)

    def test_has_three_panels(
        self, renderer: GridRenderer, sample_grid: np.ndarray
    ) -> None:
        """Verify the figure has three subplots."""
        fig = renderer.render_comparison(sample_grid, sample_grid)
        assert len(fig.get_axes()) == 3

    def test_identical_inputs_no_diff(
        self, renderer: GridRenderer, sample_grid: np.ndarray
    ) -> None:
        """Verify identical original and reconstructed produces no errors."""
        fig = renderer.render_comparison(sample_grid, sample_grid)
        assert isinstance(fig, plt.Figure)

    def test_different_inputs(
        self, renderer: GridRenderer, sample_grid: np.ndarray
    ) -> None:
        """Verify different inputs render without errors."""
        modified = sample_grid.copy()
        modified[1, 5, 5] = 1
        fig = renderer.render_comparison(sample_grid, modified)
        assert isinstance(fig, plt.Figure)

    def test_custom_title(
        self, renderer: GridRenderer, sample_grid: np.ndarray
    ) -> None:
        """Verify custom title is set."""
        fig = renderer.render_comparison(
            sample_grid, sample_grid, title="Test Comparison"
        )
        assert fig._suptitle is not None

    def test_closes_figure(
        self, renderer: GridRenderer, sample_grid: np.ndarray
    ) -> None:
        """Verify figure can be closed without errors."""
        fig = renderer.render_comparison(sample_grid, sample_grid)
        plt.close(fig)


class TestToBinary:
    """Test _to_binary conversion method."""

    def test_3d_grid_input(
        self, renderer: GridRenderer, sample_grid: np.ndarray
    ) -> None:
        """Verify 3x18x11 grid converts to 18x11 binary."""
        result = renderer._to_binary(sample_grid, threshold=0.5)
        assert result.shape == (18, 11)

    def test_2d_array_input(
        self, renderer: GridRenderer, sample_18x11: np.ndarray
    ) -> None:
        """Verify 18x11 array passes through as binary."""
        result = renderer._to_binary(sample_18x11, threshold=0.5)
        assert result.shape == (18, 11)

    def test_164d_vector_input(
        self, renderer: GridRenderer, sample_164: np.ndarray
    ) -> None:
        """Verify 164-dim vector converts to 18x11 binary."""
        result = renderer._to_binary(sample_164, threshold=0.5)
        assert result.shape == (18, 11)

    def test_threshold_affects_output(
        self, renderer: GridRenderer
    ) -> None:
        """Verify different thresholds produce different results."""
        grid = np.full((18, 11), 0.3, dtype=np.float32)
        low_thresh = renderer._to_binary(grid, threshold=0.2)
        high_thresh = renderer._to_binary(grid, threshold=0.5)
        assert np.sum(low_thresh) > np.sum(high_thresh)

    def test_unsupported_shape_raises(
        self, renderer: GridRenderer
    ) -> None:
        """Verify unsupported shapes raise ValueError."""
        data = np.zeros((3, 5, 5, 2), dtype=np.float32)
        with pytest.raises(ValueError, match="Unsupported data shape"):
            renderer._to_binary(data, threshold=0.5)

    def test_sigmoid_values_binarized(
        self, renderer: GridRenderer
    ) -> None:
        """Verify sigmoid-like values are correctly binarized."""
        grid = np.zeros((3, 18, 11), dtype=np.float32)
        grid[1, 10, 5] = 0.7
        grid[1, 10, 6] = 0.3
        result = renderer._to_binary(grid, threshold=0.5)
        assert result[10, 5] == 1
        assert result[10, 6] == 0
