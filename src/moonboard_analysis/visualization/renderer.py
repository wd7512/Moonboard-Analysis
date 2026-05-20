"""Moonboard grid renderer for matplotlib visualization."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from moonboard_analysis.data.grid_mapping import GridMapper


class GridRenderer:
    """Renders Moonboard hold grids as matplotlib figures.

    Depends on GridMapper for coordinate conversion. Follows single
    responsibility: takes grid data, returns figures. No model or data logic.

    Attributes:
        mapper: GridMapper instance for hold coordinate conversion.
    """

    _ACTIVE_COLOR = "#00d4ff"
    _INACTIVE_COLOR = "#1a1a2e"
    _DIFF_COLOR = "#ff4444"
    _BOTH_ACTIVE_COLOR = "#00ff88"
    _BACKGROUND_COLOR = "#0d0d1a"
    _GRID_COLOR = "#2a2a3e"
    _LABEL_COLOR = "#888899"

    def __init__(self, mapper: GridMapper) -> None:
        """Initialize with a GridMapper for coordinate conversion.

        Args:
            mapper: GridMapper instance.
        """
        self.mapper = mapper

    def render_single(
        self,
        grid: np.ndarray,
        title: str = "Moonboard Route",
    ) -> Figure:
        """Render a single Moonboard grid.

        Args:
            grid: 3x18x11 array with start/middle/end layers.
            title: Figure title.

        Returns:
            Matplotlib Figure with the rendered grid.
        """
        combined = np.any(grid > 0, axis=0).astype(float)
        return self._render_grid(combined, title=title)

    def render_comparison(
        self,
        original: np.ndarray,
        reconstructed: np.ndarray,
        threshold: float = 0.5,
        title: str = "Original vs Reconstructed",
    ) -> Figure:
        """Render side-by-side comparison with diff.

        Args:
            original: 3x18x11 array or 18x11 array of original holds.
            reconstructed: 3x18x11 array or 18x11 array of reconstructed holds.
            threshold: Sigmoid cutoff for binarization.
            title: Figure title.

        Returns:
            Matplotlib Figure with three panels: original, reconstructed, diff.
        """
        orig_grid = self._to_binary(original, threshold)
        recon_grid = self._to_binary(reconstructed, threshold)

        fig, axes = plt.subplots(1, 3, figsize=(18, 10))
        fig.patch.set_facecolor(self._BACKGROUND_COLOR)
        fig.suptitle(title, color="white", fontsize=16, fontweight="bold")

        self._draw_hold_grid(axes[0], orig_grid, "Original")
        self._draw_hold_grid(axes[1], recon_grid, "Reconstructed")
        self._draw_diff_grid(axes[2], orig_grid, recon_grid, "Difference")

        plt.tight_layout()
        return fig

    def _to_binary(self, data: np.ndarray, threshold: float) -> np.ndarray:
        """Convert grid or vector data to binary 18x11 array.

        Args:
            data: 3x18x11 grid, 18x11 array, or 164-dim vector.
            threshold: Cutoff for binarization.

        Returns:
            18x11 binary array.
        """
        if data.ndim == 1 and data.shape[0] == 164:
            grid = self.mapper.vector_to_grid(data)
        elif data.ndim == 3:
            grid = data
        elif data.ndim == 2:
            grid = data
        else:
            msg = f"Unsupported data shape: {data.shape}"
            raise ValueError(msg)

        if grid.ndim == 3:
            combined = np.any(grid > threshold, axis=0)
        else:
            combined = grid > threshold
        return combined.astype(float)

    def _render_grid(self, holds: np.ndarray, title: str) -> Figure:
        """Render a single 18x11 hold grid as a figure.

        Args:
            holds: 18x11 binary array.
            title: Figure title.

        Returns:
            Matplotlib Figure.
        """
        fig, ax = plt.subplots(figsize=(6, 10))
        fig.patch.set_facecolor(self._BACKGROUND_COLOR)
        ax.set_facecolor(self._BACKGROUND_COLOR)
        ax.set_title(title, color="white", fontsize=14, fontweight="bold")

        self._draw_hold_grid(ax, holds)
        plt.tight_layout()
        return fig

    def _draw_hold_grid(
        self,
        ax: plt.Axes,
        holds: np.ndarray,
        title: str | None = None,
    ) -> None:
        """Draw hold circles on an axis.

        Args:
            ax: Matplotlib axis to draw on.
            holds: 18x11 binary array.
            title: Optional panel title.
        """
        ax.set_facecolor(self._BACKGROUND_COLOR)
        ax.set_aspect("equal")
        ax.set_xlim(-0.7, 10.7)
        ax.set_ylim(-0.7, 17.7)

        if title:
            ax.set_title(title, color="white", fontsize=12, fontweight="bold")

        for row in range(18):
            for col in range(11):
                is_active = holds[row, col] > 0
                color = self._ACTIVE_COLOR if is_active else self._INACTIVE_COLOR
                edge_color = self._ACTIVE_COLOR if is_active else self._GRID_COLOR
                linewidth = 2 if is_active else 0.5

                circle = plt.Circle(
                    (col, 17 - row),
                    radius=0.35,
                    facecolor=color,
                    edgecolor=edge_color,
                    linewidth=linewidth,
                )
                ax.add_patch(circle)

        self._add_labels(ax)
        ax.axis("off")

    def _draw_diff_grid(
        self,
        ax: plt.Axes,
        original: np.ndarray,
        reconstructed: np.ndarray,
        title: str | None = None,
    ) -> None:
        """Draw difference grid highlighting mismatches.

        Green = both active, blue = original only, red = reconstructed only,
        dim = both inactive.

        Args:
            ax: Matplotlib axis to draw on.
            original: 18x11 binary array.
            reconstructed: 18x11 binary array.
            title: Optional panel title.
        """
        ax.set_facecolor(self._BACKGROUND_COLOR)
        ax.set_aspect("equal")
        ax.set_xlim(-0.7, 10.7)
        ax.set_ylim(-0.7, 17.7)

        if title:
            ax.set_title(title, color="white", fontsize=12, fontweight="bold")

        for row in range(18):
            for col in range(11):
                is_orig = original[row, col] > 0
                is_recon = reconstructed[row, col] > 0

                if is_orig and is_recon:
                    color = self._BOTH_ACTIVE_COLOR
                    edge_color = self._BOTH_ACTIVE_COLOR
                    linewidth = 2.0
                elif is_orig and not is_recon:
                    color = "#4488ff"
                    edge_color = "#4488ff"
                    linewidth = 2.0
                elif not is_orig and is_recon:
                    color = self._DIFF_COLOR
                    edge_color = self._DIFF_COLOR
                    linewidth = 2.0
                else:
                    color = self._INACTIVE_COLOR
                    edge_color = self._GRID_COLOR
                    linewidth = 0.5

                circle = plt.Circle(
                    (col, 17 - row),
                    radius=0.35,
                    facecolor=color,
                    edgecolor=edge_color,
                    linewidth=linewidth,
                )
                ax.add_patch(circle)

        self._add_labels(ax)
        ax.axis("off")

    @staticmethod
    def _add_labels(ax: plt.Axes) -> None:
        """Add column (A-K) and row (1-18) labels.

        Args:
            ax: Matplotlib axis to add labels to.
        """
        label_color = "#888899"
        font_size = 9

        for col in range(11):
            letter = chr(ord("A") + col)
            ax.text(
                col,
                17.5,
                letter,
                ha="center",
                va="center",
                color=label_color,
                fontsize=font_size,
            )

        for row in range(18):
            ax.text(
                -0.5,
                17 - row,
                str(row + 1),
                ha="center",
                va="center",
                color=label_color,
                fontsize=font_size,
            )
