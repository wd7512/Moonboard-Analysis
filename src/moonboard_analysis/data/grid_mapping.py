"""Grid mapping utilities for Moonboard hold representations.

Provides bidirectional conversion between 164-dimensional feature vectors
and 18x11 Moonboard grid representations (3 layers: start/middle/end).
"""

from __future__ import annotations

import numpy as np


class GridMapper:
    """Maps between 164-dim vectors and 18x11 Moonboard grid (3 layers).

    The Moonboard has 18 rows x 11 columns (A-K). Routes are represented
    as 3 binary layers: start holds, middle holds, and end holds (3x18x11).

    An optimized representation condenses this to 22x11 by extracting only
    the rows that contain route-relevant holds, then removes 78 positions
    that never appear in any route, yielding a 164-dim vector
    (22*11 - 78 = 164).

    Attributes:
        NULL_HOLDS: 78 hold descriptions that never appear in routes.
        _insert_indices: Sorted indices in the flattened 22x11 array
            where null holds are located.
    """

    NULL_HOLDS: list[str] = [
        "F18",
        "J18",
        "A17",
        "B17",
        "C17",
        "E17",
        "F17",
        "H17",
        "I17",
        "J17",
        "K17",
        "J15",
        "K15",
        "B14",
        "A8",
        "A7",
        "A6",
        "H6",
        "B5",
        "E5",
        "G5",
        "A4",
        "C4",
        "D4",
        "E4",
        "F4",
        "H4",
        "J4",
        "K4",
        "A3",
        "C3",
        "E3",
        "F3",
        "G3",
        "H3",
        "I3",
        "J3",
        "K3",
        "A2",
        "B2",
        "C2",
        "D2",
        "E2",
        "F2",
        "H2",
        "I2",
        "K2",
        "A1",
        "B1",
        "C1",
        "D1",
        "E1",
        "F1",
        "G1",
        "H1",
        "I1",
        "J1",
        "K1",
    ]

    _insert_indices: list[int] = []

    def __init__(self) -> None:
        """Compute _insert_indices from NULL_HOLDS on first instantiation."""
        if not GridMapper._insert_indices:
            GridMapper._insert_indices = self._compute_insert_indices()

    @staticmethod
    def _convert_key(key: str) -> tuple[int, int]:
        """Convert hold description (e.g. 'B10') to (row, col) in 18x11 grid.

        Args:
            key: Hold description with letter (A-K) and number (1-18).

        Returns:
            (row, col) tuple where row 0 is top, col 0 is column A.
        """
        letter = key[0]
        number = key[1:]
        col = ord(letter) - 65
        row = 18 - int(number)
        return row, col

    @classmethod
    def _compute_insert_indices(cls) -> list[int]:
        """Compute flattened indices of null holds in the condensed 22x11 array.

        Returns:
            Sorted list of indices in the 242-element flattened condensed array
            that correspond to null hold positions.
        """
        null_positions: set[tuple[int, int]] = set()
        for hold in cls.NULL_HOLDS:
            row, col = cls._convert_key(hold)
            null_positions.add((row, col))

        condensed = np.zeros((22, 11), dtype=float)

        for row, col in null_positions:
            if 12 <= row <= 16:
                condensed_row = row - 12
                condensed[condensed_row, col] = 1
            if 1 <= row <= 16:
                condensed_row = row - 1 + 5
                condensed[condensed_row, col] = 1
            if row == 0:
                condensed[21, col] = 1

        flat = condensed.flatten()
        indices = [int(i) for i, val in enumerate(flat) if val == 1]
        return sorted(indices)

    @staticmethod
    def _condense(moves: np.ndarray) -> np.ndarray:
        """Condense 3x18x11 grid to 22x11 array.

        Args:
            moves: 3x18x11 array with start/middle/end hold layers.

        Returns:
            22x11 condensed array.
        """
        new_moves = np.zeros((22, 11), dtype=moves.dtype)
        new_moves[0:5, :] = moves[0, 12:17, :]
        new_moves[5:21, :] = moves[1, 1:17, :]
        new_moves[-1, :] = moves[2, 0, :]
        return new_moves

    @staticmethod
    def _uncondense(moves: np.ndarray) -> np.ndarray:
        """Expand 22x11 condensed array back to 3x18x11 grid.

        Args:
            moves: 22x11 condensed array.

        Returns:
            3x18x11 array with start/middle/end hold layers.
        """
        new_moves = np.zeros((3, 18, 11), dtype=moves.dtype)
        new_moves[0, 12:17, :] = moves[0:5, :]
        new_moves[1, 1:17, :] = moves[5:21, :]
        new_moves[2, 0, :] = moves[-1, :]
        return new_moves

    def vector_to_grid(self, vec: np.ndarray) -> np.ndarray:
        """Convert 164-dim vector to 3x18x11 Moonboard grid.

        Inserts zeros at null hold positions, reshapes to 22x11, then
        expands to the full 3-layer grid.

        Args:
            vec: 1D array of shape (164,).

        Returns:
            3D array of shape (3, 18, 11) with start/middle/end layers.

        Raises:
            ValueError: If input is not 1D with 164 elements.
        """
        if vec.ndim != 1 or vec.shape[0] != 164:
            msg = f"Expected 1D array with 164 elements, got shape {vec.shape}"
            raise ValueError(msg)

        full = np.zeros(22 * 11, dtype=vec.dtype)
        vec_idx = 0
        insert_set = set(self._insert_indices)

        for i in range(22 * 11):
            if i in insert_set:
                full[i] = 0
            else:
                full[i] = vec[vec_idx]
                vec_idx += 1

        condensed = full.reshape((22, 11))
        grid = self._uncondense(condensed)
        return grid

    def grid_to_vector(self, grid: np.ndarray) -> np.ndarray:
        """Convert 3x18x11 Moonboard grid to 164-dim vector.

        Condenses the grid to 22x11, flattens, then removes null hold
        positions.

        Args:
            grid: 3D array of shape (3, 18, 11) with start/middle/end layers.

        Returns:
            1D array of shape (164,).

        Raises:
            ValueError: If input shape is not (3, 18, 11).
        """
        if grid.shape != (3, 18, 11):
            msg = f"Expected shape (3, 18, 11), got {grid.shape}"
            raise ValueError(msg)

        condensed = self._condense(grid)
        flat = condensed.flatten()
        vec = np.delete(flat, self._insert_indices)
        return vec
