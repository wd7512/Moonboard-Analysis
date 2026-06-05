"""Grid mapping utilities for Moonboard hold representations.

Provides bidirectional conversion between compressed feature vectors
and 18x11 Moonboard grid representations (3 layers: start/middle/end).

Supports multiple hold setups:
- "2016" (default): 164-dim vectors (58 null holds compressed out)
- "master2017": 242-dim vectors (no null holds — all positions used)
"""

from __future__ import annotations

import numpy as np


def detect_grid_setup(sequences: list[list[str]]) -> str:
    """Detect Moonboard hold setup from tokenised route sequences.

    Masters 2017 uses row-1 holds (A1-K1) which are always null
    on the 2016 board. If any row-1 hold appears, it's master2017.

    Args:
        sequences: Tokenised route sequences from preprocess_lstm_data().

    Returns:
        "master2017" if any row-1 hold is found, otherwise "2016".
    """
    for seq in sequences:
        for token in seq:
            if len(token) >= 2 and token[0].isalpha() and token[1:].isdigit():
                if int(token[1:]) == 1:
                    return "master2017"
    return "2016"


class GridMapper:
    """Maps between condensed vectors and 3x18x11 Moonboard grid.

    The Moonboard has 18 rows x 11 columns (A-K). Routes are represented
    as 3 binary layers: start holds, middle holds, and end holds (3x18x11).

    On the 2016 board, 58 hold positions never appear in any route, allowing
    a compressed 164-dim representation (242 - 78 = 164 after accounting for
    multi-layer mapping). On the Masters 2017 board, all 198 positions are
    used, so the full 242-dim grid is the canonical representation.

    Args:
        setup: Hold setup identifier ("2016" or "master2017").

    Raises:
        ValueError: If setup is not recognised.
    """

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

    def __init__(self, setup: str = "2016") -> None:
        if setup not in self._NULL_HOLDS:
            raise ValueError(
                f"Unknown setup: {setup!r}. Choose from: {list(self._NULL_HOLDS.keys())}"
            )
        self.setup = setup
        self.NULL_HOLDS = self._NULL_HOLDS[setup]
        if setup not in self._insert_indices:
            self._insert_indices[setup] = self._compute_insert_indices()
        self._insert_set = set(self._insert_indices[setup])

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

    def _compute_insert_indices(self) -> list[int]:
        """Compute flattened indices of null holds in the condensed 22x11 array.

        Returns:
            Sorted list of indices in the 242-element flattened condensed array
            that correspond to null hold positions.
        """
        null_positions: set[tuple[int, int]] = set()
        for hold in self.NULL_HOLDS:
            row, col = self._convert_key(hold)
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
        """Convert compressed vector to 3x18x11 Moonboard grid.

        For 2016: inserts zeros at null hold positions, reshapes to 22x11,
        then expands to the full 3-layer grid via _uncondense.

        For 2017: no null-holds compression. The vector is the flattened
        3x18x11 grid directly (594-dim).

        Args:
            vec: 1D array — (164,) for 2016 or (594,) for 2017.

        Returns:
            3D array of shape (3, 18, 11) with start/middle/end layers.

        Raises:
            ValueError: If input shape doesn't match the setup.
        """
        insert_indices = self._insert_indices[self.setup]

        if self.setup == "master2017":
            if vec.ndim != 1 or vec.shape[0] != 3 * 18 * 11:
                msg = f"Expected 1D array with {3*18*11} elements, got shape {vec.shape}"
                raise ValueError(msg)
            return vec.reshape((3, 18, 11))

        # 2016: expand compressed vector by inserting zeros at null positions
        expected_dim = 242 - len(insert_indices)
        if vec.ndim != 1 or vec.shape[0] != expected_dim:
            msg = f"Expected 1D array with {expected_dim} elements, got shape {vec.shape}"
            raise ValueError(msg)

        full = np.zeros(22 * 11, dtype=vec.dtype)
        vec_idx = 0
        for i in range(22 * 11):
            if i in self._insert_set:
                full[i] = 0
            else:
                full[i] = vec[vec_idx]
                vec_idx += 1

        condensed = full.reshape((22, 11))
        grid = self._uncondense(condensed)
        return grid

    def grid_to_vector(self, grid: np.ndarray) -> np.ndarray:
        """Convert 3x18x11 Moonboard grid to vector representation.

        For 2016: condenses the grid to 22x11, flattens, then removes
        null hold positions → 164-dim vector.

        For 2017: returns the flattened full grid → 594-dim vector.

        Args:
            grid: 3D array of shape (3, 18, 11) with start/middle/end layers.

        Returns:
            1D array — (164,) for 2016 or (594,) for 2017.

        Raises:
            ValueError: If input shape is not (3, 18, 11).
        """
        if grid.shape != (3, 18, 11):
            msg = f"Expected shape (3, 18, 11), got {grid.shape}"
            raise ValueError(msg)

        if self.setup == "master2017":
            return grid.flatten().copy()

        condensed = self._condense(grid)
        flat = condensed.flatten()

        insert_indices = self._insert_indices[self.setup]
        if not insert_indices:
            return flat

        vec = np.delete(flat, list(insert_indices))
        return vec
