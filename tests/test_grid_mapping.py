"""Tests for GridMapper bidirectional conversion."""

import numpy as np
import pytest

from moonboard_analysis.data.grid_mapping import GridMapper, detect_grid_setup


@pytest.fixture
def mapper() -> GridMapper:
    """Return a GridMapper instance."""
    return GridMapper()


class TestGridMapperConstruction:
    """Test GridMapper initialization."""

    def test_insert_indices_not_empty(self, mapper: GridMapper) -> None:
        """Verify null hold indices are computed on construction."""
        assert len(mapper._insert_indices[mapper.setup]) > 0

    def test_insert_indices_count(self, mapper: GridMapper) -> None:
        """Verify exactly 78 null hold positions are identified."""
        assert len(mapper._insert_indices[mapper.setup]) == 78

    def test_insert_indices_sorted(self, mapper: GridMapper) -> None:
        """Verify insert indices are in ascending order."""
        indices = mapper._insert_indices[mapper.setup]
        assert indices == sorted(indices)

    def test_insert_indices_within_bounds(self, mapper: GridMapper) -> None:
        """Verify all indices are within the 22x11=242 flattened range."""
        for idx in mapper._insert_indices[mapper.setup]:
            assert 0 <= idx < 242


class TestConvertKey:
    """Test hold description to grid coordinate conversion."""

    def test_a1_bottom_left(self) -> None:
        """A1 should map to bottom row (row 17), left column (col 0)."""
        row, col = GridMapper._convert_key("A1")
        assert row == 17
        assert col == 0

    def test_k18_top_right(self) -> None:
        """K18 should map to top row (row 0), right column (col 10)."""
        row, col = GridMapper._convert_key("K18")
        assert row == 0
        assert col == 10

    def test_a18_top_left(self) -> None:
        """A18 should map to top row (row 0), left column (col 0)."""
        row, col = GridMapper._convert_key("A18")
        assert row == 0
        assert col == 0

    def test_k1_bottom_right(self) -> None:
        """K1 should map to bottom row (row 17), right column (col 10)."""
        row, col = GridMapper._convert_key("K1")
        assert row == 17
        assert col == 10

    def test_e10_middle(self) -> None:
        """E10 should map to row 8, col 4."""
        row, col = GridMapper._convert_key("E10")
        assert row == 8
        assert col == 4


class TestVectorToGrid:
    """Test 164-dim vector to 3x18x11 grid conversion."""

    def test_output_shape(self, mapper: GridMapper) -> None:
        """Verify output is 3x18x11."""
        vec = np.zeros(164, dtype=np.float32)
        grid = mapper.vector_to_grid(vec)
        assert grid.shape == (3, 18, 11)

    def test_zeros_produce_zeros(self, mapper: GridMapper) -> None:
        """Zero vector should produce all-zero grid."""
        vec = np.zeros(164, dtype=np.float32)
        grid = mapper.vector_to_grid(vec)
        assert np.allclose(grid, 0)

    def test_ones_produce_nonzero(self, mapper: GridMapper) -> None:
        """All-ones vector should produce nonzero holds in the grid."""
        vec = np.ones(164, dtype=np.float32)
        grid = mapper.vector_to_grid(vec)
        assert np.sum(grid) > 0

    def test_null_holds_always_zero(self, mapper: GridMapper) -> None:
        """Null hold positions should always be zero regardless of input."""
        vec = np.ones(164, dtype=np.float32)
        grid = mapper.vector_to_grid(vec)
        for hold in mapper.NULL_HOLDS:
            row, col = GridMapper._convert_key(hold)
            for layer in range(3):
                assert grid[layer, row, col] == 0, f"Null hold {hold} is nonzero"

    def test_preserves_binary_values(self, mapper: GridMapper) -> None:
        """Binary input should produce binary output (0 or 1)."""
        rng = np.random.default_rng(42)
        vec = rng.integers(0, 2, size=164).astype(np.float32)
        grid = mapper.vector_to_grid(vec)
        unique_vals = np.unique(grid)
        for val in unique_vals:
            assert val in (0.0, 1.0), f"Non-binary value found: {val}"


class TestGridToVector:
    """Test 3x18x11 grid to 164-dim vector conversion."""

    def test_output_shape(self, mapper: GridMapper) -> None:
        """Verify output is 164-dim."""
        grid = np.zeros((3, 18, 11), dtype=np.float32)
        vec = mapper.grid_to_vector(grid)
        assert vec.shape == (164,)

    def test_zeros_produce_zeros(self, mapper: GridMapper) -> None:
        """Zero grid should produce zero vector."""
        grid = np.zeros((3, 18, 11), dtype=np.float32)
        vec = mapper.grid_to_vector(grid)
        assert np.allclose(vec, 0)


class TestRoundTrip:
    """Test vector→grid→vector and grid→vector→grid round-trips."""

    def test_vector_roundtrip(self, mapper: GridMapper) -> None:
        """vector_to_grid(grid_to_vector(x)) should equal x."""
        rng = np.random.default_rng(42)
        original = rng.random(164).astype(np.float32)
        grid = mapper.vector_to_grid(original)
        recovered = mapper.grid_to_vector(grid)
        np.testing.assert_array_almost_equal(original, recovered)

    def test_grid_roundtrip(self, mapper: GridMapper) -> None:
        """grid_to_vector(vector_to_grid(x)) should equal x for valid grids.

        Only positions in the condensed representation are preserved:
        - Start holds: rows 12-16
        - Middle holds: rows 1-16
        - End holds: row 0
        Null holds within these regions are always zeroed.
        """
        rng = np.random.default_rng(42)
        original = rng.random((3, 18, 11)).astype(np.float32)
        vec = mapper.grid_to_vector(original)
        recovered_grid = mapper.vector_to_grid(vec)

        for hold in mapper.NULL_HOLDS:
            row, col = GridMapper._convert_key(hold)
            for layer in range(3):
                assert recovered_grid[layer, row, col] == 0

        non_null_mask = np.zeros((3, 18, 11), dtype=bool)
        non_null_mask[0, 12:17, :] = True
        non_null_mask[1, 1:17, :] = True
        non_null_mask[2, 0, :] = True
        for hold in mapper.NULL_HOLDS:
            row, col = GridMapper._convert_key(hold)
            for layer in range(3):
                non_null_mask[layer, row, col] = False

        np.testing.assert_array_almost_equal(
            original[non_null_mask], recovered_grid[non_null_mask]
        )

    def test_binary_vector_roundtrip(self, mapper: GridMapper) -> None:
        """Round-trip should preserve exact binary values."""
        rng = np.random.default_rng(42)
        original = rng.integers(0, 2, size=164).astype(np.float32)
        grid = mapper.vector_to_grid(original)
        recovered = mapper.grid_to_vector(grid)
        np.testing.assert_array_equal(original, recovered)


class TestInputValidation:
    """Test input validation for conversion methods."""

    def test_vector_to_grid_wrong_ndim(self, mapper: GridMapper) -> None:
        """Should raise ValueError for non-1D input."""
        vec = np.zeros((164, 1), dtype=np.float32)
        with pytest.raises(ValueError, match="1D array"):
            mapper.vector_to_grid(vec)

    def test_vector_to_grid_wrong_size(self, mapper: GridMapper) -> None:
        """Should raise ValueError for wrong number of elements."""
        vec = np.zeros(100, dtype=np.float32)
        with pytest.raises(ValueError, match="164 elements"):
            mapper.vector_to_grid(vec)

    def test_grid_to_vector_wrong_shape(self, mapper: GridMapper) -> None:
        """Should raise ValueError for wrong grid shape."""
        grid = np.zeros((18, 11), dtype=np.float32)
        with pytest.raises(ValueError, match="3, 18, 11"):
            mapper.grid_to_vector(grid)


class TestGridMapperSetup:
    """Test GridMapper with different hold setups."""

    def test_default_setup_is_2016(self, mapper: GridMapper) -> None:
        """Default constructor should use 2016 setup."""
        assert mapper.setup == "2016"

    def test_2017_setup_vector_dim(self) -> None:
        """Master 2017 setup should produce 594-dim vectors (full grid)."""
        mapper = GridMapper(setup="master2017")
        grid = np.zeros((3, 18, 11))
        vec = mapper.grid_to_vector(grid)
        assert vec.shape == (594,)

    def test_2017_null_holds_empty(self) -> None:
        """Master 2017 has no null holds — all positions used."""
        mapper = GridMapper(setup="master2017")
        assert len(mapper.NULL_HOLDS) == 0

    def test_2017_vector_roundtrip(self) -> None:
        """Round-trip should preserve full 594-dim vector."""
        mapper = GridMapper(setup="master2017")
        rng = np.random.default_rng(42)
        original = rng.random(594).astype(np.float32)
        grid = mapper.vector_to_grid(original)
        recovered = mapper.grid_to_vector(grid)
        np.testing.assert_array_almost_equal(original, recovered)

    def test_2017_grid_roundtrip(self) -> None:
        """Round-trip should preserve full 3x18x11 grid for 2017."""
        mapper = GridMapper(setup="master2017")
        rng = np.random.default_rng(42)
        original = rng.random((3, 18, 11)).astype(np.float32)
        vec = mapper.grid_to_vector(original)
        recovered = mapper.vector_to_grid(vec)
        np.testing.assert_array_almost_equal(original, recovered)

    def test_unknown_setup_raises_error(self) -> None:
        """Unknown setup name should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown setup"):
            GridMapper(setup="nonexistent")

    def test_setups_independent(self) -> None:
        """2016 and 2017 mappers should not share state incorrectly."""
        m2016 = GridMapper(setup="2016")
        m2017 = GridMapper(setup="master2017")
        grid = np.zeros((3, 18, 11))
        vec2016 = m2016.grid_to_vector(grid)
        vec2017 = m2017.grid_to_vector(grid)
        assert len(vec2016) != len(vec2017)


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
