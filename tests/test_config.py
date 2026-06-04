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
