import pytest

from moonboard_analysis.scripts.generate_leaderboard import (
    check_files_up_to_date,
    generate_detailed_table,
    generate_simple_table,
    replace_between_markers,
)

MOCK_ENTRIES = [
    {
        "submission": "model-b",
        "model": "Model B",
        "exact_accuracy": {"mean": 0.5000, "std": 0.0100},
        "within_one_grade": {"mean": 0.7000, "std": 0.0150},
        "within_two_grades": {"mean": 0.9000, "std": 0.0050},
        "macro_f1": {"mean": 0.2000, "std": 0.0080},
        "training_time": "~5 min",
    },
    {
        "submission": "model-a",
        "model": "Model A",
        "exact_accuracy": {"mean": 0.4500, "std": 0.0120},
        "within_one_grade": {"mean": 0.6500, "std": 0.0100},
        "within_two_grades": {"mean": 0.8500, "std": 0.0080},
        "macro_f1": {"mean": 0.1800, "std": 0.0060},
        "training_time": "~10 min",
    },
]


class TestGenerateSimpleTable:
    def test_has_headers(self):
        table = generate_simple_table(MOCK_ENTRIES)
        assert "| Model | Exact (%) | Within +/-1 (%) | Within +/-2 (%) | Macro-F1 (%) |" in table
        assert "|---" in table

    def test_contains_models(self):
        table = generate_simple_table(MOCK_ENTRIES)
        assert "Model B" in table
        assert "Model A" in table

    def test_contains_plus_minus(self):
        table = generate_simple_table(MOCK_ENTRIES)
        assert "±" in table

    def test_sorts_by_exact_accuracy_descending(self):
        table = generate_simple_table(MOCK_ENTRIES)
        model_b_idx = table.index("Model B")
        model_a_idx = table.index("Model A")
        assert model_b_idx < model_a_idx, "higher exact_accuracy should appear first"

    def test_empty_entries(self):
        table = generate_simple_table([])
        assert "No entries" in table or table.strip() == ""

    def test_value_formatting(self):
        table = generate_simple_table(MOCK_ENTRIES)
        assert "50.00 ± 1.00" in table
        assert "90.00 ± 0.50" in table


class TestGenerateDetailedTable:
    def test_has_detailed_headers(self):
        table = generate_detailed_table(MOCK_ENTRIES)
        assert "| # | Submission" in table
        assert "Macro-F1 (%)" in table
        assert "Training Time" in table

    def test_has_submission_names(self):
        table = generate_detailed_table(MOCK_ENTRIES)
        assert "model-a" in table
        assert "model-b" in table

    def test_bolds_best_exact(self):
        table = generate_detailed_table(MOCK_ENTRIES)
        model_b_line = [x for x in table.split("\n") if "model-b" in x][0]
        assert "**50.00**" in model_b_line

    def test_sorts_by_exact_accuracy_descending(self):
        table = generate_detailed_table(MOCK_ENTRIES)
        model_b_idx = table.index("model-b")
        model_a_idx = table.index("model-a")
        assert model_b_idx < model_a_idx


class TestReplaceBetweenMarkers:
    def test_replaces_between_markers(self):
        content = "before\n<!-- LEADERBOARD-START -->\nold content\n<!-- LEADERBOARD-END -->\nafter"
        new_table = "| new | table |"
        result = replace_between_markers(
            content, "<!-- LEADERBOARD-START -->", "<!-- LEADERBOARD-END -->", new_table
        )
        assert "before" in result
        assert "after" in result
        assert "old content" not in result
        assert "| new | table |" in result

    def test_multiple_calls_different_markers(self):
        content = (
            "a\n<!-- START-A -->\nold-a\n<!-- END-A -->\n"
            "b\n<!-- START-B -->\nold-b\n<!-- END-B -->\nc"
        )
        result = content
        result = replace_between_markers(
            result, "<!-- START-A -->", "<!-- END-A -->", "new-a"
        )
        result = replace_between_markers(
            result, "<!-- START-B -->", "<!-- END-B -->", "new-b"
        )
        assert "new-a" in result
        assert "new-b" in result
        assert "old-a" not in result
        assert "old-b" not in result

    def test_raises_on_missing_start_marker(self):
        content = "no markers here"
        with pytest.raises(ValueError, match="not found"):
            replace_between_markers(
                content,
                "<!-- LEADERBOARD-START -->",
                "<!-- LEADERBOARD-END -->",
                "table",
            )

    def test_raises_on_missing_end_marker(self):
        content = "<!-- LEADERBOARD-START -->\nno end marker"
        with pytest.raises(ValueError, match="not found"):
            replace_between_markers(
                content,
                "<!-- LEADERBOARD-START -->",
                "<!-- LEADERBOARD-END -->",
                "table",
            )

    def test_preserves_surrounding_content(self):
        content = (
            "header\n<!-- LEADERBOARD-START -->\n"
            "stuff\n<!-- LEADERBOARD-END -->\nfooter"
        )
        result = replace_between_markers(
            content,
            "<!-- LEADERBOARD-START -->",
            "<!-- LEADERBOARD-END -->",
            "new stuff",
        )
        assert result == (
            "header\n<!-- LEADERBOARD-START -->\n\n"
            "new stuff\n\n<!-- LEADERBOARD-END -->\nfooter"
        )


class TestCheckDetectsOutdated:
    def test_detects_outdated_file(self, tmp_path):
        import json

        json_path = tmp_path / "leaderboard.json"
        data = {
            "version": "1.0",
            "dataset": "test",
            "num_routes": 0,
            "cv_folds": 0,
            "description": "",
            "entries": MOCK_ENTRIES,
        }
        json_path.write_text(json.dumps(data))
        md_path = tmp_path / "test.md"
        md_path.write_text(
            "<!-- LEADERBOARD-START -->\nstale\n<!-- LEADERBOARD-END -->\n"
        )
        target = {
            str(md_path): (
                "<!-- LEADERBOARD-START -->",
                "<!-- LEADERBOARD-END -->",
                "simple",
            )
        }
        result = check_files_up_to_date(str(json_path), target)
        assert result != 0

    def test_passes_up_to_date(self, tmp_path):
        import json

        json_path = tmp_path / "leaderboard.json"
        v2_data = {
            "version": "2.0",
            "datasets": {
                "2016": {
                    "num_routes": 0,
                    "cv_folds": 0,
                    "description": "",
                    "entries": MOCK_ENTRIES,
                },
            },
        }
        json_path.write_text(json.dumps(v2_data))
        md_path = tmp_path / "test.md"
        from moonboard_analysis.scripts.generate_leaderboard import generate_comparison_simple
        table = generate_comparison_simple(MOCK_ENTRIES, [])
        md_path.write_text(
            f"<!-- LEADERBOARD-START -->\n\n{table}\n\n<!-- LEADERBOARD-END -->\n"
        )
        target = {
            str(md_path): (
                "<!-- LEADERBOARD-START -->",
                "<!-- LEADERBOARD-END -->",
                "simple",
            )
        }
        result = check_files_up_to_date(str(json_path), target)
        assert result == 0
