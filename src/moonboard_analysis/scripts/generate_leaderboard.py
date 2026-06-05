"""Generate leaderboard tables from dual-dataset JSON source.

Handles both legacy (single-dataset) and v2 (dual-dataset) leaderboard formats.
Generates comparison tables showing 2016 and Masters 2017 metrics side by side.

Usage:
    uv run python src/moonboard_analysis/scripts/generate_leaderboard.py --check
    uv run python src/moonboard_analysis/scripts/generate_leaderboard.py --write
"""

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
LEADERBOARD_PATH = REPO_ROOT / "results" / "leaderboard.json"

_START = "<!-- LEADERBOARD-START -->"
_END = "<!-- LEADERBOARD-END -->"
_FULL_START = "<!-- LEADERBOARD-FULLDATA-START -->"
_FULL_END = "<!-- LEADERBOARD-FULLDATA-END -->"

TARGET_FILES: dict[str, tuple[str, str, str]] = {
    "README.md": (_START, _END, "simple"),
    "EXPERIMENTS.md": (_FULL_START, _FULL_END, "detailed"),
    "docs/index.md": (_START, _END, "simple"),
    "results/full_data_summary.md": (_START, _END, "simple"),
}


def load_leaderboard(path: str | Path) -> dict:
    """Load leaderboard JSON, supporting both legacy and v2 format."""
    with open(path) as f:
        data = json.load(f)

    # v2 format: {"version": "2.0", "datasets": {"2016": {...}, "master2017": {...}}}
    if "datasets" in data:
        return data

    # Legacy format: {"version": "1.0", "dataset": "...", "entries": [...]}
    # Convert to v2 format
    return {
        "version": "2.0",
        "datasets": {
            "2016": {
                "num_routes": data.get("num_routes", 0),
                "cv_folds": data.get("cv_folds", 5),
                "description": data.get("description", ""),
                "entries": data.get("entries", []),
            },
        },
    }


def _fmt_pct(mean: float, std: float) -> str:
    return f"{mean * 100:.2f} ± {std * 100:.2f}"


def _fmt_pct_parens(mean: float, std: float) -> str:
    return f"{mean * 100:.2f} (±{std * 100:.2f})"


def _bold_pct(mean: float, std: float) -> str:
    return f"**{mean * 100:.2f}** (±{std * 100:.2f})"


def _entries_sorted(entries: list[dict]) -> list[dict]:
    return sorted(entries, key=lambda e: e["exact_accuracy"]["mean"], reverse=True)


def generate_simple_table(entries: list[dict]) -> str:
    """Generate a simple leaderboard table from entries."""
    if not entries:
        return "No entries"
    sorted_entries = _entries_sorted(entries)
    lines = [
        "| Model | Exact (%) | Within +/-1 (%) | Within +/-2 (%) | Macro-F1 (%) |",
        "|-------|-----------|-----------------|-----------------|--------------|",
    ]
    for e in sorted_entries:
        exact = _fmt_pct(e["exact_accuracy"]["mean"], e["exact_accuracy"]["std"])
        w1 = _fmt_pct(e["within_one_grade"]["mean"], e["within_one_grade"]["std"])
        w2 = _fmt_pct(e["within_two_grades"]["mean"], e["within_two_grades"]["std"])
        mf1 = _fmt_pct(e["macro_f1"]["mean"], e["macro_f1"]["std"])
        lines.append(f"| {e['model']} | {exact} | {w1} | {w2} | {mf1} |")
    return "\n".join(lines)


def generate_detailed_table(entries: list[dict]) -> str:
    """Generate a detailed leaderboard table from entries."""
    if not entries:
        return "No entries"
    sorted_entries = _entries_sorted(entries)
    best_macro_f1 = max(e["macro_f1"]["mean"] for e in sorted_entries)
    lines = [
        "| # | Submission | Exact (%) | ±1 (%) | ±2 (%) | Macro-F1 (%) | Training Time |",
        "|---|-----------|-----------|--------|--------|--------------|---------------|",
    ]
    for rank, e in enumerate(sorted_entries, start=1):
        w1 = _fmt_pct_parens(e["within_one_grade"]["mean"], e["within_one_grade"]["std"])
        w2 = _fmt_pct_parens(e["within_two_grades"]["mean"], e["within_two_grades"]["std"])
        mf1 = _fmt_pct_parens(e["macro_f1"]["mean"], e["macro_f1"]["std"])
        exact_cell = _bold_pct(
            e["exact_accuracy"]["mean"], e["exact_accuracy"]["std"]
        )
        if e["macro_f1"]["mean"] == best_macro_f1:
            mf1_cell = _bold_pct(e["macro_f1"]["mean"], e["macro_f1"]["std"])
        else:
            mf1_cell = mf1
        row = f"| {rank} | {e['submission']} | {exact_cell} | {w1} | {w2}"
        row += f" | {mf1_cell} | {e.get('training_time', '')} |"
        lines.append(row)
    return "\n".join(lines)


def generate_comparison_simple(entries_2016: list[dict], entries_2017: list[dict]) -> str:
    """Generate a side-by-side comparison table for both datasets."""
    if not entries_2016 and not entries_2017:
        return "No entries"

    # Build lookup by submission name
    by_name_2016 = {e["submission"]: e for e in entries_2016}
    by_name_2017 = {e["submission"]: e for e in entries_2017}
    all_names = sorted(set(list(by_name_2016.keys()) + list(by_name_2017.keys())))

    lines = [
        "## 2016 Hold Setup",
        "",
        generate_simple_table(entries_2016),
        "",
        "## Masters 2017 Hold Setup",
        "",
        generate_simple_table(entries_2017),
        "",
        "## Side-by-Side Comparison",
        "",
        "| Model | 2016 Exact (%) | 2017 Exact (%) | 2016 Macro-F1 (%) | 2017 Macro-F1 (%) |",
        "|-------|---------------|---------------|-------------------|-------------------|",
    ]
    for name in all_names:
        e16 = by_name_2016.get(name)
        e17 = by_name_2017.get(name)
        entry = e16 if e16 is not None else e17
        if entry is None:
            continue
        model = entry["model"]
        ea16 = _fmt_pct(e16["exact_accuracy"]["mean"], e16["exact_accuracy"]["std"]) if e16 else "—"
        ea17 = _fmt_pct(e17["exact_accuracy"]["mean"], e17["exact_accuracy"]["std"]) if e17 else "—"
        mf16 = _fmt_pct(e16["macro_f1"]["mean"], e16["macro_f1"]["std"]) if e16 else "—"
        mf17 = _fmt_pct(e17["macro_f1"]["mean"], e17["macro_f1"]["std"]) if e17 else "—"
        lines.append(f"| {model} | {ea16} | {ea17} | {mf16} | {mf17} |")

    return "\n".join(lines)


def replace_between_markers(
    content: str, start_marker: str, end_marker: str, table: str
) -> str:
    if start_marker not in content:
        raise ValueError(f"Start marker {start_marker!r} not found in content")
    if end_marker not in content:
        raise ValueError(f"End marker {end_marker!r} not found in content")
    before = content[: content.index(start_marker) + len(start_marker)]
    after_start = content[content.index(start_marker) + len(start_marker) :]
    after = after_start[after_start.index(end_marker) :]
    return before + "\n\n" + table + "\n\n" + after


def _resolve_path(relative_path: str) -> Path:
    return REPO_ROOT / relative_path


def check_files_up_to_date(
    leaderboard_path: str | Path,
    target_files: dict[str, tuple[str, str, str]],
) -> int:
    """Check if all target files contain the latest leaderboard tables."""
    leaderboard = load_leaderboard(leaderboard_path)
    datasets = leaderboard.get("datasets", {})
    outdated = False

    entries_2016 = datasets.get("2016", {}).get("entries", [])
    entries_2017 = datasets.get("master2017", {}).get("entries", [])

    for relative_path, (start_marker, end_marker, table_type) in target_files.items():
        filepath = _resolve_path(relative_path)
        if table_type == "simple":
            expected_table = generate_comparison_simple(entries_2016, entries_2017)
        else:
            # Detailed tables: show each dataset separately
            parts = []
            if entries_2016:
                parts.append(f"## 2016 Hold Setup\n\n{generate_detailed_table(entries_2016)}")
            if entries_2017:
                parts.append(
                    f"## Masters 2017 Hold Setup\n\n{generate_detailed_table(entries_2017)}"
                )
            expected_table = "\n\n".join(parts)

        content = filepath.read_text()
        try:
            new_content = replace_between_markers(
                content, start_marker, end_marker, expected_table
            )
        except ValueError as e:
            print(f"ERROR: {filepath}: {e}", file=sys.stderr)
            outdated = True
            continue
        if content != new_content:
            print(f"OUTDATED: {filepath}")
            outdated = True
    return 1 if outdated else 0


def _process_all(mode: str) -> int:
    """Generate or check all leaderboard tables."""
    leaderboard = load_leaderboard(LEADERBOARD_PATH)
    datasets = leaderboard.get("datasets", {})
    entries_2016 = datasets.get("2016", {}).get("entries", [])
    entries_2017 = datasets.get("master2017", {}).get("entries", [])
    any_change = False

    for relative_path, (start_marker, end_marker, table_type) in TARGET_FILES.items():
        filepath = _resolve_path(relative_path)

        if table_type == "simple":
            new_table = generate_comparison_simple(entries_2016, entries_2017)
        else:
            parts = []
            if entries_2016:
                parts.append(f"## 2016 Hold Setup\n\n{generate_detailed_table(entries_2016)}")
            if entries_2017:
                parts.append(
                    f"## Masters 2017 Hold Setup\n\n{generate_detailed_table(entries_2017)}"
                )
            new_table = "\n\n".join(parts)

        content = filepath.read_text()
        try:
            new_content = replace_between_markers(
                content, start_marker, end_marker, new_table
            )
        except ValueError as e:
            print(f"ERROR: {filepath}: {e}", file=sys.stderr)
            return 1

        if content != new_content:
            any_change = True
            if mode == "write":
                filepath.write_text(new_content)
                print(f"UPDATED: {filepath}")
            else:
                print(f"WOULD UPDATE: {filepath}")
        else:
            print(f"OK: {filepath}")

    if mode != "write" and any_change:
        print("\nDry-run: no files modified. Pass --write to apply changes.")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate leaderboard tables from JSON source"
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--check",
        action="store_true",
        help="Verify all files are up to date (exit 1 if not)",
    )
    group.add_argument(
        "--write", action="store_true", help="Write updated tables to files"
    )
    args = parser.parse_args()
    if args.check:
        sys.exit(check_files_up_to_date(LEADERBOARD_PATH, TARGET_FILES))
    sys.exit(_process_all("write" if args.write else "dry-run"))


if __name__ == "__main__":
    main()
