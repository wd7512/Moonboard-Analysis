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
    with open(path) as f:
        return json.load(f)


def _fmt_pct(mean: float, std: float) -> str:
    return f"{mean * 100:.2f} ± {std * 100:.2f}"


def _fmt_pct_parens(mean: float, std: float) -> str:
    return f"{mean * 100:.2f} (±{std * 100:.2f})"


def _entries_sorted(entries: list[dict]) -> list[dict]:
    return sorted(entries, key=lambda e: e["exact_accuracy"]["mean"], reverse=True)


def generate_simple_table(entries: list[dict]) -> str:
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


def _bold_pct(mean: float, std: float) -> str:
    return f"**{mean * 100:.2f}** (±{std * 100:.2f})"


def generate_detailed_table(entries: list[dict]) -> str:
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
        row += f" | {mf1_cell} | {e['training_time']} |"
        lines.append(row)
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
    leaderboard = load_leaderboard(leaderboard_path)
    entries = leaderboard["entries"]
    outdated = False
    for relative_path, (start_marker, end_marker, table_type) in target_files.items():
        filepath = _resolve_path(relative_path)
        table_fn = generate_simple_table if table_type == "simple" else generate_detailed_table
        expected_table = table_fn(entries)
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
    leaderboard = load_leaderboard(LEADERBOARD_PATH)
    entries = leaderboard["entries"]
    any_change = False
    for relative_path, (start_marker, end_marker, table_type) in TARGET_FILES.items():
        filepath = _resolve_path(relative_path)
        table_fn = (
            generate_simple_table if table_type == "simple" else generate_detailed_table
        )
        new_table = table_fn(entries)
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
