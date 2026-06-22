"""One-time migration: add provenance (data_sha256, git_commit_sha) to existing result JSONs.

For git_commit_sha, uses the commit that last modified each submission's main.py.
For data_sha256, computes the hash of the data file referenced in the result path.
Skips files that already have a 'provenance' key.
"""

import hashlib
import json
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "results"
RAW_DIR = REPO_ROOT / "Raw"
SUBMISSIONS_DIR = REPO_ROOT / "submissions"

DATA_FILES = {
    "2016": RAW_DIR / "moonboard_problems_setup_2016.json",
    "master2017": RAW_DIR / "moonboard_problems_setup_Masters2017.json",
}


def compute_file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def get_last_commit_for_submission(submission_name: str) -> str:
    """Get the commit SHA that last modified the submission's main.py."""
    main_py = SUBMISSIONS_DIR / submission_name / "main.py"
    if not main_py.exists():
        return "unknown"
    try:
        result = subprocess.run(
            ["git", "log", "-1", "--format=%H", str(main_py)],
            capture_output=True,
            text=True,
            timeout=10,
            cwd=REPO_ROOT,
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
    except (subprocess.SubprocessError, FileNotFoundError):
        pass
    return "unknown"


def detect_dataset(name: str) -> str:
    """Detect which dataset a results file belongs to."""
    name_lower = name.lower()
    if "master2017" in name_lower or "master" in name_lower:
        return "master2017"
    return "2016"


def main() -> None:
    result_files = sorted(RESULTS_DIR.glob("*-2016.json"))
    result_files += sorted(RESULTS_DIR.glob("*-master2017.json"))

    # Exclude files that don't follow the pattern
    result_files = [f for f in result_files if f.name != "leaderboard.json"]

    data_hashes: dict[str, str] = {}
    migrated = 0
    skipped = 0
    errors = 0

    for filepath in result_files:
        try:
            data = json.loads(filepath.read_text())
        except (json.JSONDecodeError, OSError) as e:
            print(f"ERROR reading {filepath}: {e}")
            errors += 1
            continue

        if "provenance" in data:
            print(f"SKIP {filepath.name} (already has provenance)")
            skipped += 1
            continue

        # Determine submission name and dataset
        stem = filepath.stem  # e.g. "fast-mlp-2016"
        parts = stem.rsplit("-", 1)
        if len(parts) != 2:
            print(f"SKIP {filepath.name} (unexpected filename format)")
            skipped += 1
            continue
        submission_name = parts[0]
        dataset = detect_dataset(parts[1])

        # Get data SHA256
        data_path = DATA_FILES.get(dataset)
        if data_path is None or not data_path.exists():
            print(f"WARN {filepath.name}: data file for {dataset} not found, skipping hash")
            data_sha = ""
        else:
            if dataset not in data_hashes:
                data_hashes[dataset] = compute_file_sha256(data_path)
            data_sha = data_hashes[dataset]

        # Get git commit SHA from submission's main.py
        git_sha = get_last_commit_for_submission(submission_name)

        data["provenance"] = {
            "data_sha256": data_sha,
            "git_commit_sha": git_sha,
        }

        filepath.write_text(json.dumps(data, indent=2))
        sha_status = "set" if data_sha else "unset"
        git_status = git_sha[:8] if git_sha != "unknown" else "unknown"
        print(f"MIGRATED {filepath.name}: data_sha256={sha_status}, git_commit_sha={git_status}")
        migrated += 1

    print(f"\nDone: {migrated} migrated, {skipped} skipped, {errors} errors")


if __name__ == "__main__":
    main()
