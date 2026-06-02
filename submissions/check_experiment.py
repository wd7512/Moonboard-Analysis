"""Submission gate checker — validates against experiment registry using AST only."""

import argparse
import ast
import subprocess
import sys
from pathlib import Path

TREE_IMPORTS = frozenset({
    "RandomForestClassifier", "RandomForestRegressor", "XGBClassifier", "XGBRegressor",
    "LGBMClassifier", "LGBMRegressor", "CatBoostClassifier", "CatBoostRegressor",
    "GradientBoostingClassifier", "GradientBoostingRegressor",
    "DecisionTreeClassifier", "DecisionTreeRegressor",
})
WEIGHT_EXTS = frozenset({".pth", ".joblib", ".h5", ".onnx", ".pt", ".pkl", ".keras"})


def _ast_names(tree: ast.AST) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                names.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                names.add(node.module.split(".")[0])
            for alias in node.names:
                names.add(alias.name)
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                names.add(node.func.id)
            elif isinstance(node.func, ast.Attribute):
                names.add(node.func.attr)
    return names


def _has_func(tree: ast.AST, name: str) -> bool:
    return any(isinstance(n, ast.FunctionDef) and n.name == name for n in ast.iter_child_nodes(tree))


def _load_tree(main_py: Path) -> ast.AST | None:
    if not main_py.exists():
        return None
    try:
        return ast.parse(main_py.read_text())
    except SyntaxError:
        return None


def _check_interface(submission_dir: Path) -> tuple[bool, str]:
    tree = _load_tree(submission_dir / "main.py")
    if tree is None:
        return False, "main.py not found or has syntax errors"
    if not _has_func(tree, "train_and_evaluate"):
        return False, "must define train_and_evaluate()"
    fn = next(n for n in ast.iter_child_nodes(tree) if isinstance(n, ast.FunctionDef) and n.name == "train_and_evaluate")
    if len(fn.args.args) < 4:
        return False, "train_and_evaluate() must accept (sequences, grades, train_idx, test_idx, ...)"
    return True, ""


def _check_no_weights(submission_dir: Path) -> tuple[bool, str]:
    found = [f.name for f in submission_dir.rglob("*") if f.suffix.lower() in WEIGHT_EXTS and f.is_file()]
    if found:
        return False, f"found weight files: {', '.join(found)}"
    return True, ""


def _check_tree_policy(main_py: Path) -> tuple[bool, str]:
    """Tree methods are allowed. This check no longer gates submissions."""
    return True, ""


def _check_seeds(main_py: Path) -> tuple[bool, str]:
    tree = _load_tree(main_py)
    if tree is None:
        return True, ""
    if "set_seeds" not in _ast_names(tree):
        return False, "must call set_seeds (from moonboard_analysis.utils.reproducibility)"
    return True, ""


def _check_metrics(main_py: Path) -> tuple[bool, str]:
    tree = _load_tree(main_py)
    if tree is None:
        return True, ""
    source = main_py.read_text()
    for key in ("exact_accuracy", "within_one_grade", "within_two_grades"):
        if key not in source:
            return False, f"must compute '{key}' in results dict"
    return True, ""


def _check_feature_redundancy(main_py: Path) -> list[str]:
    tree = _load_tree(main_py)
    if tree is None:
        return []
    source = main_py.read_text()
    funcs = {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
    warnings: list[str] = []

    has_lstm = any(kw in source for kw in ("LSTMSequenceDataset", "ClimbingGradePredictor"))
    has_grid = "GridMapper" in source or "grid_to_vector" in source
    has_matrix = any(f in source for f in ("hold_to_matrix", "sequences_to_matrices"))
    has_sections = any(f in funcs for f in ("_extract_sections", "_section_to_vector"))

    flat_funcs = {"_sequences_to_vectors", "sequences_to_vectors", "_hold_to_index", "hold_to_index"}
    if (flat_funcs & funcs or "HOLD_VECTOR_DIM" in source) and not has_grid and not has_lstm and not has_sections:
        warnings.append("Flat 198-dim binary hold vector — fast-mlp/perceptron territory")
    if has_grid:
        warnings.append("GridMapper / grid binary vector — ridge/tree territory (164-dim)")
    if has_lstm:
        warnings.append("Token sequence / LSTM processing — LSTM territory")
    if has_matrix and not has_grid:
        warnings.append("Single-channel 18x11 binary matrix — 2DCNN territory")
    if has_sections and "INPUT_DIM" in source:
        warnings.append("Section-separated features + meta-features — deep-mlp territory")
    return warnings


def _check_ruff(submission_dir: Path) -> list[str]:
    try:
        result = subprocess.run(["uv", "run", "ruff", "check", str(submission_dir)], capture_output=True, text=True, timeout=60)
    except (FileNotFoundError, subprocess.TimeoutExpired) as e:
        return [f"ruff check unavailable: {e}"]
    issues = [line for line in result.stdout.splitlines() if line.strip() and "All checks passed" not in line]
    if result.returncode != 0 and not issues:
        issues = [f"ruff exited with code {result.returncode}"]
    return issues


def _timing_warn(main_py: Path, max_samples: int) -> tuple[bool, str]:
    """Warn if estimated training time exceeds 1 minute at 250K samples."""
    tree = _load_tree(main_py)
    if tree is None:
        return False, ""
    source = main_py.read_text()
    epochs = 0
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "add_argument":
            for arg in node.args:
                if isinstance(arg, ast.Constant) and arg.value == "--epochs":
                    for kw in node.keywords:
                        if kw.arg == "default" and isinstance(kw.value, ast.Constant):
                            epochs = int(kw.value.value)
    mult = 5 if any(kw in source for kw in ("ensemble", "all_prob", "ensemble_seed")) else 1
    has_torch = "torch" in source
    has_sk = "sklearn" in source

    # Scale estimate to 250K samples.
    # Heuristic: torch models ~linear in samples * epochs; sklearn roughly constant per fit
    scale = max(1, max_samples / 250_000)
    if has_torch and epochs > 0:
        est_min = max(1, (epochs * mult * scale) // 10)
    elif has_sk:
        est_min = max(1, mult)
    else:
        est_min = 1

    # Warn if projected time at 250K samples exceeds 1 minute
    if est_min > 1:
        return True, f"~{est_min} min at {max_samples:,} samples — slow (>1 min at 250K equivalent)"
    return False, ""


def _relevant_sections(text: str) -> list[str]:
    lines = text.splitlines()
    keep = []
    capture = False
    kw = ["leaderboard", "decision matrix", "feature engineering", "submission gate", "training technique"]
    for line in lines:
        if line.startswith("## ") and any(k in line.lower() for k in kw):
            capture = True
        elif line.startswith("## ") and not any(k in line.lower() for k in kw):
            capture = False
        if capture:
            keep.append(line)
    return keep if keep else lines[:50]


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate submission against experiment registry")
    parser.add_argument("--submission-dir", required=True)
    parser.add_argument("--experiments-file", default="EXPERIMENTS.md")
    parser.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args()

    sub_dir = Path(args.submission_dir)
    if not sub_dir.is_dir():
        print(f"[FAIL] Interface contract — dir not found: {sub_dir}"); sys.exit(1)
    main_py = sub_dir / "main.py"
    name = sub_dir.name
    passed = failed = 0
    warnings: list[str] = []

    for label, fn, is_warn in [
        ("Interface contract", lambda: _check_interface(sub_dir), False),
        ("No pre-trained weights", lambda: _check_no_weights(sub_dir), False),
        ("Tree-method policy", lambda: _check_tree_policy(main_py), False),
        ("Uses set_seeds", lambda: _check_seeds(main_py), False),
        ("Computes required metrics", lambda: _check_metrics(main_py), False),
    ]:
        ok, msg = fn()
        if ok:
            print(f"[PASS] {label}"); passed += 1
        else:
            print(f"[FAIL] {label} — {msg}"); failed += 1

    for w in _check_feature_redundancy(main_py):
        print(f"[WARN] Feature redundancy — {w}"); warnings.append(w)

    if args.max_samples is not None:
        warn, msg = _timing_warn(main_py, args.max_samples)
        if warn:
            print(f"[WARN] Training time — {msg}"); warnings.append(msg)
        else:
            print("[PASS] Training time estimate")

    issues = _check_ruff(sub_dir)
    if issues:
        for issue in issues[:10]:
            print(f"[WARN] Code quality — {issue}"); warnings.append(issue)
        if len(issues) > 10:
            print(f"[WARN] Code quality — ... and {len(issues)-10} more")
    else:
        print("[PASS] Code quality (ruff)")

    print(f"\nResult: {passed} passed, {failed} failed, {len(warnings)} warnings")

    exp = Path(args.experiments_file)
    if exp.exists():
        secs = _relevant_sections(exp.read_text())
        print(f"\n--- Relevant EXPERIMENTS.md sections ---")
        print("\n".join(secs))

    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
