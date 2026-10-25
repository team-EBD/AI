"""Run from ai-server: python -m experiments.output_validation_20261025.run_experiment --output PATH."""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import io
import json
from pathlib import Path
import platform
import subprocess
import sys
import tarfile
import tempfile


def source_hash(source_root):
    digest = hashlib.sha256()
    for path in sorted((source_root / "app").rglob("*.py")):
        digest.update(path.relative_to(source_root).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--baseline-ref", help="Load app code from this git revision into an isolated temporary directory")
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[3]
    git = ["git", "-c", f"safe.directory={repo.as_posix()}", "-C", str(repo)]
    revision = subprocess.check_output([*git, "rev-parse", args.baseline_ref or "HEAD"], text=True).strip()
    temporary = None
    source_root = repo / "ai-server"
    if args.baseline_ref:
        # Tracked application code only: no .env, credentials, data or git metadata.
        archive = subprocess.check_output([*git, "archive", "--format=tar", revision, "ai-server/app"])
        temporary = tempfile.TemporaryDirectory(prefix="eatlog-ai-baseline-")
        with tarfile.open(fileobj=io.BytesIO(archive)) as source:
            source.extractall(temporary.name, filter="data")
        source_root = Path(temporary.name) / "ai-server"
        sys.path.insert(0, str(source_root))
    from .cases import CASES

    rows = []
    groups = defaultdict(lambda: {"passed": 0, "total": 0})
    for case in CASES:
        try:
            case["check"]()
            passed, error = True, None
        except Exception as exc:  # record every outcome, including uncaught service errors
            passed, error = False, type(exc).__name__
        rows.append({"id": case["id"], "group": case["group"], "control": case["control"], "passed": passed, "error_type": error})
        groups[case["group"]]["total"] += 1
        groups[case["group"]]["passed"] += int(passed)
    report = {
        "kind": "synthetic_output_contract_fault_injection",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "base_revision": revision,
        "app_source_sha256": source_hash(source_root),
        "source": "git_archive" if args.baseline_ref else "working_tree",
        "python": platform.python_version(),
        "packages": {p: importlib.metadata.version(p) for p in ("fastapi", "pydantic", "pytest", "httpx")},
        "live_model_calls": 0, "production_db_queries": 0,
        "limitations": "Hand-authored fault cases, not an independent sample or a photo/nutrition accuracy benchmark. No user outcome, prevalence, confidence calibration or model robustness inference.",
        "passed": sum(r["passed"] for r in rows), "total": len(rows),
        "controls": {"passed": sum(r["passed"] for r in rows if r["control"]), "total": sum(r["control"] for r in rows)},
        "groups": dict(groups), "cases": rows,
    }
    Path(args.output).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("passed", "total", "controls", "groups")}, ensure_ascii=False))
    if temporary is not None:
        assert Path(temporary.name).resolve().parent == Path(tempfile.gettempdir()).resolve()
        temporary.cleanup()


if __name__ == "__main__":
    main()
