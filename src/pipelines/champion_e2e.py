from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List


@dataclass
class StageResult:
    state: str
    stage: str
    command: str
    return_code: int
    log_file: str
    ok: bool


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _safe_import_pandas():
    try:
        import pandas as pd  # type: ignore

        return pd
    except Exception as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "pandas is required for dataset/leaderboard validation. "
            "Install requirements and re-run."
        ) from exc


def _check_required_columns(dataset_path: Path, required_columns: List[str], skip_checks: bool) -> Dict[str, Any]:
    report: Dict[str, Any] = {
        "dataset_path": str(dataset_path),
        "exists": dataset_path.exists(),
        "rows": None,
        "columns": None,
        "missing_columns": [],
        "ok": False,
    }
    if not dataset_path.exists():
        return report

    if skip_checks:
        report["ok"] = True
        return report

    pd = _safe_import_pandas()
    df = pd.read_parquet(dataset_path)
    cols = list(df.columns)
    missing = [c for c in required_columns if c not in cols]

    report["rows"] = int(len(df))
    report["columns"] = int(len(cols))
    report["missing_columns"] = missing
    report["ok"] = len(missing) == 0 and len(df) > 0
    return report


def _run_stage(command: str, log_file: Path, dry_run: bool) -> tuple[int, bool]:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    if dry_run:
        log_file.write_text(f"[DRY RUN] {command}\n", encoding="utf-8")
        return 0, True

    completed = subprocess.run(command, shell=True, capture_output=True, text=True)
    log_file.write_text(
        f"$ {command}\n\nSTDOUT:\n{completed.stdout}\n\nSTDERR:\n{completed.stderr}\n",
        encoding="utf-8",
    )
    return completed.returncode, completed.returncode == 0


def _assert_artifacts(paths: List[str], root: Path) -> Dict[str, Any]:
    checks = []
    for rel in paths:
        abs_path = root / rel
        checks.append({"path": rel, "exists": abs_path.exists()})
    return {
        "checks": checks,
        "ok": all(item["exists"] for item in checks),
    }


def _validate_leaderboard(path: Path, required_models: List[str], skip_checks: bool) -> Dict[str, Any]:
    required_columns = [
        "model",
        "mae_total",
        "mae_margin",
        "rmse_total",
        "ece_win",
        "stability_std_mae_total",
    ]
    result: Dict[str, Any] = {
        "leaderboard": str(path),
        "exists": path.exists(),
        "missing_models": [],
        "missing_columns": [],
        "columns": [],
        "ok": False,
    }

    if not path.exists():
        return result

    if skip_checks:
        result["ok"] = True
        return result

    pd = _safe_import_pandas()
    df = pd.read_csv(path)
    result["columns"] = list(df.columns)
    result["missing_columns"] = [c for c in required_columns if c not in df.columns]

    if result["missing_columns"]:
        result["missing_models"] = required_models
        return result

    available = set(str(v).strip().lower() for v in df["model"].astype(str).tolist())
    missing = [m for m in required_models if m not in available]
    result["missing_models"] = missing
    result["ok"] = len(missing) == 0 and len(result["missing_columns"]) == 0 and len(df) > 0
    return result


def _promote_champions(champion_map: Dict[str, Dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": "src/pipelines/champion_e2e.py",
        "champions": champion_map,
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def run_pipeline(config_path: Path, promote: bool, dry_run: bool, skip_checks: bool) -> Path:
    repo_root = Path.cwd()
    config = _load_json(config_path)

    run_id = _now_utc()
    run_root = repo_root / "reports" / "champion_runs" / run_id
    latest_dir = repo_root / "reports" / "champion_runs" / "latest"

    run_root.mkdir(parents=True, exist_ok=True)
    latest_dir.mkdir(parents=True, exist_ok=True)

    report: Dict[str, Any] = {
        "run_id": run_id,
        "config": str(config_path),
        "dry_run": dry_run,
        "skip_checks": skip_checks,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "global_checks": config.get("global_checks", {}),
        "states": {},
        "stage_results": [],
        "ok": True,
    }

    stage_results: List[StageResult] = []
    champions: Dict[str, Dict[str, Any]] = {}

    required_models = [m.lower() for m in config.get("required_models", [])]
    states = config.get("states", {})

    for state, state_cfg in states.items():
        dataset_path = repo_root / state_cfg["dataset_path"]
        dataset_check = _check_required_columns(
            dataset_path,
            state_cfg.get("required_columns", []),
            skip_checks,
        )

        state_report: Dict[str, Any] = {
            "dataset_check": dataset_check,
            "stages": [],
            "leaderboard_check": None,
            "ok": bool(dataset_check.get("ok", False)),
        }

        if not dataset_check.get("ok", False):
            report["ok"] = False

        for stage in state_cfg.get("stages", []):
            command = stage["command"].replace("reports/champion_runs/latest", str(latest_dir))
            log_file = run_root / f"{state}_{stage['name']}.log"
            return_code, ok = _run_stage(command, log_file, dry_run)

            artifact_check = _assert_artifacts(stage.get("required_artifacts", []), repo_root)
            if not artifact_check["ok"]:
                ok = False

            state_stage = {
                "name": stage["name"],
                "command": command,
                "return_code": return_code,
                "ok": ok,
                "log_file": str(log_file.relative_to(repo_root)),
                "artifact_check": artifact_check,
            }
            state_report["stages"].append(state_stage)

            stage_results.append(
                StageResult(
                    state=state,
                    stage=stage["name"],
                    command=command,
                    return_code=return_code,
                    log_file=str(log_file.relative_to(repo_root)),
                    ok=ok,
                )
            )

            if not ok:
                state_report["ok"] = False
                report["ok"] = False

        leaderboard_path = repo_root / state_cfg.get("leaderboard_path", "")
        leaderboard_check = _validate_leaderboard(leaderboard_path, required_models, skip_checks)
        state_report["leaderboard_check"] = leaderboard_check
        if not leaderboard_check.get("ok", False):
            state_report["ok"] = False
            report["ok"] = False

        champions[state] = {
            "status": "pending_manual_selection" if not leaderboard_check.get("ok", False) else "eligible",
            "leaderboard": str(leaderboard_path.relative_to(repo_root)) if leaderboard_path.exists() else str(leaderboard_path),
            "targets": state_cfg.get("targets", []),
            "required_models": required_models,
        }

        report["states"][state] = state_report

    report["stage_results"] = [sr.__dict__ for sr in stage_results]
    report["completed_at"] = datetime.now(timezone.utc).isoformat()

    report_path = run_root / "run_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    champions_path = run_root / "champion_candidates.json"
    champions_path.write_text(json.dumps(champions, indent=2), encoding="utf-8")

    latest_report = latest_dir / "run_report.json"
    shutil.copy2(report_path, latest_report)
    shutil.copy2(champions_path, latest_dir / "champion_candidates.json")

    if promote and report["ok"]:
        _promote_champions(champions, repo_root / "data" / "processed" / "champion_models.json")

    return report_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Canonical champion testing and promotion orchestrator.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/champion_testing_v1.json"),
        help="Path to champion testing config JSON.",
    )
    parser.add_argument("--promote", action="store_true", help="Promote champions if all checks pass.")
    parser.add_argument("--dry-run", action="store_true", help="Log commands without executing them.")
    parser.add_argument(
        "--skip-checks",
        action="store_true",
        help="Skip dataset and leaderboard dataframe validation (for environments without pandas).",
    )

    args = parser.parse_args()

    report_path = run_pipeline(
        config_path=args.config,
        promote=bool(args.promote),
        dry_run=bool(args.dry_run),
        skip_checks=bool(args.skip_checks),
    )
    print(f"Run report written to: {report_path}")


if __name__ == "__main__":
    main()
