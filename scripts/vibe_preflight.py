from __future__ import annotations

import argparse
import importlib
import json
import os
import re
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List


def _check_python(min_major: int = 3, min_minor: int = 10) -> Dict[str, Any]:
    v = sys.version_info
    ok = (v.major, v.minor) >= (min_major, min_minor)
    return {
        "name": "python_version",
        "required": f">={min_major}.{min_minor}",
        "actual": f"{v.major}.{v.minor}.{v.micro}",
        "ok": ok,
    }


def _check_module(name: str) -> Dict[str, Any]:
    try:
        importlib.import_module(name)
        return {"name": f"module:{name}", "ok": True}
    except Exception as exc:
        return {"name": f"module:{name}", "ok": False, "error": str(exc)}


def _check_file(path: Path) -> Dict[str, Any]:
    return {"name": f"file:{path}", "ok": path.exists()}


def _check_executable(name: str) -> Dict[str, Any]:
    return {"name": f"exec:{name}", "ok": shutil.which(name) is not None}


def _check_dir_writable(path: Path) -> Dict[str, Any]:
    try:
        path.mkdir(parents=True, exist_ok=True)
        test = path / ".write_test"
        test.write_text("ok", encoding="utf-8")
        test.unlink(missing_ok=True)
        return {"name": f"writable:{path}", "ok": True}
    except Exception as exc:
        return {"name": f"writable:{path}", "ok": False, "error": str(exc)}


def run_preflight(repo_root: Path, out_json: Path) -> Path:
    required_modules = ["pandas", "numpy", "pyarrow", "sklearn"]
    required_files = [
        repo_root / "config/champion_testing_v1.json",
        repo_root / "config/champion_refresh_policy_v1.json",
        repo_root / "src/pipelines/champion_e2e.py",
        repo_root / "src/pipelines/champion_refresh_cycle.py",
        repo_root / "src/pipelines/run_champion_ops_cycle.py",
        repo_root / "src/pipelines/build_champion_leaderboard.py",
    ]
    checks: List[Dict[str, Any]] = []

    cfg_path = repo_root / "config/champion_testing_v1.json"
    cfg_payload: Dict[str, Any] = {}
    if cfg_path.exists():
        try:
            cfg_payload = json.loads(cfg_path.read_text(encoding="utf-8"))
        except Exception:
            cfg_payload = {}

    # Add dependency checks based on configured stage commands.
    cmd_blob = "\n".join(
        stage.get("command", "")
        for state_cfg in cfg_payload.get("states", {}).values()
        for stage in state_cfg.get("stages", [])
    )
    optional_by_flag = {
        "--include-xgb": "xgboost",
        "--include-cat": "catboost",
        "--include-lgbm": "lightgbm",
        "--tuner optuna": "optuna",
    }
    for marker, module_name in optional_by_flag.items():
        if marker in cmd_blob and module_name not in required_modules:
            required_modules.append(module_name)

    checks.append(_check_python())
    checks.extend(_check_module(m) for m in required_modules)
    checks.extend(_check_file(f) for f in required_files)
    checks.append(_check_executable("python"))
    checks.append(_check_dir_writable(repo_root / "reports" / "champion_runs"))

    # Validate that state target declarations match stage command overrides.
    target_arg_pattern = re.compile(r"--target-(total|margin)\s+([^\s]+)")
    for state_name, state_cfg in cfg_payload.get("states", {}).items():
        declared_targets = list(state_cfg.get("targets", []))
        target_from_cmd: Dict[str, str] = {}
        for stage in state_cfg.get("stages", []):
            if str(stage.get("name", "")) != "train":
                continue
            cmd = str(stage.get("command", ""))
            for kind, value in target_arg_pattern.findall(cmd):
                target_from_cmd[kind] = value

        total_declared = declared_targets[0] if len(declared_targets) > 0 else None
        margin_declared = declared_targets[1] if len(declared_targets) > 1 else None
        checks.append(
            {
                "name": f"target_alignment:{state_name}",
                "ok": (
                    target_from_cmd.get("total") == total_declared
                    and target_from_cmd.get("margin") == margin_declared
                ),
                "declared": {"total": total_declared, "margin": margin_declared},
                "from_train_command": target_from_cmd,
            }
        )

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "repo_root": str(repo_root),
        "environment": {
            "python": sys.executable,
            "cwd": str(Path.cwd()),
            "VIRTUAL_ENV": os.environ.get("VIRTUAL_ENV"),
        },
        "checks": checks,
        "ok": all(c.get("ok", False) for c in checks),
    }

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Vibe platform preflight checks for champion pipeline")
    parser.add_argument("--out", type=Path, default=Path("reports/champion_runs/preflight.json"))
    parser.add_argument("--allow-fail", action="store_true", help="Return success even if checks fail")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    out = run_preflight(repo_root, args.out)
    print(f"Preflight report written to: {out}")

    payload = json.loads(out.read_text(encoding="utf-8"))
    if not payload.get("ok", False):
        print("Preflight failed. See report for failing checks.")
        if not args.allow_fail:
            raise SystemExit(2)


if __name__ == "__main__":
    main()
