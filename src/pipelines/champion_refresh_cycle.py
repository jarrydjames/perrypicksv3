from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _safe_import_pandas():
    try:
        import pandas as pd  # type: ignore

        return pd
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("pandas is required for refresh-cycle data checks") from exc


def _dataset_freshness_check(dataset_path: Path, holdout_days: int, skip_checks: bool=False) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "path": str(dataset_path),
        "exists": dataset_path.exists(),
        "rows": 0,
        "latest_game_date": None,
        "holdout_cutoff": None,
        "ok": False,
    }
    if not dataset_path.exists():
        return result

    if skip_checks:
        result["ok"] = True
        return result

    pd = _safe_import_pandas()
    df = pd.read_parquet(dataset_path)
    result["rows"] = int(len(df))
    if len(df) == 0:
        return result

    game_date_col = None
    for c in ["game_date", "gameTimeUTC", "date", "game_datetime"]:
        if c in df.columns:
            game_date_col = c
            break

    if game_date_col is None:
        result["ok"] = True
        return result

    dates = pd.to_datetime(df[game_date_col], errors="coerce", utc=True).dropna()
    if len(dates) == 0:
        return result

    latest = dates.max().to_pydatetime()
    cutoff = datetime.now(timezone.utc) - timedelta(days=int(holdout_days))
    result["latest_game_date"] = latest.isoformat()
    result["holdout_cutoff"] = cutoff.isoformat()
    result["ok"] = latest >= cutoff
    return result


def _new_data_volume_check(dataset_path: Path, minimum_new_games: int, skip_checks: bool=False) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "path": str(dataset_path),
        "minimum_new_games_to_retrain": int(minimum_new_games),
        "estimated_new_games": 0,
        "ok": False,
    }
    if not dataset_path.exists():
        return result

    if skip_checks:
        result["ok"] = True
        return result

    pd = _safe_import_pandas()
    df = pd.read_parquet(dataset_path)
    if "season_end_yy" in df.columns:
        recent = df[df["season_end_yy"] == df["season_end_yy"].max()]
        estimated_new_games = int(len(recent))
    else:
        estimated_new_games = int(len(df))

    result["estimated_new_games"] = estimated_new_games
    result["ok"] = estimated_new_games >= int(minimum_new_games)
    return result


def _leaderboard_presence_check(path: Path) -> Dict[str, Any]:
    return {
        "path": str(path),
        "exists": path.exists(),
        "ok": path.exists(),
    }


def evaluate_refresh_readiness(
    policy_path: Path,
    output_path: Path,
    skip_checks: bool=False,
) -> Path:
    policy = _load_json(policy_path)

    windows = policy.get("windows", {})
    holdout_days = int(windows.get("holdout_days", 30))
    minimum_new_games = int(windows.get("minimum_new_games_to_retrain", 120))

    report: Dict[str, Any] = {
        "policy": str(policy_path),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "cadence": policy.get("cadence", {}),
        "promotion_gates": policy.get("promotion_gates", {}),
        "drift_triggers": policy.get("drift_triggers", {}),
        "rollout": policy.get("rollout", {}),
        "states": {},
        "recommendation": "unknown",
        "ok": True,
    }

    for state, state_cfg in policy.get("states", {}).items():
        dataset = Path(state_cfg["dataset"])
        leaderboard = Path(state_cfg["leaderboard"])

        freshness = _dataset_freshness_check(dataset, holdout_days=holdout_days, skip_checks=skip_checks)
        volume = _new_data_volume_check(dataset, minimum_new_games=minimum_new_games, skip_checks=skip_checks)
        leaderboard_check = _leaderboard_presence_check(leaderboard)

        state_ok = bool(freshness.get("ok", False)) and bool(leaderboard_check.get("ok", False))

        report["states"][state] = {
            "freshness": freshness,
            "new_data_volume": volume,
            "leaderboard": leaderboard_check,
            "ready_for_full_retrain": bool(state_ok and volume.get("ok", False)),
            "ready_for_calibration_only": bool(state_ok and not volume.get("ok", False)),
            "ok": state_ok,
        }

        if not state_ok:
            report["ok"] = False

    if report["ok"] and all(v["ready_for_full_retrain"] for v in report["states"].values()):
        report["recommendation"] = "full_retrain"
    elif report["ok"] and all(v["ready_for_calibration_only"] for v in report["states"].values()):
        report["recommendation"] = "calibration_only"
    elif report["ok"]:
        report["recommendation"] = "mixed_state_action"
    else:
        report["recommendation"] = "block_promotion_and_fix_data"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate recurring model refresh readiness.")
    parser.add_argument(
        "--policy",
        type=Path,
        default=Path("config/champion_refresh_policy_v1.json"),
        help="Refresh policy JSON path",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("reports/champion_runs/refresh_readiness.json"),
        help="Output report JSON",
    )
    parser.add_argument("--skip-checks", action="store_true", help="Skip dataframe-dependent checks")
    args = parser.parse_args()

    out = evaluate_refresh_readiness(args.policy, args.out, skip_checks=bool(args.skip_checks))
    print(f"Refresh readiness report written to: {out}")


if __name__ == "__main__":
    main()
