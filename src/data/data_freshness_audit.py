from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


def _safe_import_pandas():
    try:
        import pandas as pd  # type: ignore

        return pd
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("pandas is required for data freshness audit") from exc


@dataclass(frozen=True)
class DatasetPolicy:
    path: str
    date_columns: List[str]
    required_columns: List[str]
    recency_feature_columns: List[str]
    max_null_pct_per_feature: float
    min_rows: int
    required_seasons: List[int]
    min_games_current_season: int
    freshness_days: int


def _load_policy(policy_path: Path) -> Dict[str, Any]:
    with policy_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _find_first_existing(columns: List[str], candidates: List[str]) -> Optional[str]:
    for col in candidates:
        if col in columns:
            return col
    return None


def _to_dataset_policy(data: Dict[str, Any]) -> DatasetPolicy:
    return DatasetPolicy(
        path=str(data["path"]),
        date_columns=list(data.get("date_columns", [])),
        required_columns=list(data.get("required_columns", [])),
        recency_feature_columns=list(data.get("recency_feature_columns", [])),
        max_null_pct_per_feature=float(data.get("max_null_pct_per_feature", 20.0)),
        min_rows=int(data.get("min_rows", 1)),
        required_seasons=[int(s) for s in data.get("required_seasons", [])],
        min_games_current_season=int(data.get("min_games_current_season", 1)),
        freshness_days=int(data.get("freshness_days", 7)),
    )


def _audit_dataset(name: str, policy: DatasetPolicy) -> Dict[str, Any]:
    pd = _safe_import_pandas()

    path = Path(policy.path)
    result: Dict[str, Any] = {
        "dataset": name,
        "path": str(path),
        "exists": path.exists(),
        "ok": False,
        "errors": [],
        "warnings": [],
    }

    if not path.exists():
        result["errors"].append("dataset_missing")
        return result

    df = pd.read_parquet(path)
    result["rows"] = int(len(df))
    result["columns"] = int(len(df.columns))

    if len(df) < policy.min_rows:
        result["errors"].append(f"row_count_below_min:{len(df)}<{policy.min_rows}")

    missing_required = [c for c in policy.required_columns if c not in df.columns]
    result["missing_required_columns"] = missing_required
    if missing_required:
        result["errors"].append("missing_required_columns")

    date_col = _find_first_existing(list(df.columns), policy.date_columns)
    result["date_column_used"] = date_col

    if date_col is None:
        result["errors"].append("missing_date_column")
    else:
        dates = pd.to_datetime(df[date_col], errors="coerce", utc=True).dropna()
        if dates.empty:
            result["errors"].append("no_parseable_dates")
        else:
            latest = dates.max().to_pydatetime()
            cutoff = datetime.now(timezone.utc) - timedelta(days=policy.freshness_days)
            result["latest_game_date"] = latest.isoformat()
            result["freshness_cutoff"] = cutoff.isoformat()
            result["fresh_enough"] = bool(latest >= cutoff)
            if latest < cutoff:
                result["errors"].append("stale_dataset")

    season_col = "season_end_yy" if "season_end_yy" in df.columns else None
    if season_col is None and policy.required_seasons:
        result["warnings"].append("season_end_yy_missing")
    elif season_col is not None:
        seasons_present = sorted({int(s) for s in df[season_col].dropna().astype(int).tolist()})
        result["seasons_present"] = seasons_present
        missing_seasons = [s for s in policy.required_seasons if s not in seasons_present]
        result["missing_required_seasons"] = missing_seasons
        if missing_seasons:
            result["errors"].append("missing_required_seasons")

        if seasons_present:
            current_season = max(seasons_present)
            current_games = int((df[season_col].astype("Int64") == current_season).sum())
            result["current_season"] = current_season
            result["current_season_rows"] = current_games
            if current_games < policy.min_games_current_season:
                result["errors"].append("insufficient_current_season_rows")

    null_pct: Dict[str, float] = {}
    for col in policy.recency_feature_columns:
        if col in df.columns:
            pct = float(df[col].isna().mean() * 100.0)
            null_pct[col] = pct
            if pct > policy.max_null_pct_per_feature:
                result["errors"].append(f"high_null_rate:{col}:{pct:.2f}")
        else:
            result["errors"].append(f"missing_recency_feature:{col}")

    result["recency_null_pct"] = null_pct
    result["ok"] = len(result["errors"]) == 0
    return result


def run_audit(policy_path: Path, out_path: Path) -> Dict[str, Any]:
    policy_raw = _load_policy(policy_path)
    datasets = policy_raw.get("datasets", {})
    report: Dict[str, Any] = {
        "policy": str(policy_path),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "datasets": {},
        "ok": True,
    }

    for name, data in datasets.items():
        dataset_report = _audit_dataset(name, _to_dataset_policy(data))
        report["datasets"][name] = dataset_report
        if not dataset_report.get("ok", False):
            report["ok"] = False

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit parquet datasets for freshness and feature coverage")
    parser.add_argument(
        "--policy",
        type=Path,
        default=Path("config/data_freshness_policy_v1.json"),
        help="JSON policy path",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("reports/champion_runs/data_freshness_audit.json"),
        help="JSON report output path",
    )
    parser.add_argument("--strict", action="store_true", help="Exit non-zero when checks fail")
    args = parser.parse_args()

    report = run_audit(args.policy, args.out)
    print(f"Data freshness audit report written to: {args.out}")

    if args.strict and not report.get("ok", False):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
