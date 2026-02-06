#!/usr/bin/env python3
"""Backtest pregame TOTAL accuracy (MAE/bias) for current vs legacy feature logic.

Compares:
- current feature extraction (fixed team-oriented historical averaging)
- legacy buggy historical averaging used before fix

Usage:
  python scripts/backtest_pregame_total_before_after.py --max-games 500
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import pandas as pd

from src.data.historical_data import get_historical_data_manager
from src.modeling.pregame_model import get_pregame_model
from src.predict_pregame import extract_core_features


@dataclass
class RowResult:
    game_id: str
    actual_total: float
    pred_total_current: float
    pred_total_legacy: float


def _legacy_basic_from_history(hist_mgr, team_id: int, game_date: pd.Timestamp, side: str) -> Dict[str, float]:
    """Reproduce old buggy orientation behavior for basic team ratings."""
    games = hist_mgr.get_team_games(team_id, before_date=game_date, n=20)
    if len(games) == 0:
        return {
            f"{side}_off_rating": 110.0,
            f"{side}_def_rating": 110.0,
            f"{side}_pace": 100.0,
            f"{side}_efg": 0.50,
            f"{side}_ft_rate": 0.25,
            f"{side}_tov_rate": 0.15,
            f"{side}_orb_rate": 0.25,
            f"{side}_win_pct": 0.5,
        }

    # old bug: took home_* means for home team, away_* means for away team,
    # regardless of row orientation for the team itself.
    col_prefix = "home" if side == "home" else "away"

    def m(col: str, default: float) -> float:
        full = f"{col_prefix}_{col}"
        return float(games[full].mean()) if full in games.columns else default

    return {
        f"{side}_off_rating": m("off_rating", 110.0),
        f"{side}_def_rating": m("def_rating", 110.0),
        f"{side}_pace": m("pace", 100.0),
        f"{side}_efg": m("efg", 0.50),
        f"{side}_ft_rate": m("ft_rate", 0.25),
        f"{side}_tov_rate": m("tov_rate", 0.15),
        f"{side}_orb_rate": m("orb_rate", 0.25),
        f"{side}_win_pct": m("win_pct", 0.5),
    }


def _apply_basic_derivations(features: Dict[str, float]) -> None:
    """Recompute deterministic derived fields from basic rating fields."""
    features["home_net_rating"] = features["home_off_rating"] - features["home_def_rating"]
    features["away_net_rating"] = features["away_off_rating"] - features["away_def_rating"]
    features["net_rating_diff"] = features["home_net_rating"] - features["away_net_rating"]

    features["home_ts_proxy"] = features["home_efg"] * features["home_ft_rate"]
    features["away_ts_proxy"] = features["away_efg"] * features["away_ft_rate"]
    features["ts_proxy_diff"] = features["home_ts_proxy"] - features["away_ts_proxy"]

    features["home_assist_ratio_proxy"] = features["home_pace"] / 100.0
    features["away_assist_ratio_proxy"] = features["away_pace"] / 100.0
    features["assist_ratio_diff"] = features["home_assist_ratio_proxy"] - features["away_assist_ratio_proxy"]

    features["home_four_factor_weighted"] = (
        features["home_efg"] * 0.4
        + features["home_orb_rate"] * 0.3
        + features["home_tov_rate"] * -0.15
        + features["home_ft_rate"] * 0.15
    )
    features["away_four_factor_weighted"] = (
        features["away_efg"] * 0.4
        + features["away_orb_rate"] * 0.3
        + features["away_tov_rate"] * -0.15
        + features["away_ft_rate"] * 0.15
    )
    features["four_factor_weighted_diff"] = features["home_four_factor_weighted"] - features["away_four_factor_weighted"]

    features["off_rating_diff"] = features["home_off_rating"] - features["away_off_rating"]
    features["def_rating_diff"] = features["home_def_rating"] - features["away_def_rating"]
    features["pace_diff"] = features["home_pace"] - features["away_pace"]
    features["efg_diff"] = features["home_efg"] - features["away_efg"]
    features["tov_rate_diff"] = features["home_tov_rate"] - features["away_tov_rate"]
    features["orb_rate_diff"] = features["home_orb_rate"] - features["away_orb_rate"]
    features["ft_rate_diff"] = features["home_ft_rate"] - features["away_ft_rate"]

    features["home_home_win_pct"] = features["home_win_pct"] * 1.03
    features["away_road_win_pct"] = features["away_win_pct"] * 0.97
    features["home_efficiency_score"] = features["home_net_rating"]
    features["away_efficiency_score"] = features["away_net_rating"]
    features["efficiency_diff"] = features["home_efficiency_score"] - features["away_efficiency_score"]


def _build_legacy_features(home_id: int, away_id: int, game_date: pd.Timestamp) -> Dict[str, float]:
    hist_mgr = get_historical_data_manager()
    features = extract_core_features(
        home_stats=None,
        away_stats=None,
        home_team_id=home_id,
        away_team_id=away_id,
        game_date=game_date,
    )
    if not hist_mgr:
        return features

    features.update(_legacy_basic_from_history(hist_mgr, home_id, game_date, "home"))
    features.update(_legacy_basic_from_history(hist_mgr, away_id, game_date, "away"))
    _apply_basic_derivations(features)
    return features


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/processed/final_features.parquet")
    ap.add_argument("--max-games", type=int, default=500)
    args = ap.parse_args()

    path = Path(args.data)
    if not path.exists():
        raise SystemExit(f"Data file not found: {path}")

    model = get_pregame_model()
    if model is None:
        raise SystemExit("Pregame model not available (models_v3/pregame missing?)")

    df = pd.read_parquet(path)
    df["game_date"] = pd.to_datetime(df["game_date"], utc=True)
    df = df.sort_values("game_date").tail(args.max_games)

    results: List[RowResult] = []
    for _, row in df.iterrows():
        game_id = str(row.get("game_id"))
        home_id = int(row["home_team_id"])
        away_id = int(row["away_team_id"])
        game_date = pd.Timestamp(row["game_date"])
        actual_total = float(row["total"])

        current_features = extract_core_features(
            home_stats=None,
            away_stats=None,
            home_team_id=home_id,
            away_team_id=away_id,
            game_date=game_date,
        )
        legacy_features = _build_legacy_features(home_id, away_id, game_date)

        pred_current = model.predict(features=current_features, game_id=game_id)
        pred_legacy = model.predict(features=legacy_features, game_id=game_id)
        if pred_current is None or pred_legacy is None:
            continue

        results.append(
            RowResult(
                game_id=game_id,
                actual_total=actual_total,
                pred_total_current=float(pred_current.total_mean),
                pred_total_legacy=float(pred_legacy.total_mean),
            )
        )

    if not results:
        raise SystemExit("No results generated")

    out = pd.DataFrame([r.__dict__ for r in results])
    out["err_current"] = out["pred_total_current"] - out["actual_total"]
    out["err_legacy"] = out["pred_total_legacy"] - out["actual_total"]

    mae_current = float(out["err_current"].abs().mean())
    mae_legacy = float(out["err_legacy"].abs().mean())
    bias_current = float(out["err_current"].mean())
    bias_legacy = float(out["err_legacy"].mean())

    print(f"Games evaluated: {len(out)}")
    print("\nTOTAL metrics")
    print(f"  current MAE: {mae_current:.3f}")
    print(f"  legacy  MAE: {mae_legacy:.3f}")
    print(f"  MAE delta (legacy-current): {mae_legacy - mae_current:+.3f}")
    print(f"  current bias: {bias_current:+.3f}")
    print(f"  legacy  bias: {bias_legacy:+.3f}")
    print(f"  bias delta (legacy-current): {bias_legacy - bias_current:+.3f}")


if __name__ == "__main__":
    main()
