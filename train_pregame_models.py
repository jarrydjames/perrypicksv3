from __future__ import annotations

"""Train upgraded pregame models with walk-forward validation and model comparison output."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit

from src.modeling.stacked_pregame_ensemble import StackedPregameEnsemble
from src.modeling.win_probability_pregame import WinProbabilityPregameModel


PROCESSED_PATH = Path("data/processed/pregame_features.parquet")
FINAL_REPORT_PATH = Path("reports/final_pregame_model_comparison.csv")
COMPARISON_REPORT_PATH = Path("reports/pregame_model_comparison.csv")


def _find_column(df: pd.DataFrame, candidates: list[str]) -> str:
    for name in candidates:
        if name in df.columns:
            return name
    raise KeyError(f"Missing expected columns: {candidates}")


def _rolling_mae(values_true: np.ndarray, values_pred: np.ndarray, window: int) -> float:
    valid = ~np.isnan(values_pred)
    y_true = values_true[valid]
    y_pred = values_pred[valid]
    if len(y_true) < window:
        return float("nan")
    errs = np.abs(y_true - y_pred)
    return float(pd.Series(errs).rolling(window).mean().dropna().mean())


def run_training() -> pd.DataFrame:
    if not PROCESSED_PATH.exists():
        raise FileNotFoundError(f"Missing pregame training data: {PROCESSED_PATH}")

    df = pd.read_parquet(PROCESSED_PATH)
    date_col = _find_column(df, ["game_date", "date"])
    margin_col = _find_column(df, ["margin", "home_margin", "score_margin"])
    total_col = _find_column(df, ["total", "game_total"])
    home_team_col = _find_column(df, ["home_team"])
    away_team_col = _find_column(df, ["away_team"])

    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col).reset_index(drop=True)

    y_win = (df[margin_col] > 0).astype(int).values
    drop_cols = {
        date_col,
        margin_col,
        total_col,
        "home_score",
        "away_score",
        "game_id",
        home_team_col,
        away_team_col,
        "season",
    }
    feature_cols = [c for c in df.columns if c not in drop_cols and pd.api.types.is_numeric_dtype(df[c])]

    X = df[feature_cols].fillna(0.0).values
    y_margin = df[margin_col].astype(float).values
    y_total = df[total_col].astype(float).values

    game_frame = pd.DataFrame(
        {
            "game_date": df[date_col],
            "home_team": df[home_team_col],
            "away_team": df[away_team_col],
            "home_score": df["home_score"] if "home_score" in df.columns else 0.0,
            "away_score": df["away_score"] if "away_score" in df.columns else 0.0,
        }
    )

    tscv = TimeSeriesSplit(n_splits=8)
    rows: list[dict[str, float | str]] = []

    margin_preds = np.full(len(df), np.nan)
    total_preds = np.full(len(df), np.nan)
    win_preds = np.full(len(df), np.nan)

    for fold, (train_idx, test_idx) in enumerate(tscv.split(X), start=1):
        stack_margin = StackedPregameEnsemble().fit(X[train_idx], y_margin[train_idx], game_frame.iloc[train_idx])
        stack_total = StackedPregameEnsemble().fit(X[train_idx], y_total[train_idx], game_frame.iloc[train_idx])
        win_model = WinProbabilityPregameModel().fit(X[train_idx], y_win[train_idx])

        margin_preds[test_idx] = stack_margin.predict(X[test_idx], game_frame.iloc[test_idx])
        total_preds[test_idx] = stack_total.predict(X[test_idx], game_frame.iloc[test_idx])
        win_preds[test_idx] = win_model.predict_proba(X[test_idx])

        rows.append(
            {
                "model": f"stacked_fold_{fold}",
                "mae_margin": float(mean_absolute_error(y_margin[test_idx], margin_preds[test_idx])),
                "mae_total": float(mean_absolute_error(y_total[test_idx], total_preds[test_idx])),
                "brier_win": float(np.mean((win_preds[test_idx] - y_win[test_idx]) ** 2)),
            }
        )

    final_row = {
        "model": "stacked_pregame_oof",
        "mae_margin": float(mean_absolute_error(y_margin[~np.isnan(margin_preds)], margin_preds[~np.isnan(margin_preds)])),
        "mae_total": float(mean_absolute_error(y_total[~np.isnan(total_preds)], total_preds[~np.isnan(total_preds)])),
        "brier_win": float(np.mean((win_preds[~np.isnan(win_preds)] - y_win[~np.isnan(win_preds)]) ** 2)),
        "rolling_mae_margin_30": _rolling_mae(y_margin, margin_preds, 30),
        "rolling_mae_margin_60": _rolling_mae(y_margin, margin_preds, 60),
        "rolling_mae_total_30": _rolling_mae(y_total, total_preds, 30),
        "rolling_mae_total_60": _rolling_mae(y_total, total_preds, 60),
    }

    out = pd.concat([pd.DataFrame(rows), pd.DataFrame([final_row])], ignore_index=True)
    for path in [FINAL_REPORT_PATH, COMPARISON_REPORT_PATH]:
        path.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(path, index=False)

    preds = pd.DataFrame(
        {
            "game_date": df[date_col],
            "pred_margin": margin_preds,
            "pred_total": total_preds,
            "pred_home_win_prob": win_preds,
            "actual_margin": y_margin,
            "actual_total": y_total,
            "home_team": df[home_team_col],
            "away_team": df[away_team_col],
            "season": df["season"] if "season" in df.columns else df[date_col].dt.year,
            "confidence_bucket": pd.cut(np.abs(margin_preds), bins=[-np.inf, 3, 6, 9, np.inf], labels=["low", "medium", "high", "very_high"]),
        }
    )
    if "game_id" in df.columns:
        preds["game_id"] = df["game_id"]
    preds.to_csv("reports/pregame_predictions_oof.csv", index=False)

    print("Best pregame model: stacked_pregame_oof")
    print(f"Expected MAE margin: {final_row['mae_margin']:.3f}")
    print(f"Expected MAE total: {final_row['mae_total']:.3f}")
    print(f"Expected Brier win: {final_row['brier_win']:.4f}")
    return out


if __name__ == "__main__":
    run_training()
