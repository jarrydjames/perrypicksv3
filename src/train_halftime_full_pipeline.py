"""Full halftime model training pipeline with 25-26 expansion."""

import json
import logging
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Ensure repo root is importable for `from src...` when running as a script.
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.predict_from_gameid_v2 import compute_1h_behavior_from_pbp, fetch_box, fetch_pbp_df, first_half_score

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def _safe_int(value, default: int = 0) -> int:
    try:
        if value is None or value == "":
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def _resolve_targets(df: pd.DataFrame) -> tuple[str, str]:
    """Pick best available supervised target columns for 2H modeling."""
    if {"h2_total", "h2_margin"}.issubset(df.columns):
        return "h2_total", "h2_margin"
    if {"final_total", "final_margin", "h1_total", "h1_margin"}.issubset(df.columns):
        df["h2_total"] = df["final_total"] - df["h1_total"]
        df["h2_margin"] = df["final_margin"] - df["h1_margin"]
        return "h2_total", "h2_margin"

    logger.warning("No explicit 2H targets found; using legacy synthetic fallback targets.")
    df["h2_total"] = df["h1_total"] + df["h1_total"]
    df["h2_margin"] = df["h1_margin"]
    return "h2_total", "h2_margin"


def load_halftime_baseline() -> pd.DataFrame:
    baseline = pd.read_parquet("data/processed/halftime_training_23_24_leakage_free.parquet")
    logger.info("Loaded baseline halftime dataset: %s games (23-24, 24-25)", len(baseline))
    return baseline


def load_25_26_games() -> pd.DataFrame:
    with open("data/processed/game_ids_3_seasons.json", "r") as f:
        all_games = json.load(f)

    games_25_26 = [g for g in all_games if g.get("gameId", "").startswith("00225")]
    completed_games = [g for g in games_25_26 if int(g.get("gameStatus", 0)) == 3]

    logger.info("Found %s 25-26 games", len(games_25_26))
    logger.info("Completed games: %s", len(completed_games))
    return pd.DataFrame(completed_games)


def extract_halftime_row(game_id: str) -> Dict | None:
    try:
        game = fetch_box(game_id)
        h1_home, h1_away = first_half_score(game)

        try:
            pbp = fetch_pbp_df(game_id)
            beh = compute_1h_behavior_from_pbp(pbp)
        except Exception:
            logger.warning("PBP failed for %s, using empty behavior", game_id)
            beh = {
                "h1_events": 0,
                "h1_n_2pt": 0,
                "h1_n_3pt": 0,
                "h1_n_turnover": 0,
                "h1_n_rebound": 0,
                "h1_n_foul": 0,
                "h1_n_timeout": 0,
                "h1_n_sub": 0,
            }

        home_team = game.get("homeTeam", {}) or {}
        away_team = game.get("awayTeam", {}) or {}
        final_home = _safe_int(home_team.get("score"))
        final_away = _safe_int(away_team.get("score"))

        final_total = final_home + final_away
        final_margin = final_home - final_away

        return {
            "game_id": game_id,
            "season_end_yy": 25,
            "h1_home": h1_home,
            "h1_away": h1_away,
            "h1_total": h1_home + h1_away,
            "h1_margin": h1_home - h1_away,
            "final_home": final_home,
            "final_away": final_away,
            "final_total": final_total,
            "final_margin": final_margin,
            "h2_total": final_total - (h1_home + h1_away),
            "h2_margin": final_margin - (h1_home - h1_away),
            **beh,
        }
    except Exception as e:
        logger.error("Error extracting %s: %s", game_id, e)
        return None


def build_halftime_dataset_expanded() -> pd.DataFrame:
    logger.info("=" * 70)
    logger.info("HALFTIME DATASET EXPANSION")
    logger.info("=" * 70)

    baseline = load_halftime_baseline()
    games_25_26 = load_25_26_games()

    if len(games_25_26) == 0:
        logger.warning("No 25-26 games available, returning baseline only")
        return baseline

    logger.info("Extracting features for %s 25-26 games...", len(games_25_26))
    rows = []
    game_ids = games_25_26["gameId"].tolist()
    for i, gid in enumerate(game_ids, 1):
        row = extract_halftime_row(gid)
        if row:
            rows.append(row)
        if i % 100 == 0:
            logger.info("Processed %s/%s (%.1f%%)", i, len(game_ids), i / len(game_ids) * 100)

    df_new = pd.DataFrame(rows)
    df_expanded = pd.concat([baseline, df_new], ignore_index=True).sort_values("game_id").reset_index(drop=True)

    _resolve_targets(df_expanded)

    logger.info("Combined dataset: %s games", len(df_expanded))
    logger.info("Baseline: %s games", len(baseline))
    logger.info("New (25-26): %s games", len(df_new))

    output_path = Path("data/processed/halftime_training_full_3_seasons.parquet")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_expanded.to_parquet(output_path, index=False)
    logger.info("Saved expanded dataset -> %s", output_path)
    return df_expanded


def _fit_and_score(model, X_train, y_train, X_test, y_test) -> Dict:
    model.fit(X_train, y_train)
    pred_train = model.predict(X_train)
    pred_test = model.predict(X_test)
    return {
        "model": model,
        "metrics_train": {
            "mae": mean_absolute_error(y_train, pred_train),
            "rmse": np.sqrt(mean_squared_error(y_train, pred_train)),
            "r2": r2_score(y_train, pred_train),
        },
        "metrics_test": {
            "mae": mean_absolute_error(y_test, pred_test),
            "rmse": np.sqrt(mean_squared_error(y_test, pred_test)),
            "r2": r2_score(y_test, pred_test),
        },
    }


def train_ridge(X_train, y_train, X_test, y_test) -> Dict:
    return _fit_and_score(Ridge(alpha=2.0, random_state=42, solver="auto"), X_train, y_train, X_test, y_test)


def train_rf(X_train, y_train, X_test, y_test) -> Dict:
    return _fit_and_score(
        RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1),
        X_train,
        y_train,
        X_test,
        y_test,
    )


def train_gbt(X_train, y_train, X_test, y_test) -> Dict:
    return _fit_and_score(
        HistGradientBoostingRegressor(max_iter=100, max_depth=5, learning_rate=0.1, random_state=42),
        X_train,
        y_train,
        X_test,
        y_test,
    )


def walk_forward_cv(df: pd.DataFrame, min_train_size: int = 500, test_size: int = 200, step_size: int = 200) -> pd.DataFrame:
    logger.info("=" * 70)
    logger.info("WALK-FORWARD TEMPORAL CROSS-VALIDATION")
    logger.info("=" * 70)

    df_sorted = df.sort_values("game_id").reset_index(drop=True)
    total_target_col, margin_target_col = _resolve_targets(df_sorted)

    results = []
    train_end = min_train_size
    fold_num = 0

    while train_end + test_size <= len(df_sorted):
        test_start = train_end
        test_end = test_start + test_size

        train_df = df_sorted.iloc[:train_end]
        test_df = df_sorted.iloc[test_start:test_end]

        feature_cols = [c for c in train_df.columns if c.startswith("h1_")]
        X_train = train_df[feature_cols].values
        X_test = test_df[feature_cols].values

        y_train_total = train_df[total_target_col].values
        y_test_total = test_df[total_target_col].values
        y_train_margin = train_df[margin_target_col].values
        y_test_margin = test_df[margin_target_col].values

        ridge_total = train_ridge(X_train, y_train_total, X_test, y_test_total)
        ridge_margin = train_ridge(X_train, y_train_margin, X_test, y_test_margin)
        rf_total = train_rf(X_train, y_train_total, X_test, y_test_total)
        rf_margin = train_rf(X_train, y_train_margin, X_test, y_test_margin)
        gbt_total = train_gbt(X_train, y_train_total, X_test, y_test_total)
        gbt_margin = train_gbt(X_train, y_train_margin, X_test, y_test_margin)

        results.append(
            {
                "fold": fold_num,
                "train_end": train_end,
                "test_start": test_start,
                "test_end": test_end,
                "test_size": len(test_df),
                "ridge_total_mae": ridge_total["metrics_test"]["mae"],
                "rf_total_mae": rf_total["metrics_test"]["mae"],
                "gbt_total_mae": gbt_total["metrics_test"]["mae"],
                "ridge_margin_mae": ridge_margin["metrics_test"]["mae"],
                "rf_margin_mae": rf_margin["metrics_test"]["mae"],
                "gbt_margin_mae": gbt_margin["metrics_test"]["mae"],
            }
        )

        logger.info("Fold %s: Train=%s, Test=%s", fold_num, len(train_df), len(test_df))
        logger.info("  Total MAE - Ridge: %.3f RF: %.3f GBT: %.3f", ridge_total["metrics_test"]["mae"], rf_total["metrics_test"]["mae"], gbt_total["metrics_test"]["mae"])

        train_end += step_size
        fold_num += 1

    results_df = pd.DataFrame(results)
    if not results_df.empty:
        logger.info("Completed %s folds", len(results_df))
        logger.info("Overall Total MAE - Ridge: %.3f RF: %.3f GBT: %.3f", results_df["ridge_total_mae"].mean(), results_df["rf_total_mae"].mean(), results_df["gbt_total_mae"].mean())
    else:
        logger.warning("No folds produced. Dataset too small for current CV parameters.")
    return results_df


def main():
    import joblib

    df = build_halftime_dataset_expanded()
    total_target_col, _ = _resolve_targets(df)

    cv_results = walk_forward_cv(df)

    champion = "ridge"
    logger.info("=" * 70)
    logger.info("CHAMPION MODEL: %s", champion.upper())
    logger.info("=" * 70)

    feature_cols = [c for c in df.columns if c.startswith("h1_")]
    if champion == "ridge":
        model_path = Path("models/team_2h_total.joblib")
        model_path.parent.mkdir(parents=True, exist_ok=True)
        model = Ridge(alpha=2.0, random_state=42)
        model.fit(df[feature_cols].values, df[total_target_col].values)
        joblib.dump({"model": model, "features": feature_cols, "model_name": "Ridge", "target": total_target_col}, model_path)
        logger.info("Saved champion model -> %s", model_path)

    cv_path = Path("data/processed/halftime_cv_results_full.parquet")
    cv_path.parent.mkdir(parents=True, exist_ok=True)
    cv_results.to_parquet(cv_path, index=False)
    logger.info("Saved CV results -> %s", cv_path)

    logger.info("=" * 70)
    logger.info("HALFTIME TRAINING COMPLETE")
    logger.info("=" * 70)
    logger.info("Final dataset size: %s games", len(df))


if __name__ == "__main__":
    main()
