from __future__ import annotations

"""Run market-aware pregame betting backtest and produce comparison report."""

from pathlib import Path

import pandas as pd

from backtest_pregame import evaluate_spread_bets, evaluate_totals_bets, summarize_backtest

PREDS_PATH = Path("reports/pregame_predictions_oof.csv")
MARKET_PATH = Path("data/manual/market_lines.csv")
TRAINING_REPORT_PATH = Path("reports/final_pregame_model_comparison.csv")
FINAL_OUT_PATH = Path("reports/final_pregame_model_comparison.csv")
COMPARE_OUT_PATH = Path("reports/pregame_model_comparison.csv")
SEASON_OUT_PATH = Path("reports/backtest/pregame_backtest_by_season.csv")
MONTH_OUT_PATH = Path("reports/backtest/pregame_backtest_by_month.csv")
CONF_OUT_PATH = Path("reports/backtest/pregame_backtest_by_confidence.csv")


def _ensure_market_defaults(frame: pd.DataFrame) -> pd.DataFrame:
    defaults = {
        "market_spread_open": 0.0,
        "market_spread_close": 0.0,
        "market_total_open": 0.0,
        "market_total_close": 0.0,
    }
    for col, val in defaults.items():
        if col not in frame.columns:
            frame[col] = val
        frame[col] = frame[col].fillna(val)
    return frame


def _single_summary(summary: pd.DataFrame) -> dict[str, float]:
    if summary.empty:
        return {"roi": 0.0, "clv": 0.0, "sharpe": 0.0, "max_drawdown": 0.0, "hit_rate": 0.0, "bets": 0.0}
    row = summary.iloc[0]
    return {
        "roi": float(row.get("roi", 0.0)),
        "clv": float(row.get("avg_clv", 0.0)),
        "sharpe": float(row.get("sharpe", 0.0)),
        "max_drawdown": float(row.get("max_drawdown", 0.0)),
        "hit_rate": float(row.get("hit_rate", 0.0)),
        "bets": float(row.get("bets", 0.0)),
    }


def _build_final_report(spread_summary: pd.DataFrame, totals_summary: pd.DataFrame) -> pd.DataFrame:
    spread = _single_summary(spread_summary)
    totals = _single_summary(totals_summary)

    training_row: dict[str, float | str] = {
        "model": "stacked_pregame_oof",
        "mae_margin": float("nan"),
        "mae_total": float("nan"),
        "brier_win": float("nan"),
    }

    if TRAINING_REPORT_PATH.exists():
        prior = pd.read_csv(TRAINING_REPORT_PATH)
        if not prior.empty:
            preferred = prior[prior.get("model", "") == "stacked_pregame_oof"]
            pick = preferred.iloc[0] if not preferred.empty else prior.iloc[-1]
            for col in ["model", "mae_margin", "mae_total", "brier_win", "rolling_mae_margin_30", "rolling_mae_margin_60", "rolling_mae_total_30", "rolling_mae_total_60"]:
                if col in prior.columns:
                    training_row[col] = pick[col]

    final_row = {
        **training_row,
        "spread_roi": spread["roi"],
        "spread_clv": spread["clv"],
        "spread_hit_rate": spread["hit_rate"],
        "spread_bets": spread["bets"],
        "spread_sharpe": spread["sharpe"],
        "spread_max_drawdown": spread["max_drawdown"],
        "totals_roi": totals["roi"],
        "totals_clv": totals["clv"],
        "totals_hit_rate": totals["hit_rate"],
        "totals_bets": totals["bets"],
        "totals_sharpe": totals["sharpe"],
        "totals_max_drawdown": totals["max_drawdown"],
        "overall_roi": (spread["roi"] + totals["roi"]) / 2.0,
        "overall_clv": (spread["clv"] + totals["clv"]) / 2.0,
    }
    return pd.DataFrame([final_row])


def run_backtest() -> pd.DataFrame:
    if not PREDS_PATH.exists():
        raise FileNotFoundError(f"Missing predictions file: {PREDS_PATH}. Run train_pregame_models.py first.")

    preds = pd.read_csv(PREDS_PATH)
    date_col = "game_date" if "game_date" in preds.columns else "date"
    preds[date_col] = pd.to_datetime(preds[date_col]).dt.normalize()

    if MARKET_PATH.exists():
        market = pd.read_csv(MARKET_PATH)
        market["date"] = pd.to_datetime(market["date"]).dt.normalize()
        joined = preds.merge(
            market,
            left_on=[date_col, "home_team", "away_team"],
            right_on=["date", "home_team", "away_team"],
            how="left",
        )
    else:
        joined = preds.copy()

    joined = _ensure_market_defaults(joined)
    joined["game_date"] = joined[date_col]

    spread_bets = evaluate_spread_bets(joined)
    totals_bets = evaluate_totals_bets(joined)

    spread_summary = summarize_backtest(spread_bets)
    totals_summary = summarize_backtest(totals_bets)

    # Stability slices
    all_bets = pd.concat([spread_bets.assign(market="spread"), totals_bets.assign(market="totals")], ignore_index=True)
    season_summary = summarize_backtest(all_bets, groupby="season")
    month_summary = summarize_backtest(all_bets, groupby="month")
    conf_summary = summarize_backtest(all_bets, groupby="confidence_bucket") if "confidence_bucket" in all_bets.columns else pd.DataFrame()

    for path, frame in [(SEASON_OUT_PATH, season_summary), (MONTH_OUT_PATH, month_summary), (CONF_OUT_PATH, conf_summary)]:
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(path, index=False)

    final_report = _build_final_report(spread_summary, totals_summary)
    for out_path in [FINAL_OUT_PATH, COMPARE_OUT_PATH]:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        final_report.to_csv(out_path, index=False)

    print("Best pregame model: stacked_pregame_oof")
    print(f"Expected ROI: {float(final_report['overall_roi'].iloc[0]):.4f}")
    print(f"CLV: {float(final_report['overall_clv'].iloc[0]):.4f}")
    print("Stability: tracked via season/month/confidence reports + drawdown/sharpe")
    print("Recommended production model: stacked pregame ensemble + calibrated win-probability")
    return final_report


if __name__ == "__main__":
    run_backtest()
