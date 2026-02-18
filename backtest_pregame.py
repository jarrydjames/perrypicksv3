from __future__ import annotations

"""Pregame market-aware backtesting utilities."""

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class BetResult:
    stake: float
    profit: float
    clv: float
    won: bool


def american_to_decimal(odds: float) -> float:
    if odds < 0:
        return 1.0 + 100.0 / abs(odds)
    return 1.0 + odds / 100.0


def compute_roi(profit_series: pd.Series, stake_series: pd.Series) -> float:
    total_staked = float(stake_series.sum())
    if total_staked <= 0:
        return 0.0
    return float(profit_series.sum()) / total_staked


def compute_clv(pred_line: pd.Series, close_line: pd.Series, side: pd.Series) -> pd.Series:
    """Positive CLV means our price was better than close for chosen side."""
    side = side.astype(str).str.lower()
    signed = np.where(side == "home", close_line - pred_line, pred_line - close_line)
    return pd.Series(signed, index=pred_line.index, dtype=float)


def _drawdown(equity_curve: pd.Series) -> float:
    peak = equity_curve.cummax()
    dd = (equity_curve - peak) / peak.replace(0, np.nan)
    return float(dd.min()) if not dd.empty else 0.0


def evaluate_spread_bets(df: pd.DataFrame, edge_threshold: float = 1.5, stake: float = 1.0) -> pd.DataFrame:
    data = df.copy()
    # market spread is expected in home-team points (negative => home favored)
    data["model_edge"] = data["pred_margin"] + data["market_spread_close"]
    data["bet_side"] = np.where(data["model_edge"] > edge_threshold, "home", np.where(data["model_edge"] < -edge_threshold, "away", "none"))
    data = data[data["bet_side"] != "none"].copy()
    if data.empty:
        return data

    data["stake"] = stake
    # Cover checks are done using margin + spread from the perspective of the selected side
    home_cover_delta = data["actual_margin"] + data["market_spread_close"]
    away_cover_delta = -home_cover_delta
    data["push"] = np.where(data["bet_side"] == "home", home_cover_delta == 0, away_cover_delta == 0)
    data["won"] = np.where(data["bet_side"] == "home", home_cover_delta > 0, away_cover_delta > 0)

    payout = american_to_decimal(-110)
    data["profit"] = np.where(data["push"], 0.0, np.where(data["won"], stake * (payout - 1.0), -stake))
    data["clv"] = compute_clv(data["market_spread_open"], data["market_spread_close"], data["bet_side"])
    return data


def evaluate_totals_bets(df: pd.DataFrame, edge_threshold: float = 1.5, stake: float = 1.0) -> pd.DataFrame:
    data = df.copy()
    data["model_edge"] = data["pred_total"] - data["market_total_close"]
    data["bet_side"] = np.where(data["model_edge"] > edge_threshold, "over", np.where(data["model_edge"] < -edge_threshold, "under", "none"))
    data = data[data["bet_side"] != "none"].copy()
    if data.empty:
        return data

    data["stake"] = stake
    total_delta = data["actual_total"] - data["market_total_close"]
    data["push"] = total_delta == 0
    data["won"] = np.where(data["bet_side"] == "over", total_delta > 0, total_delta < 0)

    payout = american_to_decimal(-110)
    data["profit"] = np.where(data["push"], 0.0, np.where(data["won"], stake * (payout - 1.0), -stake))
    data["clv"] = np.where(data["bet_side"] == "over", data["market_total_close"] - data["market_total_open"], data["market_total_open"] - data["market_total_close"])
    return data


def summarize_backtest(bets: pd.DataFrame, groupby: str | None = None) -> pd.DataFrame:
    if bets.empty:
        return pd.DataFrame()

    data = bets.copy()
    data["game_date"] = pd.to_datetime(data["game_date"])
    data["season"] = data.get("season", data["game_date"].dt.year)
    data["month"] = data["game_date"].dt.to_period("M").astype(str)

    by = [groupby] if groupby else []
    grouped = data.groupby(by) if by else [("all", data)]

    rows = []
    for key, frame in grouped:
        profits = frame["profit"].astype(float)
        stakes = frame["stake"].astype(float)
        equity = profits.cumsum() + 100.0

        rows.append(
            {
                groupby or "bucket": key,
                "bets": int(len(frame)),
                "hit_rate": float(frame["won"].mean()),
                "roi": compute_roi(profits, stakes),
                "avg_clv": float(frame["clv"].mean()),
                "sharpe": float(profits.mean() / (profits.std(ddof=1) + 1e-8) * np.sqrt(252)),
                "max_drawdown": _drawdown(equity),
            }
        )

    return pd.DataFrame(rows)
