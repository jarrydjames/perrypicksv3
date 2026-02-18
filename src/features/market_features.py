from __future__ import annotations

"""Pregame market feature builder using free/manual market line CSVs."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class MarketFeatureConfig:
    lines_path: Path = Path("data/manual/market_lines.csv")
    public_splits_path: Path = Path("data/manual/public_splits.csv")


class MarketFeatureBuilder:
    def __init__(self, config: MarketFeatureConfig | None = None):
        self.config = config or MarketFeatureConfig()

    def _load_lines(self) -> pd.DataFrame:
        if not self.config.lines_path.exists():
            return pd.DataFrame(columns=["date", "game_id", "home_team", "away_team", "spread", "total", "home_ml", "away_ml", "opening_spread", "opening_total"])
        lines = pd.read_csv(self.config.lines_path)
        lines["date"] = pd.to_datetime(lines["date"]).dt.date
        return lines

    def _load_splits(self) -> pd.DataFrame:
        if not self.config.public_splits_path.exists():
            return pd.DataFrame(columns=["date", "game_id", "home_bets_pct", "over_bets_pct"])
        splits = pd.read_csv(self.config.public_splits_path)
        splits["date"] = pd.to_datetime(splits["date"]).dt.date
        return splits

    @staticmethod
    def _implied_prob(american_odds: float) -> float:
        if american_odds is None or np.isnan(american_odds):
            return 0.5
        if american_odds < 0:
            return (-american_odds) / ((-american_odds) + 100.0)
        return 100.0 / (american_odds + 100.0)

    def build_for_game(self, game_date: str, home_team: str, away_team: str, game_id: str | None = None) -> dict[str, float]:
        game_day = pd.to_datetime(game_date).date()
        lines = self._load_lines()
        splits = self._load_splits()

        line_row = lines[
            (lines["date"] == game_day)
            & (lines["home_team"] == home_team)
            & (lines["away_team"] == away_team)
        ]
        if game_id and "game_id" in lines.columns:
            candidate = lines[(lines["date"] == game_day) & (lines["game_id"].astype(str) == str(game_id))]
            if not candidate.empty:
                line_row = candidate

        if line_row.empty:
            return {
                "market_spread_open": 0.0,
                "market_spread_close": 0.0,
                "market_total_open": 0.0,
                "market_total_close": 0.0,
                "market_spread_move": 0.0,
                "market_total_move": 0.0,
                "market_home_win_prob": 0.5,
                "market_away_win_prob": 0.5,
                "market_over_prob_proxy": 0.5,
                "public_home_bets_pct": 0.5,
                "public_over_bets_pct": 0.5,
            }

        row = line_row.iloc[-1]
        spread_close = float(row.get("spread", 0.0))
        total_close = float(row.get("total", 0.0))
        spread_open = float(row.get("opening_spread", spread_close))
        total_open = float(row.get("opening_total", total_close))

        split_row = pd.DataFrame()
        if not splits.empty:
            split_row = splits[
                (splits["date"] == game_day)
                & ((splits.get("game_id", "").astype(str) == str(row.get("game_id", ""))) if "game_id" in splits.columns else True)
            ]

        public_home = 0.5
        public_over = 0.5
        if not split_row.empty:
            public_home = float(split_row.iloc[-1].get("home_bets_pct", 50.0)) / 100.0
            public_over = float(split_row.iloc[-1].get("over_bets_pct", 50.0)) / 100.0

        home_ml = float(row.get("home_ml", np.nan))
        away_ml = float(row.get("away_ml", np.nan))
        home_prob = self._implied_prob(home_ml)
        away_prob = self._implied_prob(away_ml)
        norm = max(home_prob + away_prob, 1e-8)

        return {
            "market_spread_open": spread_open,
            "market_spread_close": spread_close,
            "market_total_open": total_open,
            "market_total_close": total_close,
            "market_spread_move": spread_close - spread_open,
            "market_total_move": total_close - total_open,
            "market_home_win_prob": home_prob / norm,
            "market_away_win_prob": away_prob / norm,
            "market_over_prob_proxy": 0.5 + np.clip((total_close - 225.0) / 60.0, -0.25, 0.25),
            "public_home_bets_pct": public_home,
            "public_over_bets_pct": public_over,
        }
