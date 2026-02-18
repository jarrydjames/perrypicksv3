from __future__ import annotations

"""Pregame advanced team feature engineering utilities.

All features are computed from historical pregame-available box score summaries.
"""

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class TeamAdvancedFeatureConfig:
    lookback_games: int = 15
    min_games: int = 5


class AdvancedTeamFeatureBuilder:
    """Build offensive/defensive efficiency features from free box score data."""

    def __init__(self, config: TeamAdvancedFeatureConfig | None = None):
        self.config = config or TeamAdvancedFeatureConfig()

    @staticmethod
    def _safe_div(numerator: float, denominator: float) -> float:
        return float(numerator / denominator) if denominator not in (0, 0.0) else 0.0

    @staticmethod
    def _possessions(frame: pd.DataFrame, prefix: str) -> pd.Series:
        fga = frame.get(f"{prefix}_fga", 0.0)
        oreb = frame.get(f"{prefix}_oreb", 0.0)
        tov = frame.get(f"{prefix}_tov", 0.0)
        fta = frame.get(f"{prefix}_fta", 0.0)
        return 0.5 * (fga + 0.4 * fta - 1.07 * oreb + tov)

    def _aggregate_team_window(self, frame: pd.DataFrame, team: str, as_home: bool) -> dict[str, float]:
        if frame.empty:
            return {
                "off_rating": 0.0,
                "def_rating": 0.0,
                "net_rating": 0.0,
                "pace": 0.0,
                "true_shooting_pct": 0.0,
                "turnover_rate": 0.0,
                "rebound_rate": 0.0,
            }

        own = "home" if as_home else "away"
        opp = "away" if as_home else "home"

        poss = self._possessions(frame, own)
        opp_poss = self._possessions(frame, opp)
        avg_poss = (poss + opp_poss).replace(0, np.nan) / 2.0

        points_for = frame[f"{own}_score"].sum()
        points_against = frame[f"{opp}_score"].sum()
        possessions = float(np.nansum(avg_poss.values))

        fga = frame.get(f"{own}_fga", pd.Series(0.0, index=frame.index)).sum()
        fta = frame.get(f"{own}_fta", pd.Series(0.0, index=frame.index)).sum()
        tov = frame.get(f"{own}_tov", pd.Series(0.0, index=frame.index)).sum()
        oreb = frame.get(f"{own}_oreb", pd.Series(0.0, index=frame.index)).sum()
        dreb = frame.get(f"{own}_dreb", pd.Series(0.0, index=frame.index)).sum()
        opp_oreb = frame.get(f"{opp}_oreb", pd.Series(0.0, index=frame.index)).sum()

        off_rating = 100.0 * self._safe_div(points_for, possessions)
        def_rating = 100.0 * self._safe_div(points_against, possessions)
        pace = 48.0 * self._safe_div(possessions, max(len(frame), 1))
        ts = self._safe_div(points_for, 2.0 * (fga + 0.44 * fta))
        tov_rate = self._safe_div(tov, possessions)
        rebound_rate = self._safe_div(oreb + dreb, oreb + dreb + opp_oreb)

        return {
            "off_rating": off_rating,
            "def_rating": def_rating,
            "net_rating": off_rating - def_rating,
            "pace": pace,
            "true_shooting_pct": ts,
            "turnover_rate": tov_rate,
            "rebound_rate": rebound_rate,
        }

    def _team_game_window(self, games: pd.DataFrame, team: str, game_date: pd.Timestamp) -> pd.DataFrame:
        team_games = games[
            ((games["home_team"] == team) | (games["away_team"] == team))
            & (games["game_date"] < game_date)
        ].sort_values("game_date")
        return team_games.tail(self.config.lookback_games)

    def build_for_game(self, games: pd.DataFrame, game_date: str | pd.Timestamp, home_team: str, away_team: str) -> dict[str, float]:
        game_date = pd.Timestamp(game_date)
        home_window = self._team_game_window(games, home_team, game_date)
        away_window = self._team_game_window(games, away_team, game_date)

        home_home = self._aggregate_team_window(home_window[home_window["home_team"] == home_team], home_team, as_home=True)
        home_away = self._aggregate_team_window(home_window[home_window["away_team"] == home_team], home_team, as_home=False)
        away_home = self._aggregate_team_window(away_window[away_window["home_team"] == away_team], away_team, as_home=True)
        away_away = self._aggregate_team_window(away_window[away_window["away_team"] == away_team], away_team, as_home=False)

        features: dict[str, float] = {}
        metric_names: Iterable[str] = home_home.keys()
        for metric in metric_names:
            home_value = float(np.mean([home_home[metric], home_away[metric]]))
            away_value = float(np.mean([away_home[metric], away_away[metric]]))
            features[f"home_{metric}"] = home_value
            features[f"away_{metric}"] = away_value
            features[f"{metric}_diff"] = home_value - away_value

        features["home_games_in_window"] = float(len(home_window))
        features["away_games_in_window"] = float(len(away_window))
        features["advanced_features_valid"] = float(
            len(home_window) >= self.config.min_games and len(away_window) >= self.config.min_games
        )
        return features
