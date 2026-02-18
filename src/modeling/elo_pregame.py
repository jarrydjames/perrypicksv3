from __future__ import annotations

"""Free pregame Elo-style team strength model."""

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class EloPregameConfig:
    base_elo: float = 1500.0
    k_factor: float = 20.0
    margin_multiplier: float = 0.03
    home_court_advantage: float = 70.0


class EloPregameModel:
    def __init__(self, config: EloPregameConfig | None = None):
        self.config = config or EloPregameConfig()
        self.team_elos: dict[str, float] = {}

    def _expected(self, home_elo: float, away_elo: float) -> float:
        return 1.0 / (1.0 + 10 ** (-(home_elo - away_elo + self.config.home_court_advantage) / 400.0))

    def _team_elo(self, team: str) -> float:
        return self.team_elos.get(team, self.config.base_elo)

    def update(self, home_team: str, away_team: str, home_score: float, away_score: float) -> None:
        home_elo = self._team_elo(home_team)
        away_elo = self._team_elo(away_team)

        expected_home = self._expected(home_elo, away_elo)
        actual_home = 1.0 if home_score > away_score else 0.0
        margin = abs(home_score - away_score)
        margin_boost = 1.0 + self.config.margin_multiplier * margin
        delta = self.config.k_factor * margin_boost * (actual_home - expected_home)

        self.team_elos[home_team] = home_elo + delta
        self.team_elos[away_team] = away_elo - delta

    def fit(self, historical_games: pd.DataFrame) -> "EloPregameModel":
        ordered = historical_games.sort_values("game_date")
        for _, row in ordered.iterrows():
            self.update(
                home_team=row["home_team"],
                away_team=row["away_team"],
                home_score=float(row["home_score"]),
                away_score=float(row["away_score"]),
            )
        return self

    def features_for_matchup(self, home_team: str, away_team: str) -> dict[str, float]:
        home_elo = self._team_elo(home_team)
        away_elo = self._team_elo(away_team)
        expected_home = self._expected(home_elo, away_elo)

        return {
            "home_elo": home_elo,
            "away_elo": away_elo,
            "elo_diff": home_elo - away_elo,
            "elo_expected_home_win": expected_home,
            "elo_off_rating_proxy": (home_elo - self.config.base_elo) / 5.0,
            "elo_def_rating_proxy": -(away_elo - self.config.base_elo) / 5.0,
            "elo_net_rating_proxy": (home_elo - away_elo) / 5.0,
        }
