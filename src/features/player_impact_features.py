from __future__ import annotations

"""Pregame lineup and availability features from manual injury + player metric data."""

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class PlayerImpactConfig:
    injuries_path: Path = Path("data/manual/injuries.csv")
    player_ratings_path: Path = Path("data/manual/player_ratings.csv")
    default_status_weight: float = 0.5


_STATUS_IMPACT = {
    "out": 1.0,
    "doubtful": 0.75,
    "questionable": 0.4,
    "probable": 0.15,
    "available": 0.0,
}


class PlayerImpactFeatureBuilder:
    def __init__(self, config: PlayerImpactConfig | None = None):
        self.config = config or PlayerImpactConfig()

    def _load_or_empty(self, path: Path, columns: list[str]) -> pd.DataFrame:
        if not path.exists():
            return pd.DataFrame(columns=columns)
        return pd.read_csv(path)

    def _status_to_weight(self, status: str) -> float:
        if not isinstance(status, str):
            return self.config.default_status_weight
        return _STATUS_IMPACT.get(status.strip().lower(), self.config.default_status_weight)

    def build_for_game(self, game_date: str, home_team: str, away_team: str) -> dict[str, float]:
        injuries = self._load_or_empty(
            self.config.injuries_path,
            ["date", "team", "player", "status", "expected_minutes", "is_starter"],
        )
        ratings = self._load_or_empty(
            self.config.player_ratings_path,
            ["team", "player", "impact_metric", "minutes_per_game", "is_starter"],
        )

        if injuries.empty:
            return {
                "home_starters_out": 0.0,
                "away_starters_out": 0.0,
                "home_minutes_missing": 0.0,
                "away_minutes_missing": 0.0,
                "home_lineup_strength": 0.0,
                "away_lineup_strength": 0.0,
                "lineup_strength_diff": 0.0,
                "injury_impact_diff": 0.0,
            }

        injuries["date"] = pd.to_datetime(injuries["date"]).dt.date
        injuries_today = injuries[injuries["date"] == pd.to_datetime(game_date).date()].copy()
        if injuries_today.empty:
            injuries_today = injuries.copy()

        ratings = ratings.copy()
        if not ratings.empty:
            ratings["impact_metric"] = ratings["impact_metric"].fillna(0.0)
            ratings["minutes_per_game"] = ratings["minutes_per_game"].fillna(0.0)

        merged = injuries_today.merge(ratings, on=["team", "player"], how="left", suffixes=("", "_rating"))
        merged["expected_minutes"] = merged.get("expected_minutes", 0.0).fillna(0.0)
        merged["impact_metric"] = merged.get("impact_metric", 0.0).fillna(0.0)
        merged["status_weight"] = merged["status"].map(self._status_to_weight)
        merged["minutes_lost"] = merged["expected_minutes"] * merged["status_weight"]
        merged["impact_lost"] = merged["minutes_lost"] * merged["impact_metric"]
        merged["is_starter"] = merged.get("is_starter", 0).fillna(0).astype(float)

        def summarize(team: str) -> dict[str, float]:
            team_rows = merged[merged["team"] == team]
            starters_out = float(((team_rows["is_starter"] > 0.5) & (team_rows["status_weight"] >= 0.75)).sum())
            minutes_missing = float(team_rows["minutes_lost"].sum())
            injury_impact = float(team_rows["impact_lost"].sum())

            ratings_rows = ratings[ratings["team"] == team]
            lineup_strength = float((ratings_rows["impact_metric"] * ratings_rows["minutes_per_game"]).sum())
            return {
                "starters_out": starters_out,
                "minutes_missing": minutes_missing,
                "injury_impact": injury_impact,
                "lineup_strength": lineup_strength,
            }

        home = summarize(home_team)
        away = summarize(away_team)

        return {
            "home_starters_out": home["starters_out"],
            "away_starters_out": away["starters_out"],
            "home_minutes_missing": home["minutes_missing"],
            "away_minutes_missing": away["minutes_missing"],
            "home_lineup_strength": home["lineup_strength"],
            "away_lineup_strength": away["lineup_strength"],
            "lineup_strength_diff": home["lineup_strength"] - away["lineup_strength"],
            "injury_impact_diff": home["injury_impact"] - away["injury_impact"],
        }
