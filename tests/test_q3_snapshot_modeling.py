from __future__ import annotations

import pandas as pd

from src.build_dataset_q3 import _clock_to_seconds_remaining, _q3_five_minute_snapshot
import src.predict_from_gameid_v3_runtime as runtime


def test_clock_parser_supports_iso_and_mmss() -> None:
    assert _clock_to_seconds_remaining("PT5M00.00S") == 300.0
    assert _clock_to_seconds_remaining("4:30") == 270.0
    assert _clock_to_seconds_remaining(None) is None


def test_q3_snapshot_uses_first_event_at_or_below_5_minutes() -> None:
    pbp = pd.DataFrame(
        [
            {"period": 3, "clock": "PT5M20.00S", "scoreHome": 80, "scoreAway": 79},
            {"period": 3, "clock": "PT4M59.00S", "scoreHome": 82, "scoreAway": 79},
            {"period": 3, "clock": "PT4M40.00S", "scoreHome": 82, "scoreAway": 81},
        ]
    )

    home, away = _q3_five_minute_snapshot(pbp)

    assert home == 82.0
    assert away == 79.0


def test_predict_q3_converts_remaining_predictions_to_final(monkeypatch) -> None:
    monkeypatch.setattr(runtime, "extract_game_id", lambda game_input: "0022500001")

    monkeypatch.setattr("src.data.game_data.fetch_game_by_id", lambda gid: {
            "period": 3,
            "gameClock": "PT4M30.00S",
            "homeTeam": {"teamTricode": "LAL", "periods": [{"period": 1, "score": 30}, {"period": 2, "score": 25}, {"period": 3, "score": 20}]},
            "awayTeam": {"teamTricode": "BOS", "periods": [{"period": 1, "score": 28}, {"period": 2, "score": 22}, {"period": 3, "score": 24}]},
        })

    monkeypatch.setattr(runtime, "fetch_pbp_df", lambda gid: pd.DataFrame([{"period": 3, "actionType": "2pt", "clock": "PT4M30.00S"}]))
    monkeypatch.setattr(runtime, "team_totals_from_box_team", lambda team: {"fga": 80, "fgm": 40, "fta": 20, "tov": 10, "orb": 8, "fg3a": 30, "fg3m": 12})
    monkeypatch.setattr(runtime, "add_rate_features", lambda prefix, team, opp: {f"{prefix}_efg": 0.5})

    class FakePred:
        margin_mean = 3.0
        total_mean = 48.0
        margin_q10 = -2.0
        margin_q90 = 8.0
        total_q10 = 42.0
        total_q90 = 54.0
        home_win_prob = 0.6
        margin_sd = 5.0
        total_sd = 7.0
        model_name = "fake"
        feature_version = "v"

    class FakeQ3Model:
        def predict(self, **kwargs):
            return FakePred()

    monkeypatch.setattr(runtime, "get_q3_model", lambda: FakeQ3Model())

    result = runtime.predict_from_game_id("0022500001", fetch_odds=False)

    # Snapshot score from periods above: home=75, away=74
    assert result["remaining_total"] == 48.0
    assert result["remaining_margin"] == 3.0
    assert result["total"] == 197.0
    assert result["margin"] == 4.0
    assert result["total_q10"] == 191.0
    assert result["total_q90"] == 203.0
