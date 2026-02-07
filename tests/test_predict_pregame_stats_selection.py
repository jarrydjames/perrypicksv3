from __future__ import annotations

from types import SimpleNamespace

import pandas as pd

import src.predict_pregame as pregame


def test_fetch_team_stats_selects_requested_team_id(monkeypatch) -> None:
    df = pd.DataFrame(
        [
            {"TEAM_ID": 111, "OFF_RATING": 105.0},
            {"TEAM_ID": 222, "OFF_RATING": 118.0},
        ]
    )

    class FakeEndpoint:
        def __init__(self, **_: object) -> None:
            pass

        def get_data_frames(self):
            return [df]

    monkeypatch.setattr(pregame, "leaguedashteamstats", SimpleNamespace(LeagueDashTeamStats=FakeEndpoint))

    row_222, season_222 = pregame.fetch_team_stats(222, ["2025-26"])
    row_111, season_111 = pregame.fetch_team_stats(111, ["2025-26"])

    assert row_222 is not None
    assert row_111 is not None
    assert season_222 == "2025-26"
    assert season_111 == "2025-26"
    assert float(row_222["OFF_RATING"]) == 118.0
    assert float(row_111["OFF_RATING"]) == 105.0


def test_fetch_team_stats_does_not_fallback_to_first_row_on_multirow_payload(monkeypatch) -> None:
    df = pd.DataFrame(
        [
            {"OFF_RATING": 101.0},
            {"OFF_RATING": 119.0},
        ]
    )

    class FakeEndpoint:
        def __init__(self, **_: object) -> None:
            pass

        def get_data_frames(self):
            return [df]

    monkeypatch.setattr(pregame, "leaguedashteamstats", SimpleNamespace(LeagueDashTeamStats=FakeEndpoint))

    row, season = pregame.fetch_team_stats(222, ["2025-26"])

    assert row is None
    assert season is None


def test_infer_season_from_game_id() -> None:
    assert pregame.infer_season_from_game_id("0022500742") == "2025-26"
    assert pregame.infer_season_from_game_id("0012600001") == "2026-27"
    assert pregame.infer_season_from_game_id("bad") is None


def test_predict_from_game_id_uses_inferred_season_and_scheduled_datetime(monkeypatch) -> None:
    calls: dict[str, object] = {}

    def fake_fetch_team_stats(team_id: int, seasons):
        calls.setdefault("seasons", []).append((team_id, seasons))
        return pd.Series(
            {
                "OFF_RATING": 111.0,
                "DEF_RATING": 109.0,
                "PACE": 99.0,
                "EFG_PCT": 0.53,
                "FTA_RATE": 0.23,
                "TOV_PCT": 0.13,
                "OREB_PCT": 0.27,
                "GP": 10,
                "W": 6,
            }
        ), seasons[0]

    def fake_extract_core_features(home_stats, away_stats, home_team_id, away_team_id, game_date, **kwargs):
        calls["game_date"] = game_date
        return {"home_off_rating": 111.0, "away_off_rating": 110.0}

    class FakeModel:
        def predict(self, *, features, game_id):
            return SimpleNamespace(
                margin_mean=1.0,
                total_mean=200.0,
                margin_q10=-5.0,
                margin_q90=7.0,
                total_q10=190.0,
                total_q90=210.0,
                home_win_prob=0.55,
                margin_sd=4.0,
                total_sd=6.0,
                model_name="fake",
                feature_version="v",
            )

    monkeypatch.setattr(pregame, "fetch_team_stats", fake_fetch_team_stats)
    monkeypatch.setattr(pregame, "extract_core_features", fake_extract_core_features)
    monkeypatch.setattr(pregame, "get_pregame_model", lambda: FakeModel())
    monkeypatch.setattr(
        pregame,
        "build_data_freshness_context",
        lambda *args, **kwargs: {"is_stale": False, "force_historical_stats": False},
    )

    result = pregame.predict_from_game_id(
        game_id="0022500742",
        home_team="DET",
        away_team="NYK",
        game_datetime="2026-02-05T23:30:00Z",
    )

    assert result["status"] == "success"
    assert calls["seasons"] == [
        (pregame.TEAM_IDS["DET"], ["2025-26", "2024-25"]),
        (pregame.TEAM_IDS["NYK"], ["2025-26", "2024-25"]),
    ]
    assert str(calls["game_date"]) == "2026-02-05 23:30:00+00:00"


def test_predict_from_game_id_sets_warning_when_defaults_used(monkeypatch) -> None:
    def fake_fetch_team_stats(team_id: int, seasons):
        return None, None

    def fake_extract_core_features(home_stats, away_stats, home_team_id, away_team_id, game_date, **kwargs):
        return {
            "home_off_rating": 110.0,
            "away_off_rating": 110.0,
            "home_def_rating": 110.0,
            "away_def_rating": 110.0,
            "home_pace": 100.0,
            "away_pace": 100.0,
            "off_rating_diff": 0.0,
            "def_rating_diff": 0.0,
            "pace_diff": 0.0,
        }

    class FakeModel:
        def predict(self, *, features, game_id):
            return SimpleNamespace(
                margin_mean=0.0,
                total_mean=181.5,
                margin_q10=-5.0,
                margin_q90=5.0,
                total_q10=170.0,
                total_q90=193.0,
                home_win_prob=0.5,
                margin_sd=4.0,
                total_sd=6.0,
                model_name="fake",
                feature_version="v",
            )

    monkeypatch.setattr(pregame, "fetch_team_stats", fake_fetch_team_stats)
    monkeypatch.setattr(pregame, "extract_core_features", fake_extract_core_features)
    monkeypatch.setattr(pregame, "get_pregame_model", lambda: FakeModel())
    monkeypatch.setattr(
        pregame,
        "build_data_freshness_context",
        lambda *args, **kwargs: {"is_stale": False, "force_historical_stats": False},
    )

    result = pregame.predict_from_game_id(
        game_id="0022500742",
        home_team="DET",
        away_team="NYK",
        game_datetime="2026-02-05T23:30:00Z",
    )

    assert result["status"] == "warning"
    assert result["data_source"] == {
        "home_stats_season": "DEFAULTS",
        "away_stats_season": "DEFAULTS",
    }
    assert "data_warning" in result


def test_predict_from_game_id_sets_warning_when_data_stale(monkeypatch) -> None:
    def fake_fetch_team_stats(team_id: int, seasons):
        return pd.Series({
            "OFF_RATING": 111.0,
            "DEF_RATING": 109.0,
            "PACE": 99.0,
            "EFG_PCT": 0.53,
            "FTA_RATE": 0.23,
            "TOV_PCT": 0.13,
            "OREB_PCT": 0.27,
            "GP": 10,
            "W": 6,
        }), seasons[0]

    def fake_extract_core_features(home_stats, away_stats, home_team_id, away_team_id, game_date, **kwargs):
        assert kwargs["force_home_historical"] is True
        assert kwargs["force_away_historical"] is True
        return {"home_off_rating": 111.0, "away_off_rating": 109.0, "off_rating_diff": 2.0}

    class FakeModel:
        def predict(self, *, features, game_id):
            return SimpleNamespace(
                margin_mean=1.0,
                total_mean=200.0,
                margin_q10=-5.0,
                margin_q90=7.0,
                total_q10=190.0,
                total_q90=210.0,
                home_win_prob=0.55,
                margin_sd=4.0,
                total_sd=6.0,
                model_name="fake",
                feature_version="v",
            )

    monkeypatch.setattr(pregame, "fetch_team_stats", fake_fetch_team_stats)
    monkeypatch.setattr(pregame, "extract_core_features", fake_extract_core_features)
    monkeypatch.setattr(pregame, "get_pregame_model", lambda: FakeModel())
    monkeypatch.setattr(
        pregame,
        "build_data_freshness_context",
        lambda *args, **kwargs: {
            "is_stale": True,
            "force_historical_stats": True,
            "stale_reasons": ["historical data is 6 days old"],
        },
    )

    result = pregame.predict_from_game_id(
        game_id="0022500742",
        home_team="DET",
        away_team="NYK",
        game_datetime="2026-02-05T23:30:00Z",
    )

    assert result["status"] == "warning"
    assert "Stale data detected" in result["data_warning"]
    assert result["data_freshness"]["is_stale"] is True
    assert result["data_source"]["home_stats_season"] == "HISTORICAL"
    assert result["data_source"]["away_stats_season"] == "HISTORICAL"
