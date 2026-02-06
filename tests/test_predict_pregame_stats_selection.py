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

    row_222 = pregame.fetch_team_stats(222)
    row_111 = pregame.fetch_team_stats(111)

    assert row_222 is not None
    assert row_111 is not None
    assert float(row_222["OFF_RATING"]) == 118.0
    assert float(row_111["OFF_RATING"]) == 105.0
