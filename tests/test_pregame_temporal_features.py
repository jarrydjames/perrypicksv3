from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.build_dataset_pregame import add_temporal_features


def test_temporal_features_rest_days_and_h2h() -> None:
    rows = [
        {
            "game_id": "1",
            "game_date": datetime(2025, 1, 1, tzinfo=timezone.utc).isoformat(),
            "home_tri": "AAA",
            "away_tri": "BBB",
            "home_score": 100,
            "away_score": 90,
            "total": 190,
            "margin": 10,
        },
        {
            "game_id": "2",
            "game_date": datetime(2025, 1, 2, tzinfo=timezone.utc).isoformat(),
            "home_tri": "BBB",
            "away_tri": "AAA",
            "home_score": 95,
            "away_score": 105,
            "total": 200,
            "margin": -10,
        },
    ]

    enriched = add_temporal_features(rows)
    first = enriched[0]
    second = enriched[1]

    assert first["home_rest_days"] == 7.0
    assert second["away_is_b2b"] == 1.0
    assert second["h2h_games"] == 1.0
    assert second["h2h_home_win_pct"] == 0.0
    assert second["sos_diff"] == 20.0
