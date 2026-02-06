from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import requests

BOX_URL = "https://cdn.nba.com/static/json/liveData/boxscore/boxscore_{gid}.json"


def _fetch_final_score(game_id: str) -> Dict[str, float] | None:
    response = requests.get(BOX_URL.format(gid=game_id), timeout=10)
    response.raise_for_status()
    payload = response.json()
    game = (payload or {}).get("game") or {}
    home = (game.get("homeTeam") or {}).get("statistics") or {}
    away = (game.get("awayTeam") or {}).get("statistics") or {}
    home_points = home.get("points")
    away_points = away.get("points")
    if home_points is None or away_points is None:
        return None
    return {"home_points": float(home_points), "away_points": float(away_points)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Grade predictions using final boxscores")
    parser.add_argument("predictions", type=Path, help="CSV with game_id, total, margin")
    parser.add_argument("--out", type=Path, default=Path("data/predictions/graded_predictions.csv"))
    args = parser.parse_args()

    df = pd.read_csv(args.predictions)
    results: List[Dict[str, float]] = []

    for _, row in df.iterrows():
        game_id = str(row.get("game_id"))
        final = _fetch_final_score(game_id)
        if not final:
            continue
        home_points = final["home_points"]
        away_points = final["away_points"]
        actual_total = home_points + away_points
        actual_margin = home_points - away_points
        results.append(
            {
                "game_id": game_id,
                "pred_total": float(row.get("total", 0.0)),
                "pred_margin": float(row.get("margin", 0.0)),
                "actual_total": actual_total,
                "actual_margin": actual_margin,
                "total_error": float(row.get("total", 0.0)) - actual_total,
                "margin_error": float(row.get("margin", 0.0)) - actual_margin,
            }
        )

    if not results:
        raise SystemExit("No graded results produced (missing final scores?)")

    out_df = pd.DataFrame(results)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.out, index=False)
    print(f"Saved graded predictions to {args.out}")


if __name__ == "__main__":
    main()
