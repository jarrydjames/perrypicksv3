import json
import re
from datetime import datetime
from pathlib import Path

import requests

SCHEDULE_URL = "https://cdn.nba.com/static/json/staticData/scheduleLeagueV2.json"


def parse_date(s: str):
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def season_from_game_id(game_id: str) -> int:
    m = re.match(r"^002(\d{2})", game_id)
    return int(m.group(1)) if m else -1


def main(season_end_yy: int = 25):
    resp = requests.get(SCHEDULE_URL, timeout=30)
    resp.raise_for_status()
    sched = resp.json()

    games = sched.get("leagueSchedule", {}).get("gameDates", [])
    out = []
    for gd in games:
        for g in gd.get("games", []):
            gid = g.get("gameId")
            if not gid:
                continue
            if season_from_game_id(gid) != season_end_yy:
                continue

            out.append(
                {
                    "gameId": gid,
                    "gameDate": g.get("gameDateTimeUTC") or g.get("gameDateTime") or g.get("gameDate"),
                    "homeTeam": (g.get("homeTeam", {}) or {}).get("teamTricode"),
                    "awayTeam": (g.get("awayTeam", {}) or {}).get("teamTricode"),
                    "gameStatus": g.get("gameStatus"),
                }
            )

    out_path = Path(f"data/processed/game_ids_20{season_end_yy:02d}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(out, f, indent=2)

    print(f"Saved {len(out)} games -> {out_path}")


if __name__ == "__main__":
    main(season_end_yy=25)
