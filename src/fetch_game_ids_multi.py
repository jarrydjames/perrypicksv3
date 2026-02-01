"""Fetch game IDs for multiple seasons.

Usage:
    python src/fetch_game_ids_multi.py
    # Fetches all seasons: 23, 24, 25
"""
import json
import re
from datetime import datetime
import requests

SCHEDULE_URL = "https://cdn.nba.com/static/json/staticData/scheduleLeagueV2.json"

def parse_date(s: str):
    return datetime.fromisoformat(s.replace("Z", "+00:00"))

def season_from_game_id(game_id: str) -> int:
    # pattern: 002YYxxxxx where YY is season end year (25 -> 2025-26)
    m = re.match(r"^002(\d{2})", game_id)
    return int(m.group(1)) if m else -1

def fetch_season(season_end_yy: int):
    """Fetch all game IDs for a single season."""
    sched = requests.get(SCHEDULE_URL, timeout=30).json()
    
    games = sched.get("leagueSchedule", {}).get("gameDates", [])
    out = []
    for gd in games:
        for g in gd.get("games", []):
            gid = g.get("gameId")
            if not gid:
                continue
            if season_from_game_id(gid) != season_end_yy:
                continue
            
            out.append({
                "gameId": gid,
                "gameDate": g.get("gameDateTimeUTC") or g.get("gameDateTime") or g.get("gameDate"),
                "homeTeam": (g.get("homeTeam", {}) or {}).get("teamTricode"),
                "awayTeam": (g.get("awayTeam", {}) or {}).get("teamTricode"),
                "gameStatus": g.get("gameStatus"),
            })
    
    out_path = f"data/processed/game_ids_20{season_end_yy:02d}.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    
    return len(out)

def main():
    seasons = [23, 24, 25]  # 2023-24, 2024-25, 2025-26
    
    all_games = {}
    
    for season_end_yy in seasons:
        print(f"\n{'='*60}")
        print(f"Fetching season 20{season_end_yy}-20{season_end_yy+1}")
        print(f"{'='*60}")
        
        try:
            count = fetch_season(season_end_yy)
            print(f"✅ Saved {count} games -> data/processed/game_ids_20{season_end_yy:02d}.json")
        except Exception as e:
            print(f"❌ Error fetching season {season_end_yy}: {e}")
    
    print(f"\n{'='*60}")
    print("✅ All seasons fetched!")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
