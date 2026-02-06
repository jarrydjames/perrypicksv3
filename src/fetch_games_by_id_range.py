"""Fetch NBA games by game ID range for a specific season."""

import json
import time
from pathlib import Path

import requests

BOX_URL = "https://cdn.nba.com/static/json/liveData/boxscore/boxscore_{}.json"
PBP_URL = "https://cdn.nba.com/static/json/liveData/playbyplay/playbyplay_{}.json"


def fetch_game(game_id: str, include_pbp: bool = True) -> dict:
    """Fetch box score and optionally play-by-play for a game."""
    try:
        box_response = requests.get(BOX_URL.format(game_id), timeout=30)
        box_response.raise_for_status()
        box_data = box_response.json()

        pbp_data = None
        if include_pbp:
            try:
                pbp_response = requests.get(PBP_URL.format(game_id), timeout=30)
                pbp_response.raise_for_status()
                pbp_data = pbp_response.json()
            except requests.RequestException:
                pbp_data = None

        return {
            "game_id": game_id,
            "box_score": box_data,
            "play_by_play": pbp_data,
            "success": True,
        }
    except (requests.RequestException, ValueError) as e:
        return {"game_id": game_id, "error": str(e), "success": False}


def fetch_game_range(season_code: str, start_game: int, end_game: int, include_pbp: bool = False) -> tuple:
    """Fetch games in a range within a season."""
    games_fetched = 0
    games_failed = 0

    print("=" * 70)
    print(f"FETCHING GAMES: Season {season_code}, Games {start_game} to {end_game}")
    print("=" * 70)

    results = []
    total_games = max(1, end_game - start_game + 1)
    progress_interval = max(1, total_games // 10)

    for idx, game_num in enumerate(range(start_game, end_game + 1), 1):
        game_id = f"00{season_code}{game_num:05d}"
        result = fetch_game(game_id, include_pbp=include_pbp)
        results.append(result)

        if result["success"]:
            games_fetched += 1
        else:
            games_failed += 1

        if idx % progress_interval == 0 or idx == total_games:
            print(f"  Progress: {idx}/{total_games} games ({games_fetched} fetched, {games_failed} failed)")

        time.sleep(0.1)

    print(f"\nSummary: {games_fetched} games fetched, {games_failed} failed")
    return results, games_fetched, games_failed


def save_games(results: list, box_dir: str, pbp_dir: str | None = None):
    """Save fetched games to disk."""
    box_path = Path(box_dir)
    box_path.mkdir(parents=True, exist_ok=True)

    pbp_path = None
    if pbp_dir:
        pbp_path = Path(pbp_dir)
        pbp_path.mkdir(parents=True, exist_ok=True)

    saved = 0
    pbp_saved = 0
    for result in results:
        if not result["success"]:
            continue

        game_id = result["game_id"]
        with (box_path / f"{game_id}.json").open("w") as f:
            json.dump(result["box_score"], f, indent=2)

        if pbp_path and result["play_by_play"]:
            with (pbp_path / f"{game_id}.json").open("w") as f:
                json.dump(result["play_by_play"], f, indent=2)
            pbp_saved += 1

        saved += 1

    print(f"Saved {saved} games to {box_dir}")
    if pbp_path:
        print(f"Saved {pbp_saved} play-by-play files to {pbp_dir}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Fetch NBA games by game ID range")
    parser.add_argument("--season", type=str, required=True, help='Season code (e.g., "25" for 2025-26 season)')
    parser.add_argument("--start-game", type=int, required=True, help="Starting game number (e.g., 1)")
    parser.add_argument("--end-game", type=int, required=True, help="Ending game number (e.g., 700)")
    parser.add_argument("--include-pbp", action="store_true", help="Also fetch play-by-play data (slower)")
    parser.add_argument("--box-dir", type=str, default="data/raw/box", help="Output directory for box scores")
    parser.add_argument("--pbp-dir", type=str, default="data/raw/pbp", help="Output directory for play-by-play")

    args = parser.parse_args()

    results, fetched, failed = fetch_game_range(
        season_code=args.season,
        start_game=args.start_game,
        end_game=args.end_game,
        include_pbp=args.include_pbp,
    )

    save_games(results, args.box_dir, args.pbp_dir if args.include_pbp else None)

    print("\n" + "=" * 70)
    print("FETCH COMPLETE")
    print("=" * 70)
    print(f"\nGames fetched: {fetched}")
    print(f"Games failed: {failed}")


if __name__ == "__main__":
    main()
