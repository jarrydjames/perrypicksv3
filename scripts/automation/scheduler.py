from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import time
from datetime import date

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.automation import build_message, format_prediction, post_message
from src.predict_api import predict_game
from scripts.automation.game_scanner import scan_games


def main() -> None:
    parser = argparse.ArgumentParser(description="Simple scheduler for halftime/Q3 predictions")
    parser.add_argument("--sleep", type=int, default=60, help="Seconds between scans")
    parser.add_argument("--mode", default="auto", choices=["auto", "pregame", "halftime", "q3"])
    parser.add_argument("--once", action="store_true", help="Run a single scan and exit")
    parser.add_argument("--include-pregame", action="store_true", help="Include scheduled games")
    args = parser.parse_args()

    if not os.getenv("DISCORD_WEBHOOK_URL"):
        raise SystemExit("DISCORD_WEBHOOK_URL is required")

    seen = set()
    while True:
        payload = scan_games(date.today())
        game_ids = payload.get("halftime", []) + payload.get("end_q3", [])
        if args.include_pregame:
            game_ids = payload.get("pregame", []) + game_ids
        new_games = [gid for gid in game_ids if gid not in seen]
        if new_games:
            seen.update(new_games)
            lines = []
            for gid in new_games:
                pred = predict_game(gid, mode=args.mode)
                lines.append(format_prediction(gid, pred))
            message = build_message(lines)
            post_message(os.environ["DISCORD_WEBHOOK_URL"], content=message, username="PerryPicks")
        if args.once:
            break
        time.sleep(args.sleep)


if __name__ == "__main__":
    main()
