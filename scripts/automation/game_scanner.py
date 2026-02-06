from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path
import sys
from typing import List

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.scoreboard import fetch_scoreboard


def _is_halftime(status_text: str, period: int | None, clock: str | None) -> bool:
    if status_text.lower().startswith("half"):
        return True
    if period == 2 and (clock in {"PT00M00.00S", "PT12M00.00S", None}):
        return True
    return False


def _is_end_q3(status_text: str, period: int | None, clock: str | None) -> bool:
    if "end of 3" in status_text.lower():
        return True
    if period == 3 and clock in {"PT00M00.00S", None}:
        return True
    if period == 4 and clock in {"PT12M00.00S", None}:
        return True
    return False


def scan_games(scan_date: date) -> dict:
    games = fetch_scoreboard(scan_date, include_live=True)
    halftime: List[str] = []
    end_q3: List[str] = []

    for game in games:
        if _is_halftime(game.status_text, game.period, game.clock):
            halftime.append(game.game_id)
        if _is_end_q3(game.status_text, game.period, game.clock):
            end_q3.append(game.game_id)

    return {
        "date": scan_date.isoformat(),
        "halftime": halftime,
        "end_q3": end_q3,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Scan games for halftime and end-of-Q3")
    parser.add_argument("--date", type=str, default=None, help="YYYY-MM-DD (defaults to today)")
    args = parser.parse_args()

    scan_date = date.fromisoformat(args.date) if args.date else date.today()
    payload = scan_games(scan_date)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
