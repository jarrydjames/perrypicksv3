from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path
import sys
from typing import List, Tuple

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.scoreboard import fetch_scoreboard
from src.data.import_health import write_import_watermark


def _is_valid_game_id(game_id: str) -> bool:
    gid = str(game_id or "").strip()
    return len(gid) == 10 and gid.isdigit()


def _validate_game(game) -> Tuple[bool, str]:
    if not _is_valid_game_id(game.game_id):
        return False, "INVALID_GAME_ID"
    if game.home in {"UNK", "HOME", ""}:
        return False, "INVALID_HOME_TEAM"
    if game.away in {"UNK", "AWAY", ""}:
        return False, "INVALID_AWAY_TEAM"
    return True, ""


def _write_quarantine(scan_date: date, quarantined: list[dict]) -> str | None:
    if not quarantined:
        return None
    out_dir = Path("data/diagnostics")
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"quarantined_games_{scan_date.strftime('%Y%m%d')}.json"
    path.write_text(json.dumps(quarantined, indent=2), encoding="utf-8")
    return str(path)


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
    quarantined: list[dict] = []
    valid_games = []

    for game in games:
        valid, reason = _validate_game(game)
        if not valid:
            quarantined.append(
                {
                    "game_id": game.game_id,
                    "away": game.away,
                    "home": game.home,
                    "status_text": game.status_text,
                    "reason": reason,
                }
            )
            continue
        valid_games.append(game)

    pregame: List[str] = []
    halftime: List[str] = []
    end_q3: List[str] = []

    for game in valid_games:
        if game.period in {0, None}:
            pregame.append(game.game_id)
        if _is_halftime(game.status_text, game.period, game.clock):
            halftime.append(game.game_id)
        if _is_end_q3(game.status_text, game.period, game.clock):
            end_q3.append(game.game_id)

    quarantine_path = _write_quarantine(scan_date, quarantined)
    write_import_watermark(
        source="cdn_nba_schedule_boxscore",
        game_date=scan_date.isoformat(),
        valid_games=len(valid_games),
        quarantined_games=len(quarantined),
    )

    return {
        "date": scan_date.isoformat(),
        "pregame": pregame,
        "halftime": halftime,
        "end_q3": end_q3,
        "valid_games": len(valid_games),
        "quarantined_games": len(quarantined),
        "quarantine_path": quarantine_path,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Scan games for halftime and end-of-Q3")
    parser.add_argument("--date", type=str, default=None, help="YYYY-MM-DD (defaults to today)")
    parser.add_argument("--import-check-only", action="store_true", help="Only run import/schedule validation checks")
    args = parser.parse_args()

    scan_date = date.fromisoformat(args.date) if args.date else date.today()
    payload = scan_games(scan_date)
    if args.import_check_only:
        reduced = {
            "date": payload.get("date"),
            "valid_games": payload.get("valid_games"),
            "quarantined_games": payload.get("quarantined_games"),
            "quarantine_path": payload.get("quarantine_path"),
        }
        print(json.dumps(reduced, indent=2))
        return
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
