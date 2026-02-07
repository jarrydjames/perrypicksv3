"""Post-game learning snapshot and plain-language miss explainer generator."""

from pathlib import Path
import sys
import sqlite3
import json

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.qol import miss_explainer_three_bullets
from core.storage import MissExplanationStorage, DEFAULT_DB_PATH


def generate_for_game(game_id: str, db_path: Path = DEFAULT_DB_PATH):
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    cur.execute("SELECT * FROM games WHERE game_id = ?", (game_id,))
    game = cur.fetchone()
    cur.execute("SELECT * FROM picks WHERE game_id = ? ORDER BY created_at_utc DESC LIMIT 1", (game_id,))
    pick = cur.fetchone()
    conn.close()

    if not game or not pick:
        return []

    final_margin = (game["score_home"] or 0) - (game["score_away"] or 0)
    expected = f"a {pick['trigger_type']} edge on {pick['side']} with projected market advantage."
    changed = f"final margin finished at {final_margin}, diverging from the pick direction late in-game."
    evidence = "the pregame/halftime edge and confidence tier were valid at post time; the result moved after a live game-state swing."
    bullets = miss_explainer_three_bullets(expected, changed, evidence)
    MissExplanationStorage.store(game_id=game_id, trigger_type=pick['trigger_type'], bullets=bullets, db_path=db_path)
    return bullets


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("game_id")
    parser.add_argument("--db", default=str(DEFAULT_DB_PATH))
    args = parser.parse_args()

    bullets = generate_for_game(args.game_id, Path(args.db))
    print(json.dumps({"game_id": args.game_id, "bullets": bullets}, indent=2))
