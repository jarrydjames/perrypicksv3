from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional


IMPORT_DIAGNOSTICS_DIR = Path("data/diagnostics")
IMPORT_WATERMARK_FILE = IMPORT_DIAGNOSTICS_DIR / "import_watermark.json"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_import_watermark(
    *,
    source: str,
    game_date: str,
    valid_games: int,
    quarantined_games: int,
    latest_game_time_utc: Optional[str] = None,
    output_path: Path = IMPORT_WATERMARK_FILE,
) -> Path:
    """Persist import watermark metadata for downstream freshness gating."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload: Dict[str, Any] = {
        "updated_at_utc": _utc_now_iso(),
        "source": source,
        "game_date": game_date,
        "valid_games": int(valid_games),
        "quarantined_games": int(quarantined_games),
        "latest_game_time_utc": latest_game_time_utc,
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return output_path


def read_import_watermark(path: Path = IMPORT_WATERMARK_FILE) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}

