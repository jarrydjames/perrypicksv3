from __future__ import annotations

from typing import Dict, List


def format_prediction(game_id: str, pred: Dict[str, object]) -> str:
    if pred.get("status") != "success":
        return f"{game_id}: prediction failed ({pred.get('error')})"
    total = pred.get("total")
    margin = pred.get("margin")
    winner = pred.get("winner")
    model_used = pred.get("model_used") or pred.get("model")
    if total is None or margin is None:
        return f"{game_id}: prediction incomplete"
    return f"{game_id} | total={float(total):.1f} margin={float(margin):.1f} winner={winner} ({model_used})"


def build_message(lines: List[str]) -> str:
    return "\n".join(lines)
