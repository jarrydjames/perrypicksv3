"""QoL and platform utility helpers."""

from __future__ import annotations

import hashlib
from typing import Any, Dict, List, Optional


def canonical_pick_id(game_id: str, trigger_type: str, bet_type: str, side: str, model_version: str) -> str:
    raw = f"{game_id}|{trigger_type}|{bet_type}|{side}|{model_version}"
    return hashlib.sha256(raw.encode()).hexdigest()[:20]


def confidence_tier(probability: float, edge: float) -> str:
    if probability >= 0.62 and edge >= 0.04:
        return "HIGH"
    if probability >= 0.55 and edge >= 0.02:
        return "MEDIUM"
    return "LOW"


def interval_width(low: Optional[float], high: Optional[float]) -> Optional[float]:
    if low is None or high is None:
        return None
    return high - low


def explain_trigger_decision(trigger_type: str, game_state: Dict[str, Any], has_data: bool, has_picks: bool) -> Dict[str, Any]:
    return {
        "trigger_type": trigger_type,
        "status": game_state.get("status"),
        "period": game_state.get("current_period"),
        "clock": game_state.get("game_clock"),
        "has_data": has_data,
        "has_picks": has_picks,
        "decision": "fired" if has_data and has_picks else "skipped",
    }


def miss_explainer_three_bullets(
    expected_path: str,
    changed_live: str,
    deviation_evidence: str,
) -> List[str]:
    return [
        f"What we expected: {expected_path}",
        f"What changed live: {changed_live}",
        f"Why this was path deviation (not model collapse): {deviation_evidence}",
    ][:3]


def should_use_degraded_mode(odds: Dict[str, Any], game_state: Dict[str, Any]) -> bool:
    """Use degraded mode when configured or when key market data is missing."""
    import os
    if os.getenv("DEGRADED_MODE", "0") in {"1", "true", "TRUE"}:
        return True
    if not odds:
        return True
    return not any(k in odds for k in ("spread", "total", "moneyline"))
