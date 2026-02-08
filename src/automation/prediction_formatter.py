from __future__ import annotations

from typing import Dict, List


def format_prediction(game_id: str, pred: Dict[str, object]) -> str:
    """Format prediction as a detailed Discord message.
    
    Includes team names, predicted scores, winner, total, and margin.
    """
    if pred.get("status") != "success":
        return f"{game_id}: prediction failed ({pred.get('error')})"
    
    # Extract prediction data
    home_team = pred.get("home_name", pred.get("home_team", "Home"))
    away_team = pred.get("away_name", pred.get("away_team", "Away"))
    total = pred.get("total")
    margin = pred.get("margin")
    winner = pred.get("winner")
    model_used = pred.get("model_used") or pred.get("model")
    
    # Validate required fields
    if total is None or margin is None:
        return f"{game_id}: prediction incomplete"
    
    # Calculate individual scores
    home_score = (float(total) + float(margin)) / 2
    away_score = (float(total) - float(margin)) / 2
    
    # Build formatted message
    lines = [
        f"**{away_team} @ {home_team}**",
        "",
        f"📊 **Predicted Score:**",
        f"{away_team} {away_score:.1f} - {home_team} {home_score:.1f}",
        "",
        f"🏆 **Winner:** {winner}",
        "",
        f"📈 **Details:**",
        f"Total: {float(total):.1f} | Margin: {float(margin):.1f}",
        f"Model: {model_used}",
    ]
    
    return "\n".join(lines)


def build_message(lines: List[str]) -> str:
    return "\n".join(lines)
