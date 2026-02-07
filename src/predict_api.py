from __future__ import annotations
from typing import Any, Dict, Optional, Tuple
import datetime as dt
import os
import pandas as pd

from src.data.import_health import read_import_watermark


def _is_placeholder_team(tricode: Optional[str]) -> bool:
    t = str(tricode or "").strip().upper()
    return t in {"", "UNK", "HOME", "AWAY"}


def _pregame_import_gate(
    *,
    game_id: str,
    home_team: Optional[str],
    away_team: Optional[str],
    bypass: bool = False,
) -> Optional[dict]:
    # Allow bypassing the import gate for manual predictions (e.g., Streamlit UI)
    if bypass:
        return None
    
    if _is_placeholder_team(home_team) or _is_placeholder_team(away_team):
        return {
            "status": "error",
            "error": "PLACEHOLDER_GAME: invalid team tricode(s) in schedule payload",
            "game_id": game_id,
            "model_used": "IMPORT_GATE",
        }

    watermark = read_import_watermark()
    if not watermark:
        return {
            "status": "error",
            "error": "STALE_DATA: import watermark not found; run game scanner/import job first",
            "game_id": game_id,
            "model_used": "IMPORT_GATE",
        }

    updated_at = watermark.get("updated_at_utc")
    if not updated_at:
        return {
            "status": "error",
            "error": "STALE_DATA: import watermark missing updated_at_utc",
            "game_id": game_id,
            "model_used": "IMPORT_GATE",
        }

    try:
        updated_ts = pd.Timestamp(updated_at)
        if updated_ts.tzinfo is None:
            updated_ts = updated_ts.tz_localize("UTC")
        else:
            updated_ts = updated_ts.tz_convert("UTC")
        age_hours = (pd.Timestamp.now(tz="UTC") - updated_ts).total_seconds() / 3600.0
    except Exception:
        age_hours = 1e9

    max_hours = float(os.getenv("PREGAME_IMPORT_MAX_AGE_HOURS", "36"))
    if age_hours > max_hours:
        return {
            "status": "error",
            "error": f"STALE_DATA: import watermark is {age_hours:.1f}h old (max {max_hours:.1f}h)",
            "game_id": game_id,
            "model_used": "IMPORT_GATE",
            "data_freshness": {"watermark_age_hours": age_hours, "max_hours": max_hours},
        }

    return None

def detect_game_state(game_id: str) -> Tuple[str, Optional[dict]]:
    """
    Detect current game state to determine which model to use.
    
    Returns:
        Tuple of (game_state, game_data)
        - game_state: 'pregame', 'halftime', 'q3', or 'final'
        - game_data: Game data dict (if available)
    """
    import logging
    logger = logging.getLogger(__name__)
    
    try:
        from src.data.game_data import fetch_game_by_id
        game = fetch_game_by_id(game_id)
        
        if not game:
            # Game not found - assume pregame (might be upcoming)
            logger.info(f"Game {game_id} not found, assuming pregame state")
            return ('pregame', None)
        
        # Extract period and clock info
        home_periods = (game.get("homeTeam", {}) or {}).get("periods", [])
        away_periods = (game.get("awayTeam", {}) or {}).get("periods", [])
        
        # Count periods that have data
        all_periods = home_periods + away_periods
        periods_with_data = [p for p in all_periods if isinstance(p, dict)]
        
        if not periods_with_data:
            # No period data - game hasn't started
            return ('pregame', game)
        
        # Get highest period number
        max_period = 0
        for p in periods_with_data:
            period_num = p.get("period", 0)
            try:
                period_int = int(float(period_num))
                if period_int > max_period:
                    max_period = period_int
            except (ValueError, TypeError):
                pass
        
        # Get game status and clock
        game_status = game.get("gameStatus", 0)
        game_clock = game.get("gameClock", "PT00M00.00S")
        
        # Helper function to parse game clock (format: PT{minutes}M{seconds}.{milliseconds}S)
        def parse_clock(clock_str):
            try:
                if not clock_str or not clock_str.startswith('PT'):
                    return 0
                # Extract minutes
                parts = clock_str.replace('PT', '').replace('S', '').split('M')
                if len(parts) >= 2:
                    minutes = float(parts[0])
                    seconds = float(parts[1].split('.')[0])
                    return minutes + seconds / 60
                return 0
            except (ValueError, IndexError, AttributeError):
                return 0
        
        # Determine game state
        if max_period == 0:
            # Game hasn't started
            return ('pregame', game)
        elif max_period == 2:
            # Period 2 - could be halftime or in Q3
            # Check if there's any period 3 data
            has_period_3 = any(p.get('period') == 3 for p in periods_with_data)
            if has_period_3:
                return ('q3', game)
            else:
                # No period 3 data yet - assume halftime
                return ('halftime', game)
        elif max_period == 3:
            # Period 3 - check if we're halfway through
            # Q3 is 12 minutes; halfway is 6 minutes remaining
            minutes_remaining = parse_clock(game_clock)
            
            if minutes_remaining <= 6.0:
                # Halfway through Q3 or further (6 minutes or less remaining)
                return ('q3', game)
            else:
                # Less than halfway through Q3 (more than 6 minutes remaining)
                # Still use halftime model
                return ('halftime', game)
        elif max_period >= 4:
            # Period 4 or higher (Q4, OT) - use Q3 model
            return ('q3', game)
        else:
            # In progress (period 1 or in Q2)
            return ('pregame', game)
            
    except Exception as e:
        import logging
        logger.warning(f"Failed to detect game state for {game_id}: {e}")
        # Default to pregame if detection fails
        return ('pregame', None)

def extract_team_tricodes(game_data: Optional[dict], home_team: Optional[str], away_team: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
    """
    Extract team tricodes from game data if not provided.
    """
    if home_team and away_team:
        return (home_team, away_team)
    
    if game_data is None:
        return (None, None)
    
    home = game_data.get("homeTeam", {})
    away = game_data.get("awayTeam", {})
    
    home_tri = home.get("teamTricode") if home else None
    away_tri = away.get("teamTricode") if away else None
    
    return (home_tri, away_tri)

def predict_game(
    game_input: str,
    use_binned_intervals: bool = True,
    fetch_odds: bool = True,
    mode: str = 'auto',
    home_team: Optional[str] = None,
    away_team: Optional[str] = None,
    bypass_import_gate: bool = False,
) -> Dict[str, Any]:
    """
    Single public entrypoint used by app.py.

    Production runtime: uses compact sklearn models shipped in-repo.

    Model use-cases:
    - margin/spread/ML: ridge twohead (calibration-first)
    - game total: gbt twohead (small + stable)
    - team totals: derived from total+margin
    Returns rich dict (status, bands80, normal, labels, text, etc.).
    
    IMPORTANT: Game State Detection
    ------------------------------
    The system now properly detects game state to ensure correct model usage:
    - 'pregame':   Use pregame model (before game starts or early Q1)
    - 'halftime':  Use halftime model (at end of Q2, or early Q3 before halfway)
    - 'q3':        Use Q3 model (halfway through Q3 or later, Q4, OT)
    - 'final':     Use Q3 model (game finished)
    
    Auto-detection logic:
    - Period 0 or not started → PREGAME
    - Period 2 (no period 3 data) → HALFTIME
    - Period 3 (< 6 min remaining) → HALFTIME (early Q3)
    - Period 3 (>= 6 min remaining) → Q3 (halfway through Q3 or later)
    - Period 4+ → Q3
    
    Args:
        game_input: Game ID or URL
        use_binned_intervals: Legacy parameter (deprecated)
        fetch_odds: Whether to fetch odds from API
        mode: Model selection mode:
            - 'pregame': Force pregame model
            - 'halftime': Force halftime model  
            - 'q3': Force Q3 model
            - 'auto': Auto-detect based on game state (DEFAULT - RECOMMENDED)
        home_team: Home team tricode (optional, helps avoid API calls)
        away_team: Away team tricode (optional, helps avoid API calls)
        bypass_import_gate: Whether to bypass the import data freshness check (default False).
            Set True for manual predictions (e.g., Streamlit UI, ad-hoc predictions).
            Set False for production automation to ensure data freshness.
    
    Returns:
        Dict with prediction results including:
        - status: 'success' or 'error'
        - model_used: Which model was used
        - margin, total: Predictions
        - home_win_prob: Win probability
        
    Raises:
        ValueError: If game input is invalid
        Exception: If prediction fails
    """
    # `use_binned_intervals` kept for backwards compatibility; runtime predictor
    # already bakes in model-specific sigmas.
    _ = use_binned_intervals
    
    import logging
    logger = logging.getLogger(__name__)
    
    # Call prediction with comprehensive error handling
    try:
        # Step 1: Detect game state (if auto mode)
        game_state = None
        game_data = None
        
        if mode == 'auto':
            game_state, game_data = detect_game_state(game_input)
            logger.info(f"Auto-detected game state for {game_input}: {game_state}")
            
            # Extract team tricodes from game data if not provided
            if (home_team is None or away_team is None) and game_data:
                home_team, away_team = extract_team_tricodes(game_data, home_team, away_team)
        
        # Step 2: Determine which model to use based on mode/state
        use_model = mode
        
        if mode == 'auto' and game_state:
            # Map game state to model
            state_to_model = {
                'pregame': 'pregame',
                'halftime': 'halftime',
                'q3': 'q3',
                'final': 'q3',
            }
            use_model = state_to_model.get(game_state, 'pregame')
        
        # Step 3: Call appropriate model
        result = None
        
        if use_model == 'pregame':
            # PREGAME MODEL - Use for games that haven't started or early in Q1
            from src.predict_pregame import predict_from_game_id as predict_pregame
            
            # Validate we have team tricodes
            if not home_team or not away_team:
                # Try to extract from game data
                if game_data is None:
                    # Fetch game data to get team tricodes
                    from src.data.game_data import fetch_game_by_id
                    game_data = fetch_game_by_id(game_input)
                
                if game_data:
                    home_team, away_team = extract_team_tricodes(game_data, home_team, away_team)
                
                # If still no team tricodes, return error
                if not home_team or not away_team:
                    return {
                        "status": "error",
                        "error": f"Unable to determine team tricodes for game {game_input}. Please provide home_team and away_team parameters.",
                        "game_id": game_input,
                        "model_used": "ERROR",
                    }
            
            gate_error = _pregame_import_gate(
                game_id=game_input,
                home_team=home_team,
                away_team=away_team,
                bypass=bypass_import_gate,
            )
            if gate_error is not None:
                return gate_error

            result = predict_pregame(
                game_id=game_input,
                home_team=home_team,
                away_team=away_team,
                fetch_odds=fetch_odds,
                game_datetime=(game_data or {}).get("gameTimeUTC") if isinstance(game_data, dict) else None,
            )
            
            # Add game state info to result
            if result and result.get('status') == 'success':
                result['game_state'] = game_state if mode == 'auto' else 'pregame_forced'
                result['mode_requested'] = mode
        
        elif use_model == 'halftime':
            # HALFTIME MODEL - Use at end of Q2
            from src.predict_from_gameid_v2_ci import predict_from_game_id as predict_halftime
            raw_result = predict_halftime(game_input)
            
            # Normalize return structure to match expected format
            # `predict_from_gameid_v2_ci` returns rich payload with nested `pred` and interval bands.
            if raw_result and isinstance(raw_result, dict) and isinstance(raw_result.get('pred'), dict):
                pred = raw_result.get('pred', {})
                normal = raw_result.get('normal', {}) or {}

                margin_q10, margin_q90 = (normal.get('final_margin') or [None, None])[:2]
                total_q10, total_q90 = (normal.get('final_total') or [None, None])[:2]

                def _sd_from_q10_q90(q10, q90):
                    try:
                        q10 = float(q10)
                        q90 = float(q90)
                        if q90 <= q10:
                            return None
                        return (q90 - q10) / (2.0 * 1.2815515655)
                    except Exception:
                        return None

                margin_sd = _sd_from_q10_q90(margin_q10, margin_q90)
                total_sd = _sd_from_q10_q90(total_q10, total_q90)

                result = {
                    'game_id': raw_result.get('game_id'),
                    'home_name': raw_result.get('home_name'),
                    'away_name': raw_result.get('away_name'),
                    'margin': pred.get('pred_final_margin'),
                    'total': pred.get('pred_final_total'),
                    'home_score': raw_result.get('h1_home'),
                    'away_score': raw_result.get('h1_away'),
                    'margin_q10': margin_q10,
                    'margin_q90': margin_q90,
                    'total_q10': total_q10,
                    'total_q90': total_q90,
                    'home_win_prob': None,
                    'margin_sd': margin_sd,
                    'total_sd': total_sd,
                    'model_used': 'HALFTIME_V2_CI',
                    'model_name': None,
                    'feature_version': None,
                    'game_state': game_state if mode == 'auto' else 'halftime_forced',
                    'mode_requested': mode,
                    'status': 'success',
                }
            else:
                result = raw_result
                if result:
                    result['game_state'] = game_state if mode == 'auto' else 'halftime_forced'
                    result['mode_requested'] = mode
        
        elif use_model == 'q3':
            # Q3 MODEL - Use after end of Q3
            from src.predict_from_gameid_v3_runtime import predict_from_game_id as predict_q3
            result = predict_q3(game_input, fetch_odds=fetch_odds)
            
            # Q3 predictor already returns correct structure, just add metadata
            # Q3 predictor doesn't set 'status' field - check for required keys
            if result and 'margin' in result and 'total' in result:
                result['status'] = 'success'  # Set status explicitly
                result['game_state'] = game_state if mode == 'auto' else 'q3_forced'
                result['mode_requested'] = mode
        
        else:
            # Invalid mode
            return {
                "status": "error",
                "error": f"Invalid mode: {mode}. Must be 'auto', 'pregame', 'halftime', or 'q3'.",
                "game_id": game_input,
                "model_used": "ERROR",
            }
        
        # Validate that result is a dict (never a string or error)
        if not isinstance(result, dict):
            error_msg = str(result) if isinstance(result, str) else f"Unexpected result type: {type(result)}"
            raise ValueError(f"Prediction returned unexpected type: {error_msg}")
        
        # Validate that result has required keys (skip for pregame error responses)
        required_keys = ["game_id", "home_name", "away_name", "margin", "total"]
        missing_keys = [k for k in required_keys if k not in result]
        
        # Allow missing keys if status is error (pregame model might have data issues)
        if missing_keys and result.get('status') != 'error':
            raise ValueError(f"Prediction missing required keys: {missing_keys}")
        
        return result
        
    except Exception as e:
        # Re-raise with context for easier debugging
        import traceback
        logger.error(f"Prediction failed: {repr(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise
