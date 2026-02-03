from __future__ import annotations
from typing import Any, Dict, Optional
import datetime as dt

def predict_game(
    game_input: str,
    use_binned_intervals: bool = True,
    fetch_odds: bool = True,
    mode: str = 'auto',
    home_team: Optional[str] = None,
    away_team: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Single public entrypoint used by app.py.

    Production runtime: uses compact sklearn models shipped in-repo.

    Model use-cases:
    - margin/spread/ML: ridge twohead (calibration-first)
    - game total: gbt twohead (small + stable)
    - team totals: derived from total+margin
    Returns rich dict (status, bands80, normal, labels, text, etc.).
    
    Args:
        game_input: Game ID or URL
        use_binned_intervals: Legacy parameter (deprecated)
        fetch_odds: Whether to fetch odds from API
        mode: Model selection mode:
            - 'pregame': Force pregame model
            - 'halftime': Force halftime model  
            - 'q3': Force Q3 model
            - 'auto': Auto-detect based on game state (default)
        home_team: Home team tricode (optional, helps avoid API calls)
        away_team: Away team tricode (optional, helps avoid API calls)
    
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
        # Determine which model to use based on mode
        if mode == 'pregame':
            # Force pregame model (FIXED - now using correct model!)
            from src.predict_pregame import predict_from_game_id as predict_pregame
            
            # Get team tricodes if not provided
            if home_team is None or away_team is None:
                # For pregame, we MUST have team tricodes
                # Try to extract from game ID
                home_team = home_team or 'UNK'
                away_team = away_team or 'UNK'
            
            result = predict_pregame(
                game_id=game_input,
                home_team=home_team,
                away_team=away_team,
                fetch_odds=fetch_odds,
            )
        
        else:
            # Halftime or Q3 models (or auto-detect)
            from src.predict_from_gameid_v3_runtime import predict_from_game_id as predict_runtime
            result = predict_runtime(game_input, fetch_odds=fetch_odds)
        
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
