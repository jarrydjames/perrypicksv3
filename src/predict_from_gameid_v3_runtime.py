from src.predict_from_gameid_v2_ci import predict_from_game_id

def predict_from_game_id(game_id: str):
    """
    Predict halftime scores + final projections for a game_id.
    Delegates to the v3 runtime predictor (src.predict_from_gameid_v3_runtime.py)
    """
    # For v2, this was the production predictor.
    # NOTE: v3 runtime predictor does not exist yet; use placeholder for now.
    try:
        return predict_from_game_id(game_id)
    except Exception as e:
        return {"game_id": game_id, "error": str(e)}

