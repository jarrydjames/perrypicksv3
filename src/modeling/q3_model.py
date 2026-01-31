"""Q3 model stub - follows same training/calibration methodology as halftime model.

This model is game-clock aware: can evaluate predictions at any game state
(halftime, end-of-Q3, or during play), compare to odds, and recommend bets.
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import joblib
import numpy as np

# Reuse same model infrastructure as halftime
from src.modeling.base import BaseTwoHeadModel, TrainedHead
from src.modeling.uncertainty import sigma_from_residuals


@dataclass(frozen=True)
class Q3Prediction:
    """Prediction output from Q3 model."""
    
    game_id: str
    period: int
    clock: str
    
    # Predictions (same structure as halftime)
    home_win_prob: float
    margin_mean: float
    margin_sd: float
    total_mean: float
    total_sd: float
    
    # Intervals (80% confidence bands)
    margin_q10: float
    margin_q90: float
    total_q10: float
    total_q90: float
    
    # Metadata
    model_name: str
    feature_version: str


class Q3Model:
    """
    Q3 model - game-clock aware predictor.
    
    Follows exact same training/calibration methodology as halftime model:
    - Uses two-head architecture (margin + total)
    - Quantile regression for 80% confidence intervals
    - Calibration via residual q10/q90 estimation
    
    Key difference: accepts (period, clock) to evaluate at any game state,
    not just hardcoded halftime filtering.
    """
    
    # Model paths (separate from halftime models)
    MODELS_DIR = Path("models_v3/q3")
    
    TARGET_TOTAL = "q3_total"
    TARGET_MARGIN = "q3_margin"
    
    def __init__(self):
        self.models_dir = self.MODELS_DIR
        self._loaded = False
    
    def _ensure_models_dir(self) -> None:
        self.models_dir.mkdir(parents=True, exist_ok=True)
    
    def load_models(self) -> bool:
        """Load trained Q3 models if available."""
        if self._loaded:
            return True
        
        total_path = self.models_dir / "gbt_twohead.joblib"  # Fixed: was q3_total_twohead.joblib
        margin_path = self.models_dir / "ridge_twohead.joblib"  # Fixed: was q3_margin_twohead.joblib
        
        if not total_path.exists() or not margin_path.exists():
            return False
        
        self.total_model = joblib.load(total_path)
        self.margin_model = joblib.load(margin_path)
        self._loaded = True
        return True
    
    def predict(
        self,
        features: Dict[str, float],
        *,
        period: int,
        clock: str,
        game_id: str,
    ) -> Optional[Q3Prediction]:
        """
        Predict at given game state.
        
        Args:
            features: Dict of feature values (same as halftime feature_columns)
            period: Current quarter (1-4, or OT)
            clock: Clock string (e.g., "PT5M30.00S")
            game_id: Game ID for tracking
        
        Returns:
            Q3Prediction if models loaded, else None
        """
        if not self._loaded:
            if not self.load_models():
                return None
        
        # Convert features to numpy array
        feature_names = list(features.keys())
        X = np.array([[features.get(f, 0.0) for f in feature_names]])
        
        # Get trained heads
        total_head = TrainedHead(
            model=self.total_model.get("model"),
            residual_sigma=self.total_model.get("residual_sigma", 2.0),
            q10_model=self.total_model.get("q10_model"),
            q90_model=self.total_model.get("q90_model"),
        )
        
        margin_head = TrainedHead(
            model=self.margin_model.get("model"),
            residual_sigma=self.margin_model.get("residual_sigma", 2.0),
            q10_model=self.margin_model.get("q10_model"),
            q90_model=self.margin_model.get("q90_model"),
        )
        
        # Predict means
        total_mean = total_head.model.predict(X)[0]
        margin_mean = margin_head.model.predict(X)[0]
        
        # Predict quantiles for intervals
        total_q10 = total_head.q10_model.predict(X)[0]
        total_q90 = total_head.q90_model.predict(X)[0]
        margin_q10 = margin_head.q10_model.predict(X)[0]
        margin_q90 = margin_head.q90_model.predict(X)[0]
        
        # Compute home win prob from margin
        margin_sd = margin_head.residual_sigma
        home_win_prob = 1.0 - (0.5 * (1.0 + margin_mean / (np.sqrt(2) * margin_sd)))
        home_win_prob = np.clip(home_win_prob, 0.01, 0.99)
        
        return Q3Prediction(
            game_id=game_id,
            period=period,
            clock=clock,
            home_win_prob=home_win_prob,
            margin_mean=margin_mean,
            margin_sd=margin_sd,
            total_mean=total_mean,
            total_sd=total_head.residual_sigma,
            margin_q10=margin_q10,
            margin_q90=margin_q90,
            total_q10=total_q10,
            total_q90=total_q90,
            model_name="Q3 Two-Head",
            feature_version=self.total_model.get("feature_version", "v3_q3"),
        )
    
    def calibrate(
        self,
        residuals_total: np.ndarray,
        residuals_margin: np.ndarray,
    ) -> Dict[str, float]:
        """
        Calibrate intervals using residuals (same methodology as halftime).
        
        Estimates q10/q90 quantiles of residuals to generate 80% confidence bands.
        """
        q_total = np.quantile(residuals_total, [0.1, 0.9])
        q_margin = np.quantile(residuals_margin, [0.1, 0.9])
        
        return {
            "total_q10": float(q_total[0]),
            "total_q90": float(q_total[1]),
            "margin_q10": float(q_margin[0]),
            "margin_q90": float(q_margin[1]),
        }


def get_q3_model() -> Q3Model:
    """Factory function to get Q3 model singleton."""
    if "_q3_model" not in globals():
        globals()["_q3_model"] = Q3Model()
    return globals()["_q3_model"]


def load_q3_intervals() -> Dict[str, float]:
    """Load calibrated Q3 intervals if available, else use defaults."""
    intervals_path = Path("models_v3/q3/q3_intervals.joblib")
    
    if intervals_path.exists():
        return joblib.load(intervals_path)
    
    # Fallback defaults (same as halftime)
    return {
        "total_q10": -8.0,
        "total_q90": 8.0,
        "margin_q10": -6.0,
        "margin_q90": 6.0,
    }
