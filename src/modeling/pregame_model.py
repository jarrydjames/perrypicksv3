"""Pregame model - follows same methodology as halftime/Q3 models.

This model makes predictions before the game starts, using only pregame
information (team stats, form, etc.) - no game state.

Same training/calibration methodology as halftime/Q3:
- Two-head architecture (margin + total)
- Quantile regression for 80% confidence intervals
- Calibration via residual q10/q90 estimation
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import joblib
import numpy as np

# Reuse same model infrastructure as halftime/Q3
from src.modeling.base import BaseTwoHeadModel, TrainedHead


@dataclass(frozen=True)
class PregamePrediction:
    """Prediction output from pregame model."""
    
    game_id: str
    
    # Predictions (same structure as halftime/Q3)
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


class PregameModel:
    """
    Pregame model - predicts before game starts.
    
    Follows exact same training/calibration methodology as halftime/Q3:
    - Two-head architecture (margin + total)
    - Quantile regression for 80% confidence intervals
    - Calibration via residual q10/q90 estimation
    
    Key difference: no game clock input - pregame only.
    """
    
    # Model paths (separate from halftime/Q3)
    MODELS_DIR = Path("models_v3/pregame")
    
    TARGET_TOTAL = "total"
    TARGET_MARGIN = "margin"
    
    def __init__(self):
        self.models_dir = self.MODELS_DIR
        self._loaded = False
    
    def _ensure_models_dir(self) -> None:
        self.models_dir.mkdir(parents=True, exist_ok=True)
    
    def load_models(self) -> bool:
        """Load trained pregame models if available."""
        if self._loaded:
            return True
        
        total_path = self.models_dir / "gbt_twohead.joblib"
        margin_path = self.models_dir / "ridge_twohead.joblib"
        
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
        game_id: str,
    ) -> Optional[PregamePrediction]:
        """
        Predict game outcome before game starts.
        
        Args:
            features: Dict of feature values (pregame features only)
            game_id: Game ID for tracking
        
        Returns:
            PregamePrediction if models loaded, else None
        """
        if not self._loaded:
            if not self.load_models():
                return None
        
        # Convert features to numpy array
        feature_names = list(features.keys())
        X = np.array([[features.get(f, 0.0) for f in feature_names]])
        
        # Get trained heads
        # Same structure as halftime/Q3 models
        total_head = TrainedHead(
            features=list(feature_names),
            model=self.total_model.get("total", {}).get("model"),
            residual_sigma=self.total_model.get("total", {}).get("residual_sigma", 2.0),
        )
        margin_head = TrainedHead(
            features=list(feature_names),
            model=self.margin_model.get("margin", {}).get("model"),
            residual_sigma=self.margin_model.get("margin", {}).get("residual_sigma", 2.0),
        )
        
        # Extract quantile models separately (same as halftime/Q3)
        total_q10_model = self.total_model.get("total", {}).get("q10_model")
        total_q90_model = self.total_model.get("total", {}).get("q90_model")
        margin_q10_model = self.margin_model.get("margin", {}).get("q10_model")
        margin_q90_model = self.margin_model.get("margin", {}).get("q90_model")
        
        # Predict means
        # Check if main models exist
        if total_head.model is not None:
            total_mean = total_head.model.predict(X)[0]
        else:
            total_mean = 215.0  # Fallback: typical NBA game total
        
        if margin_head.model is not None:
            margin_mean = margin_head.model.predict(X)[0]
        else:
            margin_mean = 0.0  # Fallback: neutral
        
        # Predict quantiles for intervals
        if total_q10_model is not None:
            total_q10 = total_q10_model.predict(X)[0]
        else:
            total_q10 = total_mean - 8.0  # Fallback
        
        if total_q90_model is not None:
            total_q90 = total_q90_model.predict(X)[0]
        else:
            total_q90 = total_mean + 8.0  # Fallback
        
        if margin_q10_model is not None:
            margin_q10 = margin_q10_model.predict(X)[0]
        else:
            margin_q10 = 0.0  # Fallback
        
        if margin_q90_model is not None:
            margin_q90 = margin_q90_model.predict(X)[0]
        else:
            margin_q90 = 0.0  # Fallback
        
        # Compute home win prob from margin (same as halftime/Q3)
        margin_sd = margin_head.residual_sigma
        home_win_prob = 1.0 - (0.5 * (1.0 + margin_mean / (np.sqrt(2) * margin_sd)))
        home_win_prob = np.clip(home_win_prob, 0.01, 0.99)
        
        return PregamePrediction(
            game_id=game_id,
            home_win_prob=home_win_prob,
            margin_mean=margin_mean,
            margin_sd=margin_sd,
            total_mean=total_mean,
            total_sd=total_head.residual_sigma,
            margin_q10=margin_q10,
            margin_q90=margin_q90,
            total_q10=total_q10,
            total_q90=total_q90,
            model_name="pregame_gbt_ridge",
            feature_version=self.total_model.get("feature_version", "unknown"),
        )
