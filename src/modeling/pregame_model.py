"""Pregame model - Uses champion models with rate-based features."""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional
import joblib
import math
import numpy as np

from src.modeling.base import BaseTwoHeadModel, TrainedHead

@dataclass(frozen=True)
class PregamePrediction:
    """Prediction output from pregame model."""
    
    game_id: str
    home_win_prob: float
    margin_mean: float
    margin_sd: float
    total_mean: float
    total_sd: float
    margin_q10: float
    margin_q90: float
    total_q10: float
    total_q90: float
    model_name: str
    feature_version: str


class PregameModel:
    """
    Pregame model - Uses champion models from the comprehensive evaluation.
    Includes rate-based efficiency features only (leakage-safe).
    """
    
    # Use champion models from comprehensive 7-model evaluation
    MODELS_DIR = Path("models_v3/pregame")
    TARGET_TOTAL = "total"
    TARGET_MARGIN = "margin"
    
    def __init__(self):
        self.models_dir = self.MODELS_DIR
        self._loaded = False
    
    def load_models(self) -> bool:
        """Load trained pregame models if available."""
        if self._loaded:
            return True
        
        # Use twohead champion model (RandomForest - lowest sigma)
        twohead_path = self.models_dir / "randomforest_twohead.joblib"
        
        if not twohead_path.exists():
            # Fallback to GBT
            twohead_path = self.models_dir / "gbt_twohead.joblib"
            if not twohead_path.exists():
                return False
        
        twohead_raw = joblib.load(twohead_path)
        
        # Extract total and margin models from twohead
        self.total_model = twohead_raw.get('total', {})
        self.margin_model = twohead_raw.get('margin', {})
        
        # Get sigma values from twohead
        total_sigma = self.total_model.get('residual_sigma', 9.58)
        margin_sigma = self.margin_model.get('residual_sigma', 2.95)
        
        # Update sigma in model dicts
        self.total_model['residual_sigma'] = total_sigma
        self.margin_model['residual_sigma'] = margin_sigma
        
        self.features = twohead_raw.get('features', [])
        self.feature_version = "v3_twohead_champion"
        self._loaded = True
        return True
    
    def predict(
        self,
        features: Dict[str, float],
        *,
        game_id: str,
    ) -> Optional[PregamePrediction]:
        """Predict game outcome using final models."""
        if not self._loaded:
            if not self.load_models():
                return None
        
        # Use the feature list in the correct order
        feature_values = [features.get(f, 0.0) for f in self.features]
        X = np.array([feature_values])
        
        total_head = TrainedHead(
            features=self.features,
            model=self.total_model.get("model"),
            residual_sigma=self.total_model.get("residual_sigma", 15.6),
        )
        margin_head = TrainedHead(
            features=self.features,
            model=self.margin_model.get("model"),
            residual_sigma=self.margin_model.get("residual_sigma", 11.2),
        )
        
        if total_head.model is not None:
            total_mean = total_head.model.predict(X)[0]
        else:
            total_mean = 215.0
        
        if margin_head.model is not None:
            margin_mean = margin_head.model.predict(X)[0]
        else:
            margin_mean = 0.0
        
        # 80% confidence intervals (no quantile models, use residual sigma)
        sigma_total = self.total_model.get("residual_sigma", 15.6)
        sigma_margin = self.margin_model.get("residual_sigma", 11.2)
        
        total_q10 = total_mean - 1.28 * sigma_total
        total_q90 = total_mean + 1.28 * sigma_total
        margin_q10 = margin_mean - 1.28 * sigma_margin
        margin_q90 = margin_mean + 1.28 * sigma_margin
        
        margin_sd = self.margin_model.get("residual_sigma", 11.2)
        # P(home wins) = P(home-away margin > 0) under Normal(mean=margin_mean, sd=margin_sd)
        z = float(margin_mean) / max(1e-6, float(margin_sd))
        home_win_prob = 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))
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
            model_name="pregame_twohead_rf_champion",
            feature_version=self.feature_version,
        )

def get_pregame_model() -> Optional[PregameModel]:
    """Get or create pregame model instance."""
    model = PregameModel()
    if model.load_models():
        return model
    return None