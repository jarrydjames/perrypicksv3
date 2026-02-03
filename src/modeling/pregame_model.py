"""Pregame model - Uses FINAL models with 72 features including temporal and form data"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional
import joblib
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
    Pregame model - Uses FINAL models with 72 features
    Includes: team ratings, temporal features, form data, H2H stats, schedule strength
    """
    
    # Use FINAL models trained on complete feature set
    MODELS_DIR = Path("data/models")
    FEATURE_LIST_PATH = Path("data/processed/final_features_feature_list.txt")
    TARGET_TOTAL = "total"
    TARGET_MARGIN = "margin"
    
    def __init__(self):
        self.models_dir = self.MODELS_DIR
        self._loaded = False
    
    def load_models(self) -> bool:
        """Load trained pregame models if available."""
        if self._loaded:
            return True
        
        # Use FINAL models with 72 features
        total_path = self.models_dir / "ridge_total_final.pkl"
        margin_path = self.models_dir / "rf_margin_final.pkl"
        
        if not total_path.exists() or not margin_path.exists():
            return False
        
        # FINAL MODELS are sklearn objects
        total_raw = joblib.load(total_path)
        margin_raw = joblib.load(margin_path)
        
        # Wrap in expected format for compatibility
        self.total_model = {
            'model': total_raw,
            'residual_sigma': 15.6,  # From FINAL_REPORT - Test MAE for total
            'q10_model': None,  # No quantile models
            'q90_model': None,
        }
        self.margin_model = {
            'model': margin_raw,
            'residual_sigma': 11.2,  # From FINAL_REPORT - Test MAE for margin
            'q10_model': None,
            'q90_model': None,
        }
        
        # Load features from FINAL feature list
        with open(self.FEATURE_LIST_PATH) as f:
            feature_lines = f.readlines()
            # Exclude metadata columns (game_id, game_date, etc.)
            exclude_cols = {'game_id', 'game_date', 'home_team_id', 'away_team_id', 'home_score', 'away_score', 'total', 'margin'}
            self.features = [line.strip() for line in feature_lines if line.strip() and line.strip() not in exclude_cols]
        
        self.feature_version = "v3_final_72feat"
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
            model_name="pregame_ridge_rf_final",
            feature_version=self.feature_version,
        )

def get_pregame_model() -> Optional[PregameModel]:
    """Get or create pregame model instance."""
    model = PregameModel()
    if model.load_models():
        return model
    return None
