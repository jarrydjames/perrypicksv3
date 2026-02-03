"""Pregame model - OLD models with correct 34 pregame features"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, List
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
    """Pregame model - OLD models with correct 34 pregame features"""
    
    MODELS_DIR = Path("data/models")
    FEATURE_LIST_PATH = Path("data/processed/pregame_feature_list.txt")
    TARGET_TOTAL = "total"
    TARGET_MARGIN = "margin"
    
    def __init__(self):
        self.models_dir = self.MODELS_DIR
        self._loaded = False
    
    def load_models(self) -> bool:
        """Load trained pregame models if available."""
        if self._loaded:
            return True
        
        total_path = self.models_dir / "total_model_pregame.pkl"
        margin_path = self.models_dir / "margin_model_pregame.pkl"
        
        if not total_path.exists() or not margin_path.exists():
            return False
        
        # OLD MODELS are sklearn objects, wrap them
        total_raw = joblib.load(total_path)
        margin_raw = joblib.load(margin_path)
        
        self.total_model = {
            'model': total_raw,
            'residual_sigma': 8.5,
            'q10_model': None,
            'q90_model': None,
        }
        self.margin_model = {
            'model': margin_raw,
            'residual_sigma': 8.5,
            'q10_model': None,
            'q90_model': None,
        }
        
        # Load features from file
        with open(self.FEATURE_LIST_PATH) as f:
            self.features = [line.strip() for line in f if line.strip()]
        
        self.feature_version = "v1_pregame_34feat"
        
        self._loaded = True
        return True
    
    def predict(
        self,
        features: Dict[str, float],
        *,
        game_id: str,
    ) -> Optional[PregamePrediction]:
        """Predict game outcome before game starts."""
        if not self._loaded:
            if not self.load_models():
                return None
        
        # Use the feature list in the correct order
        feature_values = [features.get(f, 0.0) for f in self.features]
        X = np.array([feature_values])
        
        total_head = TrainedHead(
            features=self.features,
            model=self.total_model.get("model"),
            residual_sigma=self.total_model.get("residual_sigma", 8.5),
        )
        margin_head = TrainedHead(
            features=self.features,
            model=self.margin_model.get("model"),
            residual_sigma=self.margin_model.get("residual_sigma", 8.5),
        )
        
        if total_head.model is not None:
            total_mean = total_head.model.predict(X)[0]
        else:
            total_mean = 215.0
        
        if margin_head.model is not None:
            margin_mean = margin_head.model.predict(X)[0]
        else:
            margin_mean = 0.0
        
        sigma_total = self.total_model.get("residual_sigma", 8.5)
        sigma_margin = self.margin_model.get("residual_sigma", 8.5)
        
        total_q10 = total_mean - 1.28 * sigma_total
        total_q90 = total_mean + 1.28 * sigma_total
        margin_q10 = margin_mean - 1.28 * sigma_margin
        margin_q90 = margin_mean + 1.28 * sigma_margin
        
        margin_sd = self.margin_model.get("residual_sigma", 8.5)
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
            model_name="pregame_old_ridge_rf",
            feature_version=self.feature_version,
        )

def get_pregame_model() -> Optional[PregameModel]:
    """Get or create pregame model instance."""
    model = PregameModel()
    if model.load_models():
        return model
    return None
