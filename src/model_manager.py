"""
Perry Picks v3 - Model Manager

This module provides a unified interface for loading and using production models.
Based on MODEL_MANAGER.md selection criteria.

Usage:
    from src.model_manager import get_model, get_all_models, ModelManager
    
    # Get single model
    model = get_model('halftime', 'total')
    
    # Get all models for a game state
    models = get_all_models('pregame')
    
    # Use ModelManager for full control
    manager = ModelManager()
    ht_total = manager.get_model('halftime', 'total')
"""

import joblib
from pathlib import Path
from typing import Dict, Tuple, Literal, Optional
from dataclasses import dataclass
import numpy as np

# Type definitions
GameState = Literal['halftime', 'pregame', 'q3']
TargetType = Literal['total', 'margin']

# Model paths
MODEL_PATHS = {
    'halftime': {
        'total': 'models/team_2h_total.joblib',
        'margin': 'models/team_2h_margin.joblib',
    },
    'pregame': {
        'total': 'models_v3/pregame/ridge_total.joblib',
        'margin': 'models_v3/pregame/ridge_margin.joblib',
    },
    'q3': {
        'total': 'models_v3/q3/ridge_total.joblib',
        'margin': 'models_v3/q3/ridge_margin.joblib',
    }
}

# Model status (from backtest results)
MODEL_STATUS = {
    'halftime': {
        'total': {'status': 'READY', 'mae': 1.18, 'rmse': 3.27, 'roi': None, 'accuracy': None},
        'margin': {'status': 'READY', 'mae': 0.64, 'rmse': 1.22, 'roi': 12.24, 'accuracy': None},
    },
    'pregame': {
        'total': {'status': 'READY', 'mae': 3.64, 'rmse': 4.56, 'roi': None, 'accuracy': {'3pt': 49.3, '5pt': 73.5, '10pt': 96.9}},
        'margin': {'status': 'READY', 'mae': 3.42, 'rmse': 4.29, 'roi': 84.53, 'accuracy': None},
    },
    'q3': {
        'total': {'status': 'READY', 'mae': 5.56, 'rmse': 7.10, 'roi': None, 'accuracy': None},
        'margin': {'status': 'CAUTION', 'mae': 5.97, 'rmse': 7.48, 'roi': 7.26, 'accuracy': None},
    }
}


@dataclass
class ModelInfo:
    """Metadata about a production model."""
    game_state: GameState
    target: TargetType
    model_type: str  # 'RIDGE' or 'GBT'
    features: list
    sd: float  # Calibrated SD for 80% CI
    path: str
    status: str  # 'READY' or 'CAUTION'
    mae: float
    rmse: float
    roi: Optional[float]  # Margin only
    accuracy: Optional[Dict[str, float]]  # Total only


class ModelManager:
    """
    Manages loading and accessing production models.
    
    Uses the selection criteria defined in MODEL_MANAGER.md:
    1. Backtest performance (MAE)
    2. Betting performance (ROI/Accuracy)
    3. CV vs Backtest generalization
    4. Stability (coefficient of variation)
    """
    
    def __init__(self, base_path: Optional[Path] = None):
        """
        Initialize ModelManager.
        
        Args:
            base_path: Base directory for model files (default: current directory)
        """
        self.base_path = base_path or Path('.')
        self._cache = {}  # Cache loaded models
        
    def get_model_path(self, game_state: GameState, target: TargetType) -> Path:
        """
        Get the file path for a model.
        
        Args:
            game_state: 'halftime', 'pregame', or 'q3'
            target: 'total' or 'margin'
            
        Returns:
            Path to model file
        """
        return self.base_path / MODEL_PATHS[game_state][target]
    
    def load_model(self, game_state: GameState, target: TargetType, use_cache: bool = True) -> dict:
        """
        Load a production model from disk.
        
        Args:
            game_state: 'halftime', 'pregame', or 'q3'
            target: 'total' or 'margin'
            use_cache: If True, cache loaded models in memory
            
        Returns:
            Model object with keys: 'model', 'features', 'sd', 'model_name'
        """
        path = self.get_model_path(game_state, target)
        
        if use_cache and path in self._cache:
            return self._cache[path]
        
        try:
            model_obj = joblib.load(path)
            
            if use_cache:
                self._cache[path] = model_obj
            
            return model_obj
            
        except FileNotFoundError:
            raise FileNotFoundError(f"Model not found: {path}")
        except Exception as e:
            raise RuntimeError(f"Error loading model {path}: {e}")
    
    def get_model_info(self, game_state: GameState, target: TargetType) -> ModelInfo:
        """
        Get metadata about a model without loading the full model object.
        
        Args:
            game_state: 'halftime', 'pregame', or 'q3'
            target: 'total' or 'margin'
            
        Returns:
            ModelInfo object with model metadata
        """
        model_obj = self.load_model(game_state, target)
        status = MODEL_STATUS[game_state][target]
        
        return ModelInfo(
            game_state=game_state,
            target=target,
            model_type=model_obj.get('model_name', 'UNKNOWN'),
            features=model_obj.get('features', []),
            sd=model_obj.get('sd', 0.0),
            path=str(self.get_model_path(game_state, target)),
            status=status['status'],
            mae=status['mae'],
            rmse=status['rmse'],
            roi=status.get('roi'),
            accuracy=status.get('accuracy')
        )
    
    def predict(self, game_state: GameState, target: TargetType, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Make predictions with confidence intervals.
        
        Args:
            game_state: 'halftime', 'pregame', or 'q3'
            target: 'total' or 'margin'
            X: Feature matrix (n_samples, n_features)
            
        Returns:
            Tuple of (predictions, lower_ci, upper_ci) where:
            - predictions: Point predictions
            - lower_ci: Lower 80% confidence interval
            - upper_ci: Upper 80% confidence interval
        """
        model_obj = self.load_model(game_state, target)
        model = model_obj['model']
        sd = model_obj.get('sd', 0.0)
        
        # Make predictions
        predictions = model.predict(X)
        
        # Calculate 80% confidence intervals (1.2816 sigma)
        z_score = 1.2816
        lower_ci = predictions - (sd * z_score)
        upper_ci = predictions + (sd * z_score)
        
        return predictions, lower_ci, upper_ci
    
    def get_all_models(self, game_state: GameState) -> Dict[TargetType, dict]:
        """
        Load all models for a game state.
        
        Args:
            game_state: 'halftime', 'pregame', or 'q3'
            
        Returns:
            Dict mapping target ('total', 'margin') to model objects
        """
        return {
            'total': self.load_model(game_state, 'total'),
            'margin': self.load_model(game_state, 'margin')
        }
    
    def get_all_model_info(self) -> Dict[GameState, Dict[TargetType, ModelInfo]]:
        """
        Get metadata for all production models.
        
        Returns:
            Nested dict of model metadata
        """
        result = {}
        for gs in ['halftime', 'pregame', 'q3']:
            result[gs] = {}
            for target in ['total', 'margin']:
                result[gs][target] = self.get_model_info(gs, target)
        return result
    
    def clear_cache(self):
        """Clear cached models from memory."""
        self._cache.clear()


# Convenience functions for backward compatibility

def get_model(game_state: GameState, target: TargetType, base_path: Optional[Path] = None) -> dict:
    """
    Convenience function to load a single model.
    
    Args:
        game_state: 'halftime', 'pregame', or 'q3'
        target: 'total' or 'margin'
        base_path: Base directory for model files (default: current directory)
        
    Returns:
        Model object with keys: 'model', 'features', 'sd', 'model_name'
    """
    manager = ModelManager(base_path)
    return manager.load_model(game_state, target)


def get_all_models(game_state: GameState, base_path: Optional[Path] = None) -> Dict[TargetType, dict]:
    """
    Convenience function to load all models for a game state.
    
    Args:
        game_state: 'halftime', 'pregame', or 'q3'
        base_path: Base directory for model files (default: current directory)
        
    Returns:
        Dict mapping target ('total', 'margin') to model objects
    """
    manager = ModelManager(base_path)
    return manager.get_all_models(game_state)


def get_model_status(game_state: GameState, target: TargetType) -> dict:
    """
    Get backtest performance metrics for a model.
    
    Args:
        game_state: 'halftime', 'pregame', or 'q3'
        target: 'total' or 'margin'
        
    Returns:
        Dict with status, mae, rmse, roi, accuracy
    """
    return MODEL_STATUS[game_state][target]


def get_recommendation() -> str:
    """
    Get the recommended model based on selection criteria.
    
    Returns:
        String with recommendation
    """
    return """
RECOMMENDED MODELS (based on backtest performance):

1. LOWEST ERROR:
   - Halftime Margin (MAE: 0.64, ROI: 12.24%)
   - Halftime Total (MAE: 1.18)

2. HIGHEST ROI:
   - Pregame Margin (MAE: 3.42, ROI: 84.53%)
   - All 15 folds positive ROI

3. BEST TOTAL ACCURACY:
   - Pregame Total (MAE: 3.64, Acc@3pt: 49.3%)
   - Acc@5pt: 73.5%, Acc@10pt: 96.9%

4. USE WITH CAUTION:
   - Q3 Margin (MAE: 5.97, ROI: 7.26%)
   - GBT shows overfitting in CV vs backtest

STATUS: 5/6 models READY, 1/6 CAUTION
"""


# Example usage
if __name__ == '__main__':
    print("Model Manager Test")
    print("=" * 80)
    
    # Create manager
    manager = ModelManager()
    
    # Get all model info
    print("\nPRODUCTION MODEL INVENTORY:")
    print("-" * 80)
    for gs, targets in manager.get_all_model_info().items():
        for target, info in targets.items():
            print(f"{gs.upper():8} {target.upper():8}: {info.model_type} (MAE: {info.mae:.2f})")
            print(f"           Status: {info.status}, ROI: {info.roi}, Features: {len(info.features)}")
    
    print("\n" + get_recommendation())
