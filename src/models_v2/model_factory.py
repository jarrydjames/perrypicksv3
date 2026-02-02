"""Model factory for creating standardized models.

Provides easy interface for creating different model types:
- Ridge regression
- Random Forest
- Gradient Boosting
- XGBoost
- Quantile regression
- Ensemble models
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import joblib

from .quantile_regressor import QuantileRegressor
from .ensemble_model import EnsembleModel


class ModelFactory:
    """Factory for creating standardized models."""
    
    @staticmethod
    def create_ridge(alpha: float = 1.0, **kwargs) -> Ridge:
        """
        Create Ridge regression model.
        
        Args:
            alpha: Regularization strength
            **kwargs: Additional arguments for Ridge
            
        Returns:
            Ridge model
        """
        return Ridge(alpha=alpha, random_state=42, **kwargs)
    
    @staticmethod
    def create_random_forest(n_estimators: int = 100, max_depth: int = 6, **kwargs) -> RandomForestRegressor:
        """
        Create Random Forest model.
        
        Args:
            n_estimators: Number of trees
            max_depth: Maximum depth of trees
            **kwargs: Additional arguments for RandomForestRegressor
            
        Returns:
            RandomForestRegressor model
        """
        return RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42,
            **kwargs
        )
    
    @staticmethod
    def create_gradient_boosting(n_estimators: int = 100, max_depth: int = 6, **kwargs) -> GradientBoostingRegressor:
        """
        Create Gradient Boosting model.
        
        Args:
            n_estimators: Number of boosting stages
            max_depth: Maximum depth of trees
            **kwargs: Additional arguments for GradientBoostingRegressor
            
        Returns:
            GradientBoostingRegressor model
        """
        return GradientBoostingRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=0.1,
            random_state=42,
            **kwargs
        )
    
    @staticmethod
    def create_xgboost(n_estimators: int = 100, max_depth: int = 6, **kwargs):
        """
        Create XGBoost model.
        
        Args:
            n_estimators: Number of trees
            max_depth: Maximum depth of trees
            **kwargs: Additional arguments for XGBRegressor
            
        Returns:
            XGBRegressor model
        """
        try:
            import xgboost as xgb
            return xgb.XGBRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=0.1,
                random_state=42,
                n_jobs=-1,
                **kwargs
            )
        except ImportError:
            raise ImportError("XGBoost not installed. Install with: pip install xgboost")
    
    @staticmethod
    def create_quantile_regressor(quantiles: List[float] = [0.1, 0.5, 0.9]) -> QuantileRegressor:
        """
        Create quantile regression model.
        
        Args:
            quantiles: List of quantiles to predict
            
        Returns:
            QuantileRegressor model
        """
        return QuantileRegressor(quantiles=quantiles)
    
    @staticmethod
    def create_ensemble(
        model_types: List[str] = ['ridge', 'rf', 'gbt'],
        weights: Optional[List[float]] = None
    ) -> EnsembleModel:
        """
        Create ensemble model with specified base models.
        
        Args:
            model_types: List of model types ('ridge', 'rf', 'gbt', 'xgb')
            weights: Optional weights for each model
            
        Returns:
            EnsembleModel with base models
        """
        models = []
        
        for model_type in model_types:
            if model_type == 'ridge':
                models.append(('Ridge', ModelFactory.create_ridge()))
            elif model_type == 'rf':
                models.append(('RF', ModelFactory.create_random_forest()))
            elif model_type == 'gbt':
                models.append(('GBT', ModelFactory.create_gradient_boosting()))
            elif model_type == 'xgb':
                models.append(('XGB', ModelFactory.create_xgboost()))
            else:
                raise ValueError(f"Unknown model type: {model_type}")
        
        return EnsembleModel(models, weights=weights)
    
    @staticmethod
    def create_model(model_type: str, **kwargs):
        """
        Create a model by type name.
        
        Args:
            model_type: Model type ('ridge', 'rf', 'gbt', 'xgb', 'quantile', 'ensemble')
            **kwargs: Additional arguments for model creation
            
        Returns:
            Model instance
        """
        model_type = model_type.lower()
        
        if model_type == 'ridge':
            return ModelFactory.create_ridge(**kwargs)
        elif model_type == 'rf':
            return ModelFactory.create_random_forest(**kwargs)
        elif model_type == 'gbt':
            return ModelFactory.create_gradient_boosting(**kwargs)
        elif model_type == 'xgb':
            return ModelFactory.create_xgboost(**kwargs)
        elif model_type == 'quantile':
            return ModelFactory.create_quantile_regressor(**kwargs)
        elif model_type == 'ensemble':
            return ModelFactory.create_ensemble(**kwargs)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    @staticmethod
    def load_model(filepath: str):
        """
        Load a model from file.
        
        Args:
            filepath: Path to model file
            
        Returns:
            Loaded model
        """
        return joblib.load(filepath)
    
    @staticmethod
    def save_model(model: object, filepath: str):
        """
        Save a model to file.
        
        Args:
            model: Model to save
            filepath: Path to save model
        """
        joblib.dump(model, filepath)
        print(f"Model saved to {filepath}")


class TwoHeadModel:
    """Two-headed model for predicting both total and margin."""
    
    def __init__(self, model_type: str = 'ridge'):
        """
        Initialize two-headed model.
        
        Args:
            model_type: Type of base model to use
        """
        self.model_type = model_type
        self.total_model = None
        self.margin_model = None
        self.features = None
    
    def fit(self, X: np.ndarray, y_total: np.ndarray, y_margin: np.ndarray, features: List[str]):
        """
        Fit both total and margin models.
        
        Args:
            X: Feature matrix
            y_total: Total points target
            y_margin: Margin target
            features: Feature names
        """
        self.features = features
        
        print(f"Fitting two-headed {self.model_type} model...")
        
        # Create base models
        if self.model_type == 'ridge':
            self.total_model = ModelFactory.create_ridge(alpha=2.0)
            self.margin_model = ModelFactory.create_ridge(alpha=2.0)
        elif self.model_type == 'rf':
            self.total_model = ModelFactory.create_random_forest()
            self.margin_model = ModelFactory.create_random_forest()
        elif self.model_type == 'gbt':
            self.total_model = ModelFactory.create_gradient_boosting()
            self.margin_model = ModelFactory.create_gradient_boosting()
        elif self.model_type == 'xgb':
            self.total_model = ModelFactory.create_xgboost()
            self.margin_model = ModelFactory.create_xgboost()
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        # Fit models
        print("  Fitting total model...")
        self.total_model.fit(X, y_total)
        
        print("  Fitting margin model...")
        self.margin_model.fit(X, y_margin)
        
        print("Two-headed model fitted")
    
    def predict(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Predict both total and margin.
        
        Args:
            X: Feature matrix
            
        Returns:
            Dictionary with 'total' and 'margin' predictions
        """
        pred_total = self.total_model.predict(X)
        pred_margin = self.margin_model.predict(X)
        
        return {
            'total': pred_total,
            'margin': pred_margin,
        }
    
    def save(self, filepath: str):
        """
        Save two-headed model to file.
        
        Args:
            filepath: Path to save model
        """
        model_data = {
            'model_type': self.model_type,
            'features': self.features,
            'total_model': self.total_model,
            'margin_model': self.margin_model,
        }
        joblib.dump(model_data, filepath)
        print(f"TwoHeadModel saved to {filepath}")
    
    def load(self, filepath: str):
        """
        Load two-headed model from file.
        
        Args:
            filepath: Path to load model from
        """
        model_data = joblib.load(filepath)
        self.model_type = model_data['model_type']
        self.features = model_data['features']
        self.total_model = model_data['total_model']
        self.margin_model = model_data['margin_model']
        print(f"TwoHeadModel loaded from {filepath}")


if __name__ == '__main__':
    # Test factory
    print("Testing ModelFactory...")
    
    # Create different model types
    ridge = ModelFactory.create_ridge()
    rf = ModelFactory.create_random_forest()
    gbt = ModelFactory.create_gradient_boosting()
    qr = ModelFactory.create_quantile_regressor()
    ensemble = ModelFactory.create_ensemble(['ridge', 'rf', 'gbt'])
    
    print("\nCreated models:")
    print(f"  Ridge: {ridge}")
    print(f"  RandomForest: {rf}")
    print(f"  GradientBoosting: {gbt}")
    print(f"  QuantileRegressor: {qr}")
    print(f"  EnsembleModel: {ensemble}")
    
    # Test two-headed model
    print("\n\nTesting TwoHeadModel...")
    
    # Generate sample data
    np.random.seed(42)
    n_samples = 1000
    X = np.random.randn(n_samples, 5)
    y_total = X.sum(axis=1) + 220
    y_margin = X[:, 0] - X[:, 1] + 5
    
    features = [f'feature_{i}' for i in range(5)]
    
    thm = TwoHeadModel('ridge')
    thm.fit(X, y_total, y_margin, features)
    
    predictions = thm.predict(X[:10])
    print(f"\nPredictions for first 10 samples:")
    print(f"  Total: {predictions['total']}")
    print(f"  Margin: {predictions['margin']}")
