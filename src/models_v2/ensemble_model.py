"""Ensemble model combining multiple predictors.

Combines predictions from Ridge, Random Forest, GBT, and XGBoost
using weighted averaging and stacking.
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score
import joblib


class EnsembleModel:
    """Ensemble of multiple regression models."""
    
    def __init__(self, models: List[Tuple[str, object]], weights: Optional[List[float]] = None):
        """
        Initialize ensemble model.
        
        Args:
            models: List of (name, model) tuples
            weights: Optional weights for each model (default: equal)
        """
        self.models = models
        self.n_models = len(models)
        
        if weights is not None:
            if len(weights) != len(models):
                raise ValueError("Weights must match number of models")
            self.weights = np.array(weights) / np.sum(weights)
        else:
            self.weights = np.ones(len(models)) / len(models)
        
        self.features = None
        self.is_fitted = False
        
        # Store predictions from base models for stacking
        self.base_predictions = {}
        self.meta_model = None
        self.use_stacking = False
    
    def fit_base_models(self, X: np.ndarray, y: np.ndarray, features: List[str]):
        """
        Fit all base models.
        
        Args:
            X: Feature matrix
            y: Target values
            features: Feature names
        """
        self.features = features
        
        print(f"Fitting {self.n_models} base models...")
        for i, (name, model) in enumerate(self.models):
            print(f"  Fitting {name}... ({i+1}/{self.n_models})")
            
            # Handle different model types
            if hasattr(model, 'fit'):
                model.fit(X, y)
            elif isinstance(model, dict) and 'model' in model:
                model['model'].fit(X, y)
            else:
                raise ValueError(f"Unknown model type: {type(model)}")
        
        self.is_fitted = True
        print("All base models fitted")
    
    def predict_weighted(self, X: np.ndarray) -> np.ndarray:
        """
        Predict using weighted average.
        
        Args:
            X: Feature matrix
            
        Returns:
            Weighted average of predictions
        """
        if not self.is_fitted:
            raise ValueError("Models must be fitted before prediction")
        
        predictions = []
        
        for name, model in self.models:
            # Get prediction from model
            if hasattr(model, 'predict'):
                pred = model.predict(X)
            elif isinstance(model, dict) and 'model' in model:
                pred = model['model'].predict(X)
            else:
                raise ValueError(f"Cannot predict with model: {type(model)}")
            
            predictions.append(pred)
        
        predictions = np.array(predictions)
        
        # Weighted average
        weighted_pred = np.average(predictions, axis=0, weights=self.weights)
        
        return weighted_pred
    
    def fit_stacking(self, X: np.ndarray, y: np.ndarray, cv_folds: int = 5):
        """
        Fit stacking meta-model using cross-validation.
        
        Args:
            X: Feature matrix
            y: Target values
            cv_folds: Number of CV folds
        """
        print("\nFitting stacking meta-model...")
        
        # Generate out-of-fold predictions
        oof_predictions = []
        
        fold_size = len(X) // cv_folds
        for fold in range(cv_folds):
            start = fold * fold_size
            end = (fold + 1) * fold_size if fold < cv_folds - 1 else len(X)
            
            X_train = np.concatenate([X[:start], X[end:]])
            y_train = np.concatenate([y[:start], y[end:]])
            X_val = X[start:end]
            
            # Train base models on train set
            for name, model in self.models:
                if hasattr(model, 'fit'):
                    model.fit(X_train, y_train)
                elif isinstance(model, dict) and 'model' in model:
                    model['model'].fit(X_train, y_train)
            
            # Predict on validation set
            fold_preds = []
            for name, model in self.models:
                if hasattr(model, 'predict'):
                    pred = model.predict(X_val)
                elif isinstance(model, dict) and 'model' in model:
                    pred = model['model'].predict(X_val)
                fold_preds.append(pred)
            
            oof_predictions.append(np.array(fold_preds).T)
        
        # Concatenate OOF predictions
        oof_predictions = np.vstack(oof_predictions)
        
        # Train meta-model on OOF predictions
        self.meta_model = Ridge(alpha=1.0)
        self.meta_model.fit(oof_predictions, y)
        
        self.use_stacking = True
        print("Stacking meta-model fitted")
        
        # Retrain base models on full data
        self.fit_base_models(X, y, self.features)
    
    def predict_stacking(self, X: np.ndarray) -> np.ndarray:
        """
        Predict using stacking.
        
        Args:
            X: Feature matrix
            
        Returns:
            Stacked predictions
        """
        if not self.is_fitted or self.meta_model is None:
            raise ValueError("Models must be fitted before stacking prediction")
        
        # Get predictions from all base models
        base_preds = []
        for name, model in self.models:
            if hasattr(model, 'predict'):
                pred = model.predict(X)
            elif isinstance(model, dict) and 'model' in model:
                pred = model['model'].predict(X)
            base_preds.append(pred)
        
        base_preds = np.array(base_preds).T
        
        # Predict using meta-model
        stacked_pred = self.meta_model.predict(base_preds)
        
        return stacked_pred
    
    def predict(self, X: np.ndarray, method: str = 'weighted') -> np.ndarray:
        """
        Predict using specified method.
        
        Args:
            X: Feature matrix
            method: 'weighted' or 'stacking'
            
        Returns:
            Predictions
        """
        if method == 'weighted':
            return self.predict_weighted(X)
        elif method == 'stacking':
            return self.predict_stacking(X)
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def evaluate_models(self, X: np.ndarray, y: np.ndarray, cv_folds: int = 5) -> Dict[str, Dict[str, float]]:
        """
        Evaluate each base model using cross-validation.
        
        Args:
            X: Feature matrix
            y: Target values
            cv_folds: Number of CV folds
            
        Returns:
            Dictionary with evaluation metrics for each model
        """
        results = {}
        
        for name, model in self.models:
            if hasattr(model, 'predict'):
                cv_scores = cross_val_score(
                    model, X, y, cv=cv_folds,
                    scoring='neg_mean_absolute_error'
                )
                mae = -np.mean(cv_scores)
            elif isinstance(model, dict) and 'model' in model:
                cv_scores = cross_val_score(
                    model['model'], X, y, cv=cv_folds,
                    scoring='neg_mean_absolute_error'
                )
                mae = -np.mean(cv_scores)
            else:
                continue
            
            results[name] = {'mae': mae, 'weight': 1.0 / mae}
        
        # Normalize weights based on inverse MAE
        total_weight = sum(r['weight'] for r in results.values())
        for name in results:
            results[name]['weight'] /= total_weight
        
        return results
    
    def optimize_weights(self, X: np.ndarray, y: np.ndarray):
        """
        Optimize weights based on cross-validation performance.
        
        Args:
            X: Feature matrix
            y: Target values
        """
        print("\nOptimizing ensemble weights...")
        results = self.evaluate_models(X, y)
        
        # Update weights based on performance
        new_weights = []
        for _, model in self.models:
            model_name = None
            for name in results:
                if name in str(model):
                    model_name = name
                    break
            
            if model_name in results:
                new_weights.append(results[model_name]['weight'])
            else:
                new_weights.append(1.0 / self.n_models)
        
        self.weights = np.array(new_weights) / np.sum(new_weights)
        
        print("Optimized weights:")
        for (_, model), weight in zip(self.models, self.weights):
            model_str = str(model).split('(')[0]
            print(f"  {model_str}: {weight:.3f}")
    
    def save(self, filepath: str):
        """
        Save ensemble model to file.
        
        Args:
            filepath: Path to save model
        """
        model_data = {
            'models': self.models,
            'weights': self.weights,
            'features': self.features,
            'is_fitted': self.is_fitted,
            'use_stacking': self.use_stacking,
            'meta_model': self.meta_model,
        }
        joblib.dump(model_data, filepath)
        print(f"EnsembleModel saved to {filepath}")
    
    def load(self, filepath: str):
        """
        Load ensemble model from file.
        
        Args:
            filepath: Path to load model from
        """
        model_data = joblib.load(filepath)
        self.models = model_data['models']
        self.weights = model_data['weights']
        self.features = model_data['features']
        self.is_fitted = model_data['is_fitted']
        self.use_stacking = model_data['use_stacking']
        self.meta_model = model_data['meta_model']
        print(f"EnsembleModel loaded from {filepath}")


if __name__ == '__main__':
    # Test module
    print("Testing EnsembleModel...")
    
    # Generate sample data
    np.random.seed(42)
    n_samples = 1000
    X = np.random.randn(n_samples, 5)
    y = X[:, 0] + X[:, 1] * 2 + np.random.randn(n_samples) * 0.5
    
    features = [f'feature_{i}' for i in range(5)]
    
    # Create base models
    from sklearn.linear_model import Ridge
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    
    models = [
        ('Ridge', Ridge(alpha=1.0)),
        ('RF', RandomForestRegressor(n_estimators=100, random_state=42)),
        ('GBT', GradientBoostingRegressor(n_estimators=100, random_state=42)),
    ]
    
    # Create ensemble
    ensemble = EnsembleModel(models)
    ensemble.fit_base_models(X, y, features)
    
    # Evaluate models
    results = ensemble.evaluate_models(X, y)
    print("\nBase model performance:")
    for name, metrics in results.items():
        print(f"  {name}: MAE={metrics['mae']:.4f}, Weight={metrics['weight']:.3f}")
    
    # Optimize weights
    ensemble.optimize_weights(X, y)
    
    # Predictions
    weighted_pred = ensemble.predict(X[:10], method='weighted')
    print(f"\nWeighted predictions (first 10): {weighted_pred}")
