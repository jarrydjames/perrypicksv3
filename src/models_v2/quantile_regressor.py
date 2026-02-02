"""Quantile regression for prediction intervals.

Uses GradientBoostingRegressor with quantile loss to predict:
- Lower bound (10th percentile)
- Median prediction (50th percentile)
- Upper bound (90th percentile)
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import QuantileRegressor as SklearnQuantileRegressor
import joblib


class QuantileRegressor:
    """Quantile regression model for prediction intervals."""
    
    def __init__(self, quantiles: List[float] = [0.1, 0.5, 0.9]):
        """
        Initialize quantile regressor.
        
        Args:
            quantiles: List of quantiles to predict (default: 0.1, 0.5, 0.9)
        """
        self.quantiles = quantiles
        self.models = {}
        self.features = None
        self.is_fitted = False
    
    def fit(self, X: np.ndarray, y: np.ndarray, features: List[str]):
        """
        Fit quantile regression models.
        
        Args:
            X: Feature matrix
            y: Target values
            features: Feature names
        """
        self.features = features
        
        # Fit a model for each quantile
        for q in self.quantiles:
            if q == 0.5:
                # Use standard GBDT for median
                self.models[q] = GradientBoostingRegressor(
                    n_estimators=100,
                    max_depth=6,
                    learning_rate=0.1,
                    loss='quantile',
                    alpha=0.5,
                    random_state=42,
                )
            else:
                # Use GradientBoostingRegressor with quantile loss
                self.models[q] = GradientBoostingRegressor(
                    n_estimators=100,
                    max_depth=6,
                    learning_rate=0.1,
                    loss='quantile',
                    alpha=q,
                    random_state=42,
                )
            
            print(f"Fitting quantile {q:.1f} model...")
            self.models[q].fit(X, y)
        
        self.is_fitted = True
        print("All quantile models fitted")
    
    def predict(self, X: np.ndarray) -> Dict[float, np.ndarray]:
        """
        Predict for all quantiles.
        
        Args:
            X: Feature matrix
            
        Returns:
            Dictionary mapping quantile to predictions
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        predictions = {}
        for q in self.quantiles:
            predictions[q] = self.models[q].predict(X)
        
        return predictions
    
    def predict_interval(self, X: np.ndarray, alpha: float = 0.8) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Predict with confidence interval.
        
        Args:
            X: Feature matrix
            alpha: Confidence level (default: 0.8 for 80% CI)
            
        Returns:
            Tuple of (median, lower_bound, upper_bound)
        """
        pred_dict = self.predict(X)
        
        median = pred_dict[0.5]
        lower = pred_dict[self.quantiles[0]]
        upper = pred_dict[self.quantiles[-1]]
        
        return median, lower, upper
    
    def get_prediction_interval_width(self, X: np.ndarray) -> np.ndarray:
        """
        Get width of prediction intervals.
        
        Args:
            X: Feature matrix
            
        Returns:
            Array of interval widths
        """
        _, lower, upper = self.predict_interval(X)
        return upper - lower
    
    def save(self, filepath: str):
        """
        Save model to file.
        
        Args:
            filepath: Path to save model
        """
        model_data = {
            'quantiles': self.quantiles,
            'features': self.features,
            'models': self.models,
            'is_fitted': self.is_fitted,
        }
        joblib.dump(model_data, filepath)
        print(f"QuantileRegressor saved to {filepath}")
    
    def load(self, filepath: str):
        """
        Load model from file.
        
        Args:
            filepath: Path to load model from
        """
        model_data = joblib.load(filepath)
        self.quantiles = model_data['quantiles']
        self.features = model_data['features']
        self.models = model_data['models']
        self.is_fitted = model_data['is_fitted']
        print(f"QuantileRegressor loaded from {filepath}")
    
    def evaluate_coverage(self, X: np.ndarray, y: np.ndarray, alpha: float = 0.8) -> Dict[str, float]:
        """
        Evaluate prediction interval coverage.
        
        Args:
            X: Feature matrix
            y: True values
            alpha: Target confidence level
            
        Returns:
            Dictionary with coverage metrics
        """
        median, lower, upper = self.predict_interval(X)
        
        # Calculate coverage
        in_interval = (y >= lower) & (y <= upper)
        coverage = np.mean(in_interval)
        
        # Calculate PICP (Prediction Interval Coverage Probability)
        picp = coverage
        
        # Calculate NMPIW (Normalized Mean Prediction Interval Width)
        mpiw = np.mean(upper - lower)
        y_range = np.max(y) - np.min(y)
        nmpiw = mpiw / y_range if y_range > 0 else mpiw
        
        # Calculate CWC (Coverage Width Criterion)
        # Penalizes under-coverage and wide intervals
        gamma = 0.1  # Penalty parameter
        eta = picp - alpha
        cwc = nmpiw * (1 + gamma * np.exp(gamma * eta))
        
        return {
            'coverage': coverage,
            'picp': picp,
            'mpiw': mpiw,
            'nmpiw': nmpiw,
            'cwc': cwc,
            'target_coverage': alpha,
        }


class BayesianQuantileRegressor:
    """Bayesian approach to quantile regression using ensemble."""
    
    def __init__(self, n_estimators: int = 100, quantiles: List[float] = [0.1, 0.5, 0.9]):
        """
        Initialize Bayesian quantile regressor.
        
        Args:
            n_estimators: Number of estimators in ensemble
            quantiles: List of quantiles to predict
        """
        self.n_estimators = n_estimators
        self.quantiles = quantiles
        self.estimators = []
        self.features = None
        self.is_fitted = False
    
    def fit(self, X: np.ndarray, y: np.ndarray, features: List[str]):
        """
        Fit ensemble of quantile regressors.
        
        Args:
            X: Feature matrix
            y: Target values
            features: Feature names
        """
        self.features = features
        
        # Train multiple estimators for each quantile
        for i in range(self.n_estimators):
            est_dict = {}
            for q in self.quantiles:
                # Use subsample for each estimator
                n_samples = int(0.8 * len(X))
                indices = np.random.choice(len(X), n_samples, replace=False)
                
                model = GradientBoostingRegressor(
                    n_estimators=50,
                    max_depth=4,
                    learning_rate=0.1,
                    loss='quantile',
                    alpha=q,
                    random_state=i,
                )
                model.fit(X[indices], y[indices])
                est_dict[q] = model
            
            self.estimators.append(est_dict)
            
            if (i + 1) % 20 == 0:
                print(f"Trained {i + 1}/{self.n_estimators} estimators")
        
        self.is_fitted = True
        print(f"BayesianQuantileRegressor fitted with {self.n_estimators} estimators")
    
    def predict(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Predict with uncertainty estimates.
        
        Args:
            X: Feature matrix
            
        Returns:
            Dictionary with 'median', 'mean', 'std', 'lower', 'upper'
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        # Collect predictions from all estimators
        all_predictions = {q: [] for q in self.quantiles}
        
        for est_dict in self.estimators:
            for q in self.quantiles:
                all_predictions[q].append(est_dict[q].predict(X))
        
        # Aggregate predictions
        predictions = {}
        
        # Median predictions (50th percentile)
        median_preds = np.array(all_predictions[0.5])
        predictions['median'] = np.median(median_preds, axis=0)
        predictions['median_std'] = np.std(median_preds, axis=0)
        
        # Lower and upper bounds
        lower_preds = np.array(all_predictions[self.quantiles[0]])
        predictions['lower'] = np.mean(lower_preds, axis=0)
        predictions['lower_std'] = np.std(lower_preds, axis=0)
        
        upper_preds = np.array(all_predictions[self.quantiles[-1]])
        predictions['upper'] = np.mean(upper_preds, axis=0)
        predictions['upper_std'] = np.std(upper_preds, axis=0)
        
        return predictions
    
    def get_uncertainty(self, X: np.ndarray) -> np.ndarray:
        """
        Get prediction uncertainty (standard deviation of median predictions).
        
        Args:
            X: Feature matrix
            
        Returns:
            Array of uncertainty values
        """
        pred_dict = self.predict(X)
        return pred_dict['median_std']


if __name__ == '__main__':
    # Test the module
    print("Testing QuantileRegressor...")
    
    # Generate sample data
    np.random.seed(42)
    n_samples = 1000
    X = np.random.randn(n_samples, 5)
    y = X[:, 0] + X[:, 1] * 2 + np.random.randn(n_samples) * 0.5
    
    features = [f'feature_{i}' for i in range(5)]
    
    # Train model
    qr = QuantileRegressor(quantiles=[0.1, 0.5, 0.9])
    qr.fit(X, y, features)
    
    # Predict
    median, lower, upper = qr.predict_interval(X[:10])
    print(f"\nPredictions for first 10 samples:")
    print(f"Median: {median}")
    print(f"Lower:  {lower}")
    print(f"Upper:  {upper}")
    print(f"Interval widths: {upper - lower}")
    
    # Evaluate coverage
    coverage = qr.evaluate_coverage(X, y, alpha=0.8)
    print(f"\nCoverage metrics:")
    for key, val in coverage.items():
        print(f"  {key}: {val:.4f}")
