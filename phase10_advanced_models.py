"""
Phase 10: Train Advanced Models
Train XGBoost and LightGBM models on enhanced features.
"""

import logging
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge

# Check for optional packages
try:
    import xgboost
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    import lightgbm
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

# Import classes if available
if XGBOOST_AVAILABLE:
    from xgboost import XGBRegressor
else:
    print("Warning: XGBoost not installed, skipping...")

if LIGHTGBM_AVAILABLE:
    from lightgbm import LGBMRegressor
else:
    print("Warning: LightGBM not installed, skipping...")

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class AdvancedModelTrainer:
    """
    Train and evaluate advanced ML models.
    """

    def __init__(self, features_path: str, models_dir: str = 'data/models'):
        self.features_path = features_path
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.features_df = None
        self.models = {}

    def load_data(self):
        """Load enhanced features."""
        logger.info(f"Loading enhanced features from {self.features_path}")
        self.features_df = pd.read_parquet(self.features_path)
        logger.info(f"Loaded {len(self.features_df)} games")
        return self

    def prepare_data(self):
        """Prepare features and targets for modeling."""
        logger.info("Preparing data for modeling...")

        # Sort by date and reset index
        df = self.features_df.sort_values('game_date').reset_index(drop=True).copy()

        # Define features to use
        # Exclude: identifiers, targets, post-game info
        exclude_cols = [
            'game_id', 'game_date', 'home_team_id', 'away_team_id',
            'home_score', 'away_score', 'total', 'margin'
        ]

        feature_cols = [col for col in df.columns if col not in exclude_cols]
        logger.info(f"Using {len(feature_cols)} features")

        # Targets
        y_total = df['total'].values
        y_margin = df['margin'].values

        # Features
        X = df[feature_cols].values

        # Time-based split (70% train, 15% val, 15% test)
        n = len(df)
        train_end = int(0.70 * n)
        val_end = int(0.85 * n)

        X_train, y_total_train, y_margin_train = X[:train_end], y_total[:train_end], y_margin[:train_end]
        X_val, y_total_val, y_margin_val = X[train_end:val_end], y_total[train_end:val_end], y_margin[train_end:val_end]
        X_test, y_total_test, y_margin_test = X[val_end:], y_total[val_end:], y_margin[val_end:]

        logger.info(f"Train: {len(X_train)} games")
        logger.info(f"Val: {len(X_val)} games")
        logger.info(f"Test: {len(X_test)} games")

        self.data = {
            'X_train': X_train, 'X_val': X_val, 'X_test': X_test,
            'y_total_train': y_total_train, 'y_total_val': y_total_val, 'y_total_test': y_total_test,
            'y_margin_train': y_margin_train, 'y_margin_val': y_margin_val, 'y_margin_test': y_margin_test,
            'feature_cols': feature_cols
        }

        return self

    def create_models(self):
        """Create model instances."""
        logger.info("Creating model instances...")

        # Linear models (baselines)
        self.model_configs = {
            'Linear': LinearRegression(),
            'Ridge': Ridge(alpha=1.0, random_state=42),
            'RandomForest': RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                random_state=42,
                n_jobs=-1
            )
        }
        
        # Add XGBoost if available
        if XGBOOST_AVAILABLE:
            self.model_configs['XGBoost'] = XGBRegressor(
                n_estimators=300,
                max_depth=5,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.1,
                reg_lambda=1.0,
                min_child_weight=3,
                random_state=42,
                n_jobs=-1,
                eval_metric='mae'
            )
        
        # Add LightGBM if available
        if LIGHTGBM_AVAILABLE:
            self.model_configs['LightGBM'] = LGBMRegressor(
                n_estimators=300,
                max_depth=5,
                learning_rate=0.05,
                num_leaves=31,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.1,
                reg_lambda=1.0,
                min_child_samples=20,
                random_state=42,
                n_jobs=-1,
                verbose=-1
            )

        for name, model in self.model_configs.items():
            logger.info(f"  ✓ {name}")

        return self

    def train_and_evaluate_models(self):
        """Train and evaluate all models."""
        logger.info("=" * 70)
        logger.info("Training and Evaluating Models")
        logger.info("=" * 70)

        X_train = self.data['X_train']
        X_val = self.data['X_val']
        X_test = self.data['X_test']
        y_total_train = self.data['y_total_train']
        y_total_val = self.data['y_total_val']
        y_total_test = self.data['y_total_test']
        y_margin_train = self.data['y_margin_train']
        y_margin_val = self.data['y_margin_val']
        y_margin_test = self.data['y_margin_test']

        # Train and evaluate total models
        logger.info("\n--- TOTAL POINTS MODELS ---")
        total_results = {}
        for name, model in self.model_configs.items():
            logger.info(f"\nTraining {name} (total)...")
            model.fit(X_train, y_total_train)

            # Predict
            y_train_pred = model.predict(X_train)
            y_val_pred = model.predict(X_val)
            y_test_pred = model.predict(X_test)

            # MAE
            train_mae = mean_absolute_error(y_total_train, y_train_pred)
            val_mae = mean_absolute_error(y_total_val, y_val_pred)
            test_mae = mean_absolute_error(y_total_test, y_test_pred)

            logger.info(f"  Train MAE: {train_mae:.2f}")
            logger.info(f"  Val MAE: {val_mae:.2f}")
            logger.info(f"  Test MAE: {test_mae:.2f}")

            # Save model
            model_path = self.models_dir / f"{name.lower()}_total_enhanced.pkl"
            joblib.dump(model, model_path)
            logger.info(f"  ✓ Saved to {model_path}")

            total_results[name] = {
                'train_mae': train_mae,
                'val_mae': val_mae,
                'test_mae': test_mae,
                'model': model
            }

        # Train and evaluate margin models
        logger.info("\n--- MARGIN MODELS ---")
        margin_results = {}
        for name, model in self.model_configs.items():
            logger.info(f"\nTraining {name} (margin)...")
            model.fit(X_train, y_margin_train)

            # Predict
            y_train_pred = model.predict(X_train)
            y_val_pred = model.predict(X_val)
            y_test_pred = model.predict(X_test)

            # MAE
            train_mae = mean_absolute_error(y_margin_train, y_train_pred)
            val_mae = mean_absolute_error(y_margin_val, y_val_pred)
            test_mae = mean_absolute_error(y_margin_test, y_test_pred)

            logger.info(f"  Train MAE: {train_mae:.2f}")
            logger.info(f"  Val MAE: {val_mae:.2f}")
            logger.info(f"  Test MAE: {test_mae:.2f}")

            # Save model
            model_path = self.models_dir / f"{name.lower()}_margin_enhanced.pkl"
            joblib.dump(model, model_path)
            logger.info(f"  ✓ Saved to {model_path}")

            margin_results[name] = {
                'train_mae': train_mae,
                'val_mae': val_mae,
                'test_mae': test_mae,
                'model': model
            }

        self.models['total'] = total_results
        self.models['margin'] = margin_results

        return self

    def print_results(self):
        """Print summary of results."""
        logger.info("\n" + "=" * 70)
        logger.info("PHASE 10: RESULTS SUMMARY")
        logger.info("=" * 70)

        logger.info("\n--- TOTAL POINTS ---")
        logger.info(f"{'Model':<15} {'Train MAE':<12} {'Val MAE':<12} {'Test MAE':<12}")
        logger.info("-" * 60)
        for name, results in sorted(self.models['total'].items(), key=lambda x: x[1]['val_mae']):
            logger.info(f"{name:<15} {results['train_mae']:<12.2f} {results['val_mae']:<12.2f} {results['test_mae']:<12.2f}")

        logger.info("\n--- MARGIN ---")
        logger.info(f"{'Model':<15} {'Train MAE':<12} {'Val MAE':<12} {'Test MAE':<12}")
        logger.info("-" * 60)
        for name, results in sorted(self.models['margin'].items(), key=lambda x: x[1]['val_mae']):
            logger.info(f"{name:<15} {results['train_mae']:<12.2f} {results['val_mae']:<12.2f} {results['test_mae']:<12.2f}")

        # Find best models
        best_total = min(self.models['total'].items(), key=lambda x: x[1]['val_mae'])
        best_margin = min(self.models['margin'].items(), key=lambda x: x[1]['val_mae'])

        logger.info("\n" + "=" * 70)
        logger.info("BEST MODELS (by validation MAE):")
        logger.info(f"  Total: {best_total[0]} (Val MAE: {best_total[1]['val_mae']:.2f}, Test MAE: {best_total[1]['test_mae']:.2f})")
        logger.info(f"  Margin: {best_margin[0]} (Val MAE: {best_margin[1]['val_mae']:.2f}, Test MAE: {best_margin[1]['test_mae']:.2f})")
        logger.info("=" * 70)

        return self


def main():
    """Run Phase 10."""
    # Paths
    features_path = 'data/processed/enhanced_features.parquet'
    models_dir = 'data/models'

    # Train models
    trainer = AdvancedModelTrainer(features_path, models_dir)
    trainer.load_data()
    trainer.prepare_data()
    trainer.create_models()
    trainer.train_and_evaluate_models()
    trainer.print_results()

    logger.info("\n" + "=" * 70)
    logger.info("PHASE 10: COMPLETE")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
