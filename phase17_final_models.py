"""
Phase 17: Train Final Models with All Features
Train models on the complete feature set (80 features).
"""

import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error

# Try to import XGBoost and LightGBM
try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    from lightgbm import LGBMRegressor
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class FinalModelTrainer:
    """
    Train models on final feature set.
    """

    def __init__(self, features_path: str, models_dir: str = 'data/models'):
        self.features_path = features_path
        self.models_dir = Path(models_dir)
        self.features_df = None
        self.data = None

    def load_data(self):
        """Load final features."""
        logger.info(f"Loading final features from {self.features_path}")
        self.features_df = pd.read_parquet(self.features_path)
        logger.info(f"Loaded {len(self.features_df)} games")
        return self

    def prepare_data(self):
        """Prepare features and targets."""
        logger.info("Preparing data...")

        df = self.features_df.sort_values('game_date').reset_index(drop=True).copy()

        exclude_cols = [
            'game_id', 'game_date', 'home_team_id', 'away_team_id',
            'home_score', 'away_score', 'total', 'margin'
        ]

        feature_cols = [col for col in df.columns if col not in exclude_cols]
        logger.info(f"Using {len(feature_cols)} features")

        y_total = df['total'].values
        y_margin = df['margin'].values
        X = df[feature_cols].values

        n = len(df)
        train_end = int(0.70 * n)
        val_end = int(0.85 * n)

        X_train = X[:train_end]
        X_val = X[train_end:val_end]
        X_test = X[val_end:]

        y_total_train = y_total[:train_end]
        y_total_val = y_total[train_end:val_end]
        y_total_test = y_total[val_end:]

        y_margin_train = y_margin[:train_end]
        y_margin_val = y_margin[train_end:val_end]
        y_margin_test = y_margin[val_end:]

        self.data = {
            'X_train': X_train, 'X_val': X_val, 'X_test': X_test,
            'y_total_train': y_total_train, 'y_total_val': y_total_val, 'y_total_test': y_total_test,
            'y_margin_train': y_margin_train, 'y_margin_val': y_margin_val, 'y_margin_test': y_margin_test,
            'feature_cols': feature_cols
        }

        logger.info(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
        return self

    def train_model(self, model, model_name, target='total'):
        """Train and evaluate a single model."""
        if target == 'total':
            y_train = self.data['y_total_train']
            y_val = self.data['y_total_val']
            y_test = self.data['y_total_test']
        else:
            y_train = self.data['y_margin_train']
            y_val = self.data['y_margin_val']
            y_test = self.data['y_margin_test']

        X_train = self.data['X_train']
        X_val = self.data['X_val']
        X_test = self.data['X_test']

        logger.info(f"\nTraining {model_name} ({target})...")

        model.fit(X_train, y_train)

        train_pred = model.predict(X_train)
        val_pred = model.predict(X_val)
        test_pred = model.predict(X_test)

        train_mae = mean_absolute_error(y_train, train_pred)
        val_mae = mean_absolute_error(y_val, val_pred)
        test_mae = mean_absolute_error(y_test, test_pred)

        logger.info(f"  Train MAE: {train_mae:.2f}")
        logger.info(f"  Val MAE: {val_mae:.2f}")
        logger.info(f"  Test MAE: {test_mae:.2f}")

        # Save model
        filename = f"{model_name}_{target}_final.pkl"
        path = self.models_dir / filename
        joblib.dump(model, path)
        logger.info(f"  ✓ Saved to {path}")

        return {'train_mae': train_mae, 'val_mae': val_mae, 'test_mae': test_mae}

    def train_all_models(self):
        """Train all models."""
        logger.info("=" * 70)
        logger.info("PHASE 17: Training Final Models with 80 Features")
        logger.info("=" * 70)

        results = {}

        # ===== TOTAL MODELS =====
        logger.info("\n" + "=" * 70)
        logger.info("TOTAL POINTS MODELS")
        logger.info("=" * 70)

        # Linear
        results['linear_total'] = self.train_model(
            LinearRegression(), 'linear', 'total'
        )

        # Ridge
        results['ridge_total'] = self.train_model(
            Ridge(alpha=1.0, random_state=42), 'ridge', 'total'
        )

        # RandomForest
        results['rf_total'] = self.train_model(
            RandomForestRegressor(
                n_estimators=100, max_depth=10, random_state=42, n_jobs=-1
            ),
            'rf', 'total'
        )

        # XGBoost (if available)
        if XGBOOST_AVAILABLE:
            results['xgb_total'] = self.train_model(
                XGBRegressor(
                    n_estimators=100, max_depth=6, learning_rate=0.1,
                    random_state=42, n_jobs=-1, eval_metric='mae'
                ),
                'xgboost', 'total'
            )

        # LightGBM (if available)
        if LIGHTGBM_AVAILABLE:
            results['lgb_total'] = self.train_model(
                LGBMRegressor(
                    n_estimators=100, max_depth=6, learning_rate=0.1,
                    random_state=42, n_jobs=-1, verbose=-1
                ),
                'lightgbm', 'total'
            )

        # ===== MARGIN MODELS =====
        logger.info("\n" + "=" * 70)
        logger.info("MARGIN MODELS")
        logger.info("=" * 70)

        # Linear
        results['linear_margin'] = self.train_model(
            LinearRegression(), 'linear', 'margin'
        )

        # Ridge
        results['ridge_margin'] = self.train_model(
            Ridge(alpha=1.0, random_state=42), 'ridge', 'margin'
        )

        # RandomForest
        results['rf_margin'] = self.train_model(
            RandomForestRegressor(
                n_estimators=100, max_depth=10, random_state=42, n_jobs=-1
            ),
            'rf', 'margin'
        )

        # XGBoost (if available)
        if XGBOOST_AVAILABLE:
            results['xgb_margin'] = self.train_model(
                XGBRegressor(
                    n_estimators=100, max_depth=6, learning_rate=0.1,
                    random_state=42, n_jobs=-1, eval_metric='mae'
                ),
                'xgboost', 'margin'
            )

        # LightGBM (if available)
        if LIGHTGBM_AVAILABLE:
            results['lgb_margin'] = self.train_model(
                LGBMRegressor(
                    n_estimators=100, max_depth=6, learning_rate=0.1,
                    random_state=42, n_jobs=-1, verbose=-1
                ),
                'lightgbm', 'margin'
            )

        self.results = results
        return self

    def print_results(self):
        """Print training results."""
        logger.info("\n" + "=" * 70)
        logger.info("FINAL MODEL RESULTS")
        logger.info("=" * 70)

        logger.info("\n--- TOTAL POINTS ---")
        logger.info(f"{'Model':<15} {'Train MAE':<12} {'Val MAE':<12} {'Test MAE':<12}")
        logger.info("-" * 55)
        for name in ['linear_total', 'ridge_total', 'rf_total', 'xgb_total', 'lgb_total']:
            if name in self.results:
                r = self.results[name]
                logger.info(f"{name:<15} {r['train_mae']:<12.2f} {r['val_mae']:<12.2f} {r['test_mae']:<12.2f}")

        logger.info("\n--- MARGIN ---")
        logger.info(f"{'Model':<15} {'Train MAE':<12} {'Val MAE':<12} {'Test MAE':<12}")
        logger.info("-" * 55)
        for name in ['linear_margin', 'ridge_margin', 'rf_margin', 'xgb_margin', 'lgb_margin']:
            if name in self.results:
                r = self.results[name]
                logger.info(f"{name:<15} {r['train_mae']:<12.2f} {r['val_mae']:<12.2f} {r['test_mae']:<12.2f}")

        # Best models
        best_total = min(
            [(name, data) for name, data in self.results.items() if 'total' in name],
            key=lambda x: x[1]['test_mae']
        )
        best_margin = min(
            [(name, data) for name, data in self.results.items() if 'margin' in name],
            key=lambda x: x[1]['test_mae']
        )

        logger.info("\n" + "=" * 70)
        logger.info("BEST MODELS (by test MAE):")
        logger.info(f"  Total: {best_total[0]} (Test MAE: {best_total[1]['test_mae']:.2f})")
        logger.info(f"  Margin: {best_margin[0]} (Test MAE: {best_margin[1]['test_mae']:.2f})")
        logger.info("=" * 70)

        return self


def main():
    """Run Phase 17."""
    features_path = 'data/processed/final_features.parquet'
    models_dir = 'data/models'

    trainer = FinalModelTrainer(features_path, models_dir)
    trainer.load_data()
    trainer.prepare_data()
    trainer.train_all_models()
    trainer.print_results()

    logger.info("\n" + "=" * 70)
    logger.info("PHASE 17: COMPLETE")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
