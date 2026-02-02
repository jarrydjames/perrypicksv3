"""
Phase 22: Ensemble Model
Combine multiple models for better predictions.
"""

import logging
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

# Try to import XGBoost
try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class EnsembleBuilder:
    """Build ensemble models."""

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

    def train_base_models(self):
        """Train base models."""
        logger.info("\n" + "=" * 70)
        logger.info("TRAINING BASE MODELS")
        logger.info("=" * 70)

        X_train = self.data['X_train']
        X_val = self.data['X_val']
        y_total_train = self.data['y_total_train']
        y_total_val = self.data['y_total_val']

        self.base_models = {}

        # Ridge
        logger.info("\n1. Training Ridge...")
        ridge = Ridge(alpha=8.15, random_state=42)
        ridge.fit(X_train, y_total_train)
        ridge_pred = ridge.predict(X_val)
        ridge_mae = np.mean(np.abs(ridge_pred - y_total_val))
        self.base_models['ridge'] = {'model': ridge, 'predictions': ridge_pred, 'mae': ridge_mae}
        logger.info(f"   Ridge MAE: {ridge_mae:.2f}")

        # Random Forest
        logger.info("2. Training Random Forest...")
        rf = RandomForestRegressor(n_estimators=117, max_depth=9, random_state=42, n_jobs=-1)
        rf.fit(X_train, y_total_train)
        rf_pred = rf.predict(X_val)
        rf_mae = np.mean(np.abs(rf_pred - y_total_val))
        self.base_models['rf'] = {'model': rf, 'predictions': rf_pred, 'mae': rf_mae}
        logger.info(f"   Random Forest MAE: {rf_mae:.2f}")

        # XGBoost
        if XGBOOST_AVAILABLE:
            logger.info("3. Training XGBoost...")
            xgb = XGBRegressor(
                n_estimators=174, max_depth=7, learning_rate=0.05,
                subsample=0.78, random_state=42, n_jobs=-1, eval_metric='mae', verbosity=0
            )
            xgb.fit(X_train, y_total_train)
            xgb_pred = xgb.predict(X_val)
            xgb_mae = np.mean(np.abs(xgb_pred - y_total_val))
            self.base_models['xgb'] = {'model': xgb, 'predictions': xgb_pred, 'mae': xgb_mae}
            logger.info(f"   XGBoost MAE: {xgb_mae:.2f}")

        return self

    def build_simple_average_ensemble(self):
        """Build simple average ensemble."""
        logger.info("\n" + "=" * 70)
        logger.info("BUILDING SIMPLE AVERAGE ENSEMBLE")
        logger.info("=" * 70)

        # Collect predictions
        preds = [self.base_models[k]['predictions'] for k in self.base_models.keys()]
        avg_pred = np.mean(preds, axis=0)

        avg_mae = np.mean(np.abs(avg_pred - self.data['y_total_val']))
        logger.info(f"\nSimple Average Ensemble MAE: {avg_mae:.2f}")

        self.simple_ensemble_mae = avg_mae
        return self

    def build_weighted_ensemble(self):
        """Build weighted ensemble based on validation performance."""
        logger.info("\n" + "=" * 70)
        logger.info("BUILDING WEIGHTED ENSEMBLE")
        logger.info("=" * 70)

        # Calculate weights (inverse of MAE)
        maes = np.array([self.base_models[k]['mae'] for k in self.base_models.keys()])
        weights = 1 / maes
        weights = weights / weights.sum()  # Normalize

        logger.info("\nModel Weights:")
        for i, (k, w) in enumerate(zip(self.base_models.keys(), weights)):
            logger.info(f"  {k}: {w:.3f} (MAE: {self.base_models[k]['mae']:.2f})")

        # Weighted prediction
        preds = [self.base_models[k]['predictions'] for k in self.base_models.keys()]
        weighted_pred = np.average(preds, axis=0, weights=weights)

        weighted_mae = np.mean(np.abs(weighted_pred - self.data['y_total_val']))
        logger.info(f"\nWeighted Ensemble MAE: {weighted_mae:.2f}")

        self.weights = dict(zip(self.base_models.keys(), weights))
        self.weighted_ensemble_mae = weighted_mae
        return self

    def build_meta_learner(self):
        """Build meta-learner ensemble."""
        logger.info("\n" + "=" * 70)
        logger.info("BUILDING META-LEARNER ENSEMBLE")
        logger.info("=" * 70)

        # Create meta-features (base model predictions)
        meta_train = np.column_stack([
            self.base_models[k]['model'].predict(self.data['X_train'])
            for k in self.base_models.keys()
        ])

        meta_val = np.column_stack([
            self.base_models[k]['predictions']
            for k in self.base_models.keys()
        ])

        # Train meta-learner (linear regression)
        meta = LinearRegression()
        meta.fit(meta_train, self.data['y_total_train'])
        meta_pred = meta.predict(meta_val)

        meta_mae = np.mean(np.abs(meta_pred - self.data['y_total_val']))

        logger.info(f"\nMeta-Learner (Linear Regression) MAE: {meta_mae:.2f}")
        logger.info(f"  Coefficients: {meta.coef_}")

        self.meta_learner = meta
        self.meta_ensemble_mae = meta_mae
        return self

    def compare_all(self):
        """Compare all approaches."""
        logger.info("\n" + "=" * 70)
        logger.info("ENSEMBLE COMPARISON")
        logger.info("=" * 70)

        logger.info("\nBase Models:")
        for k, v in self.base_models.items():
            logger.info(f"  {k.upper()}: {v['mae']:.2f}")

        logger.info("\nEnsemble Methods:")
        logger.info(f"  Simple Average: {self.simple_ensemble_mae:.2f}")
        logger.info(f"  Weighted Average: {self.weighted_ensemble_mae:.2f}")
        logger.info(f"  Meta-Learner: {self.meta_ensemble_mae:.2f}")

        # Best approach
        all_maes = [
            ('ridge', self.base_models['ridge']['mae']),
            ('rf', self.base_models['rf']['mae']),
            ('simple_avg', self.simple_ensemble_mae),
            ('weighted_avg', self.weighted_ensemble_mae),
            ('meta_learner', self.meta_ensemble_mae),
        ]
        if XGBOOST_AVAILABLE:
            all_maes.append(('xgb', self.base_models['xgb']['mae']))

        best = min(all_maes, key=lambda x: x[1])
        logger.info(f"\nBEST: {best[0]} (MAE: {best[1]:.2f})")

        # Calculate improvement
        base_mae = self.base_models['ridge']['mae']
        improvement = (base_mae - best[1]) / base_mae * 100
        logger.info(f"\nImprovement over best base model: {improvement:.2f}%")

        return self

    def save_ensemble(self):
        """Save best ensemble model."""
        # Save weights for weighted ensemble
        output_path = self.models_dir / 'ensemble_weights.pkl'
        joblib.dump(self.weights, output_path)
        logger.info(f"\n✓ Saved ensemble weights to {output_path}")

        # Save meta-learner
        output_path = self.models_dir / 'meta_learner.pkl'
        joblib.dump(self.meta_learner, output_path)
        logger.info(f"✓ Saved meta-learner to {output_path}")

        return self


def main():
    """Run Phase 22."""
    features_path = 'data/processed/final_features.parquet'
    models_dir = 'data/models'

    ensemble = EnsembleBuilder(features_path, models_dir)
    ensemble.load_data()
    ensemble.prepare_data()
    ensemble.train_base_models()
    ensemble.build_simple_average_ensemble()
    ensemble.build_weighted_ensemble()
    ensemble.build_meta_learner()
    ensemble.compare_all()
    ensemble.save_ensemble()

    logger.info("\n" + "=" * 70)
    logger.info("PHASE 22: COMPLETE")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
