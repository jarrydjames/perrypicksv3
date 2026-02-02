"""
Phase 20: Complete Hyperparameter Tuning (Optimized)
Use smaller search spaces to complete tuning without timeout.
"""

import logging
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error

# Try to import XGBoost
try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

from skopt import BayesSearchCV
from skopt.space import Categorical, Integer, Real

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class FastHyperparameterTuner:
    """
    Tune hyperparameters with smaller search spaces.
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

        # Combine train + val for tuning (smaller but faster)
        X_tune = X[:val_end]
        y_total_tune = y_total[:val_end]
        y_margin_tune = y_margin[:val_end]

        self.data = {
            'X_train': X_train, 'X_val': X_val, 'X_test': X_test,
            'X_tune': X_tune,
            'y_total_train': y_total_train, 'y_total_val': y_total_val, 'y_total_test': y_total_test,
            'y_total_tune': y_total_tune,
            'y_margin_train': y_margin_train, 'y_margin_val': y_margin_val, 'y_margin_test': y_margin_test,
            'y_margin_tune': y_margin_tune,
            'feature_cols': feature_cols
        }

        logger.info(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}, Tune: {len(X_tune)}")
        return self

    def tune_ridge_fast(self):
        """Tune Ridge with small search space."""
        logger.info(f"\n{'='*70}")
        logger.info("Tuning Ridge (Fast - 10 iterations)...")
        logger.info(f"{'='*70}")

        y_total = self.data['y_total_tune']
        X = self.data['X_tune']

        # Smaller search space
        search_space = {
            'alpha': Real(0.1, 10.0, prior='log-uniform'),
        }

        model = Ridge(random_state=42)

        opt = BayesSearchCV(
            model,
            search_space,
            n_iter=10,  # Fewer iterations
            cv=3,
            scoring='neg_mean_absolute_error',
            n_jobs=1,
            random_state=42,
            verbose=0  # Reduce output
        )

        logger.info("Running optimization (10 iterations, 3-fold CV)...")
        opt.fit(X, y_total)

        logger.info(f"✓ Best params: {opt.best_params_}")
        logger.info(f"✓ Best CV MAE: {-opt.best_score_:.2f}")

        return opt.best_estimator_, -opt.best_score_

    def tune_random_forest_fast(self):
        """Tune Random Forest with small search space."""
        logger.info(f"\n{'='*70}")
        logger.info("Tuning Random Forest (Fast - 10 iterations)...")
        logger.info(f"{'='*70}")

        y_total = self.data['y_total_tune']
        X = self.data['X_tune']

        # Smaller, focused search space
        search_space = {
            'n_estimators': Integer(50, 150),
            'max_depth': Integer(5, 15),
            'min_samples_split': Integer(5, 15),
            'min_samples_leaf': Integer(1, 4),
            'max_features': Categorical(['sqrt', 0.5]),
        }

        model = RandomForestRegressor(
            random_state=42,
            n_jobs=-1
        )

        opt = BayesSearchCV(
            model,
            search_space,
            n_iter=10,  # Fewer iterations
            cv=3,  # Fewer CV folds for speed
            scoring='neg_mean_absolute_error',
            n_jobs=1,
            random_state=42,
            verbose=0
        )

        logger.info("Running optimization (10 iterations, 3-fold CV)...")
        opt.fit(X, y_total)

        logger.info(f"✓ Best params: {opt.best_params_}")
        logger.info(f"✓ Best CV MAE: {-opt.best_score_:.2f}")

        return opt.best_estimator_, -opt.best_score_

    def tune_xgboost_fast(self):
        """Tune XGBoost with small search space."""
        if not XGBOOST_AVAILABLE:
            logger.warning("XGBoost not available, skipping...")
            return None, None

        logger.info(f"\n{'='*70}")
        logger.info("Tuning XGBoost (Fast - 10 iterations)...")
        logger.info(f"{'='*70}")

        y_total = self.data['y_total_tune']
        X = self.data['X_tune']

        # Very focused search space
        search_space = {
            'n_estimators': Integer(100, 200),
            'max_depth': Integer(4, 8),
            'learning_rate': Real(0.05, 0.2, prior='log-uniform'),
            'subsample': Real(0.7, 1.0),
        }

        model = XGBRegressor(
            random_state=42,
            n_jobs=-1,
            eval_metric='mae',
            verbosity=0
        )

        opt = BayesSearchCV(
            model,
            search_space,
            n_iter=10,
            cv=3,
            scoring='neg_mean_absolute_error',
            n_jobs=1,
            random_state=42,
            verbose=0
        )

        logger.info("Running optimization (10 iterations, 3-fold CV)...")
        opt.fit(X, y_total)

        logger.info(f"✓ Best params: {opt.best_params_}")
        logger.info(f"✓ Best CV MAE: {-opt.best_score_:.2f}")

        return opt.best_estimator_, -opt.best_score_

    def tune_all_fast(self):
        """Tune all models with small search spaces."""
        logger.info("=" * 70)
        logger.info("PHASE 20: Fast Hyperparameter Tuning")
        logger.info("=" * 70)

        results = {}

        # ===== TOTAL MODELS =====
        logger.info("\n" + "=" * 70)
        logger.info("TOTAL POINTS MODELS")
        logger.info("=" * 70)

        # Ridge
        ridge_total, ridge_mae = self.tune_ridge_fast()
        if ridge_total is not None:
            results['ridge_total_tuned_fast'] = {
                'model': ridge_total,
                'cv_mae': ridge_mae
            }
            path = self.models_dir / 'ridge_total_tuned_fast.pkl'
            joblib.dump(ridge_total, path)
            logger.info(f"✓ Saved to {path}")

        # RandomForest
        rf_total, rf_mae = self.tune_random_forest_fast()
        if rf_total is not None:
            results['rf_total_tuned_fast'] = {
                'model': rf_total,
                'cv_mae': rf_mae
            }
            path = self.models_dir / 'randomforest_total_tuned_fast.pkl'
            joblib.dump(rf_total, path)
            logger.info(f"✓ Saved to {path}")

        # XGBoost
        xgb_total, xgb_mae = self.tune_xgboost_fast()
        if xgb_total is not None:
            results['xgb_total_tuned_fast'] = {
                'model': xgb_total,
                'cv_mae': xgb_mae
            }
            path = self.models_dir / 'xgboost_total_tuned_fast.pkl'
            joblib.dump(xgb_total, path)
            logger.info(f"✓ Saved to {path}")

        self.results = results
        return self

    def print_results(self):
        """Print tuning results."""
        logger.info("\n" + "=" * 70)
        logger.info("FAST TUNING RESULTS")
        logger.info("=" * 70)

        logger.info(f"\n{'Model':<30} {'CV MAE':<12}")
        logger.info("-" * 45)
        for name, data in self.results.items():
            logger.info(f"{name:<30} {data['cv_mae']:<12.2f}")

        # Best model
        best = min(self.results.items(), key=lambda x: x[1]['cv_mae'])
        logger.info("\n" + "=" * 70)
        logger.info(f"BEST MODEL: {best[0]} (CV MAE: {best[1]['cv_mae']:.2f})")
        logger.info("=" * 70)

        return self


def main():
    """Run Phase 20."""
    features_path = 'data/processed/final_features.parquet'
    models_dir = 'data/models'

    tuner = FastHyperparameterTuner(features_path, models_dir)
    tuner.load_data()
    tuner.prepare_data()
    tuner.tune_all_fast()
    tuner.print_results()

    logger.info("\n" + "=" * 70)
    logger.info("PHASE 20: COMPLETE")
    logger.info("=" * 70)
    logger.info("\nNote: Used smaller search spaces (10 iterations, 3-fold CV)")
    logger.info("      to complete tuning without timeout.")


if __name__ == "__main__":
    main()
