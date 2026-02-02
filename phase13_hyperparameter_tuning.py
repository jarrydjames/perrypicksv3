"""
Phase 13: Hyperparameter Tuning
Optimize model parameters using Bayesian optimization.
"""

import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from skopt import BayesSearchCV
from skopt.space import Categorical, Integer, Real
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


class HyperparameterTuner:
    """
    Tune hyperparameters using Bayesian optimization.
    """

    def __init__(self, features_path: str, models_dir: str = 'data/models'):
        self.features_path = features_path
        self.models_dir = Path(models_dir)
        self.features_df = None
        self.data = None

    def load_data(self):
        """Load enhanced features."""
        logger.info(f"Loading enhanced features from {self.features_path}")
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

        # Combine train + val for tuning
        X_tune = np.vstack([X_train, X_val])
        y_total_tune = np.concatenate([y_total_train, y_total_val])
        y_margin_tune = np.concatenate([y_margin_train, y_margin_val])

        self.data = {
            'X_train': X_train, 'X_val': X_val, 'X_test': X_test,
            'X_tune': X_tune,
            'y_total_train': y_total_train, 'y_total_val': y_total_val, 'y_total_test': y_total_test,
            'y_total_tune': y_total_tune,
            'y_margin_train': y_margin_train, 'y_margin_val': y_margin_val, 'y_margin_test': y_margin_test,
            'y_margin_tune': y_margin_tune,
            'feature_cols': feature_cols
        }

        logger.info(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
        logger.info(f"Tune set: {len(X_tune)} (train + val)")
        return self

    def tune_ridge(self, target='total'):
        """Tune Ridge regression."""
        logger.info(f"\n{'='*70}")
        logger.info(f"Tuning Ridge ({target})...")
        logger.info(f"{'='*70}")

        if target == 'total':
            y = self.data['y_total_tune']
        else:
            y = self.data['y_margin_tune']

        X = self.data['X_tune']

        # Search space
        search_space = {
            'alpha': Real(0.01, 10.0, prior='log-uniform'),
        }

        model = Ridge(random_state=42)

        opt = BayesSearchCV(
            model,
            search_space,
            n_iter=30,
            cv=5,
            scoring='neg_mean_absolute_error',
            n_jobs=1,  # Ridge is fast, parallelize over CV
            random_state=42,
            verbose=1
        )

        logger.info("Running Bayesian optimization...")
        opt.fit(X, y)

        logger.info(f"Best params: {opt.best_params_}")
        logger.info(f"Best CV score: {-opt.best_score_:.2f}")

        # Evaluate on test set
        if target == 'total':
            y_test = self.data['y_total_test']
        else:
            y_test = self.data['y_margin_test']

        y_pred = opt.best_estimator_.predict(self.data['X_test'])
        test_mae = mean_absolute_error(y_test, y_pred)
        logger.info(f"Test MAE: {test_mae:.2f}")

        return opt.best_estimator_, test_mae

    def tune_random_forest(self, target='total'):
        """Tune Random Forest."""
        logger.info(f"\n{'='*70}")
        logger.info(f"Tuning Random Forest ({target})...")
        logger.info(f"{'='*70}")

        if target == 'total':
            y = self.data['y_total_tune']
        else:
            y = self.data['y_margin_tune']

        X = self.data['X_tune']

        # Search space
        search_space = {
            'n_estimators': Integer(50, 300),
            'max_depth': Integer(5, 20),
            'min_samples_split': Integer(2, 10),
            'min_samples_leaf': Integer(1, 5),
            'max_features': Categorical(['sqrt', 'log2', None]),
        }

        model = RandomForestRegressor(
            random_state=42,
            n_jobs=-1
        )

        opt = BayesSearchCV(
            model,
            search_space,
            n_iter=20,  # Fewer iterations for RF (slower)
            cv=3,  # Fewer CV folds for RF (faster)
            scoring='neg_mean_absolute_error',
            n_jobs=1,
            random_state=42,
            verbose=1
        )

        logger.info("Running Bayesian optimization...")
        opt.fit(X, y)

        logger.info(f"Best params: {opt.best_params_}")
        logger.info(f"Best CV score: {-opt.best_score_:.2f}")

        # Evaluate on test set
        if target == 'total':
            y_test = self.data['y_total_test']
        else:
            y_test = self.data['y_margin_test']

        y_pred = opt.best_estimator_.predict(self.data['X_test'])
        test_mae = mean_absolute_error(y_test, y_pred)
        logger.info(f"Test MAE: {test_mae:.2f}")

        return opt.best_estimator_, test_mae

    def tune_xgboost(self, target='total'):
        """Tune XGBoost (if available)."""
        if not XGBOOST_AVAILABLE:
            logger.warning("XGBoost not available, skipping...")
            return None, None

        logger.info(f"\n{'='*70}")
        logger.info(f"Tuning XGBoost ({target})...")
        logger.info(f"{'='*70}")

        if target == 'total':
            y = self.data['y_total_tune']
        else:
            y = self.data['y_margin_tune']

        X = self.data['X_tune']

        # Search space
        search_space = {
            'n_estimators': Integer(100, 500),
            'max_depth': Integer(3, 10),
            'learning_rate': Real(0.01, 0.3, prior='log-uniform'),
            'subsample': Real(0.6, 1.0),
            'colsample_bytree': Real(0.6, 1.0),
            'reg_alpha': Real(0.0, 1.0),
            'reg_lambda': Real(0.5, 5.0),
            'min_child_weight': Integer(1, 7),
        }

        model = XGBRegressor(
            random_state=42,
            n_jobs=-1,
            eval_metric='mae'
        )

        opt = BayesSearchCV(
            model,
            search_space,
            n_iter=30,
            cv=3,
            scoring='neg_mean_absolute_error',
            n_jobs=1,
            random_state=42,
            verbose=1
        )

        logger.info("Running Bayesian optimization...")
        opt.fit(X, y)

        logger.info(f"Best params: {opt.best_params_}")
        logger.info(f"Best CV score: {-opt.best_score_:.2f}")

        # Evaluate on test set
        if target == 'total':
            y_test = self.data['y_total_test']
        else:
            y_test = self.data['y_margin_test']

        y_pred = opt.best_estimator_.predict(self.data['X_test'])
        test_mae = mean_absolute_error(y_test, y_pred)
        logger.info(f"Test MAE: {test_mae:.2f}")

        return opt.best_estimator_, test_mae

    def tune_all_models(self):
        """Tune all models."""
        logger.info("=" * 70)
        logger.info("PHASE 13: Hyperparameter Tuning")
        logger.info("=" * 70)

        results = {}

        # ===== TOTAL MODELS =====
        logger.info("\n" + "=" * 70)
        logger.info("TOTAL MODELS")
        logger.info("=" * 70)

        # Ridge
        ridge_total, ridge_total_mae = self.tune_ridge('total')
        if ridge_total is not None:
            results['ridge_total_tuned'] = {
                'model': ridge_total,
                'test_mae': ridge_total_mae
            }
            # Save
            path = self.models_dir / 'ridge_total_tuned.pkl'
            joblib.dump(ridge_total, path)
            logger.info(f"✓ Saved to {path}")

        # Random Forest
        rf_total, rf_total_mae = self.tune_random_forest('total')
        if rf_total is not None:
            results['rf_total_tuned'] = {
                'model': rf_total,
                'test_mae': rf_total_mae
            }
            path = self.models_dir / 'randomforest_total_tuned.pkl'
            joblib.dump(rf_total, path)
            logger.info(f"✓ Saved to {path}")

        # XGBoost
        xgb_total, xgb_total_mae = self.tune_xgboost('total')
        if xgb_total is not None:
            results['xgb_total_tuned'] = {
                'model': xgb_total,
                'test_mae': xgb_total_mae
            }
            path = self.models_dir / 'xgboost_total_tuned.pkl'
            joblib.dump(xgb_total, path)
            logger.info(f"✓ Saved to {path}")

        # ===== MARGIN MODELS =====
        logger.info("\n" + "=" * 70)
        logger.info("MARGIN MODELS")
        logger.info("=" * 70)

        # Ridge
        ridge_margin, ridge_margin_mae = self.tune_ridge('margin')
        if ridge_margin is not None:
            results['ridge_margin_tuned'] = {
                'model': ridge_margin,
                'test_mae': ridge_margin_mae
            }
            path = self.models_dir / 'ridge_margin_tuned.pkl'
            joblib.dump(ridge_margin, path)
            logger.info(f"✓ Saved to {path}")

        # Random Forest
        rf_margin, rf_margin_mae = self.tune_random_forest('margin')
        if rf_margin is not None:
            results['rf_margin_tuned'] = {
                'model': rf_margin,
                'test_mae': rf_margin_mae
            }
            path = self.models_dir / 'randomforest_margin_tuned.pkl'
            joblib.dump(rf_margin, path)
            logger.info(f"✓ Saved to {path}")

        # XGBoost
        xgb_margin, xgb_margin_mae = self.tune_xgboost('margin')
        if xgb_margin is not None:
            results['xgb_margin_tuned'] = {
                'model': xgb_margin,
                'test_mae': xgb_margin_mae
            }
            path = self.models_dir / 'xgboost_margin_tuned.pkl'
            joblib.dump(xgb_margin, path)
            logger.info(f"✓ Saved to {path}")

        self.results = results
        return self

    def print_results(self):
        """Print tuning results."""
        logger.info("\n" + "=" * 70)
        logger.info("TUNING RESULTS")
        logger.info("=" * 70)

        logger.info("\n--- TOTAL MODELS ---")
        logger.info(f"{'Model':<25} {'Test MAE':<12}")
        logger.info("-" * 40)
        for name, data in self.results.items():
            if 'total' in name:
                logger.info(f"{name:<25} {data['test_mae']:<12.2f}")

        logger.info("\n--- MARGIN MODELS ---")
        logger.info(f"{'Model':<25} {'Test MAE':<12}")
        logger.info("-" * 40)
        for name, data in self.results.items():
            if 'margin' in name:
                logger.info(f"{name:<25} {data['test_mae']:<12.2f}")

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
        logger.info("BEST TUNED MODELS:")
        logger.info(f"  Total: {best_total[0]} (MAE: {best_total[1]['test_mae']:.2f})")
        logger.info(f"  Margin: {best_margin[0]} (MAE: {best_margin[1]['test_mae']:.2f})")
        logger.info("=" * 70)

        return self


def main():
    """Run Phase 13."""
    features_path = 'data/processed/enhanced_features.parquet'
    models_dir = 'data/models'

    tuner = HyperparameterTuner(features_path, models_dir)
    tuner.load_data()
    tuner.prepare_data()
    tuner.tune_all_models()
    tuner.print_results()

    logger.info("\n" + "=" * 70)
    logger.info("PHASE 13: COMPLETE")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
