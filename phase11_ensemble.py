"""
Phase 11: Build Ensemble Models
Combine multiple models for improved predictions.
"""

import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class EnsembleBuilder:
    """
    Build ensemble models from base models.
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

        self.data = {
            'X_train': X_train, 'X_val': X_val, 'X_test': X_test,
            'y_total_train': y_total_train, 'y_total_val': y_total_val, 'y_total_test': y_total_test,
            'y_margin_train': y_margin_train, 'y_margin_val': y_margin_val, 'y_margin_test': y_margin_test,
        }

        logger.info(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
        return self

    def load_models(self):
        """Load base models."""
        logger.info("Loading base models...")

        self.models = {}

        # Total models
        total_models = ['linear_total_enhanced', 'ridge_total_enhanced', 'randomforest_total_enhanced', 'xgboost_total_enhanced', 'lightgbm_total_enhanced']
        for name in total_models:
            path = self.models_dir / f"{name}.pkl"
            if path.exists():
                self.models[name] = joblib.load(path)
                logger.info(f"  ✓ {name}")
            else:
                logger.warning(f"  ✗ {name} not found")

        # Margin models
        margin_models = ['linear_margin_enhanced', 'ridge_margin_enhanced', 'randomforest_margin_enhanced', 'xgboost_margin_enhanced', 'lightgbm_margin_enhanced']
        for name in margin_models:
            path = self.models_dir / f"{name}.pkl"
            if path.exists():
                self.models[name] = joblib.load(path)
                logger.info(f"  ✓ {name}")
            else:
                logger.warning(f"  ✗ {name} not found")

        return self

    def predict_ensemble_simple_avg(self, model_names, X):
        """Simple average ensemble."""
        predictions = []
        for name in model_names:
            if name in self.models:
                pred = self.models[name].predict(X)
                predictions.append(pred)
        return np.mean(predictions, axis=0) if predictions else None

    def predict_ensemble_weighted(self, model_names, X, weights):
        """Weighted average ensemble."""
        predictions = []
        for name, weight in zip(model_names, weights):
            if name in self.models:
                pred = self.models[name].predict(X)
                predictions.append(pred * weight)
        return np.sum(predictions, axis=0) if predictions else None

    def calculate_validation_weights(self, model_names, target='total'):
        """Calculate weights based on validation MAE (inverse weighting)."""
        X_val = self.data['X_val']
        if target == 'total':
            y_val = self.data['y_total_val']
        else:
            y_val = self.data['y_margin_val']

        maes = []
        for name in model_names:
            if name in self.models:
                pred = self.models[name].predict(X_val)
                mae = mean_absolute_error(y_val, pred)
                maes.append(mae)
            else:
                maes.append(float('inf'))

        # Inverse weights (lower MAE = higher weight)
        weights = [1/mae if mae > 0 else 1 for mae in maes]
        total_weight = sum(weights)
        weights = [w/total_weight for w in weights]

        return weights

    def build_ensembles(self):
        """Build ensemble models."""
        logger.info("=" * 70)
        logger.info("Building Ensemble Models")
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

        results = {}

        # ===== TOTAL ENSEMBLES =====
        logger.info("\n--- TOTAL ENSEMBLES ---")

        total_model_names = ['linear_total_enhanced', 'ridge_total_enhanced', 'randomforest_total_enhanced', 'xgboost_total_enhanced', 'lightgbm_total_enhanced']

        # Simple Average
        logger.info("\nTotal: Simple Average (Linear + Ridge + RF)")
        pred_train = self.predict_ensemble_simple_avg(total_model_names, X_train)
        pred_val = self.predict_ensemble_simple_avg(total_model_names, X_val)
        pred_test = self.predict_ensemble_simple_avg(total_model_names, X_test)

        if pred_train is not None:
            train_mae = mean_absolute_error(y_total_train, pred_train)
            val_mae = mean_absolute_error(y_total_val, pred_val)
            test_mae = mean_absolute_error(y_total_test, pred_test)

            logger.info(f"  Train MAE: {train_mae:.2f}")
            logger.info(f"  Val MAE: {val_mae:.2f}")
            logger.info(f"  Test MAE: {test_mae:.2f}")

            results['total_simple_avg'] = {
                'train_mae': train_mae,
                'val_mae': val_mae,
                'test_mae': test_mae
            }

        # Weighted Average
        logger.info("\nTotal: Weighted Average (by validation MAE)")
        weights = self.calculate_validation_weights(total_model_names, 'total')
        logger.info(f"  Weights: {weights}")
        pred_train = self.predict_ensemble_weighted(total_model_names, X_train, weights)
        pred_val = self.predict_ensemble_weighted(total_model_names, X_val, weights)
        pred_test = self.predict_ensemble_weighted(total_model_names, X_test, weights)

        if pred_train is not None:
            train_mae = mean_absolute_error(y_total_train, pred_train)
            val_mae = mean_absolute_error(y_total_val, pred_val)
            test_mae = mean_absolute_error(y_total_test, pred_test)

            logger.info(f"  Train MAE: {train_mae:.2f}")
            logger.info(f"  Val MAE: {val_mae:.2f}")
            logger.info(f"  Test MAE: {test_mae:.2f}")

            results['total_weighted'] = {
                'train_mae': train_mae,
                'val_mae': val_mae,
                'test_mae': test_mae
            }

        # Best 2 Models
        logger.info("\nTotal: Best 2 Models (Linear + Ridge)")
        best2_total = ['linear_total_enhanced', 'ridge_total_enhanced']
        pred_train = self.predict_ensemble_simple_avg(best2_total, X_train)
        pred_val = self.predict_ensemble_simple_avg(best2_total, X_val)
        pred_test = self.predict_ensemble_simple_avg(best2_total, X_test)

        if pred_train is not None:
            train_mae = mean_absolute_error(y_total_train, pred_train)
            val_mae = mean_absolute_error(y_total_val, pred_val)
            test_mae = mean_absolute_error(y_total_test, pred_test)

            logger.info(f"  Train MAE: {train_mae:.2f}")
            logger.info(f"  Val MAE: {val_mae:.2f}")
            logger.info(f"  Test MAE: {test_mae:.2f}")

            results['total_best2'] = {
                'train_mae': train_mae,
                'val_mae': val_mae,
                'test_mae': test_mae
            }

        # ===== MARGIN ENSEMBLES =====
        logger.info("\n--- MARGIN ENSEMBLES ---")

        margin_model_names = ['linear_margin_enhanced', 'ridge_margin_enhanced', 'randomforest_margin_enhanced', 'xgboost_margin_enhanced', 'lightgbm_margin_enhanced']

        # Simple Average
        logger.info("\nMargin: Simple Average (Linear + Ridge + RF)")
        pred_train = self.predict_ensemble_simple_avg(margin_model_names, X_train)
        pred_val = self.predict_ensemble_simple_avg(margin_model_names, X_val)
        pred_test = self.predict_ensemble_simple_avg(margin_model_names, X_test)

        if pred_train is not None:
            train_mae = mean_absolute_error(y_margin_train, pred_train)
            val_mae = mean_absolute_error(y_margin_val, pred_val)
            test_mae = mean_absolute_error(y_margin_test, pred_test)

            logger.info(f"  Train MAE: {train_mae:.2f}")
            logger.info(f"  Val MAE: {val_mae:.2f}")
            logger.info(f"  Test MAE: {test_mae:.2f}")

            results['margin_simple_avg'] = {
                'train_mae': train_mae,
                'val_mae': val_mae,
                'test_mae': test_mae
            }

        # Weighted Average
        logger.info("\nMargin: Weighted Average (by validation MAE)")
        weights = self.calculate_validation_weights(margin_model_names, 'margin')
        logger.info(f"  Weights: {weights}")
        pred_train = self.predict_ensemble_weighted(margin_model_names, X_train, weights)
        pred_val = self.predict_ensemble_weighted(margin_model_names, X_val, weights)
        pred_test = self.predict_ensemble_weighted(margin_model_names, X_test, weights)

        if pred_train is not None:
            train_mae = mean_absolute_error(y_margin_train, pred_train)
            val_mae = mean_absolute_error(y_margin_val, pred_val)
            test_mae = mean_absolute_error(y_margin_test, pred_test)

            logger.info(f"  Train MAE: {train_mae:.2f}")
            logger.info(f"  Val MAE: {val_mae:.2f}")
            logger.info(f"  Test MAE: {test_mae:.2f}")

            results['margin_weighted'] = {
                'train_mae': train_mae,
                'val_mae': val_mae,
                'test_mae': test_mae
            }

        # Best 2 Models
        logger.info("\nMargin: Best 2 Models (Linear + Ridge)")
        best2_margin = ['linear_margin_enhanced', 'ridge_margin_enhanced']
        pred_train = self.predict_ensemble_simple_avg(best2_margin, X_train)
        pred_val = self.predict_ensemble_simple_avg(best2_margin, X_val)
        pred_test = self.predict_ensemble_simple_avg(best2_margin, X_test)

        if pred_train is not None:
            train_mae = mean_absolute_error(y_margin_train, pred_train)
            val_mae = mean_absolute_error(y_margin_val, pred_val)
            test_mae = mean_absolute_error(y_margin_test, pred_test)

            logger.info(f"  Train MAE: {train_mae:.2f}")
            logger.info(f"  Val MAE: {val_mae:.2f}")
            logger.info(f"  Test MAE: {test_mae:.2f}")

            results['margin_best2'] = {
                'train_mae': train_mae,
                'val_mae': val_mae,
                'test_mae': test_mae
            }

        self.results = results
        return self

    def print_results(self):
        """Print ensemble results."""
        logger.info("\n" + "=" * 70)
        logger.info("ENSEMBLE RESULTS")
        logger.info("=" * 70)

        logger.info("\n--- TOTAL ENSEMBLES ---")
        logger.info(f"{'Ensemble':<20} {'Train MAE':<12} {'Val MAE':<12} {'Test MAE':<12}")
        logger.info("-" * 60)
        for name, results in sorted(self.results.items()):
            if 'total' in name:
                logger.info(f"{name:<20} {results['train_mae']:<12.2f} {results['val_mae']:<12.2f} {results['test_mae']:<12.2f}")

        logger.info("\n--- MARGIN ENSEMBLES ---")
        logger.info(f"{'Ensemble':<20} {'Train MAE':<12} {'Val MAE':<12} {'Test MAE':<12}")
        logger.info("-" * 60)
        for name, results in sorted(self.results.items()):
            if 'margin' in name:
                logger.info(f"{name:<20} {results['train_mae']:<12.2f} {results['val_mae']:<12.2f} {results['test_mae']:<12.2f}")

        # Find best
        best_total = min(
            [(name, results) for name, results in self.results.items() if 'total' in name],
            key=lambda x: x[1]['val_mae']
        )
        best_margin = min(
            [(name, results) for name, results in self.results.items() if 'margin' in name],
            key=lambda x: x[1]['val_mae']
        )

        logger.info("\n" + "=" * 70)
        logger.info("BEST ENSEMBLES:")
        logger.info(f"  Total: {best_total[0]} (Val: {best_total[1]['val_mae']:.2f}, Test: {best_total[1]['test_mae']:.2f})")
        logger.info(f"  Margin: {best_margin[0]} (Val: {best_margin[1]['val_mae']:.2f}, Test: {best_margin[1]['test_mae']:.2f})")
        logger.info("=" * 70)

        return self


def main():
    """Run Phase 11."""
    features_path = 'data/processed/enhanced_features.parquet'
    models_dir = 'data/models'

    builder = EnsembleBuilder(features_path, models_dir)
    builder.load_data()
    builder.prepare_data()
    builder.load_models()
    builder.build_ensembles()
    builder.print_results()

    logger.info("\n" + "=" * 70)
    logger.info("PHASE 11: COMPLETE")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
