"""
Phase 7: Train Models on Pre-Game Features
Train and calibrate models using only data available before tipoff.
"""

import joblib
import logging
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class PregameModelTrainer:
    """
    Train models on pre-game features (no data leakage!).
    
    Uses only features available before tipoff to predict:
    - Total points (over/under)
    - Point spread (winner and margin)
    """
    
    def __init__(self):
        self.processed_dir = Path("data/processed")
        self.models_dir = Path("data/models")
        self.models_dir.mkdir(parents=True, exist_ok=True)
    
    def load_data(self) -> tuple:
        """Load pre-game features and feature list."""
        logger.info("Loading pre-game features...")
        df = pd.read_parquet(self.processed_dir / "pregame_features.parquet")
        
        with open(self.processed_dir / "pregame_feature_list.txt") as f:
            feature_cols = [line.strip() for line in f if line.strip()]
        
        logger.info(f"  Loaded {len(df)} games")
        logger.info(f"  Features: {len(feature_cols)}")
        
        return df, feature_cols
    
    def split_data(self, df: pd.DataFrame) -> tuple:
        """Split data into train/val/test with time awareness."""
        df = df.sort_values('game_date')
        
        # Time-based split: 70% train, 15% val, 15% test
        n = len(df)
        train_end = int(n * 0.70)
        val_end = int(n * 0.85)
        
        train_df = df.iloc[:train_end]
        val_df = df.iloc[train_end:val_end]
        test_df = df.iloc[val_end:]
        
        logger.info(f"\nData split (time-aware):")
        logger.info(f"  Train: {len(train_df)} games ({df['game_date'].min()} to {train_df['game_date'].max()})")
        logger.info(f"  Val: {len(val_df)} games ({val_df['game_date'].min()} to {val_df['game_date'].max()})")
        logger.info(f"  Test: {len(test_df)} games ({test_df['game_date'].min()} to {test_df['game_date'].max()})")
        
        return train_df, val_df, test_df
    
    def prepare_matrices(self, train_df, val_df, test_df, feature_cols: List[str]) -> tuple:
        """Prepare feature matrices and target vectors."""
        # Features
        X_train = train_df[feature_cols].values
        X_val = val_df[feature_cols].values
        X_test = test_df[feature_cols].values
        
        # Targets
        y_train_total = train_df['total'].values
        y_train_margin = train_df['margin'].values
        
        y_val_total = val_df['total'].values
        y_val_margin = val_df['margin'].values
        
        y_test_total = test_df['total'].values
        y_test_margin = test_df['margin'].values
        
        logger.info(f"\nTarget statistics:")
        logger.info(f"  Train Total: mean={np.mean(y_train_total):.1f}, std={np.std(y_train_total):.1f}")
        logger.info(f"  Val Total: mean={np.mean(y_val_total):.1f}, std={np.std(y_val_total):.1f}")
        logger.info(f"  Test Total: mean={np.mean(y_test_total):.1f}, std={np.std(y_test_total):.1f}")
        
        return (X_train, y_train_total, y_train_margin,
                X_val, y_val_total, y_val_margin,
                X_test, y_test_total, y_test_margin)
    
    def train_models(self, X_train, y_train_total, y_train_margin) -> dict:
        """Train multiple model types."""
        logger.info("\nTraining models...")
        
        results = {}
        
        # Model 1: Linear Regression (baseline)
        logger.info("  Training LinearRegression...")
        lr_total = LinearRegression()
        lr_margin = LinearRegression()
        lr_total.fit(X_train, y_train_total)
        lr_margin.fit(X_train, y_train_margin)
        results['linear'] = {
            'total_model': lr_total,
            'margin_model': lr_margin,
            'name': 'LinearRegression'
        }
        
        # Model 2: Ridge (regularized)
        logger.info("  Training Ridge (alpha=1.0)...")
        ridge_total = Ridge(alpha=1.0, random_state=42)
        ridge_margin = Ridge(alpha=1.0, random_state=42)
        ridge_total.fit(X_train, y_train_total)
        ridge_margin.fit(X_train, y_train_margin)
        results['ridge'] = {
            'total_model': ridge_total,
            'margin_model': ridge_margin,
            'name': 'Ridge (alpha=1.0)'
        }
        
        # Model 3: Gradient Boosting (nonlinear)
        logger.info("  Training GradientBoosting...")
        gb_total = GradientBoostingRegressor(
            n_estimators=200, max_depth=4, learning_rate=0.1,
            subsample=0.8, random_state=42
        )
        gb_margin = GradientBoostingRegressor(
            n_estimators=200, max_depth=4, learning_rate=0.1,
            subsample=0.8, random_state=42
        )
        gb_total.fit(X_train, y_train_total)
        gb_margin.fit(X_train, y_train_margin)
        results['gradient_boosting'] = {
            'total_model': gb_total,
            'margin_model': gb_margin,
            'name': 'GradientBoosting'
        }
        
        # Model 4: Random Forest (ensemble)
        logger.info("  Training RandomForest...")
        rf_total = RandomForestRegressor(
            n_estimators=200, max_depth=10, min_samples_split=10,
            random_state=42, n_jobs=-1
        )
        rf_margin = RandomForestRegressor(
            n_estimators=200, max_depth=10, min_samples_split=10,
            random_state=42, n_jobs=-1
        )
        rf_total.fit(X_train, y_train_total)
        rf_margin.fit(X_train, y_train_margin)
        results['random_forest'] = {
            'total_model': rf_total,
            'margin_model': rf_margin,
            'name': 'RandomForest'
        }
        
        return results
    
    def evaluate_model(self, model, X, y_total, y_margin) -> dict:
        """Evaluate model predictions."""
        pred_total = model['total_model'].predict(X)
        pred_margin = model['margin_model'].predict(X)
        
        total_mae = mean_absolute_error(y_total, pred_total)
        margin_mae = mean_absolute_error(y_margin, pred_margin)
        total_rmse = np.sqrt(mean_squared_error(y_total, pred_total))
        margin_rmse = np.sqrt(mean_squared_error(y_margin, pred_margin))
        total_r2 = r2_score(y_total, pred_total)
        margin_r2 = r2_score(y_margin, pred_margin)
        
        # Bias
        total_bias = (pred_total - y_total).mean()
        margin_bias = (pred_margin - y_margin).mean()
        
        # Winner accuracy
        pred_winner = np.where(pred_margin > 0, 'home', 'away')
        actual_winner = np.where(y_margin > 0, 'home', 'away')
        winner_accuracy = (pred_winner == actual_winner).mean()
        
        return {
            'total_mae': total_mae,
            'margin_mae': margin_mae,
            'total_rmse': total_rmse,
            'margin_rmse': margin_rmse,
            'total_r2': total_r2,
            'margin_r2': margin_r2,
            'total_bias': total_bias,
            'margin_bias': margin_bias,
            'winner_accuracy': winner_accuracy,
        }
    
    def calibrate_model(self, model, X_val, y_val_total, y_val_margin) -> dict:
        """Calibrate model intercepts based on validation bias."""
        pred_total = model['total_model'].predict(X_val)
        pred_margin = model['margin_model'].predict(X_val)
        
        total_bias = (pred_total - y_val_total).mean()
        margin_bias = (pred_margin - y_val_margin).mean()
        
        # Apply calibration
        if hasattr(model['total_model'], 'intercept_'):
            model['total_model'].intercept_ -= total_bias
            logger.info(f"    Calibrated total intercept by {-total_bias:.2f} points")
        else:
            logger.info(f"    Cannot calibrate total model (no intercept): bias={total_bias:.2f}")
        
        if hasattr(model['margin_model'], 'intercept_'):
            model['margin_model'].intercept_ -= margin_bias
            logger.info(f"    Calibrated margin intercept by {-margin_bias:.2f} points")
        else:
            logger.info(f"    Cannot calibrate margin model (no intercept): bias={margin_bias:.2f}")
        
        return model
    
    def run(self):
        """Run complete training pipeline."""
        logger.info("="*70)
        logger.info("PHASE 7: TRAIN MODELS ON PREGAME FEATURES")
        logger.info("="*70)
        
        # Step 1: Load data
        df, feature_cols = self.load_data()
        
        # Step 2: Split data
        train_df, val_df, test_df = self.split_data(df)
        
        # Step 3: Prepare matrices
        (X_train, y_train_total, y_train_margin,
         X_val, y_val_total, y_val_margin,
         X_test, y_test_total, y_test_margin) = self.prepare_matrices(
            train_df, val_df, test_df, feature_cols
        )
        
        # Step 4: Train models
        results = self.train_models(X_train, y_train_total, y_train_margin)
        
        # Step 5: Evaluate on validation
        logger.info("\n" + "="*70)
        logger.info("VALIDATION RESULTS (before calibration)")
        logger.info("="*70)
        
        eval_results = []
        for model_key, model in results.items():
            eval_result = self.evaluate_model(model, X_val, y_val_total, y_val_margin)
            eval_result['name'] = model['name']
            eval_results.append(eval_result)
        
        eval_df = pd.DataFrame(eval_results)
        print("\nValidation Results:")
        print(eval_df[['name', 'total_mae', 'margin_mae', 'winner_accuracy']].to_string(index=False))
        
        # Step 6: Select best model and calibrate
        logger.info("\n" + "="*70)
        logger.info("SELECTING AND CALIBRATING BEST MODELS")
        logger.info("="*70)
        
        # Select best total model (lowest MAE)
        best_total_idx = eval_df['total_mae'].idxmin()
        best_total_name = eval_df.loc[best_total_idx, 'name']
        best_total_key = [k for k, v in results.items() if v['name'] == best_total_name][0]
        
        logger.info(f"\nBest total model: {best_total_name}")
        logger.info(f"  Total MAE: {eval_df.loc[best_total_idx, 'total_mae']:.2f}")
        
        best_total_model = self.calibrate_model(
            results[best_total_key], X_val, y_val_total, y_val_margin
        )
        
        # Select best margin model
        best_margin_idx = eval_df['margin_mae'].idxmin()
        best_margin_name = eval_df.loc[best_margin_idx, 'name']
        best_margin_key = [k for k, v in results.items() if v['name'] == best_margin_name][0]
        
        logger.info(f"\nBest margin model: {best_margin_name}")
        logger.info(f"  Margin MAE: {eval_df.loc[best_margin_idx, 'margin_mae']:.2f}")
        
        best_margin_model = self.calibrate_model(
            results[best_margin_key], X_val, y_val_total, y_val_margin
        )
        
        # Step 7: Save models
        output_path_total = self.models_dir / "total_model_pregame.pkl"
        joblib.dump(best_total_model['total_model'], output_path_total)
        logger.info(f"\nSaved total model to {output_path_total}")
        
        output_path_margin = self.models_dir / "margin_model_pregame.pkl"
        joblib.dump(best_margin_model['margin_model'], output_path_margin)
        logger.info(f"Saved margin model to {output_path_margin}")
        
        # Step 8: Final test evaluation
        logger.info("\n" + "="*70)
        logger.info("TEST RESULTS (after calibration)")
        logger.info("="*70)
        
        total_test_result = self.evaluate_model(
            best_total_model, X_test, y_test_total, y_test_margin
        )
        total_test_result['name'] = best_total_name
        
        margin_test_result = self.evaluate_model(
            best_margin_model, X_test, y_test_total, y_test_margin
        )
        margin_test_result['name'] = best_margin_name
        
        print(f"\nTotal Model ({best_total_name}):")
        print(f"  MAE: {total_test_result['total_mae']:.2f} points")
        print(f"  RMSE: {total_test_result['total_rmse']:.2f} points")
        print(f"  R²: {total_test_result['total_r2']:.3f}")
        print(f"  Bias: {total_test_result['total_bias']:.2f} points")
        
        print(f"\nMargin Model ({best_margin_name}):")
        print(f"  MAE: {margin_test_result['margin_mae']:.2f} points")
        print(f"  RMSE: {margin_test_result['margin_rmse']:.2f} points")
        print(f"  R²: {margin_test_result['margin_r2']:.3f}")
        print(f"  Bias: {margin_test_result['margin_bias']:.2f} points")
        print(f"  Winner Accuracy: {margin_test_result['winner_accuracy']:.1%}")
        
        logger.info("="*70)
        logger.info("PHASE 7 COMPLETE")
        logger.info("="*70)


def main():
    trainer = PregameModelTrainer()
    trainer.run()
    return 0


if __name__ == '__main__':
    exit(main())
