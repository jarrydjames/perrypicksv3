"""
Phase 3: Train Improved Models
- Calibrate intercept (-15.4 points)
- Smaller regularization
- Add interaction features
- Train XGBoost/LightGBM
"""

import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, TimeSeriesSplit

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    logging.warning("XGBoost not available - will skip XGBoost models")

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False
    logging.warning("LightGBM not available - will skip LightGBM models")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ImprovedModelTrainer:
    def __init__(self):
        self.processed_dir = Path("data/processed")
        self.models_dir = Path("data/models")
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
    def load_data(self):
        """Load enhanced features dataset."""
        logger.info("Loading enhanced features dataset...")
        df = pd.read_parquet(self.processed_dir / "enhanced_features.parquet")
        logger.info(f"  Loaded {len(df)} games")
        return df
    
    def prepare_features(self, df):
        """Prepare feature matrices for training."""
        # Basic features
        base_features = [
            'home_pts', 'away_pts',
            'home_efg', 'away_efg',
            'home_ftr', 'away_ftr',
            'home_tpar', 'away_tpar',
            'home_tor', 'away_tor',
            'home_orbp', 'away_orbp',
        ]
        
        # NEW features
        enhanced_features = base_features + [
            'home_pace', 'away_pace', 'avg_pace', 'pace_diff',
            'home_off_rating', 'away_off_rating', 'off_rating_diff',
            'home_def_rating', 'away_def_rating', 'def_rating_diff',
            'home_recent_pts', 'away_recent_pts',
            'home_recent_total', 'away_recent_total',
            'home_recent_win_pct', 'away_recent_win_pct',
        ]
        
        # Interaction features (product of key pairs)
        df['home_pts_x_efg'] = df['home_pts'] * df['home_efg']
        df['away_pts_x_efg'] = df['away_pts'] * df['away_efg']
        df['home_pace_x_off_rating'] = df['home_pace'] * df['home_off_rating']
        df['away_pace_x_off_rating'] = df['away_pace'] * df['away_off_rating']
        
        interaction_features = [
            'home_pts_x_efg', 'away_pts_x_efg',
            'home_pace_x_off_rating', 'away_pace_x_off_rating',
        ]
        
        all_features = enhanced_features + interaction_features
        
        # Remove rows with NaN values in critical features
        df_clean = df.dropna(subset=all_features + ['total', 'margin'])
        
        logger.info(f"  Features: {len(all_features)}")
        logger.info(f"  Clean data: {len(df_clean)} games (removed {len(df) - len(df_clean)})")
        
        return df_clean, all_features
    
    def split_data(self, df):
        """Split data into train/val/test with time awareness."""
        df = df.sort_values('game_date_dt')
        
        # Time-based split: 70% train, 15% val, 15% test
        n = len(df)
        train_end = int(n * 0.70)
        val_end = int(n * 0.85)
        
        train_df = df.iloc[:train_end]
        val_df = df.iloc[train_end:val_end]
        test_df = df.iloc[val_end:]
        
        logger.info(f"  Train: {len(train_df)} games")
        logger.info(f"  Val: {len(val_df)} games")
        logger.info(f"  Test: {len(test_df)} games")
        
        return train_df, val_df, test_df
    
    def train_models(self, X_train, y_train_total, y_train_margin, X_val, y_val_total, y_val_margin):
        """Train multiple model types and compare."""
        results = {}
        
        # Model 1: Linear Regression (unregularized baseline)
        logger.info("Training Linear Regression (baseline)...")
        lr_total = LinearRegression()
        lr_margin = LinearRegression()
        lr_total.fit(X_train, y_train_total)
        lr_margin.fit(X_train, y_train_margin)
        
        # Model 2: Ridge with SMALLER alpha (reduce regularization)
        logger.info("Training Ridge (alpha=0.1, smaller regularization)...")
        ridge_total = Ridge(alpha=0.1, random_state=42)
        ridge_margin = Ridge(alpha=0.1, random_state=42)
        ridge_total.fit(X_train, y_train_total)
        ridge_margin.fit(X_train, y_train_margin)
        
        # Model 3: Gradient Boosting (nonlinear)
        logger.info("Training Gradient Boosting...")
        gb_total = GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=42)
        gb_margin = GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=42)
        gb_total.fit(X_train, y_train_total)
        gb_margin.fit(X_train, y_train_margin)
        
        # Model 4: Random Forest (ensemble)
        logger.info("Training Random Forest...")
        rf_total = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
        rf_margin = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
        rf_total.fit(X_train, y_train_total)
        rf_margin.fit(X_train, y_train_margin)
        
        # Model 5: XGBoost (if available)
        if HAS_XGBOOST:
            logger.info("Training XGBoost...")
            xgb_total = xgb.XGBRegressor(n_estimators=100, max_depth=5, random_state=42)
            xgb_margin = xgb.XGBRegressor(n_estimators=100, max_depth=5, random_state=42)
            xgb_total.fit(X_train, y_train_total)
            xgb_margin.fit(X_train, y_train_margin)
            results['xgboost'] = {
                'total_model': xgb_total,
                'margin_model': xgb_margin,
                'name': 'XGBoost'
            }
        
        # Model 6: LightGBM (if available)
        if HAS_LIGHTGBM:
            logger.info("Training LightGBM...")
            lgb_total = lgb.LGBMRegressor(n_estimators=100, max_depth=5, random_state=42, verbose=-1)
            lgb_margin = lgb.LGBMRegressor(n_estimators=100, max_depth=5, random_state=42, verbose=-1)
            lgb_total.fit(X_train, y_train_total)
            lgb_margin.fit(X_train, y_train_margin)
            results['lightgbm'] = {
                'total_model': xgb_total if not HAS_XGBOOST else None,
                'margin_model': lgb_margin,
                'name': 'LightGBM'
            }
        
        results['linear'] = {
            'total_model': lr_total,
            'margin_model': lr_margin,
            'name': 'LinearRegression'
        }
        results['ridge_small'] = {
            'total_model': ridge_total,
            'margin_model': ridge_margin,
            'name': 'Ridge (alpha=0.1)'
        }
        results['gradient_boosting'] = {
            'total_model': gb_total,
            'margin_model': gb_margin,
            'name': 'GradientBoosting'
        }
        results['random_forest'] = {
            'total_model': rf_total,
            'margin_model': rf_margin,
            'name': 'RandomForest'
        }
        
        return results
    
    def evaluate_model(self, model, X, y_total, y_margin):
        """Evaluate model predictions."""
        pred_total = model['total_model'].predict(X)
        pred_margin = model['margin_model'].predict(X)
        
        total_mae = mean_absolute_error(y_total, pred_total)
        margin_mae = mean_absolute_error(y_margin, pred_margin)
        total_r2 = r2_score(y_total, pred_total)
        margin_r2 = r2_score(y_margin, pred_margin)
        
        # Calculate bias
        total_bias = (pred_total - y_total).mean()
        margin_bias = (pred_margin - y_margin).mean()
        
        return {
            'total_mae': total_mae,
            'margin_mae': margin_mae,
            'total_r2': total_r2,
            'margin_r2': margin_r2,
            'total_bias': total_bias,
            'margin_bias': margin_bias,
        }
    
    def calibrate_intercept(self, model, X_val, y_val_total, y_val_margin):
        """
        Calibrate model intercepts to reduce bias.
        Adjust intercept by the bias amount.
        """
        pred_total = model['total_model'].predict(X_val)
        pred_margin = model['margin_model'].predict(X_val)
        
        # Calculate bias
        total_bias = (pred_total - y_val_total).mean()
        margin_bias = (pred_margin - y_val_margin).mean()
        
        # Apply calibration
        model['total_model'].intercept_ -= total_bias
        model['margin_model'].intercept_ -= margin_bias
        
        logger.info(f"  Calibrated total model by {-total_bias:.2f} points")
        logger.info(f"  Calibrated margin model by {-margin_bias:.2f} points")
        
        return model
    
    def run(self):
        """Run complete training pipeline."""
        logger.info("="*70)
        logger.info("PHASE 3: TRAIN IMPROVED MODELS")
        logger.info("="*70)
        
        # Step 1: Load data
        df = self.load_data()
        
        # Step 2: Prepare features
        df_clean, feature_cols = self.prepare_features(df)
        
        # Step 3: Split data
        train_df, val_df, test_df = self.split_data(df_clean)
        
        # Prepare matrices
        X_train = train_df[feature_cols].values
        y_train_total = train_df['total'].values
        y_train_margin = train_df['margin'].values
        
        X_val = val_df[feature_cols].values
        y_val_total = val_df['total'].values
        y_val_margin = val_df['margin'].values
        
        X_test = test_df[feature_cols].values
        y_test_total = test_df['total'].values
        y_test_margin = test_df['margin'].values
        
        # Step 4: Train models
        results = self.train_models(
            X_train, y_train_total, y_train_margin,
            X_val, y_val_total, y_val_margin
        )
        
        # Step 5: Evaluate and calibrate
        logger.info("="*70)
        logger.info("VALIDATION RESULTS (before calibration)")
        logger.info("="*70)
        
        eval_results = []
        for model_key, model in results.items():
            eval_result = self.evaluate_model(
                model, X_val, y_val_total, y_val_margin
            )
            eval_result['name'] = model['name']
            eval_results.append(eval_result)
        
        # Display results
        eval_df = pd.DataFrame(eval_results)
        print("\nValidation Results:")
        print(eval_df.to_string(index=False))
        
        # Select best model by lowest total MAE
        best_total_idx = eval_df['total_mae'].idxmin()
        best_total_model_name = eval_df.loc[best_total_idx, 'name']
        best_total_key = [k for k, v in results.items() if v['name'] == best_total_model_name][0]
        
        logger.info(f"\nBest total model: {best_total_model_name} (MAE: {eval_df.loc[best_total_idx, 'total_mae']:.2f})")
        
        # Calibrate best model
        best_total_model_calibrated = self.calibrate_intercept(
            results[best_total_key], X_val, y_val_total, y_val_margin
        )
        
        # Save best model
        output_path = self.models_dir / "total_model_improved.pkl"
        joblib.dump(best_total_model_calibrated['total_model'], output_path)
        logger.info(f"Saved improved total model to {output_path}")
        
        # For margin, also select best
        best_margin_idx = eval_df['margin_mae'].idxmin()
        best_margin_model_name = eval_df.loc[best_margin_idx, 'name']
        best_margin_key = [k for k, v in results.items() if v['name'] == best_margin_model_name][0]
        
        logger.info(f"Best margin model: {best_margin_model_name} (MAE: {eval_df.loc[best_margin_idx, 'margin_mae']:.2f})")
        
        # Calibrate margin model
        best_margin_model_calibrated = self.calibrate_intercept(
            results[best_margin_key], X_val, y_val_total, y_val_margin
        )
        
        # Save margin model
        output_path = self.models_dir / "margin_model_improved.pkl"
        joblib.dump(best_margin_model_calibrated['margin_model'], output_path)
        logger.info(f"Saved improved margin model to {output_path}")
        
        # Step 6: Final test evaluation
        logger.info("="*70)
        logger.info("TEST RESULTS (after calibration)")
        logger.info("="*70)
        
        best_total_test_result = self.evaluate_model(
            best_total_model_calibrated, X_test, y_test_total, y_test_margin
        )
        best_total_test_result['name'] = best_total_model_name
        
        best_margin_test_result = self.evaluate_model(
            best_margin_model_calibrated, X_test, y_test_total, y_test_margin
        )
        best_margin_test_result['name'] = best_margin_model_name
        
        print("\nTest Results (Calibrated Models):")
        print(f"\nTotal Model ({best_total_model_name}):")
        print(f"  MAE: {best_total_test_result['total_mae']:.2f} points")
        print(f"  R²: {best_total_test_result['total_r2']:.3f}")
        print(f"  Bias: {best_total_test_result['total_bias']:.2f} points")
        
        print(f"\nMargin Model ({best_margin_model_name}):")
        print(f"  MAE: {best_margin_test_result['margin_mae']:.2f} points")
        print(f"  R²: {best_margin_test_result['margin_r2']:.3f}")
        print(f"  Bias: {best_margin_test_result['margin_bias']:.2f} points")
        
        logger.info("="*70)
        logger.info("PHASE 3 COMPLETE")
        logger.info("="*70)


def main():
    trainer = ImprovedModelTrainer()
    trainer.run()
    return 0


if __name__ == '__main__':
    exit(main())
