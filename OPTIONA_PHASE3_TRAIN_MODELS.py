"""
OPTION A PHASE 3: Train Models on Leakage-Free Pregame Dataset
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, train_test_split
import joblib

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ModelTrainer:
    def __init__(self, data_path: str = "data/processed/pregame_leakage_free.parquet"):
        self.data_path = Path(data_path)
        self.df = None
        self.feature_cols = None
        self.target_cols = ['total', 'margin']
        
        # Models to train
        self.models = {
            'Linear Regression': LinearRegression(),
            'Ridge Regression': Ridge(alpha=1.0),
            'Random Forest': RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                random_state=42,
                n_jobs=-1
            ),
            'Gradient Boosting': GradientBoostingRegressor(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=42
            ),
        }
    
    def load_data(self):
        logger.info(f"Loading data from {self.data_path}")
        self.df = pd.read_parquet(self.data_path)
        logger.info(f"Loaded {len(self.df)} games, {len(self.df.columns)} features")
        
        # Define feature columns (exclude ID and targets)
        exclude_cols = ['game_id', 'season', 'game_date', 'total', 'margin']
        self.feature_cols = [col for col in self.df.columns if col not in exclude_cols]
        logger.info(f"Using {len(self.feature_cols)} features: {self.feature_cols}")
    
    def prepare_data(self, test_size: float = 0.2, val_size: float = 0.1):
        """
        Split data into train/val/test sets.
        
        Split by time (game_date) to avoid lookahead bias:
        - Train: oldest games
        - Val: middle games
        - Test: newest games
        """
        logger.info("Preparing train/val/test split by time...")
        
        # Sort by date
        df_sorted = self.df.sort_values('game_date').reset_index(drop=True)
        
        # Calculate split indices
        n = len(df_sorted)
        train_end = int(n * (1 - test_size - val_size))
        val_end = int(n * (1 - test_size))
        
        # Split
        train_df = df_sorted.iloc[:train_end]
        val_df = df_sorted.iloc[train_end:val_end]
        test_df = df_sorted.iloc[val_end:]
        
        logger.info(f"Train: {len(train_df)} games ({train_end/n*100:.1f}%)")
        logger.info(f"Val: {len(val_df)} games ({(val_end-train_end)/n*100:.1f}%)")
        logger.info(f"Test: {len(test_df)} games ({(n-val_end)/n*100:.1f}%)")
        
        # Extract features and targets
        def get_X_y(df):
            X = df[self.feature_cols].values
            y_total = df['total'].values
            y_margin = df['margin'].values
            # Drop rows with missing targets
            mask = ~(np.isnan(y_total) | np.isnan(y_margin))
            X = X[mask]
            y_total = y_total[mask]
            y_margin = y_margin[mask]
            return X, y_total, y_margin
        
        X_train, y_train_total, y_train_margin = get_X_y(train_df)
        X_val, y_val_total, y_val_margin = get_X_y(val_df)
        X_test, y_test_total, y_test_margin = get_X_y(test_df)
        
        logger.info(f"Final train size: {len(X_train)} (after removing missing targets)")
        logger.info(f"Final val size: {len(X_val)}")
        logger.info(f"Final test size: {len(X_test)}")
        
        return {
            'train': (X_train, y_train_total, y_train_margin),
            'val': (X_val, y_val_total, y_val_margin),
            'test': (X_test, y_test_total, y_test_margin),
        }
    
    def train_model(self, model_name: str, model, X_train, y_train, X_val, y_val, 
                    target_name: str) -> Dict:
        """Train a single model and return metrics."""
        logger.info(f"Training {model_name} for {target_name}...")
        
        # Train
        model.fit(X_train, y_train)
        
        # Predictions
        train_pred = model.predict(X_train)
        val_pred = model.predict(X_val)
        
        # Metrics
        train_mae = mean_absolute_error(y_train, train_pred)
        train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
        val_mae = mean_absolute_error(y_val, val_pred)
        val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))
        
        return {
            'model': model,
            'train_mae': train_mae,
            'train_rmse': train_rmse,
            'val_mae': val_mae,
            'val_rmse': val_rmse,
        }
    
    def train_all_models(self, splits):
        """Train all models for both targets."""
        X_train, y_train_total, y_train_margin = splits['train']
        X_val, y_val_total, y_val_margin = splits['val']
        X_test, y_test_total, y_test_margin = splits['test']
        
        results = {}
        
        for target_name, target_col in [('Total', y_train_total), ('Margin', y_train_margin)]:
            logger.info("="*70)
            logger.info(f"TRAINING MODELS FOR TARGET: {target_name}")
            logger.info("="*70)
            
            y_train = target_col
            y_val = y_val_total if target_name == 'Total' else y_val_margin
            
            target_results = {}
            
            for model_name, model in self.models.items():
                result = self.train_model(
                    model_name, model,
                    X_train, y_train,
                    X_val, y_val,
                    target_name
                )
                target_results[model_name] = result
                
                logger.info(f"  {model_name}:")
                logger.info(f"    Train MAE: {result['train_mae']:.2f}, RMSE: {result['train_rmse']:.2f}")
                logger.info(f"    Val MAE: {result['val_mae']:.2f}, RMSE: {result['val_rmse']:.2f}")
            
            results[target_name] = target_results
        
        return results
    
    def evaluate_best_models(self, results, splits):
        """Evaluate best models on test set."""
        X_test, y_test_total, y_test_margin = splits['test']
        
        logger.info("="*70)
        logger.info("BEST MODEL EVALUATION ON TEST SET")
        logger.info("="*70)
        
        for target_name in ['Total', 'Margin']:
            target_results = results[target_name]
            
            # Find best model (lowest val MAE)
            best_model_name = min(target_results.keys(), 
                               key=lambda x: target_results[x]['val_mae'])
            best_result = target_results[best_model_name]
            best_model = best_result['model']
            
            y_test = y_test_total if target_name == 'Total' else y_test_margin
            test_pred = best_model.predict(X_test)
            
            test_mae = mean_absolute_error(y_test, test_pred)
            test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
            test_r2 = r2_score(y_test, test_pred)
            
            logger.info(f"\n{target_name} - Best Model: {best_model_name}")
            logger.info(f"  Test MAE: {test_mae:.2f}")
            logger.info(f"  Test RMSE: {test_rmse:.2f}")
            logger.info(f"  Test R²: {test_r2:.3f}")
            logger.info(f"  Val MAE: {best_result['val_mae']:.2f}")
            
            # Save best model
            model_path = Path(f"data/models/{target_name.lower()}_model.pkl")
            model_path.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(best_model, model_path)
            logger.info(f"  Saved to: {model_path}")
    
    def save_results(self, results, splits):
        """Save detailed results to file."""
        output_path = Path("data/processed/phase3_training_results.txt")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            f.write("="*70 + "\n")
            f.write("PHASE 3 MODEL TRAINING RESULTS\n")
            f.write("="*70 + "\n\n")
            
            for target_name in ['Total', 'Margin']:
                f.write(f"{target_name} Target\n")
                f.write("-"*50 + "\n")
                
                target_results = results[target_name]
                for model_name, result in target_results.items():
                    f.write(f"\n{model_name}:\n")
                    f.write(f"  Train MAE: {result['train_mae']:.3f}\n")
                    f.write(f"  Train RMSE: {result['train_rmse']:.3f}\n")
                    f.write(f"  Val MAE: {result['val_mae']:.3f}\n")
                    f.write(f"  Val RMSE: {result['val_rmse']:.3f}\n")
        
        logger.info(f"\nResults saved to {output_path}")
    
    def run(self):
        """Run the complete training pipeline."""
        logger.info("="*70)
        logger.info("OPTION A PHASE 3: TRAIN MODELS ON LEAKAGE-FREE DATA")
        logger.info("="*70)
        
        # Load data
        self.load_data()
        
        # Prepare splits
        splits = self.prepare_data()
        
        # Train all models
        results = self.train_all_models(splits)
        
        # Evaluate best models
        self.evaluate_best_models(results, splits)
        
        # Save results
        self.save_results(results, splits)
        
        logger.info("\n" + "="*70)
        logger.info("PHASE 3 COMPLETE - MODELS TRAINED AND EVALUATED")
        logger.info("="*70)


def main():
    trainer = ModelTrainer()
    trainer.run()
    return 0


if __name__ == '__main__':
    exit(main())
