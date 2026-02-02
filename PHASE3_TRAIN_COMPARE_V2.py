"""
PHASE 3: Train and Compare V2 Models vs Baseline

This script:
1. Loads V2 enhanced features dataset
2. Trains models with same hyperparameters as baseline
3. Compares performance to baseline metrics

BASELINE (4-day OOS, 31 games):
- Total MAE: 19.06
- Margin MAE: 11.91
- Winner Accuracy: 64.5%

V2 TARGET:
- Total MAE: < 15
- Margin MAE: < 10
- Winner Accuracy: > 70%
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple
import logging
from sklearn.model_selection import cross_val_score, KFold
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ModelEvaluator:
    """Evaluate and compare V2 models against baseline."""
    
    def __init__(self):
        self.baseline_metrics = {
            'total_mae': 19.06,
            'margin_mae': 11.91,
            'total_rmse': 22.36,
            'margin_rmse': 14.41,
            'winner_accuracy': 0.645,
        }
    
    def load_v2_dataset(self, path: str = 'data/processed/pregame_v2_enhanced.parquet') -> pd.DataFrame:
        """Load V2 enhanced dataset."""
        df = pd.read_parquet(path)
        logger.info(f"✅ Loaded V2 dataset: {len(df)} games, {len(df.columns)} features")
        return df
    
    def prepare_features_and_targets(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
        """
        Prepare features and targets for modeling.
        
        Returns:
            X, y_total, y_margin, feature_names
        """
        # Base features (14)
        base_features = [
            'home_efg', 'home_ftr', 'home_tpar', 'home_tor', 'home_orbp',
            'away_efg', 'away_ftr', 'away_tpar', 'away_tor', 'away_orbp',
            'home_fga', 'home_fgm', 'away_fga', 'away_fgm'
        ]
        
        # V2 features (all others except game_id and targets)
        exclude_cols = ['game_id', 'total', 'margin', 'home_tri', 'away_tri']
        v2_features = [c for c in df.columns if c not in exclude_cols]
        
        # Use all V2 features
        feature_names = v2_features
        X = df[feature_names].values
        
        # Targets
        y_total = df['total'].values
        y_margin = df['margin'].values
        
        # Handle missing values
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        logger.info(f"✅ Features: {len(feature_names)}, Samples: {len(X)}")
        
        return X_scaled, y_total, y_margin, feature_names
    
    def evaluate_model(self, X, y, model, model_name: str, n_folds: int = 5) -> Dict:
        """Evaluate model with cross-validation."""
        
        # K-fold CV
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
        
        mae_scores = []
        rmse_scores = []
        r2_scores = []
        
        for train_idx, test_idx in kf.split(X):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            mae_scores.append(mean_absolute_error(y_test, y_pred))
            rmse_scores.append(np.sqrt(mean_squared_error(y_test, y_pred)))
            r2_scores.append(r2_score(y_test, y_pred))
        
        return {
            'model': model_name,
            'mae_mean': np.mean(mae_scores),
            'mae_std': np.std(mae_scores),
            'rmse_mean': np.mean(rmse_scores),
            'r2_mean': np.mean(r2_scores),
        }
    
    def train_and_evaluate_v2(self, df: pd.DataFrame):
        """Train and evaluate V2 models."""
        
        logger.info("="*70)
        logger.info("TRAINING AND EVALUATING V2 MODELS")
        logger.info("="*70)
        
        # Prepare data
        X, y_total, y_margin, feature_names = self.prepare_features_and_targets(df)
        
        # Models to test
        models = {
            'Ridge': Ridge(alpha=2.0, random_state=42),
            'RandomForest': RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42),
            'GBT': HistGradientBoostingRegressor(max_iter=100, max_depth=5, learning_rate=0.1, random_state=42),
        }
        
        # Evaluate total prediction
        logger.info("\n📊 TOTAL PREDICTION:")
        total_results = []
        for name, model in models.items():
            result = self.evaluate_model(X, y_total, model, name)
            total_results.append(result)
            logger.info(f"   {result['model']:15s}: MAE={result['mae_mean']:.2f}, RMSE={result['rmse_mean']:.2f}, R²={result['r2_mean']:.4f}")
        
        # Evaluate margin prediction
        logger.info("\n📊 MARGIN PREDICTION:")
        margin_results = []
        for name, model in models.items():
            result = self.evaluate_model(X, y_margin, model, name)
            margin_results.append(result)
            logger.info(f"   {result['model']:15s}: MAE={result['mae_mean']:.2f}, RMSE={result['rmse_mean']:.2f}, R²={result['r2_mean']:.4f}")
        
        # Find best models
        best_total = min(total_results, key=lambda x: x['mae_mean'])
        best_margin = min(margin_results, key=lambda x: x['mae_mean'])
        
        logger.info(f"\n✅ BEST TOTAL MODEL: {best_total['model']} (MAE={best_total['mae_mean']:.2f})")
        logger.info(f"✅ BEST MARGIN MODEL: {best_margin['model']} (MAE={best_margin['mae_mean']:.2f})")
        
        return {
            'total_results': total_results,
            'margin_results': margin_results,
            'best_total': best_total,
            'best_margin': best_margin,
            'feature_names': feature_names,
        }
    
    def compare_to_baseline(self, v2_results: Dict):
        """Compare V2 results to baseline metrics."""
        
        logger.info("="*70)
        logger.info("COMPARING V2 TO BASELINE")
        logger.info("="*70)
        
        baseline_total_mae = self.baseline_metrics['total_mae']
        baseline_margin_mae = self.baseline_metrics['margin_mae']
        
        v2_total_mae = v2_results['best_total']['mae_mean']
        v2_margin_mae = v2_results['best_margin']['mae_mean']
        
        # Calculate improvements
        total_improvement = (baseline_total_mae - v2_total_mae) / baseline_total_mae * 100
        margin_improvement = (baseline_margin_mae - v2_margin_mae) / baseline_margin_mae * 100
        
        logger.info(f"\n📊 TOTAL MAE:")
        logger.info(f"   Baseline: {baseline_total_mae:.2f}")
        logger.info(f"   V2:       {v2_total_mae:.2f}")
        logger.info(f"   Improvement: {total_improvement:+.1f}%")
        
        logger.info(f"\n📊 MARGIN MAE:")
        logger.info(f"   Baseline: {baseline_margin_mae:.2f}")
        logger.info(f"   V2:       {v2_margin_mae:.2f}")
        logger.info(f"   Improvement: {margin_improvement:+.1f}%")
        
        # Check if targets met
        target_total_mae = 15.0
        target_margin_mae = 10.0
        
        total_target_met = v2_total_mae < target_total_mae
        margin_target_met = v2_margin_mae < target_margin_mae
        
        logger.info(f"\n🎯 TARGETS:")
        logger.info(f"   Total MAE < {target_total_mae}:   {'✅ MET' if total_target_met else '❌ NOT MET'} ({v2_total_mae:.2f})")
        logger.info(f"   Margin MAE < {target_margin_mae}: {'✅ MET' if margin_target_met else '❌ NOT MET'} ({v2_margin_mae:.2f})")
        
        return {
            'total_improvement_pct': total_improvement,
            'margin_improvement_pct': margin_improvement,
            'total_target_met': total_target_met,
            'margin_target_met': margin_target_met,
        }
    
    def generate_report(self, v2_results: Dict, comparison: Dict) -> str:
        """Generate comprehensive comparison report."""
        
        lines = []
        lines.append("="*70)
        lines.append("PHASE 3: V2 MODEL EVALUATION REPORT")
        lines.append("="*70)
        lines.append("")
        
        lines.append("BASELINE METRICS (4-day OOS, 31 games):")
        lines.append("-"*70)
        for key, value in self.baseline_metrics.items():
            lines.append(f"   {key}: {value:.2f}")
        lines.append("")
        
        lines.append("V2 RESULTS (Enhanced Features):")
        lines.append("-"*70)
        lines.append(f"   Total MAE: {v2_results['best_total']['mae_mean']:.2f}")
        lines.append(f"   Margin MAE: {v2_results['best_margin']['mae_mean']:.2f}")
        lines.append("")
        
        lines.append("IMPROVEMENTS:")
        lines.append("-"*70)
        lines.append(f"   Total MAE: {comparison['total_improvement_pct']:+.1f}%")
        lines.append(f"   Margin MAE: {comparison['margin_improvement_pct']:+.1f}%")
        lines.append("")
        
        lines.append("TARGETS:")
        lines.append("-"*70)
        lines.append(f"   Total MAE < 15:   {'✅ MET' if comparison['total_target_met'] else '❌ NOT MET'}")
        lines.append(f"   Margin MAE < 10: {'✅ MET' if comparison['margin_target_met'] else '❌ NOT MET'}")
        lines.append("")
        
        lines.append("FEATURE COUNTS:")
        lines.append("-"*70)
        lines.append(f"   Base features: 14")
        lines.append(f"   V2 new features: {len(v2_results['feature_names']) - 14}")
        lines.append(f"   Total features: {len(v2_results['feature_names'])}")
        lines.append("")
        
        lines.append("="*70)
        
        return '\n'.join(lines)
    
    def run_evaluation(self) -> Tuple[Dict, str]:
        """Run complete V2 evaluation pipeline."""
        
        try:
            # Load V2 dataset
            df = self.load_v2_dataset()
            
            # Train and evaluate
            v2_results = self.train_and_evaluate_v2(df)
            
            # Compare to baseline
            comparison = self.compare_to_baseline(v2_results)
            
            # Generate report
            report = self.generate_report(v2_results, comparison)
            print("\n" + report)
            
            # Save report
            report_path = Path('data/processed/phase3_v2_evaluation_report.txt')
            report_path.parent.mkdir(exist_ok=True)
            with open(report_path, 'w') as f:
                f.write(report)
            logger.info(f"✅ Saved report to {report_path}")
            
            return v2_results, report
        
        except Exception as e:
            logger.error(f"❌ Evaluation failed: {e}")
            import traceback
            traceback.print_exc()
            raise


def main():
    """Main entry point."""
    try:
        evaluator = ModelEvaluator()
        v2_results, report = evaluator.run_evaluation()
        
        logger.info("="*70)
        logger.info("✅ PHASE 3 COMPLETE - V2 EVALUATION FINISHED")
        logger.info("="*70)
        
        return 0
        
    except Exception as e:
        logger.error(f"❌ PHASE 3 FAILED: {e}")
        return 1


if __name__ == '__main__':
    exit(main())
