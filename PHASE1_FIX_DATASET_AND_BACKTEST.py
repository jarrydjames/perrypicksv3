"""
PHASE 1: Fix Data Leakage and Run Proper Backtest

This script:
1. Fetches schedule data for all seasons
2. Joins pregame dataset with schedule to add dates
3. Runs strict temporal backtest (NO DATA LEAKAGE)
4. Generates accurate baseline metrics

Expected output: Much higher MAE than 3.51, but realistic
"""

import sys
import json
import numpy as np
import pandas as pd
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Tuple, Dict

# Add project root
sys.path.insert(0, '/Users/jarrydhawley/Desktop/Predictor/PerryPicks v3')

from src.data.schedule import fetch_game_ids_for_seasons, save_game_ids, load_game_ids

# Scikit-learn
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('phase1_fix_backtest.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class Phase1Backtest:
    """Complete pipeline to fix data leakage and run proper backtest."""
    
    def __init__(self):
        self.pregame_df = None
        self.schedule_df = None
        self.merged_df = None
        self.results = None
        
    def load_pregame_data(self) -> bool:
        """Load pregame dataset from parquet file."""
        try:
            self.pregame_df = pd.read_parquet('data/processed/pregame_team_v2.parquet')
            logger.info(f"✅ Loaded {len(self.pregame_df)} games from pregame dataset")
            logger.info(f"Columns: {list(self.pregame_df.columns[:10])}...")
            return True
        except Exception as e:
            logger.error(f"❌ Error loading pregame dataset: {e}")
            return False
    
    def fetch_schedule_data(self) -> bool:
        """Fetch schedule data for seasons 23-24, 24-25, 25-26."""
        try:
            logger.info("="*70)
            logger.info("FETCHING SCHEDULE DATA")
            logger.info("="*70)
            
            # Fetch schedule for all relevant seasons
            seasons = [23, 24, 25]  # 2023-24, 2024-25, 2025-26
            games = fetch_game_ids_for_seasons(season_end_yy=seasons)
            
            logger.info(f"✅ Fetched {len(games)} games from schedule API")
            
            # Convert to DataFrame
            self.schedule_df = pd.DataFrame([g.__dict__ for g in games])
            self.schedule_df['gameId'] = self.schedule_df['gameId'].astype(str)
            
            # Save schedule for reference
            Path('data/raw').mkdir(exist_ok=True)
            save_game_ids('data/raw/schedule_all.json', games)
            logger.info("✅ Saved schedule to data/raw/schedule_all.json")
            
            return True
        except Exception as e:
            logger.error(f"❌ Error fetching schedule: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def merge_datasets(self) -> bool:
        """Merge pregame data with schedule to add dates."""
        try:
            logger.info("="*70)
            logger.info("MERGING DATASETS")
            logger.info("="*70)
            
            # Check game_id column names
            if 'game_id' in self.pregame_df.columns:
                pregames_games = self.pregame_df['game_id'].astype(str).tolist()
            elif 'gameId' in self.pregame_df.columns:
                pregames_games = self.pregame_df['gameId'].astype(str).tolist()
            else:
                logger.error("❌ No game_id column found in pregame dataset")
                return False
            
            schedule_games = set(self.schedule_df['gameId'].tolist())
            
            logger.info(f"📊 Pregame games: {len(pregames_games)}")
            logger.info(f"📊 Schedule games: {len(schedule_games)}")
            logger.info(f"📊 Overlap: {len(set(pregames_games) & schedule_games)}")
            
            # Merge on game_id
            self.pregame_df['gameId'] = self.pregame_df['game_id'].astype(str)
            self.merged_df = self.pregame_df.merge(
                self.schedule_df[['gameId', 'gameDate']],
                on='gameId',
                how='left'
            )
            
            # Parse date
            self.merged_df['game_date'] = pd.to_datetime(self.merged_df['gameDate'])
            
            # Check for missing dates
            missing_dates = self.merged_df['game_date'].isna().sum()
            if missing_dates > 0:
                logger.warning(f"⚠️ {missing_dates} games have missing dates")
            else:
                logger.info("✅ All games have dates")
            
            # Sort by date
            self.merged_df = self.merged_df.sort_values('game_date').reset_index(drop=True)
            
            # Show date range
            min_date = self.merged_df['game_date'].min()
            max_date = self.merged_df['game_date'].max()
            logger.info(f"📅 Date range: {min_date} to {max_date}")
            
            # Save merged dataset
            self.merged_df.to_parquet('data/processed/pregame_with_dates.parquet', index=False)
            logger.info("✅ Saved merged dataset to data/processed/pregame_with_dates.parquet")
            
            return True
        except Exception as e:
            logger.error(f"❌ Error merging datasets: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def walk_forward_cv_strict(self, 
                               feature_cols: List[str],
                               target_cols: List[str],
                               min_train_size: int = 500,
                               test_size: int = 200,
                               step_size: int = 200) -> pd.DataFrame:
        """Perform strict temporal walk-forward CV with NO DATA LEAKAGE."""
        
        logger.info("="*70)
        logger.info("WALK-FORWARD TEMPORAL CROSS-VALIDATION (STRICT, LEAKAGE-FREE)")
        logger.info("="*70)
        logger.info(f"Min train size: {min_train_size}")
        logger.info(f"Test size: {test_size}")
        logger.info(f"Step size: {step_size}")
        
        # CRITICAL: Sort by DATE and ensure strict temporal separation
        df_sorted = self.merged_df.sort_values('game_date').reset_index(drop=True)
        logger.info(f"✅ Sorted {len(df_sorted)} games by date")
        
        results = []
        fold_num = 0
        train_end_idx = min_train_size
        
        total_folds = 0
        skipped_folds = 0
        
        while train_end_idx + test_size + step_size <= len(df_sorted):
            test_start_idx = train_end_idx
            test_end_idx = test_start_idx + test_size
            
            # Split by indices
            train_df = df_sorted.iloc[:train_end_idx]
            test_df = df_sorted.iloc[test_start_idx:test_end_idx]
            
            # CRITICAL: Verify no date overlap
            train_dates = train_df['game_date'].unique()
            test_dates = test_df['game_date'].unique()
            
            max_train_date = train_dates.max()
            min_test_date = test_dates.min()
            
            # Skip if date overlap detected
            if max_train_date >= min_test_date:
                logger.warning(f"⚠️ Fold {fold_num}: Date overlap! Train max={max_train_date}, Test min={min_test_date}")
                train_end_idx += step_size
                fold_num += 1
                skipped_folds += 1
                continue
            
            # Prepare features and targets
            X_train = train_df[feature_cols].values
            X_test = test_df[feature_cols].values
            
            results_fold = {
                'fold': fold_num,
                'train_size': len(train_df),
                'test_size': len(test_df),
                'train_date_range': (train_dates.min(), train_dates.max()),
                'test_date_range': (test_dates.min(), test_dates.max()),
            }
            
            # Train and evaluate each target
            for target in target_cols:
                y_train = train_df[target].values
                y_test = test_df[target].values
                
                # Ridge
                ridge = self._train_ridge(X_train, y_train, X_test, y_test)
                results_fold[f'{target}_ridge_mae_test'] = ridge['mae_test']
                results_fold[f'{target}_ridge_rmse_test'] = ridge['rmse_test']
                results_fold[f'{target}_ridge_r2_test'] = ridge['r2_test']
                
                # Random Forest
                rf = self._train_rf(X_train, y_train, X_test, y_test)
                results_fold[f'{target}_rf_mae_test'] = rf['mae_test']
                results_fold[f'{target}_rf_rmse_test'] = rf['rmse_test']
                results_fold[f'{target}_rf_r2_test'] = rf['r2_test']
                
                # GBT
                gbt = self._train_gbt(X_train, y_train, X_test, y_test)
                results_fold[f'{target}_gbt_mae_test'] = gbt['mae_test']
                results_fold[f'{target}_gbt_rmse_test'] = gbt['rmse_test']
                results_fold[f'{target}_gbt_r2_test'] = gbt['r2_test']
            
            results.append(results_fold)
            total_folds += 1
            
            if fold_num % 2 == 0:
                logger.info(f"📊 Fold {fold_num}: Train={len(train_df)}, Test={len(test_df)}")
                logger.info(f"   Train dates: {train_dates.min()} to {train_dates.max()}")
                logger.info(f"   Test dates: {test_dates.min()} to {test_dates.max()}")
                logger.info(f"   Total MAE: Ridge={ridge['mae_test']:.2f}, RF={rf['mae_test']:.2f}, GBT={gbt['mae_test']:.2f}")
            
            train_end_idx += step_size
            fold_num += 1
        
        results_df = pd.DataFrame(results)
        
        logger.info("="*70)
        logger.info(f"✅ Completed {len(results_df)} folds (skipped {skipped_folds} due to overlap)")
        logger.info("="*70)
        
        for target in target_cols:
            logger.info(f"📊 {target.upper()}:")
            logger.info(f"   Ridge MAE: {results_df[f'{target}_ridge_mae_test'].mean():.2f} ± {results_df[f'{target}_ridge_mae_test'].std():.2f}")
            logger.info(f"   RF MAE:    {results_df[f'{target}_rf_mae_test'].mean():.2f} ± {results_df[f'{target}_rf_mae_test'].std():.2f}")
            logger.info(f"   GBT MAE:   {results_df[f'{target}_gbt_mae_test'].mean():.2f} ± {results_df[f'{target}_gbt_mae_test'].std():.2f}")
        
        return results_df
    
    def _train_ridge(self, X_train, y_train, X_test, y_test, alpha: float = 2.0) -> Dict:
        """Train Ridge regression."""
        model = Ridge(alpha=alpha, random_state=42, solver='auto')
        model.fit(X_train, y_train)
        
        pred_test = model.predict(X_test)
        
        return {
            'mae_test': mean_absolute_error(y_test, pred_test),
            'rmse_test': np.sqrt(mean_squared_error(y_test, pred_test)),
            'r2_test': r2_score(y_test, pred_test),
        }
    
    def _train_rf(self, X_train, y_train, X_test, y_test) -> Dict:
        """Train Random Forest."""
        model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1,
        )
        model.fit(X_train, y_train)
        
        pred_test = model.predict(X_test)
        
        return {
            'mae_test': mean_absolute_error(y_test, pred_test),
            'rmse_test': np.sqrt(mean_squared_error(y_test, pred_test)),
            'r2_test': r2_score(y_test, pred_test),
        }
    
    def _train_gbt(self, X_train, y_train, X_test, y_test) -> Dict:
        """Train Gradient Boosting."""
        model = HistGradientBoostingRegressor(
            max_iter=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=42,
        )
        model.fit(X_train, y_train)
        
        pred_test = model.predict(X_test)
        
        return {
            'mae_test': mean_absolute_error(y_test, pred_test),
            'rmse_test': np.sqrt(mean_squared_error(y_test, pred_test)),
            'r2_test': r2_score(y_test, pred_test),
        }
    
    def run_backtest(self) -> Tuple[pd.DataFrame, str]:
        """Run complete backtest pipeline."""
        
        logger.info("="*70)
        logger.info("PHASE 1: FIX DATA LEAKAGE AND RUN PROPER BACKTEST")
        logger.info("="*70)
        
        # Step 1: Load pregame data
        if not self.load_pregame_data():
            raise Exception("Failed to load pregame data")
        
        # Step 2: Fetch schedule data
        if not self.fetch_schedule_data():
            raise Exception("Failed to fetch schedule data")
        
        # Step 3: Merge datasets
        if not self.merge_datasets():
            raise Exception("Failed to merge datasets")
        
        # Step 4: Define features and targets
        feature_cols = [
            'home_efg', 'home_ftr', 'home_tpar', 'home_tor', 'home_orbp',
            'away_efg', 'away_ftr', 'away_tpar', 'away_tor', 'away_orbp',
            'home_fga', 'home_fgm', 'away_fga', 'away_fgm'
        ]
        target_cols = ['total', 'margin']
        
        # Check columns exist
        missing_features = [f for f in feature_cols if f not in self.merged_df.columns]
        missing_targets = [t for t in target_cols if t not in self.merged_df.columns]
        
        if missing_features:
            raise Exception(f"Missing features: {missing_features}")
        if missing_targets:
            raise Exception(f"Missing targets: {missing_targets}")
        
        # Step 5: Run strict backtest
        cv_results = self.walk_forward_cv_strict(
            feature_cols=feature_cols,
            target_cols=target_cols,
            min_train_size=500,
            test_size=200,
            step_size=200
        )
        
        # Step 6: Generate report
        report = self._generate_report(cv_results, feature_cols, target_cols)
        
        # Step 7: Save results
        output_dir = Path('data/processed')
        output_dir.mkdir(exist_ok=True)
        
        cv_results.to_parquet(output_dir / 'phase1_cv_results_leakage_free.parquet', index=False)
        logger.info("✅ Saved CV results to data/processed/phase1_cv_results_leakage_free.parquet")
        
        with open(output_dir / 'phase1_report_leakage_free.txt', 'w') as f:
            f.write(report)
        logger.info("✅ Saved report to data/processed/phase1_report_leakage_free.txt")
        
        self.results = cv_results
        
        return cv_results, report
    
    def _generate_report(self, cv_results: pd.DataFrame, 
                        feature_cols: List[str], 
                        target_cols: List[str]) -> str:
        """Generate comprehensive report."""
        
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        lines = []
        lines.append("="*70)
        lines.append("PHASE 1: LEAKAGE-FREE HISTORICAL BACKTEST")
        lines.append("="*70)
        lines.append(f"Timestamp: {timestamp}")
        lines.append("")
        
        lines.append("DATASET SUMMARY")
        lines.append("-"*70)
        lines.append(f"Total games: {len(self.merged_df)}")
        lines.append(f"Features ({len(feature_cols)}): {', '.join(feature_cols[:5])}...")
        lines.append(f"Targets: {', '.join(target_cols)}")
        lines.append("")
        
        lines.append("COMPARISON WITH LEAKED RESULTS")
        lines.append("-"*70)
        lines.append("OLD (LEAKED) Results:")
        lines.append("  Total MAE: 3.51 points")
        lines.append("  Total RMSE: 4.39 points")
        lines.append("  Total R²: 0.949")
        lines.append("")
        
        lines.append("CROSS-VALIDATION RESULTS (LEAKAGE-FREE)")
        lines.append("-"*70)
        lines.append(f"Folds: {len(cv_results)}")
        lines.append("")
        
        for target in target_cols:
            lines.append(f"{target.upper()} TARGET:")
            lines.append("")
            lines.append("  Model           | MAE (test)    | RMSE (test)   | R² (test)")
            lines.append("  " + "-"*68)
            lines.append(f"  Ridge           | {cv_results[f'{target}_ridge_mae_test'].mean():6.2f}         | {cv_results[f'{target}_ridge_rmse_test'].mean():6.2f}        | {cv_results[f'{target}_ridge_r2_test'].mean():.4f}")
            lines.append(f"  Random Forest   | {cv_results[f'{target}_rf_mae_test'].mean():6.2f}         | {cv_results[f'{target}_rf_rmse_test'].mean():6.2f}        | {cv_results[f'{target}_rf_r2_test'].mean():.4f}")
            lines.append(f"  GBT             | {cv_results[f'{target}_gbt_mae_test'].mean():6.2f}         | {cv_results[f'{target}_gbt_rmse_test'].mean():6.2f}        | {cv_results[f'{target}_gbt_r2_test'].mean():.4f}")
            lines.append("")
        
        # Calculate improvement
        ridge_total_mae = cv_results['total_ridge_mae_test'].mean()
        old_mae = 3.51
        lines.append("LEAKAGE IMPACT")
        lines.append("-"*70)
        lines.append(f"Old Total MAE (LEAKED):  {old_mae:.2f}")
        lines.append(f"New Total MAE (FIXED):   {ridge_total_mae:.2f}")
        lines.append(f"Leakage Error:           {ridge_total_mae - old_mae:.2f} points ({(ridge_total_mae/old_mae - 1)*100:.1f}%)")
        lines.append("")
        
        lines.append("MODEL SELECTION")
        lines.append("-"*70)
        
        avg_mae_ridge = cv_results['total_ridge_mae_test'].mean()
        avg_mae_rf = cv_results['total_rf_mae_test'].mean()
        avg_mae_gbt = cv_results['total_gbt_mae_test'].mean()
        
        champion = 'ridge'
        if avg_mae_rf < avg_mae_ridge:
            champion = 'rf'
        if avg_mae_gbt < min(avg_mae_ridge, avg_mae_rf):
            champion = 'gbt'
        
        lines.append(f"Selected Champion: {champion.upper()}")
        lines.append("")
        
        lines.append("="*70)
        
        return '\n'.join(lines)


def main():
    """Main entry point."""
    try:
        backtest = Phase1Backtest()
        cv_results, report = backtest.run_backtest()
        
        print("\n" + report)
        
        logger.info("="*70)
        logger.info("✅ PHASE 1 COMPLETE - DATA LEAKAGE FIXED")
        logger.info("="*70)
        
        return 0
        
    except Exception as e:
        logger.error(f"❌ PHASE 1 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    exit(main())
