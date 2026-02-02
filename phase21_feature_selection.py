"""
Phase 21: Feature Selection
Identify most important features and remove noise.
"""

import logging
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.feature_selection import mutual_info_regression, RFE
from sklearn.model_selection import cross_val_score

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class FeatureSelector:
    """Select most important features."""

    def __init__(self, features_path: str, output_dir: str = 'data/analysis'):
        self.features_path = features_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
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

        # Combine for feature selection
        X_full = X[:val_end]
        y_full = y_total[:val_end]

        self.data = {
            'X_train': X_train, 'X_val': X_val, 'X_test': X_test,
            'X_full': X_full,
            'y_total_train': y_total_train, 'y_total_val': y_total_val, 'y_total_test': y_total_test,
            'y_total_full': y_full,
            'feature_cols': feature_cols
        }

        logger.info(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
        return self

    def analyze_feature_importance(self):
        """Analyze feature importance using multiple methods."""
        logger.info("\n" + "=" * 70)
        logger.info("FEATURE IMPORTANCE ANALYSIS")
        logger.info("=" * 70)

        X = self.data['X_full']
        y = self.data['y_total_full']
        feature_cols = self.data['feature_cols']

        # Method 1: Random Forest Importance
        logger.info("\n1. Random Forest Feature Importance...")
        rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        rf.fit(X, y)
        rf_importance = pd.DataFrame({
            'feature': feature_cols,
            'rf_importance': rf.feature_importances_
        }).sort_values('rf_importance', ascending=False)

        # Method 2: Mutual Information
        logger.info("2. Mutual Information...")
        mi = mutual_info_regression(X, y, random_state=42)
        mi_importance = pd.DataFrame({
            'feature': feature_cols,
            'mi_importance': mi
        }).sort_values('mi_importance', ascending=False)

        # Method 3: Ridge Coefficients (absolute)
        logger.info("3. Ridge Coefficient Magnitude...")
        ridge = Ridge(alpha=8.15, random_state=42)
        ridge.fit(X, y)
        ridge_importance = pd.DataFrame({
            'feature': feature_cols,
            'ridge_coef': np.abs(ridge.coef_)
        }).sort_values('ridge_coef', ascending=False)

        # Combine rankings
        rf_importance['rf_rank'] = rf_importance['rf_importance'].rank(ascending=False)
        mi_importance['mi_rank'] = mi_importance['mi_importance'].rank(ascending=False)
        ridge_importance['ridge_rank'] = ridge_importance['ridge_coef'].rank(ascending=False)

        # Merge
        combined = rf_importance.merge(mi_importance, on='feature')
        combined = combined.merge(ridge_importance, on='feature')

        # Calculate average rank
        combined['avg_rank'] = combined[['rf_rank', 'mi_rank', 'ridge_rank']].mean(axis=1)
        combined = combined.sort_values('avg_rank')

        logger.info(f"\nTop 20 Features by Average Rank:")
        logger.info(f"{'Rank':<6} {'Feature':<35} {'RF Imp':<10} {'MI':<10} {'Ridge':<10}")
        logger.info("-" * 75)
        for idx, row in combined.head(20).iterrows():
            logger.info(f"{row['avg_rank']:<6.1f} {row['feature']:<35} {row['rf_importance']:<10.4f} {row['mi_importance']:<10.4f} {row['ridge_coef']:<10.4f}")

        # Save importance
        combined.to_csv(self.output_dir / 'feature_importance.csv', index=False)
        logger.info(f"\n✓ Saved feature importance to {self.output_dir / 'feature_importance.csv'}")

        self.feature_importance = combined
        return self

    def select_top_features(self, n_features: int = 50):
        """Select top N features."""
        logger.info("\n" + "=" * 70)
        logger.info(f"SELECTING TOP {n_features} FEATURES")
        logger.info("=" * 70)

        top_features = self.feature_importance.head(n_features)['feature'].tolist()
        logger.info(f"\nSelected {len(top_features)} features")

        # Test performance with reduced feature set
        feature_cols_all = self.data['feature_cols']
        top_indices = [feature_cols_all.index(f) for f in top_features]

        X_train_reduced = self.data['X_train'][:, top_indices]
        X_val_reduced = self.data['X_val'][:, top_indices]
        X_test_reduced = self.data['X_test'][:, top_indices]

        # Train models with reduced features
        ridge = Ridge(alpha=8.15, random_state=42)
        rf = RandomForestRegressor(n_estimators=117, max_depth=9, random_state=42, n_jobs=-1)

        ridge.fit(X_train_reduced, self.data['y_total_train'])
        rf.fit(X_train_reduced, self.data['y_total_train'])

        ridge_pred = ridge.predict(X_val_reduced)
        rf_pred = rf.predict(X_val_reduced)

        ridge_mae = np.mean(np.abs(ridge_pred - self.data['y_total_val']))
        rf_mae = np.mean(np.abs(rf_pred - self.data['y_total_val']))

        logger.info(f"\nValidation MAE with {n_features} features:")
        logger.info(f"  Ridge: {ridge_mae:.2f}")
        logger.info(f"  Random Forest: {rf_mae:.2f}")

        self.selected_features = top_features
        self.top_indices = top_indices

        return self

    def save_selected_features(self):
        """Save selected feature list."""
        output_path = self.output_dir / 'selected_features.txt'
        with open(output_path, 'w') as f:
            f.write("# Top features selected by importance\n")
            f.write("# Format: feature_name\n\n")
            for i, feat in enumerate(self.selected_features, 1):
                f.write(f"{feat}\n")

        logger.info(f"\n✓ Saved selected features to {output_path}")
        return self

    def print_recommendations(self):
        """Print feature selection recommendations."""
        logger.info("\n" + "=" * 70)
        logger.info("FEATURE SELECTION RECOMMENDATIONS")
        logger.info("=" * 70)

        logger.info("\n1. Top 10 Most Important Features:")
        for idx, row in self.feature_importance.head(10).iterrows():
            logger.info(f"   {row['feature']}")

        logger.info("\n2. Low-Importance Features (consider removing):")
        for idx, row in self.feature_importance.tail(10).iterrows():
            logger.info(f"   {row['feature']}")

        logger.info("\n3. Recommendation:")
        logger.info("   - Use top 50 features for main model")
        logger.info("   - Consider removing bottom 20 features")
        logger.info("   - Re-evaluate feature importance quarterly")

        return self


def main():
    """Run Phase 21."""
    features_path = 'data/processed/final_features.parquet'

    selector = FeatureSelector(features_path)
    selector.load_data()
    selector.prepare_data()
    selector.analyze_feature_importance()
    selector.select_top_features(n_features=50)
    selector.save_selected_features()
    selector.print_recommendations()

    logger.info("\n" + "=" * 70)
    logger.info("PHASE 21: COMPLETE")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
