"""
Generate conformal uncertainty intervals for XGBoost model.
"""

import sys
sys.path.insert(0, '/Users/jarrydhawley/Desktop/Predictor/PerryPicks v3')

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import QuantileRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from src.registry import ModelRegistryExtended

print("=" * 80)
print("XGBoost Conformal Uncertainty")
print("=" * 80)
print("")

# Load dataset
df = pd.read_parquet('data/processed/halftime_with_temporal_features_total.parquet')
print(f"Loaded dataset: {len(df)} rows")

# Features and target
h1_features = [col for col in df.columns if col.startswith('h1_')]
X = df[h1_features]
y = df['h2_total']

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"Train set: {len(X_train)} samples")
print(f"Test set: {len(X_test)} samples")
print("")

# Load XGBoost model from registry
registry = ModelRegistryExtended(registry_dir="model_registry_comprehensive")
xgb_model, xgb_metadata = registry.get_model("e4cf457130a6f773")  # XGBoost model ID
print(f"Loaded XGBoost model: e4cf4571...")

# Generate predictions
y_pred_train = xgb_model.predict(X_train)
y_pred_test = xgb_model.predict(X_test)

# Calculate residuals
residuals_train = y_train.values - y_pred_train

# Train quantile regressors on residuals
alpha = 0.1  # 90% coverage
lower_model = QuantileRegressor(alpha=alpha/2, quantile=alpha/2, solver='highs-ds')
upper_model = QuantileRegressor(alpha=alpha/2, quantile=1-alpha/2, solver='highs-ds')

lower_model.fit(X_train, residuals_train)
upper_model.fit(X_train, residuals_train)

# Generate uncertainty intervals on test set
residuals_pred_lower = lower_model.predict(X_test)
residuals_pred_upper = upper_model.predict(X_test)

lower_bounds = y_pred_test + residuals_pred_lower
upper_bounds = y_pred_test + residuals_pred_upper

# Calculate coverage
in_interval = (y_test.values >= lower_bounds) & (y_test.values <= upper_bounds)
empirical_coverage = in_interval.mean()

# Calculate metrics
mae = mean_absolute_error(y_test, y_pred_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
r2 = r2_score(y_test, y_pred_test)

print(f"XGBoost Performance (Test Set):")
print(f"  MAE: {mae:.4f}")
print(f"  RMSE: {rmse:.4f}")
print(f"  R²: {r2:.4f}")
print("")

print(f"Conformal Uncertainty Results:")
print(f"  Target coverage: {(1-alpha)*100:.0f}%")
print(f"  Empirical coverage: {empirical_coverage*100:.2f}%")
print(f"  Coverage error: {abs(empirical_coverage - (1-alpha))*100:.2f}%")
print("")

# Create output DataFrame
output_df = pd.DataFrame({
    'season_end_yy': df.loc[X_test.index, 'season_end_yy'].values,
    'game_id': df.loc[X_test.index, 'game_id'].values,
    'h1_home': df.loc[X_test.index, 'h1_home'].values,
    'h1_away': df.loc[X_test.index, 'h1_away'].values,
    'h1_total': df.loc[X_test.index, 'h1_total'].values,
    'h1_margin': df.loc[X_test.index, 'h1_margin'].values,
    'h2_total_true': y_test.values,
    'h2_total_pred': y_pred_test,
    'lower_90%_ci': lower_bounds,
    'upper_90%_ci': upper_bounds,
    'interval_width': upper_bounds - lower_bounds,
    'is_in_interval': in_interval,
})

# Save to CSV
output_path = 'data/processed/xgboost_predictions_with_intervals.csv'
output_df.to_csv(output_path, index=False)
print(f"Saved predictions to: {output_path}")

# Print summary
interval_width_mean = (upper_bounds - lower_bounds).mean()
interval_width_median = np.median(upper_bounds - lower_bounds)
print(f"Interval Statistics:")
print(f"  Mean interval width: {interval_width_mean:.2f}")
print(f"  Median interval width: {interval_width_median:.2f}")
print(f"  Std interval width: {(upper_bounds - lower_bounds).std():.2f}")

print("\n" + "=" * 80)
print("COMPLETE")
print("=" * 80)
print(f"\nTest set size: {len(output_df)}")
print(f"Coverage: {empirical_coverage*100:.2f}%")
print(f"Target: {(1-alpha)*100:.0f}%")
print("")
