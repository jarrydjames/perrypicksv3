"""
Export predictions and uncertainty intervals from Phase 4 to CSV.
"""

import sys
sys.path.insert(0, '/Users/jarrydhawley/Desktop/Predictor/PerryPicks v3')

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from src.conformal import run_conformal_uncertainty

# Load dataset
df = pd.read_parquet('data/processed/halftime_with_temporal_features_total.parquet')
print(f"Loaded dataset: {len(df)} rows")

# Features and target
h1_features = [col for col in df.columns if col.startswith('h1_')]
X = df[h1_features]
y = df['h2_total']

# Split data (same as Phase 4)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Run conformal uncertainty
h1_features = [col for col in df.columns if col.startswith('h1_')]
report, results = run_conformal_uncertainty(
    df,
    h1_features,
    'h2_total',
    alpha=0.1,
    random_state=42,
    test_size=0.2,
)

# Extract results
cqr = results['cqr']
coverage = results['coverage']
calibration_error = results['calibration_error']
interval_quality = results['interval_quality']

# Extract models and calibration data
lower_model = cqr['lower_model']
upper_model = cqr['upper_model']
cal_q = cqr['cal_q']
cal_idx = cqr['cal_idx']

cal_lower = cqr['cal_lower']
cal_upper = cqr['cal_upper']

# Train mean model (LinearRegression) for point predictions
mean_model = LinearRegression()
mean_model.fit(X_train, y_train)

# Generate predictions and intervals for calibration set
X_cal = X.iloc[cal_idx]
lower_cal = lower_model.predict(X_cal)
upper_cal = upper_model.predict(X_cal)

# Apply conformal correction
lower_cal_corrected = lower_cal - cal_q
upper_cal_corrected = upper_cal + cal_q

# Get point predictions
y_pred_cal = mean_model.predict(X_cal)

# Get actual values for calibration set
y_true_cal = y.iloc[cal_idx].values

# Create output DataFrame
output_df = pd.DataFrame({
    'season_end_yy': df.loc[cal_idx, 'season_end_yy'].values,
    'game_id': df.loc[cal_idx, 'game_id'].values,
    'h1_home': df.loc[cal_idx, 'h1_home'].values,
    'h1_away': df.loc[cal_idx, 'h1_away'].values,
    'h1_total': df.loc[cal_idx, 'h1_total'].values,
    'h1_margin': df.loc[cal_idx, 'h1_margin'].values,
    'h2_total_true': y_true_cal,
    'h2_total_pred': y_pred_cal,
    'lower_90%_ci': lower_cal_corrected,
    'upper_90%_ci': upper_cal_corrected,
    'interval_width': upper_cal_corrected - lower_cal_corrected,
    'is_in_interval': (y_true_cal >= lower_cal_corrected) & (y_true_cal <= upper_cal_corrected),
})

# Save to CSV
output_path = 'data/processed/h2_total_predictions_with_intervals.csv'
output_df.to_csv(output_path, index=False)
print(f"Exported {len(output_df)} predictions to: {output_path}")

# Print summary
print("\nSummary Statistics:")
print(f"Mean predicted H2 total: {output_df['h2_total_pred'].mean():.2f}")
print(f"Mean actual H2 total: {output_df['h2_total_true'].mean():.2f}")
print(f"Mean interval width: {output_df['interval_width'].mean():.2f}")
print(f"Median interval width: {output_df['interval_width'].median():.2f}")
print(f"Coverage: {(output_df['is_in_interval'].sum() / len(output_df)) * 100:.2f}%")
print(f"Calibration set size: {len(output_df)}")

# Show first 5 rows
print("\nFirst 5 predictions:")
print(output_df.head(5).to_string(index=False))
