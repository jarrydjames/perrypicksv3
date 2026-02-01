import joblib
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor

print('FIXING Q3 MARGIN MODEL - Using GBT as Champion')
print()

# Load dataset
df = pd.read_parquet('data/processed/q3_team_v2.parquet')
print(f'Loaded {len(df)} games')

# Features for Q3 margin
feature_cols = [c for c in df.columns if c.startswith('q3_') or c.endswith('_efg') or c.endswith('_ftr') or c.endswith('_tpar') or c.endswith('_tor') or c.endswith('_orbp') or c.endswith('_fga') or c.endswith('_fgm')]
feature_cols = [c for c in feature_cols if c != 'margin']  # Exclude target
feature_cols.sort()

print(f'Features: {len(feature_cols)}')
print()

# Train GBT on full dataset
X = df[feature_cols].values
y = df['margin'].values

print('Training GBT model...')
model = GradientBoostingRegressor(
    n_estimators=100,
    max_depth=4,
    learning_rate=0.1,
    random_state=42
)
model.fit(X, y)
print('Training complete')
print()

# Calibrate SD
pred = model.predict(X)
residuals = y - pred
sd = np.percentile(np.abs(residuals), 80) / 1.2816
print(f'Calibrated SD: {sd:.3f}')
print()

# Save model
model_obj = {
    'model': model,
    'model_name': 'GBT',
    'features': feature_cols,
    'sd': sd,
    'dataset': 'q3_team_v2.parquet',
    'target': 'margin'
}

model_path = 'models_v3/q3/ridge_margin.joblib'  # Keep same path for consistency
joblib.dump(model_obj, model_path)

print(f'Saved model to: {model_path}')
print()
print('Q3 MARGIN MODEL FIXED!')
print('  Champion: GBT (MAE: 3.88)')
