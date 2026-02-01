import joblib
import pandas as pd
import numpy as np
from datetime import datetime

print('BACKTESTING PREGAME MARGIN MODEL')
print()

# Load dataset
df = pd.read_parquet('data/processed/pregame_team_v2.parquet')
print(f'Loaded {len(df)} games')

# Load model
model_obj = joblib.load('models_v3/pregame/ridge_margin.joblib')
model = model_obj['model']
feature_cols = model_obj['features']

print(f'Model: {model_obj["model_name"]}')
print(f'Features: {len(feature_cols)}')
print()

# Walk-forward backtesting parameters
test_size = 200
min_train_size = 500

results = []

# Walk-forward backtesting
num_folds = (len(df) - min_train_size) // test_size
print(f'Running {num_folds} folds of backtesting...')
print()

for i in range(num_folds):
    train_end = min_train_size + i * test_size
    test_start = train_end
    test_end = test_start + test_size
    
    if test_end > len(df):
        break
    
    train_df = df.iloc[:train_end].copy()
    test_df = df.iloc[test_start:test_end].copy()
    
    # Train on train data
    X_train = train_df[feature_cols].values
    y_train = train_df['margin'].values
    
    model.fit(X_train, y_train)
    
    # Predict on test data
    X_test = test_df[feature_cols].values
    y_test = test_df['margin'].values
    pred = model.predict(X_test)
    
    # Calculate metrics
    mae = np.mean(np.abs(y_test - pred))
    rmse = np.sqrt(np.mean((y_test - pred) ** 2))
    
    # Simple ROI calculation (predicting winner)
    correct = np.sum(np.sign(pred) == np.sign(y_test))
    roi = (correct - len(y_test) / 2) / (len(y_test) / 2) * 100
    
    results.append({
        'train_end': train_end,
        'test_start': test_start,
        'test_end': test_end,
        'test_size': len(y_test),
        'total_mae': np.nan,  # Not applicable for margin-only backtest
        'total_rmse': np.nan,
        'margin_mae': mae,
        'margin_rmse': rmse,
        'roi': roi
    })
    
    print(f'Fold {i+1}/{num_folds}: MAE={mae:.3f}, RMSE={rmse:.3f}, ROI={roi:.1f}%')

print()
print('Saving results...')

# Save results
results_df = pd.DataFrame(results)
results_df.to_parquet('data/processed/pregame_backtest_results.parquet', index=False)

print(f'Saved {len(results_df)} folds')
print()
print('AVERAGE RESULTS:')
print(f'  Margin MAE: {results_df["margin_mae"].mean():.3f}')
print(f'  Margin RMSE: {results_df["margin_rmse"].mean():.3f}')
print(f'  ROI: {results_df["roi"].mean():.1f}%')
print()
print('PREGAME MARGIN BACKTEST COMPLETE!')
