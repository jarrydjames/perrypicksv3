import joblib
import pandas as pd
import numpy as np
from datetime import datetime

print('BACKTESTING PREGAME MODEL (TOTAL + MARGIN)')
print()

# Load dataset
df = pd.read_parquet('data/processed/pregame_team_v2.parquet')
print(f'Loaded {len(df)} games')

# Load models
total_model = joblib.load('models_v3/pregame/ridge_total.joblib')
margin_model = joblib.load('models_v3/pregame/ridge_margin.joblib')

print(f'Total Model: {total_model["model_name"]}')
print(f'Margin Model: {margin_model["model_name"]}')
print(f'Features: {len(total_model["features"])}')
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
    X_train = train_df[total_model['features']].values
    y_train_total = train_df['total'].values
    y_train_margin = train_df['margin'].values
    
    total_model['model'].fit(X_train, y_train_total)
    margin_model['model'].fit(X_train, y_train_margin)
    
    # Predict on test data
    X_test = test_df[total_model['features']].values
    y_test_total = test_df['total'].values
    y_test_margin = test_df['margin'].values
    
    pred_total = total_model['model'].predict(X_test)
    pred_margin = margin_model['model'].predict(X_test)
    
    # Calculate metrics
    total_mae = np.mean(np.abs(y_test_total - pred_total))
    total_rmse = np.sqrt(np.mean((y_test_total - pred_total) ** 2))
    margin_mae = np.mean(np.abs(y_test_margin - pred_margin))
    margin_rmse = np.sqrt(np.mean((y_test_margin - pred_margin) ** 2))
    
    # Simple ROI calculation (predicting winner based on margin)
    correct = np.sum(np.sign(pred_margin) == np.sign(y_test_margin))
    roi = (correct - len(y_test_margin) / 2) / (len(y_test_margin) / 2) * 100
    
    results.append({
        'train_end': train_end,
        'test_start': test_start,
        'test_end': test_end,
        'test_size': len(y_test_total),
        'total_mae': total_mae,
        'total_rmse': total_rmse,
        'margin_mae': margin_mae,
        'margin_rmse': margin_rmse,
        'roi': roi
    })
    
    print(f'Fold {i+1}/{num_folds}: Total MAE={total_mae:.3f}, Margin MAE={margin_mae:.3f}, ROI={roi:.1f}%')

print()
print('Saving results...')

# Save results
results_df = pd.DataFrame(results)
results_df.to_parquet('data/processed/pregame_backtest_results_complete.parquet', index=False)

print(f'Saved {len(results_df)} folds')
print()
print('AVERAGE RESULTS:')
print(f'  Total MAE: {results_df["total_mae"].mean():.3f} (±{results_df["total_mae"].std():.3f})')
print(f'  Total RMSE: {results_df["total_rmse"].mean():.3f} (±{results_df["total_rmse"].std():.3f})')
print(f'  Margin MAE: {results_df["margin_mae"].mean():.3f} (±{results_df["margin_mae"].std():.3f})')
print(f'  Margin RMSE: {results_df["margin_rmse"].mean():.3f} (±{results_df["margin_rmse"].std():.3f})')
print(f'  ROI: {results_df["roi"].mean():.1f}% (±{results_df["roi"].std():.1f}%)')
print()
print('PREGAME COMPLETE BACKTEST DONE!')
