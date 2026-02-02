import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

# Load data
df = pd.read_parquet('data/processed/pregame_leakage_free.parquet')
df_sorted = df.sort_values('game_date').reset_index(drop=True)

# Splits
n = len(df_sorted)
train_end = int(n * 0.7)
val_end = int(n * 0.9)

train_df = df_sorted.iloc[:train_end]
val_df = df_sorted.iloc[train_end:val_end]
test_df = df_sorted.iloc[val_end:]

# ALL features (points + Four Factors)
feature_cols = ['home_pts', 'away_pts', 
                'home_efg', 'home_ftr', 'home_tpar', 'home_tor', 'home_orbp',
                'away_efg', 'away_ftr', 'away_tpar', 'away_tor', 'away_orbp']

def clean_data(df):
    X = df[feature_cols].values
    y_total = df['total'].values
    y_margin = df['margin'].values
    mask = ~(np.isnan(y_total) | np.isnan(y_margin))
    return X[mask], y_total[mask], y_margin[mask]

X_train, y_train_total, y_train_margin = clean_data(train_df)
X_val, y_val_total, y_val_margin = clean_data(val_df)
X_test, y_test_total, y_test_margin = clean_data(test_df)

print(f'Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}')
print(f'Features: {feature_cols}')
print()

# Models
models = {
    'Linear': LinearRegression(),
    'Ridge': Ridge(alpha=1.0),
    'RF': RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42),
    'GB': GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=42),
}

for target_name, y_train, y_val, y_test in [('Total', y_train_total, y_val_total, y_test_total),
                                            ('Margin', y_train_margin, y_val_margin, y_test_margin)]:
    print(f'{'='*60}')
    print(f'TARGET: {target_name}')
    print('='*60)
    
    best_model = None
    best_val_mae = float('inf')
    
    for model_name, model in models.items():
        model.fit(X_train, y_train)
        train_pred = model.predict(X_train)
        val_pred = model.predict(X_val)
        
        train_mae = mean_absolute_error(y_train, train_pred)
        val_mae = mean_absolute_error(y_val, val_pred)
        train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
        val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))
        
        print(f'{model_name}: Train MAE={train_mae:.2f}, Val MAE={val_mae:.2f}')
        
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_model = model
            best_model_name = model_name
    
    # Test best model
    test_pred = best_model.predict(X_test)
    test_mae = mean_absolute_error(y_test, test_pred)
    test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
    test_r2 = r2_score(y_test, test_pred)
    
    print(f'\nBEST ({target_name}): {best_model_name}')
    print(f'  Test MAE: {test_mae:.2f}, RMSE: {test_rmse:.2f}, R²={test_r2:.3f}')
    print(f'  Val MAE: {best_val_mae:.2f}')
    
    # Save
    joblib.dump(best_model, f'data/models/{target_name.lower()}_model.pkl')
    print(f'  Saved to data/models/{target_name.lower()}_model.pkl')
    print()
