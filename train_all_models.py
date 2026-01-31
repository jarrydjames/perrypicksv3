"""
Comprehensive Model Training - Train 7 models and compare.

Models:
1. Ridge Regression (baseline)
2. Random Forest
3. XGBoost
4. Neural Networks (MLPRegressor)
5. ElasticNet
6. Gradient Boosting
7. LightGBM
"""

import sys
sys.path.insert(0, '/Users/jarrydhawley/Desktop/Predictor/PerryPicks v3')

import pandas as pd
import numpy as np
from datetime import datetime, timezone
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

# Local imports
from src.registry import ModelRegistryExtended, ModelMetadata

print("=" * 80)
print("COMPREHENSIVE MODEL TRAINING - 7 MODELS")
print("=" * 80)
print("")

# Load dataset
df = pd.read_parquet('data/processed/halftime_with_temporal_features_total.parquet')
print(f"Loaded dataset: {len(df)} rows, {len(df.columns)} columns")

# Features and target
h1_features = [col for col in df.columns if col.startswith('h1_')]
X = df[h1_features]
y = df['h2_total']

print(f"Features: {len(h1_features)} (h1_*)")
print(f"Target: h2_total")
print("")

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"Train set: {len(X_train)} samples ({len(X_train)/len(X)*100:.1f}%)")
print(f"Test set: {len(X_test)} samples ({len(X_test)/len(X)*100:.1f}%)")
print("")

# Initialize model registry
registry = ModelRegistryExtended(registry_dir="model_registry_comprehensive")
print("Model registry initialized: model_registry_comprehensive/")
print("")

# Define models
models = {
    'ridge': {
        'name': 'ridge_regression',
        'model': Ridge(alpha=2.0, random_state=42),
        'type': 'ridge',
        'hyperparameters': {'alpha': 2.0, 'solver': 'auto', 'random_state': 42},
        'is_baseline': True,
        'tags': ['baseline', 'ridge', 'linear'],
    },
    'random_forest': {
        'name': 'random_forest',
        'model': RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
        ),
        'type': 'random_forest',
        'hyperparameters': {
            'n_estimators': 100,
            'max_depth': 10,
            'min_samples_split': 5,
            'min_samples_leaf': 2,
            'random_state': 42,
            'n_jobs': -1,
        },
        'is_baseline': False,
        'tags': ['random_forest', 'tree', 'ensemble'],
    },
    'xgboost': {
        'name': 'xgboost',
        'model': xgb.XGBRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
        ),
        'type': 'xgboost',
        'hyperparameters': {
            'n_estimators': 100,
            'max_depth': 6,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
            'n_jobs': -1,
        },
        'is_baseline': False,
        'tags': ['xgboost', 'boosting', 'ensemble'],
    },
    'mlp': {
        'name': 'neural_network',
        'model': MLPRegressor(
            hidden_layer_sizes=(64, 32, 16),
            activation='relu',
            solver='adam',
            learning_rate_init=0.001,
            max_iter=500,
            early_stopping=True,
            validation_fraction=0.1,
            random_state=42,
        ),
        'type': 'neural_network',
        'hyperparameters': {
            'hidden_layer_sizes': (64, 32, 16),
            'activation': 'relu',
            'solver': 'adam',
            'learning_rate_init': 0.001,
            'max_iter': 500,
            'early_stopping': True,
            'validation_fraction': 0.1,
            'random_state': 42,
        },
        'is_baseline': False,
        'tags': ['neural_network', 'mlp', 'deep_learning'],
    },
    'elasticnet': {
        'name': 'elastic_net',
        'model': ElasticNet(
            alpha=1.0,
            l1_ratio=0.5,
            random_state=42,
        ),
        'type': 'elastic_net',
        'hyperparameters': {
            'alpha': 1.0,
            'l1_ratio': 0.5,
            'random_state': 42,
        },
        'is_baseline': False,
        'tags': ['elastic_net', 'linear', 'regularization'],
    },
    'gradient_boosting': {
        'name': 'gradient_boosting',
        'model': GradientBoostingRegressor(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            subsample=0.8,
            random_state=42,
        ),
        'type': 'gradient_boosting',
        'hyperparameters': {
            'n_estimators': 100,
            'max_depth': 5,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'random_state': 42,
        },
        'is_baseline': False,
        'tags': ['gradient_boosting', 'boosting', 'ensemble'],
    },
    'lightgbm': {
        'name': 'lightgbm',
        'model': lgb.LGBMRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            verbose=-1,
        ),
        'type': 'lightgbm',
        'hyperparameters': {
            'n_estimators': 100,
            'max_depth': 6,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
            'n_jobs': -1,
        },
        'is_baseline': False,
        'tags': ['lightgbm', 'boosting', 'ensemble'],
    },
}

# Train all models
results = {}
model_ids = {}

for model_key, model_info in models.items():
    print(f"Training {model_key.upper()}...")
    
    # Train model
    model = model_info['model']
    model.fit(X_train, y_train)
    
    # Generate predictions
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    # Calculate metrics
    mae_train = mean_absolute_error(y_train, y_pred_train)
    mae_test = mean_absolute_error(y_test, y_pred_test)
    rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
    rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
    r2_train = r2_score(y_train, y_pred_train)
    r2_test = r2_score(y_test, y_pred_test)
    
    # Store results
    results[model_key] = {
        'mae_train': mae_train,
        'mae_test': mae_test,
        'rmse_train': rmse_train,
        'rmse_test': rmse_test,
        'r2_train': r2_train,
        'r2_test': r2_test,
        'y_pred_train': y_pred_train,
        'y_pred_test': y_pred_test,
    }
    
    # Create metadata
    metadata = ModelMetadata(
        model_name=model_info['name'],
        version="v1.0.0",
        hyperparameters=model_info['hyperparameters'],
        metrics={
            'mae': mae_test,
            'rmse': rmse_test,
            'r2': r2_test,
        },
        dataset_info={
            'n_samples': len(df),
            'n_features': len(h1_features),
            'dataset': 'halftime_with_temporal_features_total.parquet',
            'checksum': '0b8b8bffc5916f58',
        },
        features=h1_features,
        target='h2_total',
        model_type=model_info['type'],
        is_baseline=model_info['is_baseline'],
        is_deployed=False,
        tags=model_info['tags'],
        notes=f"Trained on {len(X_train)} samples, tested on {len(X_test)} samples",
    )
    
    # Register model
    model_id = registry.register_model(
        model=model,
        metadata=metadata,
    )
    model_ids[model_key] = model_id
    
    print(f"  MAE (train): {mae_train:.4f}")
    print(f"  MAE (test): {mae_test:.4f}")
    print(f"  RMSE (test): {rmse_test:.4f}")
    print(f"  R² (test): {r2_test:.4f}")
    print(f"  Model ID: {model_id[:8]}...")
    print("")

# Create comparison DataFrame
comparison_df = pd.DataFrame({
    'Model': [
        'Ridge Regression',
        'Random Forest',
        'XGBoost',
        'Neural Network',
        'ElasticNet',
        'Gradient Boosting',
        'LightGBM',
    ],
    'MAE (train)': [results['ridge']['mae_train'], results['random_forest']['mae_train'], results['xgboost']['mae_train'], results['mlp']['mae_train'], results['elasticnet']['mae_train'], results['gradient_boosting']['mae_train'], results['lightgbm']['mae_train']],
    'MAE (test)': [results['ridge']['mae_test'], results['random_forest']['mae_test'], results['xgboost']['mae_test'], results['mlp']['mae_test'], results['elasticnet']['mae_test'], results['gradient_boosting']['mae_test'], results['lightgbm']['mae_test']],
    'RMSE (test)': [results['ridge']['rmse_test'], results['random_forest']['rmse_test'], results['xgboost']['rmse_test'], results['mlp']['rmse_test'], results['elasticnet']['rmse_test'], results['gradient_boosting']['rmse_test'], results['lightgbm']['rmse_test']],
    'R² (test)': [results['ridge']['r2_test'], results['random_forest']['r2_test'], results['xgboost']['r2_test'], results['mlp']['r2_test'], results['elasticnet']['r2_test'], results['gradient_boosting']['r2_test'], results['lightgbm']['r2_test']],
})

# Sort by MAE (test)
comparison_df = comparison_df.sort_values('MAE (test)')
comparison_df['Rank'] = range(1, len(comparison_df) + 1)
comparison_df = comparison_df[['Rank', 'Model', 'MAE (train)', 'MAE (test)', 'RMSE (test)', 'R² (test)']]

print("=" * 80)
print("MODEL COMPARISON - RANKED BY MAE (TEST)")
print("=" * 80)
print(comparison_df.to_string(index=False))
print("")

# Find best model
best_model_key = comparison_df.iloc[0]['Model'].lower().replace(' ', '_')
if best_model_key not in results:
    # Map model names to keys
    model_name_map = {
        'ridge_regression': 'ridge',
        'random_forest': 'random_forest',
        'xgboost': 'xgboost',
        'neural_network': 'mlp',
        'elastic_net': 'elasticnet',
        'gradient_boosting': 'gradient_boosting',
        'lightgbm': 'lightgbm',
    }
    best_model_key = model_name_map.get(comparison_df.iloc[0]['Model'].lower().replace(' ', '_'), best_model_key)

print(f"Best Model: {comparison_df.iloc[0]['Model']}")
print(f"  MAE (test): {comparison_df.iloc[0]['MAE (test)']:.4f}")
print(f"  RMSE (test): {comparison_df.iloc[0]['RMSE (test)']:.4f}")
print(f"  R² (test): {comparison_df.iloc[0]['R² (test)']:.4f}")
print("")

# Deploy best model
best_model_id = model_ids[best_model_key]
registry.deploy_model(best_model_id)
print(f"Deployed best model: {best_model_id[:8]}...")
print("")

# Save comparison to CSV
comparison_path = 'data/processed/model_comparison.csv'
comparison_df.to_csv(comparison_path, index=False)
print(f"Saved model comparison to: {comparison_path}")
print("")

# Create predictions DataFrame
predictions_df = pd.DataFrame({
    'season_end_yy': df.loc[X_test.index, 'season_end_yy'].values,
    'game_id': df.loc[X_test.index, 'game_id'].values,
    'h1_home': df.loc[X_test.index, 'h1_home'].values,
    'h1_away': df.loc[X_test.index, 'h1_away'].values,
    'h1_total': df.loc[X_test.index, 'h1_total'].values,
    'h1_margin': df.loc[X_test.index, 'h1_margin'].values,
    'h2_total_true': y_test.values,
    'pred_ridge': results['ridge']['y_pred_test'],
    'pred_random_forest': results['random_forest']['y_pred_test'],
    'pred_xgboost': results['xgboost']['y_pred_test'],
    'pred_neural_network': results['mlp']['y_pred_test'],
    'pred_elasticnet': results['elasticnet']['y_pred_test'],
    'pred_gradient_boosting': results['gradient_boosting']['y_pred_test'],
    'pred_lightgbm': results['lightgbm']['y_pred_test'],
})

# Save predictions to CSV
predictions_path = 'data/processed/model_predictions.csv'
predictions_df.to_csv(predictions_path, index=False)
print(f"Saved predictions to: {predictions_path}")
print("")

print("=" * 80)
print("TRAINING COMPLETE")
print("=" * 80)
print(f"\nTotal models trained: {len(models)}")
print(f"Models registered: {len(model_ids)}")
print(f"Best model: {comparison_df.iloc[0]['Model']}")
print(f"Best MAE: {comparison_df.iloc[0]['MAE (test)']:.4f}")
print(f"\nOutput files:")
print(f"  - {comparison_path}")
print(f"  - {predictions_path}")
print(f"  - model_registry_comprehensive/")
print("")
