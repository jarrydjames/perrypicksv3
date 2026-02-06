"""
Comprehensive Training: Train ALL 7 Models for ALL States (Pregame, Halftime, Q3)

Models (7):
1. Ridge Regression
2. Random Forest
3. XGBoost
4. Neural Network (MLPRegressor)
5. ElasticNet
6. Gradient Boosting
7. LightGBM

States (3):
- Pregame (targets: total, margin)
- Halftime (targets: h2_total, h2_margin)
- Q3 (targets: q3_total, q3_margin)

This script trains ALL models, backtests ALL models, and generates comprehensive results.
"""

import sys
sys.path.insert(0, "/Users/jarrydhawley/Desktop/Predictor/PerryPicks v3")

import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path
from datetime import datetime
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

# Configuration
DATA_DIR = Path("data/processed")
MODELS_DIR = Path("models_v3")
OUTPUT_DIR = Path("data/processed")

# Dataset configurations
DATASETS = {
    "pregame": {
        "path": DATA_DIR / "pregame_team_v2.parquet",
        "targets": ["total", "margin"],
        "feature_prefix": "",  # All columns except game_id, home_tri, away_tri
        "output_suffix": "pregame",
    },
    "halftime": {
        "path": DATA_DIR / "halftime_with_temporal_features_total.parquet",
        "targets": ["h2_total", "h2_margin"],
        "feature_prefix": "h1_",  # h1_* features
        "output_suffix": "halftime",
    },
    "q3": {
        "path": DATA_DIR / "q3_team_v2.parquet",
        "targets": ["q3_total", "q3_margin"],
        "feature_prefix": "",  # All columns except game_id, home_tri, away_tri, q3_*
        "output_suffix": "q3",
    },
}

# Model definitions
def get_models():
    """Return all 7 models with consistent hyperparameters."""
    return {
        'ridge': {
            'name': 'Ridge Regression',
            'model': Ridge(alpha=2.0, random_state=42),
            'params': {'alpha': 2.0, 'random_state': 42},
        },
        'random_forest': {
            'name': 'Random Forest',
            'model': RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1,
            ),
            'params': {
                'n_estimators': 100,
                'max_depth': 10,
                'min_samples_split': 5,
                'min_samples_leaf': 2,
                'random_state': 42,
                'n_jobs': -1,
            },
        },
        'xgboost': {
            'name': 'XGBoost',
            'model': xgb.XGBRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1,
                verbosity=0,
            ),
            'params': {
                'n_estimators': 100,
                'max_depth': 6,
                'learning_rate': 0.1,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'random_state': 42,
                'n_jobs': -1,
            },
        },
        'mlp': {
            'name': 'Neural Network',
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
            'params': {
                'hidden_layer_sizes': (64, 32, 16),
                'activation': 'relu',
                'solver': 'adam',
                'learning_rate_init': 0.001,
                'max_iter': 500,
                'early_stopping': True,
                'validation_fraction': 0.1,
                'random_state': 42,
            },
        },
        'elasticnet': {
            'name': 'ElasticNet',
            'model': ElasticNet(
                alpha=1.0,
                l1_ratio=0.5,
                random_state=42,
            ),
            'params': {
                'alpha': 1.0,
                'l1_ratio': 0.5,
                'random_state': 42,
            },
        },
        'gradient_boosting': {
            'name': 'Gradient Boosting',
            'model': GradientBoostingRegressor(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                subsample=0.8,
                random_state=42,
            ),
            'params': {
                'n_estimators': 100,
                'max_depth': 5,
                'learning_rate': 0.1,
                'subsample': 0.8,
                'random_state': 42,
            },
        },
        'lightgbm': {
            'name': 'LightGBM',
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
            'params': {
                'n_estimators': 100,
                'max_depth': 6,
                'learning_rate': 0.1,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'random_state': 42,
                'n_jobs': -1,
            },
        },
    }


def get_features(df, state):
    """Extract feature columns for a state, ensuring no data leakage."""
    config = DATASETS[state]
    
    if state == "pregame":
        # CRITICAL: Only use rate-based efficiency features
        # DO NOT use any point totals, shot totals, or counting stats
        # These would leak the target information
        rate_features = [
            'home_efg', 'home_ftr', 'home_tpar', 'home_tor', 'home_orbp',
            'away_efg', 'away_ftr', 'away_tpar', 'away_tor', 'away_orbp',
        ]
        # Verify all rate features exist
        features = [col for col in rate_features if col in df.columns]
        print(f"  Using {len(features)} rate-based features (NO POINT TOTALS to prevent leakage)")
        print(f"  Features: {features}")
        
    elif state == "halftime":
        # Use ONLY h1_* features (first half - safe)
        # Targets are h2_* (second half)
        features = [col for col in df.columns if col.startswith("h1_")]
        # Double-check: exclude any h2-related features
        features = [col for col in features if "h2" not in col]
        print(f"  Using {len(features)} h1 features (first half stats only)")
        print(f"  Features: {features}")
        
    elif state == "q3":
        # CRITICAL: Only use pregame team rate-based stats
        # DO NOT use q3 game state features (q3_home, q3_away, etc.)
        # These would leak the target information
        rate_features = [
            'home_efg', 'home_ftr', 'home_tpar', 'home_tor', 'home_orbp',
            'away_efg', 'away_ftr', 'away_tpar', 'away_tor', 'away_orbp',
        ]
        # Verify all rate features exist
        features = [col for col in rate_features if col in df.columns]
        print(f"  Using {len(features)} rate-based features (NO Q3 GAME STATE to prevent leakage)")
        print(f"  Features: {features}")
        
    else:
        raise ValueError(f"Unknown state: {state}")
    
    return features


def train_and_evaluate(state, model_key, model_info, X_train, y_train, X_test, y_test, target_name):
    """Train and evaluate a single model."""
    model = model_info['model']
    
    # Train model
    model.fit(X_train, y_train)
    
    # Predictions
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    # Metrics
    mae_train = mean_absolute_error(y_train, y_pred_train)
    mae_test = mean_absolute_error(y_test, y_pred_test)
    rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
    rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
    r2_train = r2_score(y_train, y_pred_train)
    r2_test = r2_score(y_test, y_pred_test)
    
    return {
        'model': model,
        'mae_train': mae_train,
        'mae_test': mae_test,
        'rmse_train': rmse_train,
        'rmse_test': rmse_test,
        'r2_train': r2_train,
        'r2_test': r2_test,
        'y_pred_train': y_pred_train,
        'y_pred_test': y_pred_test,
    }


def train_state(state):
    """Train all models for a specific state."""
    print("=" * 80)
    print(f"TRAINING: {state.upper()}")
    print("=" * 80)
    print()
    
    config = DATASETS[state]
    
    # Load dataset
    df = pd.read_parquet(config["path"])
    print(f"Loaded dataset: {len(df)} rows, {len(df.columns)} columns")
    
    # Get features
    features = get_features(df, state)
    print(f"Features: {len(features)}")
    print(f"Targets: {config['targets']}")
    print()
    
    # Prepare results storage
    results = {}
    
    for target in config["targets"]:
        print(f"\n--- TARGET: {target} ---")
        
        # Check if target exists
        if target not in df.columns:
            print(f"⚠️  Target {target} not found in dataset, skipping")
            continue
        
        # Prepare data
        X = df[features].values
        y = df[target].values
        
        # Remove NaN
        mask = ~np.isnan(y).any(axis=0) if y.ndim > 1 else ~np.isnan(y)
        X = X[mask]
        y = y[mask]
        
        print(f"Samples after removing NaN: {len(X)}")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        print(f"Train: {len(X_train)}, Test: {len(X_test)}")
        
        # Train all models
        target_results = {}
        models = get_models()
        
        for model_key, model_info in models.items():
            print(f"  Training {model_info['name']}...", end=" ")
            
            try:
                result = train_and_evaluate(
                    state, model_key, model_info,
                    X_train, y_train, X_test, y_test, target
                )
                target_results[model_key] = result
                print(f"✅ MAE={result['mae_test']:.4f}")
            except Exception as e:
                print(f"❌ {str(e)[:50]}")
                target_results[model_key] = None
        
        results[target] = target_results
    
    return results, features


def save_models(state, results, features):
    """Save trained models for a state."""
    print(f"\nSaving models for {state}...")
    
    out_dir = MODELS_DIR / state
    out_dir.mkdir(parents=True, exist_ok=True)
    
    models = get_models()
    
    for target, target_results in results.items():
        for model_key, result in target_results.items():
            if result is None:
                continue
            
            model = result['model']
            model_name = models[model_key]['name'].lower().replace(' ', '_')
            model_file = out_dir / f"{model_name}_{target}.joblib"
            
            joblib.dump({
                'model': model,
                'features': features,
                'target': target,
                'state': state,
                'metrics': {
                    'mae_train': result['mae_train'],
                    'mae_test': result['mae_test'],
                    'rmse_test': result['rmse_test'],
                    'r2_test': result['r2_test'],
                },
            }, model_file)
            print(f"  Saved: {model_file.name}")


def generate_comparison(results, state):
    """Generate comparison table for a state."""
    comparison_rows = []
    
    for target, target_results in results.items():
        for model_key, result in target_results.items():
            if result is None:
                continue
            
            comparison_rows.append({
                'State': state,
                'Target': target,
                'Model': get_models()[model_key]['name'],
                'MAE (Train)': result['mae_train'],
                'MAE (Test)': result['mae_test'],
                'RMSE (Test)': result['rmse_test'],
                'R² (Test)': result['r2_test'],
            })
    
    df = pd.DataFrame(comparison_rows)
    
    # Sort by State -> Target -> MAE (Test)
    df = df.sort_values(['State', 'Target', 'MAE (Test)'])
    df['Rank'] = df.groupby(['State', 'Target']).cumcount() + 1
    
    return df


def main():
    """Main execution."""
    print("=" * 80)
    print("COMPREHENSIVE 7-MODEL TRAINING - ALL STATES, ALL TARGETS")
    print("=" * 80)
    print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Train all states
    all_results = {}
    all_features = {}
    all_comparisons = []
    
    for state in ["pregame", "halftime", "q3"]:
        results, features = train_state(state)
        all_results[state] = results
        all_features[state] = features
        
        # Save models
        save_models(state, results, features)
        
        # Generate comparison
        comparison = generate_comparison(results, state)
        all_comparisons.append(comparison)
        
        print()
        print(comparison.to_string(index=False))
        print()
    
    # Combine all comparisons
    combined_comparison = pd.concat(all_comparisons, ignore_index=True)
    
    # Save combined comparison
    comparison_file = OUTPUT_DIR / "all_7_models_comparison.csv"
    combined_comparison.to_csv(comparison_file, index=False)
    print(f"\nSaved combined comparison to: {comparison_file}")
    
    # Save results
    results_file = OUTPUT_DIR / "all_7_models_results.json"
    with open(results_file, 'w') as f:
        # Convert to serializable format
        serializable_results = {}
        for state, state_results in all_results.items():
            serializable_results[state] = {}
            for target, target_results in state_results.items():
                serializable_results[state][target] = {}
                for model_key, result in target_results.items():
                    if result is not None:
                        serializable_results[state][target][model_key] = {
                            'mae_train': float(result['mae_train']),
                            'mae_test': float(result['mae_test']),
                            'rmse_test': float(result['rmse_test']),
                            'r2_test': float(result['r2_test']),
                        }
        json.dump(serializable_results, f, indent=2)
    print(f"Saved results to: {results_file}")
    
    print()
    print("=" * 80)
    print("TRAINING COMPLETE")
    print("=" * 80)
    print(f"End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    print(f"Total models trained: {len(combined_comparison)}")
    print(f"States: 3 (pregame, halftime, q3)")
    print(f"Models per state: 7")
    print(f"Targets per state: {len(DATASETS['pregame']['targets'])}")
    print()


if __name__ == "__main__":
    main()