import joblib
import pandas as pd
import numpy as np

def calibrate_sd(df, model, feature_cols, target_col):
    X = df[feature_cols].values
    y = df[target_col].values
    pred = model.predict(X)
    residuals = y - pred
    sd = np.percentile(np.abs(residuals), 80) / 1.2816
    return sd

print('CALIBRATING AND UPDATING MODELS')
print()

# Pregame
print('Pregame models:')
df_pg = pd.read_parquet('data/processed/pregame_team_v2.parquet')
feature_cols_pg = [c for c in df_pg.columns if c.endswith('_efg') or c.endswith('_ftr') or c.endswith('_tpar') or c.endswith('_tor') or c.endswith('_orbp') or c.endswith('_fga') or c.endswith('_fgm')]

for target in ['total', 'margin']:
    model_path = f'models_v3/pregame/ridge_{target}.joblib'
    model_obj = joblib.load(model_path)
    model = model_obj['model']
    sd = calibrate_sd(df_pg, model, feature_cols_pg, target)
    
    model_obj['sd'] = sd
    joblib.dump(model_obj, model_path)
    print(f'  ridge_{target}: SD={sd:.2f}')

print()

# Q3
print('Q3 models:')
df_q3 = pd.read_parquet('data/processed/q3_team_v2.parquet')
feature_cols_q3 = [c for c in df_q3.columns if c.startswith('q3_') or c.endswith('_efg') or c.endswith('_ftr') or c.endswith('_tpar') or c.endswith('_tor') or c.endswith('_orbp')]

for target in ['total', 'margin']:
    model_path = f'models_v3/q3/ridge_{target}.joblib'
    model_obj = joblib.load(model_path)
    model = model_obj['model']
    sd = calibrate_sd(df_q3, model, feature_cols_q3, target)
    
    model_obj['sd'] = sd
    joblib.dump(model_obj, model_path)
    print(f'  ridge_{target}: SD={sd:.2f}')

print()
print('ALL MODELS UPDATED WITH CALIBRATED SD')
