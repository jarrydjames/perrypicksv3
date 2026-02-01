import joblib

print('UPDATING MODEL NAMES')
print()

# Update Pregame models
for target in ['total', 'margin']:
    model_path = f'models_v3/pregame/ridge_{target}.joblib'
    model_obj = joblib.load(model_path)
    model_obj['model_name'] = 'RIDGE'
    joblib.dump(model_obj, model_path)
    print(f'Updated pregame ridge_{target}')

# Update Q3 models
for target in ['total', 'margin']:
    model_path = f'models_v3/q3/ridge_{target}.joblib'
    model_obj = joblib.load(model_path)
    model_obj['model_name'] = 'RIDGE'
    joblib.dump(model_obj, model_path)
    print(f'Updated q3 ridge_{target}')

print()
print('ALL MODELS UPDATED')
