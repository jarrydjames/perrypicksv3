"""Compile all model readouts into a single comprehensive report."""

import os
import joblib
import pandas as pd
from datetime import datetime

def generate_summary():
    """Generate comprehensive summary of all models."""
    
    lines = []
    lines.append('='*70)
    lines.append('PERRY PICKS - COMPREHENSIVE MODEL BACKTEST REPORT')
    lines.append('='*70)
    lines.append(f'Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    lines.append('')
    
    # Halftime
    lines.append('-'*70)
    lines.append('HALFTIME MODEL')
    lines.append('-'*70)
    lines.append('')
    
    if os.path.exists('data/processed/halftime_cv_results.parquet'):
        cv = pd.read_parquet('data/processed/halftime_cv_results.parquet')
        lines.append(f'Folds completed: {len(cv)}')
        lines.append(f'h2_total MAE (Ridge): {cv["h2_total_ridge_mae"].mean():.3f}')
        lines.append(f'h2_total MAE (RF): {cv["h2_total_rf_mae"].mean():.3f}')
        lines.append(f'h2_total MAE (GBT): {cv["h2_total_gbt_mae"].mean():.3f}')
        
        # Check for champion model
        if os.path.exists('models/team_2h_total.joblib'):
            model = joblib.load('models/team_2h_total.joblib')
            lines.append(f'Champion: {model.get("model_name", "UNKNOWN")}')
            lines.append(f'Calibrated SD: {model.get("sd", 0):.3f}')
    else:
        lines.append('Halftime results not available yet')
    lines.append('')
    
    # Pregame
    lines.append('-'*70)
    lines.append('PREGAME MODEL')
    lines.append('-'*70)
    lines.append('')
    
    if os.path.exists('data/processed/pregame_readout.txt'):
        with open('data/processed/pregame_readout.txt') as f:
            content = f.read()
            lines.extend(content.split('\n')[:40])
            lines.append('(Full readout in data/processed/pregame_readout.txt)')
    else:
        lines.append('Pregame results not available yet')
    lines.append('')
    
    # Q3
    lines.append('-'*70)
    lines.append('Q3 MODEL')
    lines.append('-'*70)
    lines.append('')
    
    if os.path.exists('data/processed/q3_readout.txt'):
        with open('data/processed/q3_readout.txt') as f:
            content = f.read()
            lines.extend(content.split('\n')[:40])
            lines.append('(Full readout in data/processed/q3_readout.txt)')
    else:
        lines.append('Q3 results not available yet')
    lines.append('')
    
    # Dataset sizes
    lines.append('-'*70)
    lines.append('DATASET SIZES')
    lines.append('-'*70)
    lines.append('')
    
    try:
        if os.path.exists('data/processed/pregame_team_v2.parquet'):
            pg = pd.read_parquet('data/processed/pregame_team_v2.parquet')
            lines.append(f'Pregame: {len(pg)} games')
        
        if os.path.exists('data/processed/q3_team_v2.parquet'):
            q3 = pd.read_parquet('data/processed/q3_team_v2.parquet')
            lines.append(f'Q3: {len(q3)} games')
        
        if os.path.exists('data/processed/halftime_training_23_24_leakage_free.parquet'):
            ht = pd.read_parquet('data/processed/halftime_training_23_24_leakage_free.parquet')
            lines.append(f'Halftime: {len(ht)} games')
    except Exception as e:
        lines.append(f'Error reading datasets: {e}')
    
    lines.append('')
    lines.append('='*70)
    
    return '\n'.join(lines)

if __name__ == '__main__':
    summary = generate_summary()
    print(summary)
    
    with open('data/processed/COMPREHENSIVE_BACKTEST_REPORT.txt', 'w') as f:
        f.write(summary)
    
    print('\n\nSaved to: data/processed/COMPREHENSIVE_BACKTEST_REPORT.txt')
