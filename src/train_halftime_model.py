"""
Train halftime model with temporal features.

This is a wrapper around the training infrastructure in src/modeling/train_models.py
that specifically uses the dataset with temporal features added.

Usage:
    python3 src/train_halftime_model.py
    python3 src/train_halftime_model.py --include-xgb
    python3 src/train_halftime_model.py --include-cat
"""
import sys
sys.path.insert(0, '/Users/jarrydhawley/Desktop/Predictor/PerryPicks v3')

from pathlib import Path
import argparse

from src.modeling.train_models import train_from_parquet

def main():
    parser = argparse.ArgumentParser(description='Train halftime model with temporal features')
    parser.add_argument('--dataset', type=str, 
                       default='data/processed/halftime_with_temporal_features.parquet',
                       help='Path to training dataset with temporal features')
    parser.add_argument('--output-dir', type=str, 
                       default='models_v3/halftime',
                       help='Output directory for trained models')
    parser.add_argument('--include-xgb', action='store_true',
                       help='Include XGBoost models')
    parser.add_argument('--include-cat', action='store_true',
                       help='Include CatBoost models')
    
    args = parser.parse_args()
    
    print('=' * 70)
    print('HALFTIME MODEL TRAINING WITH TEMPORAL FEATURES')
    print('=' * 70)
    print(f'\nDataset: {args.dataset}')
    print(f'Output directory: {args.output_dir}')
    print(f'Include XGBoost: {args.include_xgb}')
    print(f'Include CatBoost: {args.include_cat}')
    print('=' * 70)
    
    # Check if dataset exists
    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        print(f'❌ ERROR: Dataset not found: {args.dataset}')
        print(f'\nPlease run: python3 src/merge_temporal_halftime.py')
        print(f'To merge temporal features with halftime stats')
        sys.exit(1)
    
    # Train models using existing infrastructure
    print('\nTraining models (Ridge, RandomForest, GBT)...')
    
    if args.include_xgb:
        print('Including XGBoost models...')
    
    if args.include_cat:
        print('Including CatBoost models...')
    
    train_from_parquet(
        parquet_path=dataset_path,
        out_dir=Path(args.output_dir),
        include_xgb=args.include_xgb,
        include_cat=args.include_cat
    )
    
    # Summary
    print('\n' + '=' * 70)
    print('TRAINING COMPLETE')
    print('=' * 70)
    print(f'\nModels saved to: {args.output_dir}')
    print('=' * 70)
    print('\nNext:')
    print('  1. Update app.py to use new model')
    print('  2. Set up cron job for daily refresh')
    print('  3. Test predictions with new model')

if __name__ == '__main__':
    main()
