"""
Train halftime model with temporal features.

This is a wrapper around the training infrastructure in src/modeling/train_models.py
that specifically uses the dataset with temporal features added.

Usage:
    python3 src/train_halftime_model.py
    python3 src/train_halftime_model.py --retune
    python3 src/train_halftime_model.py --model-type gbt
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
    parser.add_argument('--model-type', type=str, default='gbt',
                       choices=['gbt', 'rf', 'ridge', 'all'],
                       help='Model type to train (default: gbt)')
    parser.add_argument('--include-xgb', action='store_true',
                       help='Include XGBoost models')
    parser.add_argument('--include-cat', action='store_true',
                       help='Include CatBoost models')
    parser.add_argument('--output-dir', type=str, 
                       default='models_v3/halftime',
                       help='Output directory for trained models')
    parser.add_argument('--n-estimators', type=int, default=100,
                       help='Number of estimators for tree-based models')
    parser.add_argument('--max-depth', type=int, default=6,
                       help='Maximum tree depth')
    parser.add_argument('--learning-rate', type=float, default=0.05,
                       help='Learning rate for boosting models')
    parser.add_argument('--retune', action='store_true',
                       help='Retune hyperparameters (default: use existing)')
    parser.add_argument('--test-split', type=float, default=0.2,
                       help='Test split fraction (e.g., 0.2 means train on 80%, test on 20%)')
    parser.add_argument('--random-seed', type=int, default=42,
                       help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    print('=' * 70)
    print('HALFTIME MODEL TRAINING WITH TEMPORAL FEATURES')
    print('=' * 70)
    print(f'\nDataset: {args.dataset}')
    print(f'Model type: {args.model_type}')
    print(f'Output directory: {args.output_dir}')
    print(f'N estimators: {args.n_estimators}')
    print(f'Max depth: {args.max_depth}')
    print(f'Learning rate: {args.learning_rate}')
    print(f'Test split: {args.test_split}')
    print(f'Random seed: {args.random_seed}')
    print('=' * 70)
    
    # Check if dataset exists
    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        print(f'❌ ERROR: Dataset not found: {args.dataset}')
        print(f'\nPlease run: python3 src/merge_temporal_halftime.py')
        print(f'To merge temporal features with halftime stats')
        sys.exit(1)
    
    # Determine which models to train
    include_xgb = args.include_xgb
    include_cat = args.include_cat
    
    if args.model_type == 'gbt':
        models = ['gbt']
    elif args.model_type == 'rf':
        models = ['rf']
    elif args.model_type == 'ridge':
        models = ['ridge']
    elif args.model_type == 'all':
        models = ['gbt', 'rf', 'ridge']
        if include_xgb:
            models.append('xgb')
        if include_cat:
            models.append('cat')
    else:
        print(f'Unknown model type: {args.model_type}')
        sys.exit(1)
    
    # Train models
    print('\nTraining models...')
    
    result = train_from_parquet(
        parquet_path=dataset_path,
        out_dir=args.output_dir,
        models=models,
        include_xgb=include_xgb,
        include_catboost=include_cat,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        test_split=args.test_split,
        random_seed=args.random_seed
    )
    
    # Summary
    print('\n' + '=' * 70)
    print('TRAINING COMPLETE')
    print('=' * 70)
    print(f'\nModels trained: {result.models_trained}')
    print(f'\nModels saved to: {args.output_dir}')
    print(f'\nEvaluation metrics:')
    for model_name, metrics in result.eval_metrics.items():
        print(f'  {model_name}:')
        for metric_name, value in metrics.items():
            print(f'    {metric_name}: {value:.4f}')
    print('=' * 70)
    print('\nNext:')
    print('  1. Update app.py to use new model')
    print('  2. Set up cron job for daily refresh')
    print('  3. Test predictions with new model')

if __name__ == '__main__':
    main()
