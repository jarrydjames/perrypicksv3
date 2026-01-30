"""
Merge temporal features with halftime statistics.

This combines game-level temporal features with first-half team statistics
for each game, creating an enriched dataset ready for model training.
"""
import pandas as pd
import os

def main():
    print('=' * 70)
    print('MERGE TEMPORAL + HALFTIME FEATURES')
    print('=' * 70)
    
    # Load temporal features
    print('\nLoading temporal features...')
    temporal = pd.read_parquet('data/processed/games_with_temporal_features.parquet')
    print(f'  Games: {len(temporal)}')
    
    # Load halftime data
    print('Loading halftime training data...')
    halftime = pd.read_parquet('data/processed/halftime_training_23_24_leakage_free.parquet')
    print(f'  Games: {len(halftime)}')
    
    # Show available columns from both
    print('\nTemporal features available:')
    temporal_cols = [c for c in temporal.columns if c not in ['game_id', 'game_date', 'game_code', 'home_team_id', 'home_team_name', 'home_team_city', 'home_team_code', 'home_score', 'away_team_id', 'away_team_name', 'away_team_city', 'away_team_code', 'away_score', 'periods', 'home_h1', 'away_h1', 'home_q3', 'away_q3', 'home_final', 'away_final', 'margin', 'total', 'team_id_x', 'team_id_y', 'game_date_x', 'home_team_id', 'home_pts_scored_avg_5', 'home_pts_allowed_avg_5', 'home_margin_avg_5', 'home_current_streak_5', 'home_days_since_last', 'home_is_back_to_back', 'away_team_id', 'away_pts_scored_avg_5', 'away_pts_allowed_avg_5', 'away_margin_avg_5', 'away_current_streak_5', 'away_days_since_last', 'away_is_back_to_back', 'game_date_y']]
    for col in temporal_cols[:15]:
        print(f'  - {col}')
    
    print('\nHalftime features available:')
    halftime_cols = [c for c in halftime.columns if c not in ['periods', 'home_team_id', 'home_team_name', 'home_team_city', 'home_team_code', 'home_score', 'away_team_id', 'away_team_name', 'away_team_city', 'away_team_code', 'away_score', 'home_h1', 'away_h1', 'home_q3', 'away_q3', 'home_final', 'away_final', 'margin', 'total']]
    for col in halftime_cols[:15]:
        print(f'  - {col}')
    
    # Merge on game_id
    print('\nMerging on game_id...')
    merged = halftime.merge(
        temporal[['game_id', 'game_date', 
                 'home_team_id', 'home_pts_scored_avg_5', 'home_pts_allowed_avg_5', 'home_margin_avg_5', 'home_current_streak_5', 
                 'home_days_since_last', 'home_is_back_to_back',
                 'away_team_id', 'away_pts_scored_avg_5', 'away_pts_allowed_avg_5', 'away_margin_avg_5', 'away_current_streak_5',
                 'away_days_since_last', 'away_is_back_to_back']],
        on='game_id',
        how='left'
    )
    
    print(f'  Merged games: {len(merged)}')
    print(f'  Features: {len(merged.columns)}')
    
    # Save
    output_path = 'data/processed/halftime_with_temporal_features.parquet'
    merged.to_parquet(output_path, index=False)
    print(f'\nSaved merged dataset: {output_path}')
    
    # Summary
    print('\n' + '=' * 70)
    print('MERGE COMPLETE')
    print('=' * 70)
    print(f'\nFinal dataset: {len(merged)} games')
    print(f'Features: {len(merged.columns)}')
    print(f'\nNew temporal features added:')
    print(f'  home_pts_scored_avg_5')
    print(f'  home_pts_allowed_avg_5')
    print(f'  home_margin_avg_5')
    print(f'  home_current_streak_5')
    print(f'  home_days_since_last')
    print(f'  home_is_back_to_back')
    print(f'  away_pts_scored_avg_5')
    print(f'  away_pts_allowed_avg_5')
    print(f'  away_margin_avg_5')
    print(f'  away_current_streak_5')
    print(f'  away_days_since_last')
    print(f'  away_is_back_to_back')
    print('=' * 70)

if __name__ == '__main__':
    main()
