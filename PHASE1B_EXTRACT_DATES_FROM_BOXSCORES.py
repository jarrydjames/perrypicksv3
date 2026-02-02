"""
PHASE 1B: Extract Game Dates from Boxscore Files

Extracts dates from existing boxscore JSON files to add temporal ordering
to the entire pregame dataset (all 3,520 games).
"""

import json
import pandas as pd
from pathlib import Path
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def extract_dates_from_boxscores(boxscore_dir: str = 'data/raw/box') -> pd.DataFrame:
    """Extract game IDs and dates from boxscore JSON files."""
    
    logger.info("="*70)
    logger.info("EXTRACTING DATES FROM BOXSCORE FILES")
    logger.info("="*70)
    
    boxscore_path = Path(boxscore_dir)
    
    if not boxscore_path.exists():
        logger.error(f"❌ Boxscore directory not found: {boxscore_dir}")
        return None
    
    # Get all JSON files
    json_files = list(boxscore_path.glob('*.json'))
    logger.info(f"📊 Found {len(json_files)} boxscore files")
    
    # Extract dates
    game_dates = []
    
    for i, json_file in enumerate(json_files):
        try:
            game_id = json_file.stem
            
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            # Try to find game date
            game_date = None
            
            # Try different paths
            if 'game' in data:
                game = data['game']
                game_date = game.get('gameEt') or game.get('gameTimeUTC') or game.get('gameTimeLocal')
            elif 'meta' in data:
                meta = data['meta']
                game_date = meta.get('gameDate') or meta.get('gameTime')
            
            if game_date:
                # Try to parse
                try:
                    if 'T' in game_date:
                        # ISO format
                        dt = pd.to_datetime(game_date)
                    else:
                        # Try other formats
                        dt = pd.to_datetime(game_date)
                    
                    game_dates.append({
                        'gameId': game_id,
                        'gameDate': game_date,
                        'game_date': dt
                    })
                except Exception as e:
                    logger.warning(f"⚠️ Could not parse date for {game_id}: {game_date}")
            else:
                logger.warning(f"⚠️ No date found for {game_id}")
            
            if (i + 1) % 500 == 0:
                logger.info(f"   Processed {i + 1}/{len(json_files)} files...")
        
        except Exception as e:
            logger.error(f"❌ Error processing {json_file}: {e}")
    
    df = pd.DataFrame(game_dates)
    logger.info(f"✅ Extracted dates for {len(df)} games")
    
    return df


def merge_with_pregame(pregame_path: str, dates_df: pd.DataFrame) -> pd.DataFrame:
    """Merge dates with pregame dataset."""
    
    logger.info("="*70)
    logger.info("MERGING DATES WITH PREGAME DATASET")
    logger.info("="*70)
    
    # Load pregame data
    pregame_df = pd.read_parquet(pregame_path)
    logger.info(f"📊 Loaded {len(pregame_df)} pregame games")
    
    # Merge
    pregame_df['gameId'] = pregame_df['game_id'].astype(str)
    dates_df['gameId'] = dates_df['gameId'].astype(str)
    
    merged = pregame_df.merge(dates_df[['gameId', 'game_date']], on='gameId', how='left')
    
    # Check missing dates
    missing = merged['game_date'].isna().sum()
    logger.info(f"📊 Missing dates: {missing}/{len(merged)} ({missing/len(merged)*100:.1f}%)")
    
    # Sort by date
    merged = merged.sort_values('game_date').reset_index(drop=True)
    
    # Show date range
    min_date = merged['game_date'].min()
    max_date = merged['game_date'].max()
    logger.info(f"📅 Date range: {min_date} to {max_date}")
    
    # Save
    output_path = 'data/processed/pregame_with_full_dates.parquet'
    merged.to_parquet(output_path, index=False)
    logger.info(f"✅ Saved merged dataset to {output_path}")
    
    return merged


def main():
    """Main entry point."""
    try:
        # Extract dates from boxscores
        dates_df = extract_dates_from_boxscores('data/raw/box')
        
        if dates_df is None or len(dates_df) == 0:
            raise Exception("Failed to extract dates from boxscores")
        
        # Merge with pregame data
        merged_df = merge_with_pregame('data/processed/pregame_team_v2.parquet', dates_df)
        
        # Save dates separately
        dates_df.to_parquet('data/processed/game_dates_from_boxscores.parquet', index=False)
        logger.info("✅ Saved dates to data/processed/game_dates_from_boxscores.parquet")
        
        logger.info("="*70)
        logger.info("✅ PHASE 1B COMPLETE - DATES EXTRACTED FROM BOXSCORES")
        logger.info("="*70)
        
        return 0
        
    except Exception as e:
        logger.error(f"❌ PHASE 1B FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    exit(main())
