def main():
    logger.info("="*70)
    logger.info("OPTION A PHASE 2: BUILD TRUE LEAKAGE-FREE PREGAME DATASET")
    logger.info("="*70)
    
    # Create builder
    builder = TruePregameBuilder()
    
    # Build dataset (process ALL games)
    df = builder.build_pregame_dataset(max_games=None)  # None = all games
    
    # Save dataset
    builder.save_dataset(df, "data/processed/pregame_leakage_free.parquet")
    
    logger.info("\n" + "="*70)
    logger.info("✅ PHASE 2 COMPLETE - LEAKAGE-FREE PREGAME DATASET BUILT")
    logger.info("="*70)
    
    return 0

if __name__ == '__main__':
    import sys
    sys.path.insert(0, '.')
    from OPTIONA_PHASE2_BUILD_TRUE_PREGAME import TruePregameBuilder, logger
    exit(main())
