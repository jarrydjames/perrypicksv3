"""
Dataset de-duplication utility for PerryPicks v3

Removes exact duplicate rows and provides summary statistics.
"""
import pandas as pd
import sys
from pathlib import Path


def deduplicate_dataset(input_path: str, output_path: str = None, drop_exact: bool = False) -> pd.DataFrame:
    """
    De-duplicate dataset by removing exact duplicate rows.
    
    Args:
        input_path: Path to input parquet file
        output_path: Path to output parquet file (optional)
        drop_exact: If True, removes exact duplicate rows
                   If False, returns de-duplicated dataframe without saving
    
    Returns:
        De-duplicated dataframe
    """
    df = pd.read_parquet(input_path)
    
    print(f"Original dataset: {len(df)} rows, {len(df.columns)} columns")
    
    # Check for exact duplicates
    exact_duplicates = df.duplicated(keep=False)
    exact_duplicate_count = exact_duplicates.sum()
    unique_rows = df.drop_duplicates()
    unique_count = len(unique_rows)
    
    # Check for duplicate primary keys
    key_duplicates = df.duplicated(subset=['season_end_yy', 'game_id'], keep=False)
    key_duplicate_count = key_duplicates.sum()
    unique_games = df[['season_end_yy', 'game_id']].drop_duplicates().shape[0]
    
    print(f"\nDuplicate Analysis:")
    print(f"  Exact duplicate rows: {exact_duplicate_count}")
    print(f"  Duplicate primary keys: {key_duplicate_count}")
    print(f"  Unique rows: {unique_count}")
    print(f"  Unique games: {unique_games}")
    print(f"  Rows per game: {len(df) / unique_games:.2f}x")
    
    if drop_exact:
        df_dedup = df.drop_duplicates()
        print(f"\nDe-duplicated dataset: {len(df_dedup)} rows")
        print(f"  Removed: {len(df) - len(df_dedup)} exact duplicate rows")
        
        if output_path:
            df_dedup.to_parquet(output_path, index=False)
            print(f"\nSaved to: {output_path}")
            print(f"  File size: {Path(output_path).stat().st_size / 1024 / 1024:.2f} MB")
        
        return df_dedup
    else:
        print(f"\nDrop exact duplicates: Use --drop flag to remove {exact_duplicate_count} duplicate rows")
        return df


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python deduplicate_dataset.py <input.parquet> [output.parquet] [--drop]")
        print("\nOptions:")
        print("  input.parquet    Path to input parquet file")
        print("  output.parquet   Path to output parquet file (optional)")
        print("  --drop          Remove exact duplicate rows and save")
        sys.exit(1)
    
    input_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else None
    drop_exact = "--drop" in sys.argv
    
    # Auto-generate output path if not provided and --drop flag set
    if drop_exact and output_path is None:
        path_obj = Path(input_path)
        output_path = str(path_obj.parent / f"{path_obj.stem}_deduplicated.parquet")
    
    df_dedup = deduplicate_dataset(input_path, output_path, drop_exact)
