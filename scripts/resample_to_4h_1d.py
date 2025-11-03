#!/usr/bin/env python3
"""
Resample 1h data to 4h and 1d timeframes for ETHUSDT.

This script reads the existing 1h data from historical_data/binance/ethusdt/processed/ethusdt_1h/
and creates 4h and 1d resampled data in the same structure.
"""

import sys
from pathlib import Path
from datetime import datetime
import pandas as pd
import logging

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils.logger import system_logger

logger = system_logger.getChild("Resample4h1d")

def resample_ohlcv(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """
    Resample OHLCV data to specified timeframe.
    
    Args:
        df: DataFrame with timestamp index and OHLCV columns
        timeframe: Target timeframe ('4h' or '1d')
    
    Returns:
        Resampled DataFrame
    """
    # Define resampling rules
    resample_rules = {
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }
    
    # Convert timeframe to pandas frequency
    freq_map = {
        '4h': '4H',
        '1d': '1D'
    }
    
    if timeframe not in freq_map:
        raise ValueError(f"Unsupported timeframe: {timeframe}")
    
    freq = freq_map[timeframe]
    
    # Perform resampling
    logger.info(f"🔄 Resampling to {timeframe} using frequency {freq}")
    
    # Resample only OHLCV columns
    ohlcv_cols = ['open', 'high', 'low', 'close', 'volume']
    available_cols = [col for col in ohlcv_cols if col in df.columns]
    
    logger.info(f"   Available OHLCV columns: {available_cols}")
    logger.info(f"   Data index type: {type(df.index)}")
    logger.info(f"   Data index range: {df.index.min()} to {df.index.max()}")
    
    resampled = df[available_cols].resample(freq).agg({
        col: resample_rules[col] for col in available_cols
    })
    
    logger.info(f"   Rows after resampling: {len(resampled)}")
    logger.info(f"   NaN rows before dropna: {resampled.isna().all(axis=1).sum()}")
    
    # Handle any other columns (take last value)
    other_cols = [col for col in df.columns if col not in ohlcv_cols]
    if other_cols:
        for col in other_cols:
            if col not in resampled.columns:
                resampled[col] = df[col].resample(freq).last()
    
    # Drop rows with ALL NaN values (but keep rows with at least some data)
    resampled = resampled.dropna(how='all')
    
    logger.info(f"✅ Resampled: {len(df)} → {len(resampled)} rows")
    
    return resampled


def load_1h_data(base_path: Path) -> pd.DataFrame:
    """
    Load all 1h data from partitioned parquet files.
    
    Args:
        base_path: Path to ethusdt_1h directory
    
    Returns:
        Combined DataFrame
    """
    logger.info(f"📂 Loading 1h data from {base_path}")
    
    # Find all parquet files in year/month subdirectories (partitioned data)
    parquet_files = list(base_path.rglob("*.parquet"))
    
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found in {base_path}")
    
    logger.info(f"📊 Found {len(parquet_files)} parquet files")
    
    # Load and combine all files
    dfs = []
    for file_path in sorted(parquet_files):
        logger.info(f"  Loading {file_path.name}...")
        df = pd.read_parquet(file_path)
        dfs.append(df)
    
    # Combine all DataFrames
    combined_df = pd.concat(dfs, ignore_index=True)
    
    logger.info(f"✅ Loaded {len(combined_df):,} rows from {len(dfs)} files")
    
    # Ensure timestamp is datetime and set as index
    if 'timestamp' not in combined_df.columns:
        # Try to use close_time or open_time if timestamp is missing
        if 'close_time' in combined_df.columns:
            combined_df['timestamp'] = combined_df['close_time']
        elif 'open_time' in combined_df.columns:
            combined_df['timestamp'] = combined_df['open_time']
        else:
            raise ValueError("No timestamp column found in data")
    
    # Convert to datetime if not already
    if not pd.api.types.is_datetime64_any_dtype(combined_df['timestamp']):
        # Try milliseconds first, then seconds
        try:
            combined_df['timestamp'] = pd.to_datetime(combined_df['timestamp'], unit='ms', utc=True)
        except:
            combined_df['timestamp'] = pd.to_datetime(combined_df['timestamp'], unit='s', utc=True)
    
    # Set as index and sort
    combined_df = combined_df.set_index('timestamp')
    combined_df = combined_df.sort_index()
    
    return combined_df


def save_resampled_data(df: pd.DataFrame, output_path: Path, timeframe: str):
    """
    Save resampled data in partitioned format by year.
    
    Args:
        df: DataFrame to save (with timestamp index)
        output_path: Base output directory
        timeframe: Timeframe string for metadata
    """
    logger.info(f"💾 Saving resampled data to {output_path}")
    
    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Reset index to get timestamp as column
    df_save = df.reset_index()
    
    # Add year column for partitioning
    df_save['year'] = df_save['timestamp'].dt.year
    
    # Partition by year
    for year, year_group in df_save.groupby('year'):
        year_dir = output_path / f"year={year}"
        year_dir.mkdir(parents=True, exist_ok=True)
        
        # Save to parquet
        output_file = year_dir / f"ethusdt_{timeframe}_{year}.parquet"
        year_group.drop('year', axis=1).to_parquet(output_file, index=False)
        
        logger.info(f"  📁 Saved year {year}: {len(year_group)} rows → {output_file.name}")
    
    logger.info(f"✅ Saved {len(df)} rows to {output_path}")


def main():
    """Main execution function."""
    logger.info("=" * 80)
    logger.info("🔄 RESAMPLING 1H DATA TO 4H AND 1D TIMEFRAMES")
    logger.info("=" * 80)
    
    # Define paths
    base_dir = Path("/Users/remyroche/Documents/Ares/historical_data/binance/ethusdt/processed")
    input_path = base_dir / "ethusdt_1h"
    
    # Check if input directory exists
    if not input_path.exists():
        logger.error(f"❌ Input directory not found: {input_path}")
        return 1
    
    try:
        # Load 1h data
        logger.info("\n📊 STEP 1: Loading 1h data")
        logger.info("-" * 80)
        df_1h = load_1h_data(input_path)
        
        logger.info(f"\n📈 Data Summary:")
        logger.info(f"   Total rows: {len(df_1h):,}")
        logger.info(f"   Date range: {df_1h.index.min()} to {df_1h.index.max()}")
        logger.info(f"   Columns: {', '.join(df_1h.columns.tolist())}")
        
        # Resample to 4h
        logger.info("\n🔄 STEP 2: Resampling to 4h")
        logger.info("-" * 80)
        df_4h = resample_ohlcv(df_1h, '4h')
        output_4h = base_dir / "ethusdt_4h"
        save_resampled_data(df_4h, output_4h, '4h')
        
        # Resample to 1d
        logger.info("\n🔄 STEP 3: Resampling to 1d")
        logger.info("-" * 80)
        df_1d = resample_ohlcv(df_1h, '1d')
        output_1d = base_dir / "ethusdt_1d"
        save_resampled_data(df_1d, output_1d, '1d')
        
        # Summary
        logger.info("\n" + "=" * 80)
        logger.info("✅ RESAMPLING COMPLETE")
        logger.info("=" * 80)
        logger.info(f"   1h data:  {len(df_1h):,} rows")
        logger.info(f"   4h data:  {len(df_4h):,} rows (saved to {output_4h})")
        logger.info(f"   1d data:  {len(df_1d):,} rows (saved to {output_1d})")
        logger.info("=" * 80)
        
        return 0
        
    except Exception as e:
        logger.exception(f"❌ Error during resampling: {e}")
        return 1


if __name__ == "__main__":
    exit(main())

