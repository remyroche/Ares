#!/usr/bin/env python3
"""
Script to resample 1-minute OHLCV data to higher timeframes (5m, 15m, 30m)
for cross-timeframe analysis.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
import logging
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def convert_timestamp_to_datetime(df):
    """Convert timestamp column to datetime index."""
    if 'timestamp' in df.columns:
        # Check if timestamp is in milliseconds (common in crypto data)
        sample_timestamp = df['timestamp'].iloc[0]
        if sample_timestamp > 1e12:  # Milliseconds
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        elif sample_timestamp > 1e9:  # Seconds
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
        else:  # Already datetime or other format
            df['timestamp'] = pd.to_datetime(df['timestamp'])

    return df

def resample_ohlcv(df, timeframe):
    """
    Resample OHLCV data to specified timeframe.

    Args:
        df: DataFrame with OHLCV data
        timeframe: Target timeframe (e.g., '5T', '15T', '30T')

    Returns:
        Resampled DataFrame
    """
    # Define resampling rules for OHLCV
    resample_rules = {
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }

    # Resample the data
    resampled = df.resample(timeframe).agg(resample_rules)

    # Drop rows with NaN values (incomplete periods)
    resampled = resampled.dropna()

    return resampled

def generate_multi_timeframe_data(input_path, output_dir):
    """
    Generate multiple timeframes from 1m data.

    Args:
        input_path: Path to 1m data file
        output_dir: Directory to save resampled data
    """
    logger.info(f"🔄 Processing: {input_path}")

    # Load the 1m data
    try:
        df = pd.read_parquet(input_path)
        logger.info(f"📊 Loaded {len(df)} rows from {input_path.name}")
    except Exception as e:
        logger.error(f"❌ Failed to load {input_path}: {e}")
        return False

    # Convert timestamp to datetime and set as index
    df = convert_timestamp_to_datetime(df)
    df = df.set_index('timestamp')

    # Ensure we have required OHLCV columns
    required_cols = ['open', 'high', 'low', 'close', 'volume']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        logger.error(f"❌ Missing required columns: {missing_cols}")
        return False

    # Define timeframes to generate
    timeframes = {
        '5m': '5T',
        '15m': '15T',
        '30m': '30T'
    }

    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate each timeframe
    for tf_name, tf_freq in timeframes.items():
        try:
            logger.info(f"🔄 Resampling to {tf_name} timeframe...")

            # Resample the data
            resampled_df = resample_ohlcv(df[required_cols], tf_freq)

            # Add metadata columns
            resampled_df['symbol'] = df['symbol'].iloc[0] if 'symbol' in df.columns else 'ETHUSDT'
            resampled_df['exchange'] = df['exchange'].iloc[0] if 'exchange' in df.columns else 'binance'
            resampled_df['timeframe'] = tf_name
            resampled_df['batch_number'] = df['batch_number'].iloc[0] if 'batch_number' in df.columns else 1
            resampled_df['session_id'] = df['session_id'].iloc[0] if 'session_id' in df.columns else f"resampled_{tf_name}"
            resampled_df['download_timestamp'] = datetime.now().timestamp() * 1000

            # Reset index to have timestamp as column
            resampled_df = resampled_df.reset_index()

            # Generate output filename
            base_name = input_path.stem.replace('_1m_', f'_{tf_name}_')
            output_path = output_dir / f"{base_name}.parquet"

            # Save the resampled data
            resampled_df.to_parquet(output_path, index=False)

            # Save metadata
            metadata_path = output_dir / f"{base_name}.metadata.json"
            metadata = {
                'original_file': str(input_path),
                'resampled_timeframe': tf_name,
                'original_rows': len(df),
                'resampled_rows': len(resampled_df),
                'date_range': {
                    'start': str(resampled_df['timestamp'].min()),
                    'end': str(resampled_df['timestamp'].max())
                },
                'resampling_timestamp': datetime.now().isoformat(),
                'compression_ratio': len(resampled_df) / len(df)
            }

            import json
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2, default=str)

            logger.info(f"✅ Generated {len(resampled_df)} {tf_name} bars (from {len(df)} 1m bars)")
            logger.info(f"💾 Saved to: {output_path}")

        except Exception as e:
            logger.error(f"❌ Failed to resample {tf_name}: {e}")
            continue

    return True

def main():
    """Main function to resample all 1m data files."""
    # Find all 1m data files
    data_dir = Path('historical_data/binance/ethusdt/klines')
    pattern = 'klines_binance_ETHUSDT_1m_*.parquet'

    logger.info(f"🔍 Looking for 1m data files in: {data_dir}")
    logger.info(f"📋 Pattern: {pattern}")

    # Find all matching files
    import glob
    search_pattern = data_dir / pattern
    files = glob.glob(str(search_pattern))

    if not files:
        logger.error(f"❌ No 1m data files found matching pattern: {search_pattern}")
        return False

    logger.info(f"📁 Found {len(files)} 1m data files to process:")
    for f in files:
        logger.info(f"  - {Path(f).name}")

    # Process each file
    success_count = 0
    for file_path in files:
        file_path = Path(file_path)
        if generate_multi_timeframe_data(file_path, data_dir):
            success_count += 1
        else:
            logger.warning(f"⚠️ Failed to process: {file_path}")

    logger.info(f"🎯 Processing complete: {success_count}/{len(files)} files processed successfully")

    if success_count > 0:
        logger.info("✅ Multi-timeframe data generation complete!")
        logger.info("🔄 You can now run cross-timeframe analysis with full multi-timeframe support")
    else:
        logger.error("❌ No files were processed successfully")

    return success_count > 0

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
