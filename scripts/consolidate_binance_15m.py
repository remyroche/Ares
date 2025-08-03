#!/usr/bin/env python3
"""
Script to consolidate all Binance 15m klines data into a single file.
This will create a proper consolidated file for regime training.
"""

import os
import glob
import pandas as pd
from pathlib import Path
import sys

# Add the project root to the path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.utils.logger import setup_logging, system_logger

def consolidate_binance_15m_data():
    """Consolidate all Binance 15m klines data files."""
    logger = system_logger.getChild("ConsolidateBinance15m")
    
    print("🔄 Consolidating Binance 15m data...")
    logger.info("🔄 Starting Binance 15m data consolidation...")
    
    # Find all 15m Binance files
    pattern = "data_cache/klines_BINANCE_ETHUSDT_15m_*.csv"
    source_files = sorted(glob.glob(pattern))
    
    logger.info(f"📁 Found {len(source_files)} 15m Binance files")
    logger.info(f"📋 Source files:")
    for i, file in enumerate(source_files[:5], 1):
        file_size = os.path.getsize(file)
        logger.info(f"   {i}. {os.path.basename(file)} ({file_size:,} bytes)")
    if len(source_files) > 5:
        logger.info(f"   ... and {len(source_files) - 5} more files")
    
    if not source_files:
        logger.error("❌ No 15m Binance files found")
        return False
    
    # Output file
    output_file = "data_cache/klines_BINANCE_ETHUSDT_15m_consolidated.csv"
    logger.info(f"💾 Output file: {output_file}")
    
    # Consolidate all files
    all_data = []
    total_records = 0
    
    for i, file in enumerate(source_files, 1):
        logger.info(f"📖 [{i}/{len(source_files)}] Processing {os.path.basename(file)}...")
        
        try:
            # Read the CSV file
            df = pd.read_csv(file)
            logger.info(f"   📊 Loaded {len(df)} records")
            
            # Validate data
            if len(df) == 0:
                logger.warning(f"   ⚠️ Empty file: {os.path.basename(file)}")
                continue
            
            # Check columns
            expected_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
            if not all(col in df.columns for col in expected_columns):
                logger.warning(f"   ⚠️ Missing columns in {os.path.basename(file)}")
                logger.warning(f"   📋 Expected: {expected_columns}")
                logger.warning(f"   📋 Found: {list(df.columns)}")
                continue
            
            # Convert timestamp
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
            
            # Convert numeric columns
            numeric_cols = ['open', 'high', 'low', 'close', 'volume']
            df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')
            
            # Check for reasonable price data
            if df['close'].isna().all():
                logger.warning(f"   ⚠️ Invalid price data in {os.path.basename(file)}")
                continue
            
            # Check price range (ETH should be reasonable)
            min_price = df['low'].min()
            max_price = df['high'].max()
            if min_price < 100 or max_price > 10000:
                logger.warning(f"   ⚠️ Unreasonable price range in {os.path.basename(file)}: ${min_price:.2f} - ${max_price:.2f}")
                continue
            
            logger.info(f"   ✅ Valid data: {len(df)} records, price range: ${min_price:.2f} - ${max_price:.2f}")
            logger.info(f"   📅 Date range: {df.index.min()} to {df.index.max()}")
            
            all_data.append(df)
            total_records += len(df)
            
        except Exception as e:
            logger.error(f"   ❌ Error processing {os.path.basename(file)}: {e}")
            continue
    
    if not all_data:
        logger.error("❌ No valid data files found")
        return False
    
    logger.info(f"📊 Consolidating {len(all_data)} dataframes...")
    
    # Combine all dataframes
    consolidated_df = pd.concat(all_data, ignore_index=False)
    logger.info(f"📈 Combined dataframe shape: {consolidated_df.shape}")
    
    # Remove duplicates
    initial_count = len(consolidated_df)
    consolidated_df = consolidated_df[~consolidated_df.index.duplicated(keep='first')]
    final_count = len(consolidated_df)
    duplicates_removed = initial_count - final_count
    
    logger.info(f"🧹 Removed {duplicates_removed} duplicate records")
    logger.info(f"📊 Final dataframe shape: {consolidated_df.shape}")
    
    # Sort by timestamp
    consolidated_df.sort_index(inplace=True)
    logger.info(f"📅 Final date range: {consolidated_df.index.min()} to {consolidated_df.index.max()}")
    logger.info(f"💰 Final price range: ${consolidated_df['low'].min():.2f} to ${consolidated_df['high'].max():.2f}")
    
    # Save consolidated file
    logger.info(f"💾 Saving consolidated data to {output_file}...")
    consolidated_df.to_csv(output_file)
    
    file_size = os.path.getsize(output_file)
    logger.info(f"✅ Consolidated file saved: {file_size:,} bytes")
    logger.info(f"📊 Total records: {len(consolidated_df)}")
    
    # Verify the saved file
    logger.info(f"🔍 Verifying saved file...")
    verification_df = pd.read_csv(output_file)
    verification_df['timestamp'] = pd.to_datetime(verification_df['timestamp'])
    verification_df.set_index('timestamp', inplace=True)
    
    logger.info(f"✅ Verification successful:")
    logger.info(f"   - Records: {len(verification_df)}")
    logger.info(f"   - Date range: {verification_df.index.min()} to {verification_df.index.max()}")
    logger.info(f"   - Price range: ${verification_df['low'].min():.2f} to ${verification_df['high'].max():.2f}")
    
    print(f"✅ Successfully consolidated {len(consolidated_df)} records")
    print(f"📁 Output file: {output_file}")
    print(f"📅 Date range: {consolidated_df.index.min()} to {consolidated_df.index.max()}")
    print(f"💰 Price range: ${consolidated_df['low'].min():.2f} to ${consolidated_df['high'].max():.2f}")
    
    return True

if __name__ == "__main__":
    success = consolidate_binance_15m_data()
    if not success:
        sys.exit(1) 