#!/usr/bin/env python3
"""
Fix Aggtrades Consolidation Issue

This script consolidates individual daily aggtrades files into the expected
consolidated format that the system requires.
"""

import os
import pandas as pd
import glob
from pathlib import Path
import logging
from typing import List, Optional
import numpy as np
import psutil
import gc

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def log_memory_usage(stage: str):
    """Log current memory usage."""
    try:
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        memory_mb = memory_info.rss / 1024 / 1024
        logger.info(f"Memory usage at {stage}: {memory_mb:.1f} MB")
    except Exception:
        pass  # Ignore memory monitoring errors

def find_aggtrades_files(data_cache_dir: str, exchange: str, symbol: str) -> List[str]:
    """Find all aggtrades parquet files for the given exchange and symbol."""
    pattern = os.path.join(data_cache_dir, f"aggtrades_{exchange}_{symbol}_*.parquet")
    files = glob.glob(pattern)
    logger.info(f"Found {len(files)} aggtrades files matching pattern: {pattern}")
    return sorted(files)

def load_and_validate_aggtrades_file(file_path: str) -> Optional[pd.DataFrame]:
    """Load and validate a single aggtrades file."""
    try:
        df = pd.read_parquet(file_path)
        
        # Check required columns
        required_columns = ['timestamp', 'price', 'quantity']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            logger.warning(f"Missing required columns in {file_path}: {missing_columns}")
            return None
            
        # Check data types and convert if needed
        if df['timestamp'].dtype != 'int64':
            try:
                df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
                df['timestamp'] = (df['timestamp'].astype('int64') // 10**6).astype('int64')
            except Exception as e:
                logger.warning(f"Failed to convert timestamp in {file_path}: {e}")
                return None
                
        # Ensure numeric types
        df['price'] = pd.to_numeric(df['price'], errors='coerce')
        df['quantity'] = pd.to_numeric(df['quantity'], errors='coerce')
        
        # Remove rows with invalid data
        df = df.dropna(subset=['timestamp', 'price', 'quantity'])
        
        if len(df) == 0:
            logger.warning(f"No valid data in {file_path}")
            return None
            
        logger.info(f"Loaded {len(df)} valid rows from {os.path.basename(file_path)}")
        return df
        
    except Exception as e:
        logger.error(f"Failed to load {file_path}: {e}")
        return None

def consolidate_aggtrades_files(data_cache_dir: str, exchange: str, symbol: str, batch_size: int = 10) -> bool:
    """Consolidate all aggtrades files into a single consolidated file in batches to save memory."""
    try:
        # Find all aggtrades files
        aggtrades_files = find_aggtrades_files(data_cache_dir, exchange, symbol)
        
        if not aggtrades_files:
            logger.error(f"No aggtrades files found for {exchange}_{symbol}")
            return False
            
        logger.info(f"Processing {len(aggtrades_files)} files in batches of {batch_size}")
        
        # Process files in batches
        consolidated_df = None
        total_rows = 0
        batch_count = 0
        
        for i in range(0, len(aggtrades_files), batch_size):
            batch_count += 1
            batch_files = aggtrades_files[i:i + batch_size]
            logger.info(f"Processing batch {batch_count}: files {i+1}-{min(i+batch_size, len(aggtrades_files))}")
            
            # Load batch files
            batch_dataframes = []
            batch_rows = 0
            
            for file_path in batch_files:
                df = load_and_validate_aggtrades_file(file_path)
                if df is not None and len(df) > 0:
                    batch_dataframes.append(df)
                    batch_rows += len(df)
                    
            if not batch_dataframes:
                logger.warning(f"No valid data in batch {batch_count}")
                continue
                
            # Combine batch dataframes
            batch_combined = pd.concat(batch_dataframes, ignore_index=True)
            logger.info(f"Batch {batch_count}: {len(batch_combined):,} rows")
            
            # Remove duplicates within batch
            initial_batch_count = len(batch_combined)
            batch_combined = batch_combined.drop_duplicates(subset=['timestamp', 'price', 'quantity'], keep='first')
            if initial_batch_count != len(batch_combined):
                logger.info(f"Removed {initial_batch_count - len(batch_combined)} duplicates in batch {batch_count}")
            
            # Add to consolidated dataframe
            if consolidated_df is None:
                consolidated_df = batch_combined.copy()
            else:
                consolidated_df = pd.concat([consolidated_df, batch_combined], ignore_index=True)
                
            total_rows += len(batch_combined)
            
            # Clear batch dataframes to free memory
            del batch_dataframes, batch_combined
            gc.collect()  # Force garbage collection
            
            logger.info(f"Batch {batch_count} completed. Total rows so far: {total_rows:,}")
            log_memory_usage(f"after batch {batch_count}")
                
        if consolidated_df is None or len(consolidated_df) == 0:
            logger.error("No valid aggtrades data found")
            return False
        
        # Final deduplication across all batches
        logger.info("Performing final deduplication...")
        initial_count = len(consolidated_df)
        consolidated_df = consolidated_df.drop_duplicates(subset=['timestamp', 'price', 'quantity'], keep='first')
        final_count = len(consolidated_df)
        
        if initial_count != final_count:
            logger.info(f"Removed {initial_count - final_count} duplicate rows in final deduplication")
            
        # Sort by timestamp
        logger.info("Sorting by timestamp...")
        consolidated_df = consolidated_df.sort_values('timestamp').reset_index(drop=True)
        
        # Ensure proper data types
        consolidated_df['timestamp'] = consolidated_df['timestamp'].astype('int64')
        consolidated_df['price'] = consolidated_df['price'].astype('float64')
        consolidated_df['quantity'] = consolidated_df['quantity'].astype('float64')
        
        # Add is_buyer_maker column if missing (set to False as default)
        if 'is_buyer_maker' not in consolidated_df.columns:
            consolidated_df['is_buyer_maker'] = False
            logger.info("Added missing is_buyer_maker column (defaulting to False)")
            
        # Add agg_trade_id if missing
        if 'agg_trade_id' not in consolidated_df.columns:
            consolidated_df['agg_trade_id'] = range(len(consolidated_df))
            logger.info("Added missing agg_trade_id column")
            
        # Save consolidated file
        output_file = os.path.join(data_cache_dir, f"aggtrades_{exchange}_{symbol}_consolidated.parquet")
        consolidated_df.to_parquet(output_file, index=False)
        
        logger.info(f"✅ Successfully created consolidated aggtrades file: {output_file}")
        logger.info(f"   - Total rows: {len(consolidated_df):,}")
        logger.info(f"   - Date range: {pd.to_datetime(consolidated_df['timestamp'].min(), unit='ms')} to {pd.to_datetime(consolidated_df['timestamp'].max(), unit='ms')}")
        logger.info(f"   - Columns: {list(consolidated_df.columns)}")
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to consolidate aggtrades files: {e}")
        return False

def main():
    """Main function to fix aggtrades consolidation."""
    data_cache_dir = "/Users/remyroche/Documents/Ares/data_cache"
    exchange = "BINANCE"
    symbol = "ETHUSDT"
    batch_size = 10  # Process 10 files at a time to save memory
    
    logger.info("🔧 Starting aggtrades consolidation fix...")
    logger.info(f"Data directory: {data_cache_dir}")
    logger.info(f"Exchange: {exchange}")
    logger.info(f"Symbol: {symbol}")
    logger.info(f"Batch size: {batch_size}")
    
    log_memory_usage("start")
    
    # Check if data cache directory exists
    if not os.path.exists(data_cache_dir):
        logger.error(f"Data cache directory does not exist: {data_cache_dir}")
        return False
        
    # Consolidate aggtrades files
    success = consolidate_aggtrades_files(data_cache_dir, exchange, symbol, batch_size)
    
    if success:
        logger.info("✅ Aggtrades consolidation completed successfully!")
        
        # Verify the consolidated file
        consolidated_file = os.path.join(data_cache_dir, f"aggtrades_{exchange}_{symbol}_consolidated.parquet")
        if os.path.exists(consolidated_file):
            file_size = os.path.getsize(consolidated_file) / (1024 * 1024)  # MB
            logger.info(f"📁 Consolidated file size: {file_size:.2f} MB")
            
            # Quick validation
            try:
                df = pd.read_parquet(consolidated_file)
                logger.info(f"📊 Validation: {len(df):,} rows, {len(df.columns)} columns")
                logger.info(f"📅 Date range: {pd.to_datetime(df['timestamp'].min(), unit='ms')} to {pd.to_datetime(df['timestamp'].max(), unit='ms')}")
            except Exception as e:
                logger.warning(f"Validation failed: {e}")
    else:
        logger.error("❌ Aggtrades consolidation failed!")
        
    return success

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
