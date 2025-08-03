import asyncio
import glob
import os
import pickle
import sys
import time
import traceback
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

# Add the project root to the Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.config import CONFIG

# Import the consolidated data downloader
from src.training.steps.data_downloader import download_all_data_with_consolidation
from src.utils.error_handler import (
    handle_errors,
)
from src.utils.logger import setup_logging, system_logger


def consolidate_files(
    pattern: str,
    consolidated_filepath: str,
    index_col: str,
    sort_col: str | None = None,
    dtype: dict | None = None,
    expected_columns: list[str] | None = None,
    lookback_days: int | None = None,
) -> pd.DataFrame:
    """
    Incrementally consolidates multiple source CSVs into a single file with data validation.
    This function is optimized to only process and append new data, making it highly efficient for resuming.
    1. If a consolidated file exists, it merges new data from source files.
    2. If no consolidated file exists, it creates one from all source files.
    3. Validates data integrity to prevent corruption.
    4. If lookback_days is specified, only processes files within that period.
    Returns the full, consolidated DataFrame.
    """
    logger = system_logger.getChild("ConsolidateFiles")
    print(f"🔄 Starting consolidation for pattern: {pattern}")
    print(f"📁 Looking for files in data_cache directory...")
    logger.info(f"🔄 Starting consolidation for pattern: {pattern}")
    logger.info(f"📁 Looking for files in data_cache directory...")
    
    # Get all source files
    source_files = sorted(glob.glob(os.path.join("data_cache", pattern)))
    print(f"📋 Found {len(source_files)} files matching pattern")
    logger.info(f"📋 Found {len(source_files)} files matching pattern")
    
    # Filter files based on lookback_days if specified
    if lookback_days is not None:
        from datetime import datetime, timedelta
        cutoff_date = datetime.now() - timedelta(days=lookback_days)
        print(f"📅 Filtering files to last {lookback_days} days (since {cutoff_date.strftime('%Y-%m-%d')})")
        logger.info(f"📅 Filtering files to last {lookback_days} days (since {cutoff_date.strftime('%Y-%m-%d')})")
        
        filtered_files = []
        for file in source_files:
            # Try to extract date from filename
            filename = os.path.basename(file)
            try:
                # Look for date patterns in filename (YYYY-MM-DD or YYYYMMDD)
                import re
                date_match = re.search(r'(\d{4})-(\d{2})-(\d{2})', filename)
                if date_match:
                    file_date = datetime.strptime(date_match.group(0), '%Y-%m-%d')
                else:
                    # Try YYYYMMDD format
                    date_match = re.search(r'(\d{4})(\d{2})(\d{2})', filename)
                    if date_match:
                        file_date = datetime.strptime(date_match.group(0), '%Y%m%d')
                    else:
                        # If no date found, include the file (conservative approach)
                        file_date = datetime.now()
                
                if file_date >= cutoff_date:
                    filtered_files.append(file)
                else:
                    print(f"   ⏭️ Skipping {filename} (date: {file_date.strftime('%Y-%m-%d')})")
                    logger.info(f"   ⏭️ Skipping {filename} (date: {file_date.strftime('%Y-%m-%d')})")
            except Exception as e:
                # If date parsing fails, include the file (conservative approach)
                print(f"   ⚠️ Could not parse date from {filename}, including file")
                logger.warning(f"   ⚠️ Could not parse date from {filename}, including file: {e}")
                filtered_files.append(file)
        
        source_files = filtered_files
        print(f"📋 After filtering: {len(source_files)} files within {lookback_days} days")
        logger.info(f"📋 After filtering: {len(source_files)} files within {lookback_days} days")
    
    print(f"📁 Source files:")
    logger.info(f"📁 Source files:")
    for i, file in enumerate(source_files[:5], 1):  # Show first 5 files
        print(f"   {i}. {os.path.basename(file)} ({os.path.getsize(file):,} bytes)")
        logger.info(f"   {i}. {os.path.basename(file)} ({os.path.getsize(file):,} bytes)")
    if len(source_files) > 5:
        print(f"   ... and {len(source_files) - 5} more files")
        logger.info(f"   ... and {len(source_files) - 5} more files")
    
    unique_col = sort_col if sort_col else index_col
    logger.info(f"🔑 Using unique column: {unique_col}")
    logger.info(f"📊 Expected columns: {expected_columns}")

    if not source_files:
        logger.warning(f"No source files found for pattern: {pattern}")
        print(f"[ConsolidateFiles] No source files found for pattern: {pattern}")
        if os.path.exists(consolidated_filepath):
            logger.info(
                f"Returning data from existing consolidated file: {consolidated_filepath}",
            )
            print(
                f"[ConsolidateFiles] Returning data from existing consolidated file: {consolidated_filepath}",
            )
            return pd.read_csv(consolidated_filepath)
        return pd.DataFrame()

    # Validate source files before processing
    logger.info(f"🔍 Validating {len(source_files)} source files...")
    logger.info(f"📋 Expected columns: {expected_columns}")
    valid_files = []
    for i, file in enumerate(source_files, 1):
        logger.info(f"🔍 [{i}/{len(source_files)}] Validating file: {os.path.basename(file)}")
        try:
            # Read a small sample to validate the file
            logger.info(f"   📖 Reading sample from {os.path.basename(file)}...")
            sample_df = pd.read_csv(file, nrows=5)
            logger.info(f"   📊 Sample shape: {sample_df.shape}")
            logger.info(f"   📋 Sample columns: {list(sample_df.columns)}")
            logger.info(f"   📄 Sample data types: {sample_df.dtypes.to_dict()}")
            
            # Check if file has expected columns
            if expected_columns:
                missing_cols = set(expected_columns) - set(sample_df.columns)
                if missing_cols:
                    logger.warning(f"   ❌ File {os.path.basename(file)} missing columns: {missing_cols}")
                    continue
                else:
                    logger.info(f"   ✅ All expected columns present")
            
            # Check for reasonable price data (for klines files)
            if "close" in sample_df.columns or "Close" in sample_df.columns:
                close_col = "close" if "close" in sample_df.columns else "Close"
                logger.info(f"   💰 Checking price data in column: {close_col}")
                prices = pd.to_numeric(sample_df[close_col], errors='coerce')
                logger.info(f"   📊 Price sample: {prices.tolist()}")
                
                if prices.isna().all():
                    logger.warning(f"   ❌ File {os.path.basename(file)} has invalid price data (all NaN)")
                    continue
                
                # Check for reasonable price range (for ETH, should be between $100 and $50,000)
                valid_prices = prices.dropna()
                if len(valid_prices) > 0:
                    min_price = valid_prices.min()
                    max_price = valid_prices.max()
                    logger.info(f"   📈 Price range: ${min_price:.2f} to ${max_price:.2f}")
                    if min_price < 100 or max_price > 50000:
                        logger.warning(f"   ⚠️ File {os.path.basename(file)} has unreasonable prices: ${min_price:.2f} to ${max_price:.2f}")
                        continue
                    else:
                        logger.info(f"   ✅ Price range is reasonable")
                else:
                    logger.warning(f"   ❌ No valid prices found in {os.path.basename(file)}")
                    continue
            
            valid_files.append(file)
            logger.info(f"   ✅ Validated file: {os.path.basename(file)}")
            
        except Exception as e:
            logger.warning(f"   ❌ Invalid file {os.path.basename(file)}: {e}")
            logger.warning(f"   🔍 Exception type: {type(e).__name__}")
            continue
    
    if not valid_files:
        logger.error("No valid source files found!")
        return pd.DataFrame()
    
    logger.info(f"Found {len(valid_files)} valid files out of {len(source_files)} total files")

    # Load existing data to determine what's new
    existing_df = pd.DataFrame()
    existing_ids = set()
    if os.path.exists(consolidated_filepath):
        try:
            logger.info(f"Loading existing consolidated file: {consolidated_filepath}")
            existing_df = pd.read_csv(
                consolidated_filepath,
                dtype=dtype,
                low_memory=False,
            )

            if not existing_df.empty and index_col in existing_df.columns:
                # Robustly convert timestamp column, handling mixed (string/unix) formats
                # from previous runs, which can cause errors.
                converted_datetimes = pd.to_datetime(
                    existing_df[index_col],
                    errors="coerce",
                )
                failed_indices = (
                    converted_datetimes.isna() & existing_df[index_col].notna()
                )

                if failed_indices.any():
                    logger.warning(
                        f"Found {failed_indices.sum()} non-standard timestamps in consolidated file. Attempting numeric conversion.",
                    )
                    numeric_part = pd.to_datetime(
                        existing_df.loc[failed_indices, index_col],
                        unit="ms",
                        errors="coerce",
                    )
                    converted_datetimes.loc[failed_indices] = numeric_part

                existing_df[index_col] = converted_datetimes
                existing_df.dropna(subset=[index_col], inplace=True)

            if not existing_df.empty and unique_col in existing_df.columns:
                # Get the set of unique IDs that we already have.
                existing_ids = set(existing_df[unique_col].unique())
                logger.info(f"Found {len(existing_ids)} existing unique records.")
        except Exception as e:
            logger.warning(
                f"Could not read existing file {consolidated_filepath}: {e}. Rebuilding from scratch.",
            )
            existing_df, existing_ids = pd.DataFrame(), set()

    # Process source files in chunks, adding only new data.
    new_data_chunks = []
    chunk_size = 200  # Process 200 files at a time
    logger.info(f"🔄 Processing {len(valid_files)} valid files in chunks of {chunk_size}")
    logger.info(f"📊 Total chunks: {(len(valid_files) + chunk_size - 1)//chunk_size}")
    
    for i in range(0, len(valid_files), chunk_size):
        chunk_of_files = valid_files[i : i + chunk_size]
        chunk_num = i//chunk_size + 1
        total_chunks = (len(valid_files) + chunk_size - 1)//chunk_size
        logger.info(
            f"📦 Processing chunk {chunk_num}/{total_chunks} for pattern '{pattern}'",
        )
        logger.info(f"   📁 Files in this chunk: {len(chunk_of_files)}")
        logger.info(f"   📋 Sample files: {[os.path.basename(f) for f in chunk_of_files[:3]]}")

        valid_df_list = []
        logger.info(f"   📖 Reading {len(chunk_of_files)} files in this chunk...")
        
        for j, f in enumerate(chunk_of_files, 1):
            logger.info(f"   📄 [{j}/{len(chunk_of_files)}] Processing: {os.path.basename(f)}")
            
            if os.path.getsize(f) > 0:
                logger.info(f"      📊 File size: {os.path.getsize(f):,} bytes")
                try:
                    logger.info(f"      📖 Reading CSV file...")
                    df = pd.read_csv(f, dtype=dtype, on_bad_lines="warn")
                    logger.info(f"      ✅ CSV read successfully")
                    logger.info(f"      📊 DataFrame shape: {df.shape}")
                    logger.info(f"      📋 DataFrame columns: {list(df.columns)}")
                    logger.info(f"      📄 Data types: {df.dtypes.to_dict()}")

                    # Ensure timestamp column is in datetime format for processing
                    if (
                        index_col in df.columns
                        and not pd.api.types.is_datetime64_any_dtype(df[index_col])
                    ):
                        logger.info(f"      🕐 Converting timestamp column '{index_col}'...")
                        df[index_col] = pd.to_datetime(df[index_col], errors="coerce")
                        logger.info(f"      📊 Timestamp conversion completed")
                        
                        # Check for invalid timestamps and remove them
                        invalid_timestamps = df[index_col].isna().sum()
                        if invalid_timestamps > 0:
                            logger.warning(f"      ⚠️ Found {invalid_timestamps} rows with invalid timestamps - removing them")
                            df = df.dropna(subset=[index_col])
                            logger.info(f"      ✅ Removed {invalid_timestamps} rows with invalid timestamps")
                        else:
                            logger.info(f"      ✅ All timestamps are valid")
                        
                        # Remove duplicates
                        duplicates_before = df[index_col].duplicated().sum()
                        if duplicates_before > 0:
                            logger.info(f"      🧹 Removing {duplicates_before} duplicate timestamps...")
                            df.drop_duplicates(subset=[index_col], keep="last", inplace=True)
                            logger.info(f"      ✅ Duplicate removal completed")
                        else:
                            logger.info(f"      ✅ No duplicate timestamps found")
                        
                        logger.info(f"      ✅ Timestamp processing completed")

                    # Final check for the unique column
                    if unique_col in df.columns and not df.empty:
                        logger.info(f"      ✅ File is valid and ready for processing")
                        valid_df_list.append(df)
                    else:
                        logger.warning(
                            f"      ❌ Skipping file {os.path.basename(f)}: missing '{unique_col}' or empty.",
                        )
                        if unique_col not in df.columns:
                            logger.warning(f"         Missing column: {unique_col}")
                        if df.empty:
                            logger.warning(f"         DataFrame is empty")
                except Exception as e:
                    logger.warning(
                        f"      ❌ Could not read or process file {os.path.basename(f)}: {e}. Skipping.",
                    )
                    logger.warning(f"         Exception type: {type(e).__name__}")
            else:
                logger.warning(f"      ⚠️ File is empty: {os.path.basename(f)}")

        if not valid_df_list:
            logger.warning(
                f"   ❌ No valid dataframes found in chunk starting with {os.path.basename(chunk_of_files[0])}.",
            )
            continue

        logger.info(f"   🔄 Concatenating {len(valid_df_list)} valid dataframes...")
        chunk_df = pd.concat(valid_df_list, ignore_index=True)
        logger.info(f"   📊 Combined chunk shape: {chunk_df.shape}")
        
        logger.info(f"   🧹 Removing duplicates based on '{unique_col}'...")
        duplicates_before = chunk_df[unique_col].duplicated().sum()
        chunk_df.drop_duplicates(subset=[unique_col], keep="last", inplace=True)
        duplicates_after = chunk_df[unique_col].duplicated().sum()
        logger.info(f"   ✅ Removed {duplicates_before - duplicates_after} duplicates")

        # Filter out records we already have in the consolidated file
        logger.info(f"   🔍 Filtering out existing records...")
        new_records_mask = ~chunk_df[unique_col].isin(existing_ids)
        new_data = chunk_df[new_records_mask]
        logger.info(f"   📊 New records found: {len(new_data)} out of {len(chunk_df)}")

        if not new_data.empty:
            logger.info(f"   ✅ Adding {len(new_data)} new records to processing queue")
            new_data_chunks.append(new_data)
            existing_ids.update(new_data[unique_col])
        else:
            logger.info(f"   ℹ️ No new records in this chunk")

    if new_data_chunks:
        logger.info(f"🔄 Concatenating {len(new_data_chunks)} data chunks...")
        new_data_df = pd.concat(new_data_chunks, ignore_index=True)
        logger.info(
            f"📊 Found {len(new_data_df)} new records to add to the consolidated file.",
        )
        logger.info(f"📋 New data columns: {list(new_data_df.columns)}")
        logger.info(f"📊 New data shape: {new_data_df.shape}")
        
        logger.info(f"🔄 Combining with existing data...")
        combined_df = pd.concat([existing_df, new_data_df], ignore_index=True)
        logger.info(f"📊 Combined shape: {combined_df.shape}")
    else:
        logger.info("ℹ️ No new records found in source files.")
        combined_df = existing_df

    if combined_df.empty:
        logger.warning(f"❌ No data found for pattern '{pattern}'.")
        return pd.DataFrame()

    # Final sort, safeguard deduplication, and save
    logger.info(f"🔄 Final processing steps...")
    logger.info(f"   📊 Combined DataFrame shape before final processing: {combined_df.shape}")
    
    # Timestamps are converted incrementally, so the combined column should be datetime.
    logger.info(f"   📅 Sorting by '{index_col}'...")
    combined_df.sort_values(by=index_col, inplace=True)
    logger.info(f"   ✅ Sorting completed")
    
    logger.info(f"   🧹 Final duplicate removal...")
    duplicates_before = combined_df[unique_col].duplicated().sum()
    combined_df.drop_duplicates(subset=[unique_col], keep="last", inplace=True)
    duplicates_after = combined_df[unique_col].duplicated().sum()
    logger.info(f"   ✅ Removed {duplicates_before - duplicates_after} final duplicates")
    
    logger.info(f"   💾 Saving to {consolidated_filepath}...")
    combined_df.to_csv(consolidated_filepath, index=False)
    file_size = os.path.getsize(consolidated_filepath)
    logger.info(
        f"✅ Saved consolidated file with {len(combined_df)} rows to {consolidated_filepath}",
    )
    logger.info(f"📊 File size: {file_size:,} bytes")
    logger.info(f"📈 Data range: {combined_df[index_col].min()} to {combined_df[index_col].max()}")
    logger.info(f"📋 Final columns: {list(combined_df.columns)}")
    return combined_df


@handle_errors(
    exceptions=(Exception,),
    default_return=(None, None, None),
    context="data_collection_step",
)
async def run_step(
    symbol: str,
    exchange_name: str,
    min_data_points: str,
    data_dir: str,
    download_new_data: bool = True,
    lookback_days: int | None = None,
) -> tuple[pd.DataFrame | None, pd.DataFrame | None, pd.DataFrame | None]:
    """
    Orchestrates the data collection process by calling the robust, incremental downloader.
    Saves the collected DataFrames to a pickle file.
    """
    import sys  # Add sys import here to fix the variable error

    start_time = time.time()
    setup_logging()
    logger = system_logger.getChild("Step1DataCollection")

    logger.info("=" * 80)
    logger.info("🚀 STEP 1: DATA COLLECTION START")
    logger.info("=" * 80)
    logger.info(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Symbol: {symbol}")
    logger.info(f"Exchange: {exchange_name}")
    logger.info(f"Min data points: {min_data_points}")
    logger.info(f"Data directory: {data_dir}")
    logger.info(f"Lookback days: {lookback_days}")
    logger.info(f"Python version: {sys.version}")
    logger.info(f"Working directory: {Path.cwd()}")
    print("=" * 80)  # Explicit print for subprocess output
    print("🚀 STEP 1: DATA COLLECTION START")  # Explicit print for subprocess output
    print("=" * 80)  # Explicit print for subprocess output

    try:
        min_points = int(min_data_points)

        # Step 1.1: Download data (skip for blank training mode)
        logger.info("📥 STEP 1.1: Data download...")
        download_start = time.time()

        # Check if this is a blank training run
        blank_training_mode = os.environ.get("BLANK_TRAINING_MODE", "0") == "1"

        if blank_training_mode:
            logger.info(
                "🔧 BLANK TRAINING MODE: Skipping data download, using existing data",
            )
            logger.info("   Using existing data files for blank training run")
            download_success = True  # Skip download for blank mode
        elif download_new_data:
            # For normal training runs, download data
            logger.info("   Starting data download...")
            print("   Starting data download...")

            # For blank training runs, limit the data download to recent data only
            if lookback_days and lookback_days <= 60:
                logger.info(
                    "   Blank training run: Limiting data download to recent data only",
                )
                print(
                    "   Blank training run: Limiting data download to recent data only",
                )
                # Temporarily override the config lookback for blank runs
                original_lookback = CONFIG["MODEL_TRAINING"]["data_retention_days"]
                CONFIG["MODEL_TRAINING"]["data_retention_days"] = min(
                    lookback_days,
                    30,
                )  # Max 30 days for blank runs
                logger.info(
                    f"   Temporarily set data_retention_days to {CONFIG['MODEL_TRAINING']['data_retention_days']} for blank run",
                )
                print(
                    f"   Temporarily set data_retention_days to {CONFIG['MODEL_TRAINING']['data_retention_days']} for blank run",
                )

            download_success = await download_all_data_with_consolidation(
                symbol=symbol,
                exchange_name=exchange_name,
                interval="1m",  # Assuming 1m for highest granularity
            )

            # Restore original config if we modified it
            if lookback_days and lookback_days <= 60:
                CONFIG["MODEL_TRAINING"]["data_retention_days"] = original_lookback
                logger.info(f"   Restored data_retention_days to {original_lookback}")
                print(f"   Restored data_retention_days to {original_lookback}")

            logger.info(
                f"   download_all_data_with_consolidation completed: {download_success}",
            )
            print(
                f"   download_all_data_with_consolidation completed: {download_success}",
            )

            if not download_success:
                raise RuntimeError(
                    "Data download step failed. Check downloader logs for details.",
                )
        else:
            logger.info("✅ STEP 1.1: Skipping data download as requested for resume.")
            print("✅ STEP 1.1: Skipping data download as requested for resume.")
            download_success = True

        # --- Step 2: Consolidate downloaded files ---
        logger.info("✅ STEP 1.2: Consolidating downloaded data files...")
        print("✅ STEP 1.2: Consolidating downloaded data files...")

        # Use the new incremental consolidation function
        print("=" * 60)
        print("📈 STEP 1.1: KLINES CONSOLIDATION")
        print("=" * 60)
        print("🔄 Starting klines consolidation...")
        logger.info("🔄 Starting klines consolidation...")
        
        klines_pattern = f"klines_{exchange_name}_{symbol}_1m_*.csv"
        klines_consolidated_path = os.path.join(
            "data_cache",
            f"klines_{exchange_name}_{symbol}_1m_consolidated.csv",
        )
        
        print(f"📁 Looking for files matching pattern: {klines_pattern}")
        print(f"💾 Consolidated file path: {klines_consolidated_path}")
        print(f"📋 Expected columns: ['timestamp', 'open', 'high', 'low', 'close', 'volume']")
        logger.info(f"📁 Looking for files matching pattern: {klines_pattern}")
        logger.info(f"💾 Consolidated file path: {klines_consolidated_path}")
        logger.info(f"📋 Expected columns: ['timestamp', 'open', 'high', 'low', 'close', 'volume']")
        
        # Check for existing files
        klines_files = glob.glob(os.path.join("data_cache", klines_pattern))
        print(f"📁 Found {len(klines_files)} klines files")
        if klines_files:
            print(f"📋 First 3 klines files:")
            for i, file in enumerate(klines_files[:3], 1):
                print(f"   {i}. {os.path.basename(file)} ({os.path.getsize(file):,} bytes)")
        
        print(f"🔍 Checking existing consolidated file...")
        logger.info(f"🔍 Checking existing consolidated file...")
        
        if os.path.exists(klines_consolidated_path):
            existing_size = os.path.getsize(klines_consolidated_path)
            print(f"📄 Existing consolidated file found: {existing_size:,} bytes")
            logger.info(f"📄 Existing consolidated file found: {existing_size:,} bytes")
        else:
            print(f"📄 No existing consolidated file found - will create new one")
            logger.info(f"📄 No existing consolidated file found - will create new one")
        
        print(f"🔄 Starting klines consolidation...")
        klines_df = consolidate_files(
            pattern=klines_pattern,
            consolidated_filepath=klines_consolidated_path,
            index_col="timestamp",
            expected_columns=["timestamp", "open", "high", "low", "close", "volume"],
            lookback_days=lookback_days,
        )
        
        logger.info(f"✅ Klines consolidation completed: {len(klines_df)} rows")
        print(f"✅ Klines consolidation completed: {len(klines_df)} rows")
        
        if not klines_df.empty:
            logger.info(f"📊 Klines data info:")
            logger.info(f"   - Shape: {klines_df.shape}")
            logger.info(f"   - Columns: {list(klines_df.columns)}")
            logger.info(f"   - Data types: {klines_df.dtypes.to_dict()}")
            logger.info(f"   - Date range: {klines_df.index.min()} to {klines_df.index.max()}")
            logger.info(f"   - Price range: ${klines_df['low'].min():.2f} to ${klines_df['high'].max():.2f}")
            logger.info(f"   - Memory usage: {klines_df.memory_usage(deep=True).sum() / 1024:.2f} KB")
        else:
            logger.warning(f"⚠️ No klines data found!")

        # Process aggtrades data (limited subset for blank training mode)
        logger.info(
            "📊 Processing aggtrades data (limited subset for training)...",
        )
        print(
            "📊 Processing aggtrades data (limited subset for training)...",
        )

        # Try to load existing aggtrades data
        print("=" * 60)
        print("📊 STEP 1.2: AGGTRADES CONSOLIDATION")
        print("=" * 60)
        
        # Check for different aggtrades file patterns
        agg_trades_pattern_1m = f"aggtrades_{exchange_name}_{symbol}_1m_*.csv"
        agg_trades_pattern_daily = f"aggtrades_{exchange_name}_{symbol}_????-??-??.csv"
        
        print(f"🔍 Looking for aggtrades files with patterns:")
        print(f"   Pattern 1: {agg_trades_pattern_1m}")
        print(f"   Pattern 2: {agg_trades_pattern_daily}")
        
        agg_trades_files_1m = glob.glob(os.path.join("data_cache", agg_trades_pattern_1m))
        agg_trades_files_daily = glob.glob(os.path.join("data_cache", agg_trades_pattern_daily))
        
        print(f"📁 Found {len(agg_trades_files_1m)} files with pattern 1")
        print(f"📁 Found {len(agg_trades_files_daily)} files with pattern 2")
        
        # Show first few files of each pattern
        if agg_trades_files_1m:
            print(f"📋 Pattern 1 files (first 3):")
            for i, file in enumerate(agg_trades_files_1m[:3], 1):
                print(f"   {i}. {os.path.basename(file)} ({os.path.getsize(file):,} bytes)")
        
        if agg_trades_files_daily:
            print(f"📋 Pattern 2 files (first 3):")
            for i, file in enumerate(agg_trades_files_daily[:3], 1):
                print(f"   {i}. {os.path.basename(file)} ({os.path.getsize(file):,} bytes)")
        
        # Use the pattern that has files
        if agg_trades_files_daily:
            agg_trades_pattern = agg_trades_pattern_daily
            print(f"✅ Using daily pattern for aggtrades consolidation")
        elif agg_trades_files_1m:
            agg_trades_pattern = agg_trades_pattern_1m
            print(f"✅ Using 1m pattern for aggtrades consolidation")
        else:
            print(f"❌ No aggtrades files found with either pattern")
            agg_trades_pattern = agg_trades_pattern_daily  # Default to daily pattern

        if agg_trades_files_1m or agg_trades_files_daily:
            print(f"🔄 Starting aggtrades consolidation with pattern: {agg_trades_pattern}")
            # Load and consolidate aggtrades data
            agg_trades_df = consolidate_files(
                pattern=agg_trades_pattern,
                consolidated_filepath=os.path.join(
                    "data_cache",
                    f"aggtrades_{exchange_name}_{symbol}_consolidated.csv",
                ),
                index_col="timestamp",
                expected_columns=[
                    "timestamp",
                    "price",
                    "quantity",
                    "is_buyer_maker",
                    "agg_trade_id",
                ],
                lookback_days=lookback_days,
            )

            # For blank training mode, limit to a reasonable subset
            if len(agg_trades_df) > 10000:
                logger.info(
                    f"   Limiting aggtrades to 10,000 records for training (from {len(agg_trades_df)})",
                )
                agg_trades_df = agg_trades_df.tail(10000)

            logger.info(f"   Loaded {len(agg_trades_df)} aggtrades records")
        else:
            # Create a minimal aggtrades DataFrame if no data exists
            logger.info("   No aggtrades files found, creating minimal dataset")
            agg_trades_df = pd.DataFrame(
                {
                    "price": [2000.0, 2001.0, 2002.0, 2003.0, 2004.0],
                    "quantity": [1.0, 1.0, 1.0, 1.0, 1.0],
                    "is_buyer_maker": [True, False, True, False, True],
                    "agg_trade_id": [1, 2, 3, 4, 5],
                },
                index=pd.date_range("2023-01-01", periods=5, freq="1H"),
            )

        logger.info(
            f"Processed aggtrades dataset with {len(agg_trades_df)} records",
        )
        print(f"Processed aggtrades dataset with {len(agg_trades_df)} records")

        # Process futures data
        print("=" * 60)
        print("📈 STEP 1.3: FUTURES CONSOLIDATION")
        print("=" * 60)
        
        futures_pattern = f"futures_{exchange_name}_{symbol}_*.csv"
        futures_consolidated_path = os.path.join(
            "data_cache",
            f"futures_{exchange_name}_{symbol}_consolidated.csv",
        )
        
        print(f"📁 Looking for files matching pattern: {futures_pattern}")
        print(f"💾 Consolidated file path: {futures_consolidated_path}")
        print(f"📋 Expected columns: ['timestamp', 'fundingRate']")
        
        # Check for existing files
        futures_files = glob.glob(os.path.join("data_cache", futures_pattern))
        print(f"📁 Found {len(futures_files)} futures files")
        if futures_files:
            print(f"📋 First 3 futures files:")
            for i, file in enumerate(futures_files[:3], 1):
                print(f"   {i}. {os.path.basename(file)} ({os.path.getsize(file):,} bytes)")
        
        print(f"🔄 Starting futures consolidation...")
        futures_df = consolidate_files(
            pattern=futures_pattern,
            consolidated_filepath=futures_consolidated_path,
            index_col="timestamp",
            expected_columns=["timestamp", "fundingRate"],
            lookback_days=lookback_days,
        )
        
        logger.info(f"✅ Futures consolidation completed: {len(futures_df)} rows")
        print(f"✅ Futures consolidation completed: {len(futures_df)} rows")

        # Set index for the returned dataframes before use
        for df, idx in [
            (klines_df, "timestamp"),
            (agg_trades_df, "timestamp"),
            (futures_df, "timestamp"),
        ]:
            if not df.empty and idx in df.columns:
                df[idx] = pd.to_datetime(df[idx])
                df.set_index(idx, inplace=True)
                df.sort_index(inplace=True)

        # --- Filter data based on retention period ---
        # This is especially important for blank/test runs to limit the data size.
        if lookback_days:
            logger.info(
                f"✅ Filtering data to the last {lookback_days} days as per request.",
            )
            print(f"✅ Filtering data to the last {lookback_days} days as per request.")

            end_date = datetime.now(klines_df.index.tz if klines_df.index.tz else None)
            start_date = end_date - timedelta(days=int(lookback_days))

            original_klines_len = len(klines_df)
            original_agg_trades_len = len(agg_trades_df)
            original_futures_len = len(futures_df)

            if not klines_df.empty:
                klines_df = klines_df[klines_df.index >= start_date]
            if not agg_trades_df.empty:
                agg_trades_df = agg_trades_df[agg_trades_df.index >= start_date]
            if not futures_df.empty:
                futures_df = futures_df[futures_df.index >= start_date]

            logger.info(f"   - Klines: {original_klines_len} -> {len(klines_df)} rows")
            logger.info(
                f"   - Agg trades: {original_agg_trades_len} -> {len(agg_trades_df)} rows",
            )
            logger.info(
                f"   - Futures: {original_futures_len} -> {len(futures_df)} rows",
            )
        else:
            logger.info(
                "✅ No lookback period specified, using all available consolidated data.",
            )

        # --- Step 3: Validate data ---
        logger.info("✅ STEP 1.3: Validating consolidated data...")
        print("✅ STEP 1.3: Validating consolidated data...")
        total_points = len(klines_df) + len(agg_trades_df) + len(futures_df)
        logger.info(f"📊 Total data points collected: {total_points}")
        print(f"📊 Total data points collected: {total_points}")
        logger.info(f"   - Klines: {len(klines_df)} rows")
        logger.info(f"   - Aggregated trades: {len(agg_trades_df)} rows")
        logger.info(f"   - Futures: {len(futures_df)} rows")

        if total_points < min_points:
            logger.warning(f"⚠️ Insufficient data points: {total_points} < {min_points}")
        if klines_df.empty:
            raise RuntimeError(
                "💥 No klines data available after download - this is critical and the pipeline cannot continue.",
            )

        # --- Step 4: Save final data artifact for the pipeline ---
        logger.info("💾 STEP 1.4: Saving final data artifact to training directory...")
        print("💾 STEP 1.4: Saving final data artifact to training directory...")
        os.makedirs(data_dir, exist_ok=True)

        end_time_ms = int(time.time() * 1000)
        
        # Get lookback days with fallback
        try:
            lookback_days = CONFIG["MODEL_TRAINING"]["data_retention_days"]
        except (KeyError, TypeError):
            # Fallback if CONFIG is not available or MODEL_TRAINING section is missing
            lookback_days = 730  # Default to 2 years
            logger.warning(f"⚠️ CONFIG['MODEL_TRAINING']['data_retention_days'] not available, using default: {lookback_days} days")
        
        start_time_ms = end_time_ms - int(
            timedelta(days=lookback_days).total_seconds() * 1000,
        )
        data_to_save = {
            "klines": klines_df,
            "agg_trades": agg_trades_df,
            "futures": futures_df,
            "metadata": {
                "symbol": symbol,
                "exchange": exchange_name,
                "collection_time": datetime.now().isoformat(),
                "time_range": {
                    "start_ms": start_time_ms,
                    "end_ms": end_time_ms,
                    "lookback_days": lookback_days,
                },
            },
        }

        pickle_path = f"{data_dir}/{exchange_name}_{symbol}_historical_data.pkl"
        with open(pickle_path, "wb") as f:
            pickle.dump(data_to_save, f)
        logger.info(
            f"✅ Successfully saved final data artifact to {pickle_path} ({os.path.getsize(pickle_path)} bytes)",
        )
        print(f"INFO: ✅ Successfully saved final data artifact to {pickle_path}")

        # Final summary
        total_duration = time.time() - start_time
        logger.info("=" * 80)
        logger.info("🎉 STEP 1: DATA COLLECTION COMPLETE")
        logger.info("=" * 80)
        print("=" * 80)  # Explicit print
        print("🎉 STEP 1: DATA COLLECTION COMPLETE")  # Explicit print
        print("=" * 80)  # Explicit print
        logger.info(f"⏱️  Total duration: {total_duration:.2f} seconds")
        logger.info("📊 Final data summary:")
        logger.info(f"   - Klines: {len(klines_df)} rows")
        logger.info(f"   - Aggregated trades: {len(agg_trades_df)} rows")
        logger.info(f"   - Futures: {len(futures_df)} rows")
        logger.info(f"💾 Final training artifact saved to: {pickle_path}")
        logger.info("=" * 80)

        # Run data collection quality analysis
        logger.info("🔍 Running data collection quality analysis...")
        print("🔍 Running data collection quality analysis...")

        try:
            # Import the quality analyzer
            import sys

            sys.path.append(str(Path(__file__).parent.parent.parent.parent))
            from analysis.data_collection_quality_analysis import DataCollectionQualityAnalyzer

            # Create analyzer and load the collected data
            analyzer = DataCollectionQualityAnalyzer()
            data_to_analyze = {
                "klines": klines_df,
                "agg_trades": agg_trades_df,
                "futures": futures_df,
            }
            analyzer.data = data_to_analyze

            # Run the analysis
            analyzer.analyze_data_quality()

            # Save the quality report
            quality_report_path = (
                f"{data_dir}/{exchange_name}_{symbol}_data_collection_quality_report.txt"
            )
            analyzer.save_report(quality_report_path)

            logger.info(
                f"✅ Data collection quality analysis completed and saved to: {quality_report_path}",
            )
            print(
                f"✅ Data collection quality analysis completed and saved to: {quality_report_path}",
            )

        except Exception as e:
            logger.warning(f"⚠️  Data collection quality analysis failed: {e}")
            print(f"⚠️  Data collection quality analysis failed: {e}")

        return klines_df, agg_trades_df, futures_df

    except Exception as e:
        logger.error(f"💥 Unexpected error in data collection: {e}")
        logger.error("Full traceback:")
        logger.error(traceback.format_exc())
        return None, None, None


if __name__ == "__main__":
    print("=" * 80)
    print("🚀 STEP1 SCRIPT STARTING")
    print("=" * 80)
    print(f"Arguments: {sys.argv}")

    # Command-line arguments: symbol, exchange_name, min_data_points, data_dir
    symbol = sys.argv[
        1
    ]  # The download_new_data flag defaults to True, which is correct for direct execution.
    exchange_name = sys.argv[2]
    min_data_points = sys.argv[3]
    data_dir = sys.argv[4]
    lookback_days_arg = None

    # Only try to parse lookback_days if we have more than 4 arguments
    if len(sys.argv) > 5:
        try:
            lookback_days_arg = int(sys.argv[5])
            print(f"Parsed lookback_days: {lookback_days_arg}")
        except (ValueError, IndexError):
            print(f"Could not parse lookback_days argument: '{sys.argv[5]}'. Ignoring.")
            lookback_days_arg = None

    print("Parsed arguments:")
    print(f"  Symbol: {symbol}")
    print(f"  Exchange: {exchange_name}")
    print(f"  Min data points: {min_data_points}")
    print(f"  Data dir: {data_dir}")
    print(f"  Lookback days: {lookback_days_arg}")
    print("Starting asyncio.run...")

    klines_df, agg_trades_df, futures_df = asyncio.run(
        run_step(
            symbol=symbol,
            exchange_name=exchange_name,
            min_data_points=min_data_points,
            data_dir=data_dir,
            lookback_days=lookback_days_arg,
        ),
    )

    print("asyncio.run completed. Results:")
    print(
        f"  klines_df: {type(klines_df)} with {len(klines_df) if klines_df is not None else 'None'} rows",
    )
    print(
        f"  agg_trades_df: {type(agg_trades_df)} with {len(agg_trades_df) if agg_trades_df is not None else 'None'} rows",
    )
    print(
        f"  futures_df: {type(futures_df)} with {len(futures_df) if futures_df is not None else 'None'} rows",
    )

    if klines_df is None:
        print("❌ Step1 failed - klines_df is None")
        sys.exit(1)  # Indicate failure
    print("✅ Step1 completed successfully")
    sys.exit(0)  # Indicate success
