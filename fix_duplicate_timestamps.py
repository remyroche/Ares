#!/usr/bin/env python3
"""
Fix Duplicate Timestamps in ETH/USDT Data

This script fixes the duplicate timestamp issue in the consolidated ETH/USDT 1-minute data.
Root cause: Data was collected twice and merged without deduplication (not resampling).

Issue: 53.12% of records have duplicate timestamps with conflicting OHLC data.
Strategy:
1. For duplicate timestamps, keep the record with highest volume (most complete data)
2. Convert timestamps from seconds to milliseconds (required format)
3. Verify data integrity after cleaning
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def fix_duplicate_timestamps(input_file: str, output_file: str) -> bool:
    """Fix duplicate timestamps in the data file."""
    try:
        logger.info(f"Loading data from {input_file}")
        df = pd.read_parquet(input_file)

        original_count = len(df)
        logger.info(f"Original data: {original_count:,} records")

        # Check for duplicates
        duplicates = df['timestamp'].duplicated()
        duplicate_count = duplicates.sum()
        logger.info(f"Found {duplicate_count:,} duplicate timestamps")

        if duplicate_count == 0:
            logger.info("No duplicates found - data is clean")
            return True

        # Convert timestamps from seconds to milliseconds if needed
        if df['timestamp'].max() < 1e10:  # Likely in seconds
            logger.info("Converting timestamps from seconds to milliseconds")
            df['timestamp'] = (df['timestamp'] * 1000).astype('int64')

        # Group by timestamp and resolve conflicts
        logger.info("Resolving duplicate timestamps...")

        # For each timestamp group, keep the record with highest volume
        resolved_records = []

        for timestamp, group in df.groupby('timestamp'):
            if len(group) == 1:
                # No conflict, keep as is
                resolved_records.append(group.iloc[0])
            else:
                # Conflict: keep record with highest volume
                # If tie, keep first occurrence
                best_record = group.loc[group['volume'].idxmax()]
                resolved_records.append(best_record)

                # Log conflicts for monitoring
                if len(group) > 2:
                    logger.warning(f"Timestamp {timestamp}: {len(group)} conflicting records")

        # Create new dataframe
        df_cleaned = pd.DataFrame(resolved_records)

        # Sort by timestamp
        df_cleaned = df_cleaned.sort_values('timestamp').reset_index(drop=True)

        cleaned_count = len(df_cleaned)
        removed_count = original_count - cleaned_count

        logger.info(f"Cleaned data: {cleaned_count:,} records")
        logger.info(f"Removed {removed_count:,} duplicate records")

        # Verify no duplicates remain
        final_duplicates = df_cleaned['timestamp'].duplicated().sum()
        if final_duplicates > 0:
            logger.error(f"ERROR: Still have {final_duplicates} duplicates after cleaning!")
            return False

        # Save cleaned data
        logger.info(f"Saving cleaned data to {output_file}")
        df_cleaned.to_parquet(output_file, index=False)

        # Verify the output
        df_verify = pd.read_parquet(output_file)
        verify_duplicates = df_verify['timestamp'].duplicated().sum()
        logger.info(f"Verification: {len(df_verify):,} records, {verify_duplicates} duplicates")

        if verify_duplicates == 0:
            logger.info("✅ SUCCESS: Duplicate timestamps fixed!")
            return True
        else:
            logger.error("❌ FAILED: Duplicates still exist in output")
            return False

    except Exception as e:
        logger.error(f"Error fixing duplicates: {e}")
        return False

def analyze_conflicts(input_file: str):
    """Analyze the nature of conflicts in duplicate timestamps."""
    try:
        df = pd.read_parquet(input_file)

        # Get duplicate groups
        duplicate_groups = df[df['timestamp'].duplicated(keep=False)].groupby('timestamp')

        logger.info("Analyzing conflict patterns...")

        conflict_summary = {
            'total_conflicts': 0,
            'ohlc_differences': 0,
            'volume_differences': 0,
            'sample_conflicts': []
        }

        for i, (timestamp, group) in enumerate(duplicate_groups):
            if len(group) > 1:
                conflict_summary['total_conflicts'] += 1

                # Check for OHLC differences
                ohlc_cols = ['open', 'high', 'low', 'close']
                ohlc_different = any(group[col].nunique() > 1 for col in ohlc_cols)
                if ohlc_different:
                    conflict_summary['ohlc_differences'] += 1

                # Check for volume differences
                volume_different = group['volume'].nunique() > 1
                if volume_different:
                    conflict_summary['volume_differences'] += 1

                # Store sample conflicts
                if len(conflict_summary['sample_conflicts']) < 5:
                    conflict_summary['sample_conflicts'].append({
                        'timestamp': timestamp,
                        'count': len(group),
                        'ohlc_different': ohlc_different,
                        'volume_different': volume_different
                    })

            if i >= 1000:  # Limit analysis to first 1000 conflicts
                break

        logger.info(f"Conflict Analysis Summary:")
        logger.info(f"  Total conflicts: {conflict_summary['total_conflicts']}")
        logger.info(f"  OHLC differences: {conflict_summary['ohlc_differences']}")
        logger.info(f"  Volume differences: {conflict_summary['volume_differences']}")

        for sample in conflict_summary['sample_conflicts']:
            logger.info(f"  Sample: TS {sample['timestamp']}, {sample['count']} records, "
                       f"OHLC diff: {sample['ohlc_different']}, Vol diff: {sample['volume_different']}")

        return conflict_summary

    except Exception as e:
        logger.error(f"Error analyzing conflicts: {e}")
        return None

def main():
    """Main function to fix duplicate timestamps."""
    input_file = "historical_data/binance/ethusdt/processed/ethusdt_1m/features_ethusdt_1m_consolidated.parquet"
    output_file = "historical_data/binance/ethusdt/processed/ethusdt_1m/features_ethusdt_1m_consolidated_fixed.parquet"

    # Check if input file exists
    if not Path(input_file).exists():
        logger.error(f"Input file not found: {input_file}")
        return False

    logger.info("=== Duplicate Timestamp Fix Script ===")
    logger.info(f"Input: {input_file}")
    logger.info(f"Output: {output_file}")

    # First, analyze the conflicts
    logger.info("Step 1: Analyzing conflicts...")
    conflict_analysis = analyze_conflicts(input_file)

    # Fix the duplicates
    logger.info("Step 2: Fixing duplicates...")
    success = fix_duplicate_timestamps(input_file, output_file)

    if success:
        logger.info("=== Fix Complete ===")
        logger.info(f"Fixed file saved to: {output_file}")

        # Show before/after stats
        df_original = pd.read_parquet(input_file)
        df_fixed = pd.read_parquet(output_file)

        logger.info(f"Before: {len(df_original):,} records, {df_original['timestamp'].duplicated().sum():,} duplicates")
        logger.info(f"After:  {len(df_fixed):,} records, {df_fixed['timestamp'].duplicated().sum():,} duplicates")

        return True
    else:
        logger.error("=== Fix Failed ===")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
