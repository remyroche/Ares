#!/usr/bin/env python3
"""
Clean Parquet Data Duplicates

This script removes duplicate rows from parquet files in the historical data storage.
It distinguishes between:
1. TRUE duplicates: Same timestamp AND same values across all columns
2. Timestamp conflicts: Same timestamp but different values (DATA QUALITY ISSUE)

Usage:
    python3 scripts/clean_parquet_duplicates.py [--dry-run] [--symbol ETHUSDT] [--exchange binance]
"""

import argparse
import glob
import os
import shutil
from pathlib import Path
from datetime import datetime
import pandas as pd
import sys


def find_parquet_files(base_path: str, symbol: str = None, exchange: str = None) -> list:
    """Find all parquet files matching the criteria."""
    if symbol and exchange:
        pattern = f"{base_path}/{exchange.lower()}/{symbol.lower()}/**/*.parquet"
    elif exchange:
        pattern = f"{base_path}/{exchange.lower()}/**/*.parquet"
    else:
        pattern = f"{base_path}/**/*.parquet"
    
    files = glob.glob(pattern, recursive=True)
    return sorted(files)


def analyze_duplicates(df: pd.DataFrame, file_path: str) -> dict:
    """
    Analyze duplicate timestamps in a dataframe.
    Distinguishes between true duplicates and timestamp conflicts.
    """
    if not hasattr(df, 'index'):
        return {
            'has_duplicates': False,
            'total_rows': len(df),
            'duplicate_timestamps': 0,
            'true_duplicates': 0,
            'timestamp_conflicts': 0
        }
    
    # Find rows with duplicate timestamps
    duplicate_timestamp_mask = df.index.duplicated(keep=False)
    duplicate_timestamp_count = duplicate_timestamp_mask.sum()
    
    if duplicate_timestamp_count == 0:
        return {
            'has_duplicates': False,
            'total_rows': len(df),
            'duplicate_timestamps': 0,
            'true_duplicates': 0,
            'timestamp_conflicts': 0
        }
    
    # Check if duplicates are TRUE duplicates (same values) or conflicts (different values)
    # Get rows with duplicate timestamps
    duplicate_rows = df[duplicate_timestamp_mask]
    
    # Check for true duplicates (duplicate across ALL columns including index)
    true_duplicate_mask = df.duplicated(keep=False)  # Checks all columns + index
    true_duplicate_count = true_duplicate_mask.sum()
    
    # Timestamp conflicts = duplicate timestamps but different values
    timestamp_conflicts = duplicate_timestamp_count - true_duplicate_count
    
    result = {
        'has_duplicates': duplicate_timestamp_count > 0,
        'total_rows': len(df),
        'duplicate_timestamps': duplicate_timestamp_count,
        'true_duplicates': true_duplicate_count,
        'timestamp_conflicts': timestamp_conflicts,
        'unique_rows': len(df) - df.index.duplicated().sum(),
        'duplicate_percentage': (duplicate_timestamp_count / len(df) * 100) if len(df) > 0 else 0
    }
    
    # If there are timestamp conflicts, provide examples
    if timestamp_conflicts > 0:
        # Find timestamps that have conflicts
        conflict_timestamps = []
        for ts in df[duplicate_timestamp_mask].index.unique():
            ts_rows = df.loc[ts] if isinstance(df.loc[ts], pd.DataFrame) else pd.DataFrame([df.loc[ts]])
            if len(ts_rows) > 1:
                # Check if values differ
                if not ts_rows.duplicated().all():
                    conflict_timestamps.append(ts)
                    if len(conflict_timestamps) >= 3:  # Limit examples
                        break
        
        result['conflict_examples'] = conflict_timestamps[:3]
    
    return result


def clean_parquet_file(file_path: str, dry_run: bool = False, create_backup: bool = True) -> dict:
    """
    Clean duplicates from a single parquet file.
    Only removes TRUE duplicates (same timestamp AND same values).
    Warns about timestamp conflicts (same timestamp, different values).
    """
    result = {
        'file': file_path,
        'success': False,
        'error': None,
        'stats': {}
    }
    
    try:
        # Read parquet file
        print(f"\n📂 Processing: {file_path}")
        df = pd.read_parquet(file_path)
        print(f"   Loaded {len(df)} rows")
        
        # Analyze duplicates
        stats = analyze_duplicates(df, file_path)
        result['stats'] = stats
        
        if not stats['has_duplicates']:
            print(f"   ✅ No duplicates found")
            result['success'] = True
            result['message'] = 'No duplicates found'
            return result
        
        # Report findings
        print(f"   📊 Duplicate Analysis:")
        print(f"      - Duplicate timestamps: {stats['duplicate_timestamps']} ({stats['duplicate_percentage']:.1f}%)")
        print(f"      - TRUE duplicates (same values): {stats['true_duplicates']}")
        print(f"      - Timestamp conflicts (different values): {stats['timestamp_conflicts']}")
        
        # Warn about timestamp conflicts
        if stats['timestamp_conflicts'] > 0:
            print(f"   ⚠️  WARNING: Found {stats['timestamp_conflicts']} timestamp conflicts!")
            print(f"      These are rows with the same timestamp but DIFFERENT values.")
            print(f"      This indicates a DATA QUALITY ISSUE that needs investigation.")
            if 'conflict_examples' in stats:
                print(f"      Example conflict timestamps: {stats['conflict_examples'][:3]}")
            result['warning'] = f"Timestamp conflicts detected: {stats['timestamp_conflicts']} rows"
        
        if stats['true_duplicates'] == 0:
            print(f"   ℹ️  No true duplicates to remove (only timestamp conflicts)")
            result['success'] = True
            result['message'] = 'No true duplicates, only timestamp conflicts'
            return result
        
        if dry_run:
            print(f"   🔍 DRY RUN: Would remove {stats['true_duplicates']} true duplicates")
            result['success'] = True
            result['message'] = f"Would remove {stats['true_duplicates']} true duplicates"
            return result
        
        # Create backup if requested
        if create_backup:
            backup_path = f"{file_path}.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            shutil.copy2(file_path, backup_path)
            result['backup'] = backup_path
            print(f"   💾 Backup created: {backup_path}")
        
        # Remove TRUE duplicates only (same timestamp AND same values)
        df_clean = df[~df.duplicated(keep='first')]  # This checks ALL columns including index
        
        # Save cleaned data
        df_clean.to_parquet(file_path)
        
        rows_removed = len(df) - len(df_clean)
        print(f"   ✅ Removed {rows_removed} true duplicates, kept {len(df_clean)} unique rows")
        
        result['success'] = True
        result['message'] = f"Removed {rows_removed} true duplicates, kept {len(df_clean)} unique rows"
        result['stats']['rows_after'] = len(df_clean)
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        result['error'] = str(e)
        result['message'] = f"Error: {e}"
    
    return result


def main():
    parser = argparse.ArgumentParser(
        description='Clean duplicate rows from parquet files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Dry run to see what would be cleaned
  python3 scripts/clean_parquet_duplicates.py --dry-run

  # Clean ETHUSDT data on binance
  python3 scripts/clean_parquet_duplicates.py --symbol ETHUSDT --exchange binance

  # Clean all data without backups (faster)
  python3 scripts/clean_parquet_duplicates.py --no-backup
        """
    )
    parser.add_argument('--dry-run', action='store_true', 
                       help='Show what would be done without making changes')
    parser.add_argument('--symbol', type=str, 
                       help='Filter by symbol (e.g., ETHUSDT)')
    parser.add_argument('--exchange', type=str, 
                       help='Filter by exchange (e.g., binance)')
    parser.add_argument('--no-backup', action='store_true',
                       help='Skip creating backup files (faster but riskier)')
    parser.add_argument('--base-path', type=str, 
                       default='historical_data/storage',
                       help='Base path for historical data (default: historical_data/storage)')
    
    args = parser.parse_args()
    
    # Validate base path exists
    if not os.path.exists(args.base_path):
        print(f"❌ Error: Base path '{args.base_path}' does not exist")
        sys.exit(1)
    
    # Find parquet files
    print(f"\n🔍 Searching for parquet files in: {args.base_path}")
    if args.symbol:
        print(f"   Symbol filter: {args.symbol}")
    if args.exchange:
        print(f"   Exchange filter: {args.exchange}")
    
    files = find_parquet_files(args.base_path, args.symbol, args.exchange)
    
    if not files:
        print(f"❌ No parquet files found matching criteria")
        sys.exit(1)
    
    print(f"✅ Found {len(files)} parquet files to process")
    
    if args.dry_run:
        print(f"\n🔍 DRY RUN MODE - No changes will be made\n")
    
    # Process files
    results = []
    total_true_duplicates = 0
    total_timestamp_conflicts = 0
    files_with_conflicts = []
    
    for file_path in files:
        result = clean_parquet_file(
            file_path, 
            dry_run=args.dry_run,
            create_backup=not args.no_backup
        )
        results.append(result)
        
        if result['success'] and result['stats'].get('true_duplicates', 0) > 0:
            total_true_duplicates += result['stats']['true_duplicates']
        
        if result['stats'].get('timestamp_conflicts', 0) > 0:
            total_timestamp_conflicts += result['stats']['timestamp_conflicts']
            files_with_conflicts.append(file_path)
    
    # Summary
    print(f"\n{'='*80}")
    print(f"📊 SUMMARY")
    print(f"{'='*80}")
    print(f"Files processed: {len(results)}")
    print(f"Files with true duplicates: {sum(1 for r in results if r['stats'].get('true_duplicates', 0) > 0)}")
    print(f"Total true duplicates found: {total_true_duplicates}")
    print(f"Total timestamp conflicts found: {total_timestamp_conflicts}")
    
    if files_with_conflicts:
        print(f"\n⚠️  WARNING: {len(files_with_conflicts)} files have timestamp conflicts:")
        for file_path in files_with_conflicts[:10]:  # Show first 10
            print(f"   - {file_path}")
        if len(files_with_conflicts) > 10:
            print(f"   ... and {len(files_with_conflicts) - 10} more")
        print(f"\n   Timestamp conflicts indicate DATA QUALITY ISSUES:")
        print(f"   - Same timestamp but different OHLCV values")
        print(f"   - May indicate data collection errors or timezone issues")
        print(f"   - Recommend investigating the source data")
    
    if args.dry_run:
        print(f"\n🔍 This was a DRY RUN - no changes were made")
        print(f"   Run without --dry-run to actually clean the files")
    else:
        print(f"\n✅ Cleaning complete!")
        if not args.no_backup:
            print(f"   Backups were created with .backup_YYYYMMDD_HHMMSS extension")
    
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()
