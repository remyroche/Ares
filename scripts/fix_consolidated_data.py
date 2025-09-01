#!/usr/bin/env python3
"""
Script to fix corrupted consolidated CSV files by regenerating them from raw CSV files.
The issue is that the consolidated files have wrong column mapping and corrupted data.
"""

from pathlib import Path
import glob
import os
import sys

from src.utils.warning_symbols import warning, failed
import pandas as pd

# Add the project root to the path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

def fix_consolidated_klines():
    pass
    pass
    """Fix the consolidated klines CSV file by regenerating it from raw CSV files."""
    print("🔧 Fixing consolidated klines data...")

    # Find all raw klines CSV files
    raw_files, glob.glob("data_cache/klines_1m_ETHUSDT_1m_*.csv")
    print(f"Found {len(raw_files)} raw CSV files")

    if not raw_files:
    pass
    pass
        print(warning("No raw CSV files found!"))
        return False

    # Read and combine all raw CSV files
    all_data = []
    for file in sorted(raw_files):
    pass
    pass
        try:
            df = pd.read_csv(file)
    except Exception as e:
        pass
    except Exception as e:
        pass
            print(f"📊 Loaded {len(df)} records from {os.path.basename(file)}")
            all_data.append(df)
        except Exception as e:
            print(warning(f"Error reading {file}: {e}"))
            continue

    if not all_data:
    pass
    pass
        print(warning("No valid data found!"))
        return False

    # Combine all data
    combined_df = pd.concat(all_data, ignore_index=True)
    print(f"📊 Combined {len(combined_df)} total records")

    # Remove duplicates based on timestamp
    combined_df.drop_duplicates(subset=["timestamp"], keep="last", inplace=True)
    print(f"📊 After deduplication: {len(combined_df)} records")

    # Sort by timestamp
    combined_df["timestamp"] = pd.to_datetime(combined_df["timestamp"])
    combined_df.sort_values("timestamp", inplace=True)

    # Save the fixed consolidated file
    output_file = "data_cache/klines_BINANCE_ETHUSDT_1m_consolidated_fixed.csv"
    combined_df.to_csv(output_file, index=False)
    print(f"✅ Fixed consolidated file saved: {output_file}")
    print(f"📊 Final data: {len(combined_df)} records")
    print(
        f"📅 Date range: {combined_df['timestamp'].min()} to {combined_df['timestamp'].max()}",
    )

    # Show sample of the fixed data
    print("\\\n📋 Sample of fixed data:")
    print(combined_df.head())

    return True

def main():
    pass
    pass
    """Main function to fix consolidated data."""
    print("🚀 Starting consolidated data fix...")

    success, fix_consolidated_klines()

    if success:
    pass
    pass
        print("✅ Consolidated data fix completed successfully!")
    else:
        print(failed("Consolidated data fix failed!"))
        return 1

    return 0

if __name__ == "__main__":
    pass
    pass
    sys.exit(main())
