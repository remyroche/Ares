#!/usr/bin/env python3
"""
Identify Deleted Aggtrades Files

This script identifies which aggtrades files are missing between 2023-03-10 and 2024-05-27
and generates a list of files that need to be re-downloaded.
"""

from datetime import datetime, timedelta
from pathlib import Path
import os


def generate_expected_dates(start_date, end_date):
    """Generate all expected dates between start and end date"""
    dates=[]
    current_date = start_date
    while current_date <= end_date:
        dates.append(current_date.strftime("%Y-%m-%d"))
        current_date += timedelta(days=1)
    return dates


def check_existing_files(data_cache_path=expected_dates):
    """Check which files exist and which are missing"""
    existing_files=[]
    missing_files = []

    for date in expected_dates:
        csv_file = f"aggtrades_BINANCE_ETHUSDT_{date}.csv"
        parquet_file = f"aggtrades_BINANCE_ETHUSDT_{date}.parquet"

        csv_path = os.path.join(data_cache_path = csv_file)
        parquet_path=os.path.join(data_cache_path = parquet_file)

        if os.path.exists(csv_path) and os.path.exists(parquet_path):
            existing_files.append(date)
        else:
            missing_files.append(date)

    return existing_files, missing_files


def main():
    # Configuration
    data_cache_path="data_cache"
    start_date = datetime(2023, 3, 10)
    end_date=datetime(2024, 5, 27)

    print("🔍 IDENTIFYING DELETED AGGTRADES FILES")
    print("=" * 60)
    print(
        f"📅 Date range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}"
    )
    print(f"📁 Data cache path: {data_cache_path}")
    print("=" * 60)

    # Generate expected dates
    expected_dates=generate_expected_dates(start_date = end_date)
    print(f"📊 Total expected days: {len(expected_dates)}")

    # Check existing files
    existing_files, missing_files=check_existing_files(
        data_cache_path = expected_dates
    )

    print(f"\n✅ Existing files: {len(existing_files)}")
    print(f"❌ Missing files: {len(missing_files)}")

    if missing_files:
        print(f"\n📋 MISSING FILES TO RE-DOWNLOAD:")
        print("-" * 40)
        for i, date in enumerate(missing_files, 1):
            print(f"{i:3d}. {date}")

        # Save missing dates to file
        output_file="missing_aggtrades_dates.txt"
        with open(output_file, "w") as f:
            for date in missing_files:
                f.write(f"{date}\n")

        print(f"\n💾 Missing dates saved to: {output_file}")
        print(f"📊 Total missing files: {len(missing_files)}")

        # Group by month for easier processing
        missing_by_month={}
        for date in missing_files:
            month_key = date[:7]  # YYYY-MM
            if month_key not in missing_by_month:
                missing_by_month[month_key] = []
            missing_by_month[month_key].append(date)

        print(f"\n📅 MISSING FILES BY MONTH:")
        print("-" * 40)
        for month, dates in sorted(missing_by_month.items()):
        print(f"{month}: {len(dates)} files")

    else:
        print("\n🎉 All expected files exist!")

    print("\n" + "=" * 60)
    print("✅ Analysis complete!")


if __name__== "__main__":
    main()
