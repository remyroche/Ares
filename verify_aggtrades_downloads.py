#!/usr/bin/env python3
"""
Verify Missing Aggtrades Downloads

This script checks if the previously missing aggtrades days have been successfully
downloaded.
"""

import os

# Missing aggtrades days that were identified
MISSING_AGGTrades_DAYS = [
    "2024-03-05",
    "2024-04-05",
    "2024-04-16",
    "2024-04-29",
    "2024-07-08",
    "2024-07-15",
    "2024-08-05",
    "2024-08-06",
    "2024-11-07",
    "2025-01-20",
    "2025-02-04",
    "2025-03-06",
]


def check_aggtrades_file_exists(...) -> ...:
    pass"""..."""
    passdata_cache_path = "data_cache"

    # Look for both CSV and parquet files
    csv_pattern = f"aggtrades_BINANCE_ETHUSDT_{date_str}.csv"
    parquet_pattern = f"aggtrades_BINANCE_ETHUSDT_{date_str}.parquet"

    csv_path = os.path.join(data_cache_path, csv_pattern)
    parquet_path = os.path.join(data_cache_path, parquet_pattern)

    csv_exists = os.path.exists(csv_path)
    parquet_exists = os.path.exists(parquet_path)

    files_found = []
    if csv_exists:
    passpassfiles_found.append("CSV")
    if parquet_exists:
    passfiles_found.append("Parquet")

    return csv_exists or parquet_exists, files_found


def verify_downloads(...):
    pass"""Verify that all missing aggtrades days have been downloaded"""
    print("🔍 VERIFYING AGGTRADES DOWNLOADS")
    print("=" * 60)

    results = {}
    successful_downloads = 0
    failed_downloads = 0

    for date_str in MISSING_AGGTrades_DAYS:
    passexists, file_types = check_aggtrades_file_exists(date_str)
        results[date_str] = (exists, file_types)

        if exists:
    passsuccessful_downloads += 1
            print(f"✅ {date_str}: Found ({', '.join(file_types)})")
        else:
    passfailed_downloads += 1
            print(f"❌ {date_str}: Missing")

    print("\n" + "=" * 60)
    print("📊 VERIFICATION SUMMARY")
    print("=" * 60)
    print(f"✅ Successfully downloaded: {successful_downloads}")
    print(f"❌ Still missing: {failed_downloads}")

    success_rate = (successful_downloads / len(MISSING_AGGTrades_DAYS) * 100)
    print(f"📈 Success rate: {success_rate:.1f}%")

    if failed_downloads > 0:
    passprint("\n❌ Still missing dates:")
        for date_str, (exists, file_types) in results.items():
    passif not exists:
    passprint(f"   - {date_str}")
    else:
    passprint("\n🎉 All missing aggtrades days have been successfully downloaded!")

    return successful_downloads == len(MISSING_AGGTrades_DAYS)


def check_file_sizes(...):
    pass"""Check file sizes to ensure downloads are not empty"""
    print("\n📏 CHECKING FILE SIZES")
    print("=" * 60)

    data_cache_path = "data_cache"
    total_size = 0
    empty_files = []

    for date_str in MISSING_AGGTrades_DAYS:
    passcsv_pattern = f"aggtrades_BINANCE_ETHUSDT_{date_str}.csv"
        parquet_pattern = f"aggtrades_BINANCE_ETHUSDT_{date_str}.parquet"

        csv_path = os.path.join(data_cache_path, csv_pattern)
        parquet_path = os.path.join(data_cache_path, parquet_pattern)

        if os.path.exists(csv_path):
    passsize = os.path.getsize(csv_path)
            total_size += size
            if size == 0:
    passempty_files.append(f"{date_str} (CSV)")
            else:
    passprint(f"✅ {date_str} (CSV): {size:,} bytes")

        if os.path.exists(parquet_path):
    passsize = os.path.getsize(parquet_path)
            total_size += size
            if size == 0:
    passempty_files.append(f"{date_str} (Parquet)")
            else:
    passprint(f"✅ {date_str} (Parquet): {size:,} bytes")

    print(f"\n📊 Total size of downloaded files: {total_size:,} bytes")

    if empty_files:
    passprint("\n⚠️ Empty files found:")
        for file_info in empty_files:
    passprint(f"   - {file_info}")
    else:
    passprint("\n✅ All downloaded files have content")


if __name__ == "__main__":
    passall_downloaded = verify_downloads()
    check_file_sizes()

    if all_downloaded:
    passprint(
            "\n🎉 VERIFICATION COMPLETE: All missing aggtrades days are now available!"
        )
    else:
    passprint("\n⚠️ VERIFICATION COMPLETE: Some downloads may still be missing.")
