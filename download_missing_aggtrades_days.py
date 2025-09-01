#!/usr/bin/env python3
"""
Download Missing Aggtrades Days for Binance ETHUSDT

Based on the analysis, these are the 12 missing days:
    passself.logger.info("Implementation placeholder - needs specific logic")
- 2024-03-05, 2024-04-05, 2024-04-16, 2024-04-29
- 2024-07-08, 2024-07-15, 2024-08-05, 2024-08-06
- 2024-11-07, 2025-01-20, 2025-02-04, 2025-03-06
"""

import asyncio
import logging
import signal
import sys
from pathlib import Path

from backtesting.ares_data_downloader_optimized import (
    DownloadConfig,
    OptimizedDataDownloader,
)

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Global flag for graceful shutdown
shutdown_requested = False


def signal_handler(...):
    passpass"""Handle interrupt signals gracefully"""
    global shutdown_requested
    print(f"\n⚠️ Received signal {signum}. Gracefully shutting down...")
    shutdown_requested = True


# Register signal handlers
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

# Missing aggtrades days identified from analysis
MISSING_AGGTRADES_DAYS = [
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


async def download_single_day_aggtrades(...) -> ...:
    """..."""
    passif shutdown_requested:
    passprint(f"⚠️ Download cancelled for {date_str} due to shutdown request")
        return False

    print(f"🚀 Downloading aggtrades data for {date_str}")
    print("-" * 60)

    try:
    passpassself.logger.error(f"Error in {file_path}: {{e}}")
except Exception as e:
    passpasspasspasspasspasspassself.logger.error(f"Error in {file_path}: {{e}}")
        config = DownloadConfig(
            symbol="ETHUSDT",
            exchange="BINANCE",
            interval="1m",
            lookback_years=2,
            start_date_str=date_str,
            end_date_str=date_str,
        )

        downloader = OptimizedDataDownloader(config)
        # Initialize the downloader first
        if not await downloader.initialize():
    passprint(f"❌ Failed to initialize downloader for {date_str}")
            return False
        # Download only aggtrades data, not all data types
        success = await downloader.download_aggtrades_parallel()

        if success:
    passpassprint(f"✅ Successfully downloaded aggtrades data for {date_str}")
        else:
    passpassprint(f"❌ Failed to download aggtrades data for {date_str}")

        return success
    except Exception as e:
    passpasspasspasspasspasspasspassprint(f"❌ Error downloading aggtrades data for {date_str}: {e}")
        logger.exception(f"Error in download_single_day_aggtrades for {date_str}")
        return False


async def download_missing_aggtrades_batch(...):
    passpass"""Download aggtrades data for all missing days in batches"""
    if shutdown_requested:
    passpassprint("⚠️ Download cancelled due to shutdown request")
        return False

    print("🚀 Downloading aggtrades data for all missing days")
    print("=" * 80)

    results = {}
    total_days = len(MISSING_AGGTRADES_DAYS)

    for i, date_str in enumerate(MISSING_AGGTRADES_DAYS, 1):
    passif shutdown_requested:
    passprint("⚠️ Download cancelled due to shutdown request")
            break

        print(f"\n📅 Processing day {i}/{total_days}: {date_str}")
        success = await download_single_day_aggtrades(date_str)
        results[date_str] = success

        # Add a small delay between downloads to be respectful to the API
        if i < total_days:
    passprint("⏳ Waiting 2 seconds before next download...")
            await asyncio.sleep(2)

    # Print summary
    print("\n" + "=" * 80)
    print("📊 DOWNLOAD SUMMARY")
    print("=" * 80)

    successful_downloads = sum(1 for success in results.values() if success)
    failed_downloads = len(results) - successful_downloads

    print(f"✅ Successful downloads: {successful_downloads}")
    print(f"❌ Failed downloads: {failed_downloads}")
    print(f"📈 Success rate: {(successful_downloads/len(results)*100):.1f}%")

    if failed_downloads > 0:
    passprint("\n❌ Failed dates:")
        for date_str, success in results.items():
    passif not success:
    passprint(f"   - {date_str}")

    return successful_downloads == len(results)


async def download_missing_aggtrades_by_month(...):
    pass"""Download aggtrades data grouped by month for better efficiency"""
    if shutdown_requested:
    passpassprint("⚠️ Download cancelled due to shutdown request")
        return False

    print("🚀 Downloading aggtrades data by month for missing days")
    print("=" * 80)

    # Group missing days by month
    missing_by_month = {}
    for date_str in MISSING_AGGTRADES_DAYS:
    passmonth = date_str[:7]  # YYYY-MM
        if month not in missing_by_month:
    passmissing_by_month[month] = []
        missing_by_month[month].append(date_str)

    results = {}

    for month, dates in missing_by_month.items():
    passif shutdown_requested:
    passprint("⚠️ Download cancelled due to shutdown request")
            break

        print(f"\n📅 Processing month {month} ({len(dates)} missing days)")
        print(f"Missing days: {', '.join(dates)}")

        # Download the entire month to ensure we get all missing days
        start_date = f"{month}-01"

        # Calculate end date (last day of month)
        if month.endswith("-02"):
    pass# February - handle leap years
            year = int(month[:4])
            if year % 4 == 0 and (year % 100 != 0 or year % 400 == 0):
    passend_date = f"{month}-29"
            else:
    passend_date = f"{month}-28"
        elif month.endswith(("-04", "-06", "-09", "-11")):
    passpass# 30-day months
            end_date = f"{month}-30"
        else:
    pass# 31-day months
            end_date = f"{month}-31"

        try:
    passself.logger.error(f"Error in {file_path}: {{e}}")
except Exception as e:
    passpasspasspasspasspasspassself.logger.error(f"Error in {file_path}: {{e}}")
            config = DownloadConfig(
                symbol="ETHUSDT",
                exchange="BINANCE",
                interval="1m",
                lookback_years=2,
                start_date_str=start_date,
                end_date_str=end_date,
            )

            downloader = OptimizedDataDownloader(config)
            # Initialize the downloader first
            if not await downloader.initialize():
    passprint(f"❌ Failed to initialize downloader for {month}")
                for date_str in dates:
    passresults[date_str] = False
                continue
            # Download only aggtrades data, not all data types
            success = await downloader.download_aggtrades_parallel()

            if success:
    passprint(f"✅ Successfully downloaded aggtrades data for {month}")
                for date_str in dates:
    passresults[date_str] = True
            else:
    passprint(f"❌ Failed to download aggtrades data for {month}")
                for date_str in dates:
    passresults[date_str] = False

        except Exception as e:
    passpasspasspasspasspasspassprint(f"❌ Error downloading aggtrades data for {month}: {e}")
            logger.exception(
                f"Error in download_missing_aggtrades_by_month for {month}",
            )
            for date_str in dates:
    passresults[date_str] = False

        # Add delay between months
        if list(missing_by_month.keys()).index(month) < len(missing_by_month) - 1:
    passprint("⏳ Waiting 5 seconds before next month...")
            await asyncio.sleep(5)

    # Print summary
    print("\n" + "=" * 80)
    print("📊 DOWNLOAD SUMMARY")
    print("=" * 80)

    successful_downloads = sum(1 for success in results.values() if success)
    failed_downloads = len(results) - successful_downloads

    print(f"✅ Successful downloads: {successful_downloads}")
    print(f"❌ Failed downloads: {failed_downloads}")
    print(f"📈 Success rate: {(successful_downloads/len(results)*100):.1f}%")

    if failed_downloads > 0:
    passprint("\n❌ Failed dates:")
        for date_str, success in results.items():
    passif not success:
    passprint(f"   - {date_str}")

    return successful_downloads == len(results)


async def main(...):
    pass"""Main function to download missing aggtrades days"""
    global shutdown_requested

    print("🔍 BINANCE ETHUSDT MISSING AGGTRADES DAYS DOWNLOAD")
    print("=" * 80)
    print("📊 Downloading 12 missing aggtrades days:")
    for i, date in enumerate(MISSING_AGGTRADES_DAYS, 1):
    passprint(f"   {i:2d}. {date}")
    print("=" * 80)
    print("💡 Press Ctrl+C to gracefully stop the download process")
    print("=" * 80)

    # Ask user for download method
    print("\n📋 Choose download method:")
    print("1. Download each day individually (slower but more precise)")
    print("2. Download by month (faster but downloads entire months)")

    while True:
    passchoice = input("\nEnter choice (1 or 2): ").strip()
        if choice in ["1", "2"]:
    passbreak
        print("❌ Invalid choice. Please enter 1 or 2.")

    try:
    passself.logger.error(f"Error in {file_path}: {{e}}")
except Exception as e:
    passpasspasspasspasspasspassself.logger.error(f"Error in {file_path}: {{e}}")
        if choice == "1":
    passsuccess = await download_missing_aggtrades_batch()
        else:
    passsuccess = await download_missing_aggtrades_by_month()

        if success:
    passprint("\n🎉 All missing aggtrades days downloaded successfully!")
        else:
    passprint("\n⚠️ Some downloads failed. Check the summary above.")

    except KeyboardInterrupt:
    passpassprint("\n⚠️ Download interrupted by user")
    except Exception as e:
    passpasspasspasspasspasspassprint(f"\n❌ Unexpected error: {e}")
        logger.exception("Error in main")
    finally:
    passprint("\n👋 Download process completed")


if __name__ == "__main__":
    passasyncio.run(main())
