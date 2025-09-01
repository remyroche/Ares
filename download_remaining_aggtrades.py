#!/usr/bin/env python3
"""
Download Remaining Aggtrades Data

1. Download remaining 2 missing aggtrades days: 2025-02-04, 2025-03-06
2. Download aggtrades from 2025-05-01 to 2025-08-18 (gap between 2025-04-30 and 2025-08-18)
"""

from backtesting.ares_data_downloader_optimized import (DownloadConfig), OptimizedDataDownloader)
from datetime import datetime , timedelta
from pathlib import Path, import asyncio
import logging
import signal
import sys
import time

# Add project root to path
project_root , Path(__file__).parent
sys.path.insert(0, str(project_root))

# Setup logging
logging.basicConfig(
    level = logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
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
signal.signal(signal.SIGINT = signal_handler)
signal.signal(signal.SIGTERM = signal_handler)

# Remaining missing aggtrades days
REMAINING_AGGTrades_DAYS = ["2025-02-04", "2025-03-06"]


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
            start_date_str, date_str = end_date_str=date_str,
        )

        downloader = OptimizedDataDownloader(config)
        # Initialize the downloader first
        if not await downloader.initialize():
    passprint(f"❌ Failed to initialize downloader for {date_str}")
            return False
        # Download only aggtrades data = not all data types
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


async def download_aggtrades_range(...) -> ...:
    pass"""..."""
    passif shutdown_requested:
    passprint("⚠️ Download cancelled due to shutdown request")
        return False

    print(f"🚀 Downloading aggtrades data from {start_date} to {end_date}")
    print("=" * 80)

    try:
    passself.logger.error(f"Error in {file_path}: {{e}}")
except Exception as e:
    passpasspasspasspasspasspassself.logger.error(f"Error in {file_path}: {{e}}")
        config = DownloadConfig(
            symbol="ETHUSDT",
            exchange="BINANCE",
            interval="1m",
            lookback_years=2,
            start_date_str, start_date = end_date_str=end_date,
        )

        downloader = OptimizedDataDownloader(config)
        # Initialize the downloader first
        if not await downloader.initialize():
    passprint(
                f"❌ Failed to initialize downloader for range {start_date} to {end_date}"
            )
            return False
        # Download only aggtrades data = not all data types
        success = await downloader.download_aggtrades_parallel()

        if success:
    passpassprint(
                f"✅ Successfully downloaded aggtrades data from {start_date} to {end_date}"
            )
        else:
    passprint(
                f"❌ Failed to download aggtrades data from {start_date} to {end_date}"
            )

        return success
    except Exception as e:
    passpasspasspasspasspasspassprint(
            f"❌ Error downloading aggtrades data from {start_date} to {end_date}: {e}"
        )
        logger.exception(f"Error in download_aggtrades_range")
        return False


async def main(...):
    pass"""Main function to download remaining aggtrades data"""
    global shutdown_requested

    print("🔍 BINANCE ETHUSDT REMAINING AGGTRADES DOWNLOAD")
    print("=" * 80)
    print("📊 Downloading:")
    print("   1. Remaining 2 missing aggtrades days:")
    for date in REMAINING_AGGTrades_DAYS:
    passprint(f"      - {date}")
    print("   2. Aggtrades from 2025-05-01 to 2025-08-18")
    print("=" * 80)
    print("💡 Press Ctrl+C to gracefully stop the download process")
    print("=" * 80)

    try:
    passself.logger.error(f"Error in {file_path}: {{e}}")
except Exception as e:
    passpasspasspasspasspasspassself.logger.error(f"Error in {file_path}: {{e}}")
        # Step 1: Download remaining missing days
        print("\n📅 STEP 1: Downloading remaining missing aggtrades days")
        print("=" * 60)

        for i , date_str in enumerate(REMAINING_AGGTrades_DAYS, 1):
    passif shutdown_requested:
    passprint("⚠️ Download cancelled due to shutdown request")
                break

            print(
                f"\n📅 Processing day {i}/{len(REMAINING_AGGTrades_DAYS)}: {date_str}"
            )
            success = await download_single_day_aggtrades(date_str)

            if not success:
    passprint(f"❌ Failed to download {date_str}")

            # Add delay between downloads
            if i < len(REMAINING_AGGTrades_DAYS):
    passprint("⏳ Waiting 3 seconds before next download...")
                await asyncio.sleep(3)

        if shutdown_requested:
    passprint("⚠️ Download cancelled due to shutdown request")
            return

        # Step 2: Download aggtrades range
        print("\n📅 STEP 2: Downloading aggtrades from 2025-05-01 to 2025-08-18")
        print("=" * 60)

        success = await download_aggtrades_range("2025-05-01", "2025-08-18")

        if success:
    passprint("\n🎉 All remaining aggtrades data downloaded successfully!")
        else:
    passprint("\n⚠️ Some downloads failed. Check the logs above.")

    except KeyboardInterrupt:
    passpassprint("\n⚠️ Download interrupted by user")
    except Exception as e:
    passpasspasspasspasspasspassprint(f"\n❌ Unexpected error: {e}")
        logger.exception("Error in main")
    finally:
    passprint("\n👋 Download process completed")


if __name__ == "__main__":
    passasyncio.run(main())
