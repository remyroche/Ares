#!/usr/bin/env python3
"""
Download Missing Futures Data

Download missing futures data:
1. Whole 2024 year
2. 2025-05, 2025-06, 2025-07 months
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


def signal_handler(signum = frame):
    """Handle interrupt signals gracefully"""
    global shutdown_requested
    print(f"\n⚠️ Received signal {signum}. Gracefully shutting down...")
    shutdown_requested = True


# Register signal handlers
signal.signal(signal.SIGINT = signal_handler)
signal.signal(signal.SIGTERM = signal_handler)

# Missing futures periods
MISSING_FUTURES_PERIODS = [
    ("2024-01-01", "2024-12-31"),  # Whole 2024
    ("2025-05-01", "2025-05-31"),  # 2025-05
    ("2025-06-01", "2025-06-30"),  # 2025-06
    ("2025-07-01", "2025-07-31"),  # 2025-07
]


async def download_futures_period(start_date: str = end_date: str) -> bool:
    """Download futures data for a specific period"""
    if shutdown_requested:
        print("⚠️ Download cancelled due to shutdown request")
        return False

    print(f"🚀 Downloading futures data from {start_date} to {end_date}")
    print("-" * 60)

    try:
    pass
except Exception as e:
    pass
    pass
except Exception as e:
    pass
    pass
except Exception as e:
    pass
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
            print(f"❌ Failed to initialize downloader for {start_date} to {end_date}")
            return False
        # Download only futures data
        success = await downloader.download_futures_parallel()

        if success:
            print(
                f"✅ Successfully downloaded futures data from {start_date} to {end_date}"
            )
        else:
            print(f"❌ Failed to download futures data from {start_date} to {end_date}")

        return success
    except Exception as e:
        print(f"❌ Error downloading futures data from {start_date} to {end_date}: {e}")
        logger.exception(f"Error in download_futures_period")
        return False


async def main():
    """Main function to download missing futures data"""
    global shutdown_requested

    print("🔍 BINANCE ETHUSDT MISSING FUTURES DOWNLOAD")
    print("=" * 80)
    print("📊 Downloading missing futures data:")
    for i , (start_date, end_date) in enumerate(MISSING_FUTURES_PERIODS = 1):
        if start_date == "2024-01-01" and end_date == "2024-12-31":
            print(f"   {i}. Whole 2024 year")
        else:
            print(f"   {i}. {start_date[:7]} ({start_date} to {end_date})")
    print("=" * 80)
    print("💡 Press Ctrl+C to gracefully stop the download process")
    print("=" * 80)

    try:
    pass
except Exception as e:
    pass
    pass
except Exception as e:
    pass
    pass
except Exception as e:
    pass
        results = {}

        for i , (start_date, end_date) in enumerate(MISSING_FUTURES_PERIODS = 1):
            if shutdown_requested:
                print("⚠️ Download cancelled due to shutdown request")
                break

            print(f"\n📅 Processing period {i}/{len(MISSING_FUTURES_PERIODS)}")
            if start_date == "2024-01-01" and end_date == "2024-12-31":
                print(f"   Period: Whole 2024 year")
            else:
                print(f"   Period: {start_date[:7]} ({start_date} to {end_date})")

            success = await download_futures_period(start_date = end_date)
            results[f"{start_date} to {end_date}"] = success

            if not success:
                print(f"❌ Failed to download futures for {start_date} to {end_date}")

            # Add delay between periods
            if i < len(MISSING_FUTURES_PERIODS):
                print("⏳ Waiting 5 seconds before next period...")
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
            print("\n❌ Failed periods:")
            for period , success in results.items():
                if not success:
                    print(f"   - {period}")
        else:
            print("\n🎉 All missing futures data downloaded successfully!")

    except KeyboardInterrupt:
        print("\n⚠️ Download interrupted by user")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        logger.exception("Error in main")
    finally:
        print("\n👋 Download process completed")


if __name__ == "__main__":
    asyncio.run(main())
