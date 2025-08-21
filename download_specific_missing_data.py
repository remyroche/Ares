#!/usr/bin/env python3
"""
Download Specific Missing Data for Binance ETHUSDT

Based on user requirements:
1. Futures: 2023 and 2025-01 to 2025-04
2. Aggtrades: Since 2025-02-22 and between 2025-01-01 and 2025-02-17
"""

import asyncio
import logging
import signal
import sys
import time
from datetime import datetime
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


def signal_handler(signum, frame):
    """Handle interrupt signals gracefully"""
    global shutdown_requested
    print(f"\n⚠️ Received signal {signum}. Gracefully shutting down...")
    shutdown_requested = True


# Register signal handlers
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


async def download_futures_2023():
    """Download futures data for 2023"""
    if shutdown_requested:
        print("⚠️ Download cancelled due to shutdown request")
        return False

    print("🚀 Downloading futures data for 2023")
    print("=" * 80)

    try:
        config = DownloadConfig(
            symbol="ETHUSDT",
            exchange="BINANCE",
            interval="1m",
            lookback_years=2,
            start_date_str="2023-01-01",
            end_date_str="2023-12-31",
        )

        downloader = OptimizedDataDownloader(config)
        success = await downloader.run_optimized_download()

        if success:
            print("✅ Successfully downloaded futures data for 2023")
        else:
            print("❌ Failed to download futures data for 2023")

        return success
    except Exception as e:
        print(f"❌ Error downloading futures data for 2023: {e}")
        logger.exception("Error in download_futures_2023")
        return False


async def download_futures_2025_01_to_04():
    """Download futures data for 2025-01 to 2025-04"""
    if shutdown_requested:
        print("⚠️ Download cancelled due to shutdown request")
        return False

    print("🚀 Downloading futures data for 2025-01 to 2025-04")
    print("=" * 80)

    try:
        config = DownloadConfig(
            symbol="ETHUSDT",
            exchange="BINANCE",
            interval="1m",
            lookback_years=2,
            start_date_str="2025-01-01",
            end_date_str="2025-04-30",
        )

        downloader = OptimizedDataDownloader(config)
        success = await downloader.run_optimized_download()

        if success:
            print("✅ Successfully downloaded futures data for 2025-01 to 2025-04")
        else:
            print("❌ Failed to download futures data for 2025-01 to 2025-04")

        return success
    except Exception as e:
        print(f"❌ Error downloading futures data for 2025-01 to 2025-04: {e}")
        logger.exception("Error in download_futures_2025_01_to_04")
        return False


async def download_aggtrades_since_2025_02_22():
    """Download aggtrades data since 2025-02-22 (the last file we have)"""
    if shutdown_requested:
        print("⚠️ Download cancelled due to shutdown request")
        return False

    print("🚀 Downloading aggtrades data since 2025-02-22")
    print("=" * 80)

    try:
        # Start from 2025-02-23 (day after the last file we have)
        config = DownloadConfig(
            symbol="ETHUSDT",
            exchange="BINANCE",
            interval="1m",
            lookback_years=2,
            start_date_str="2025-02-23",
            end_date_str=datetime.now().strftime("%Y-%m-%d"),  # Up to today
        )

        downloader = OptimizedDataDownloader(config)
        success = await downloader.run_optimized_download()

        if success:
            print("✅ Successfully downloaded aggtrades data since 2025-02-22")
        else:
            print("❌ Failed to download aggtrades data since 2025-02-22")

        return success
    except Exception as e:
        print(f"❌ Error downloading aggtrades data since 2025-02-22: {e}")
        logger.exception("Error in download_aggtrades_since_2025_02_22")
        return False


async def download_aggtrades_2025_01_01_to_2025_02_17():
    """Download aggtrades data between 2025-01-01 and 2025-02-17"""
    if shutdown_requested:
        print("⚠️ Download cancelled due to shutdown request")
        return False

    print("🚀 Downloading aggtrades data for 2025-01-01 to 2025-02-17")
    print("=" * 80)

    try:
        config = DownloadConfig(
            symbol="ETHUSDT",
            exchange="BINANCE",
            interval="1m",
            lookback_years=2,
            start_date_str="2025-01-01",
            end_date_str="2025-02-17",
        )

        downloader = OptimizedDataDownloader(config)
        success = await downloader.run_optimized_download()

        if success:
            print(
                "✅ Successfully downloaded aggtrades data for 2025-01-01 to 2025-02-17",
            )
        else:
            print("❌ Failed to download aggtrades data for 2025-01-01 to 2025-02-17")

        return success
    except Exception as e:
        print(f"❌ Error downloading aggtrades data for 2025-01-01 to 2025-02-17: {e}")
        logger.exception("Error in download_aggtrades_2025_01_01_to_2025_02_17")
        return False


async def main():
    """Main function to download specific missing data"""
    global shutdown_requested

    print("🔍 BINANCE ETHUSDT SPECIFIC MISSING DATA DOWNLOAD")
    print("=" * 80)
    print("📊 Based on user requirements, downloading:")
    print("   • Futures: 2023-01-01 to 2023-12-31")
    print("   • Futures: 2025-01-01 to 2025-04-30")
    print("   • Aggtrades: 2025-02-23 to today (since last file)")
    print("   • Aggtrades: 2025-01-01 to 2025-02-17")
    print("=" * 80)
    print("💡 Press Ctrl+C to gracefully stop the download process")
    print("=" * 80)

    results = {}

    try:
        # Download futures data
        print("\n📊 PHASE 1: Downloading missing futures data")
        print("-" * 60)

        if not shutdown_requested:
            results["futures_2023"] = await download_futures_2023()

        if not shutdown_requested:
            results["futures_2025_01_to_04"] = await download_futures_2025_01_to_04()

        # Download aggtrades data
        if not shutdown_requested:
            print("\n📊 PHASE 2: Downloading missing aggtrades data")
            print("-" * 60)

            results[
                "aggtrades_since_2025_02_22"
            ] = await download_aggtrades_since_2025_02_22()

        if not shutdown_requested:
            results[
                "aggtrades_2025_01_01_to_2025_02_17"
            ] = await download_aggtrades_2025_01_01_to_2025_02_17()

    except KeyboardInterrupt:
        print("\n⚠️ Download interrupted by user")
        shutdown_requested = True
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        logger.exception("Unexpected error in main")
        return False

    # Summary
    print("\n📊 DOWNLOAD SUMMARY")
    print("=" * 80)

    if shutdown_requested:
        print("⚠️ Download process was interrupted")

    success_count = sum(1 for success in results.values() if success)
    total_count = len(results)

    for task, success in results.items():
        status = "✅ SUCCESS" if success else "❌ FAILED"
        print(f"   {task}: {status}")

    print(f"\n📈 Overall: {success_count}/{total_count} downloads successful")

    if success_count == total_count and not shutdown_requested:
        print("🎉 All missing data downloaded successfully!")
        return True
    if shutdown_requested:
        print("⚠️ Download process was interrupted - some data may be incomplete")
        return False
    print("⚠️ Some downloads failed. Check logs for details.")
    return False


if __name__ == "__main__":
    start_time = time.time()
    try:
        success = asyncio.run(main())
        end_time = time.time()
        duration = end_time - start_time
        print(f"\n⏱️ Total execution time: {duration:.2f} seconds")
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n⚠️ Process interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        logger.exception("Fatal error")
        sys.exit(1)
