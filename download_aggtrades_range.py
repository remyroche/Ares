#!/usr/bin/env python3
"""
Download Aggtrades Range: 2025-05-01 to 2025-08-18

This script downloads aggtrades data for the gap between existing files.
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


async def download_aggtrades_range(...) -> ...:
    """..."""
    passif shutdown_requested:
    passprint("⚠️ Download cancelled due to shutdown request")
        return False

    print(f"🚀 Downloading aggtrades data from {start_date} to {end_date}")
    print("=" * 80)

    try:
    passpass  # TODO: Add proper exception handling
except Exception as e:
    passpasspasspasspasspasspasspass  # TODO: Add proper exception handling
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
    pass"""Main function to download aggtrades range"""
    global shutdown_requested

    print("🔍 BINANCE ETHUSDT AGGTRADES RANGE DOWNLOAD")
    print("=" * 80)
    print("📊 Downloading aggtrades from 2025-05-01 to 2025-08-18")
    print("   (Gap between existing files)")
    print("=" * 80)
    print("💡 Press Ctrl+C to gracefully stop the download process")
    print("=" * 80)

    try:
    passsuccess = await download_aggtrades_range("2025-05-01", "2025-08-18")

        if success:
    passprint("\n🎉 Aggtrades range downloaded successfully!")
        else:
    passprint("\n⚠️ Download failed. Check the logs above.")

    except KeyboardInterrupt:
    passpassprint("\n⚠️ Download interrupted by user")
    except Exception as e:
    passpasspasspasspasspasspassprint(f"\n❌ Unexpected error: {e}")
        logger.exception("Unexpected error in main")
    finally:
    passprint("\n🏁 Download process completed")


if __name__ == "__main__":
    passasyncio.run(main())
