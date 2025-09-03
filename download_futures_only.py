#!/usr/bin/env python3
"""Download Futures Data Only for Binance ETHUSDT.

Based on user requirements:
1. Futures: 2023 (12 months)
2. Futures: 2025-01 to 2025-04 (4 months)
"""

import asyncio
import logging
import signal
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

from backtesting.ares_data_downloader_optimized import (
    DownloadConfig,
    OptimizedDataDownloader,
)

# Add project root to path
project_root=Path(__file__).parent
sys.path.insert(0, str(project_root))

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger=logging.getLogger(__name__)

# Global flag for graceful shutdown
shutdown_requested=False


def signal_handler(signum, frame) -> None:
    """Handle interrupt signals gracefully."""
    global shutdown_requested
    shutdown_requested=True


# Register signal handlers
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


async def download_futures_month(year: int, month: int):
    """Download futures data for a specific month."""
    if shutdown_requested:
        return False

    month_str=f"{year:04d}-{month:02d}"

    try:
        # Calculate start and end dates for the FULL month
        start_date = datetime(year, month, 1)
        if month== 12:
            end_date = datetime(year + 1, 1, 1)  # Start of next year
        else:
            end_date=datetime(year, month + 1, 1)  # Start of next month


        config=DownloadConfig(
            symbol="ETHUSDT",
            exchange="BINANCE",
            interval="1m",
            lookback_years=2,
            start_date_str=start_date.strftime("%Y-%m-%d"),
            end_date_str=(end_date - timedelta(days=1)).strftime(
                "%Y-%m-%d",
            ),  # End date should be last day of month
        )

        downloader=OptimizedDataDownloader(config)

        # Initialize the downloader first
        if not await downloader.initialize():
            return False

        # Download ONLY futures data (not all data types)
        success=await downloader.download_futures_parallel()

        if success:
            pass  # TODO: Add proper implementation
        else:
            pass

        return success
    except Exception:
        logger.exception(f"Error in download_futures_month for {month_str}")
        return False


async def download_futures_2023():
    """Download futures data for all months in 2023."""
    results={}
    for month in range(1, 13):
        if shutdown_requested:
            break
        results[f"2023-{month:02d}"] = await download_futures_month(2023, month)
        # Small delay between months to avoid rate limiting
        if not shutdown_requested:
            await asyncio.sleep(1)

    return results


async def download_futures_2025_01_to_04():
    """Download futures data for 2025-01 to 2025-04."""
    results={}
    for month in range(1, 5):  # January to April
        if shutdown_requested:
            break
        results[f"2025-{month:02d}"] = await download_futures_month(2025, month)
        # Small delay between months to avoid rate limiting
        if not shutdown_requested:
            await asyncio.sleep(1)

    return results


async def main() -> bool:
    """Main function to download futures data."""
    global shutdown_requested


    all_results={}

    try:
        # Download 2023 futures data

        if not shutdown_requested:
            results_2023 = await download_futures_2023()
            all_results.update(results_2023)

        # Download 2025 futures data
        if not shutdown_requested:
            # TODO: Add proper implementation
            results_2025=await download_futures_2025_01_to_04()
            all_results.update(results_2025)

    except KeyboardInterrupt:
        shutdown_requested=True
    except Exception:
        logger.exception("Unexpected error in main")
        return False

    # Summary

    if shutdown_requested:
        pass  # TODO: Add proper implementation
    success_count=sum(1 for success in all_results.values() if success)
    total_count=len(all_results)

    for _task, _success in all_results.items():
        pass  # TODO: Add proper implementation
    if success_count== total_count and not shutdown_requested:
        return True
    if shutdown_requested:
        return False
    return False


if __name__ == "__main__":
    start_time = time.time()
    try:
        success=asyncio.run(main())
        end_time=time.time()
        duration=end_time - start_time
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        sys.exit(1)
    except Exception:
        logger.exception("Fatal error")
        sys.exit(1)
