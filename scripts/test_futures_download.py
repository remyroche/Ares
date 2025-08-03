#!/usr/bin/env python3
"""
Test script to check the futures data download functionality.
"""

import asyncio
import sys
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.training.steps.data_downloader import download_futures_data
from src.utils.logger import setup_logging, system_logger


async def test_futures_download():
    """Test the futures data download functionality."""
    setup_logging()
    logger = system_logger.getChild("TestFuturesDownload")

    logger.info("Testing futures data download functionality...")

    # The `download_futures_data` function initializes its own ccxt client,
    # so we can pass None for the client argument.
    client = None
    exchange_name = "BINANCE"
    symbol = "ETHUSDT"
    # Set lookback period to 2 years
    lookback_years = 2.0

    logger.info(
        f"Running test for {symbol} on {exchange_name} with a lookback of {lookback_years:.3f} years.",
    )

    success = await download_futures_data(client, exchange_name, symbol, lookback_years)

    logger.info(f"Test completed. Success: {success}")
    if success:
        logger.info(
            f"Check the 'data_cache' directory for files like 'futures_{exchange_name}_{symbol}_YYYY-MM.csv'",
        )


if __name__ == "__main__":
    asyncio.run(test_futures_download())
