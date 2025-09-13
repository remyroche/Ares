#!/usr/bin/env python3
"""
Standalone script to download 3 years of ETHUSDT data
"""

import asyncio
import sys
import os
from datetime import datetime, timedelta
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

try:
    from src.utils.data.historical_data_downloader import HistoricalDataDownloader
    from src.utils.logger import system_logger

    async def download_ethusdt():
        """Download 3 years of ETHUSDT data."""
        logger = system_logger.getChild("ETHUSDT_Downloader")

        logger.info("🚀 Starting ETHUSDT data download...")

        # Initialize downloader
        downloader = HistoricalDataDownloader("historical_data")

        # Download 3 years of data
        success = await downloader.download_historical_klines(
            symbol="ETHUSDT",
            interval="1m",
            years=3,
            api_key="",  # Add your API key if needed
            api_secret=""  # Add your API secret if needed
        )

        if success:
            logger.info("✅ ETHUSDT data download completed successfully!")
            # Get summary
            summary = downloader.get_data_summary("ETHUSDT")
            logger.info(f"📊 Download summary: {summary}")
        else:
            logger.error("❌ ETHUSDT data download failed!")
            return False

        return True

    if __name__ == "__main__":
        asyncio.run(download_ethusdt())

except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Please ensure all dependencies are installed:")
    print("pip install -r requirements_data_collection.txt")
    sys.exit(1)
