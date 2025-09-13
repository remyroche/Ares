#!/usr/bin/env python3
"""
Script to download complete 3 years of ETHUSDT data
This will backup existing data and download fresh 3-year dataset
"""

import asyncio
import sys
import os
import shutil
from datetime import datetime
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

try:
    from src.utils.data.historical_data_downloader import HistoricalDataDownloader
    from src.utils.logger import system_logger

    async def download_full_3years():
        """Download complete 3 years of ETHUSDT data."""
        logger = system_logger.getChild("Full_ETHUSDT_Downloader")

        data_dir = Path("historical_data/binance/ethusdt/raw")
        backup_dir = Path("historical_data/binance/ethusdt/raw_backup")

        # Check if data exists
        if data_dir.exists() and list(data_dir.glob("*.parquet")):
            logger.info(f"📦 Backing up existing data to {backup_dir}")

            # Create backup directory
            backup_dir.mkdir(parents=True, exist_ok=True)

            # Move existing files to backup
            for file in data_dir.glob("*.parquet"):
                shutil.move(str(file), str(backup_dir / file.name))

            logger.info("✅ Existing data backed up")

        logger.info("🚀 Starting fresh 3-year ETHUSDT data download...")

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
            logger.info("✅ Full 3-year ETHUSDT data download completed successfully!")

            # Get summary
            summary = downloader.get_data_summary("ETHUSDT")
            logger.info(f"📊 Download summary: {summary}")

            # Verify we have 3 years
            if summary and 'date_range' in summary and summary['date_range']:
                start_date, end_date = summary['date_range']
                days_diff = (end_date - start_date).days
                years_diff = days_diff / 365

                logger.info(f"📅 Data spans {days_diff} days ({years_diff:.1f} years)")

                if years_diff >= 2.8:  # Allow some tolerance
                    logger.info("🎉 SUCCESS: Downloaded approximately 3 years of data!")
                else:
                    logger.warning(f"⚠️ Only {years_diff:.1f} years downloaded. Expected ~3 years.")

        else:
            logger.error("❌ Full ETHUSDT data download failed!")

            # Restore backup if download failed
            if backup_dir.exists():
                logger.info("🔄 Restoring backup data...")
                for file in backup_dir.glob("*.parquet"):
                    shutil.move(str(file), str(data_dir / file.name))
                logger.info("✅ Backup restored")

            return False

        return True

    if __name__ == "__main__":
        asyncio.run(download_full_3years())

except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Please ensure all dependencies are installed:")
    print("pip install -r requirements_data_collection.txt")
    sys.exit(1)
