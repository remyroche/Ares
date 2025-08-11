#!/usr/bin/env python3
"""
Simple data loader script that uses existing working data collection methods.
"""

import argparse
import asyncio
import subprocess
import sys
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.training.steps.data_downloader import download_all_data_with_consolidation
from src.utils.logger import setup_logging, system_logger
from src.utils.warning_symbols import (
    error,
    warning,
    critical,
    problem,
    failed,
    invalid,
    missing,
    timeout,
    connection_error,
    validation_error,
    initialization_error,
    execution_error,
)


async def load_data(symbol: str, exchange: str, interval: str = "1m"):
    """Load data using existing working methods with consolidation into pkl files."""
    setup_logging()
    logger = system_logger.getChild("SimpleDataLoader")

    logger.info(f"Loading data for {symbol} on {exchange} with interval {interval}")

    try:
        # Step 1: Download raw data
        logger.info("🔄 Step 1: Downloading raw data...")
        success = await download_all_data_with_consolidation(symbol, exchange, interval)

        if not success:
            print(failed("❌ Data downloading failed")))
            return False

        logger.info("✅ Data downloading completed successfully")

        # Step 2: Run consolidation to create pkl files
        logger.info("🔄 Step 2: Consolidating data into pkl files...")

        # Get configuration for data consolidation
        from src.config import CONFIG

        lookback_days = CONFIG.get("lookback_days", 365)
        min_data_points = CONFIG.get("min_data_points", "1000")
        data_dir = CONFIG.get("data_dir", "data")

        # Run the consolidation script
        consolidation_process = subprocess.Popen(
            [
                sys.executable,
                "src/training/steps/step1_data_collection.py",
                symbol,
                exchange,
                min_data_points,
                data_dir,
                str(lookback_days),
                str(CONFIG.get("DATA_CONFIG", {}).get("exclude_recent_days", 0)),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True,
            cwd=project_root,
        )

        # Read output in real-time
        while True:
            output = consolidation_process.stdout.readline()
            if output == "" and consolidation_process.poll() is not None:
                break
            if output:
                print(output.strip())
                logger.info(output.strip())

        consolidation_return_code = consolidation_process.poll()

        if consolidation_return_code == 0:
            logger.info("✅ Data consolidation completed successfully")
            logger.info(
                f"📁 Consolidated data saved to: data/{exchange}_{symbol}_historical_data.pkl",
            )
            return True
        logger.error(
            f"❌ Data consolidation failed with return code: {consolidation_return_code}",
        )
        return False

    except Exception as e:
        print(error("❌ Error during data loading: {e}")))
        return False


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Simple data loader")
    parser.add_argument(
        "--symbol",
        type=str,
        required=True,
        help="Trading symbol (e.g., ETHUSDT)",
    )
    parser.add_argument(
        "--exchange",
        type=str,
        required=True,
        help="Exchange name (e.g., BINANCE)",
    )
    parser.add_argument(
        "--interval",
        type=str,
        default="1m",
        help="K-line interval (default: 1m)",
    )

    args = parser.parse_args()

    success = asyncio.run(load_data(args.symbol, args.exchange, args.interval))

    if success:
        print("✅ Data loading completed successfully")
        sys.exit(0)
    else:
        print(failed("Data loading failed")))
        sys.exit(1)


if __name__ == "__main__":
    main()
