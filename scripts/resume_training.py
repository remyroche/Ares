#!/usr/bin/env python3
"""
Resume Training Script for Ares Trading Bot

This script first ensures that all historical data is up-to-date by running
the data collection and consolidation step. It then resumes the main training
pipeline from Step 2 (Preliminary Optimization).

This is useful if a training run failed after the data collection step, or if
you want to retrain on the latest data without starting from scratch.

Usage:
    python scripts/resume_training.py <SYMBOL> [EXCHANGE]
    Example: python scripts/resume_training.py BTCUSDT BINANCE
"""

import asyncio
import sys
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.training.training_manager import TrainingManager
from src.utils.logger import setup_logging, system_logger
from src.database.sqlite_manager import SQLiteManager
from src.config import CONFIG
from src.training.steps.step1_data_collection import run_step as run_data_collection_step


async def main():
    """Main function to run the resumed training pipeline."""
    setup_logging()
    logger = system_logger.getChild("ResumeTraining")

    if len(sys.argv) < 2:
        logger.error("A symbol argument is required.")
        print(__doc__)
        sys.exit(1)

    symbol = sys.argv[1].upper()
    exchange = sys.argv[2].upper() if len(sys.argv) > 2 else "BINANCE"

    logger.info(f"Attempting to resume training for {symbol} on {exchange}...")

    # Step 1: Consolidate and update data cache before resuming.
    # This ensures the latest data is downloaded and the consolidated .pkl file is ready.
    logger.info("Step 1: Consolidating and updating data cache before resuming...")
    data_dir = "data/training"
    min_data_points = str(CONFIG["MODEL_TRAINING"]["min_data_points"])

    # Run data collection step WITHOUT downloading new data.
    klines_df, _, _ = await run_data_collection_step(
        symbol=symbol,
        exchange_name=exchange,
        min_data_points=min_data_points,
        data_dir=data_dir,
        download_new_data=False,
    )

    if klines_df is None:
        logger.error("Data consolidation step failed. Cannot resume training.")
        sys.exit(1)

    logger.info("Data consolidation successful. Proceeding with training pipeline from Step 2.")

    db_manager = None
    try:
        db_manager = SQLiteManager()
        await db_manager.initialize()

        training_manager = TrainingManager(db_manager)

        # This will now start from Step 2, as Step 1 (data part) is complete.
        run_id = await training_manager.resume_training_pipeline(symbol, exchange)

        if run_id:
            logger.info(f"Resumed training pipeline completed successfully for {symbol}. MLflow Run ID: {run_id}")
        else:
            logger.error(f"Resumed training pipeline failed for {symbol}.")
            sys.exit(1)
    finally:
        if db_manager:
            await db_manager.close()

if __name__ == "__main__":
    asyncio.run(main())