#!/usr/bin/env python3
"""
Script to run a 'blank' training pipeline with minimal optimization parameters.

This is intended for testing the end-to-end functionality of the training pipeline
without performing a full, computationally intensive hyperparameter optimization.
It overrides CONFIG values for optimization trials and data lookback to be very small.

Usage:
    python scripts/blank_training_run.py --symbol BTCUSDT --exchange BINANCE
"""

import asyncio
import sys
import time
import traceback
from pathlib import Path
from datetime import datetime

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import necessary modules
from src.database.sqlite_manager import SQLiteManager
from src.training.training_manager import TrainingManager
from src.utils.logger import setup_logging, system_logger
from src.config import CONFIG  # Import the global CONFIG dictionary
import argparse


async def main():
    """
    Main function to orchestrate the blank training run.
    It temporarily modifies CONFIG parameters for quick execution.
    """
    start_time = time.time()
    setup_logging()
    logger = system_logger.getChild("BlankTrainingRun")
    
    logger.info("=" * 80)
    logger.info("🚀 BLANK TRAINING RUN INITIALIZATION")
    logger.info("=" * 80)
    logger.info(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Python version: {sys.version}")
    logger.info(f"Working directory: {Path.cwd()}")
    logger.info(f"Script path: {Path(__file__).absolute()}")

    parser = argparse.ArgumentParser(
        description="Run a 'blank' training pipeline for testing."
    )
    parser.add_argument(
        "--symbol",
        type=str,
        default=CONFIG.get("SYMBOL", "BTCUSDT"),
        help="The trading symbol for the blank run (e.g., BTCUSDT).",
    )
    parser.add_argument(
        "--exchange",
        type=str,
        default=CONFIG.get("EXCHANGE", "BINANCE"),
        help="The exchange for the blank run (e.g., BINANCE).",
    )
    args = parser.parse_args()

    logger.info(f"📋 Command line arguments:")
    logger.info(f"   Symbol: {args.symbol}")
    logger.info(f"   Exchange: {args.exchange}")

    logger.info(
        f"🚀 Starting 'blank' training pipeline for {args.symbol} on {args.exchange}..."
    )
    logger.info("Temporarily overriding CONFIG parameters for quick test run.")
    
    # For a blank run, we reduce the number of optimization trials to make it run faster.
    # NOTE: The data lookback is now passed as a parameter to the training manager.
    CONFIG["MODEL_TRAINING"]["hyperparameter_tuning"]["max_trials"] = (
        5  # For Optuna in base_ensemble
    )
    CONFIG["MODEL_TRAINING"]["coarse_hpo"] = {
        "n_trials": 5
    }  # For CoarseOptimizer's HPO
    
    # Set blank training mode flag for NaN handling
    CONFIG["BLANK_TRAINING_MODE"] = True

    db_manager = None
    try:
        logger.info("🔧 Initializing SQLiteManager...")
        # Initialize SQLiteManager
        db_manager = SQLiteManager()
        await db_manager.initialize()
        logger.info("✅ SQLiteManager initialized successfully")

        logger.info("🔧 Initializing TrainingManager...")
        # Instantiate TrainingManager
        training_manager = TrainingManager(db_manager)
        logger.info("✅ TrainingManager initialized successfully")

        logger.info("🚀 Starting full training pipeline...")
        training_start_time = time.time()
        
        # Run the full training pipeline with a short lookback period for a quick test.
        success = await training_manager.run_full_training(
            args.symbol, args.exchange, lookback_days_override=60
        )
        
        training_duration = time.time() - training_start_time
        logger.info(f"⏱️  Training pipeline completed in {training_duration:.2f} seconds")

        if success:
            logger.info("✅ 'Blank' training pipeline completed successfully!")
            logger.info(f"📊 Training summary:")
            logger.info(f"   Symbol: {args.symbol}")
            logger.info(f"   Exchange: {args.exchange}")
            logger.info(f"   Duration: {training_duration:.2f} seconds")
            logger.info(f"   Status: SUCCESS")
        else:
            logger.error("❌ 'Blank' training pipeline failed.")
            logger.error(f"📊 Training summary:")
            logger.error(f"   Symbol: {args.symbol}")
            logger.error(f"   Exchange: {args.exchange}")
            logger.error(f"   Duration: {training_duration:.2f} seconds")
            logger.error(f"   Status: FAILED")
            sys.exit(1)

    except Exception as e:
        logger.critical("💥 CRITICAL ERROR during blank training run")
        logger.critical(f"Error type: {type(e).__name__}")
        logger.critical(f"Error message: {str(e)}")
        logger.critical("Full traceback:")
        logger.critical(traceback.format_exc())
        
        # Log additional debugging information
        logger.critical(f"📊 Error context:")
        logger.critical(f"   Symbol: {args.symbol}")
        logger.critical(f"   Exchange: {args.exchange}")
        logger.critical(f"   Python path: {sys.path[:3]}...")  # First 3 entries
        logger.critical(f"   Working directory: {Path.cwd()}")
        
        sys.exit(1)
    finally:
        # Ensure database connection is closed
        if db_manager and db_manager.conn:
            logger.info("🔧 Closing database connection...")
            await db_manager.close()
            logger.info("✅ Database connection closed")
        
        total_duration = time.time() - start_time
        logger.info("=" * 80)
        logger.info("🏁 BLANK TRAINING RUN COMPLETION")
        logger.info("=" * 80)
        logger.info(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"Total duration: {total_duration:.2f} seconds")
        logger.info(f"Symbol: {args.symbol}")
        logger.info(f"Exchange: {args.exchange}")
        logger.info("🏁 'Blank' training pipeline finished.")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n⚠️  Training interrupted by user (Ctrl+C)")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Unexpected error in main execution: {e}")
        sys.exit(1)
