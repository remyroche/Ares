#!/usr/bin/env python3
# scripts/run_enhanced_training.py

"""
Enhanced Training Runner for Large Datasets

This script provides a simple interface to run enhanced training with all
optimizations enabled for handling large datasets efficiently.

Usage:
    python scripts/run_enhanced_training.py --symbol ETHUSDT --lookback 730
    python scripts/run_enhanced_training.py --demo  # Run efficiency demo
    python scripts/run_enhanced_training.py --checkpoint  # Run checkpoint demo
"""

import argparse
import asyncio
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.config import CONFIG
from src.database.sqlite_manager import SQLiteManager
from src.training.enhanced_training_manager import EnhancedTrainingManager
from src.utils.logger import system_logger
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


async def run_enhanced_training(symbol: str, lookback_days: int, timeframe: str = "1h"):
    """Run enhanced training with all optimizations."""
    logger = system_logger.getChild("EnhancedTrainingRunner")

    logger.info("🚀 Starting Enhanced Training with Large Dataset Optimizations")
    logger.info(f"Symbol: {symbol}")
    logger.info(f"Lookback days: {lookback_days}")
    logger.info(f"Timeframe: {timeframe}")

    # Initialize database
    db_manager = SQLiteManager()
    await db_manager.initialize()

    # Initialize enhanced training manager
    training_manager = EnhancedTrainingManager(db_manager)

    # Run training
    session_id = await training_manager.run_full_training(
        symbol=symbol,
        exchange_name="BINANCE",
        timeframe=timeframe,
        lookback_days_override=lookback_days,
    )

    if session_id:
        logger.info(f"✅ Training completed successfully! Session ID: {session_id}")

        # Display efficiency stats
        stats = training_manager.get_efficiency_stats()
        logger.info("📊 Final Efficiency Statistics:")
        for key, value in stats.items():
            logger.info(f"  {key}: {value}")

        return True
    print(failed("❌ Training failed!")))
    return False


async def run_efficiency_demo():
    """Run efficiency features demonstration."""
    logger = system_logger.getChild("EfficiencyDemo")

    logger.info("🔧 Running Efficiency Features Demonstration")

    # Initialize components
    db_manager = SQLiteManager()
    await db_manager.initialize()

    training_manager = EnhancedTrainingManager(db_manager)

    # Initialize efficiency optimizer
    symbol = "ETHUSDT"
    timeframe = "1h"
    await training_manager.initialize_efficiency_optimizer(symbol, timeframe)

    if training_manager.efficiency_optimizer:
        # Demonstrate memory optimization
        logger.info("📊 Memory optimization demo:")
        import numpy as np
        import pandas as pd

        # Create test data
        test_data = pd.DataFrame(
            {
                "timestamp": pd.date_range("2022-01-01", periods=50000, freq="1H"),
                "open": np.random.randn(50000) * 100 + 2000,
                "high": np.random.randn(50000) * 100 + 2100,
                "low": np.random.randn(50000) * 100 + 1900,
                "close": np.random.randn(50000) * 100 + 2000,
                "volume": np.random.randint(1000, 10000, 50000),
            },
        )

        initial_memory = test_data.memory_usage(deep=True).sum()
        logger.info(f"  Initial memory usage: {initial_memory / (1024**2):.2f} MB")

        # Optimize memory
        optimized_data = (
            training_manager.efficiency_optimizer.optimize_dataframe_memory(test_data)
        )

        final_memory = optimized_data.memory_usage(deep=True).sum()
        reduction = (initial_memory - final_memory) / initial_memory * 100
        logger.info(f"  Memory reduction: {reduction:.1f}%")

        # Demonstrate segmentation
        logger.info("📊 Data segmentation demo:")
        segments = training_manager.efficiency_optimizer.segment_data_by_time(
            test_data,
            segment_days=30,
        )
        logger.info(f"  Created {len(segments)} segments")

        for i, (start_date, end_date, segment_data) in enumerate(segments[:3]):
            logger.info(
                f"  Segment {i+1}: {start_date} to {end_date}, {len(segment_data)} rows",
            )

        # Demonstrate database stats
        logger.info("📊 Database statistics:")
        stats = training_manager.efficiency_optimizer.get_database_stats()
        for key, value in stats.items():
            logger.info(f"  {key}: {value}")

    logger.info("✅ Efficiency features demonstration completed")


async def run_checkpoint_demo():
    """Run checkpoint and resume demonstration."""
    logger = system_logger.getChild("CheckpointDemo")

    logger.info("🔧 Running Checkpoint and Resume Demonstration")

    # Initialize components
    db_manager = SQLiteManager()
    await db_manager.initialize()

    training_manager = EnhancedTrainingManager(db_manager)

    # Initialize efficiency optimizer
    symbol = "ETHUSDT"
    timeframe = "1h"
    await training_manager.initialize_efficiency_optimizer(symbol, timeframe)

    if training_manager.efficiency_optimizer:
        # Create a test checkpoint
        logger.info("📝 Creating test checkpoint...")
        test_metadata = {
            "stage": "data_collection",
            "progress": 0.5,
            "last_processed_date": "2023-06-15",
            "features_computed": 150,
        }

        training_manager.efficiency_optimizer.create_processing_checkpoint(
            f"training_{symbol}",
            test_metadata,
        )

        # Retrieve the checkpoint
        logger.info("📖 Retrieving checkpoint...")
        checkpoint = training_manager.efficiency_optimizer.get_latest_checkpoint(
            f"training_{symbol}",
        )

        if checkpoint:
            logger.info(f"  Checkpoint timestamp: {checkpoint['timestamp']}")
            logger.info(f"  Checkpoint metadata: {checkpoint['metadata']}")

        # Demonstrate resume functionality
        logger.info("🔄 Testing resume functionality...")
        resume_success = await training_manager.resume_training_from_checkpoint(symbol)
        logger.info(f"  Resume success: {resume_success}")

    logger.info("✅ Checkpoint and resume demonstration completed")


def main():
    """Main function with command line interface."""
    parser = argparse.ArgumentParser(
        description="Enhanced Training Runner for Large Datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run training with 2 years of data
  python scripts/run_enhanced_training.py --symbol ETHUSDT --lookback 730

  # Run with custom timeframe
  python scripts/run_enhanced_training.py --symbol BTCUSDT --lookback 365 --timeframe 4h

  # Run efficiency demo
  python scripts/run_enhanced_training.py --demo

  # Run checkpoint demo
  python scripts/run_enhanced_training.py --checkpoint
        """,
    )

    parser.add_argument("--symbol", default="ETHUSDT", help="Trading symbol")
    parser.add_argument(
        "--lookback",
        type=int,
        default=730,
        help="Lookback days (default: 730 = 2 years)",
    )
    parser.add_argument("--timeframe", default="1h", help="Timeframe (default: 1h)")
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run efficiency features demonstration",
    )
    parser.add_argument(
        "--checkpoint",
        action="store_true",
        help="Run checkpoint/resume demonstration",
    )
    parser.add_argument(
        "--blank-mode",
        action="store_true",
        help="Run in blank mode with limited data and parameters for quick testing",
    )

    args = parser.parse_args()

    # Update configuration based on command line arguments
    CONFIG["trading_symbol"] = args.symbol
    CONFIG["trading_interval"] = args.timeframe

    # Handle blank mode - use limited data and parameters
    if args.blank_mode:
        CONFIG["MODEL_TRAINING"]["data_retention_days"] = (
            args.lookback
        )  # Use command line lookback
        CONFIG["MODEL_TRAINING"]["hyperparameter_tuning"]["max_trials"] = (
            3  # Reduced trials
        )
        CONFIG["MODEL_TRAINING"]["coarse_hpo"]["n_trials"] = 3  # Reduced trials
        CONFIG["BLANK_TRAINING_MODE"] = True
        print(
            f"🔧 Running in BLANK MODE with limited data ({args.lookback} days) and reduced optimization trials",
        )
    else:
        CONFIG["MODEL_TRAINING"]["data_retention_days"] = args.lookback

    # Run appropriate function
    if args.demo:
        asyncio.run(run_efficiency_demo())
    elif args.checkpoint:
        asyncio.run(run_checkpoint_demo())
    else:
        success = asyncio.run(
            run_enhanced_training(
                symbol=args.symbol,
                lookback_days=args.lookback,
                timeframe=args.timeframe,
            ),
        )
        sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
