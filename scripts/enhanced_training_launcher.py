#!/usr/bin/env python3
# scripts/enhanced_training_launcher.py

"""
Enhanced Training Launcher for Large Datasets

This script demonstrates how to use the EnhancedTrainingManager to efficiently
handle large datasets (2+ years of historical data) with various optimization strategies.

Key Features:
1. Intelligent caching with SQLite storage
2. Time-based data segmentation
3. Memory-efficient processing
4. Database-backed feature storage
5. Checkpoint and resume capabilities
"""

import asyncio
import sys
import time
from datetime import datetime
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.config import CONFIG
from src.database.sqlite_manager import SQLiteManager
from src.training.enhanced_training_manager import EnhancedTrainingManager
from src.utils.logger import system_logger


async def main():
    """Main function to demonstrate enhanced training with large datasets."""
    logger = system_logger.getChild("EnhancedTrainingLauncher")

    logger.info("=" * 80)
    logger.info("🚀 ENHANCED TRAINING LAUNCHER FOR LARGE DATASETS")
    logger.info("=" * 80)
    logger.info(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Configuration
    symbol = "ETHUSDT"
    exchange_name = "BINANCE"
    timeframe = "1h"
    lookback_days = 730  # 2 years

    logger.info("Configuration:")
    logger.info(f"  Symbol: {symbol}")
    logger.info(f"  Exchange: {exchange_name}")
    logger.info(f"  Timeframe: {timeframe}")
    logger.info(f"  Lookback days: {lookback_days}")
    logger.info(
        f"  Efficiency optimizations: {CONFIG.get('ENHANCED_TRAINING', {}).get('enable_efficiency_optimizations', True)}",
    )

    # Initialize database manager
    logger.info("🔧 Initializing database manager...")
    db_manager = SQLiteManager()
    await db_manager.initialize()

    # Initialize enhanced training manager
    logger.info("🔧 Initializing enhanced training manager...")
    training_manager = EnhancedTrainingManager(db_manager)

    # Display efficiency configuration
    logger.info("📊 Efficiency Configuration:")
    enhanced_config = CONFIG.get("ENHANCED_TRAINING", {})
    for key, value in enhanced_config.items():
        logger.info(f"  {key}: {value}")

    # Check system resources
    logger.info("💻 System Resources:")
    import psutil

    # Memory info
    memory = psutil.virtual_memory()
    logger.info(f"  Total RAM: {memory.total / (1024**3):.1f} GB")
    logger.info(f"  Available RAM: {memory.available / (1024**3):.1f} GB")
    logger.info(f"  Memory usage: {memory.percent:.1f}%")

    # Disk info
    disk = psutil.disk_usage(".")
    logger.info(f"  Total disk: {disk.total / (1024**3):.1f} GB")
    logger.info(f"  Available disk: {disk.free / (1024**3):.1f} GB")
    logger.info(f"  Disk usage: {disk.percent:.1f}%")

    # CPU info
    cpu_count = psutil.cpu_count()
    logger.info(f"  CPU cores: {cpu_count}")

    # Check if we have enough resources
    if memory.percent > 90:
        logger.warning(
            "⚠️ High memory usage detected. Consider closing other applications.",
        )

    if disk.percent > 90:
        logger.warning("⚠️ High disk usage detected. Consider freeing up space.")

    # Start enhanced training
    logger.info("🚀 Starting enhanced training pipeline...")
    start_time = time.time()

    try:
        # Run the enhanced training pipeline
        session_id = await training_manager.run_full_training(
            symbol=symbol,
            exchange_name=exchange_name,
            timeframe=timeframe,
            lookback_days_override=lookback_days,
        )

        if session_id:
            total_duration = time.time() - start_time
            logger.info("✅ Enhanced training completed successfully!")
            logger.info(f"  Session ID: {session_id}")
            logger.info(
                f"  Total duration: {total_duration:.2f} seconds ({total_duration/3600:.2f} hours)",
            )

            # Display efficiency stats
            efficiency_stats = training_manager.get_efficiency_stats()
            logger.info("📊 Efficiency Statistics:")
            for key, value in efficiency_stats.items():
                logger.info(f"  {key}: {value}")

        else:
            logger.error("❌ Enhanced training failed!")
            return 1

    except KeyboardInterrupt:
        logger.info("⏹️ Training interrupted by user")
        return 1

    except Exception as e:
        logger.error(f"❌ Training failed with error: {e}", exc_info=True)
        return 1

    logger.info("=" * 80)
    logger.info("✅ Enhanced training launcher completed")
    logger.info("=" * 80)

    return 0


async def demonstrate_efficiency_features():
    """Demonstrate individual efficiency features."""
    logger = system_logger.getChild("EfficiencyDemo")

    logger.info("🔧 Demonstrating efficiency features...")

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

        # Create a large test DataFrame
        test_data = pd.DataFrame(
            {
                "timestamp": pd.date_range("2022-01-01", periods=100000, freq="1H"),
                "open": np.random.randn(100000) * 100 + 2000,
                "high": np.random.randn(100000) * 100 + 2100,
                "low": np.random.randn(100000) * 100 + 1900,
                "close": np.random.randn(100000) * 100 + 2000,
                "volume": np.random.randint(1000, 10000, 100000),
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


async def demonstrate_checkpoint_resume():
    """Demonstrate checkpoint and resume functionality."""
    logger = system_logger.getChild("CheckpointDemo")

    logger.info("🔧 Demonstrating checkpoint and resume functionality...")

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


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Enhanced Training Launcher for Large Datasets",
    )
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
    parser.add_argument("--symbol", default="ETHUSDT", help="Trading symbol")
    parser.add_argument("--timeframe", default="1h", help="Timeframe")
    parser.add_argument("--lookback", type=int, default=730, help="Lookback days")

    args = parser.parse_args()

    if args.demo:
        asyncio.run(demonstrate_efficiency_features())
    elif args.checkpoint:
        asyncio.run(demonstrate_checkpoint_resume())
    else:
        # Update configuration based on command line arguments
        CONFIG["trading_symbol"] = args.symbol
        CONFIG["trading_interval"] = args.timeframe
        CONFIG["MODEL_TRAINING"]["data_retention_days"] = args.lookback

        exit_code = asyncio.run(main())
        sys.exit(exit_code)
