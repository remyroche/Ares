#!/usr/bin/env python3
"""
Script to run a 'blank' training pipeline with minimal optimization parameters.

This is intended for testing the end-to-end functionality of the training pipeline
without performing a full, computationally intensive hyperparameter optimization.
It overrides CONFIG values for optimization trials and data lookback to be very small.

Usage:
    python scripts/blank_training_run.py --symbol BTCUSDT --exchange BINANCE
"""

from src.training.steps.data_preparation_components.aggtrades_data_formatting import (
    AggTradesDataFormatter
)
from src.training.steps.data_preparation_components.training_validation_config import (
    TrainingValidationConfig
)
import os
import traceback
from datetime import datetime
from pathlib import Path
from src.config import CONFIG  # Import the global CONFIG dictionary
from src.utils.logger import setup_logging, system_logger
import argparse
import asyncio
import sys
import time

import pandas as pd
import pickle
from src.database.sqlite_manager import SQLiteManager
from src.training.enhanced_training_manager import EnhancedTrainingManager

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

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
        description="Run a 'blank' training pipeline for testing.",
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

    logger.info("📋 Command line arguments:")
    logger.info(f"   Symbol: {args.symbol}")
    logger.info(f"   Exchange: {args.exchange}")

    logger.info(
        f"🚀 Starting 'blank' training pipeline for {args.symbol} on {args.exchange}...",
    )
    print(
        f"🚀 Starting 'blank' training pipeline for {args.symbol} on {args.exchange}...",
    )
    logger.info("Temporarily overriding CONFIG parameters for quick test run.")
    print("Temporarily overriding CONFIG parameters for quick test run.")

    # For a blank run, we reduce the number of optimization trials to make it run faster.
    # NOTE: The data lookback is now passed as a parameter to the training manager.
    CONFIG["MODEL_TRAINING"]["hyperparameter_tuning"]["max_trials"] = (
        3  # For Optuna in base_ensemble
    )
    CONFIG["MODEL_TRAINING"]["coarse_hpo"] = {
        "n_trials": 3,
    }  # For CoarseOptimizer's HPO

    # Set blank training mode flag for NaN handling
    CONFIG["BLANK_TRAINING_MODE"] = True

    db_manager = None
    try:
        logger.info("🔧 Initializing SQLiteManager...")
        print("🔧 Initializing SQLiteManager...")
        # Initialize SQLiteManager
        db_manager = SQLiteManager({})
        await db_manager.initialize()
        logger.info("✅ SQLiteManager initialized successfully")
        print("✅ SQLiteManager initialized successfully")

        logger.info("🔧 Initializing TrainingManager...")
        print("🔧 Initializing TrainingManager...")
        # Instantiate EnhancedTrainingManager
        training_manager = EnhancedTrainingManager(db_manager)
        logger.info("✅ TrainingManager initialized successfully")
        print("✅ TrainingManager initialized successfully")

        logger.info("🚀 Starting full training pipeline...")
        print("🚀 Starting full training pipeline...")
        training_start_time = time.time()

        # Skip data downloading and use existing data only
        logger.info("⏭️  Skipping data downloading - using existing data only")
        print("⏭️  Skipping data downloading - using existing data only")

        # Check if required data files exist (using actual file patterns)
        klines_filename = f"data_cache/klines_{args.exchange}_{args.symbol}_1m_consolidated_fixed.csv"  # Use fixed 1m consolidated
        agg_trades_filename = f"data_cache/aggtrades_{args.exchange}_{args.symbol}_2025-07-31.csv"  # Use most recent
        futures_filename = (
            f"data_cache/futures_{args.exchange}_{args.symbol}_consolidated.csv"
        )

        # Check if all required data files exist

        required_files = [klines_filename, agg_trades_filename, futures_filename]
        missing_files = [f for f in required_files if not os.path.exists(f)]

        if missing_files:
            print(f"❌ Missing required data files: {missing_files}")
            logger.error(
                "Please run data downloading first or ensure data files exist.",
            )
            print("Expected files:")
            for _file in required_files:
                print(f"  - {_file}")
            sys.exit(1)

        logger.info("✅ All required data files found, proceeding with training")
        print("✅ All required data files found, proceeding with training")

        # Run the training pipeline stages manually = skipping data collection
        logger.info(
            "🚀 Running training pipeline stages manually (skipping data collection)",
        )
        print("🚀 Running training pipeline stages manually (skipping data collection)")
        logger.info(
            f"   Parameters: symbol={args.symbol}, exchange={args.exchange}, lookback_days_override=60",
        )
        print(
            f"   Parameters: symbol={args.symbol}, exchange={args.exchange}, lookback_days_override=60",
        )

        try:
            print("🔄 Starting training pipeline stages...")

            # Stage 1: Setup training environment
            logger.info("🔧 Stage 1: Setting up training environment...")
            print("🔧 Stage 1: Setting up training environment...")
            setup_success = await training_manager._setup_training_environment(
                args.symbol,
                "1m",
            )
            if not setup_success:
                msg = "Stage 1 (Setup) failed"
                raise Exception(msg)
            logger.info("✅ Stage 1 completed successfully")
            print("✅ Stage 1 completed successfully")

            # Use existing data file instead of collecting new data
            csv_data_file = f"data_cache/klines_{args.exchange}_{args.symbol}_1m_consolidated_fixed.csv"
            logger.info(f"📁 Using existing CSV data file: {csv_data_file}")
            print(f"📁 Using existing CSV data file: {csv_data_file}")

            # Use the existing CSV formatting script to ensure proper data format
            logger.info(
                "🔄 Using CSV formatting script to ensure proper data format...",
            )
            print("🔄 Using CSV formatting script to ensure proper data format...")

            # Import and run the CSV formatting script
            auto_reformat_aggtrades_files_for_exchange = AggTradesDataFormatter(
                args.exchange, args.symbol
            )

            # Run the CSV formatting to ensure data files for this exchange/symbol are in correct format
            logger.info("📋 Running CSV format validation and conversion...")
            print("📋 Running CSV format validation and conversion...")
            auto_reformat_aggtrades_files_for_exchange.reformat_aggtrades_files()

            # Create the required pickle file from existing CSV data
            logger.info("🔄 Creating pickle file from existing CSV data...")
            print("🔄 Creating pickle file from existing CSV data...")

            # Read the CSV data (now properly formatted)
            if not os.path.exists(csv_data_file):
                msg = f"CSV data file not found: {csv_data_file}"
                raise Exception(msg)

            klines_df = pd.read_csv(csv_data_file)
            logger.info(
                f"📊 Loaded CSV data: {len(klines_df)} rows = {len(klines_df.columns)} columns",
            )
            print(
                f"📊 Loaded CSV data: {len(klines_df)} rows = {len(klines_df.columns)} columns",
            )

            # Add quality validation checks
            logger.info("🔍 Running data quality validation...")
            print("🔍 Running data quality validation...")

            validation_config = TrainingValidationConfig()
            validate_data_format, validate_data_quality = validation_config.validate_data(
                klines_df
            )

            # Load additional data for validation
            agg_trades_file = (
                f"data_cache/aggtrades_{args.exchange}_{args.symbol}_consolidated.csv"
            )
            futures_file = (
                f"data_cache/futures_{args.exchange}_{args.symbol}_consolidated.csv"
            )

            validation_data = {"klines": klines_df}

            # Load aggregated trades if available
            if os.path.exists(agg_trades_file):
                agg_trades_df = pd.read_csv(agg_trades_file)
                validation_data["agg_trades"] = agg_trades_df
                logger.info(f"📊 Loaded agg trades: {len(agg_trades_df)} rows")
            else:
                logger.warning(
                    "⚠️  Aggregated trades file not found = skipping validation",
                )

            # Load futures data if available
            if os.path.exists(futures_file):
                futures_df = pd.read_csv(futures_file)
                validation_data["futures"] = futures_df
                logger.info(f"📊 Loaded futures: {len(futures_df)} rows")
            else:
                print("⚠️  Futures file not found = skipping validation")

            # Validate data format
            format_valid, format_errors = validate_data_format(validation_data)
            if not format_valid:
                print(f"❌ Data format validation failed: {format_errors}")
                print(f"Data format validation failed: {format_errors}")
                msg = "Data format validation failed"
                raise Exception(msg)
            logger.info("✅ Data format validation passed")
            print("✅ Data format validation passed")

            # Validate data quality
            quality_valid, quality_errors = validate_data_quality(validation_data)
            if not quality_valid:
                print(f"❌ Data quality validation failed: {quality_errors}")
                print(f"Data quality validation failed: {quality_errors}")
                msg = "Data quality validation failed"
                raise Exception(msg)
            logger.info("✅ Data quality validation passed")
            print("✅ Data quality validation passed")

            # Create the pickle file in the expected format
            pickle_data = {"klines": klines_df}
            pickle_file = (
                f"data/training/{args.exchange}_{args.symbol}_collected_data.pkl"
            )
            os.makedirs("data/training", exist_ok=True)

            with open(pickle_file, "wb") as f:
                pickle.dump(pickle_data, f)

            logger.info(f"✅ Created pickle file: {pickle_file}")
            print(f"✅ Created pickle file: {pickle_file}")

            # Use the pickle file for training
            data_file = pickle_file

            # Stage 3: Model Optimization and Training
            logger.info("🔧 Stage 3: Optimizing and training models...")
            print("🔧 Stage 3: Optimizing and training models...")
            training_success = await training_manager._optimize_and_train_models(
                args.symbol,
                data_file,
            )
            if not training_success:
                msg = "Stage 3 (Training) failed"
                raise Exception(msg)
            logger.info("✅ Stage 3 completed successfully")
            print("✅ Stage 3 completed successfully")

            # Stage 4: Model Validation and Testing
            logger.info("🔧 Stage 4: Validating and testing models...")
            print("🔧 Stage 4: Validating and testing models...")
            validation_success = await training_manager._validate_and_test_models(
                data_file,
            )
            if not validation_success:
                msg = "Stage 4 (Validation) failed"
                raise Exception(msg)
            logger.info("✅ Stage 4 completed successfully")
            print("✅ Stage 4 completed successfully")

            # Stage 5: Final Setup and Artifact Management
            session_id = f"{args.symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            run_id = "blank_run_" + session_id
            logger.info("🔧 Stage 5: Finalizing training...")
            print("🔧 Stage 5: Finalizing training...")
            finalize_success = await training_manager._finalize_training(
                session_id,
                run_id,
            )
            if not finalize_success:
                msg = "Stage 5 (Finalization) failed"
                raise Exception(msg)
            logger.info("✅ Stage 5 completed successfully")
            print("✅ Stage 5 completed successfully")

            success = True
            logger.info("✅ All training stages completed successfully")
            print("✅ All training stages completed successfully")

        except Exception as e:
            print(f"❌ Training pipeline failed with exception: {e}")
            print(f"Training pipeline failed with exception: {e}")
            print(f"Exception type: {type(e).__name__}")
            print(f"Exception type: {type(e).__name__}")
            print("Full traceback:")
            print("Full traceback:")

            logger.exception(traceback.format_exc())
            print(traceback.format_exc())
            raise

        training_duration = time.time() - training_start_time
        logger.info(
            f"⏱️  Training pipeline completed in {training_duration:.2f} seconds",
        )

        if success:
            logger.info("✅ 'Blank' training pipeline completed successfully!")
            logger.info("📊 Training summary:")
            logger.info(f"   Symbol: {args.symbol}")
            logger.info(f"   Exchange: {args.exchange}")
            logger.info(f"   Duration: {training_duration:.2f} seconds")
            logger.info("   Status: SUCCESS")
        else:
            print("❌ 'Blank' training pipeline failed.")
            print("📊 Training summary:")
            print(f"   Symbol: {args.symbol}")
            print(f"   Exchange: {args.exchange}")
            print(f"   Duration: {training_duration:.2f} seconds")
            print("   Status: FAILED")
            sys.exit(1)

    except Exception:
        print("💥 CRITICAL ERROR during blank training run")
        print(f"Error type: {type(e).__name__}")
        print(f"Error message: {str(e)}")
        print("Full traceback:")
        logger.critical(traceback.format_exc())

        # Log additional debugging information
        print("📊 Error context:")
        print(f"   Symbol: {args.symbol}")
        print(f"   Exchange: {args.exchange}")
        print(f"   Python path: {sys.path[:3]}...")  # First 3 entries
        print(f"   Working directory: {Path.cwd()}")

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
