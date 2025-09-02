#!/usr/bin/env python3
"""
Script to run a 'blank' training pipeline with minimal optimization parameters.

This is intended for testing end-to-end functionality without a heavy HPO run.
It temporarily overrides CONFIG values for a quick execution path.

Usage:
    python scripts/blank_training_run.py --symbol BTCUSDT --exchange BINANCE
"""

from __future__ import annotations

import argparse
import asyncio
import os
from datetime import datetime
from pathlib import Path
from typing import Any
import sys
import time

import pandas as pd

from src.config import CONFIG
from src.utils.logger import setup_logging, system_logger
from src.utils.error_handler import handle_errors
from src.training.enhanced_training_manager import EnhancedTrainingManager
from src.training.steps.data_preparation_components.aggtrades_data_formatting import (
    auto_reformat_aggtrades_files_for_exchange,
)
from src.training.steps.data_preparation_components.training_validation_config import (
    validate_data_format,
    validate_data_quality,
)

# Ensure project root in path
project_root=Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


@handle_errors(default_return=False, context="blank_training_run_main")
async def main() -> bool:
    """Orchestrate the blank training run with minimal parameters."""
    start_time=time.time()
    setup_logging()
    logger=system_logger.getChild("BlankTrainingRun")

    logger.info("=" * 80)
    logger.info("BLANK TRAINING RUN INITIALIZATION")
    logger.info("=" * 80)
    logger.info(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Python version: {sys.version}")
    logger.info(f"Working directory: {Path.cwd()}")
    logger.info(f"Script path: {Path(__file__).absolute()}")

    parser=argparse.ArgumentParser(
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
    args=parser.parse_args()

    logger.info("Command line arguments:")
    logger.info(f"   Symbol: {args.symbol}")
    logger.info(f"   Exchange: {args.exchange}")

    logger.info(
        f"Starting 'blank' training pipeline for {args.symbol} on {args.exchange}...",
    )

    # Minimal overrides for a quick run
    CONFIG.setdefault("MODEL_TRAINING", {}).setdefault("hyperparameter_tuning", {})[
        "max_trials"
    ] = 3
    CONFIG["MODEL_TRAINING"]["coarse_hpo"] = {"n_trials": 3}
    CONFIG["BLANK_TRAINING_MODE"] = True

    # Initialize Training Manager
    logger.info("Initializing EnhancedTrainingManager...")
    training_manager=EnhancedTrainingManager(CONFIG)

    # Attempt to initialize internal components
    if hasattr(training_manager, "initialize_components"):
        await training_manager.initialize_components()

    # Use existing data only
    logger.info("Using existing data only (no downloads)")

    # Required files (adjust to your environment)
    klines_filename=f"data_cache/klines_{args.exchange}_{args.symbol}_1m_consolidated_fixed.csv"
    agg_trades_filename = f"data_cache/aggtrades_{args.exchange}_{args.symbol}_2025-07-31.csv"
    futures_filename = f"data_cache/futures_{args.exchange}_{args.symbol}_consolidated.csv"

    required_files = [klines_filename, agg_trades_filename, futures_filename]
    missing_files=[f for f in required_files if not os.path.exists(f)]
    if missing_files:
        logger.error(f"Missing required data files: {missing_files}")
        logger.error("Please ensure data files exist before running the blank pipeline.")
        logger.error("Expected files:")
        for _file in required_files:
            logger.error(f"  - {_file}")
        return False

    logger.info("All required data files found, proceeding with validation and run")

    # Ensure aggtrades formatting is correct (non-interactive)
    logger.info("Ensuring aggtrades CSV format is correct...")
    auto_reformat_aggtrades_files_for_exchange(args.exchange, args.symbol)

    # Load the CSV data and validate
    csv_data_file=klines_filename
    if not os.path.exists(csv_data_file):
        logger.error(f"CSV data file not found: {csv_data_file}")
        return False

    klines_df=pd.read_csv(csv_data_file)
    logger.info(
        f"Loaded CSV data: {len(klines_df)} rows x {len(klines_df.columns)} columns",
    )

    # Data format and quality validation
    logger.info("Running data format and quality validation...")
    validation_data: dict[str, Any] = {"klines": klines_df}

    if os.path.exists(agg_trades_filename):
        validation_data["agg_trades"] = pd.read_csv(agg_trades_filename)
        logger.info("Loaded aggregated trades for validation")
    else:
        logger.warning("Aggregated trades file not found, skipping that part of validation")

    if os.path.exists(futures_filename):
        validation_data["futures"] = pd.read_csv(futures_filename)
        logger.info("Loaded futures data for validation")
    else:
        logger.warning("Futures file not found, skipping that part of validation")

    format_valid, format_errors=validate_data_format(validation_data)
    if not format_valid:
        logger.error(f"Data format validation failed: {format_errors}")
        return False

    quality_valid, quality_errors=validate_data_quality(validation_data)
    if not quality_valid:
        logger.error(f"Data quality validation failed: {quality_errors}")
        return False

    logger.info("Data validation passed")

    # Optionally create a pickle artifact used by some downstream steps
    pickle_dir=Path("data/training")
    pickle_dir.mkdir(parents=True, exist_ok=True)
    pickle_file=pickle_dir / f"{args.exchange}_{args.symbol}_collected_data.pkl"
    try:
        import pickle  # local import to avoid overhead if unused

        with open(pickle_file, "wb") as f:
            pickle.dump({"klines": klines_df}, f)
        logger.info(f"Created pickle file: {pickle_file}")
    except Exception as e:  # noqa: BLE001
        logger.warning(f"Failed to create pickle artifact: {e}")

    # Execute a lightweight training path
    logger.info("Starting lightweight optimized training execution...")
    result: dict[str, Any] = {}
    if hasattr(training_manager, "execute_optimized_training"):
        result=await training_manager.execute_optimized_training(
            args.symbol, args.exchange, timeframe="1m"
        )
    elif hasattr(training_manager, "execute_enhanced_training"):
        result_success=await training_manager.execute_enhanced_training(
            {"symbol": args.symbol, "exchange": args.exchange, "training_mode": "blank"}
        )
        result={"success": bool(result_success)}

    duration=time.time() - start_time
    logger.info(f"Training pipeline completed in {duration:.2f} seconds")

    if result is not None:
        logger.info("'Blank' training pipeline completed successfully!")
        logger.info("Training summary:")
        logger.info(f"   Symbol: {args.symbol}")
        logger.info(f"   Exchange: {args.exchange}")
        logger.info(f"   Duration: {duration:.2f} seconds")
        return True

    logger.error("'Blank' training pipeline failed.")
    return False


if __name__== "__main__":
    asyncio.run(main())
