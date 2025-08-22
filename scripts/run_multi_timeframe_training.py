#!/usr/bin/env python3
# scripts/run_multi_timeframe_training.py

"""
Multi-Timeframe Training Script

This script runs training across multiple timeframes with ensemble creation
and cross-timeframe validation.
"""

from datetime import datetime
from pathlib import Path
from src.utils.logger import system_logger
import argparse
import asyncio
import sys

from src.config import CONFIG
from src.database.sqlite_manager import SQLiteManager
# Add project root to path)
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.training.multi_timeframe_training_manager import MultiTimeframeTrainingManager

async def run_multi_timeframe_training(symbol: str, timeframes: list[str], lookback_days: int = 730, enable_ensemble: bool = True, parallel: bool = True) -> None:
    """Run multi-timeframe training."""
    logger = system_logger.getChild("MultiTimeframeTrainingRunner")

    logger.info("🚀 Starting Multi-Timeframe Training")
    logger.info(f"Symbol: {symbol}")
    logger.info(f"Timeframes: {timeframes}")
    logger.info(f"Lookback days: {lookback_days}")
    logger.info(f"Ensemble enabled: {enable_ensemble}")
    logger.info(f"Parallel training: {parallel}")

    # Initialize database
    db_manager = SQLiteManager()
    await db_manager.initialize()

    # Initialize multi-timeframe training manager
    mtf_manager = MultiTimeframeTrainingManager(CONFIG)
    await mtf_manager.initialize()

    # Update configuration for parallel training
    if not parallel:
        mtf_manager.mtf_config["enable_parallel_training"] = False

    # Run multi-timeframe training
    results = await mtf_manager.run_multi_timeframe_training(
        symbol=symbol,
        exchange_name="BINANCE",
        timeframes=timeframes,
        lookback_days=lookback_days,
        use_multi_timeframe_features=True,  # Enable multi-timeframe features
    )

    # Display results
    logger.info("📊 Multi-Timeframe Training Results:")
    logger.info(f"Summary: {results.get('summary', {})}")

    # Display timeframe results
    timeframe_results = results.get("timeframe_results", {})
    for timeframe , result in timeframe_results.items():
        status = result.get("status", "unknown")
        logger.info(f"  {timeframe}: {status}")
        if status == "success":
            logger.info(f"    Session ID: {result.get('session_id', 'N/A')}")

    # Display ensemble results
    ensemble_results = results.get("ensemble_results", {})
    if ensemble_results.get("status") == "success":
        logger.info("✅ Ensemble model created successfully")
        logger.info(f"  Timeframes used: {ensemble_results.get('timeframes_used', [])}")
    else:
        logger.info("❌ Ensemble creation failed")

    # Display validation results
    validation_results = results.get("validation_results", {})
    if validation_results.get("status") == "success":
        logger.info("✅ Cross-timeframe validation completed")
    else:
        logger.info("❌ Cross-timeframe validation failed")

    # Display recommendations
    recommendations = results.get("recommendations", [])
    if recommendations:
        logger.info("💡 Recommendations:")
        for rec in recommendations:
            logger.info(f"  - {rec}")

    return results

async def run_quick_multi_timeframe_test(symbol: str):
    """Run a quick multi-timeframe test with limited data."""
    logger = system_logger.getChild("QuickMultiTimeframeTest")

    logger.info("🧪 Running Quick Multi-Timeframe Test")

    # Use limited timeframes and data for quick testing
    timeframes = ["1h", "4h"]  # Reduced timeframes
    lookback_days = 60  # Limited data (expanded for better regime coverage)

    # Update configuration for quick test
    CONFIG["MULTI_TIMEFRAME_TRAINING"]["enable_parallel_training"] = False
    CONFIG["MULTI_TIMEFRAME_TRAINING"]["enable_ensemble"] = True
    CONFIG["MULTI_TIMEFRAME_TRAINING"]["enable_cross_validation"] = False

    return await run_multi_timeframe_training(
        symbol, timeframes=timeframes,
        lookback_days=lookback_days, enable_ensemble=True,
        parallel=False)

async def run_ensemble_only(symbol: str, timeframes: list[str]):
    """Run ensemble creation only (assumes models already trained)."""
    logger = system_logger.getChild("EnsembleOnly")

    logger.info("🎯 Running Ensemble Creation Only")
    logger.info(f"Symbol: {symbol}")
    logger.info(f"Timeframes: {timeframes}")

    # Initialize components
    db_manager = SQLiteManager()
    await db_manager.initialize()

    mtf_manager = MultiTimeframeTrainingManager(db_manager)

    # Simulate successful timeframe results (in real scenario, these would be loaded)
    timeframe_results = {}
    for timeframe in timeframes:
        timeframe_results[timeframe] = {
            "status": "success",
            "session_id": f"simulated_{timeframe}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "timeframe": timeframe,
            "timestamp": datetime.now().isoformat(),
        }

    # Create ensemble
    ensemble_results = await mtf_manager._create_ensemble_models(
        symbol=symbol,
        timeframe_results=timeframe_results,
    )

    # Validate ensemble
    validation_results = await mtf_manager._cross_timeframe_validation(
        symbol=symbol,
        timeframe_results=timeframe_results,
        ensemble_results=ensemble_results,
    )

    # Generate report
    final_results = await mtf_manager._generate_multi_timeframe_report(
        symbol=symbol,
        timeframe_results=timeframe_results,
        ensemble_results=validation_results,
    )

    logger.info("✅ Ensemble creation completed")
    return final_results

async def analyze_timeframe_correlations(symbol: str, timeframes: list[str]):
    """Analyze correlations between timeframes."""
    logger = system_logger.getChild("TimeframeAnalysis")

    logger.info("📊 Analyzing Timeframe Correlations")
    logger.info(f"Symbol: {symbol}")
    logger.info(f"Timeframes: {timeframes}")

    # Initialize components
    db_manager = SQLiteManager()
    await db_manager.initialize()

    mtf_manager = MultiTimeframeTrainingManager(db_manager)

    # Simulate successful timeframe results
    successful_timeframes = {}
    for timeframe in timeframes:
        successful_timeframes[timeframe] = {
            "status": "success",
            "session_id": f"analysis_{timeframe}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "timeframe": timeframe,
            "timestamp": datetime.now().isoformat(),
        }

    # Analyze correlations
    analysis_results = await mtf_manager._analyze_cross_timeframe_performance(
        symbol,
        successful_timeframes,
        {},
    )

    logger.info("📊 Analysis Results:")
    logger.info(
        f"  Timeframe correlations: {analysis_results.get('timeframe_correlations', {})}",
    )
    logger.info(f"  Consistency score: {analysis_results.get('consistency_score', 0)}")
    logger.info(
        f"  Diversification benefit: {analysis_results.get('diversification_benefit', 0)}",
    )
    logger.info(f"  Optimal weights: {analysis_results.get('optimal_weights', {})}")

    return analysis_results


def list_available_timeframes():
    """List all available timeframes and their purposes."""
    print("📊 Available Timeframes and Their Purposes")
    print("=" * 60)

    print("1m - High frequency, micro-trend analysis")
    print("5m - Short-term trading and noise reduction")
    print("15m - Balance between detail and trend")
    print("1h - Medium-term trend analysis")
    print("4h - Long-term trends and regime detection")
    print("1d - Macro trends and portfolio decisions")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Multi-Timeframe Training")
    parser.add_argument("symbol", help="Trading symbol (e.g., BTCUSDT)")
    parser.add_argument(
        "--timeframes",
        nargs="+",
        default=["1h", "4h", "1d"],
        help="Timeframes to train",
    )
    parser.add_argument(
        "--lookback-days",
        type=int,
        default=730,
        help="Number of days to look back",
    )
    parser.add_argument(
        "--no-ensemble",
        action="store_true",
        help="Disable ensemble creation",
    )
    parser.add_argument(
        "--sequential",
        action="store_true",
        help="Disable parallel training",
    )
    return parser


async def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()

    if args.symbol.lower() == "list":
        list_available_timeframes()
        return 0

    await run_multi_timeframe_training(
        symbol=args.symbol,
        timeframes=args.timeframes,
        lookback_days=args.lookback_days,
        enable_ensemble=not args.no_ensemble,
        parallel=not args.sequential,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
