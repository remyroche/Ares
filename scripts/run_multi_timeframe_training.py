#!/usr/bin/env python3
# scripts/run_multi_timeframe_training.py

"""
Multi-Timeframe Training Script

This script runs training across multiple timeframes with ensemble creation
and cross-timeframe validation.
"""

import argparse
import asyncio
import sys
from datetime import datetime
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.config import CONFIG
from src.database.sqlite_manager import SQLiteManager
from src.training.multi_timeframe_training_manager import MultiTimeframeTrainingManager
from src.utils.logger import system_logger


async def run_multi_timeframe_training(
    symbol: str,
    timeframes: list[str],
    lookback_days: int = 730,
    enable_ensemble: bool = True,
    parallel: bool = True,
):
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
    for timeframe, result in timeframe_results.items():
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
    lookback_days = 30  # Limited data

    # Update configuration for quick test
    CONFIG["MULTI_TIMEFRAME_TRAINING"]["enable_parallel_training"] = False
    CONFIG["MULTI_TIMEFRAME_TRAINING"]["enable_ensemble"] = True
    CONFIG["MULTI_TIMEFRAME_TRAINING"]["enable_cross_validation"] = False

    results = await run_multi_timeframe_training(
        symbol=symbol,
        timeframes=timeframes,
        lookback_days=lookback_days,
        enable_ensemble=True,
        parallel=False,
    )

    return results


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
        symbol,
        timeframe_results,
    )

    # Validate ensemble
    validation_results = await mtf_manager._cross_timeframe_validation(
        symbol,
        timeframe_results,
        ensemble_results,
    )

    # Generate report
    final_results = await mtf_manager._generate_multi_timeframe_report(
        symbol,
        timeframe_results,
        ensemble_results,
        validation_results,
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

    # Get timeframe definitions
    timeframes = CONFIG.get("TIMEFRAMES", {})
    timeframe_sets = CONFIG.get("TIMEFRAME_SETS", {})
    default_set = CONFIG.get("DEFAULT_TIMEFRAME_SET", "swing")

    print("\n🎯 Individual Timeframes:")
    print("-" * 40)

    for tf, info in timeframes.items():
        print(f"\n{tf}:")
        print(f"  Purpose: {info.get('purpose', 'Unknown')}")
        print(f"  Trading Style: {info.get('trading_style', 'Unknown')}")
        print(f"  Lookback Days: {info.get('lookback_days', 'Unknown')}")
        print(f"  Ensemble Weight: {info.get('ensemble_weight', 'Unknown')}")
        print(f"  Description: {info.get('description', 'No description')}")

    print("\n📋 Predefined Timeframe Sets:")
    print("-" * 40)

    for set_name, set_info in timeframe_sets.items():
        is_default = " (DEFAULT)" if set_name == default_set else ""
        print(f"\n{set_name}{is_default}:")
        print(f"  Timeframes: {', '.join(set_info.get('timeframes', []))}")
        print(f"  Description: {set_info.get('description', 'No description')}")
        print(f"  Use Case: {set_info.get('use_case', 'No use case specified')}")

    print("\n🔧 Configuration:")
    print("-" * 40)
    print(f"Default timeframe set: {default_set}")
    print(f"Total timeframes defined: {len(timeframes)}")
    print(f"Total timeframe sets: {len(timeframe_sets)}")


def main():
    """Main function with command line interface."""
    parser = argparse.ArgumentParser(
        description="Multi-Timeframe Training with Ensemble Creation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List all available timeframes and their purposes
  python scripts/run_multi_timeframe_training.py --list-timeframes
  
  # Full multi-timeframe training
  python scripts/run_multi_timeframe_training.py --symbol ETHUSDT --timeframes 1h,4h,1d
  
  # Quick test with limited data
  python scripts/run_multi_timeframe_training.py --symbol ETHUSDT --quick-test
  
  # Ensemble only (assumes models already trained)
  python scripts/run_multi_timeframe_training.py --symbol ETHUSDT --ensemble-only --timeframes 1h,4h,1d
  
  # Analyze timeframe correlations
  python scripts/run_multi_timeframe_training.py --symbol ETHUSDT --analyze --timeframes 1h,4h,1d
  
  # Sequential training (no parallel)
  python scripts/run_multi_timeframe_training.py --symbol ETHUSDT --timeframes 1h,4h,1d --sequential
        """,
    )

    parser.add_argument("--symbol", default="ETHUSDT", help="Trading symbol")
    parser.add_argument(
        "--timeframes",
        help="Comma-separated list of timeframes (e.g., 1h,4h,1d)",
    )
    parser.add_argument(
        "--lookback",
        type=int,
        default=None,
        help="Lookback days (default: from DATA_CONFIG)",
    )
    parser.add_argument(
        "--no-ensemble",
        action="store_true",
        help="Disable ensemble creation",
    )
    parser.add_argument(
        "--sequential",
        action="store_true",
        help="Use sequential training (no parallel)",
    )
    parser.add_argument(
        "--quick-test",
        action="store_true",
        help="Run quick test with limited data",
    )
    parser.add_argument(
        "--ensemble-only",
        action="store_true",
        help="Run ensemble creation only",
    )
    parser.add_argument(
        "--analyze",
        action="store_true",
        help="Analyze timeframe correlations only",
    )
    parser.add_argument(
        "--list-timeframes",
        action="store_true",
        help="List all available timeframes and their purposes",
    )

    args = parser.parse_args()

    # Parse timeframes
    if args.timeframes:
        timeframes = [tf.strip() for tf in args.timeframes.split(",")]
    else:
        # Get default timeframe set
        default_set = CONFIG.get("DEFAULT_TIMEFRAME_SET", "swing")
        timeframe_sets = CONFIG.get("TIMEFRAME_SETS", {})
        timeframes = timeframe_sets.get(default_set, {}).get(
            "timeframes",
            ["1h", "4h", "1d"],
        )

    # Update configuration
    CONFIG["trading_symbol"] = args.symbol
    CONFIG["MULTI_TIMEFRAME_TRAINING"]["enable_parallel_training"] = not args.sequential

    # Use centralized lookback_days
    if args.lookback is None:
        args.lookback = CONFIG.get("DATA_CONFIG", {}).get("default_lookback_days", 730)

    # Run appropriate function
    if args.list_timeframes:
        list_available_timeframes()
    elif args.quick_test:
        success = asyncio.run(run_quick_multi_timeframe_test(args.symbol))
    elif args.ensemble_only:
        success = asyncio.run(run_ensemble_only(args.symbol, timeframes))
    elif args.analyze:
        success = asyncio.run(analyze_timeframe_correlations(args.symbol, timeframes))
    else:
        success = asyncio.run(
            run_multi_timeframe_training(
                symbol=args.symbol,
                timeframes=timeframes,
                lookback_days=args.lookback,
                enable_ensemble=not args.no_ensemble,
                parallel=not args.sequential,
            ),
        )

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
