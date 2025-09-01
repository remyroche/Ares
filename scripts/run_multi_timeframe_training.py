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
from typing import Any, Dict

from src.config import CONFIG
from src.database.sqlite_manager import SQLiteManager
# Add project root to path)
import project_root = Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.training.steps.multi_timeframe_training.multi_timeframe_training_manager import MultiTimeframeTrainingManager
from src.utils.error_handler import handle_errors


import @handle_errors
@handle_errors(exceptions=(Exception,), default_return=None, context="run_multi_timeframe_training")
async def run_multi_timeframe_training(
    symbol: str,
    timeframes: list[str],
    lookback_days: int = 730,
    enable_ensemble: bool = True,
    parallel: bool = True,
) -> Dict[str, Any] | None:
    """Run multi-timeframe training."""
    logger = system_logger.getChild("MultiTimeframeTrainingRunner")

    logger.info("🚀 Starting Multi-Timeframe Training")
    logger.info(f"Symbol: {symbol}")
    logger.info(f"Timeframes: {timeframes}")
    logger.info(f"Lookback days: {lookback_days}")
    logger.info(f"Ensemble enabled: {enable_ensemble}")
    logger.info(f"Parallel training: {parallel}")

    # Initialize database
    db_manager = SQLiteManager(CONFIG)
    await db_manager.initialize()

    # Initialize multi-timeframe training manager
    mtf_manager = MultiTimeframeTrainingManager(CONFIG)
    await mtf_manager.initialize()

    # Update configuration for parallel training
    if not parallel:
    pass
    pass
        mtf_manager.multi_timeframe_config["enable_parallel_training"] = False

    # Prepare input for execution based on available manager API
    multi_timeframe_training_input: Dict[str, Any] = {
        "symbol": symbol,
        "exchange": "BINANCE",
        "timeframes": timeframes,
        "lookback_days": lookback_days,
        "enable_ensemble": enable_ensemble,
        "enable_parallel_training": parallel,
    }

    # Execute multi-timeframe training via manager
    success = await mtf_manager.execute_multi_timeframe_training(
        multi_timeframe_training_input
    )

    results: Dict[str, Any] = {
        "summary": {
            "status": "success" if success else "failed",
            "symbol": symbol,
            "timeframes": timeframes,
        }
    }

    # Display results
    logger.info("📊 Multi-Timeframe Training Results:")
    logger.info(f"Summary: {results.get('summary', {})}")

    # Display timeframe results
    timeframe_results = results.get("timeframe_results", {})
    for timeframe, result in timeframe_results.items():
    pass
    pass
        status = result.get("status", "unknown")
        logger.info(f"  {timeframe}: {status}")
        if status == "success":
    pass
    pass
            logger.info(f"    Session ID: {result.get('session_id', 'N/A')}")

    # Display ensemble results
    ensemble_results = results.get("ensemble_results", {})
    if ensemble_results.get("status") == "success":
    pass
    pass
        logger.info("✅ Ensemble model created successfully")
        logger.info(f"  Timeframes used: {ensemble_results.get('timeframes_used', [])}")
    else:
        logger.info("❌ Ensemble creation failed")

    # Display validation results
    validation_results = results.get("validation_results", {})
    if validation_results.get("status") == "success":
    pass
    pass
        logger.info("✅ Cross-timeframe validation completed")
    else:
        logger.info("❌ Cross-timeframe validation failed")

    # Display recommendations
    recommendations = results.get("recommendations", [])
    if recommendations:
    pass
    pass
        logger.info("💡 Recommendations:")
        for rec in recommendations:
    pass
    pass
            logger.info(f"  - {rec}")

    return results


@handle_errors(exceptions=(Exception,), default_return=None, context="run_quick_multi_timeframe_test")
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
        symbol,
        timeframes=timeframes,
        lookback_days=lookback_days,
        enable_ensemble=True,
        parallel=False,
    )


@handle_errors(exceptions=(Exception,), default_return=None, context="run_ensemble_only")
async def run_ensemble_only(symbol: str, timeframes: list[str]):
    """Run ensemble creation only (assumes models already trained)."""
    logger = system_logger.getChild("EnsembleOnly")

    logger.info("🎯 Running Ensemble Creation Only")
    logger.info(f"Symbol: {symbol}")
    logger.info(f"Timeframes: {timeframes}")

    # Initialize components
    db_manager = SQLiteManager(CONFIG)
    await db_manager.initialize()

    mtf_manager = MultiTimeframeTrainingManager(CONFIG)
    await mtf_manager.initialize()

    # Simulate successful timeframe results (in real scenario, these would be loaded)
    timeframe_results: Dict[str, Any] = {}
    for timeframe in timeframes:
    pass
    pass
        timeframe_results[timeframe] = {
            "status": "success",
            "session_id": f"simulated_{timeframe}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "timeframe": timeframe,
            "timestamp": datetime.now().isoformat(),
        }

    # The following internal methods may exist; guard calls if present
    ensemble_results = {}
    if hasattr(mtf_manager, "_create_ensemble_models"):
    pass
    pass
        ensemble_results = await getattr(mtf_manager, "_create_ensemble_models")(
            symbol=symbol,
            timeframe_results=timeframe_results,
        )

    validation_results = {}
    if hasattr(mtf_manager, "_cross_timeframe_validation"):
    pass
    pass
        validation_results = await getattr(mtf_manager, "_cross_timeframe_validation")(
            symbol=symbol,
            timeframe_results=timeframe_results,
            ensemble_results=ensemble_results,
        )

    final_results: Dict[str, Any] = {}
    if hasattr(mtf_manager, "_generate_multi_timeframe_report"):
    pass
    pass
        final_results = await getattr(mtf_manager, "_generate_multi_timeframe_report")(
            symbol=symbol,
            timeframe_results=timeframe_results,
            ensemble_results=validation_results,
        )

    logger.info("✅ Ensemble creation completed")
    return final_results


@handle_errors(exceptions=(Exception,), default_return=None, context="analyze_timeframe_correlations")
async def analyze_timeframe_correlations(symbol: str, timeframes: list[str]):
    """Analyze correlations between timeframes."""
    logger = system_logger.getChild("TimeframeAnalysis")

    logger.info("📊 Analyzing Timeframe Correlations")
    logger.info(f"Symbol: {symbol}")
    logger.info(f"Timeframes: {timeframes}")

    # Initialize components
    db_manager = SQLiteManager(CONFIG)
    await db_manager.initialize()

    mtf_manager = MultiTimeframeTrainingManager(CONFIG)
    await mtf_manager.initialize()

    # Simulate successful timeframe results
    successful_timeframes: Dict[str, Any] = {}
    for timeframe in timeframes:
    pass
    pass
        successful_timeframes[timeframe] = {
            "status": "success",
            "session_id": f"analysis_{timeframe}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "timeframe": timeframe,
            "timestamp": datetime.now().isoformat(),
        }

    analysis_results: Dict[str, Any] = {}
    if hasattr(mtf_manager, "_analyze_cross_timeframe_performance"):
    pass
    pass
        analysis_results = await getattr(
            mtf_manager, "_analyze_cross_timeframe_performance"
        )(
            symbol=symbol,
            timeframe_results=successful_timeframes,
            ensemble_results={},
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


def list_available_timeframes() -> None:
    pass
    pass
    """List all available timeframes and their purposes."""
    print("📊 Available Timeframes and Their Purposes")
    print("=" * 60)

    # Get timeframe definitions
    timeframes = CONFIG.get("TIMEFRAMES", {})
    timeframe_sets = CONFIG.get("TIMEFRAME_SETS", {})
    default_set = CONFIG.get("DEFAULT_TIMEFRAME_SET", "swing")

    print("\\\n🎯 Individual Timeframes:")
    print("-" * 40)

    for tf, info in timeframes.items():
    pass
    pass
        print(f"\\\n{tf}:")
        print(f"  Purpose: {info.get('purpose', 'Unknown')}")
        print(f"  Trading Style: {info.get('trading_style', 'Unknown')}")
        print(f"  Lookback Days: {info.get('lookback_days', 'Unknown')}")
        print(f"  Ensemble Weight: {info.get('ensemble_weight', 'Unknown')}")
        print(f"  Description: {info.get('description', 'No description')}")

    print("\\\n📋 Predefined Timeframe Sets:")
    print("-" * 40)

    for set_name, set_info in timeframe_sets.items():
    pass
    pass
        is_default = " (DEFAULT)" if set_name == default_set else ""
        print(f"\\\n{set_name}{is_default}:")
        print(f"  Timeframes: {', '.join(set_info.get('timeframes', []))}")
        print(f"  Description: {set_info.get('description', 'No description')}")
        print(f"  Use Case: {set_info.get('use_case', 'No use case specified')}")

    print("\\\n🔧 Configuration:")
    print("-" * 40)
    print(f"Default timeframe set: {default_set}")
    print(f"Total timeframes defined: {len(timeframes)}")
    print(f"Total timeframe sets: {len(timeframe_sets)}")


def main() -> None:
    pass
    pass
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
    pass
    pass
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
    pass
    pass
        args.lookback = CONFIG.get("DATA_CONFIG", {}).get("default_lookback_days", 730)

    # Run appropriate function
    if args.list_timeframes:
    pass
    pass
        list_available_timeframes()
        success = True
    elif args.quick_test:
        success = asyncio.run(run_quick_multi_timeframe_test(args.symbol)) is not None
    elif args.ensemble_only:
        success = asyncio.run(run_ensemble_only(args.symbol, timeframes)) is not None
    elif args.analyze:
        success = (
            asyncio.run(analyze_timeframe_correlations(args.symbol, timeframes))
            is not None
        )
    else:
        result = asyncio.run(
            run_multi_timeframe_training(
                symbol=args.symbol,
                timeframes=timeframes,
                lookback_days=args.lookback,
                enable_ensemble=not args.no_ensemble,
                parallel=not args.sequential,
            ),
        )
        success = result is not None

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    pass
    pass
    main()
