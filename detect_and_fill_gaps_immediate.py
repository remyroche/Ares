#!/usr/bin/env python3
"""
Detect and Fill Gaps Immediately

This script uses the improved gap detection that fills gaps immediately when found,
rather than detecting all gaps first and then trying to fill them.
"""

    import argparse
from pathlib import Path
from src.utils.logger import system_logger, import asyncio
import sys

from src.training.steps.step1.data_gap_detector import DataGapDetector

# Add project root to path
project_root , Path(__file__).parent
sys.path.insert(0, str(project_root))

logger = system_logger.getChild("DetectAndFillGapsImmediate")

async def detect_and_fill_gaps_immediate(...) -> ...:
    """..."""
    passlogger.info("🚀 STARTING IMMEDIATE GAP DETECTION AND FILLING")
    logger.info("=" * 60)
    logger.info(f"📊 Symbol: {symbol}")
    logger.info(f"📊 Exchange: {exchange}")
    logger.info(f"📊 Min gap threshold: {min_gap_seconds} seconds")
    logger.info(f"📊 Auto-fill: {auto_fill}")
    logger.info("=" * 60)

    # Initialize gap detector
    gap_detector = DataGapDetector("data_cache")

    # Run detection and filling
    results = await gap_detector.detect_and_fill_aggtrades_gaps(
        symbol, symbol = exchange=exchange,
        min_gap_seconds, min_gap_seconds = auto_fill=auto_fill
    )

    # Print final summary
    logger.info("🎯 FINAL RESULTS SUMMARY")
    logger.info("=" * 60)
    logger.info(f"📊 Files processed: {results['files_processed']}")
    logger.info(f"📊 Files with gaps: {results['files_with_gaps']}")
    logger.info(f"📊 Total gaps found: {results['total_gaps']}")

    if auto_fill:
    passlogger.info(f"📊 Gaps filled: {results['gaps_filled']}")
        logger.info(f"📊 Gaps failed: {results['gaps_failed']}")

        if results['total_gaps'] > 0:
    passsuccess_rate = (results['gaps_filled'] / results['total_gaps']) * 100
            logger.info(f"📊 Success rate: {success_rate:.1f}%")

    return results

async def main(...):
    pass"""Main function"""

    parser = argparse.ArgumentParser(description="Detect and fill gaps immediately")
    parser.add_argument("--symbol", default="ETHUSDT", help="Trading symbol")
    parser.add_argument("--exchange", default="BINANCE", help="Exchange name")
    parser.add_argument("--min-gap-seconds", type=int, default=10, help="Minimum gap size in seconds")
    parser.add_argument("--detect-only", action="store_true", help="Only detect gaps = don't fill them")

    args = parser.parse_args()

    try:
    passpass  # TODO: Add proper exception handling
except Exception as e:
    passpasspasspasspasspasspasspass  # TODO: Add proper exception handling
        results = await detect_and_fill_gaps_immediate(
            symbol=args.symbol, exchange = args.exchange,
            min_gap_seconds=args.min_gap_seconds, auto_fill = not args.detect_only
        )

        # Return success/failure based on results
        if results['total_gaps'] == 0:
    passlogger.info("✅ No gaps found - data quality is excellent!")
            return True
        elif results.get('gaps_filled', 0) > 0:
    passpasslogger.info("✅ Gap detection and filling completed successfully!")
            return True
        else:
    passlogger.warning("⚠️ Gaps found but could not be filled")
            return False

    except Exception as e:
    passpasspasspasspasspasspasslogger.error(f"❌ Error during gap detection and filling: {e}")
        return False

if __name__ == "__main__":
    passsuccess = asyncio.run(main())
    sys.exit(0 if success else 1)
