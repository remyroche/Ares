#!/usr/bin/env python3
"""
Fixed HMM Regime Discovery Script

This script runs the HMM regime discovery step with proper 6-month data range
and comprehensive error handling to fix the issues encountered.
"""

import asyncio
import sys
from pathlib import Path

from src.config.constants import DEFAULT_LOOKBACK_DAYS
from src.training.steps.step3_hmm_regime_discovery import run_step
from src.utils.logger import setup_logging, system_logger

# Add the project root to the Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


async def main():
    """Run the fixed HMM regime discovery step."""
    setup_logging()
    logger=system_logger.getChild("FixedHMMRegimeDiscovery")

    logger.info("🚀 Starting Fixed HMM Regime Discovery")
    logger.info("=" * 80)
    logger.info(f"📊 Using {DEFAULT_LOOKBACK_DAYS} days lookback (6 months)")
    logger.info("📊 Target symbol: ETHUSDT")
    logger.info("📊 Target exchange: BINANCE")
    logger.info("=" * 80)

    try:
        # Run the HMM regime discovery step with fixed parameters
        success = await run_step(
            symbol="ETHUSDT",
            exchange="BINANCE",
            data_dir="data/training",
            timeframe="1m",  # Start with 1m; will process all timeframes
            lookback_days=DEFAULT_LOOKBACK_DAYS,  # Use exactly 180 days
            force_reload=False,  # Use cache if available
        )

        if success:
            logger.info("✅ HMM Regime Discovery completed successfully!")
            logger.info("📊 Check the reports directory for detailed regime analysis")
            return True
        logger.error("❌ HMM Regime Discovery failed")
        return False

    except Exception as e:
        logger.exception(f"❌ Error during HMM Regime Discovery: {e}")
        logger.exception(f"Stack trace: {sys.exc_info()}")
        return False


if __name__== "__main__":
    try:
        success = asyncio.run(main())
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        sys.exit(1)
