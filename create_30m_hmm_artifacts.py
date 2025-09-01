#!/usr/bin/env python3
"""
Script to create missing HMM artifacts for 30m timeframe.
This will run the step1_7 HMM regime discovery process specifically for 30m.
"""

import traceback
import asyncio
from pathlib import Path
from src.training.steps.step3_hmm_regime_discovery import run_step
from src.utils.logger import system_logger
import os
import sys

# Add the project root to the path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

async def create_30m_hmm_artifacts():
    """Create missing HMM artifacts for 30m timeframe."""
    logger = system_logger.getChild("Create30mArtifacts")

    logger.info("🔧 Starting creation of missing 30m HMM artifacts...")

    # Parameters for the HMM regime discovery
    symbol = "ETHUSDT"
    exchange = "BINANCE"
    data_dir = "data/training"
    timeframe = "30m"  # Specifically target 30m
    lookback_days = 180  # Use 6 months of data

    logger.info(
        f"📋 Parameters: symbol={symbol}, exchange={exchange}, timeframe={timeframe}",
    )

    try:
        # Run the HMM regime discovery for 30m timeframe
        success = await run_step(
            symbol=symbol,
            exchange=exchange,
            data_dir=data_dir,
            timeframe=timeframe,
            lookback_days=lookback_days
        )

        if success:
            logger.info("✅ Successfully created 30m HMM artifacts!")

            # Verify the artifacts were created
            artifacts_to_check = [
                f"{exchange}_{symbol}_hmm_block_states_{timeframe}.parquet",
                f"{exchange}_{symbol}_hmm_composite_clusters_{timeframe}.parquet",
                f"{exchange}_{symbol}_hmm_composite_intensity_{timeframe}.parquet",
                f"{exchange}_{symbol}_hmm_composite_meta_{timeframe}.json",
            ]

            logger.info("🔍 Verifying created artifacts...")
            for artifact in artifacts_to_check:
                artifact_path = os.path.join(data_dir, artifact)
                if os.path.exists(artifact_path):
                    size = os.path.getsize(artifact_path)
                    logger.info(f"✅ {artifact}: {size:,} bytes")
                else:
                    logger.warning(f"❌ {artifact}: Not found")

        else:
            logger.error("❌ Failed to create 30m HMM artifacts")
            return False

    except Exception as e:
        logger.exception(f"❌ Error creating 30m HMM artifacts: {e}")
        logger.exception(f"Traceback: {traceback.format_exc()}")
        return False

    logger.info("🎉 30m HMM artifact creation process completed!")
    return True

if __name__ == "__main__":
    # Run the async function
    success = asyncio.run(create_30m_hmm_artifacts())

    if success:
        print("✅ 30m HMM artifacts created successfully!")
        sys.exit(0)
    else:
        print("❌ Failed to create 30m HMM artifacts")
        sys.exit(1)
