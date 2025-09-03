#!/usr/bin/env python3
"""Run the complete feature engineering and selection pipeline."""

import asyncio
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.training.steps.step06_feature_engineering import run_step as run_step06
from src.training.steps.step07_enhanced_matrix_operations import run_step as run_step07
from src.training.steps.step08_advanced_feature_selection import run_step as run_step08
from src.utils.logger import system_logger


async def run_feature_pipeline(symbol: str, exchange: str, timeframe: str = "1m"):
    """Run the complete feature engineering and selection pipeline."""
    
    logger = system_logger.getChild("FeaturePipeline")
    
    # Step 6: Feature Engineering
    logger.info("Running Step 6: Feature Engineering...")
    if not await run_step06(symbol, exchange, timeframe):
        logger.error("Step 6 failed!")
        return False
    
    # Step 7: Matrix Operations & Initial Filtering
    logger.info("Running Step 7: Matrix Operations & Filtering...")
    if not await run_step07(symbol, exchange, timeframe):
        logger.error("Step 7 failed!")
        return False
    
    # Step 8: Advanced Feature Selection
    logger.info("Running Step 8: Advanced Feature Selection...")
    if not await run_step08(symbol, exchange, timeframe):
        logger.error("Step 8 failed!")
        return False
    
    logger.info("✅ Feature pipeline completed successfully!")
    return True


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python run_feature_pipeline.py <symbol> <exchange> [timeframe]")
        sys.exit(1)
    
    symbol = sys.argv[1]
    exchange = sys.argv[2]
    timeframe = sys.argv[3] if len(sys.argv) > 3 else "1m"
    
    success = asyncio.run(run_feature_pipeline(symbol, exchange, timeframe))
    sys.exit(0 if success else 1)
