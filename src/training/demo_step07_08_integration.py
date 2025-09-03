#!/usr/bin/env python3
"""Demo script showing Step 7 and Step 8 integration."""

import asyncio
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.training.steps.step07_enhanced_matrix_operations import run_step as run_step07
from src.training.steps.step08_advanced_feature_selection import run_step as run_step08
from src.utils.logger import system_logger


async def demo_feature_selection_pipeline(
    symbol: str = "BTCUSDT",
    exchange: str = "binance",
    timeframe: str = "1m"
):
    """Demonstrate the feature selection pipeline (Steps 7-8)."""
    
    logger = system_logger.getChild("FeatureSelectionDemo")
    
    logger.info(f"🚀 Starting feature selection pipeline for {symbol} on {exchange}")
    
    # Step 7: Initial feature filtering
    logger.info("\n" + "="*60)
    logger.info("📊 STEP 7: Enhanced Matrix Operations & Initial Filtering")
    logger.info("="*60)
    
    step7_success = await run_step07(
        symbol=symbol,
        exchange=exchange,
        timeframe=timeframe,
        force_rerun=True
    )
    
    if not step7_success:
        logger.error("❌ Step 7 failed!")
        return False
    
    logger.info("✅ Step 7 completed successfully")
    logger.info("   - Applied matrix operations")
    logger.info("   - Filtered bottom 33% of features")
    logger.info("   - Output: ~200 features")
    
    # Step 8: Advanced feature selection
    logger.info("\n" + "="*60)
    logger.info("🎯 STEP 8: Advanced Feature Selection")
    logger.info("="*60)
    
    step8_success = await run_step08(
        symbol=symbol,
        exchange=exchange,
        timeframe=timeframe,
        force_rerun=True
    )
    
    if not step8_success:
        logger.error("❌ Step 8 failed!")
        return False
    
    logger.info("✅ Step 8 completed successfully")
    logger.info("   - Phase 1: mRMR/RF selection to ~150 features")
    logger.info("   - Phase 2: Boruta selection to 100/80/60 features")
    logger.info("   - Generated interpretability reports")
    
    # Summary
    logger.info("\n" + "="*60)
    logger.info("📈 FEATURE SELECTION PIPELINE SUMMARY")
    logger.info("="*60)
    logger.info(f"Symbol: {symbol}")
    logger.info(f"Exchange: {exchange}")
    logger.info(f"Timeframe: {timeframe}")
    logger.info("\nFeature Progression:")
    logger.info("  Step 6 output: ~300+ features")
    logger.info("  Step 7 output: ~200 features (filtered)")
    logger.info("  Step 8 Phase 1: ~150 features (mRMR/RF)")
    logger.info("  Step 8 Phase 2: 100/80/60 feature sets (Boruta)")
    logger.info("\nOutput Files:")
    logger.info("  - data/training/{exchange}_{symbol}_{timeframe}_features_filtered_*.parquet")
    logger.info("  - data/selected_features/{exchange}_{symbol}_{timeframe}_top*.parquet")
    logger.info("  - data/selected_features/{exchange}_{symbol}_{timeframe}_interpretability_report.json")
    
    return True


async def main():
    """Main demo function."""
    # Run demo
    success = await demo_feature_selection_pipeline(
        symbol="BTCUSDT",
        exchange="binance",
        timeframe="1m"
    )
    
    if success:
        system_logger.info("\n✅ Feature selection pipeline completed successfully!")
    else:
        system_logger.error("\n❌ Feature selection pipeline failed!")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())