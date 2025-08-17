#!/usr/bin/env python3
"""
Test script for enhanced step1_7 with advanced feature engineering.
This script tests the enhanced version that uses VectorizedAdvancedFeatureEngineering
instead of basic features.
"""

import asyncio
import sys
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.utils.logger import system_logger


async def test_enhanced_step1_7():
    """Test the enhanced step1_7 with advanced feature engineering."""
    logger = system_logger.getChild("TestEnhancedStep1_7")

    print("🚀 Testing enhanced step1_7 with advanced feature engineering...")
    logger.info("🚀 Testing enhanced step1_7 with advanced feature engineering...")

    try:
        # Import the enhanced step1_7
        from src.training.steps.step1_7_hmm_regime_discovery_enhanced import (
            run_step_enhanced,
        )

        print("✅ Successfully imported enhanced step1_7")
        logger.info("✅ Successfully imported enhanced step1_7")

        # Test with a small dataset for quick verification
        success = await run_step_enhanced(
            symbol="ETHUSDT",
            exchange="BINANCE",
            data_dir="data/training",
            timeframe="1m",
            lookback_days=7,  # Small dataset for testing
            force_rerun=True,
            cluster_algorithm="kmeans",
            target_num_clusters=5,  # Fewer clusters for testing
            min_combination_frequency=0.01,
            generate_metrics_report=True,
        )

        if success:
            print("✅ Enhanced step1_7 completed successfully!")
            logger.info("✅ Enhanced step1_7 completed successfully!")
            print("🎉 Advanced feature engineering is working correctly!")
            return True
        else:
            print("❌ Enhanced step1_7 failed!")
            logger.error("❌ Enhanced step1_7 failed!")
            return False

    except Exception as e:
        print(f"❌ Error testing enhanced step1_7: {e}")
        logger.error(f"❌ Error testing enhanced step1_7: {e}")
        return False


if __name__ == "__main__":
    # Run the test
    success = asyncio.run(test_enhanced_step1_7())

    if success:
        print("\n🎉 Test completed successfully!")
        print("✅ Enhanced step1_7 with advanced feature engineering is working!")
        print(
            "✅ The system will now use sophisticated features instead of basic ones."
        )
        sys.exit(0)
    else:
        print("\n❌ Test failed!")
        print("❌ Please check the logs for more details.")
        sys.exit(1)
