#!/usr/bin/env python3
"""
Test script to verify backtesting sub-pipeline fix.
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_backtesting_import():
    """Test that backtesting sub-pipeline can be imported."""
    try:
        print("🔄 Testing backtesting sub-pipeline import...")

        # Test individual backtesting components
        from src.training.steps.backtesting.sub_pipeline import (
            BacktestingSubPipeline,
            SubPipelineConfig,
            ExecutionMode
        )
        print("✅ BacktestingSubPipeline imported successfully")

        # Test configuration
        config = SubPipelineConfig(
            symbol="BTCUSDT",
            exchange="binance",
            mode=ExecutionMode.FULL
        )
        print("✅ SubPipelineConfig created successfully")

        # Test pipeline instance
        pipeline = BacktestingSubPipeline(config)
        print("✅ BacktestingSubPipeline instance created successfully")

        # Test available sub-pipelines
        available = pipeline.get_available_sub_pipelines()
        print(f"✅ Available sub-pipelines: {available}")

        return True

    except Exception as e:
        print(f"❌ Backtesting import failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_main_pipeline_backtesting():
    """Test main pipeline with backtesting enabled."""
    try:
        print("🔄 Testing main pipeline backtesting availability...")

        # Import main pipeline
        from src.training.steps.main_training_pipeline import (
            MainTrainingPipeline,
            MainPipelineConfig,
            PipelineStage
        )
        print("✅ MainTrainingPipeline imported successfully")

        # Create config with backtesting enabled
        config = MainPipelineConfig(
            enabled_stages=[PipelineStage.BACKTESTING]
        )
        print("✅ MainPipelineConfig created with backtesting")

        # Create pipeline instance
        pipeline = MainTrainingPipeline(config)
        print("✅ MainTrainingPipeline instance created")

        # Check if backtesting is available
        backtesting_available = pipeline.backtesting_pipeline is not None
        print(f"📊 Backtesting pipeline available: {backtesting_available}")

        if backtesting_available:
            print("✅ SUCCESS: Backtesting sub-pipeline is available!")
            return True
        else:
            print("❌ FAILURE: Backtesting sub-pipeline is not available")
            return False

    except Exception as e:
        print(f"❌ Main pipeline test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main test function."""
    print("🚀 Starting backtesting fix verification...")
    print("=" * 50)

    # Test 1: Individual backtesting import
    success1 = test_backtesting_import()
    print()

    # Test 2: Main pipeline backtesting
    success2 = test_main_pipeline_backtesting()
    print()

    print("=" * 50)
    if success1 and success2:
        print("🎉 ALL TESTS PASSED! Backtesting fix is successful.")
        print("📝 The warning 'Backtesting sub-pipeline not available' should be resolved.")
        return 0
    else:
        print("❌ SOME TESTS FAILED! Backtesting fix needs more work.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
