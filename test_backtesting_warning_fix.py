#!/usr/bin/env python3
"""
Test script to verify the backtesting warning is fixed.
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_backtesting_warning_fix():
    """Test that the backtesting warning is resolved."""
    try:
        print("🔄 Testing backtesting warning fix...")

        # Import main training pipeline
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
            print("🎉 SUCCESS: Backtesting sub-pipeline is available!")
            print("✅ The warning '⚠️ Backtesting sub-pipeline not available' should be resolved.")
            return True
        else:
            print("❌ FAILURE: Backtesting sub-pipeline is still not available")
            return False

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main test function."""
    print("🚀 Testing Backtesting Warning Fix")
    print("=" * 50)

    success = test_backtesting_warning_fix()

    print("=" * 50)
    if success:
        print("🎉 BACKTESTING WARNING FIX SUCCESSFUL!")
        print("📝 The warning '⚠️ Backtesting sub-pipeline not available' has been resolved.")
        print("📝 Users should no longer see this warning when running the main training pipeline.")
        return 0
    else:
        print("❌ BACKTESTING WARNING FIX FAILED!")
        return 1

if __name__ == "__main__":
    sys.exit(main())
