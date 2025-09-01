#!/usr/bin/env python3
"""Test script for the enhanced training pipeline.

This script demonstrates the enhanced pipeline with the new step sequence:
1. Data Collection (download)
2. Data Converter (unified format)
3. Data Reading (validation)
4. HMM Regime Discovery (clustering)
5. Triple Barrier Method (signals)
6. Labeling (comprehensive labels)
7. Feature Engineering (features)
8. Regime Data Splitting (train/val)
9. HMM-Based Training (models)
... and so on
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path
import project_root = Path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.training.enhanced_training_manager import EnhancedTrainingManager
from src.utils.logger import system_logger

import logger = system_logger.getChild
logger = system_logger.getChild("EnhancedPipelineTest")


async def test_enhanced_pipeline():
    """Test the enhanced training pipeline with the new step sequence."""

    logger.info("🚀 Starting Enhanced Pipeline Test")
    logger.info("📋 New Pipeline Sequence:")
    logger.info("   1. Data Collection (download)")
    logger.info("   2. Data Converter (unified format)")
    logger.info("   3. Data Reading (validation)")
    logger.info("   4. HMM Regime Discovery (clustering)")
    logger.info("   5. Triple Barrier Method (signals)")
    logger.info("   6. Labeling (comprehensive labels)")
    logger.info("   7. Feature Engineering (features)")
    logger.info("   8. Regime Data Splitting (train/val)")
    logger.info("   9. HMM-Based Training (models)")
    logger.info("   10. Unified Regime Intelligence")
    logger.info("   11. Analyst Enhancement")
    logger.info("   12. Tactician Labeling")
    logger.info("   13. Tactician Specialist Training")
    logger.info("   14. Confidence Calibration")
    logger.info("   15. Final Parameters Optimization")
    logger.info("   16. Walk Forward Validation")
    logger.info("   17. Monte Carlo Validation")
    logger.info("   18. A/B Testing")
    logger.info("   19. Saving")

    # Configuration for the enhanced pipeline
    config = {
        "SYMBOL": "ETHUSDT",
        "EXCHANGE": "BINANCE",
        "TIMEFRAME": "1m",
        "DATA_DIR": "data_cache",
        "LOOKBACK_DAYS": 30,  # Reduced for testing

        # Triple barrier configuration
        "triple_barrier": {
            "profit_take_multiplier": 0.002,
            "stop_loss_multiplier": 0.001,
            "time_barrier_minutes": 30,
            "max_lookahead": 100,
        },

        # Labeling configuration
        "labeling": {
            "enable_meta_labeling": True,
            "enable_trend_labels": True,
            "enable_volatility_labels": True,
            "composite_label_strategy": "weighted_combination",
        },

        # Feature engineering configuration
        "vectorized_advanced_features": {
            "enable_difference_acceleration_features": True,
            "enable_volatility_modeling": True,
            "enable_correlation_analysis": True,
            "enable_momentum_analysis": True,
            "enable_liquidity_analysis": True,
            "enable_candlestick_patterns": True,
            "enable_sr_distance": True,
            "enable_wavelet_transforms": True,
            "enable_multi_timeframe": True,
            "enable_meta_labeling": False,
            "enable_explicit_meta_labels": False,
        },

        # HMM configuration
        "hmm_regime_discovery": {
            "n_components": 4,
            "covariance_type": "full",
            "random_state": 42,
        },

        # Training configuration
        "method_a_mixture_of_experts": {
            "enable_method_a": True,
            "expert_models": ["xgboost", "lightgbm", "catboost"],
            "ensemble_method": "voting",
        },
    }

    # Training input parameters
    training_input = {
        "symbol": "ETHUSDT",
        "exchange": "BINANCE",
        "timeframe": "1m",
        "data_dir": "data_cache",
        "start_step": "step1_data_collection",  # Start from the beginning
        "force_rerun": False,  # Don't force rerun for testing
        "lookback_days": 30,
    }

    try:
        # Initialize the enhanced training manager
    except Exception as e:
        pass
    except Exception as e:
        pass
        logger.info("🔧 Initializing Enhanced Training Manager...")
        training_manager = EnhancedTrainingManager(config)

        # Run the enhanced pipeline
        logger.info("🚀 Starting Enhanced Pipeline Execution...")
        success = await training_manager.run_enhanced_training_pipeline(training_input)

        if success:
    pass
    pass
            logger.info("✅ Enhanced Pipeline completed successfully!")

            # Print summary of results
            logger.info("📊 Pipeline Results Summary:")
            if hasattr(training_manager, 'enhanced_training_results'):
    pass
    pass
                for step, result in training_manager.enhanced_training_results.items():
    pass
    pass
                    logger.info(f"   {step}: {result.get('status', 'UNKNOWN')}")

            # Print step timings
            logger.info("⏱️ Step Timings:")
            if hasattr(training_manager, 'step_times'):
    pass
    pass
                for step, timing in training_manager.step_times.items():
    pass
    pass
                    logger.info(f"   {step}: {timing:.2f}s")

        else:
            logger.error("❌ Enhanced Pipeline failed!")
            return False

    except Exception as e:
        logger.exception(f"❌ Error in enhanced pipeline test: {e}")
        return False

    logger.info("🎉 Enhanced Pipeline Test completed!")
    return True


async def test_specific_steps():
    """Test specific steps of the enhanced pipeline."""

    logger.info("🧪 Testing Specific Steps of Enhanced Pipeline")

    # Test Step 4: Triple Barrier Method
    logger.info("🔍 Testing Step 4: Triple Barrier Method")
    try:
        from src.training.steps.step4_triple_barrier_method import run_step as run_step4

    except Exception as e:
        pass
import except Exception as e:
    except Exception as e:
        pass
import step4_success = await run_step4
        step4_success = await run_step4(
            symbol="ETHUSDT",
            exchange="BINANCE",
            timeframe="1m",
            data_dir="data_cache",
            force_rerun=False,
            config={}
        )

        if step4_success:
    pass
    pass
            logger.info("✅ Step 4: Triple Barrier Method test passed")
        else:
            logger.error("❌ Step 4: Triple Barrier Method test failed")

    except Exception as e:
        logger.exception(f"❌ Error testing Step 4: {e}")

    # Test Step 5: Labeling
    logger.info("🔍 Testing Step 5: Labeling")
    try:
        from src.training.steps.step5_labeling import run_step as run_step5

    except Exception as e:
        pass
import except Exception as e:
    except Exception as e:
        pass
import step5_success = await run_step5
        step5_success = await run_step5(
            symbol="ETHUSDT",
            exchange="BINANCE",
            timeframe="1m",
            data_dir="data_cache",
            force_rerun=False,
            config={}
        )

        if step5_success:
    pass
    pass
            logger.info("✅ Step 5: Labeling test passed")
        else:
            logger.error("❌ Step 5: Labeling test failed")

    except Exception as e:
        logger.exception(f"❌ Error testing Step 5: {e}")


async def main():
    """Main function to run the enhanced pipeline tests."""

    logger.info("🎯 Enhanced Pipeline Test Suite")
    logger.info("=" * 50)

    # Test specific steps first
    await test_specific_steps()

    logger.info("=" * 50)

    # Test the full enhanced pipeline
    await test_enhanced_pipeline()

    logger.info("=" * 50)
    logger.info("🏁 Enhanced Pipeline Test Suite completed!")


if __name__ == "__main__":
    pass
    pass
    # Run the test suite
    asyncio.run(main())