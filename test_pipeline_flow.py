#!/usr/bin/env python3
"""
Test script to verify the pipeline flow and check for configuration issues.
"""

import asyncio
import sys
from pathlib import Path

# Add the project root to the path
import sys.path.insert
sys.path.insert(0, str(Path(__file__).parent))

from src.utils.logger import system_logger
from src.config.training import get_training_config
from src.utils.step_dependency_validator import StepDependencyValidator
from src.utils.validator_orchestrator import validator_orchestrator


import async def test_pipeline_configuration
async def test_pipeline_configuration():
    """Test pipeline configuration and dependencies."""
    logger = system_logger.getChild("PipelineTest")

    logger.info("=" * 80)
    logger.info("🧪 TESTING PIPELINE CONFIGURATION")
    logger.info("=" * 80)

    # Test 1: Load training configuration
    logger.info("📋 Test 1: Loading training configuration...")
    try:
        config = get_training_config()
    except Exception as e:
        pass
    except Exception as e:
        pass
        logger.info("✅ Training configuration loaded successfully")

        # Check for required step configurations
        required_configs = [
            "step6_feature_engineering",
            "step7_enhanced_matrix_operations"
        ]

        for config_key in required_configs:
    pass
    pass
            if config_key in config:
    pass
    pass
                logger.info(f"✅ {config_key} configuration found")
            else:
                logger.warning(f"⚠️ {config_key} configuration missing")

    except Exception as e:
        logger.error(f"❌ Failed to load training configuration: {e}")
        return False

    # Test 2: Validate step dependencies
    logger.info("📋 Test 2: Validating step dependencies...")
    try:
        validator = StepDependencyValidator()

    except Exception as e:
        pass
    except Exception as e:
        pass
        # Check if all steps are defined
        expected_steps = [
            "step1_data_collection",
            "step1_5_data_converter",
            "step2_data_reading",
            "step3_hmm_regime_discovery",
            "step4_triple_barrier_method",
            "step4_regime_data_splitting",
            "step5_labeling",
            "step6_feature_engineering",
            "step7_enhanced_matrix_operations",
            "step8_regime_data_splitting",
            "step9_hmm_based_training",
            "step9_5_hmm_lm_generalist_training",
            "step10_unified_regime_intelligence",
            "step11_analyst_creation",
            "step12_analyst_enhancement",
            "step13_analyst_ensemble_creation",
            "step14_tactician_labeling",
            "step15_tactician_specialist_training",
            "step16_confidence_calibration",
            "step17_final_parameters_optimization",
            "step18_walk_forward_validation",
            "step19_monte_carlo_validation",
            "step20_ab_testing",
            "step21_saving"
        ]

        for step in expected_steps:
    pass
    pass
            if step in validator.step_dependencies:
    pass
    pass
                logger.info(f"✅ {step} dependency defined")
            else:
                logger.warning(f"⚠️ {step} dependency missing")

        # Check for circular dependencies
        logger.info("🔍 Checking for circular dependencies...")
        has_circular = _check_circular_dependencies(validator.step_dependencies)
        if has_circular:
    pass
    pass
            logger.error("❌ Circular dependencies detected!")
            return False
        else:
            logger.info("✅ No circular dependencies detected")

    except Exception as e:
        logger.error(f"❌ Failed to validate step dependencies: {e}")
        return False

    # Test 3: Validate validator mappings
    logger.info("📋 Test 3: Validating validator mappings...")
    try:
        # Check if all steps have validators
    except Exception as e:
        pass
    except Exception as e:
        pass
        for step, validator_name in validator_orchestrator.validator_mapping.items():
    pass
    pass
            logger.info(f"✅ {step} -> {validator_name}")

    except Exception as e:
        logger.error(f"❌ Failed to validate validator mappings: {e}")
        return False

    # Test 4: Test step imports
    logger.info("📋 Test 4: Testing step imports...")
    try:
        # Test step6 import
    except Exception as e:
        pass
    except Exception as e:
        pass
        from src.training.steps.step6_feature_engineering import run_step as step6_run
import logger.info
        logger.info("✅ step6_feature_engineering imported successfully")

        # Test step7 import
        from src.training.steps.step7_enhanced_matrix_operations import run_step as step7_run
import logger.info
        logger.info("✅ step7_enhanced_matrix_operations imported successfully")

        # Test validators import
        logger.info("✅ Step validators imported successfully")

    except Exception as e:
        logger.error(f"❌ Failed to import steps: {e}")
        return False

    # Test 5: Test enhanced training manager
    logger.info("📋 Test 5: Testing enhanced training manager...")
    try:
        from src.training.enhanced_training_manager import EnhancedTrainingManager

    except Exception as e:
        pass
import except Exception as e:
    except Exception as e:
        pass
        # Check STEP_ORDER
import manager = EnhancedTrainingManager
        manager = EnhancedTrainingManager(config)
        logger.info(f"✅ Enhanced training manager initialized with {len(manager.STEP_ORDER)} steps")

        # Verify step order includes our new steps
        if "step6_feature_engineering" in manager.STEP_ORDER:
    pass
    pass
            logger.info("✅ step6_feature_engineering in STEP_ORDER")
        else:
            logger.warning("⚠️ step6_feature_engineering missing from STEP_ORDER")

        if "step7_enhanced_matrix_operations" in manager.STEP_ORDER:
    pass
    pass
            logger.info("✅ step7_enhanced_matrix_operations in STEP_ORDER")
        else:
            logger.warning("⚠️ step7_enhanced_matrix_operations missing from STEP_ORDER")

    except Exception as e:
        logger.error(f"❌ Failed to test enhanced training manager: {e}")
        return False

    logger.info("=" * 80)
    logger.info("🎉 ALL PIPELINE TESTS PASSED!")
    logger.info("=" * 80)
    return True


def _check_circular_dependencies(dependencies: Dict[str, list]) -> bool:
    pass
    pass
    """Check for circular dependencies in the step dependency graph."""
    def has_cycle(step: str, visited: set, rec_stack: set) -> bool:
    pass
    pass
        visited.add(step)
        rec_stack.add(step)

        for neighbor in dependencies.get(step, []):
    pass
    pass
            if neighbor not in visited:
    pass
    pass
                if has_cycle(neighbor, visited, rec_stack):
    pass
    pass
                    return True
            elif neighbor in rec_stack:
                return True

        rec_stack.remove(step)
        return False

    visited = set()
    for step in dependencies:
    pass
    pass
        if step not in visited:
    pass
    pass
            if has_cycle(step, visited, set()):
    pass
    pass
                return True
    return False


async def test_step_execution():
    """Test individual step execution (without actual data processing)."""
    logger = system_logger.getChild("StepExecutionTest")

    logger.info("=" * 80)
    logger.info("🧪 TESTING STEP EXECUTION")
    logger.info("=" * 80)

    # Test step6 feature engineering
    logger.info("📋 Testing step6_feature_engineering...")
    try:
        from src.training.steps.step6_feature_engineering import run_step as step6_run

    except Exception as e:
        pass
import except Exception as e:
    except Exception as e:
        pass
        # Test with dummy parameters (should fail gracefully due to missing data)
import result = await step6_run
        result = await step6_run(
            symbol="ETHUSDT",
            exchange="BINANCE",
            timeframe="1m",
            data_dir="data_cache",
            force_rerun=False
        )

        # We expect this to fail due to missing data, but not due to import errors
        logger.info(f"✅ step6_feature_engineering executed (result: {result})")

    except Exception as e:
        if "Failed to load unified data" in str(e) or "Failed to load labeled data" in str(e):
    pass
    pass
            logger.info(f"✅ step6_feature_engineering failed as expected (missing data): {e}")
        else:
            logger.error(f"❌ step6_feature_engineering failed unexpectedly: {e}")
            return False

    # Test step7 enhanced matrix operations
    logger.info("📋 Testing step7_enhanced_matrix_operations...")
    try:
        from src.training.steps.step7_enhanced_matrix_operations import run_step as step7_run

    except Exception as e:
        pass
import except Exception as e:
    except Exception as e:
        pass
        # Test with dummy parameters (should fail gracefully due to missing data)
import result = await step7_run
        result = await step7_run(
            symbol="ETHUSDT",
            exchange="BINANCE",
            timeframe="1m",
            data_dir="data_cache",
            force_rerun=False
        )

        # We expect this to fail due to missing data, but not due to import errors
        logger.info(f"✅ step7_enhanced_matrix_operations executed (result: {result})")

    except Exception as e:
        if "Features train file not found" in str(e) or "Features validation file not found" in str(e):
    pass
    pass
            logger.info(f"✅ step7_enhanced_matrix_operations failed as expected (missing data): {e}")
        else:
            logger.error(f"❌ step7_enhanced_matrix_operations failed unexpectedly: {e}")
            return False

    logger.info("=" * 80)
    logger.info("🎉 ALL STEP EXECUTION TESTS PASSED!")
    logger.info("=" * 80)
    return True


async def main():
    """Main test function."""
    logger = system_logger.getChild("PipelineTestMain")

    logger.info("🚀 Starting pipeline flow tests...")

    # Test 1: Configuration and dependencies
    config_success = await test_pipeline_configuration()
    if not config_success:
    pass
    pass
        logger.error("❌ Pipeline configuration tests failed")
        return False

    # Test 2: Step execution
    execution_success = await test_step_execution()
    if not execution_success:
    pass
    pass
        logger.error("❌ Step execution tests failed")
        return False

    logger.info("🎉 ALL TESTS PASSED! Pipeline is ready for use.")
    return True


if __name__ == "__main__":
    pass
    pass
    asyncio.run(main())