#!/usr/bin/env python3
"""Test script to verify SR optimization integration.

This script tests:
1. The new step2_5_sr_optimization step
2. That HMM clustering uses optimized parameters
3. That subsequent steps use optimized parameters
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path
import project_root = Path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.utils.logger import system_logger

import logger = system_logger.getChild
logger = system_logger.getChild("SROptimizationIntegrationTest")


async def test_sr_optimization_step():
    """Test the SR optimization step."""
    try:
        logger.info("🧪 Testing SR optimization step...")

    except Exception as e:
        pass
    except Exception as e:
        pass
        # Import the step
        from src.training.steps.step2_5_sr_optimization import run_step

        # Test configuration
import test_config = {
        test_config = {
            "sr_detection_optimization": {
                "n_trials": 5,  # Reduced for testing
                "cv_folds": 3,
                "test_size": 0.2,
                "optimization_timeout": 60,  # 1 minute for testing
                "performance_thresholds": {
                    "min_sharpe_ratio": 0.2,
                    "max_drawdown": -0.3,
                    "min_win_rate": 0.4,
                    "min_profit_factor": 1.1,
                    "min_signal_clarity": 0.05,
                }
            },
            "sr_breakout_predictor": {
                "use_optimized_params": True,
                "enable_detailed_reporting": True,
                "report_directory": "reports/sr_analysis",
                "report_format": "json",
                "report_retention_days": 30
            }
        }

        # Run the step
        success = await run_step(test_config)

        if success:
    pass
    pass
            logger.info("✅ SR optimization step test passed")
            return True
        else:
            logger.error("❌ SR optimization step test failed")
            return False

    except Exception as e:
        logger.error(f"❌ SR optimization step test error: {e}")
        return False


async def test_hmm_uses_optimized_params():
    """Test that HMM clustering uses optimized parameters."""
    try:
        logger.info("🧪 Testing HMM uses optimized parameters...")

    except Exception as e:
        pass
    except Exception as e:
        pass
        # Import HMM step
        from src.training.steps.step3_hmm_regime_discovery import HMMRegimeDiscoveryStep

        # Test configuration with optimized parameters
import test_config = {
        test_config = {
            "sr_breakout_predictor": {
                "use_optimized_params": True,
                "optimization_results_file": "optimization_results.json"
            },
            "sr_detection_optimization": {
                "optimized_method_weights": {"fractal": 0.8, "volume": 0.6},
                "optimized_strength_weights": {"volume": 0.7, "price": 0.3},
                "optimized_dbscan_params": {"eps": 0.1, "min_samples": 5},
                "optimized_timeframe_weights": {"1m": 0.4, "5m": 0.6},
                "optimized_advanced_params": {"fibonacci_sensitivity": 0.8}
            }
        }

        # Create HMM step
        hmm_step = HMMRegimeDiscoveryStep(test_config)

        # Check if SR predictor is initialized with optimized params
        if hasattr(hmm_step, 'sr_predictor'):
    pass
    pass
            sr_predictor = hmm_step.sr_predictor
            if hasattr(sr_predictor, 'use_optimized_params'):
    pass
    pass
                if sr_predictor.use_optimized_params:
    pass
    pass
                    logger.info("✅ HMM step uses optimized parameters")
                    return True
                else:
                    logger.error("❌ HMM step does not use optimized parameters")
                    return False
            else:
                logger.error("❌ SR predictor does not have use_optimized_params attribute")
                return False
        else:
            logger.error("❌ HMM step does not have SR predictor")
            return False

    except Exception as e:
        logger.error(f"❌ HMM optimized params test error: {e}")
        return False


async def test_subsequent_steps_use_optimized_params():
    """Test that subsequent steps use optimized parameters."""
    try:
        logger.info("🧪 Testing subsequent steps use optimized parameters...")

    except Exception as e:
        pass
    except Exception as e:
        pass
        # Test configuration with optimized parameters
        test_config = {
            "sr_breakout_predictor": {
                "use_optimized_params": True,
                "optimization_results_file": "optimization_results.json"
            },
            "sr_detection_optimization": {
                "optimized_method_weights": {"fractal": 0.8, "volume": 0.6},
                "optimized_strength_weights": {"volume": 0.7, "price": 0.3},
                "optimized_dbscan_params": {"eps": 0.1, "min_samples": 5},
                "optimized_timeframe_weights": {"1m": 0.4, "5m": 0.6},
                "optimized_advanced_params": {"fibonacci_sensitivity": 0.8}
            }
        }

        # Test various components that should use optimized parameters
        components_to_test = [
            ("SR Breakout Predictor", "src.tactician.sr_breakout_predictor", "SRBreakoutPredictor"),
            ("Analyst Unified Regime Classifier", "src.analyst.unified_regime_classifier", "UnifiedRegimeClassifier"),
            ("Tactician SR Backtesting Validator", "src.tactician.sr_backtesting_validator", "SRBacktestingValidator"),
        ]

        all_passed = True

        for component_name, module_path, class_name in components_to_test:
    pass
    pass
            try:
                # Import the module
    except Exception as e:
        pass
    except Exception as e:
        pass
                module = __import__(module_path, fromlist=[class_name])
                component_class = getattr(module, class_name)

                # Create instance with test config
                component = component_class(test_config)

                # Check if it uses optimized parameters
                if hasattr(component, 'sr_predictor'):
    pass
    pass
                    sr_predictor = component.sr_predictor
                    if hasattr(sr_predictor, 'use_optimized_params') and sr_predictor.use_optimized_params:
    pass
    pass
                        logger.info(f"✅ {component_name} uses optimized parameters")
                    else:
                        logger.error(f"❌ {component_name} does not use optimized parameters")
                        all_passed = False
                else:
                    logger.warning(f"⚠️ {component_name} does not have SR predictor")

            except Exception as e:
                logger.warning(f"⚠️ Could not test {component_name}: {e}")

        return all_passed

    except Exception as e:
        logger.error(f"❌ Subsequent steps test error: {e}")
        return False


async def test_pipeline_integration():
    """Test that the pipeline correctly includes the new step."""
    try:
        logger.info("🧪 Testing pipeline integration...")

    except Exception as e:
        pass
    except Exception as e:
        pass
        # Test step dependency validator
        from src.utils.step_dependency_validator import StepDependencyValidator

import validator = StepDependencyValidator
        validator = StepDependencyValidator()

        # Check if step2_5_sr_optimization is in dependencies
        if "step2_5_sr_optimization" in validator.step_dependencies:
    pass
    pass
            logger.info("✅ step2_5_sr_optimization found in step dependencies")
        else:
            logger.error("❌ step2_5_sr_optimization not found in step dependencies")
            return False

        # Check if step3_hmm_regime_discovery depends on step2_5_sr_optimization
        step3_deps = validator.step_dependencies.get("step3_hmm_regime_discovery", [])
        if "step2_5_sr_optimization" in step3_deps:
    pass
    pass
            logger.info("✅ step3_hmm_regime_discovery depends on step2_5_sr_optimization")
        else:
            logger.error("❌ step3_hmm_regime_discovery does not depend on step2_5_sr_optimization")
            return False

        # Test enhanced training manager
        from src.training.enhanced_training_manager import EnhancedTrainingManager

        # Check if step2_5_sr_optimization is in STEP_ORDER
import if "step2_5_sr_optimization" in EnhancedTrainingManager.STEP_ORDER:
        if "step2_5_sr_optimization" in EnhancedTrainingManager.STEP_ORDER:
    pass
    pass
            logger.info("✅ step2_5_sr_optimization found in STEP_ORDER")
        else:
            logger.error("❌ step2_5_sr_optimization not found in STEP_ORDER")
            return False

        # Check if step2_5_sr_optimization is in CRITICAL_ARTIFACTS
        if "step2_5_sr_optimization" in EnhancedTrainingManager.CRITICAL_ARTIFACTS:
    pass
    pass
            logger.info("✅ step2_5_sr_optimization found in CRITICAL_ARTIFACTS")
        else:
            logger.error("❌ step2_5_sr_optimization not found in CRITICAL_ARTIFACTS")
            return False

        return True

    except Exception as e:
        logger.error(f"❌ Pipeline integration test error: {e}")
        return False


async def run_all_tests():
    """Run all tests."""
    logger.info("🚀 Starting SR optimization integration tests...")

    tests = [
        ("SR Optimization Step", test_sr_optimization_step),
        ("HMM Uses Optimized Params", test_hmm_uses_optimized_params),
        ("Subsequent Steps Use Optimized Params", test_subsequent_steps_use_optimized_params),
        ("Pipeline Integration", test_pipeline_integration),
    ]

    results = {}

    for test_name, test_func in tests:
    pass
    pass
        logger.info(f"\\\n{'='*60}")
        logger.info(f"Running test: {test_name}")
        logger.info(f"{'='*60}")

        try:
            result = await test_func()
    except Exception as e:
        pass
    except Exception as e:
        pass
            results[test_name] = result

            if result:
    pass
    pass
                logger.info(f"✅ {test_name} PASSED")
            else:
                logger.error(f"❌ {test_name} FAILED")

        except Exception as e:
            logger.error(f"❌ {test_name} ERROR: {e}")
            results[test_name] = False

    # Summary
    logger.info(f"\\\n{'='*60}")
    logger.info("TEST SUMMARY")
    logger.info(f"{'='*60}")

    passed = sum(1 for result in results.values() if result)
    total = len(results)

    for test_name, result in results.items():
    pass
    pass
        status = "✅ PASSED" if result else "❌ FAILED"
        logger.info(f"{test_name:<40} {status}")

    logger.info(f"\\\nOverall: {passed}/{total} tests passed")

    if passed == total:
    pass
    pass
        logger.info("🎉 All tests passed! SR optimization integration is working correctly.")
        return True
    else:
        logger.error(f"❌ {total - passed} tests failed. Please check the implementation.")
        return False


if __name__ == "__main__":
    pass
    pass
    # Run all tests
    success = asyncio.run(run_all_tests())

    if success:
    pass
    pass
        print("\\\n🎉 All SR optimization integration tests passed!")
        sys.exit(0)
    else:
        print("\\\n❌ Some SR optimization integration tests failed!")
        sys.exit(1)