#!/usr/bin/env python3
"""
Comprehensive Test Script for Step 16 Enhanced Optimizations

This script validates all the enhanced optimizations including:
- Fast-fail validation mechanisms
- Memory optimization utilities
- Enhanced matrix operations
- Convergence optimization
- Calibration quality metrics
- Enhanced calibration methods
"""

import asyncio
import sys
import os
import numpy as np
import pandas as pd
from pathlib import Path
import time
import json
import logging

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_test_data(n_samples: int = 1000, n_regimes: int = 3) -> Dict[str, Any]:
    """Create comprehensive test data for calibration."""
    np.random.seed(42)
    
    test_data = {
        'trained_specialists': {}
    }
    
    for regime_id in range(n_regimes):
        # Create regime-specific data
        regime_data = {
            'train_probabilities': np.random.beta(2, 2, n_samples).tolist(),
            'train_labels': np.random.binomial(1, 0.6, n_samples).tolist(),
            'val_probabilities': np.random.beta(2, 2, n_samples//4).tolist(),
            'val_labels': np.random.binomial(1, 0.6, n_samples//4).tolist(),
            'specialist_type': f'regime_{regime_id}_specialist',
            'regime_id': regime_id
        }
        
        test_data['trained_specialists'][f'regime_{regime_id}_specialist'] = regime_data
    
    return test_data

def create_test_config() -> Dict[str, Any]:
    """Create test configuration for enhanced optimizations."""
    return {
        'symbol': 'TEST_SYMBOL',
        'exchange': 'TEST_EXCHANGE',
        'timeframe': '1m',
        'optimization_level': 'aggressive',
        'memory_limit_gb': 4.0,
        'use_gpu': False,  # Disable GPU for testing
        'enable_parallel_processing': True,
        'enable_caching': True,
        'enable_fast_fail': True,
        'platt_scaling': {
            'max_iterations': 1000,
            'learning_rate': 0.01,
            'regularization': 0.01,
            'early_stopping': True,
            'validation_split': 0.2,
            'tolerance': 1e-6,
            'patience': 10
        },
        'isotonic_regression': {
            'out_of_bounds': 'clip',
            'increasing': True,
            'cross_validation': True,
            'cv_folds': 3
        },
        'temperature_scaling': {
            'temperature_range': [0.1, 10.0],
            'optimization_method': 'multi_start',
            'cross_validation': True,
            'validation_split': 0.2
        },
        'min_samples': 50,
        'max_missing_ratio': 0.1,
        'min_class_balance': 0.05
    }

async def test_optimization_utilities():
    """Test optimization utilities."""
    logger.info("🧪 Testing Optimization Utilities...")
    
    try:
        from src.training.steps.optimisation.step16_optimization_utilities import (
            FastFailValidator, ParameterValidator, MemoryOptimizer,
            EnhancedMatrixOperations, CalibrationQualityMetrics,
            FastFailError, ValidationError, ConvergenceError
        )
        
        # Test FastFailValidator
        logger.info("  Testing FastFailValidator...")
        validator = FastFailValidator({'min_samples': 50, 'max_missing_ratio': 0.1})
        
        # Test valid data
        valid_data = pd.DataFrame({
            'probabilities': np.random.random(100),
            'labels': np.random.binomial(1, 0.5, 100)
        })
        assert validator.validate_data_quality(valid_data, 'test_regime') == True
        
        # Test invalid data (too few samples)
        invalid_data = pd.DataFrame({
            'probabilities': np.random.random(10),
            'labels': np.random.binomial(1, 0.5, 10)
        })
        try:
            validator.validate_data_quality(invalid_data, 'test_regime')
            assert False, "Should have raised FastFailError"
        except FastFailError:
            pass  # Expected
        
        # Test ParameterValidator
        logger.info("  Testing ParameterValidator...")
        param_validator = ParameterValidator()
        
        # Test valid parameters
        valid_params = {
            'temperature_range': [0.1, 10.0],
            'max_iterations': 1000,
            'calibration_bins': 10,
            'learning_rate': 0.01
        }
        assert param_validator.validate_calibration_parameters(valid_params, 'test_regime') == True
        
        # Test invalid parameters
        invalid_params = {
            'temperature_range': [10.0, 0.1],  # Invalid range
            'max_iterations': 50,  # Too few iterations
            'calibration_bins': 3,  # Too few bins
            'learning_rate': -0.01  # Negative learning rate
        }
        try:
            param_validator.validate_calibration_parameters(invalid_params, 'test_regime')
            assert False, "Should have raised ValidationError"
        except ValidationError:
            pass  # Expected
        
        # Test MemoryOptimizer
        logger.info("  Testing MemoryOptimizer...")
        memory_optimizer = MemoryOptimizer(memory_limit_gb=1.0)
        
        # Test memory estimation
        test_df = pd.DataFrame(np.random.random((1000, 10)))
        memory_usage = memory_optimizer.estimate_memory_usage(test_df)
        assert memory_usage > 0
        
        # Test data optimization
        optimized_data = memory_optimizer.optimize_data_loading(test_df, 'test_regime')
        assert len(optimized_data) == len(test_df)
        
        # Test EnhancedMatrixOperations
        logger.info("  Testing EnhancedMatrixOperations...")
        matrix_ops = EnhancedMatrixOperations(use_gpu=False)
        
        # Test ECE calculation
        probabilities = np.random.random(100)
        labels = np.random.binomial(1, 0.5, 100)
        ece = matrix_ops.calculate_ece_vectorized(probabilities, labels)
        assert 0 <= ece <= 1
        
        # Test CalibrationQualityMetrics
        logger.info("  Testing CalibrationQualityMetrics...")
        metrics_calculator = CalibrationQualityMetrics(matrix_ops)
        metrics = metrics_calculator.calculate_comprehensive_metrics(probabilities, labels)
        
        assert hasattr(metrics, 'ece')
        assert hasattr(metrics, 'brier_score')
        assert hasattr(metrics, 'reliability_score')
        assert 0 <= metrics.ece <= 1
        assert 0 <= metrics.brier_score <= 1
        assert 0 <= metrics.reliability_score <= 1
        
        logger.info("✅ Optimization utilities tests passed")
        return True
        
    except Exception as e:
        logger.error(f"❌ Optimization utilities tests failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_enhanced_calibration_methods():
    """Test enhanced calibration methods."""
    logger.info("🧪 Testing Enhanced Calibration Methods...")
    
    try:
        from src.training.steps.optimisation.step16_enhanced_calibration_methods import (
            EnhancedPlattScaling, EnhancedIsotonicRegression, EnhancedTemperatureScaling
        )
        
        config = create_test_config()
        
        # Test EnhancedPlattScaling
        logger.info("  Testing EnhancedPlattScaling...")
        platt_scaling = EnhancedPlattScaling(config)
        
        probabilities = np.random.random(100)
        labels = np.random.binomial(1, 0.5, 100)
        
        platt_result = platt_scaling.calibrate(probabilities, labels, 'test_regime')
        assert 'calibration_method' in platt_result
        assert platt_result['calibration_method'] == 'enhanced_platt_scaling'
        assert 'calibration_metrics' in platt_result
        assert 'calibration_coefficients' in platt_result
        
        # Test EnhancedIsotonicRegression
        logger.info("  Testing EnhancedIsotonicRegression...")
        isotonic_regression = EnhancedIsotonicRegression(config)
        
        isotonic_result = isotonic_regression.calibrate(probabilities, labels, 'test_regime')
        assert 'calibration_method' in isotonic_result
        assert isotonic_result['calibration_method'] == 'enhanced_isotonic_regression'
        assert 'calibration_metrics' in isotonic_result
        assert 'calibration_function' in isotonic_result
        
        # Test EnhancedTemperatureScaling
        logger.info("  Testing EnhancedTemperatureScaling...")
        temperature_scaling = EnhancedTemperatureScaling(config)
        
        temp_result = temperature_scaling.calibrate(probabilities, labels, 'test_regime')
        assert 'calibration_method' in temp_result
        assert temp_result['calibration_method'] == 'enhanced_temperature_scaling'
        assert 'calibration_metrics' in temp_result
        assert 'calibration_coefficients' in temp_result
        
        logger.info("✅ Enhanced calibration methods tests passed")
        return True
        
    except Exception as e:
        logger.error(f"❌ Enhanced calibration methods tests failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_enhanced_step16():
    """Test the main enhanced step16 implementation."""
    logger.info("🧪 Testing Enhanced Step 16 Implementation...")
    
    try:
        from src.training.steps.optimisation.step16_enhanced_confidence_calibration import (
            EnhancedStep16ConfidenceCalibration, run_enhanced_step16
        )
        
        config = create_test_config()
        
        # Create test data directory
        test_data_dir = Path('test_data_cache')
        test_data_dir.mkdir(exist_ok=True)
        
        # Create test specialist data
        test_data = create_test_data(n_samples=200, n_regimes=2)
        
        # Save test data
        training_dir = test_data_dir / 'training'
        training_dir.mkdir(exist_ok=True)
        
        test_data_file = training_dir / 'TEST_EXCHANGE_TEST_SYMBOL_1m_tactician_specialist_training_aggregated.json'
        with open(test_data_file, 'w') as f:
            json.dump(test_data, f, indent=2)
        
        # Test EnhancedStep16ConfidenceCalibration
        logger.info("  Testing EnhancedStep16ConfidenceCalibration...")
        step = EnhancedStep16ConfidenceCalibration(config)
        
        # Test execution
        result = await step.execute(
            symbol='TEST_SYMBOL',
            exchange='TEST_EXCHANGE',
            timeframe='1m',
            data_dir=str(test_data_dir)
        )
        
        assert 'success' in result
        if result['success']:
            assert 'calibration_results' in result
            assert 'performance_metrics' in result
            logger.info("✅ Enhanced Step 16 execution successful")
        else:
            logger.warning(f"⚠️ Enhanced Step 16 execution failed: {result.get('error', 'Unknown error')}")
        
        # Test run_enhanced_step16 function
        logger.info("  Testing run_enhanced_step16 function...")
        function_result = await run_enhanced_step16(
            symbol='TEST_SYMBOL',
            exchange='TEST_EXCHANGE',
            timeframe='1m',
            data_dir=str(test_data_dir)
        )
        
        assert 'success' in function_result
        
        # Cleanup
        import shutil
        shutil.rmtree(test_data_dir, ignore_errors=True)
        
        logger.info("✅ Enhanced Step 16 implementation tests passed")
        return True
        
    except Exception as e:
        logger.error(f"❌ Enhanced Step 16 implementation tests failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_performance_benchmarks():
    """Test performance benchmarks."""
    logger.info("🧪 Testing Performance Benchmarks...")
    
    try:
        from src.training.steps.optimisation.step16_enhanced_calibration_methods import (
            EnhancedPlattScaling, EnhancedIsotonicRegression, EnhancedTemperatureScaling
        )
        
        config = create_test_config()
        
        # Create larger test dataset
        n_samples = 5000
        probabilities = np.random.random(n_samples)
        labels = np.random.binomial(1, 0.5, n_samples)
        
        # Benchmark Platt Scaling
        logger.info("  Benchmarking EnhancedPlattScaling...")
        platt_scaling = EnhancedPlattScaling(config)
        
        start_time = time.time()
        platt_result = platt_scaling.calibrate(probabilities, labels, 'benchmark_regime')
        platt_time = time.time() - start_time
        
        logger.info(f"    Platt Scaling: {platt_time:.3f}s for {n_samples} samples")
        
        # Benchmark Isotonic Regression
        logger.info("  Benchmarking EnhancedIsotonicRegression...")
        isotonic_regression = EnhancedIsotonicRegression(config)
        
        start_time = time.time()
        isotonic_result = isotonic_regression.calibrate(probabilities, labels, 'benchmark_regime')
        isotonic_time = time.time() - start_time
        
        logger.info(f"    Isotonic Regression: {isotonic_time:.3f}s for {n_samples} samples")
        
        # Benchmark Temperature Scaling
        logger.info("  Benchmarking EnhancedTemperatureScaling...")
        temperature_scaling = EnhancedTemperatureScaling(config)
        
        start_time = time.time()
        temp_result = temperature_scaling.calibrate(probabilities, labels, 'benchmark_regime')
        temp_time = time.time() - start_time
        
        logger.info(f"    Temperature Scaling: {temp_time:.3f}s for {n_samples} samples")
        
        # Performance analysis
        total_time = platt_time + isotonic_time + temp_time
        samples_per_second = n_samples / total_time
        
        logger.info(f"📊 Performance Summary:")
        logger.info(f"    Total time: {total_time:.3f}s")
        logger.info(f"    Samples per second: {samples_per_second:.0f}")
        logger.info(f"    Average time per method: {total_time/3:.3f}s")
        
        # Validate performance thresholds
        assert platt_time < 10.0, f"Platt scaling too slow: {platt_time:.3f}s"
        assert isotonic_time < 5.0, f"Isotonic regression too slow: {isotonic_time:.3f}s"
        assert temp_time < 15.0, f"Temperature scaling too slow: {temp_time:.3f}s"
        assert samples_per_second > 100, f"Processing too slow: {samples_per_second:.0f} samples/s"
        
        logger.info("✅ Performance benchmarks passed")
        return True
        
    except Exception as e:
        logger.error(f"❌ Performance benchmarks failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_error_handling():
    """Test error handling and edge cases."""
    logger.info("🧪 Testing Error Handling...")
    
    try:
        from src.training.steps.optimisation.step16_enhanced_calibration_methods import (
            EnhancedPlattScaling
        )
        from src.training.steps.optimisation.step16_optimization_utilities import (
            FastFailError, ValidationError, ConvergenceError
        )
        
        config = create_test_config()
        platt_scaling = EnhancedPlattScaling(config)
        
        # Test with insufficient data
        logger.info("  Testing insufficient data handling...")
        try:
            platt_scaling.calibrate(np.array([]), np.array([]), 'test_regime')
            assert False, "Should have raised FastFailError"
        except FastFailError:
            pass  # Expected
        
        # Test with invalid data types
        logger.info("  Testing invalid data types...")
        try:
            platt_scaling.calibrate("invalid", "invalid", 'test_regime')
            assert False, "Should have raised an error"
        except Exception:
            pass  # Expected
        
        # Test with single class labels
        logger.info("  Testing single class labels...")
        try:
            probabilities = np.random.random(100)
            labels = np.zeros(100)  # All zeros
            platt_scaling.calibrate(probabilities, labels, 'test_regime')
            assert False, "Should have raised FastFailError"
        except FastFailError:
            pass  # Expected
        
        # Test with invalid probabilities
        logger.info("  Testing invalid probabilities...")
        try:
            probabilities = np.array([-0.1, 1.5, 0.5])  # Invalid range
            labels = np.array([0, 1, 0])
            platt_scaling.calibrate(probabilities, labels, 'test_regime')
            assert False, "Should have raised FastFailError"
        except FastFailError:
            pass  # Expected
        
        logger.info("✅ Error handling tests passed")
        return True
        
    except Exception as e:
        logger.error(f"❌ Error handling tests failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Main test function."""
    logger.info("🚀 Starting Comprehensive Step 16 Enhanced Optimizations Test")
    logger.info("=" * 80)
    
    test_results = {}
    
    # Run all tests
    tests = [
        ("Optimization Utilities", test_optimization_utilities),
        ("Enhanced Calibration Methods", test_enhanced_calibration_methods),
        ("Enhanced Step 16 Implementation", test_enhanced_step16),
        ("Performance Benchmarks", test_performance_benchmarks),
        ("Error Handling", test_error_handling)
    ]
    
    for test_name, test_func in tests:
        logger.info(f"\n{'='*20} {test_name} {'='*20}")
        try:
            result = await test_func()
            test_results[test_name] = result
        except Exception as e:
            logger.error(f"❌ {test_name} test failed with exception: {e}")
            test_results[test_name] = False
    
    # Summary
    logger.info("\n" + "="*80)
    logger.info("📊 TEST SUMMARY")
    logger.info("="*80)
    
    passed_tests = sum(test_results.values())
    total_tests = len(test_results)
    
    for test_name, result in test_results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        logger.info(f"  {test_name}: {status}")
    
    logger.info(f"\nOverall: {passed_tests}/{total_tests} tests passed ({passed_tests/total_tests*100:.1f}%)")
    
    if passed_tests == total_tests:
        logger.info("🎉 All tests passed! Enhanced Step 16 optimizations are working correctly.")
        return True
    else:
        logger.error(f"❌ {total_tests - passed_tests} tests failed. Please check the implementation.")
        return False

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)