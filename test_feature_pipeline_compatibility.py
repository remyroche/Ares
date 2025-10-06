#!/usr/bin/env python3
"""
Feature Pipeline Compatibility Test

This script tests the compatibility and optimization of the feature engineering
pipeline components: Feature Bank, Normalizer, and Scaler.
"""

import sys
import os
import logging
import pandas as pd
import numpy as np
import time
from typing import Dict, Any, List

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_test_data(n_samples: int = 1000) -> pd.DataFrame:
    """Create test data for pipeline testing."""
    np.random.seed(42)
    
    # Generate synthetic OHLCV data
    dates = pd.date_range('2023-01-01', periods=n_samples, freq='1H')
    
    # Generate price data with trend and volatility
    returns = np.random.normal(0.0001, 0.02, n_samples)
    prices = 100 * np.exp(np.cumsum(returns))
    
    # Generate OHLCV data
    data = pd.DataFrame({
        'timestamp': dates,
        'open': prices * (1 + np.random.normal(0, 0.001, n_samples)),
        'high': prices * (1 + np.abs(np.random.normal(0, 0.01, n_samples))),
        'low': prices * (1 - np.abs(np.random.normal(0, 0.01, n_samples))),
        'close': prices,
        'volume': np.random.lognormal(10, 1, n_samples)
    })
    
    # Ensure high >= max(open, close) and low <= min(open, close)
    data['high'] = np.maximum(data['high'], np.maximum(data['open'], data['close']))
    data['low'] = np.minimum(data['low'], np.minimum(data['open'], data['close']))
    
    data.set_index('timestamp', inplace=True)
    return data

def test_feature_bank_compatibility():
    """Test Feature Bank functionality and compatibility."""
    logger.info("🧪 Testing Feature Bank compatibility...")
    
    try:
        from src.feature_generation.core.feature_bank import get_global_feature_bank
        
        # Get test data
        data = create_test_data(500)
        
        # Initialize feature bank
        feature_bank = get_global_feature_bank()
        
        # Test basic feature generation
        start_time = time.time()
        features = feature_bank.generate_features(
            data=data,
            categories=['momentum', 'volatility', 'volume'],
            lookback_optimization=False
        )
        generation_time = time.time() - start_time
        
        logger.info(f"✅ Feature Bank test passed")
        logger.info(f"   Generated {len(features.columns)} features in {generation_time:.3f}s")
        logger.info(f"   Features shape: {features.shape}")
        
        return {
            'success': True,
            'features_generated': len(features.columns),
            'generation_time': generation_time,
            'features_shape': features.shape
        }
        
    except Exception as e:
        logger.error(f"❌ Feature Bank test failed: {e}")
        return {'success': False, 'error': str(e)}

def test_normalizer_compatibility():
    """Test Normalizer functionality and compatibility."""
    logger.info("🧪 Testing Normalizer compatibility...")
    
    try:
        from src.training.steps.market_analysis.hybrid_nas_tas_regime.shared_utils.data_normalization import (
            NormalizationConfig, NormalizationMethod, create_data_normalizer
        )
        
        # Get test data
        data = create_test_data(500)
        
        # Create normalizer
        config = NormalizationConfig(
            method=NormalizationMethod.Z_SCORE,
            use_hardware_acceleration=True,
            use_matrix_operations=True
        )
        normalizer = create_data_normalizer(config)
        
        # Test normalization
        start_time = time.time()
        result = normalizer.normalize_data(data)
        normalization_time = time.time() - start_time
        
        if result.success:
            logger.info(f"✅ Normalizer test passed")
            logger.info(f"   Normalized {len(result.normalized_data.columns)} columns in {normalization_time:.3f}s")
            logger.info(f"   Hardware optimization: {result.hardware_optimization_applied}")
            logger.info(f"   Matrix operations: {result.matrix_operations_used}")
            
            return {
                'success': True,
                'columns_normalized': len(result.normalized_data.columns),
                'normalization_time': normalization_time,
                'hardware_optimization': result.hardware_optimization_applied,
                'matrix_operations': result.matrix_operations_used
            }
        else:
            logger.error(f"❌ Normalization failed: {result.error_message}")
            return {'success': False, 'error': result.error_message}
            
    except Exception as e:
        logger.error(f"❌ Normalizer test failed: {e}")
        return {'success': False, 'error': str(e)}

def test_scaler_compatibility():
    """Test Scaler functionality and compatibility."""
    logger.info("🧪 Testing Scaler compatibility...")
    
    try:
        from src.utils.intensity_scaler import get_intensity_config, apply_intensity_scaling
        
        # Test intensity configuration
        config = get_intensity_config()
        
        # Test scaling application
        test_config = {
            'model_training': {
                'max_trials': 100,
                'n_trials': 50,
                'epochs': 100,
                'batch_size': 1024
            },
            'validation': {
                'monte_carlo_samples': 1000,
                'ab_test_rounds': 5
            }
        }
        
        start_time = time.time()
        scaled_config = apply_intensity_scaling(test_config, config.intensity_percentage)
        scaling_time = time.time() - start_time
        
        logger.info(f"✅ Scaler test passed")
        logger.info(f"   Intensity percentage: {config.intensity_percentage}")
        logger.info(f"   Training mode: {config.training_mode}")
        logger.info(f"   Scaling time: {scaling_time:.3f}s")
        
        return {
            'success': True,
            'intensity_percentage': config.intensity_percentage,
            'training_mode': config.training_mode,
            'scaling_time': scaling_time
        }
        
    except Exception as e:
        logger.error(f"❌ Scaler test failed: {e}")
        return {'success': False, 'error': str(e)}

def test_matrix_operations_compatibility():
    """Test Matrix Operations functionality and compatibility."""
    logger.info("🧪 Testing Matrix Operations compatibility...")
    
    try:
        from src.utils.matrix_operations import get_unified_matrix_operations
        
        # Initialize matrix operations
        matrix_ops = get_unified_matrix_operations()
        
        # Test basic operations
        test_data = np.random.randn(100, 10)
        
        start_time = time.time()
        
        # Test matrix multiplication
        result = matrix_ops.matrix_multiply(test_data.T, test_data)
        
        # Test array optimization
        optimized = matrix_ops.optimize_array(test_data, dtype=np.float32)
        
        # Test covariance computation
        cov_matrix = matrix_ops.compute_covariance(test_data)
        
        operation_time = time.time() - start_time
        
        logger.info(f"✅ Matrix Operations test passed")
        logger.info(f"   Operations completed in {operation_time:.3f}s")
        logger.info(f"   Matrix ops available: {matrix_ops.is_available()}")
        
        return {
            'success': True,
            'operation_time': operation_time,
            'matrix_ops_available': matrix_ops.is_available()
        }
        
    except Exception as e:
        logger.error(f"❌ Matrix Operations test failed: {e}")
        return {'success': False, 'error': str(e)}

def test_hardware_manager_compatibility():
    """Test Hardware Manager functionality and compatibility."""
    logger.info("🧪 Testing Hardware Manager compatibility...")
    
    try:
        from src.utils.hardware.unified_hardware_manager import (
            get_unified_hardware_manager, WorkloadType, OptimizationLevel
        )
        
        # Initialize hardware manager
        hardware_manager = get_unified_hardware_manager(conservative_mode=True)
        
        # Test optimization for feature engineering workload
        start_time = time.time()
        success = hardware_manager.optimize_for_workload(
            WorkloadType.FEATURE_ENGINEERING,
            OptimizationLevel.BALANCED
        )
        optimization_time = time.time() - start_time
        
        # Get system status
        status = hardware_manager.get_system_status()
        
        logger.info(f"✅ Hardware Manager test passed")
        logger.info(f"   Optimization success: {success}")
        logger.info(f"   Optimization time: {optimization_time:.3f}s")
        logger.info(f"   System initialized: {status.get('initialized', False)}")
        
        return {
            'success': True,
            'optimization_success': success,
            'optimization_time': optimization_time,
            'system_initialized': status.get('initialized', False)
        }
        
    except Exception as e:
        logger.error(f"❌ Hardware Manager test failed: {e}")
        return {'success': False, 'error': str(e)}

def test_optimized_pipeline_integration():
    """Test the complete optimized pipeline integration."""
    logger.info("🧪 Testing Optimized Pipeline integration...")
    
    try:
        from src.feature_generation.utils.optimized_feature_pipeline import (
            get_optimized_feature_pipeline, PipelineConfig
        )
        
        # Create test data
        data = create_test_data(1000)
        
        # Configure pipeline
        config = PipelineConfig(
            enable_matrix_operations=True,
            enable_gpu_acceleration=True,
            enable_hardware_optimization=True,
            auto_normalize=True,
            normalization_method="zscore",
            enable_intensity_scaling=True
        )
        
        # Get pipeline
        pipeline = get_optimized_feature_pipeline(config)
        
        # Test complete pipeline
        start_time = time.time()
        result = pipeline.process_features(
            data=data,
            categories=['momentum', 'volatility', 'volume', 'trend'],
            target_column='close'
        )
        pipeline_time = time.time() - start_time
        
        if result.success:
            logger.info(f"✅ Optimized Pipeline test passed")
            logger.info(f"   Generated {len(result.features.columns)} features")
            logger.info(f"   Pipeline time: {pipeline_time:.3f}s")
            logger.info(f"   Memory usage: {result.memory_usage:.2f}MB")
            logger.info(f"   Hardware accelerations: {result.performance_stats.get('hardware_accelerations', 0)}")
            logger.info(f"   Vectorized operations: {result.performance_stats.get('vectorized_operations', 0)}")
            
            return {
                'success': True,
                'features_generated': len(result.features.columns),
                'pipeline_time': pipeline_time,
                'memory_usage': result.memory_usage,
                'hardware_accelerations': result.performance_stats.get('hardware_accelerations', 0),
                'vectorized_operations': result.performance_stats.get('vectorized_operations', 0)
            }
        else:
            logger.error(f"❌ Pipeline failed: {result.error_message}")
            return {'success': False, 'error': result.error_message}
            
    except Exception as e:
        logger.error(f"❌ Optimized Pipeline test failed: {e}")
        return {'success': False, 'error': str(e)}

def run_compatibility_tests():
    """Run all compatibility tests."""
    logger.info("🚀 Starting Feature Pipeline Compatibility Tests")
    logger.info("=" * 60)
    
    test_results = {}
    
    # Test individual components
    test_results['feature_bank'] = test_feature_bank_compatibility()
    test_results['normalizer'] = test_normalizer_compatibility()
    test_results['scaler'] = test_scaler_compatibility()
    test_results['matrix_operations'] = test_matrix_operations_compatibility()
    test_results['hardware_manager'] = test_hardware_manager_compatibility()
    
    # Test integrated pipeline
    test_results['optimized_pipeline'] = test_optimized_pipeline_integration()
    
    # Generate summary report
    logger.info("=" * 60)
    logger.info("📊 COMPATIBILITY TEST SUMMARY")
    logger.info("=" * 60)
    
    total_tests = len(test_results)
    passed_tests = sum(1 for result in test_results.values() if result.get('success', False))
    
    for test_name, result in test_results.items():
        status = "✅ PASS" if result.get('success', False) else "❌ FAIL"
        logger.info(f"{test_name:20} : {status}")
        
        if not result.get('success', False):
            logger.info(f"{'':20}   Error: {result.get('error', 'Unknown error')}")
    
    logger.info("=" * 60)
    logger.info(f"Total Tests: {total_tests}")
    logger.info(f"Passed: {passed_tests}")
    logger.info(f"Failed: {total_tests - passed_tests}")
    logger.info(f"Success Rate: {(passed_tests/total_tests)*100:.1f}%")
    logger.info("=" * 60)
    
    return test_results

if __name__ == "__main__":
    try:
        results = run_compatibility_tests()
        
        # Exit with error code if any tests failed
        failed_tests = [name for name, result in results.items() if not result.get('success', False)]
        if failed_tests:
            logger.error(f"❌ {len(failed_tests)} test(s) failed: {failed_tests}")
            sys.exit(1)
        else:
            logger.info("🎉 All compatibility tests passed!")
            sys.exit(0)
            
    except Exception as e:
        logger.error(f"❌ Test execution failed: {e}")
        sys.exit(1)