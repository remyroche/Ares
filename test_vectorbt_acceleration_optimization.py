#!/usr/bin/env python3
"""
Test script to demonstrate full VectorBT usage in feature generation acceleration.

This script tests:
1. UnifiedVectorizationManager
2. VectorBTRollingOptimizer
3. Optimized acceleration generators
4. Performance comparison
"""

import numpy as np
import pandas as pd
import time
import logging
from typing import List, Dict, Any

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_sample_data(n_samples: int = 10000) -> pd.DataFrame:
    """Create sample OHLCV data for testing."""
    np.random.seed(42)
    
    # Generate price data with trend and volatility
    dates = pd.date_range('2020-01-01', periods=n_samples, freq='1min')
    
    # Generate price with trend and noise
    trend = np.linspace(100, 200, n_samples)
    noise = np.random.normal(0, 0.5, n_samples)
    close_prices = trend + noise
    
    # Generate OHLCV data
    data = pd.DataFrame({
        'open': close_prices + np.random.normal(0, 0.1, n_samples),
        'high': close_prices + np.abs(np.random.normal(0, 0.2, n_samples)),
        'low': close_prices - np.abs(np.random.normal(0, 0.2, n_samples)),
        'close': close_prices,
        'volume': np.random.lognormal(10, 1, n_samples)
    }, index=dates)
    
    return data

def test_unified_vectorization_manager():
    """Test UnifiedVectorizationManager functionality."""
    logger.info("🧪 Testing UnifiedVectorizationManager...")
    
    try:
        from src.feature_generation.utils.unified_vectorization_manager import (
            get_unified_vectorization_manager, 
            UnifiedVectorizationConfig
        )
        
        # Create test data
        data = create_sample_data(5000)
        
        # Initialize manager
        config = UnifiedVectorizationConfig(
            enable_vectorbt=True,
            enable_gpu=False,
            enable_parallel=True,
            memory_limit_gb=4.0
        )
        manager = get_unified_vectorization_manager(config)
        
        # Test rolling operations
        logger.info("Testing rolling operations...")
        start_time = time.time()
        
        # Test individual rolling operation
        rolling_mean = manager.rolling_operation(data['close'], 'mean', 20)
        rolling_std = manager.rolling_operation(data['close'], 'std', 20)
        
        # Test batch rolling operations
        batch_operations = [
            {'name': 'close_mean_10', 'column': 'close', 'operation': 'mean', 'window': 10},
            {'name': 'close_std_10', 'column': 'close', 'operation': 'std', 'window': 10},
            {'name': 'close_mean_20', 'column': 'close', 'operation': 'mean', 'window': 20},
            {'name': 'close_std_20', 'column': 'close', 'operation': 'std', 'window': 20},
            {'name': 'volume_mean_10', 'column': 'volume', 'operation': 'mean', 'window': 10},
        ]
        
        batch_result = manager.batch_rolling_operations(data, batch_operations)
        
        execution_time = time.time() - start_time
        
        # Test DataFrame optimization
        optimized_data = manager.optimize_dataframe(data)
        
        # Get performance stats
        stats = manager.get_performance_stats()
        
        logger.info(f"✅ UnifiedVectorizationManager test completed in {execution_time:.3f}s")
        logger.info(f"Performance stats: {stats}")
        logger.info(f"Batch result shape: {batch_result.shape}")
        logger.info(f"Optimized data memory usage: {optimized_data.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ UnifiedVectorizationManager test failed: {e}")
        return False

def test_vectorbt_rolling_optimizer():
    """Test VectorBTRollingOptimizer functionality."""
    logger.info("🧪 Testing VectorBTRollingOptimizer...")
    
    try:
        from src.feature_generation.utils.vectorbt_rolling_optimizer import (
            get_vectorbt_rolling_optimizer,
            VectorBTRollingOptimizer
        )
        
        # Create test data
        data = create_sample_data(5000)
        
        # Initialize optimizer
        optimizer = get_vectorbt_rolling_optimizer(enable_gpu=False, enable_parallel=True)
        
        # Test various rolling operations
        logger.info("Testing rolling operations...")
        start_time = time.time()
        
        # Test different operations
        operations = ['mean', 'std', 'var', 'min', 'max', 'sum']
        windows = [10, 20, 50]
        
        results = {}
        for operation in operations:
            for window in windows:
                key = f'{operation}_{window}'
                results[key] = optimizer._rolling_operation(data['close'], operation, window)
        
        # Test correlation
        corr_result = optimizer.rolling_corr(data['close'], data['volume'], 20)
        
        execution_time = time.time() - start_time
        
        # Get performance stats
        stats = optimizer.get_performance_stats()
        
        logger.info(f"✅ VectorBTRollingOptimizer test completed in {execution_time:.3f}s")
        logger.info(f"Performance stats: {stats}")
        logger.info(f"Generated {len(results)} rolling features")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ VectorBTRollingOptimizer test failed: {e}")
        return False

def test_optimized_acceleration_generators():
    """Test optimized acceleration generators."""
    logger.info("🧪 Testing optimized acceleration generators...")
    
    try:
        from src.feature_generation.categories.acceleration_optimized import (
            create_optimized_acceleration_generators,
            OptimizedAccelerationFeatureGenerator
        )
        
        # Create test data
        data = create_sample_data(5000)
        
        # Test individual generator
        logger.info("Testing OptimizedAccelerationFeatureGenerator...")
        generator = OptimizedAccelerationFeatureGenerator()
        
        start_time = time.time()
        result = generator.generate_features(data)
        execution_time = time.time() - start_time
        
        logger.info(f"✅ OptimizedAccelerationFeatureGenerator completed in {execution_time:.3f}s")
        logger.info(f"Generated features shape: {result.shape}")
        
        # Test batch generators
        logger.info("Testing batch acceleration generators...")
        generators = create_optimized_acceleration_generators()
        
        start_time = time.time()
        batch_results = []
        for gen in generators[:5]:  # Test first 5 generators
            try:
                feature_result = gen.generate_feature(data)
                if isinstance(feature_result, pd.Series):
                    batch_results.append(feature_result)
            except Exception as e:
                logger.warning(f"Generator {gen.__class__.__name__} failed: {e}")
        
        execution_time = time.time() - start_time
        
        logger.info(f"✅ Batch acceleration generators completed in {execution_time:.3f}s")
        logger.info(f"Generated {len(batch_results)} features")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Optimized acceleration generators test failed: {e}")
        return False

def test_vectorbt_acceleration_generators():
    """Test VectorBT acceleration generators."""
    logger.info("🧪 Testing VectorBT acceleration generators...")
    
    try:
        from src.feature_generation.categories.vectorbt_acceleration import (
            create_vectorbt_acceleration_generators,
            VectorBTMomentumGenerator
        )
        
        # Create test data
        data = create_sample_data(5000)
        
        # Test individual generator
        logger.info("Testing VectorBTMomentumGenerator...")
        generator = VectorBTMomentumGenerator(period=20)
        
        start_time = time.time()
        result = generator.generate_feature(data)
        execution_time = time.time() - start_time
        
        logger.info(f"✅ VectorBTMomentumGenerator completed in {execution_time:.3f}s")
        logger.info(f"Generated feature shape: {result.shape}")
        
        # Test batch generators
        logger.info("Testing batch VectorBT acceleration generators...")
        generators = create_vectorbt_acceleration_generators()
        
        start_time = time.time()
        batch_results = []
        for gen in generators[:5]:  # Test first 5 generators
            try:
                feature_result = gen.generate_feature(data)
                if isinstance(feature_result, pd.Series):
                    batch_results.append(feature_result)
            except Exception as e:
                logger.warning(f"Generator {gen.__class__.__name__} failed: {e}")
        
        execution_time = time.time() - start_time
        
        logger.info(f"✅ Batch VectorBT acceleration generators completed in {execution_time:.3f}s")
        logger.info(f"Generated {len(batch_results)} features")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ VectorBT acceleration generators test failed: {e}")
        return False

def performance_comparison():
    """Compare performance between different approaches."""
    logger.info("🧪 Running performance comparison...")
    
    try:
        from src.feature_generation.utils.unified_vectorization_manager import get_unified_vectorization_manager
        from src.feature_generation.utils.vectorbt_rolling_optimizer import get_vectorbt_rolling_optimizer
        from src.feature_generation.categories.acceleration_optimized import OptimizedMomentumGenerator
        from src.feature_generation.categories.vectorbt_acceleration import VectorBTMomentumGenerator
        
        # Create test data
        data = create_sample_data(10000)
        
        # Test different approaches
        approaches = {
            'UnifiedVectorizationManager': lambda: get_unified_vectorization_manager().rolling_operation(data['close'], 'mean', 20),
            'VectorBTRollingOptimizer': lambda: get_vectorbt_rolling_optimizer().rolling_mean(data['close'], 20),
            'OptimizedMomentumGenerator': lambda: OptimizedMomentumGenerator(20).generate_feature(data),
            'VectorBTMomentumGenerator': lambda: VectorBTMomentumGenerator(20).generate_feature(data),
            'Pandas Fallback': lambda: data['close'].rolling(20).mean()
        }
        
        results = {}
        for name, func in approaches.items():
            try:
                start_time = time.time()
                result = func()
                execution_time = time.time() - start_time
                results[name] = {
                    'time': execution_time,
                    'success': True,
                    'result_shape': result.shape if hasattr(result, 'shape') else len(result)
                }
                logger.info(f"✅ {name}: {execution_time:.3f}s")
            except Exception as e:
                results[name] = {
                    'time': float('inf'),
                    'success': False,
                    'error': str(e)
                }
                logger.error(f"❌ {name}: {e}")
        
        # Find fastest approach
        successful_results = {k: v for k, v in results.items() if v['success']}
        if successful_results:
            fastest = min(successful_results.items(), key=lambda x: x[1]['time'])
            logger.info(f"🏆 Fastest approach: {fastest[0]} ({fastest[1]['time']:.3f}s)")
        
        return results
        
    except Exception as e:
        logger.error(f"❌ Performance comparison failed: {e}")
        return {}

def main():
    """Run all tests."""
    logger.info("🚀 Starting VectorBT acceleration optimization tests...")
    
    test_results = {}
    
    # Test UnifiedVectorizationManager
    test_results['UnifiedVectorizationManager'] = test_unified_vectorization_manager()
    
    # Test VectorBTRollingOptimizer
    test_results['VectorBTRollingOptimizer'] = test_vectorbt_rolling_optimizer()
    
    # Test optimized acceleration generators
    test_results['OptimizedAccelerationGenerators'] = test_optimized_acceleration_generators()
    
    # Test VectorBT acceleration generators
    test_results['VectorBTAccelerationGenerators'] = test_vectorbt_acceleration_generators()
    
    # Performance comparison
    performance_results = performance_comparison()
    
    # Summary
    logger.info("📊 Test Summary:")
    for test_name, success in test_results.items():
        status = "✅ PASSED" if success else "❌ FAILED"
        logger.info(f"  {test_name}: {status}")
    
    successful_tests = sum(test_results.values())
    total_tests = len(test_results)
    logger.info(f"Overall: {successful_tests}/{total_tests} tests passed")
    
    if performance_results:
        logger.info("📈 Performance Results:")
        for approach, result in performance_results.items():
            if result['success']:
                logger.info(f"  {approach}: {result['time']:.3f}s")
            else:
                logger.info(f"  {approach}: FAILED - {result['error']}")
    
    logger.info("🎉 VectorBT acceleration optimization tests completed!")

if __name__ == "__main__":
    main()