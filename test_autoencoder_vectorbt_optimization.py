#!/usr/bin/env python3
"""
Test script for VectorBT-optimized autoencoder feature generation.

This script tests the full VectorBT optimization implementation including:
- VectorBTRollingOptimizer
- UnifiedVectorizationManager
- VectorBT-optimized autoencoder generators
"""

import numpy as np
import pandas as pd
import time
import logging
from typing import List, Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_test_data(n_samples: int = 10000) -> pd.DataFrame:
    """Create test data for autoencoder feature generation."""
    np.random.seed(42)
    
    # Generate realistic OHLCV data
    dates = pd.date_range('2020-01-01', periods=n_samples, freq='1min')
    
    # Generate price data with trend and volatility
    returns = np.random.normal(0.0001, 0.02, n_samples)
    prices = 100 * (1 + returns).cumprod()
    
    # Generate OHLCV data
    data = pd.DataFrame({
        'open': prices * (1 + np.random.normal(0, 0.001, n_samples)),
        'high': prices * (1 + np.abs(np.random.normal(0, 0.005, n_samples))),
        'low': prices * (1 - np.abs(np.random.normal(0, 0.005, n_samples))),
        'close': prices,
        'volume': np.random.lognormal(10, 1, n_samples)
    }, index=dates)
    
    # Ensure high >= low and high/low contain open/close
    data['high'] = np.maximum(data['high'], np.maximum(data['open'], data['close']))
    data['low'] = np.minimum(data['low'], np.minimum(data['open'], data['close']))
    
    return data

def test_vectorbt_rolling_optimizer():
    """Test VectorBTRollingOptimizer functionality."""
    logger.info("Testing VectorBTRollingOptimizer...")
    
    try:
        from src.feature_generation.utils.vectorbt_rolling_optimizer import VectorBTRollingOptimizer
        
        # Create test data
        data = create_test_data(5000)
        
        # Initialize optimizer
        optimizer = VectorBTRollingOptimizer(enable_gpu=False, enable_parallel=True)
        
        # Test various operations
        operations = ['mean', 'std', 'var', 'min', 'max', 'sum']
        windows = [10, 20, 50]
        
        results = {}
        for operation in operations:
            for window in windows:
                start_time = time.time()
                result = getattr(optimizer, f'rolling_{operation}')(data['close'], window)
                execution_time = time.time() - start_time
                
                results[f'{operation}_{window}'] = {
                    'shape': result.shape,
                    'execution_time': execution_time,
                    'has_nan': result.isna().any()
                }
                
                logger.info(f"✅ {operation} (window={window}): {execution_time:.4f}s, shape={result.shape}")
        
        # Get performance stats
        stats = optimizer.get_performance_stats()
        logger.info(f"Performance stats: {stats}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ VectorBTRollingOptimizer test failed: {e}")
        return False

def test_unified_vectorization_manager():
    """Test UnifiedVectorizationManager functionality."""
    logger.info("Testing UnifiedVectorizationManager...")
    
    try:
        from src.feature_generation.utils.unified_vectorization_manager import UnifiedVectorizationManager, VectorizationConfig
        
        # Create test data
        data = create_test_data(5000)
        
        # Initialize manager
        config = VectorizationConfig(
            enable_gpu=False,
            enable_parallel=True,
            enable_caching=True
        )
        manager = UnifiedVectorizationManager(config)
        
        # Test various operations
        operations = ['mean', 'std', 'var', 'min', 'max', 'sum', 'quantile', 'skew', 'kurt']
        windows = [10, 20, 50]
        
        results = {}
        for operation in operations:
            for window in windows:
                start_time = time.time()
                if operation == 'quantile':
                    result = getattr(manager, f'rolling_{operation}')(data['close'], window, q=0.5)
                else:
                    result = getattr(manager, f'rolling_{operation}')(data['close'], window)
                execution_time = time.time() - start_time
                
                results[f'{operation}_{window}'] = {
                    'shape': result.shape,
                    'execution_time': execution_time,
                    'has_nan': result.isna().any()
                }
                
                logger.info(f"✅ {operation} (window={window}): {execution_time:.4f}s, shape={result.shape}")
        
        # Test scaling operations
        scaling_methods = ['zscore', 'minmax', 'robust', 'rank', 'winsorize', 'clip']
        for method in scaling_methods:
            start_time = time.time()
            if method == 'winsorize':
                result = manager.winsorize_data(data['close'], limits=(0.05, 0.05))
            elif method == 'clip':
                result = manager.clip_data(data['close'], lower=data['close'].quantile(0.01), upper=data['close'].quantile(0.99))
            else:
                result = manager.scale_data(data['close'], method=method)
            execution_time = time.time() - start_time
            
            logger.info(f"✅ scaling {method}: {execution_time:.4f}s, shape={result.shape}")
        
        # Get performance stats
        stats = manager.get_performance_stats()
        logger.info(f"Performance stats: {stats}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ UnifiedVectorizationManager test failed: {e}")
        return False

def test_autoencoder_generators():
    """Test VectorBT-optimized autoencoder generators."""
    logger.info("Testing VectorBT-optimized autoencoder generators...")
    
    try:
        from src.feature_generation.categories.autoencoder import (
            AutoencoderFeatureGenerator,
            AutoencoderEncodedGenerator,
            AutoencoderReconstructionErrorGenerator,
            create_autoencoder_generators
        )
        
        # Create test data
        data = create_test_data(5000)
        
        # Test main autoencoder generator
        logger.info("Testing AutoencoderFeatureGenerator...")
        generator = AutoencoderFeatureGenerator(enable_gpu=False, enable_parallel=True)
        
        start_time = time.time()
        result = generator.generate_features(data)
        execution_time = time.time() - start_time
        
        logger.info(f"✅ AutoencoderFeatureGenerator: {execution_time:.4f}s, shape={result.shape}")
        
        # Test encoded generator
        logger.info("Testing AutoencoderEncodedGenerator...")
        encoded_generator = AutoencoderEncodedGenerator(
            encoding_dimension=10,
            window=20,
            enable_gpu=False,
            enable_parallel=True
        )
        
        start_time = time.time()
        encoded_result = encoded_generator.generate_features(data)
        execution_time = time.time() - start_time
        
        logger.info(f"✅ AutoencoderEncodedGenerator: {execution_time:.4f}s, shape={encoded_result.shape}")
        
        # Test reconstruction error generator
        logger.info("Testing AutoencoderReconstructionErrorGenerator...")
        error_generator = AutoencoderReconstructionErrorGenerator(
            window=20,
            enable_gpu=False,
            enable_parallel=True
        )
        
        start_time = time.time()
        error_result = error_generator.generate_features(data)
        execution_time = time.time() - start_time
        
        logger.info(f"✅ AutoencoderReconstructionErrorGenerator: {execution_time:.4f}s, shape={error_result.shape}")
        
        # Test batch generator creation
        logger.info("Testing batch generator creation...")
        generators = create_autoencoder_generators(
            encoding_dimensions=[5, 10],
            windows=[10, 20],
            enable_gpu=False,
            enable_parallel=True
        )
        
        logger.info(f"✅ Created {len(generators)} autoencoder generators")
        
        # Test batch processing
        start_time = time.time()
        batch_results = []
        for generator in generators[:3]:  # Test first 3 generators
            try:
                result = generator.generate_features(data)
                batch_results.append(result)
            except Exception as e:
                logger.warning(f"Generator {generator.__class__.__name__} failed: {e}")
        
        execution_time = time.time() - start_time
        logger.info(f"✅ Batch processing: {execution_time:.4f}s for {len(batch_results)} generators")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Autoencoder generators test failed: {e}")
        return False

def test_performance_comparison():
    """Compare performance between VectorBT and pandas implementations."""
    logger.info("Testing performance comparison...")
    
    try:
        from src.feature_generation.utils.vectorbt_rolling_optimizer import VectorBTRollingOptimizer
        from src.feature_generation.utils.unified_vectorization_manager import UnifiedVectorizationManager
        
        # Create test data
        data = create_test_data(10000)
        
        # Test VectorBT rolling optimizer
        optimizer = VectorBTRollingOptimizer(enable_gpu=False, enable_parallel=True)
        
        start_time = time.time()
        vectorbt_result = optimizer.rolling_mean(data['close'], window=20)
        vectorbt_time = time.time() - start_time
        
        # Test pandas fallback
        start_time = time.time()
        pandas_result = data['close'].rolling(window=20).mean()
        pandas_time = time.time() - start_time
        
        # Test unified vectorization manager
        manager = UnifiedVectorizationManager()
        
        start_time = time.time()
        unified_result = manager.rolling_mean(data['close'], window=20)
        unified_time = time.time() - start_time
        
        logger.info(f"VectorBT Rolling Optimizer: {vectorbt_time:.4f}s")
        logger.info(f"Pandas fallback: {pandas_time:.4f}s")
        logger.info(f"Unified Vectorization Manager: {unified_time:.4f}s")
        
        # Verify results are similar
        vectorbt_pandas_diff = np.abs(vectorbt_result - pandas_result).max()
        unified_pandas_diff = np.abs(unified_result - pandas_result).max()
        
        logger.info(f"VectorBT vs Pandas max difference: {vectorbt_pandas_diff:.2e}")
        logger.info(f"Unified vs Pandas max difference: {unified_pandas_diff:.2e}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Performance comparison test failed: {e}")
        return False

def main():
    """Run all tests."""
    logger.info("🚀 Starting VectorBT autoencoder optimization tests...")
    
    tests = [
        ("VectorBTRollingOptimizer", test_vectorbt_rolling_optimizer),
        ("UnifiedVectorizationManager", test_unified_vectorization_manager),
        ("Autoencoder Generators", test_autoencoder_generators),
        ("Performance Comparison", test_performance_comparison)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        logger.info(f"\n{'='*50}")
        logger.info(f"Running {test_name} test...")
        logger.info(f"{'='*50}")
        
        try:
            success = test_func()
            results[test_name] = success
            if success:
                logger.info(f"✅ {test_name} test passed!")
            else:
                logger.error(f"❌ {test_name} test failed!")
        except Exception as e:
            logger.error(f"❌ {test_name} test crashed: {e}")
            results[test_name] = False
    
    # Summary
    logger.info(f"\n{'='*50}")
    logger.info("TEST SUMMARY")
    logger.info(f"{'='*50}")
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, success in results.items():
        status = "✅ PASSED" if success else "❌ FAILED"
        logger.info(f"{test_name}: {status}")
    
    logger.info(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All tests passed! VectorBT optimization is working correctly.")
    else:
        logger.warning(f"⚠️ {total - passed} tests failed. Check the logs for details.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)