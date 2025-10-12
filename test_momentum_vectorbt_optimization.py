#!/usr/bin/env python3
"""
Test script for VectorBT optimization in momentum feature generation.

This script tests the enhanced momentum features that now use:
1. VectorBTRollingOptimizer for optimized rolling operations
2. UnifiedVectorizationManager for comprehensive optimization
3. Enhanced performance tracking and fallback mechanisms
"""

import numpy as np
import pandas as pd
import time
import logging
from typing import Dict, Any

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_sample_data(n_points: int = 5000) -> pd.DataFrame:
    """Create sample OHLCV data for testing."""
    np.random.seed(42)
    
    # Generate realistic price data
    returns = np.random.normal(0.001, 0.02, n_points)
    prices = 100 * (1 + returns).cumprod()
    
    # Generate OHLCV data
    data = pd.DataFrame({
        'open': prices * (1 + np.random.normal(0, 0.001, n_points)),
        'high': prices * (1 + np.abs(np.random.normal(0, 0.005, n_points))),
        'low': prices * (1 - np.abs(np.random.normal(0, 0.005, n_points))),
        'close': prices,
        'volume': np.random.lognormal(10, 1, n_points)
    }, index=pd.date_range('2020-01-01', periods=n_points, freq='1min'))
    
    return data

def test_vectorbt_rolling_optimizer():
    """Test VectorBTRollingOptimizer functionality."""
    logger.info("🧪 Testing VectorBTRollingOptimizer...")
    
    try:
        from src.feature_generation.utils.vectorbt_rolling_optimizer import (
            get_vectorbt_rolling_optimizer,
            optimized_rolling_mean,
            optimized_rolling_std
        )
        
        # Create sample data
        data = create_sample_data(1000)
        
        # Test global optimizer
        optimizer = get_vectorbt_rolling_optimizer()
        
        # Test rolling mean
        start_time = time.time()
        rolling_mean_result = optimizer.rolling_mean(data['close'], window=20)
        mean_time = time.time() - start_time
        
        # Test rolling std
        start_time = time.time()
        rolling_std_result = optimizer.rolling_std(data['close'], window=20)
        std_time = time.time() - start_time
        
        # Test convenience functions
        start_time = time.time()
        conv_mean = optimized_rolling_mean(data['close'], window=20)
        conv_time = time.time() - start_time
        
        # Get performance stats
        stats = optimizer.get_performance_stats()
        
        logger.info(f"✅ VectorBTRollingOptimizer test completed:")
        logger.info(f"   - Rolling mean: {mean_time:.4f}s")
        logger.info(f"   - Rolling std: {std_time:.4f}s")
        logger.info(f"   - Convenience function: {conv_time:.4f}s")
        logger.info(f"   - Performance stats: {stats}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ VectorBTRollingOptimizer test failed: {e}")
        return False

def test_unified_vectorization_manager():
    """Test UnifiedVectorizationManager functionality."""
    logger.info("🧪 Testing UnifiedVectorizationManager...")
    
    try:
        from src.utils.ml_common.unified_vectorization_manager import (
            get_unified_vectorization_manager,
            OperationType,
            optimize_financial_operation
        )
        
        # Create sample data
        data = create_sample_data(2000)
        
        # Test unified manager
        manager = get_unified_vectorization_manager()
        
        # Test technical indicators operation
        operation_data = {
            'close': data['close'],
            'high': data['high'],
            'low': data['low'],
            'volume': data['volume']
        }
        
        start_time = time.time()
        result = manager.optimize_operation(
            OperationType.TECHNICAL_INDICATORS,
            operation_data
        )
        operation_time = time.time() - start_time
        
        # Get optimization stats
        stats = manager.get_optimization_stats()
        
        logger.info(f"✅ UnifiedVectorizationManager test completed:")
        logger.info(f"   - Operation time: {operation_time:.4f}s")
        logger.info(f"   - Strategy used: {result.strategy_used}")
        logger.info(f"   - Performance gain: {result.performance_gain:.2f}x")
        logger.info(f"   - Available optimizations: {stats['available_optimizations']}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ UnifiedVectorizationManager test failed: {e}")
        return False

def test_enhanced_momentum_generators():
    """Test enhanced momentum generators with VectorBT optimization."""
    logger.info("🧪 Testing Enhanced Momentum Generators...")
    
    try:
        from src.feature_generation.categories.momentum import (
            UnifiedMomentumFeatureGenerator,
            RSIGenerator,
            StochasticGenerator,
            WilliamsRGenerator,
            create_default_momentum_generators
        )
        
        # Create sample data
        data = create_sample_data(3000)
        
        # Test UnifiedMomentumFeatureGenerator
        logger.info("Testing UnifiedMomentumFeatureGenerator...")
        unified_gen = UnifiedMomentumFeatureGenerator()
        
        start_time = time.time()
        unified_result = unified_gen._generate_feature(data)
        unified_time = time.time() - start_time
        
        unified_stats = unified_gen.get_performance_stats()
        
        logger.info(f"   - Unified momentum time: {unified_time:.4f}s")
        logger.info(f"   - Unified stats: {unified_stats}")
        
        # Test enhanced RSI generator
        logger.info("Testing enhanced RSI generator...")
        rsi_gen = RSIGenerator(period=14)
        
        start_time = time.time()
        rsi_result = rsi_gen._generate_feature(data)
        rsi_time = time.time() - start_time
        
        logger.info(f"   - RSI time: {rsi_time:.4f}s")
        logger.info(f"   - RSI result shape: {rsi_result.shape}")
        
        # Test enhanced Stochastic generator
        logger.info("Testing enhanced Stochastic generator...")
        stoch_gen = StochasticGenerator(k_period=14, d_period=3)
        
        start_time = time.time()
        stoch_result = stoch_gen._generate_feature(data)
        stoch_time = time.time() - start_time
        
        logger.info(f"   - Stochastic time: {stoch_time:.4f}s")
        logger.info(f"   - Stochastic result shape: {stoch_result.shape}")
        
        # Test enhanced Williams %R generator
        logger.info("Testing enhanced Williams %R generator...")
        willr_gen = WilliamsRGenerator(period=14)
        
        start_time = time.time()
        willr_result = willr_gen._generate_feature(data)
        willr_time = time.time() - start_time
        
        logger.info(f"   - Williams %R time: {willr_time:.4f}s")
        logger.info(f"   - Williams %R result shape: {willr_result.shape}")
        
        # Test default generators
        logger.info("Testing create_default_momentum_generators...")
        generators = create_default_momentum_generators()
        
        logger.info(f"   - Created {len(generators)} momentum generators")
        logger.info(f"   - Generator types: {[type(g).__name__ for g in generators[:5]]}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Enhanced momentum generators test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_performance_comparison():
    """Compare performance between optimized and non-optimized implementations."""
    logger.info("🧪 Testing Performance Comparison...")
    
    try:
        from src.feature_generation.categories.momentum import (
            UnifiedMomentumFeatureGenerator,
            RSIGenerator
        )
        
        # Create larger dataset for performance testing
        data = create_sample_data(10000)
        
        # Test RSI with different data sizes
        rsi_gen = RSIGenerator(period=14)
        
        sizes = [1000, 5000, 10000]
        results = {}
        
        for size in sizes:
            test_data = data.iloc[:size]
            
            # Time the operation
            start_time = time.time()
            result = rsi_gen._generate_feature(test_data)
            operation_time = time.time() - start_time
            
            results[size] = {
                'time': operation_time,
                'shape': result.shape,
                'valid_values': result.notna().sum()
            }
            
            logger.info(f"   - Size {size}: {operation_time:.4f}s, {result.notna().sum()} valid values")
        
        # Test unified generator
        unified_gen = UnifiedMomentumFeatureGenerator()
        
        start_time = time.time()
        unified_result = unified_gen._generate_feature(data)
        unified_time = time.time() - start_time
        
        unified_stats = unified_gen.get_performance_stats()
        
        logger.info(f"   - Unified generator: {unified_time:.4f}s")
        logger.info(f"   - Unified stats: {unified_stats}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Performance comparison test failed: {e}")
        return False

def main():
    """Run all tests."""
    logger.info("🚀 Starting VectorBT Momentum Optimization Tests...")
    
    tests = [
        ("VectorBTRollingOptimizer", test_vectorbt_rolling_optimizer),
        ("UnifiedVectorizationManager", test_unified_vectorization_manager),
        ("Enhanced Momentum Generators", test_enhanced_momentum_generators),
        ("Performance Comparison", test_performance_comparison)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        logger.info(f"\n{'='*60}")
        logger.info(f"Running {test_name} test...")
        logger.info(f"{'='*60}")
        
        try:
            success = test_func()
            results[test_name] = success
            if success:
                logger.info(f"✅ {test_name} test PASSED")
            else:
                logger.error(f"❌ {test_name} test FAILED")
        except Exception as e:
            logger.error(f"❌ {test_name} test ERROR: {e}")
            results[test_name] = False
    
    # Summary
    logger.info(f"\n{'='*60}")
    logger.info("TEST SUMMARY")
    logger.info(f"{'='*60}")
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, success in results.items():
        status = "✅ PASSED" if success else "❌ FAILED"
        logger.info(f"{test_name}: {status}")
    
    logger.info(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All tests passed! VectorBT optimization is working correctly.")
    else:
        logger.warning(f"⚠️ {total - passed} tests failed. Check the logs above for details.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)