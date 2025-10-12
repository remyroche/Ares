#!/usr/bin/env python3
"""
Test script to verify VectorBT optimizations in entropy feature generation.
"""

import sys
import os
import numpy as np
import pandas as pd
import time
import logging

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_vectorbt_optimization():
    """Test VectorBT optimization in entropy feature generation."""
    
    try:
        # Import the entropy generators
        from src.feature_generation.categories.entropy import (
            EntropyFeatureGenerator,
            PriceEntropyGenerator,
            VolumeEntropyGenerator,
            ReturnEntropyGenerator,
            calculate_vectorized_entropy,
            VECTORBT_OPTIMIZATION_AVAILABLE
        )
        
        logger.info("✅ Successfully imported entropy generators")
        logger.info(f"VectorBT Optimization Available: {VECTORBT_OPTIMIZATION_AVAILABLE}")
        
        # Create sample data
        np.random.seed(42)
        dates = pd.date_range('2020-01-01', periods=1000, freq='1min')
        data = pd.DataFrame({
            'close': 100 + np.cumsum(np.random.randn(1000) * 0.01),
            'high': 100 + np.cumsum(np.random.randn(1000) * 0.01) + np.random.rand(1000) * 2,
            'low': 100 + np.cumsum(np.random.randn(1000) * 0.01) - np.random.rand(1000) * 2,
            'volume': np.random.lognormal(10, 1, 1000)
        }, index=dates)
        
        logger.info(f"Created sample data with shape: {data.shape}")
        
        # Test 1: Basic entropy calculation
        logger.info("\n🧪 Test 1: Basic entropy calculation")
        start_time = time.time()
        
        entropy_result = calculate_vectorized_entropy(data['close'], window=20, use_vectorbt=True)
        
        end_time = time.time()
        logger.info(f"✅ Entropy calculation completed in {end_time - start_time:.4f} seconds")
        logger.info(f"Result shape: {entropy_result.shape}")
        logger.info(f"Result stats: mean={entropy_result.mean():.4f}, std={entropy_result.std():.4f}")
        
        # Test 2: PriceEntropyGenerator
        logger.info("\n🧪 Test 2: PriceEntropyGenerator")
        start_time = time.time()
        
        price_entropy_gen = PriceEntropyGenerator(window=20)
        price_entropy_result = price_entropy_gen._generate_feature(data)
        
        end_time = time.time()
        logger.info(f"✅ PriceEntropyGenerator completed in {end_time - start_time:.4f} seconds")
        logger.info(f"Result shape: {price_entropy_result.shape}")
        logger.info(f"Uses VectorBT: {getattr(price_entropy_gen, 'use_vectorbt', False)}")
        
        # Test 3: VolumeEntropyGenerator
        logger.info("\n🧪 Test 3: VolumeEntropyGenerator")
        start_time = time.time()
        
        volume_entropy_gen = VolumeEntropyGenerator(window=20)
        volume_entropy_result = volume_entropy_gen._generate_feature(data)
        
        end_time = time.time()
        logger.info(f"✅ VolumeEntropyGenerator completed in {end_time - start_time:.4f} seconds")
        logger.info(f"Result shape: {volume_entropy_result.shape}")
        logger.info(f"Uses VectorBT: {getattr(volume_entropy_gen, 'use_vectorbt', False)}")
        
        # Test 4: ReturnEntropyGenerator
        logger.info("\n🧪 Test 4: ReturnEntropyGenerator")
        start_time = time.time()
        
        return_entropy_gen = ReturnEntropyGenerator(window=20)
        return_entropy_result = return_entropy_gen._generate_feature(data)
        
        end_time = time.time()
        logger.info(f"✅ ReturnEntropyGenerator completed in {end_time - start_time:.4f} seconds")
        logger.info(f"Result shape: {return_entropy_result.shape}")
        logger.info(f"Uses VectorBT: {getattr(return_entropy_gen, 'use_vectorbt', False)}")
        
        # Test 5: VectorBT Rolling Operations
        logger.info("\n🧪 Test 5: VectorBT Rolling Operations")
        if hasattr(price_entropy_gen, 'vectorbt_optimizer') and price_entropy_gen.vectorbt_optimizer:
            start_time = time.time()
            
            # Test various rolling operations
            rolling_mean = price_entropy_gen.vectorbt_optimizer.rolling_mean(data['close'], window=20)
            rolling_std = price_entropy_gen.vectorbt_optimizer.rolling_std(data['close'], window=20)
            rolling_var = price_entropy_gen.vectorbt_optimizer.rolling_var(data['close'], window=20)
            
            end_time = time.time()
            logger.info(f"✅ VectorBT rolling operations completed in {end_time - start_time:.4f} seconds")
            logger.info(f"Rolling mean shape: {rolling_mean.shape}")
            logger.info(f"Rolling std shape: {rolling_std.shape}")
            logger.info(f"Rolling var shape: {rolling_var.shape}")
            
            # Get performance stats
            stats = price_entropy_gen.vectorbt_optimizer.get_performance_stats()
            logger.info(f"VectorBT Performance Stats: {stats}")
        else:
            logger.warning("⚠️ VectorBT optimizer not available")
        
        # Test 6: UnifiedVectorizationManager
        logger.info("\n🧪 Test 6: UnifiedVectorizationManager")
        if hasattr(price_entropy_gen, 'unified_manager') and price_entropy_gen.unified_manager:
            try:
                from src.utils.ml_common.unified_vectorization_manager import OperationType, OperationConfig
                
                config = OperationConfig(
                    operation_type=OperationType.FEATURE_ENGINEERING,
                    data_size=len(data),
                    data_dimensions=data.shape
                )
                
                start_time = time.time()
                result = price_entropy_gen.unified_manager.optimize_operation(
                    OperationType.FEATURE_ENGINEERING,
                    data,
                    config,
                    feature_type="entropy"
                )
                end_time = time.time()
                
                logger.info(f"✅ UnifiedVectorizationManager completed in {end_time - start_time:.4f} seconds")
                logger.info(f"Strategy used: {result.strategy_used}")
                logger.info(f"Performance gain: {result.performance_gain:.2f}x")
                logger.info(f"Computation time: {result.computation_time:.4f} seconds")
            except Exception as e:
                logger.warning(f"⚠️ UnifiedVectorizationManager test failed: {e}")
        else:
            logger.warning("⚠️ UnifiedVectorizationManager not available")
        
        logger.info("\n🎉 All VectorBT optimization tests completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_performance_comparison():
    """Compare performance between VectorBT and pandas implementations."""
    
    try:
        from src.feature_generation.categories.entropy import calculate_vectorized_entropy
        
        # Create larger dataset for performance testing
        np.random.seed(42)
        data_size = 10000
        dates = pd.date_range('2020-01-01', periods=data_size, freq='1min')
        data = pd.DataFrame({
            'close': 100 + np.cumsum(np.random.randn(data_size) * 0.01)
        }, index=dates)
        
        logger.info(f"\n🚀 Performance comparison with {data_size} data points")
        
        # Test VectorBT implementation
        start_time = time.time()
        vectorbt_result = calculate_vectorized_entropy(data['close'], window=20, use_vectorbt=True)
        vectorbt_time = time.time() - start_time
        
        # Test pandas implementation
        start_time = time.time()
        pandas_result = calculate_vectorized_entropy(data['close'], window=20, use_vectorbt=False)
        pandas_time = time.time() - start_time
        
        speedup = pandas_time / vectorbt_time if vectorbt_time > 0 else 0
        
        logger.info(f"VectorBT time: {vectorbt_time:.4f} seconds")
        logger.info(f"Pandas time: {pandas_time:.4f} seconds")
        logger.info(f"Speedup: {speedup:.2f}x")
        
        # Verify results are similar
        correlation = np.corrcoef(vectorbt_result.dropna(), pandas_result.dropna())[0, 1]
        logger.info(f"Result correlation: {correlation:.4f}")
        
        return speedup > 1.0 and correlation > 0.95
        
    except Exception as e:
        logger.error(f"❌ Performance comparison failed: {e}")
        return False

if __name__ == "__main__":
    logger.info("🧪 Starting VectorBT optimization tests for entropy feature generation...")
    
    # Run basic functionality tests
    basic_tests_passed = test_vectorbt_optimization()
    
    # Run performance comparison
    performance_tests_passed = test_performance_comparison()
    
    if basic_tests_passed and performance_tests_passed:
        logger.info("\n✅ All tests passed! VectorBT optimization is working correctly.")
        sys.exit(0)
    else:
        logger.error("\n❌ Some tests failed. Please check the implementation.")
        sys.exit(1)