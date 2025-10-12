#!/usr/bin/env python3
"""
Direct test for order flow features integration without importing the full module structure.
"""

import pandas as pd
import numpy as np
import time
import logging
import sys
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_sample_data(n_points: int = 100) -> pd.DataFrame:
    """Create sample OHLCV data for testing."""
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=n_points, freq='1min')
    
    # Generate realistic price data
    price = 100 + np.cumsum(np.random.randn(n_points) * 0.01)
    volume = np.random.lognorm(10, 1, n_points)
    
    data = pd.DataFrame({
        'open': price + np.random.randn(n_points) * 0.001,
        'high': price + np.abs(np.random.randn(n_points)) * 0.002,
        'low': price - np.abs(np.random.randn(n_points)) * 0.002,
        'close': price,
        'volume': volume,
        'bid': price - np.random.rand(n_points) * 0.001,
        'ask': price + np.random.rand(n_points) * 0.001,
        'market_buys': volume * np.random.rand(n_points),
        'market_sells': volume * np.random.rand(n_points)
    }, index=dates)
    
    return data

def test_vectorbt_rolling_optimizer():
    """Test VectorBTRollingOptimizer directly."""
    logger.info("Testing VectorBTRollingOptimizer...")
    
    try:
        # Add the src directory to the path
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
        
        from feature_generation.utils.vectorbt_rolling_optimizer import VectorBTRollingOptimizer
        
        # Create optimizer
        optimizer = VectorBTRollingOptimizer(enable_gpu=False, enable_parallel=True)
        
        # Create test data
        data = create_sample_data(100)
        
        # Test rolling operations
        close = data['close']
        
        # Test rolling mean
        mean_result = optimizer.rolling_mean(close, window=20)
        logger.info(f"✅ Rolling mean: {len(mean_result)} points, {mean_result.notna().sum()} valid values")
        
        # Test rolling std
        std_result = optimizer.rolling_std(close, window=20)
        logger.info(f"✅ Rolling std: {len(std_result)} points, {std_result.notna().sum()} valid values")
        
        # Test rolling sum
        sum_result = optimizer.rolling_sum(data['volume'], window=20)
        logger.info(f"✅ Rolling sum: {len(sum_result)} points, {sum_result.notna().sum()} valid values")
        
        # Get performance stats
        stats = optimizer.get_performance_stats()
        logger.info(f"📊 Performance stats: {stats}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ VectorBTRollingOptimizer test failed: {e}")
        return False

def test_order_flow_generators_direct():
    """Test order flow generators by directly importing and using them."""
    logger.info("Testing order flow generators directly...")
    
    try:
        # Add the src directory to the path
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
        
        # Import the specific classes we need
        from feature_generation.categories.order_flow import (
            TakerBuyRatioGenerator,
            TakerSellRatioGenerator,
            MarketAggressionIndexGenerator,
            OrderFlowImbalanceGenerator
        )
        
        # Create test data
        data = create_sample_data(100)
        logger.info(f"📊 Created test data: {data.shape}")
        
        # Test each generator
        generators = [
            TakerBuyRatioGenerator(window=20),
            TakerSellRatioGenerator(window=20),
            MarketAggressionIndexGenerator(window=20),
            OrderFlowImbalanceGenerator(window=20)
        ]
        
        results = {}
        for generator in generators:
            try:
                feature_name = generator.config.name
                result = generator._generate_feature(data)
                results[feature_name] = result
                logger.info(f"✅ Generated {feature_name}: {len(result)} points, {result.notna().sum()} valid values")
                
                # Check if rolling_optimizer is being used
                if hasattr(generator, 'rolling_optimizer') and generator.rolling_optimizer is not None:
                    logger.info(f"   🔧 Using VectorBTRollingOptimizer for {feature_name}")
                else:
                    logger.warning(f"   ⚠️  Not using VectorBTRollingOptimizer for {feature_name}")
                    
            except Exception as e:
                logger.error(f"❌ Failed to generate {generator.__class__.__name__}: {e}")
                return False
        
        logger.info(f"✅ Successfully generated {len(results)} features")
        return True
        
    except Exception as e:
        logger.error(f"❌ Order flow generators test failed: {e}")
        return False

def test_batch_processing_direct():
    """Test batch processing functionality directly."""
    logger.info("Testing batch processing directly...")
    
    try:
        # Add the src directory to the path
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
        
        from feature_generation.categories.order_flow import process_order_flow_features_batch
        
        # Create test data
        data = create_sample_data(100)
        
        # Define batch processing configuration
        feature_configs = [
            {'name': 'taker_buy_ratio_5', 'type': 'taker_buy_ratio', 'window': 5, 'column': 'close'},
            {'name': 'taker_buy_ratio_20', 'type': 'taker_buy_ratio', 'window': 20, 'column': 'close'},
            {'name': 'market_aggression_10', 'type': 'market_aggression_index', 'window': 10, 'column': 'close'},
        ]
        
        # Process batch
        result_df = process_order_flow_features_batch(data, feature_configs)
        
        logger.info(f"✅ Batch processing completed: {len(result_df.columns)} features")
        logger.info(f"📊 Result shape: {result_df.shape}")
        
        # Check if any features were generated
        valid_features = result_df.notna().sum().sum()
        logger.info(f"📈 Total valid feature values: {valid_features}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Batch processing test failed: {e}")
        return False

def main():
    """Main test function."""
    logger.info("🚀 Starting direct order flow features integration test...")
    
    tests = [
        ("VectorBTRollingOptimizer", test_vectorbt_rolling_optimizer),
        ("Order Flow Generators", test_order_flow_generators_direct),
        ("Batch Processing", test_batch_processing_direct),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        logger.info(f"\n{'='*50}")
        logger.info(f"Running: {test_name}")
        logger.info('='*50)
        
        try:
            success = test_func()
            results[test_name] = success
        except Exception as e:
            logger.error(f"❌ Test {test_name} crashed: {e}")
            results[test_name] = False
    
    # Print summary
    logger.info("\n" + "="*60)
    logger.info("📋 TEST SUMMARY")
    logger.info("="*60)
    
    passed = 0
    total = len(results)
    
    for test_name, success in results.items():
        if success:
            logger.info(f"✅ {test_name}: PASSED")
            passed += 1
        else:
            logger.info(f"❌ {test_name}: FAILED")
    
    logger.info(f"\n🎯 Results: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All tests passed! VectorBT integration is working correctly.")
    else:
        logger.warning("⚠️  Some tests failed. Check the logs above for details.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)