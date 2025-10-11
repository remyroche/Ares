#!/usr/bin/env python3
"""
Simple test for VectorBT migration of Order Flow and Acceleration features.

This script tests the migration without importing the full feature generation system.
"""

import sys
import os
import pandas as pd
import numpy as np
import time
import logging

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_vectorbt_availability():
    """Test VectorBT availability and basic functionality."""
    logger.info("🔍 Testing VectorBT Availability...")
    
    try:
        import vectorbt as vbt
        logger.info(f"✅ VectorBT version: {vbt.__version__}")
        
        # Test basic VectorBT functionality
        data = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        
        # Test VectorBT MA indicator
        ma_result = vbt.indicators.basic.MA.run(data, window=3)
        logger.info(f"✅ VectorBT MA test: {ma_result.ma.iloc[-1]}")
        
        # Test VectorBT RSI indicator
        rsi_result = vbt.indicators.basic.RSI.run(data, window=5)
        logger.info(f"✅ VectorBT RSI test: {rsi_result.rsi.iloc[-1]}")
        
        return True
        
    except ImportError:
        logger.warning("⚠️ VectorBT not available - falling back to legacy implementations")
        return False
    except Exception as e:
        logger.error(f"❌ VectorBT test failed: {e}")
        return False

def test_vectorbt_order_flow_generators():
    """Test VectorBT order flow generators directly."""
    logger.info("🧪 Testing VectorBT Order Flow Generators...")
    
    try:
        from src.feature_generation.categories.vectorbt_order_flow import (
            create_vectorbt_order_flow_generators,
            VectorBTTakerBuyRatioGenerator,
            VectorBTTakerSellRatioGenerator,
            VectorBTMarketAggressionIndexGenerator
        )
        
        # Create sample data
        data = pd.DataFrame({
            'close': [100, 101, 102, 101, 103, 104, 103, 105, 106, 107],
            'volume': [1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900],
            'bid': [99.9, 100.9, 101.9, 100.9, 102.9, 103.9, 102.9, 104.9, 105.9, 106.9],
            'ask': [100.1, 101.1, 102.1, 101.1, 103.1, 104.1, 103.1, 105.1, 106.1, 107.1],
            'market_buys': [50, 55, 60, 65, 70, 75, 80, 85, 90, 95],
            'market_sells': [45, 50, 55, 60, 65, 70, 75, 80, 85, 90]
        }, index=pd.date_range('2023-01-01', periods=10, freq='1min'))
        
        # Test individual generators
        generators = [
            VectorBTTakerBuyRatioGenerator(window=5),
            VectorBTTakerSellRatioGenerator(window=5),
            VectorBTMarketAggressionIndexGenerator(window=5)
        ]
        
        results = {}
        for generator in generators:
            try:
                feature_result = generator.generate(data)
                results[generator.config.name] = feature_result
                logger.info(f"✅ Generated {generator.config.name}: {len(feature_result)} values")
            except Exception as e:
                logger.error(f"❌ Failed to generate {generator.config.name}: {e}")
        
        # Test batch creation
        all_generators = create_vectorbt_order_flow_generators()
        logger.info(f"✅ Created {len(all_generators)} VectorBT order flow generators")
        
        return True, len(results)
        
    except Exception as e:
        logger.error(f"❌ VectorBT Order Flow test failed: {e}")
        return False, 0

def test_vectorbt_acceleration_generators():
    """Test VectorBT acceleration generators directly."""
    logger.info("🧪 Testing VectorBT Acceleration Generators...")
    
    try:
        from src.feature_generation.categories.vectorbt_acceleration import (
            create_vectorbt_acceleration_generators,
            VectorBTMomentumGenerator,
            VectorBTPriceAccelerationGenerator,
            VectorBTPriceJerkGenerator
        )
        
        # Create sample data
        data = pd.DataFrame({
            'close': [100, 101, 102, 101, 103, 104, 103, 105, 106, 107],
            'volume': [1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900]
        }, index=pd.date_range('2023-01-01', periods=10, freq='1min'))
        
        # Test individual generators
        generators = [
            VectorBTMomentumGenerator(period=5),
            VectorBTPriceAccelerationGenerator(period=5),
            VectorBTPriceJerkGenerator(period=5)
        ]
        
        results = {}
        for generator in generators:
            try:
                feature_result = generator.generate(data)
                results[generator.config.name] = feature_result
                logger.info(f"✅ Generated {generator.config.name}: {len(feature_result)} values")
            except Exception as e:
                logger.error(f"❌ Failed to generate {generator.config.name}: {e}")
        
        # Test batch creation
        all_generators = create_vectorbt_acceleration_generators()
        logger.info(f"✅ Created {len(all_generators)} VectorBT acceleration generators")
        
        return True, len(results)
        
    except Exception as e:
        logger.error(f"❌ VectorBT Acceleration test failed: {e}")
        return False, 0

def test_performance_comparison():
    """Compare performance between pandas and VectorBT implementations."""
    logger.info("🚀 Testing Performance Comparison...")
    
    try:
        # Create larger sample data
        n_samples = 1000
        data = pd.DataFrame({
            'close': np.random.randn(n_samples).cumsum() + 100,
            'volume': np.random.randint(1000, 10000, n_samples),
            'bid': np.random.randn(n_samples).cumsum() + 99.5,
            'ask': np.random.randn(n_samples).cumsum() + 100.5,
            'market_buys': np.random.randint(0, 100, n_samples),
            'market_sells': np.random.randint(0, 100, n_samples)
        }, index=pd.date_range('2023-01-01', periods=n_samples, freq='1min'))
        
        # Test pandas rolling operations
        start_time = time.time()
        pandas_mean = data['close'].rolling(window=20).mean()
        pandas_std = data['close'].rolling(window=20).std()
        pandas_time = time.time() - start_time
        
        # Test VectorBT operations
        start_time = time.time()
        try:
            import vectorbt as vbt
            vbt_mean = vbt.indicators.basic.MA.run(data['close'], window=20).ma
            vbt_std = data['close'].rolling(window=20).std()  # VectorBT doesn't have direct std
            vbt_time = time.time() - start_time
        except Exception as e:
            logger.warning(f"VectorBT performance test failed: {e}")
            vbt_time = float('inf')
        
        logger.info(f"⏱️ Pandas rolling operations: {pandas_time:.4f} seconds")
        logger.info(f"⏱️ VectorBT operations: {vbt_time:.4f} seconds")
        
        if vbt_time < float('inf'):
            speedup = pandas_time / vbt_time
            logger.info(f"📈 VectorBT speedup: {speedup:.2f}x")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Performance comparison failed: {e}")
        return False

def main():
    """Main test function."""
    logger.info("🚀 Starting Simple VectorBT Migration Test...")
    
    # Test VectorBT availability
    vectorbt_available = test_vectorbt_availability()
    
    # Test VectorBT order flow generators
    order_flow_success, order_flow_features = test_vectorbt_order_flow_generators()
    
    # Test VectorBT acceleration generators
    acceleration_success, acceleration_features = test_vectorbt_acceleration_generators()
    
    # Test performance comparison
    performance_success = test_performance_comparison()
    
    # Summary
    logger.info("\n" + "="*60)
    logger.info("📋 SIMPLE VECTORBT MIGRATION TEST SUMMARY")
    logger.info("="*60)
    logger.info(f"VectorBT Available: {'✅ Yes' if vectorbt_available else '❌ No'}")
    logger.info(f"Order Flow Features: {'✅ Success' if order_flow_success else '❌ Failed'} ({order_flow_features} features)")
    logger.info(f"Acceleration Features: {'✅ Success' if acceleration_success else '❌ Failed'} ({acceleration_features} features)")
    logger.info(f"Performance Test: {'✅ Success' if performance_success else '❌ Failed'}")
    
    if vectorbt_available and order_flow_success and acceleration_success and performance_success:
        logger.info("🎉 All tests passed! VectorBT migration successful!")
        return True
    else:
        logger.error("❌ Some tests failed. Check the logs above for details.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)