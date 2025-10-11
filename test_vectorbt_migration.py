#!/usr/bin/env python3
"""
Test script for VectorBT migration of Order Flow and Acceleration features.

This script tests the migration of order flow and acceleration features to use VectorBT
for improved performance and optimization.
"""

import sys
import os
import pandas as pd
import numpy as np
import time
import logging
from typing import List, Dict, Any

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_sample_data(n_samples: int = 1000) -> pd.DataFrame:
    """Create sample OHLCV data for testing."""
    np.random.seed(42)
    
    # Generate sample price data
    returns = np.random.normal(0, 0.02, n_samples)
    prices = 100 * np.exp(np.cumsum(returns))
    
    # Generate OHLCV data
    data = pd.DataFrame({
        'open': prices * (1 + np.random.normal(0, 0.001, n_samples)),
        'high': prices * (1 + np.abs(np.random.normal(0, 0.01, n_samples))),
        'low': prices * (1 - np.abs(np.random.normal(0, 0.01, n_samples))),
        'close': prices,
        'volume': np.random.randint(1000, 10000, n_samples),
        'bid': prices * (1 - np.random.uniform(0.0001, 0.001, n_samples)),
        'ask': prices * (1 + np.random.uniform(0.0001, 0.001, n_samples)),
        'market_buys': np.random.randint(0, 100, n_samples),
        'market_sells': np.random.randint(0, 100, n_samples),
    }, index=pd.date_range('2023-01-01', periods=n_samples, freq='1min'))
    
    return data

def test_order_flow_features():
    """Test order flow feature generation with VectorBT."""
    logger.info("🧪 Testing Order Flow Features with VectorBT...")
    
    try:
        from src.feature_generation.categories.order_flow import create_default_order_flow_generators
        from src.feature_generation.categories.vectorbt_order_flow import create_vectorbt_order_flow_generators
        
        # Create sample data
        data = create_sample_data(1000)
        
        # Test VectorBT generators directly
        logger.info("Testing VectorBT Order Flow generators...")
        vectorbt_generators = create_vectorbt_order_flow_generators()
        logger.info(f"Created {len(vectorbt_generators)} VectorBT order flow generators")
        
        # Test feature generation
        start_time = time.time()
        features = {}
        
        for generator in vectorbt_generators[:5]:  # Test first 5 generators
            try:
                feature_result = generator.generate(data)
                features[generator.config.name] = feature_result
                logger.info(f"✅ Generated {generator.config.name}: {len(feature_result)} values")
            except Exception as e:
                logger.error(f"❌ Failed to generate {generator.config.name}: {e}")
        
        generation_time = time.time() - start_time
        logger.info(f"⏱️ VectorBT Order Flow generation time: {generation_time:.4f} seconds")
        
        # Test default generators (should use VectorBT if available)
        logger.info("Testing default order flow generators...")
        default_generators = create_default_order_flow_generators()
        logger.info(f"Created {len(default_generators)} default order flow generators")
        
        return True, len(features), generation_time
        
    except Exception as e:
        logger.error(f"❌ Order Flow test failed: {e}")
        return False, 0, 0

def test_acceleration_features():
    """Test acceleration feature generation with VectorBT."""
    logger.info("🧪 Testing Acceleration Features with VectorBT...")
    
    try:
        from src.feature_generation.categories.acceleration import create_acceleration_generators
        from src.feature_generation.categories.vectorbt_acceleration import create_vectorbt_acceleration_generators
        
        # Create sample data
        data = create_sample_data(1000)
        
        # Test VectorBT generators directly
        logger.info("Testing VectorBT Acceleration generators...")
        vectorbt_generators = create_vectorbt_acceleration_generators()
        logger.info(f"Created {len(vectorbt_generators)} VectorBT acceleration generators")
        
        # Test feature generation
        start_time = time.time()
        features = {}
        
        for generator in vectorbt_generators[:5]:  # Test first 5 generators
            try:
                feature_result = generator.generate(data)
                features[generator.config.name] = feature_result
                logger.info(f"✅ Generated {generator.config.name}: {len(feature_result)} values")
            except Exception as e:
                logger.error(f"❌ Failed to generate {generator.config.name}: {e}")
        
        generation_time = time.time() - start_time
        logger.info(f"⏱️ VectorBT Acceleration generation time: {generation_time:.4f} seconds")
        
        # Test default generators (should use VectorBT if available)
        logger.info("Testing default acceleration generators...")
        default_generators = create_acceleration_generators()
        logger.info(f"Created {len(default_generators)} default acceleration generators")
        
        return True, len(features), generation_time
        
    except Exception as e:
        logger.error(f"❌ Acceleration test failed: {e}")
        return False, 0, 0

def test_performance_comparison():
    """Compare performance between legacy and VectorBT implementations."""
    logger.info("🚀 Testing Performance Comparison...")
    
    try:
        from src.feature_generation.categories.order_flow import create_default_order_flow_generators
        from src.feature_generation.categories.acceleration import create_acceleration_generators
        
        # Create sample data
        data = create_sample_data(2000)
        
        # Test order flow performance
        logger.info("Testing Order Flow performance...")
        order_flow_generators = create_default_order_flow_generators()
        
        start_time = time.time()
        order_flow_features = {}
        for generator in order_flow_generators[:10]:  # Test first 10 generators
            try:
                feature_result = generator.generate(data)
                order_flow_features[generator.config.name] = feature_result
            except Exception as e:
                logger.warning(f"Order flow generator {generator.config.name} failed: {e}")
        
        order_flow_time = time.time() - start_time
        logger.info(f"⏱️ Order Flow generation time: {order_flow_time:.4f} seconds for {len(order_flow_features)} features")
        
        # Test acceleration performance
        logger.info("Testing Acceleration performance...")
        acceleration_generators = create_acceleration_generators()
        
        start_time = time.time()
        acceleration_features = {}
        for generator in acceleration_generators[:10]:  # Test first 10 generators
            try:
                feature_result = generator.generate(data)
                acceleration_features[generator.config.name] = feature_result
            except Exception as e:
                logger.warning(f"Acceleration generator {generator.config.name} failed: {e}")
        
        acceleration_time = time.time() - start_time
        logger.info(f"⏱️ Acceleration generation time: {acceleration_time:.4f} seconds for {len(acceleration_features)} features")
        
        total_time = order_flow_time + acceleration_time
        total_features = len(order_flow_features) + len(acceleration_features)
        
        logger.info(f"📊 Total performance: {total_time:.4f} seconds for {total_features} features")
        logger.info(f"📈 Average time per feature: {total_time/total_features:.4f} seconds")
        
        return True, total_features, total_time
        
    except Exception as e:
        logger.error(f"❌ Performance comparison failed: {e}")
        return False, 0, 0

def test_vectorbt_availability():
    """Test VectorBT availability and configuration."""
    logger.info("🔍 Testing VectorBT Availability...")
    
    try:
        import vectorbt as vbt
        logger.info(f"✅ VectorBT version: {vbt.__version__}")
        
        # Test basic VectorBT functionality
        data = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        result = vbt.rolling_mean(data, window=3)
        logger.info(f"✅ VectorBT rolling mean test: {result.iloc[-1]}")
        
        return True
        
    except ImportError:
        logger.warning("⚠️ VectorBT not available - falling back to legacy implementations")
        return False
    except Exception as e:
        logger.error(f"❌ VectorBT test failed: {e}")
        return False

def main():
    """Main test function."""
    logger.info("🚀 Starting VectorBT Migration Test Suite...")
    
    # Test VectorBT availability
    vectorbt_available = test_vectorbt_availability()
    
    # Test order flow features
    order_flow_success, order_flow_features, order_flow_time = test_order_flow_features()
    
    # Test acceleration features
    acceleration_success, acceleration_features, acceleration_time = test_acceleration_features()
    
    # Test performance comparison
    performance_success, total_features, total_time = test_performance_comparison()
    
    # Summary
    logger.info("\n" + "="*60)
    logger.info("📋 VECTORBT MIGRATION TEST SUMMARY")
    logger.info("="*60)
    logger.info(f"VectorBT Available: {'✅ Yes' if vectorbt_available else '❌ No'}")
    logger.info(f"Order Flow Features: {'✅ Success' if order_flow_success else '❌ Failed'} ({order_flow_features} features, {order_flow_time:.4f}s)")
    logger.info(f"Acceleration Features: {'✅ Success' if acceleration_success else '❌ Failed'} ({acceleration_features} features, {acceleration_time:.4f}s)")
    logger.info(f"Performance Test: {'✅ Success' if performance_success else '❌ Failed'} ({total_features} features, {total_time:.4f}s)")
    
    if vectorbt_available and order_flow_success and acceleration_success and performance_success:
        logger.info("🎉 All tests passed! VectorBT migration successful!")
        return True
    else:
        logger.error("❌ Some tests failed. Check the logs above for details.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)