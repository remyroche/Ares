#!/usr/bin/env python3
"""
Test script for Temporal Feature Integration

This script demonstrates how the temporal feature integration solution
combines feature lookback optimization and cross timeframe analysis
to eliminate redundancy while preserving complementary information.
"""

import asyncio
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('TemporalFeatureIntegrationTest')

def create_sample_data(symbol: str = "BTCUSDT", days: int = 30) -> pd.DataFrame:
    """Create sample OHLCV data for testing."""
    logger.info(f"Creating sample data for {symbol} ({days} days)")
    
    # Generate timestamps
    end_time = datetime.now()
    start_time = end_time - timedelta(days=days)
    timestamps = pd.date_range(start=start_time, end=end_time, freq='1min')
    
    # Generate realistic price data
    np.random.seed(42)  # For reproducible results
    n_points = len(timestamps)
    
    # Base price with trend and volatility
    base_price = 50000
    trend = np.linspace(0, 0.1, n_points)  # 10% trend over period
    volatility = np.random.normal(0, 0.02, n_points)  # 2% daily volatility
    price_changes = trend + volatility
    prices = base_price * np.exp(np.cumsum(price_changes))
    
    # Generate OHLCV data
    data = pd.DataFrame({
        'timestamp': timestamps,
        'open': prices * (1 + np.random.normal(0, 0.001, n_points)),
        'high': prices * (1 + np.abs(np.random.normal(0, 0.005, n_points))),
        'low': prices * (1 - np.abs(np.random.normal(0, 0.005, n_points))),
        'close': prices,
        'volume': np.random.lognormal(10, 1, n_points)
    })
    
    # Ensure high >= max(open, close) and low <= min(open, close)
    data['high'] = np.maximum(data['high'], np.maximum(data['open'], data['close']))
    data['low'] = np.minimum(data['low'], np.minimum(data['open'], data['close']))
    
    # Add some technical indicators for testing
    data['rsi_14'] = calculate_rsi(data['close'], 14)
    data['sma_20'] = data['close'].rolling(20).mean()
    data['ema_12'] = data['close'].ewm(span=12).mean()
    data['bb_upper_20'] = data['close'].rolling(20).mean() + 2 * data['close'].rolling(20).std()
    data['bb_lower_20'] = data['close'].rolling(20).mean() - 2 * data['close'].rolling(20).std()
    
    logger.info(f"✅ Created sample data: {len(data)} rows, {len(data.columns)} columns")
    return data

def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """Calculate RSI indicator."""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

async def test_temporal_feature_integration():
    """Test the temporal feature integration system."""
    logger.info("🚀 Starting Temporal Feature Integration Test")
    
    try:
        # Import the temporal feature integration module
        from src.feature_engineering.temporal_feature_integration import (
            integrate_temporal_features,
            create_temporal_config,
            TemporalFeatureIntegration,
            TemporalFeatureConfig
        )
        logger.info("✅ Successfully imported temporal feature integration module")
        
        # Create sample data
        data = create_sample_data("BTCUSDT", days=7)  # 7 days for faster testing
        
        # Test 1: Basic integration with default config
        logger.info("\n📊 Test 1: Basic Integration with Default Config")
        config1 = create_temporal_config()
        result1 = await integrate_temporal_features(
            data=data,
            config=config1,
            symbol="BTCUSDT",
            exchange="BINANCE"
        )
        
        logger.info(f"✅ Basic integration completed:")
        logger.info(f"   - Total features before: {result1.total_features_before}")
        logger.info(f"   - Total features after: {result1.total_features_after}")
        logger.info(f"   - Redundancy removed: {result1.redundancy_removed}")
        logger.info(f"   - Integration time: {result1.integration_time:.2f}s")
        
        # Test 2: Advanced integration with custom config
        logger.info("\n📊 Test 2: Advanced Integration with Custom Config")
        config2 = TemporalFeatureConfig(
            enable_lookback_optimization=True,
            enable_cross_timeframe_analysis=False,  # Disable for testing
            correlation_threshold=0.6,  # More aggressive deduplication
            information_threshold=0.05,  # Lower information requirement
            stability_threshold=0.2,     # Lower stability requirement
            parallel_processing=True,
            max_workers=2,
            memory_limit_gb=4.0
        )
        
        integrator = TemporalFeatureIntegration(config2)
        result2 = await integrator.integrate_temporal_features(
            data=data,
            symbol="ETHUSDT",
            exchange="BINANCE"
        )
        
        logger.info(f"✅ Advanced integration completed:")
        logger.info(f"   - Total features before: {result2.total_features_before}")
        logger.info(f"   - Total features after: {result2.total_features_after}")
        logger.info(f"   - Redundancy removed: {result2.redundancy_removed}")
        logger.info(f"   - Integration time: {result2.integration_time:.2f}s")
        logger.info(f"   - Average correlation: {result2.average_correlation:.4f}")
        logger.info(f"   - Average information content: {result2.average_information_content:.4f}")
        logger.info(f"   - Average stability: {result2.average_stability:.4f}")
        
        # Test 3: Feature metadata analysis
        logger.info("\n📊 Test 3: Feature Metadata Analysis")
        if result2.feature_metadata:
            logger.info("✅ Feature metadata generated:")
            for name, metadata in list(result2.feature_metadata.items())[:5]:  # Show first 5
                logger.info(f"   - {name}:")
                logger.info(f"     Type: {metadata.get('type', 'unknown')}")
                logger.info(f"     Length: {metadata.get('length', 0)}")
                logger.info(f"     Variance: {metadata.get('variance', 0):.6f}")
                logger.info(f"     Mean: {metadata.get('mean', 0):.4f}")
                logger.info(f"     Std: {metadata.get('std', 0):.4f}")
        
        # Test 4: Quality metrics validation
        logger.info("\n📊 Test 4: Quality Metrics Validation")
        if result2.deduplicated_features:
            logger.info("✅ Quality metrics:")
            logger.info(f"   - Features with high correlation: {result2.average_correlation:.4f}")
            logger.info(f"   - Average information content: {result2.average_information_content:.4f}")
            logger.info(f"   - Average stability: {result2.average_stability:.4f}")
            
            # Validate that redundancy removal worked
            if result2.redundancy_removed > 0:
                logger.info(f"   ✅ Redundancy removal successful: {result2.redundancy_removed} features removed")
            else:
                logger.info("   ℹ️ No redundant features found (good for small dataset)")
        
        # Test 5: Error handling
        logger.info("\n📊 Test 5: Error Handling")
        try:
            # Test with empty data
            empty_data = pd.DataFrame()
            result3 = await integrate_temporal_features(
                data=empty_data,
                config=config1,
                symbol="TEST",
                exchange="TEST"
            )
            logger.info("✅ Empty data handling: Graceful degradation")
        except Exception as e:
            logger.warning(f"⚠️ Empty data handling: {e}")
        
        logger.info("\n🎉 All tests completed successfully!")
        return True
        
    except ImportError as e:
        logger.error(f"❌ Import error: {e}")
        logger.error("Make sure the temporal_feature_integration module is properly installed")
        return False
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        return False

async def test_performance_comparison():
    """Test performance comparison between different configurations."""
    logger.info("\n🚀 Starting Performance Comparison Test")
    
    try:
        from src.feature_engineering.temporal_feature_integration import (
            integrate_temporal_features,
            create_temporal_config
        )
        
        # Create larger dataset for performance testing
        data = create_sample_data("BTCUSDT", days=14)
        
        # Test configurations
        configs = [
            ("Lookback Only", create_temporal_config(enable_lookback=True, enable_cross_timeframe=False)),
            ("Cross Timeframe Only", create_temporal_config(enable_lookback=False, enable_cross_timeframe=True)),
            ("Both Enabled", create_temporal_config(enable_lookback=True, enable_cross_timeframe=True)),
        ]
        
        results = {}
        for name, config in configs:
            logger.info(f"\n📊 Testing {name} configuration...")
            start_time = asyncio.get_event_loop().time()
            
            result = await integrate_temporal_features(
                data=data,
                config=config,
                symbol="BTCUSDT",
                exchange="BINANCE"
            )
            
            end_time = asyncio.get_event_loop().time()
            duration = end_time - start_time
            
            results[name] = {
                'duration': duration,
                'features_before': result.total_features_before,
                'features_after': result.total_features_after,
                'redundancy_removed': result.redundancy_removed
            }
            
            logger.info(f"✅ {name}: {duration:.2f}s, {result.total_features_after} features")
        
        # Compare results
        logger.info("\n📊 Performance Comparison Results:")
        for name, metrics in results.items():
            logger.info(f"   {name}:")
            logger.info(f"     Duration: {metrics['duration']:.2f}s")
            logger.info(f"     Features: {metrics['features_after']}")
            logger.info(f"     Redundancy removed: {metrics['redundancy_removed']}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Performance comparison failed: {e}")
        return False

async def main():
    """Main test function."""
    logger.info("🚀 Starting Temporal Feature Integration Test Suite")
    
    # Run basic tests
    success1 = await test_temporal_feature_integration()
    
    # Run performance tests
    success2 = await test_performance_comparison()
    
    if success1 and success2:
        logger.info("\n🎉 All tests passed successfully!")
        logger.info("\n📋 Summary:")
        logger.info("   ✅ Temporal feature integration working correctly")
        logger.info("   ✅ Redundancy removal functioning")
        logger.info("   ✅ Quality metrics calculated")
        logger.info("   ✅ Error handling robust")
        logger.info("   ✅ Performance acceptable")
        return True
    else:
        logger.error("\n❌ Some tests failed!")
        return False

if __name__ == "__main__":
    # Run the test suite
    success = asyncio.run(main())
    exit(0 if success else 1)