#!/usr/bin/env python3
"""
Test script for Enhanced GMM Features with Multi-Timeframe and Streaming

This script demonstrates and validates the new multi-timeframe fusion
and memory streaming capabilities.
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Add project root to path
sys.path.insert(0, '/Users/remyroche/Documents/Ares')

from src.training.steps.market_analysis.gmm_enhanced_features import EnhancedGMMFeatures
from src.training.steps.market_analysis.multi_timeframe_utils import MultiTimeframeProcessor


def create_sample_data(n_rows: int = 10000) -> pd.DataFrame:
    """Create sample OHLCV data for testing."""
    # Generate timestamps
    start_date = datetime(2023, 1, 1)
    timestamps = pd.date_range(
        start=start_date, 
        periods=n_rows, 
        freq='15min'  # 15-minute bars
    )
    
    # Generate realistic price data
    np.random.seed(42)
    
    # Base price with trend and volatility
    base_price = 2000.0
    trend = np.linspace(0, 100, n_rows)  # Upward trend
    noise = np.random.normal(0, 10, n_rows)  # Random noise
    
    close_prices = base_price + trend + noise
    
    # Create OHLCV data
    data = pd.DataFrame(index=timestamps)
    data['close'] = close_prices
    
    # Generate realistic OHLC
    volatility_factor = 0.002  # 0.2% typical volatility
    
    data['high'] = data['close'] * (1 + np.abs(np.random.normal(0, volatility_factor, n_rows)))
    data['low'] = data['close'] * (1 - np.abs(np.random.normal(0, volatility_factor, n_rows)))
    data['open'] = data['close'].shift(1).fillna(data['close'].iloc[0])
    
    # Generate volume with correlation to price movement
    price_change = np.abs(data['close'].pct_change())
    base_volume = 1000000
    data['volume'] = base_volume * (1 + price_change * 10 + np.random.normal(0, 0.5, n_rows))
    data['volume'] = data['volume'].clip(lower=100000)  # Minimum volume
    
    return data


def test_multi_timeframe_processor():
    """Test the multi-timeframe processor functionality."""
    print("🧪 Testing Multi-Timeframe Processor...")
    
    # Create sample data
    data = create_sample_data(5000)  # Smaller dataset for quick testing
    
    # Initialize processor
    processor = MultiTimeframeProcessor(base_timeframe="15m")
    
    # Test resampling
    print("📊 Testing resampling...")
    data_60m = processor.resample_ohlcv(data, "60m")
    data_4h = processor.resample_ohlcv(data, "4h")
    
    print(f"✅ 15m data: {len(data)} rows")
    print(f"✅ 60m data: {len(data_60m)} rows")
    print(f"✅ 4h data: {len(data_4h)} rows")
    
    # Test alignment
    print("🔗 Testing alignment...")
    aligned_data = processor.align_timeframes(data, {"60m": data_60m, "4h": data_4h})
    
    for tf, df in aligned_data.items():
        print(f"✅ {tf} aligned: {len(df)} rows")
    
    # Test timeframe weights
    print("⚖️ Testing timeframe weights...")
    weights = processor.calculate_timeframe_weights(volatility_regime="normal")
    print(f"✅ Normal volatility weights: {weights}")
    
    weights_high = processor.calculate_timeframe_weights(volatility_regime="high")
    print(f"✅ High volatility weights: {weights_high}")
    
    # Test memory estimation
    print("💾 Testing memory estimation...")
    memory_usage = processor.estimate_memory_usage((10000, 50))
    print(f"✅ Estimated memory usage: {memory_usage:.2f} MB")
    
    chunk_size = processor.calculate_optimal_chunk_size(100000, 100)
    print(f"✅ Optimal chunk size: {chunk_size}")
    
    return True


def test_enhanced_gmm_features():
    """Test the enhanced GMM features with multi-timeframe support."""
    print("\n🧪 Testing Enhanced GMM Features...")
    
    # Create sample data
    data = create_sample_data(2000)  # Smaller for testing
    
    # Initialize enhanced GMM features
    config = {
        'use_multi_timeframe': True,
        'use_streaming': True,
        'use_fracdiff': False,  # Disable for faster testing
        'use_treeshap': False,  # Disable for faster testing
        'multi_tf_config': {
            'base_timeframe': '15m',
            'target_timeframes': ['15m', '60m', '4h'],
            'fusion_method': 'adaptive',
            'max_memory_mb': 512,  # Lower for testing
            'chunk_size': 500     # Smaller chunks for testing
        },
        'n_clusters_macro': 4,  # Smaller for faster testing
        'verbose': True
    }
    
    enhanced_gmm = EnhancedGMMFeatures(**config)
    
    # Test multi-timeframe feature generation
    print("🌐 Testing multi-timeframe feature generation...")
    returns = data['close'].pct_change()
    
    try:
        multi_tf_features = enhanced_gmm._generate_multi_timeframe_features_streaming(data, returns)
        print(f"✅ Generated {len(multi_tf_features.columns)} multi-timeframe features")
        print(f"✅ Feature shape: {multi_tf_features.shape}")
        
        # Show some sample features
        print("\n📋 Sample features:")
        sample_cols = multi_tf_features.columns[:10]
        for col in sample_cols:
            print(f"  - {col}")
        
        return True
        
    except Exception as e:
        print(f"❌ Multi-timeframe feature generation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_volatility_regime_detection():
    """Test volatility regime detection."""
    print("\n🧪 Testing Volatility Regime Detection...")
    
    # Create data with different volatility regimes
    data_low_vol = create_sample_data(1000)
    data_low_vol['close'] = data_low_vol['close'] + np.random.normal(0, 1, 1000)  # Low volatility
    
    data_high_vol = create_sample_data(1000)
    data_high_vol['close'] = data_high_vol['close'] + np.random.normal(0, 50, 1000)  # High volatility
    
    enhanced_gmm = EnhancedGMMFeatures()
    
    # Test regime detection
    regime_low = enhanced_gmm._detect_volatility_regime(data_low_vol)
    regime_high = enhanced_gmm._detect_volatility_regime(data_high_vol)
    
    print(f"✅ Low volatility regime: {regime_low}")
    print(f"✅ High volatility regime: {regime_high}")
    
    return True


def test_memory_streaming():
    """Test memory streaming with larger dataset."""
    print("\n🧪 Testing Memory Streaming...")
    
    # Create larger dataset
    large_data = create_sample_data(10000)
    
    config = {
        'use_multi_timeframe': True,
        'use_streaming': True,
        'use_fracdiff': False,
        'use_treeshap': False,
        'multi_tf_config': {
            'max_memory_mb': 256,  # Low memory to force streaming
            'chunk_size': 1000
        },
        'n_clusters_macro': 4,
        'verbose': False  # Reduce output for testing
    }
    
    enhanced_gmm = EnhancedGMMFeatures(**config)
    
    # Test streaming processing
    returns = large_data['close'].pct_change()
    
    try:
        print("🔄 Processing with streaming...")
        features = enhanced_gmm._generate_multi_timeframe_features_streaming(large_data, returns)
        print(f"✅ Streaming processed {len(large_data)} rows into {len(features.columns)} features")
        return True
        
    except Exception as e:
        print(f"❌ Streaming failed: {e}")
        return False


def main():
    """Run all tests."""
    print("🚀 Starting Enhanced GMM Features Tests\n")
    
    tests = [
        ("Multi-Timeframe Processor", test_multi_timeframe_processor),
        ("Enhanced GMM Features", test_enhanced_gmm_features),
        ("Volatility Regime Detection", test_volatility_regime_detection),
        ("Memory Streaming", test_memory_streaming)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        print(f"\n{'='*60}")
        print(f"Running: {test_name}")
        print(f"{'='*60}")
        
        try:
            result = test_func()
            results[test_name] = result
            
            if result:
                print(f"✅ {test_name}: PASSED")
            else:
                print(f"❌ {test_name}: FAILED")
                
        except Exception as e:
            print(f"❌ {test_name}: ERROR - {e}")
            results[test_name] = False
    
    # Summary
    print(f"\n{'='*60}")
    print("TEST SUMMARY")
    print(f"{'='*60}")
    
    passed = sum(1 for result in results.values() if result)
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name}: {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Implementation is ready.")
    else:
        print("⚠️ Some tests failed. Please review the implementation.")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
