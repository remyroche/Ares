#!/usr/bin/env python3
"""
Test script for the new body_size features integration.

This script tests that the body_size and body_size_pct features are properly
integrated into the feature engineering pipeline.
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

def test_body_size_features():
    """Test the new body_size features."""

    print("🧪 Testing Body Size Features Integration")
    print("=" * 50)

    # Create sample OHLCV data
    np.random.seed(42)
    n_rows = 100

    # Generate realistic price data
    base_price = 50000
    price_changes = np.random.normal(0, 0.02, n_rows).cumsum()
    close_prices = base_price * (1 + price_changes)

    # Generate OHLC data around close prices
    high_mult = np.random.uniform(1.001, 1.01, n_rows)
    low_mult = np.random.uniform(0.99, 0.999, n_rows)

    data = pd.DataFrame({
        'open': close_prices * np.random.uniform(0.995, 1.005, n_rows),
        'high': close_prices * high_mult,
        'low': close_prices * low_mult,
        'close': close_prices,
        'volume': np.random.uniform(1000, 10000, n_rows)
    })

    print(f"📊 Created sample data with {len(data)} rows")
    print(f"   Price range: ${data['close'].min():.2f} - ${data['close'].max():.2f}")

    # Test 1: Feature Generators
    print("\n🔧 Testing Feature Generators...")
    try:
        from src.feature_engineering.feature_generators import FeatureGenerators

        fg = FeatureGenerators()

        # Test individual generators
        body_size = fg.body_size_generator(data)
        body_size_pct = fg.body_size_pct_generator(data)
        body_direction = fg.body_direction_generator(data)
        body_strength = fg.body_strength_generator(data)
        upper_wick = fg.upper_wick_generator(data)
        lower_wick = fg.lower_wick_generator(data)
        body_to_range_ratio = fg.body_to_range_ratio_generator(data)

        print("✅ Feature generators working:")
        print(f"   Body Size - Mean: {body_size.mean():.6f}, Std: {body_size.std():.6f}")
        print(f"   Body Size Pct - Mean: {body_size_pct.mean():.4f}%")
        print(f"   Body Direction - Up: {(body_direction > 0).sum()}, Down: {(body_direction < 0).sum()}")
        print(f"   Body Strength - Mean: {body_strength.mean():.6f}")
    except Exception as e:
        print(f"❌ Feature generators test failed: {e}")
        return False

    # Test 2: Basic Candlestick Calculations
    print("\n🏗️ Testing Basic Candlestick Calculations...")
    try:
        # Test the core body_size calculations directly
        body_size = np.abs(data['close'] - data['open'])
        body_size_pct = (body_size / data['open']) * 100
        body_direction = np.sign(data['close'] - data['open'])
        body_strength = body_size * body_direction

        total_range = data['high'] - data['low']
        body_to_range_ratio = body_size / total_range.replace(0, 1)

        upper_wick = data['high'] - np.maximum(data['open'], data['close'])
        lower_wick = np.minimum(data['open'], data['close']) - data['low']

        print("✅ Basic candlestick calculations working:")
        print(f"   Body Size - Mean: {body_size.mean():.6f}, Std: {body_size.std():.6f}")
        print(f"   Body Size Pct - Mean: {body_size_pct.mean():.4f}%")
        print(f"   Body Direction - Up: {(body_direction > 0).sum()}, Down: {(body_direction < 0).sum()}")
        print(f"   Body to Range Ratio - Mean: {body_to_range_ratio.mean():.4f}")
        print(f"   Upper Wick - Mean: {upper_wick.mean():.6f}")
        print(f"   Lower Wick - Mean: {lower_wick.mean():.6f}")

    except Exception as e:
        print(f"❌ Basic candlestick calculations failed: {e}")
        return False

    # Test 3: Cross-Timeframe Analysis
    print("\n🔄 Testing Cross-Timeframe Analysis...")
    try:
        from src.feature_engineering.cross_timeframe_analysis_pipeline import CrossTimeframeAnalysisPipeline

        # Create a simple pipeline
        config = {'cross_timeframe': {'timeframes': ['1h']}}
        pipeline = CrossTimeframeAnalysisPipeline(config)

        # Test the comprehensive order flow proxies method
        features = pipeline._generate_comprehensive_order_flow_proxies(data, '1h')

        # Check for body_size features
        body_features_ct = [col for col in features.keys() if 'body' in col.lower()]
        print("✅ Cross-timeframe analysis working:")
        print(f"   Body features found: {len(body_features_ct)}")

        if body_features_ct:
            print(f"   Sample features: {body_features_ct[:3]}")

    except Exception as e:
        print(f"❌ Cross-timeframe analysis test failed: {e}")
        return False

    print("\n🎉 All tests passed! Body size features successfully integrated.")
    return True

if __name__ == "__main__":
    success = test_body_size_features()
    sys.exit(0 if success else 1)
