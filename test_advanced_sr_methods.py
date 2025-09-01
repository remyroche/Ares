#!/usr/bin/env python3
"""
Test script for advanced S/R methods in sr_breakout_predictor.py
"""

import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Mock the imports to avoid dependency issues
class MockLogger:
    def info(self, msg): print(f"INFO: {msg}")
    def error(self, msg): print(f"ERROR: {msg}")
    def warning(self, msg): print(f"WARNING: {msg}")

class MockSystemLogger:
    def getChild(self, name): return MockLogger()

# Mock the decorators
def validate_data_quality(validation_level="WARNING"):
    def decorator(func):
        return func
    return decorator

def handle_errors(exceptions=(Exception,), default_return=None, context=""):
    def decorator(func):
        return func
    return decorator

def handle_specific_errors(error_handlers=None, default_return=None, context=""):
    def decorator(func):
        return func
    return decorator

# Mock the centralized decorators
class MockCentralizedDecorators:
    @staticmethod
    def validate_data_quality(validation_level="WARNING"):
        def decorator(func):
            return func
        return decorator

# Create mock modules
import sys
sys.modules['src.utils.logger'] = type('MockModule', (), {'system_logger': MockSystemLogger()})
sys.modules['src.utils.error_handler'] = type('MockModule', (), {
    'handle_errors': handle_errors,
    'handle_specific_errors': handle_specific_errors
})
sys.modules['src.utils.centralized_decorators'] = MockCentralizedDecorators()

# Now import the actual SRBreakoutPredictor
from src.tactician.sr_breakout_predictor import SRBreakoutPredictor

def create_sample_market_data():
    """Create sample market data for testing."""
    np.random.seed(42)

    # Create 100 data points
    dates = pd.date_range(start='2024-01-01', periods=100, freq='1H')

    # Generate realistic price data with trends and volatility
    base_price = 100.0
    prices = [base_price]

    for i in range(1, 100):
        # Add trend and random walk
        trend = 0.001 * np.sin(i * 0.1)  # Cyclical trend
        random_walk = np.random.normal(0, 0.005)  # Random component
        new_price = prices[-1] * (1 + trend + random_walk)
        prices.append(new_price)

    # Create OHLCV data
    data = []
    for i, price in enumerate(prices):
        # Create realistic OHLC from base price
        volatility = 0.01 * (1 + 0.5 * np.sin(i * 0.2))  # Variable volatility
        high = price * (1 + np.random.uniform(0, volatility))
        low = price * (1 - np.random.uniform(0, volatility))
        open_price = prices[i-1] if i > 0 else price
        close_price = price

        # Volume with some correlation to price movement
        base_volume = 1000000
        volume_factor = 1 + abs(close_price - open_price) / open_price * 10
        volume = int(base_volume * volume_factor * np.random.uniform(0.5, 1.5))

        data.append({
            'open': open_price,
            'high': high,
            'low': low,
            'close': close_price,
            'volume': volume
        })

    return pd.DataFrame(data, index=dates)

def create_multi_timeframe_data():
    """Create sample multi-timeframe data."""
    base_data = create_sample_market_data()

    # Create different timeframes by resampling
    multi_tf_data = {}

    # 1-minute data (original)
    multi_tf_data['1m'] = base_data.copy()

    # 5-minute data
    multi_tf_data['5m'] = base_data.resample('5T').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna()

    # 15-minute data
    multi_tf_data['15m'] = base_data.resample('15T').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna()

    # 1-hour data
    multi_tf_data['1h'] = base_data.resample('1H').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna()

    return multi_tf_data

async def test_advanced_sr_methods():
    """Test all advanced S/R methods."""
    print("🚀 Testing Advanced S/R Methods")
    print("=" * 60)

    # Create sample data
    market_data = create_sample_market_data()
    multi_tf_data = create_multi_timeframe_data()

    print(f"✅ Created sample market data: {len(market_data)} data points")
    print(f"✅ Created multi-timeframe data: {list(multi_tf_data.keys())}")

    # Initialize SRBreakoutPredictor
    config = {
        "sr_breakout_predictor": {
            "enable_sr_breakout_tactics": True,
            "sr_proximity_threshold": 0.02,
            "breakout_confidence_threshold": 0.6,
            "sr_detection_method": "fractal",
            "min_sr_strength": 0.3,
            "max_sr_levels": 10,
            "sr_lookback_periods": 100,
            "volume_weight": 0.7,
            "price_weight": 0.3,
            "atr_multiplier": 1.5,
            "breakout_confirmation_periods": 3,
            "false_breakout_filter": True
        }
    }

    sr_predictor = SRBreakoutPredictor(config)

    # Test initialization
    print("\n📋 Testing Initialization...")
    init_success = await sr_predictor.initialize()
    print(f"✅ Initialization: {'SUCCESS' if init_success else 'FAILED'}")

    if not init_success:
        print("❌ Cannot proceed with tests - initialization failed")
        return

    # Test 1: Fibonacci Levels
    print("\n🔢 Testing Fibonacci Levels...")
    try:
        fib_levels = await sr_predictor.calculate_fibonacci_levels(market_data)
        print(f"✅ Fibonacci Levels: {len(fib_levels)} levels calculated")
        for level_name, price in list(fib_levels.items())[:5]:  # Show first 5
            print(f"   {level_name}: {price:.2f}")
    except Exception as e:
        print(f"❌ Fibonacci Levels Error: {e}")

    # Test 2: Elliott Wave Analysis
    print("\n🌊 Testing Elliott Wave Analysis...")
    try:
        elliott_levels = await sr_predictor.detect_elliott_wave_levels(market_data)
        print(f"✅ Elliott Wave: {elliott_levels.get('pattern_type', 'unknown')} pattern detected")
        print(f"   Confidence: {elliott_levels.get('confidence', 0):.2f}")
        if 'wave1' in elliott_levels:
            print(f"   Wave 1: {elliott_levels['wave1']}")
    except Exception as e:
        print(f"❌ Elliott Wave Error: {e}")

    # Test 3: Order Flow Analysis
    print("\n📊 Testing Order Flow Analysis...")
    try:
        order_flow = await sr_predictor.analyze_order_flow_levels(market_data)
        print(f"✅ Order Flow Analysis: POC at {order_flow.get('poc', 0):.2f}")
        print(f"   Value Area: {order_flow.get('value_area', {}).get('low', 0):.2f} - {order_flow.get('value_area', {}).get('high', 0):.2f}")
        print(f"   HVN Levels: {len(order_flow.get('hvn_levels', []))}")
        print(f"   Imbalances: {len(order_flow.get('imbalances', []))}")
    except Exception as e:
        print(f"❌ Order Flow Error: {e}")

    # Test 4: Multi-Timeframe Confluence
    print("\n⏰ Testing Multi-Timeframe Confluence...")
    try:
        mtf_confluence = await sr_predictor.detect_multi_timeframe_confluence(multi_tf_data)
        print(f"✅ Multi-Timeframe Confluence: {len(mtf_confluence)} strong confluence levels")
        for level_key, level_data in list(mtf_confluence.items())[:3]:  # Show first 3
            print(f"   {level_key}: {level_data['type']} at {level_data['price']:.2f} ({len(level_data['timeframes'])} timeframes)")
    except Exception as e:
        print(f"❌ Multi-Timeframe Confluence Error: {e}")

    # Test 5: Comprehensive S/R Analysis
    print("\n🎯 Testing Comprehensive S/R Analysis...")
    try:
        comprehensive = await sr_predictor.get_comprehensive_sr_analysis(market_data, multi_tf_data)
        print(f"✅ Comprehensive Analysis: {len(comprehensive.get('analysis_methods', []))} methods used")
        print(f"   Methods: {comprehensive.get('analysis_methods', [])}")
        print(f"   Current Price: {comprehensive.get('current_price', 0):.2f}")
        print(f"   Nearest Support: {comprehensive.get('nearest_support', 0):.2f}")
        print(f"   Nearest Resistance: {comprehensive.get('nearest_resistance', 0):.2f}")
    except Exception as e:
        print(f"❌ Comprehensive Analysis Error: {e}")

    # Test 6: Basic S/R Context (should include advanced methods)
    print("\n🔍 Testing Enhanced S/R Context...")
    try:
        current_price = market_data['close'].iloc[-1]
        sr_context = await sr_predictor.get_sr_context(market_data, current_price)
        print(f"✅ Enhanced S/R Context: {len(sr_context)} context elements")

        # Check for advanced methods in context
        advanced_methods = ['fibonacci_levels', 'elliott_wave_levels', 'order_flow_analysis']
        for method in advanced_methods:
            if method in sr_context:
                print(f"   ✅ {method}: Included in context")
            else:
                print(f"   ❌ {method}: Missing from context")
    except Exception as e:
        print(f"❌ Enhanced S/R Context Error: {e}")

    print("\n" + "=" * 60)
    print("🎉 Advanced S/R Methods Testing Complete!")
    print("=" * 60)

if __name__ == "__main__":
    asyncio.run(test_advanced_sr_methods())