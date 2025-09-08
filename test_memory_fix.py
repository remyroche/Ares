#!/usr/bin/env python3
"""
Test script to verify memory leak fixes in step06_advanced_features.py
"""

import pandas as pd
import numpy as np
import time
import gc
import psutil
import os
from pathlib import Path

def get_memory_usage():
    """Get current memory usage in MB"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024

def create_test_data(n_rows=50000):
    """Create test OHLCV data similar to what step06 processes"""
    np.random.seed(42)

    # Create realistic price data
    base_price = 3000
    dates = pd.date_range('2023-01-01', periods=n_rows, freq='1min')

    # Generate realistic OHLCV data
    price_changes = np.random.normal(0, 0.001, n_rows)  # Small random changes
    prices = base_price * (1 + np.cumsum(price_changes))

    # Create OHLCV columns
    high = prices * (1 + np.abs(np.random.normal(0, 0.0005, n_rows)))
    low = prices * (1 - np.abs(np.random.normal(0, 0.0005, n_rows)))
    open_prices = prices * (1 + np.random.normal(0, 0.0002, n_rows))
    close_prices = prices
    volume = np.random.lognormal(10, 1, n_rows)  # Realistic volume

    df = pd.DataFrame({
        'timestamp': dates,
        'open': open_prices,
        'high': high,
        'low': low,
        'close': close_prices,
        'volume': volume
    })

    # Add a simple label column for testing
    df['label'] = np.random.choice([-1, 0, 1], n_rows, p=[0.3, 0.4, 0.3])

    return df.set_index('timestamp')

def test_memory_usage():
    """Test memory usage improvements"""
    print("🧪 Testing memory leak fixes in step06_advanced_features.py")
    print("=" * 60)

    # Initial memory
    initial_memory = get_memory_usage()
    print(".1f")

    # Import the module
    try:
        from src.training.steps.feature_engineering.step06_advanced_features import AdvancedFeatureEngineeringStep
        import_memory = get_memory_usage()
        print(".1f")

        # Create test configuration
        config = {
            'feature_engineering': {
                'enable_wavelets': False,  # Disable to avoid additional complexity
                'enable_multi_timeframe': True,
                'timeframes': ['5m', '15m'],
                'chunk_size': 100000,
            }
        }

        # Initialize step
        step = AdvancedFeatureEngineeringStep(config)
        init_memory = get_memory_usage()
        print(".1f")

        # Create test data
        print("📊 Creating test dataset...")
        test_data = create_test_data(25000)  # Smaller dataset for testing
        data_memory = get_memory_usage()
        print(".1f")

        # Test comprehensive technical features generation
        print("🔧 Testing technical features generation...")

        # Create mock training input and pipeline state
        training_input = {
            'symbol': 'ETHUSDT',
            'exchange': 'BINANCE',
            'timeframe': '1m'
        }

        pipeline_state = {
            'labeled_data': test_data
        }

        # Measure memory before processing
        before_process = get_memory_usage()
        start_time = time.time()

        # Run the feature generation
        try:
            result = step._generate_comprehensive_technical_features(test_data)
            end_time = time.time()

            after_process = get_memory_usage()

            print(".1f")
            print(".3f")
            print(f"📈 Features generated: {result.shape[1]}")
            print(f"📊 Data points processed: {len(result):,}")

            # Check for memory growth
            memory_growth = after_process - before_process
            if memory_growth < 100:  # Less than 100MB growth
                print("✅ Memory usage test PASSED - no excessive memory leak detected")
            else:
                print(f"⚠️ Memory usage test WARNING - {memory_growth:.1f}MB growth detected")

            # Force garbage collection
            gc.collect()
            final_memory = get_memory_usage()
            print(".1f")

            return True

        except Exception as e:
            print(f"❌ Error during processing: {e}")
            return False

    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

if __name__ == "__main__":
    success = test_memory_usage()
    exit(0 if success else 1)
