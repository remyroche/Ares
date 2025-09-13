#!/usr/bin/env python3
"""
Test script to verify data quality fixes for klines data processing.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from src.utils.data.quality.data_quality import DataQualityFramework

def create_test_klines_data():
    """Create test klines data with correlated features."""
    # Create timestamp index (1-minute intervals)
    start_time = datetime(2024, 1, 1, 0, 0, 0)
    timestamps = [start_time + timedelta(minutes=i) for i in range(100)]

    # Create OHLCV data
    np.random.seed(42)
    close_prices = 50000 + np.cumsum(np.random.normal(0, 50, 100))

    data = {
        'timestamp': timestamps,
        'open': close_prices + np.random.normal(0, 10, 100),
        'high': close_prices + np.abs(np.random.normal(0, 20, 100)),
        'low': close_prices - np.abs(np.random.normal(0, 20, 100)),
        'close': close_prices,
        'volume': np.random.exponential(100, 100),
    }

    df = pd.DataFrame(data)
    df = df.set_index('timestamp')

    # Add correlated features that should be filtered out
    df['close_return'] = df['close'].pct_change()
    df['close_log_return'] = np.log(df['close'] / df['close'].shift(1))

    # Bollinger Bands (perfectly correlated)
    window = 20
    df['bb_middle'] = df['close'].rolling(window=window).mean()
    std = df['close'].rolling(window=window).std()
    df['bb_upper'] = df['bb_middle'] + 2 * std
    df['bb_lower'] = df['bb_middle'] - 2 * std


    # Add some real features that might have correlations
    df['rsi_14'] = np.random.uniform(30, 70, 100)
    df['macd'] = np.random.normal(0, 10, 100)
    df['volatility_20'] = df['close_return'].rolling(20).std()

    return df

def test_data_quality_fixes():
    """Test the data quality fixes."""
    print("🧪 Testing data quality fixes...")

    # Create test data
    df = create_test_klines_data()
    print(f"📊 Created test dataset with {len(df)} rows and {len(df.columns)} columns")
    print(f"📋 Columns: {list(df.columns)}")

    # Initialize data quality framework
    framework = DataQualityFramework()

    # Validate data quality
    print("\n🔍 Running data quality validation...")
    result = framework.validate_dataframe_quality(df)

    print(f"✅ Validation completed: {'PASSED' if result.passed else 'FAILED'}")
    print(f"📊 Quality Score: {result.quality_score:.2f}")
    print(f"⚠️ Warnings: {len(result.warnings)}")
    print(f"❌ Issues: {len(result.issues)}")

    # Check for correlation filtering
    if 'excluded_correlated_features' in result.metrics:
        excluded = result.metrics['excluded_correlated_features']
        print(f"🚫 Excluded {len(excluded)} correlated features: {excluded}")

    if 'high_correlations' in result.metrics:
        high_corr = result.metrics['high_correlations']
        print(f"🔗 Found {len(high_corr)} high correlations after filtering")

    # Check timestamp validation
    if 'timestamp_issues' in result.metrics:
        ts_issues = result.metrics['timestamp_issues']
        print(f"⏰ Found {len(ts_issues)} timestamp issues")

    print("\n📋 Detailed Results:")
    if result.warnings:
        for warning in result.warnings[:5]:  # Show first 5 warnings
            print(f"  ⚠️ {warning}")

    if result.issues:
        for issue in result.issues[:3]:  # Show first 3 issues
            print(f"  ❌ {issue}")

if __name__ == "__main__":
    test_data_quality_fixes()
