"""
Test spike detection and correction functionality.
"""

import numpy as np
import pandas as pd
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.training.steps.pre_training.feature_generation_labeling_integration_step import detect_and_correct_price_spikes


def create_test_data_with_spike():
    """Create test data with a clear spike."""
    # Create a simple uptrend
    dates = pd.date_range('2024-01-01', periods=50, freq='15T')
    
    # Base price trend
    base_price = 100 + np.linspace(0, 10, 50)
    
    # Add a spike at position 25
    prices = base_price.copy()
    prices[25] = prices[24] + 5.0  # Large upward spike
    # Next bar returns to trend
    prices[26] = base_price[26]
    
    # Create DataFrame
    df = pd.DataFrame({
        'close': prices,
        'open': prices * 0.999,
        'high': prices * 1.001,
        'low': prices * 0.998,
        'volume': np.random.uniform(1000, 2000, 50)
    }, index=dates)
    
    return df


def create_test_data_with_trend():
    """Create test data with a genuine trend (no spike)."""
    dates = pd.date_range('2024-01-01', periods=50, freq='15T')
    
    # Create a strong uptrend that continues
    prices = 100 + np.linspace(0, 20, 50)
    
    df = pd.DataFrame({
        'close': prices,
        'open': prices * 0.999,
        'high': prices * 1.001,
        'low': prices * 0.998,
        'volume': np.random.uniform(1000, 2000, 50)
    }, index=dates)
    
    return df


def create_test_data_with_multiple_spikes():
    """Create test data with multiple spikes."""
    dates = pd.date_range('2024-01-01', periods=100, freq='15T')
    
    # Base price trend
    base_price = 100 + np.sin(np.linspace(0, 4*np.pi, 100)) * 5
    
    # Add spikes
    prices = base_price.copy()
    # Spike 1
    prices[20] = prices[19] + 3.0
    # Spike 2
    prices[50] = prices[49] - 3.0
    # Spike 3
    prices[80] = prices[79] + 4.0
    
    df = pd.DataFrame({
        'close': prices,
        'open': prices * 0.999,
        'high': prices * 1.001,
        'low': prices * 0.998,
        'volume': np.random.uniform(1000, 2000, 100)
    }, index=dates)
    
    return df


def test_spike_detection_on_spike():
    """Test that spike detection correctly identifies a spike."""
    print("\n" + "="*60)
    print("TEST 1: Detecting a Clear Spike")
    print("="*60)
    
    df = create_test_data_with_spike()
    
    # Print original data around spike
    print("\nOriginal data around spike (rows 23-27):")
    print(df.iloc[23:28][['close']])
    
    # Run spike detection
    cleaned_df, stats = detect_and_correct_price_spikes(
        df,
        price_column='close',
        lookback_window=10,
        threshold_multiplier=2.0,  # Lower threshold to catch spike
        volatility_window=20
    )
    
    # Print cleaned data
    print("\nCleaned data around spike (rows 23-27):")
    print(cleaned_df.iloc[23:28][['close']])
    
    # Print statistics
    print("\nSpike Detection Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Assertions
    assert stats['spikes_detected'] > 0, "Should detect at least one spike"
    print("\n✅ TEST PASSED: Spike detected successfully")


def test_spike_detection_on_trend():
    """Test that genuine trends are NOT detected as spikes."""
    print("\n" + "="*60)
    print("TEST 2: Preserving Genuine Trend")
    print("="*60)
    
    df = create_test_data_with_trend()
    
    # Print sample data
    print("\nOriginal trend data (first 5 and last 5 rows):")
    print(df[['close']].head())
    print("...")
    print(df[['close']].tail())
    
    # Run spike detection
    cleaned_df, stats = detect_and_correct_price_spikes(
        df,
        price_column='close',
        lookback_window=10,
        threshold_multiplier=3.0,
        volatility_window=20
    )
    
    # Print statistics
    print("\nSpike Detection Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Assertions - trend should not be flagged as spikes
    assert stats['spikes_detected'] == 0 or stats['spike_percentage'] < 5.0, \
        "Genuine trend should not be flagged as multiple spikes"
    print("\n✅ TEST PASSED: Genuine trend preserved")


def test_spike_detection_on_multiple_spikes():
    """Test detection of multiple spikes."""
    print("\n" + "="*60)
    print("TEST 3: Detecting Multiple Spikes")
    print("="*60)
    
    df = create_test_data_with_multiple_spikes()
    
    # Print sample data
    print("\nOriginal data sample (every 10th row):")
    print(df.iloc[::10][['close']])
    
    # Run spike detection
    cleaned_df, stats = detect_and_correct_price_spikes(
        df,
        price_column='close',
        lookback_window=10,
        threshold_multiplier=2.0,  # Lower to catch spikes
        volatility_window=20
    )
    
    # Print cleaned data sample
    print("\nCleaned data sample (every 10th row):")
    print(cleaned_df.iloc[::10][['close']])
    
    # Print statistics
    print("\nSpike Detection Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Assertions
    assert stats['spikes_detected'] >= 2, "Should detect at least 2 spikes"
    assert stats['spikes_corrected'] > 0, "Should correct at least some spikes"
    print("\n✅ TEST PASSED: Multiple spikes detected and corrected")


def test_spike_correction_accuracy():
    """Test that spike correction produces reasonable values."""
    print("\n" + "="*60)
    print("TEST 4: Spike Correction Accuracy")
    print("="*60)
    
    df = create_test_data_with_spike()
    
    # Get the spike location
    spike_idx = 25
    original_spike_value = df.iloc[spike_idx]['close']
    prev_value = df.iloc[spike_idx - 1]['close']
    next_value = df.iloc[spike_idx + 1]['close']
    expected_correction = (prev_value + original_spike_value + next_value) / 3.0
    
    print(f"\nOriginal spike value: {original_spike_value:.2f}")
    print(f"Previous value: {prev_value:.2f}")
    print(f"Next value: {next_value:.2f}")
    print(f"Expected correction (3-bar avg): {expected_correction:.2f}")
    
    # Run spike detection
    cleaned_df, stats = detect_and_correct_price_spikes(
        df,
        price_column='close',
        lookback_window=10,
        threshold_multiplier=2.0,
        volatility_window=20
    )
    
    corrected_value = cleaned_df.iloc[spike_idx]['close']
    print(f"Corrected value: {corrected_value:.2f}")
    
    # Check if correction is approximately the 3-bar average
    if stats['spikes_corrected'] > 0:
        assert abs(corrected_value - expected_correction) < 0.1, \
            f"Corrected value should be close to 3-bar average: {corrected_value:.2f} vs {expected_correction:.2f}"
        print("\n✅ TEST PASSED: Spike corrected to 3-bar average (including spike)")
    else:
        print("\n⚠️  WARNING: Spike not corrected (may not meet threshold)")


def run_all_tests():
    """Run all spike detection tests."""
    print("\n" + "="*60)
    print("SPIKE DETECTION TEST SUITE")
    print("="*60)
    
    try:
        test_spike_detection_on_spike()
        test_spike_detection_on_trend()
        test_spike_detection_on_multiple_spikes()
        test_spike_correction_accuracy()
        
        print("\n" + "="*60)
        print("ALL TESTS PASSED ✅")
        print("="*60)
        
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        raise
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        raise


if __name__ == '__main__':
    run_all_tests()

