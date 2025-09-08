#!/usr/bin/env python3
"""
Test script for entropy features in the Ares feature engineering system.
"""

import pandas as pd
import numpy as np
import sys
import os

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from training.steps.data_collection.feature_engineering.feature_components import EntropyFeatureEngine

def create_sample_data():
    """Create sample price data for testing."""
    np.random.seed(42)
    n_points = 1000

    # Create trending price data with some noise
    t = np.linspace(0, 4*np.pi, n_points)
    trend = 100 + 10 * np.sin(t)
    noise = np.random.normal(0, 2, n_points)
    price = trend + noise

    # Create OHLCV data
    data = pd.DataFrame({
        'timestamp': pd.date_range('2023-01-01', periods=n_points, freq='1min'),
        'open': price,
        'high': price + np.abs(np.random.normal(0, 1, n_points)),
        'low': price - np.abs(np.random.normal(0, 1, n_points)),
        'close': price + np.random.normal(0, 0.5, n_points),
        'volume': np.random.lognormal(10, 1, n_points)
    })

    data.set_index('timestamp', inplace=True)
    return data

def test_entropy_features():
    """Test the entropy feature generation."""
    print("🧠 Testing Entropy Features in Ares")
    print("=" * 50)

    # Create sample data
    data = create_sample_data()
    print(f"📊 Created sample data with {len(data)} rows")
    print(f"📈 Price range: {data['close'].min():.2f} - {data['close'].max():.2f}")

    # Initialize entropy engine
    config = {}
    entropy_engine = EntropyFeatureEngine(config)
    print("✅ Initialized EntropyFeatureEngine")

    # Generate entropy features
    print("\n🔬 Generating entropy features...")
    data_with_entropy = entropy_engine.create_entropy_features(data)

    # Check what features were generated
    entropy_cols = [col for col in data_with_entropy.columns if 'entropy_' in col]
    print(f"✅ Generated {len(entropy_cols)} entropy features:")
    print("\n📋 Entropy Features Generated:")
    for col in sorted(entropy_cols):
        values = data_with_entropy[col].dropna()
        if len(values) > 0:
            print("20")

    # Test specific entropy calculations
    print("\n🔍 Testing individual entropy calculations...")

    returns = data['close'].pct_change()
    shannon_entropy = entropy_engine._calculate_shannon_entropy(returns, 50)
    print(".4f")

    sample_entropy = entropy_engine._calculate_sample_entropy(data['close'], 50)
    print(".4f")

    permutation_entropy = entropy_engine._calculate_permutation_entropy(data['close'], 50)
    print(".4f")

    spectral_entropy = entropy_engine._calculate_spectral_entropy(data['close'], 50)
    print(".4f")

    print("\n🎉 Entropy feature testing completed successfully!")
    print(f"📊 Final dataset shape: {data_with_entropy.shape}")

    return data_with_entropy

if __name__ == "__main__":
    test_entropy_features()
