#!/usr/bin/env python3
"""
Test script for the optimized feature lookback optimization.
"""

import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Import the optimized feature lookback optimization
from src.training.steps.pre_training.feature_lookback_optimization.feature_lookback_optimization import (
    FeatureLookbackOptimizationComponent,
    OptimizedFeatureLookbackConfig
)

async def test_optimized_feature_lookback():
    """Test the optimized feature lookback optimization."""

    print("🧪 Testing Optimized Feature Lookback Optimization...")

    # Create sample market data
    print("📊 Creating sample market data...")
    dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='5min')
    n_bars = len(dates)

    # Generate realistic OHLCV data
    np.random.seed(42)
    close_prices = 100 * np.exp(np.cumsum(np.random.randn(n_bars) * 0.001))

    data = pd.DataFrame({
        'timestamp': dates,
        'open': close_prices * (1 + np.random.randn(n_bars) * 0.002),
        'high': close_prices * (1 + np.abs(np.random.randn(n_bars) * 0.003)),
        'low': close_prices * (1 - np.abs(np.random.randn(n_bars) * 0.003)),
        'close': close_prices,
        'volume': np.random.randint(1000, 10000, n_bars)
    })

    print(f"✅ Created data with {len(data)} bars")

    # Create optimized configuration
    config = OptimizedFeatureLookbackConfig(
        default_timeframe="5m",
        min_lookback=5,
        max_lookback=20,  # Smaller range for testing
        lookback_step=5,
        enable_volatility_normalization=True,
        enable_multi_target_scheme=True
    )

    # Initialize the optimizer
    print("🔧 Initializing FeatureLookbackOptimizationComponent...")
    optimizer = FeatureLookbackOptimizationComponent()

    # Test the unified optimization method directly
    print("🚀 Testing unified optimization method...")

    # Get feature columns (simulate having some features)
    feature_columns = ['close', 'volume']  # Simple features for testing

    try:
        # Test unified optimization
        result = optimizer.optimize_lookback_periods_unified(
            data=data,
            feature_columns=feature_columns,
            target_column='close',
            optimization_config=None,
            enable_directional=True,
            enable_multi_target=False
        )

        print("✅ Unified optimization completed successfully!")
        print(f"   → Result keys: {list(result.keys())}")

        # Check if we have results for our features
        for feature in feature_columns:
            if feature in result:
                print(f"   → {feature}: Found in results")
            else:
                print(f"   → {feature}: Not found in results")

        # Test getting eligible features
        print("🔍 Testing eligible features identification...")
        eligible_features = optimizer._get_eligible_features()
        print(f"✅ Found {len(eligible_features)} eligible features")

        # Test forward return calculation
        print("📈 Testing FPT forward return calculation...")
        forward_returns = optimizer._calculate_forward_returns_fpt(data, lookback=10)
        print(f"✅ Generated {len(forward_returns.dropna())} forward return samples")

        print("🎉 All tests passed successfully!")
        return True

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(test_optimized_feature_lookback())
    exit(0 if success else 1)