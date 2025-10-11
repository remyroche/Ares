#!/usr/bin/env python3
"""
Test script to verify VectorBT integration for feature generation categories.

This script tests the VectorBT integration for:
- Advanced Statistical Features (13 features)
- Support/Resistance Features (13 features) 
- Legacy Features (19 features)
"""

import pandas as pd
import numpy as np
import sys
import os
from datetime import datetime, timedelta

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def create_sample_data(n_points=2000):
    """Create sample OHLCV data for testing."""
    np.random.seed(42)
    
    # Generate price data with some trend and volatility
    base_price = 100
    returns = np.random.normal(0.001, 0.02, n_points)
    prices = [base_price]
    
    for ret in returns[1:]:
        prices.append(prices[-1] * (1 + ret))
    
    # Create OHLCV data
    data = []
    for i, price in enumerate(prices):
        # Add some intraday volatility
        high = price * (1 + abs(np.random.normal(0, 0.01)))
        low = price * (1 - abs(np.random.normal(0, 0.01)))
        open_price = prices[i-1] if i > 0 else price
        volume = np.random.randint(1000, 10000)
        
        data.append({
            'open': open_price,
            'high': high,
            'low': low,
            'close': price,
            'volume': volume
        })
    
    # Create datetime index
    start_date = datetime.now() - timedelta(days=n_points)
    dates = [start_date + timedelta(days=i) for i in range(n_points)]
    
    df = pd.DataFrame(data, index=dates)
    return df

def test_advanced_statistical_features():
    """Test Advanced Statistical Features with VectorBT integration."""
    print("🧪 Testing Advanced Statistical Features...")
    
    try:
        from feature_generation.categories.advanced_statistical import (
            HurstExponentGenerator,
            JumpIndicatorsGenerator,
            CVaRGenerator,
            MaxDrawdownGenerator,
            RollingSkewnessKurtosisGenerator,
            TrendPersistenceGenerator,
            create_default_advanced_statistical_generators
        )
        
        # Create sample data
        data = create_sample_data(1500)  # Large dataset to trigger VectorBT
        
        # Test individual generators
        generators = [
            HurstExponentGenerator(window=20),
            JumpIndicatorsGenerator(window=20, k_multiplier=3.0),
            CVaRGenerator(window=20, confidence_level=0.05),
            MaxDrawdownGenerator(window=20),
            RollingSkewnessKurtosisGenerator(window=20, stat_type='skewness'),
            TrendPersistenceGenerator(window=20)
        ]
        
        results = {}
        for generator in generators:
            try:
                result = generator.generate(data)
                results[generator.config.name] = result
                print(f"  ✅ {generator.config.name}: {len(result)} values generated")
            except Exception as e:
                print(f"  ❌ {generator.config.name}: {str(e)}")
        
        # Test default generators
        default_generators = create_default_advanced_statistical_generators()
        print(f"  📊 Created {len(default_generators)} default generators")
        
        return len(results)
        
    except Exception as e:
        print(f"  ❌ Advanced Statistical Features test failed: {str(e)}")
        return 0

def test_support_resistance_features():
    """Test Support/Resistance Features with VectorBT integration."""
    print("🧪 Testing Support/Resistance Features...")
    
    try:
        from feature_generation.categories.support_resistance import (
            SupportLevelGenerator,
            ResistanceLevelGenerator,
            PivotPointGenerator,
            FibonacciLevelGenerator,
            create_default_support_resistance_generators
        )
        
        # Create sample data
        data = create_sample_data(1500)  # Large dataset to trigger VectorBT
        
        # Test individual generators
        generators = [
            SupportLevelGenerator(level=1, window=20),
            ResistanceLevelGenerator(level=1, window=20),
            PivotPointGenerator(window=20),
            FibonacciLevelGenerator(level=0.618, window=20)
        ]
        
        results = {}
        for generator in generators:
            try:
                result = generator.generate(data)
                results[generator.config.name] = result
                print(f"  ✅ {generator.config.name}: {len(result)} values generated")
            except Exception as e:
                print(f"  ❌ {generator.config.name}: {str(e)}")
        
        # Test default generators
        default_generators = create_default_support_resistance_generators()
        print(f"  📊 Created {len(default_generators)} default generators")
        
        return len(results)
        
    except Exception as e:
        print(f"  ❌ Support/Resistance Features test failed: {str(e)}")
        return 0

def test_legacy_features():
    """Test Legacy Features with VectorBT integration."""
    print("🧪 Testing Legacy Features...")
    
    try:
        from feature_generation.categories.legacy import (
            LegacyRSIGenerator,
            LegacyMACDGenerator,
            LegacyBollingerBandsGenerator,
            LegacySMAGenerator,
            LegacyEMAGenerator,
            LegacyATRGenerator,
            LegacyStochasticGenerator,
            LegacyWilliamsRGenerator,
            LegacyOBVGenerator,
            create_default_legacy_generators
        )
        
        # Create sample data
        data = create_sample_data(1500)  # Large dataset to trigger VectorBT
        
        # Test individual generators
        generators = [
            LegacyRSIGenerator(period=14),
            LegacyMACDGenerator(fast=12, slow=26, signal=9),
            LegacyBollingerBandsGenerator(period=20, std_dev=2.0),
            LegacySMAGenerator(period=20),
            LegacyEMAGenerator(period=21),
            LegacyATRGenerator(period=14),
            LegacyStochasticGenerator(k_period=14, d_period=3),
            LegacyWilliamsRGenerator(period=14),
            LegacyOBVGenerator()
        ]
        
        results = {}
        for generator in generators:
            try:
                result = generator.generate(data)
                results[generator.config.name] = result
                print(f"  ✅ {generator.config.name}: {len(result)} values generated")
            except Exception as e:
                print(f"  ❌ {generator.config.name}: {str(e)}")
        
        # Test default generators
        default_generators = create_default_legacy_generators()
        print(f"  📊 Created {len(default_generators)} default generators")
        
        return len(results)
        
    except Exception as e:
        print(f"  ❌ Legacy Features test failed: {str(e)}")
        return 0

def test_vectorbt_availability():
    """Test VectorBT availability and basic functionality."""
    print("🔍 Checking VectorBT availability...")
    
    try:
        import vectorbt as vbt
        print(f"  ✅ VectorBT version: {vbt.__version__}")
        
        # Test basic VectorBT functionality
        test_data = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        rolling_mean_result = vbt.generic.rolling_mean(test_data, window=3)
        print(f"  ✅ VectorBT rolling operations working: {len(rolling_mean_result)} values")
        
        return True
        
    except ImportError:
        print("  ❌ VectorBT not available - features will use pandas fallback")
        return False
    except Exception as e:
        print(f"  ⚠️ VectorBT available but has issues: {str(e)}")
        return False

def main():
    """Run all VectorBT integration tests."""
    print("🚀 Starting VectorBT Integration Tests")
    print("=" * 50)
    
    # Check VectorBT availability
    vectorbt_available = test_vectorbt_availability()
    print()
    
    # Test each feature category
    advanced_stats_count = test_advanced_statistical_features()
    print()
    
    support_resistance_count = test_support_resistance_features()
    print()
    
    legacy_count = test_legacy_features()
    print()
    
    # Summary
    print("📊 Test Summary")
    print("=" * 50)
    print(f"VectorBT Available: {'✅ Yes' if vectorbt_available else '❌ No'}")
    print(f"Advanced Statistical Features: {advanced_stats_count}/13 working")
    print(f"Support/Resistance Features: {support_resistance_count}/13 working")
    print(f"Legacy Features: {legacy_count}/19 working")
    
    total_working = advanced_stats_count + support_resistance_count + legacy_count
    total_expected = 13 + 13 + 19  # 45 total features
    
    print(f"\nOverall: {total_working}/{total_expected} features working")
    
    if total_working == total_expected:
        print("🎉 All VectorBT integrations successful!")
        return True
    else:
        print("⚠️ Some features may need attention")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)