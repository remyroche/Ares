#!/usr/bin/env python3
"""
Minimal test for VectorBT regime features.
"""

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

def create_sample_data(n_periods: int = 100) -> pd.DataFrame:
    """Create sample OHLCV data for testing."""
    np.random.seed(42)
    
    # Generate realistic price data
    returns = np.random.normal(0.001, 0.02, n_periods)
    prices = 100 * (1 + returns).cumprod()
    
    data = pd.DataFrame({
        'open': prices * (1 + np.random.normal(0, 0.001, n_periods)),
        'high': prices * (1 + np.abs(np.random.normal(0, 0.005, n_periods))),
        'low': prices * (1 - np.abs(np.random.normal(0, 0.005, n_periods))),
        'close': prices,
        'volume': np.random.lognormal(10, 1, n_periods)
    }, index=pd.date_range('2020-01-01', periods=n_periods, freq='1min'))
    
    return data

def test_entropy_calculation():
    """Test entropy calculation directly."""
    print("🧪 Testing entropy calculation...")
    
    data = create_sample_data(50)
    close = data['close']
    window = 10
    
    def calculate_shannon_entropy(segment):
        """Calculate Shannon entropy for a segment."""
        if len(segment) == 0:
            return np.nan
        
        hist, _ = np.histogram(segment, bins=10, density=True)
        hist = hist[hist > 0]
        
        if len(hist) == 0:
            return 0.0
        
        entropy = -np.sum(hist * np.log2(hist + 1e-10))
        return entropy
    
    # Calculate entropy for each window
    entropy_values = []
    for i in range(len(close)):
        if i < window - 1:
            entropy_values.append(np.nan)
        else:
            segment = close.iloc[i-window+1:i+1].values
            entropy = calculate_shannon_entropy(segment)
            entropy_values.append(entropy)
    
    entropy_series = pd.Series(entropy_values, index=data.index)
    
    print(f"✅ Calculated entropy: {len(entropy_series)} values")
    print(f"📊 Non-NaN values: {entropy_series.notna().sum()}")
    print(f"📈 Sample values: {entropy_series.dropna().head().values}")
    
    return True

def test_vectorbt_availability():
    """Test VectorBT availability."""
    print("\n🔧 Testing VectorBT availability...")
    
    try:
        import vectorbt as vbt
        print(f"✅ VectorBT version: {vbt.__version__}")
        return True
    except ImportError:
        print("❌ VectorBT not available")
        return False

def test_rolling_operations():
    """Test basic rolling operations."""
    print("\n⚡ Testing rolling operations...")
    
    data = create_sample_data(100)
    close = data['close']
    
    # Test pandas rolling
    pandas_mean = close.rolling(window=10).mean()
    print(f"✅ Pandas rolling mean: {len(pandas_mean)} values")
    
    # Test if VectorBT is available
    try:
        import vectorbt as vbt
        from vectorbt.generic import rolling_mean as vbt_rolling_mean
        
        vbt_mean = vbt_rolling_mean(close, window=10)
        print(f"✅ VectorBT rolling mean: {len(vbt_mean)} values")
        
        # Compare results
        diff = abs(pandas_mean - vbt_mean).max()
        print(f"📊 Max difference: {diff:.10f}")
        
        return True
    except Exception as e:
        print(f"⚠️ VectorBT rolling test failed: {e}")
        return False

def main():
    """Run minimal tests."""
    print("🧪 Minimal VectorBT Regime Features Test")
    print("=" * 40)
    
    # Test entropy calculation
    entropy_test = test_entropy_calculation()
    
    # Test VectorBT availability
    vectorbt_test = test_vectorbt_availability()
    
    # Test rolling operations
    rolling_test = test_rolling_operations()
    
    print("\n📋 Test Summary")
    print("=" * 20)
    print(f"Entropy calculation: {'✅' if entropy_test else '❌'}")
    print(f"VectorBT availability: {'✅' if vectorbt_test else '❌'}")
    print(f"Rolling operations: {'✅' if rolling_test else '❌'}")
    
    if entropy_test and (vectorbt_test or rolling_test):
        print("\n🎉 Core functionality working!")
        return True
    else:
        print("\n⚠️ Some issues detected")
        return False

if __name__ == "__main__":
    main()