#!/usr/bin/env python3
"""
Simple test for VectorBT regime features.
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

def test_basic_functionality():
    """Test basic functionality without VectorBT dependencies."""
    print("🧪 Testing basic regime feature functionality...")
    
    # Create sample data
    data = create_sample_data(50)
    print(f"📊 Created sample data with {len(data)} periods")
    
    # Test basic entropy calculation
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
    
    # Test entropy calculation
    close = data['close']
    window = 10
    
    entropy_values = []
    for i in range(len(close)):
        if i < window - 1:
            entropy_values.append(np.nan)
        else:
            segment = close.iloc[i-window+1:i+1].values
            entropy = calculate_shannon_entropy(segment)
            entropy_values.append(entropy)
    
    entropy_series = pd.Series(entropy_values, index=data.index)
    print(f"✅ Calculated entropy: {len(entropy_series)} values, {entropy_series.isna().sum()} NaNs")
    print(f"📈 Sample entropy values: {entropy_series.dropna().head().values}")
    
    return True

def test_imports():
    """Test if we can import the enhanced regime features."""
    print("\n🔧 Testing imports...")
    
    try:
        # Test basic imports
        import sys
        sys.path.append('/workspace')
        
        from src.feature_generation.categories.advanced_regime_features import (
            RegimeEntropyGenerator,
            create_advanced_regime_generators
        )
        print("✅ Successfully imported regime feature generators")
        
        # Test generator creation
        generator = RegimeEntropyGenerator(10)
        print("✅ Successfully created RegimeEntropyGenerator")
        
        # Test with sample data
        data = create_sample_data(100)
        result = generator._generate_feature(data)
        print(f"✅ Generated feature: {len(result)} values, {result.isna().sum()} NaNs")
        
        return True
        
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False

def main():
    """Run simple tests."""
    print("🧪 Simple VectorBT Regime Features Test")
    print("=" * 40)
    
    # Test basic functionality
    basic_test = test_basic_functionality()
    
    # Test imports
    import_test = test_imports()
    
    print("\n📋 Test Summary")
    print("=" * 20)
    print(f"Basic functionality: {'✅' if basic_test else '❌'}")
    print(f"Import test: {'✅' if import_test else '❌'}")
    
    if basic_test and import_test:
        print("\n🎉 All tests passed!")
    else:
        print("\n⚠️ Some tests failed")
    
    return basic_test and import_test

if __name__ == "__main__":
    main()