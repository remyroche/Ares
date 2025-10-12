#!/usr/bin/env python3
"""
Test script to verify that duplicate cleanup was successful.
This script tests that the base class methods work correctly after removing duplicates.
"""

import sys
import os
import pandas as pd
import numpy as np

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_vectorized_feature_generator():
    """Test that VectorizedFeatureGenerator methods work correctly."""
    print("Testing VectorizedFeatureGenerator...")
    
    try:
        from feature_generation.core.feature_generator import VectorizedFeatureGenerator, FeatureConfig, FeatureCategory
        
        # Create a test config
        config = FeatureConfig(
            name="test_feature",
            category=FeatureCategory.CUSTOM,
            description="Test feature",
            required_columns=["close"],
            default_lookback=20
        )
        
        # Create generator
        generator = VectorizedFeatureGenerator(config)
        
        # Create test data
        data = pd.DataFrame({
            'close': np.random.randn(100),
            'volume': np.random.randint(1000, 10000, 100)
        })
        
        # Test optimize_dataframe_processing
        optimized_data = generator.optimize_dataframe_processing(data)
        assert isinstance(optimized_data, pd.DataFrame), "optimize_dataframe_processing should return DataFrame"
        assert len(optimized_data) == len(data), "Optimized data should have same length"
        
        # Test vectorized_rolling_operations
        operations = ['mean', 'std']
        windows = [10, 20]
        result = generator.vectorized_rolling_operations(data, operations, windows)
        assert isinstance(result, pd.DataFrame), "vectorized_rolling_operations should return DataFrame"
        
        print("✅ VectorizedFeatureGenerator tests passed")
        return True
        
    except Exception as e:
        print(f"❌ VectorizedFeatureGenerator tests failed: {e}")
        return False

def test_volume_generator():
    """Test that volume generators work after duplicate removal."""
    print("Testing Volume generators...")
    
    try:
        from feature_generation.categories.volume import VolumeSMAGenerator
        
        # Create generator
        generator = VolumeSMAGenerator(period=20)
        
        # Create test data
        data = pd.DataFrame({
            'close': np.random.randn(100),
            'volume': np.random.randint(1000, 10000, 100)
        })
        
        # Test that the generator still works
        result = generator.generate(data)
        assert result.success, f"Volume generator failed: {result.error_message}"
        assert isinstance(result.data, pd.Series), "Result should be a Series"
        
        print("✅ Volume generator tests passed")
        return True
        
    except Exception as e:
        print(f"❌ Volume generator tests failed: {e}")
        return False

def test_acceleration_generator():
    """Test that acceleration generators work after duplicate removal."""
    print("Testing Acceleration generators...")
    
    try:
        from feature_generation.categories.acceleration import AccelerationFeatureGenerator
        
        # Create generator
        generator = AccelerationFeatureGenerator()
        
        # Create test data
        data = pd.DataFrame({
            'close': np.random.randn(100),
            'high': np.random.randn(100) + 0.1,
            'low': np.random.randn(100) - 0.1,
            'volume': np.random.randint(1000, 10000, 100)
        })
        
        # Test that the generator still works
        result = generator.generate(data)
        assert result.success, f"Acceleration generator failed: {result.error_message}"
        assert isinstance(result.data, pd.Series), "Result should be a Series"
        
        print("✅ Acceleration generator tests passed")
        return True
        
    except Exception as e:
        print(f"❌ Acceleration generator tests failed: {e}")
        return False

def test_entropy_generator():
    """Test that entropy generators work after duplicate removal."""
    print("Testing Entropy generators...")
    
    try:
        from feature_generation.categories.entropy import EntropyFeatureGenerator
        
        # Create generator
        generator = EntropyFeatureGenerator()
        
        # Create test data
        data = pd.DataFrame({
            'close': np.random.randn(100),
            'volume': np.random.randint(1000, 10000, 100)
        })
        
        # Test that the generator still works
        result = generator.generate(data)
        assert result.success, f"Entropy generator failed: {result.error_message}"
        assert isinstance(result.data, pd.Series), "Result should be a Series"
        
        print("✅ Entropy generator tests passed")
        return True
        
    except Exception as e:
        print(f"❌ Entropy generator tests failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🧪 Running duplicate cleanup verification tests...\n")
    
    tests = [
        test_vectorized_feature_generator,
        test_volume_generator,
        test_acceleration_generator,
        test_entropy_generator
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Duplicate cleanup was successful.")
        return True
    else:
        print("⚠️ Some tests failed. Please check the implementation.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)