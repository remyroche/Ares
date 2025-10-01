#!/usr/bin/env python3
"""
Test script to verify fast fail behavior in regime feature generation.
"""

import pandas as pd
import numpy as np
import sys
import os

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from feature_generation.categories.regime_feature_integration import RegimeFeatureIntegration, RegimeFeatureConfig

def test_fast_fail():
    """Test that regime feature generation fails fast when there are issues."""

    # Create a simple test DataFrame
    test_data = pd.DataFrame({
        'close': np.random.randn(100),
        'high': np.random.randn(100),
        'low': np.random.randn(100),
        'open': np.random.randn(100),
        'volume': np.random.randn(100)
    })

    # Create regime feature config
    config = RegimeFeatureConfig()

    # Create generator
    generator = RegimeFeatureIntegration(config)

    try:
        # Test the _generate_feature method which should fast fail on error
        result = generator._generate_feature(test_data)
        print("❌ Expected fast fail but got result")
        return False
    except ValueError as e:
        print(f"✅ Fast fail working correctly: {e}")
        return True
    except Exception as e:
        print(f"❌ Unexpected error type: {type(e).__name__}: {e}")
        return False

def test_regime_features_generation():
    """Test the generate_features method."""

    # Create a simple test DataFrame
    test_data = pd.DataFrame({
        'close': np.random.randn(100),
        'high': np.random.randn(100),
        'low': np.random.randn(100),
        'open': np.random.randn(100),
        'volume': np.random.randn(100)
    })

    # Create regime feature config
    config = RegimeFeatureConfig()

    # Create generator
    generator = RegimeFeatureIntegration(config)

    try:
        # Test the generate_features method which should fast fail on error
        result = generator.generate_features(test_data)
        print("❌ Expected fast fail but got result")
        return False
    except ValueError as e:
        print(f"✅ Fast fail working correctly: {e}")
        return True
    except Exception as e:
        print(f"❌ Unexpected error type: {type(e).__name__}: {e}")
        return False

if __name__ == "__main__":
    print("Testing fast fail behavior...")

    test1_passed = test_fast_fail()
    test2_passed = test_regime_features_generation()

    if test1_passed and test2_passed:
        print("✅ All tests passed!")
        sys.exit(0)
    else:
        print("❌ Some tests failed!")
        sys.exit(1)
