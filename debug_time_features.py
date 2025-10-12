#!/usr/bin/env python3
"""
Debug time features import issues.
"""

import sys
import os

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_imports():
    """Test imports step by step."""
    print("🔍 Testing imports step by step...")
    
    try:
        print("1. Testing basic imports...")
        import pandas as pd
        import numpy as np
        print("✅ Basic imports successful")
    except Exception as e:
        print(f"❌ Basic imports failed: {e}")
        return False
    
    try:
        print("2. Testing feature_generator import...")
        from src.feature_generation.core.feature_generator import FeatureGenerator, FeatureConfig, FeatureCategory
        print("✅ FeatureGenerator import successful")
    except Exception as e:
        print(f"❌ FeatureGenerator import failed: {e}")
        return False
    
    try:
        print("3. Testing time features import...")
        from src.feature_generation.categories.time import HourGenerator
        print("✅ HourGenerator import successful")
    except Exception as e:
        print(f"❌ HourGenerator import failed: {e}")
        return False
    
    try:
        print("4. Testing time features factory...")
        from src.feature_generation.categories.time import create_default_time_generators
        print("✅ Time features factory import successful")
    except Exception as e:
        print(f"❌ Time features factory import failed: {e}")
        return False
    
    return True

def test_hour_generator():
    """Test HourGenerator specifically."""
    print("\n🔍 Testing HourGenerator...")
    
    try:
        from src.feature_generation.categories.time import HourGenerator
        import pandas as pd
        import numpy as np
        
        # Create test data
        dates = pd.date_range('2020-01-01', periods=100, freq='1min')
        test_data = pd.DataFrame({
            'close': 100 + np.cumsum(np.random.randn(100) * 0.01)
        }, index=dates)
        
        # Create generator
        generator = HourGenerator()
        print(f"✅ HourGenerator created: {generator.config.name}")
        
        # Test feature generation
        feature = generator.generate_feature(test_data)
        print(f"✅ Feature generated: {feature.shape if feature is not None else 'None'}")
        
        return True
        
    except Exception as e:
        print(f"❌ HourGenerator test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main debug function."""
    print("🐛 Debug Time Features")
    print("=" * 30)
    
    # Test imports
    import_success = test_imports()
    
    if import_success:
        # Test specific generator
        generator_success = test_hour_generator()
        
        if generator_success:
            print("\n🎉 All tests passed!")
        else:
            print("\n❌ Generator test failed!")
    else:
        print("\n❌ Import test failed!")
    
    print("\n🏁 Debug completed!")

if __name__ == "__main__":
    main()