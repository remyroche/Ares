#!/usr/bin/env python3
"""
Test standalone compatibility layer.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

def test_standalone_compatibility():
    """Test the standalone compatibility layer."""
    print("🔍 Testing standalone compatibility layer...")
    
    try:
        # Test direct import of standalone compatibility
        from src.feature_engineering.standalone_hmm_compatibility import FeatureGenerators
        print("✅ Successfully imported standalone FeatureGenerators")
        
        # Test instantiation
        fg = FeatureGenerators()
        print("✅ FeatureGenerators instantiated successfully")
        
        # Test method existence
        if hasattr(fg, 'generate_features_for_hmm'):
            print("✅ FeatureGenerators has generate_features_for_hmm method")
        else:
            print("❌ FeatureGenerators missing generate_features_for_hmm method")
            return False
        
        # Test method call with dummy data
        dummy_data = {'test': 'data'}
        result = fg.generate_features_for_hmm(dummy_data)
        print("✅ generate_features_for_hmm method called successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ Standalone compatibility test failed: {e}")
        import traceback
        print(f"❌ Error details: {traceback.format_exc()}")
        return False

def test_original_import():
    """Test importing from the original location."""
    print("\n🔍 Testing original import location...")
    
    try:
        # This should now use the standalone compatibility
        from src.feature_engineering.feature_generators import FeatureGenerators
        print("✅ Successfully imported FeatureGenerators from original location")
        
        # Test instantiation
        fg = FeatureGenerators()
        print("✅ FeatureGenerators instantiated successfully")
        
        # Test method existence
        if hasattr(fg, 'generate_features_for_hmm'):
            print("✅ FeatureGenerators has generate_features_for_hmm method")
        else:
            print("❌ FeatureGenerators missing generate_features_for_hmm method")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Original import test failed: {e}")
        import traceback
        print(f"❌ Error details: {traceback.format_exc()}")
        return False

def main():
    """Run all tests."""
    print("🚀 Testing Standalone HMM Feature Compatibility")
    print("=" * 50)
    
    tests = [
        test_standalone_compatibility,
        test_original_import
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"❌ Test {test.__name__} crashed: {e}")
            results.append(False)
    
    print("\n" + "=" * 50)
    print("📊 Test Results Summary:")
    print(f"✅ Passed: {sum(results)}/{len(results)}")
    print(f"❌ Failed: {len(results) - sum(results)}/{len(results)}")
    
    if all(results):
        print("\n🎉 All tests passed! Standalone compatibility layer is working.")
        print("🎯 HMM processes should now be able to find their features!")
    else:
        print("\n⚠️ Some tests failed. Compatibility layer needs fixing.")
    
    return all(results)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)