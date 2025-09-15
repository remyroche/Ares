#!/usr/bin/env python3
"""
Simple test to verify the compatibility layer works without external dependencies.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

def test_simple_compatibility():
    """Test the simple compatibility layer."""
    print("🔍 Testing simple compatibility layer...")
    
    try:
        # Test direct import of simple compatibility
        from src.feature_generation.compatibility.simple_hmm_compatibility import FeatureGenerators
        print("✅ Successfully imported simple FeatureGenerators")
        
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
        print(f"❌ Simple compatibility test failed: {e}")
        import traceback
        print(f"❌ Error details: {traceback.format_exc()}")
        return False

def test_compatibility_redirect():
    """Test the compatibility redirect."""
    print("\n🔍 Testing compatibility redirect...")
    
    try:
        # Test import from compatibility module
        from src.feature_engineering.feature_generators_compatibility import FeatureGenerators
        print("✅ Successfully imported FeatureGenerators from compatibility module")
        
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
        print(f"❌ Compatibility redirect test failed: {e}")
        import traceback
        print(f"❌ Error details: {traceback.format_exc()}")
        return False

def main():
    """Run all tests."""
    print("🚀 Testing Simple HMM Feature Compatibility")
    print("=" * 50)
    
    tests = [
        test_simple_compatibility,
        test_compatibility_redirect
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
        print("\n🎉 All tests passed! Simple compatibility layer is working.")
    else:
        print("\n⚠️ Some tests failed. Compatibility layer needs fixing.")
    
    return all(results)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)