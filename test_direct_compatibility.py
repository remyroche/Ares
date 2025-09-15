#!/usr/bin/env python3
"""
Test direct compatibility module.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

def test_direct_compatibility():
    """Test the direct compatibility module."""
    print("🔍 Testing direct compatibility module...")
    
    try:
        # Test direct import of standalone compatibility
        from src.hmm_feature_compatibility import FeatureGenerators
        print("✅ Successfully imported FeatureGenerators from direct compatibility")
        
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
        print(f"❌ Direct compatibility test failed: {e}")
        import traceback
        print(f"❌ Error details: {traceback.format_exc()}")
        return False

def test_hmm_training_import():
    """Test that HMM training can import what it needs."""
    print("\n🔍 Testing HMM training import...")
    
    try:
        # Test that the HMM training module can be imported
        from src.training.steps.market_analysis.hmm_models_training.hmm_models_training_enhanced import HMMModelsTrainingEnhanced
        print("✅ HMMModelsTrainingEnhanced imported successfully")
        
        # Test instantiation (this will test the feature generator initialization)
        training = HMMModelsTrainingEnhanced()
        print("✅ HMMModelsTrainingEnhanced instantiated successfully")
        
        # Check if feature generator was initialized
        if hasattr(training, 'feature_generator') and training.feature_generator is not None:
            print("✅ Feature generator initialized successfully")
        else:
            print("⚠️ Feature generator not initialized")
        
        return True
        
    except Exception as e:
        print(f"❌ HMM training import test failed: {e}")
        import traceback
        print(f"❌ Error details: {traceback.format_exc()}")
        return False

def main():
    """Run all tests."""
    print("🚀 Testing Direct HMM Feature Compatibility")
    print("=" * 50)
    
    tests = [
        test_direct_compatibility,
        test_hmm_training_import
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
        print("\n🎉 All tests passed! Direct compatibility layer is working.")
        print("🎯 HMM processes should now be able to find their features!")
    else:
        print("\n⚠️ Some tests failed. Compatibility layer needs fixing.")
    
    return all(results)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)