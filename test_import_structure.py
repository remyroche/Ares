#!/usr/bin/env python3
"""
Simple test to verify import structure without external dependencies.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

def test_import_structure():
    """Test that the import structure works correctly."""
    print("🔍 Testing import structure...")
    
    try:
        # Test 1: Import from old location
        print("Testing import from src.feature_engineering.feature_generators...")
        from src.feature_engineering.feature_generators import FeatureGenerators
        print("✅ Successfully imported FeatureGenerators from old location")
        
        # Test 2: Check if it has the expected method
        fg = FeatureGenerators()
        if hasattr(fg, 'generate_features_for_hmm'):
            print("✅ FeatureGenerators has generate_features_for_hmm method")
        else:
            print("❌ FeatureGenerators missing generate_features_for_hmm method")
            return False
        
        # Test 3: Import from new location
        print("Testing import from src.feature_generation...")
        from src.feature_generation import FeatureGenerators as NewFeatureGenerators
        print("✅ Successfully imported FeatureGenerators from new location")
        
        # Test 4: Check if new version has the method
        new_fg = NewFeatureGenerators()
        if hasattr(new_fg, 'generate_features_for_hmm'):
            print("✅ New FeatureGenerators has generate_features_for_hmm method")
        else:
            print("❌ New FeatureGenerators missing generate_features_for_hmm method")
            return False
        
        # Test 5: Check if they're the same class (compatibility working)
        if type(fg) == type(new_fg):
            print("✅ Compatibility layer working - both imports return same class")
        else:
            print("⚠️ Different classes returned - compatibility may not be working")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def test_hmm_components_import():
    """Test that HMM components can be imported."""
    print("\n🔍 Testing HMM components import...")
    
    try:
        # Test HMM regime discovery
        from src.training.steps.market_analysis.components.hmm_regime_discovery import HMMRegimeDiscoveryComponent
        print("✅ HMMRegimeDiscoveryComponent imported successfully")
        
        # Test HMM clustering
        from src.training.steps.market_analysis.components.hmm_clustering import HMMClusteringComponent
        print("✅ HMMClusteringComponent imported successfully")
        
        # Test HMM models training
        from src.training.steps.market_analysis.hmm_models_training.hmm_models_training_enhanced import HMMModelsTrainingEnhanced
        print("✅ HMMModelsTrainingEnhanced imported successfully")
        
        return True
        
    except ImportError as e:
        print(f"❌ HMM components import failed: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def main():
    """Run all tests."""
    print("🚀 Testing HMM Feature Import Structure")
    print("=" * 50)
    
    tests = [
        test_import_structure,
        test_hmm_components_import
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
        print("\n🎉 All tests passed! Import structure is working correctly.")
    else:
        print("\n⚠️ Some tests failed. Import structure needs fixing.")
    
    return all(results)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)