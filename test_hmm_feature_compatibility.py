#!/usr/bin/env python3
"""
Test script to verify that HMM processes can find the features they expect.
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

def test_feature_generators_import():
    """Test that FeatureGenerators can be imported from the expected location."""
    print("🔍 Testing FeatureGenerators import...")
    
    try:
        from src.feature_engineering.feature_generators import FeatureGenerators
        print("✅ Successfully imported FeatureGenerators from src.feature_engineering.feature_generators")
        return True
    except ImportError as e:
        print(f"❌ Failed to import FeatureGenerators: {e}")
        return False

def test_generate_features_for_hmm():
    """Test that the generate_features_for_hmm method works."""
    print("\n🔍 Testing generate_features_for_hmm method...")
    
    try:
        from src.feature_engineering.feature_generators import FeatureGenerators
        
        # Create test data
        np.random.seed(42)
        dates = pd.date_range('2023-01-01', periods=1000, freq='1H')
        test_data = pd.DataFrame({
            'open': 100 + np.random.randn(1000).cumsum() * 0.1,
            'high': 100 + np.random.randn(1000).cumsum() * 0.1 + 0.5,
            'low': 100 + np.random.randn(1000).cumsum() * 0.1 - 0.5,
            'close': 100 + np.random.randn(1000).cumsum() * 0.1,
            'volume': np.random.randint(1000, 10000, 1000)
        }, index=dates)
        
        # Initialize feature generator
        fg = FeatureGenerators()
        print("✅ FeatureGenerators initialized successfully")
        
        # Test generate_features_for_hmm
        features = fg.generate_features_for_hmm(test_data)
        print(f"✅ generate_features_for_hmm completed successfully")
        print(f"📊 Generated {features.shape[1]} features for {features.shape[0]} samples")
        
        # Check if we have the expected number of features (should be around 100)
        if features.shape[1] >= 50:  # Reasonable minimum
            print(f"✅ Feature count looks good: {features.shape[1]} features")
            return True
        else:
            print(f"⚠️ Feature count seems low: {features.shape[1]} features")
            return False
            
    except Exception as e:
        print(f"❌ generate_features_for_hmm failed: {e}")
        import traceback
        print(f"❌ Error details: {traceback.format_exc()}")
        return False

def test_hmm_processes_import():
    """Test that HMM processes can import what they need."""
    print("\n🔍 Testing HMM processes imports...")
    
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
        
    except Exception as e:
        print(f"❌ HMM processes import failed: {e}")
        import traceback
        print(f"❌ Error details: {traceback.format_exc()}")
        return False

def test_feature_generation_system():
    """Test the new unified feature generation system."""
    print("\n🔍 Testing new unified feature generation system...")
    
    try:
        from src.feature_generation import FeatureGenerators as NewFeatureGenerators
        print("✅ New FeatureGenerators imported successfully")
        
        # Test that it has the expected method
        fg = NewFeatureGenerators()
        if hasattr(fg, 'generate_features_for_hmm'):
            print("✅ New FeatureGenerators has generate_features_for_hmm method")
            return True
        else:
            print("❌ New FeatureGenerators missing generate_features_for_hmm method")
            return False
            
    except Exception as e:
        print(f"❌ New feature generation system test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🚀 Testing HMM Feature Compatibility")
    print("=" * 50)
    
    tests = [
        test_feature_generators_import,
        test_generate_features_for_hmm,
        test_hmm_processes_import,
        test_feature_generation_system
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
        print("\n🎉 All tests passed! HMM processes should be able to find their features.")
    else:
        print("\n⚠️ Some tests failed. HMM processes may have issues finding features.")
    
    return all(results)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)