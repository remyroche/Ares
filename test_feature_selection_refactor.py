"""
Test script to verify the refactored feature selection framework functionality.
"""

import numpy as np
import pandas as pd
import sys
import os
from typing import List, Dict, Any

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_modular_framework():
    """Test the new modular feature selection framework."""
    print("🧪 Testing new modular feature selection framework...")
    
    try:
        # Import the new framework
        from src.training.utils.feature_selection import FeatureSelectionFramework
        print("✅ Successfully imported new modular framework")
        
        # Create sample data
        np.random.seed(42)
        n_samples, n_features = 1000, 50
        X = np.random.randn(n_samples, n_features)
        y = np.random.randn(n_samples)
        feature_names = [f'feature_{i}' for i in range(n_features)]
        
        print(f"📊 Created sample data: {X.shape[0]} samples, {X.shape[1]} features")
        
        # Initialize framework
        config = {
            'enable_gpu': False,  # Disable GPU for testing
            'enable_parallel': False,  # Disable parallel processing for testing
            'max_workers': 1,
            'memory_threshold': 0.8,
            'random_state': 42
        }
        
        framework = FeatureSelectionFramework(config)
        print("✅ Successfully initialized framework")
        
        # Test basic functionality
        print("🔍 Testing basic feature selection...")
        
        # Run a simple feature selection
        result = framework.run_comprehensive_feature_selection(
            X, y, feature_names,
            target_features=20,
            model_type='default',
            enable_stability_analysis=False,  # Disable for faster testing
            enable_temporal_analysis=False,
            enable_causal_analysis=False
        )
        
        if result.get('success', False):
            selected_features = result.get('final_selected_features', [])
            print(f"✅ Feature selection completed successfully")
            print(f"📊 Selected {len(selected_features)} features")
            print(f"🎯 Selected features: {selected_features[:10]}...")  # Show first 10
            
            # Test individual components
            print("🔍 Testing individual components...")
            
            # Test data validator
            validation_result = framework.data_validator.validate_data_quality(X, y)
            print(f"✅ Data validation: {validation_result.get('is_valid', False)}")
            
            # Test mRMR selector
            mrmr_result = framework.mrmr_selector.select_features(X, y, feature_names, 10)
            print(f"✅ mRMR selection: {len(mrmr_result.get('selected_features', []))} features")
            
            # Test correlation filter
            corr_result = framework.correlation_filter.select_features(X, y, feature_names)
            print(f"✅ Correlation filtering: {len(corr_result.get('selected_features', []))} features")
            
            return True
            
        else:
            print(f"❌ Feature selection failed: {result.get('pipeline_summary', {}).get('error', 'Unknown error')}")
            return False
            
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_backwards_compatibility():
    """Test backwards compatibility with the original interface."""
    print("\n🧪 Testing backwards compatibility...")
    
    try:
        # Test importing from the new location
        from src.training.utils.feature_selection import FeatureSelectionFramework as NewFramework
        print("✅ Successfully imported from new location")
        
        # Test that the interface is compatible
        config = {'random_state': 42}
        framework = NewFramework(config)
        
        # Test that the main method exists
        assert hasattr(framework, 'run_comprehensive_feature_selection'), "Main method missing"
        print("✅ Main interface method exists")
        
        # Test that other expected methods exist
        expected_methods = [
            'get_model_target_features',
            'get_optimization_stats',
            'check_system_requirements'
        ]
        
        for method in expected_methods:
            assert hasattr(framework, method), f"Method {method} missing"
        
        print("✅ All expected methods exist")
        return True
        
    except Exception as e:
        print(f"❌ Backwards compatibility test failed: {e}")
        return False

def test_component_isolation():
    """Test that individual components can be used independently."""
    print("\n🧪 Testing component isolation...")
    
    try:
        # Test individual component imports
        from src.training.utils.feature_selection import (
            DataValidator, MRMRSelector, CorrelationBasedFilter,
            StabilityAnalyzer, QualityMetricsCalculator
        )
        print("✅ Successfully imported individual components")
        
        # Create sample data
        np.random.seed(42)
        X = np.random.randn(100, 20)
        y = np.random.randn(100)
        feature_names = [f'feature_{i}' for i in range(20)]
        
        # Test DataValidator
        validator = DataValidator()
        validation_result = validator.validate_data_quality(X, y)
        print(f"✅ DataValidator works independently: {validation_result.get('is_valid', False)}")
        
        # Test MRMRSelector
        mrmr_selector = MRMRSelector()
        mrmr_result = mrmr_selector.select_features(X, y, feature_names, 10)
        print(f"✅ MRMRSelector works independently: {len(mrmr_result.get('selected_features', []))} features")
        
        # Test CorrelationBasedFilter
        corr_filter = CorrelationBasedFilter()
        corr_result = corr_filter.select_features(X, y, feature_names)
        print(f"✅ CorrelationBasedFilter works independently: {len(corr_result.get('selected_features', []))} features")
        
        return True
        
    except Exception as e:
        print(f"❌ Component isolation test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🚀 Starting feature selection framework refactor tests...\n")
    
    tests = [
        ("Modular Framework", test_modular_framework),
        ("Backwards Compatibility", test_backwards_compatibility),
        ("Component Isolation", test_component_isolation)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n{'='*50}")
        print(f"Running: {test_name}")
        print('='*50)
        
        try:
            result = test_func()
            results.append((test_name, result))
            if result:
                print(f"✅ {test_name} PASSED")
            else:
                print(f"❌ {test_name} FAILED")
        except Exception as e:
            print(f"❌ {test_name} FAILED with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print(f"\n{'='*50}")
    print("TEST SUMMARY")
    print('='*50)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name}: {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The refactoring is successful.")
        return True
    else:
        print("⚠️ Some tests failed. Please review the issues above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)