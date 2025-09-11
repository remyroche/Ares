from src.utils.tprint import tprint

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
    tprint("🧪 Testing new modular feature selection framework...")
    
    try:
        # Import the new framework
        from src.training.utils.feature_selection import FeatureSelectionFramework
        tprint("✅ Successfully imported new modular framework")
        
        # Create sample data
        np.random.seed(42)
        n_samples, n_features = 1000, 50
        X = np.random.randn(n_samples, n_features)
        y = np.random.randn(n_samples)
        feature_names = [f'feature_{i}' for i in range(n_features)]
        
        tprint(f"📊 Created sample data: {X.shape[0]} samples, {X.shape[1]} features")
        
        # Initialize framework
        config = {
            'enable_gpu': False,  # Disable GPU for testing
            'enable_parallel': False,  # Disable parallel processing for testing
            'max_workers': 1,
            'memory_threshold': 0.8,
            'random_state': 42
        }
        
        framework = FeatureSelectionFramework(config)
        tprint("✅ Successfully initialized framework")
        
        # Test basic functionality
        tprint("🔍 Testing basic feature selection...")
        
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
            tprint(f"✅ Feature selection completed successfully")
            tprint(f"📊 Selected {len(selected_features)} features")
            tprint(f"🎯 Selected features: {selected_features[:10]}...")  # Show first 10
            
            # Test individual components
            tprint("🔍 Testing individual components...")
            
            # Test data validator
            validation_result = framework.data_validator.validate_data_quality(X, y)
            tprint(f"✅ Data validation: {validation_result.get('is_valid', False)}")
            
            # Test mRMR selector
            mrmr_result = framework.mrmr_selector.select_features(X, y, feature_names, 10)
            tprint(f"✅ mRMR selection: {len(mrmr_result.get('selected_features', []))} features")
            
            # Test correlation filter
            corr_result = framework.correlation_filter.select_features(X, y, feature_names)
            tprint(f"✅ Correlation filtering: {len(corr_result.get('selected_features', []))} features")
            
            return True
            
        else:
            tprint(f"❌ Feature selection failed: {result.get('pipeline_summary', {}).get('error', 'Unknown error')}")
            return False
            
    except Exception as e:
        tprint(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_backwards_compatibility():
    """Test backwards compatibility with the original interface."""
    tprint("\n🧪 Testing backwards compatibility...")
    
    try:
        # Test importing from the new location
        from src.training.utils.feature_selection import FeatureSelectionFramework as NewFramework
        tprint("✅ Successfully imported from new location")
        
        # Test that the interface is compatible
        config = {'random_state': 42}
        framework = NewFramework(config)
        
        # Test that the main method exists
        assert hasattr(framework, 'run_comprehensive_feature_selection'), "Main method missing"
        tprint("✅ Main interface method exists")
        
        # Test that other expected methods exist
        expected_methods = [
            'get_model_target_features',
            'get_optimization_stats',
            'check_system_requirements'
        ]
        
        for method in expected_methods:
            assert hasattr(framework, method), f"Method {method} missing"
        
        tprint("✅ All expected methods exist")
        return True
        
    except Exception as e:
        tprint(f"❌ Backwards compatibility test failed: {e}")
        return False

def test_component_isolation():
    """Test that individual components can be used independently."""
    tprint("\n🧪 Testing component isolation...")
    
    try:
        # Test individual component imports
        from src.training.utils.feature_selection import (
            DataValidator, MRMRSelector, CorrelationBasedFilter,
            StabilityAnalyzer, QualityMetricsCalculator
        )
        tprint("✅ Successfully imported individual components")
        
        # Create sample data
        np.random.seed(42)
        X = np.random.randn(100, 20)
        y = np.random.randn(100)
        feature_names = [f'feature_{i}' for i in range(20)]
        
        # Test DataValidator
        validator = DataValidator()
        validation_result = validator.validate_data_quality(X, y)
        tprint(f"✅ DataValidator works independently: {validation_result.get('is_valid', False)}")
        
        # Test MRMRSelector
        mrmr_selector = MRMRSelector()
        mrmr_result = mrmr_selector.select_features(X, y, feature_names, 10)
        tprint(f"✅ MRMRSelector works independently: {len(mrmr_result.get('selected_features', []))} features")
        
        # Test CorrelationBasedFilter
        corr_filter = CorrelationBasedFilter()
        corr_result = corr_filter.select_features(X, y, feature_names)
        tprint(f"✅ CorrelationBasedFilter works independently: {len(corr_result.get('selected_features', []))} features")
        
        return True
        
    except Exception as e:
        tprint(f"❌ Component isolation test failed: {e}")
        return False

def main():
    """Run all tests."""
    tprint("🚀 Starting feature selection framework refactor tests...\n")
    
    tests = [
        ("Modular Framework", test_modular_framework),
        ("Backwards Compatibility", test_backwards_compatibility),
        ("Component Isolation", test_component_isolation)
    ]
    
    results = []
    for test_name, test_func in tests:
        tprint(f"\n{'='*50}")
        tprint(f"Running: {test_name}")
        tprint('='*50)
        
        try:
            result = test_func()
            results.append((test_name, result))
            if result:
                tprint(f"✅ {test_name} PASSED")
            else:
                tprint(f"❌ {test_name} FAILED")
        except Exception as e:
            tprint(f"❌ {test_name} FAILED with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    tprint(f"\n{'='*50}")
    tprint("TEST SUMMARY")
    tprint('='*50)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        tprint(f"{test_name}: {status}")
    
    tprint(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        tprint("🎉 All tests passed! The refactoring is successful.")
        return True
    else:
        tprint("⚠️ Some tests failed. Please review the issues above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)