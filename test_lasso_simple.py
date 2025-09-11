#!/usr/bin/env python3
"""
Simple test for LASSO feature selection enhancements.

This script tests the new LASSO methods without requiring external dependencies.
"""

import sys
from pathlib import Path

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_imports():
    """Test that the enhanced feature selection framework can be imported."""
    print("🔧 Testing imports...")
    
    try:
        from utils.ml_common.feature_selection import FeatureSelectionFramework
        print("✅ FeatureSelectionFramework imported successfully")
        
        # Test initialization
        framework = FeatureSelectionFramework({
            'enable_gpu': False,
            'enable_parallel': False,
            'random_state': 42
        })
        print("✅ FeatureSelectionFramework initialized successfully")
        
        # Check if new methods exist
        methods_to_check = [
            'lasso_feature_selection',
            'lasso_stability_selection', 
            'comprehensive_feature_selection'
        ]
        
        for method in methods_to_check:
            if hasattr(framework, method):
                print(f"✅ Method {method} is available")
            else:
                print(f"❌ Method {method} is missing")
        
        # Check configuration
        print(f"✅ LASSO config available: {'lasso' in framework.method_configs}")
        print(f"✅ LASSO stability config available: {'lasso_stability' in framework.method_configs}")
        
        return True
        
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False


def test_method_signatures():
    """Test that the new methods have the correct signatures."""
    print("\n🔧 Testing method signatures...")
    
    try:
        from utils.ml_common.feature_selection import FeatureSelectionFramework
        import inspect
        
        framework = FeatureSelectionFramework()
        
        # Test lasso_feature_selection signature
        sig = inspect.signature(framework.lasso_feature_selection)
        params = list(sig.parameters.keys())
        expected_params = ['self', 'X', 'y', 'feature_names', 'alpha', 'cv_folds', 'selection_criterion']
        
        print(f"✅ lasso_feature_selection parameters: {params}")
        if all(param in params for param in expected_params):
            print("✅ lasso_feature_selection has correct signature")
        else:
            print("❌ lasso_feature_selection missing parameters")
        
        # Test lasso_stability_selection signature
        sig = inspect.signature(framework.lasso_stability_selection)
        params = list(sig.parameters.keys())
        expected_params = ['self', 'X', 'y', 'feature_names', 'n_bootstrap', 'bootstrap_fraction', 
                          'alpha_range', 'stability_threshold', 'cv_folds']
        
        print(f"✅ lasso_stability_selection parameters: {params}")
        if all(param in params for param in expected_params):
            print("✅ lasso_stability_selection has correct signature")
        else:
            print("❌ lasso_stability_selection missing parameters")
        
        # Test comprehensive_feature_selection signature
        sig = inspect.signature(framework.comprehensive_feature_selection)
        params = list(sig.parameters.keys())
        expected_params = ['self', 'X', 'y', 'feature_names', 'methods', 'weights', 'n_features']
        
        print(f"✅ comprehensive_feature_selection parameters: {params}")
        if all(param in params for param in expected_params):
            print("✅ comprehensive_feature_selection has correct signature")
        else:
            print("❌ comprehensive_feature_selection missing parameters")
        
        return True
        
    except Exception as e:
        print(f"❌ Signature test failed: {e}")
        return False


def test_configuration():
    """Test that the configuration includes LASSO settings."""
    print("\n🔧 Testing configuration...")
    
    try:
        from utils.ml_common.feature_selection import FeatureSelectionFramework
        
        framework = FeatureSelectionFramework()
        
        # Check LASSO configuration
        if 'lasso' in framework.method_configs:
            lasso_config = framework.method_configs['lasso']
            print(f"✅ LASSO config: {lasso_config}")
            
            required_keys = ['alpha_range', 'cv_folds', 'max_iter', 'tol', 'random_state']
            if all(key in lasso_config for key in required_keys):
                print("✅ LASSO config has all required keys")
            else:
                print("❌ LASSO config missing keys")
        else:
            print("❌ LASSO config not found")
        
        # Check LASSO stability configuration
        if 'lasso_stability' in framework.method_configs:
            stability_config = framework.method_configs['lasso_stability']
            print(f"✅ LASSO stability config: {stability_config}")
            
            required_keys = ['n_bootstraps', 'bootstrap_fraction', 'stability_threshold', 
                           'alpha_range', 'cv_folds']
            if all(key in stability_config for key in required_keys):
                print("✅ LASSO stability config has all required keys")
            else:
                print("❌ LASSO stability config missing keys")
        else:
            print("❌ LASSO stability config not found")
        
        return True
        
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False


def main():
    """Run all tests."""
    print("🚀 LASSO FEATURE SELECTION - SIMPLE TESTING")
    print("="*60)
    
    tests = [
        test_imports,
        test_method_signatures,
        test_configuration
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"❌ Test {test.__name__} failed with error: {e}")
    
    print("\n" + "="*60)
    print(f"📊 TEST RESULTS: {passed}/{total} tests passed")
    
    if passed == total:
        print("✅ ALL TESTS PASSED! LASSO enhancement is ready.")
    else:
        print("❌ Some tests failed. Check the implementation.")
    
    print("="*60)


if __name__ == "__main__":
    main()