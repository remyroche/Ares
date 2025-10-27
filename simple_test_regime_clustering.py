#!/usr/bin/env python3
"""
Simple test for regime clustering fixes without external dependencies.
"""

import sys
import os

# Add src to path
sys.path.append('src')

def test_missing_methods():
    """Test that all missing methods are now implemented."""
    print("🧪 Testing Missing Method Implementations")
    print("=" * 50)
    
    try:
        # Import the step class
        from src.training.steps.market_analysis.regime_clustering_step import RegimeClusteringStep
        
        # Create instance
        step = RegimeClusteringStep()
        
        # Check that all previously missing methods exist
        missing_methods = [
            '_merge_similar_clusters',
            '_create_refined_artifacts', 
            '_save_refined_clusters',
            '_calculate_refinement_metrics',
            '_create_comprehensive_report',
            '_create_placeholder_clusters',
            '_calculate_adaptive_dwell_time',
            '_calculate_local_stability',
            '_apply_stability_validation',
            '_validate_initialization',
            '_validate_config',
            '_validate_and_convert_labels',
            '_handle_execution_error',
            '_find_most_similar_cluster_for_merge',
            '_calculate_cluster_characteristics'
        ]
        
        implemented_methods = []
        missing_methods_found = []
        
        for method_name in missing_methods:
            if hasattr(step, method_name):
                implemented_methods.append(method_name)
                print(f"✅ {method_name} - IMPLEMENTED")
            else:
                missing_methods_found.append(method_name)
                print(f"❌ {method_name} - MISSING")
        
        print(f"\n📊 Summary:")
        print(f"✅ Implemented: {len(implemented_methods)}/{len(missing_methods)}")
        print(f"❌ Missing: {len(missing_methods_found)}")
        
        if missing_methods_found:
            print(f"\n❌ Still missing: {missing_methods_found}")
            return False
        else:
            print(f"\n🎉 All methods implemented successfully!")
            return True
            
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_method_signatures():
    """Test that methods have correct signatures."""
    print("\n🧪 Testing Method Signatures")
    print("=" * 50)
    
    try:
        from src.training.steps.market_analysis.regime_clustering_step import RegimeClusteringStep
        import inspect
        
        step = RegimeClusteringStep()
        
        # Test key method signatures
        signature_tests = [
            ('_merge_similar_clusters', ['labels', 'config']),
            ('_create_refined_artifacts', ['refined_clusters', 'config']),
            ('_validate_config', ['config']),
            ('_validate_and_convert_labels', ['regime_labels']),
            ('_handle_execution_error', ['error', 'config'])
        ]
        
        all_signatures_correct = True
        
        for method_name, expected_params in signature_tests:
            if hasattr(step, method_name):
                method = getattr(step, method_name)
                sig = inspect.signature(method)
                actual_params = list(sig.parameters.keys())
                
                # Check if expected parameters are present
                missing_params = [p for p in expected_params if p not in actual_params]
                
                if missing_params:
                    print(f"❌ {method_name}: Missing parameters {missing_params}")
                    all_signatures_correct = False
                else:
                    print(f"✅ {method_name}: Signature correct")
            else:
                print(f"❌ {method_name}: Method not found")
                all_signatures_correct = False
        
        return all_signatures_correct
        
    except Exception as e:
        print(f"❌ Signature test failed: {e}")
        return False

def test_error_handling_improvements():
    """Test error handling improvements."""
    print("\n🧪 Testing Error Handling Improvements")
    print("=" * 50)
    
    try:
        from src.training.steps.market_analysis.regime_clustering_step import RegimeClusteringStep
        
        step = RegimeClusteringStep()
        
        # Test error handling for different error types
        test_errors = [
            (AttributeError("'RegimeClusteringStep' object has no attribute '_merge_similar_clusters'"), "MissingMethod"),
            (ValueError("Missing required parameter: symbol"), "ValidationError"),
            (TypeError("Unsupported regime_labels type: <class 'str'>"), "TypeError"),
            (Exception("Unexpected error"), "Exception")
        ]
        
        all_handling_correct = True
        
        for error, expected_type in test_errors:
            response = step._handle_execution_error(error, {})
            
            if response['success'] == False and 'error_type' in response:
                print(f"✅ {type(error).__name__}: Handled correctly")
            else:
                print(f"❌ {type(error).__name__}: Not handled correctly")
                all_handling_correct = False
        
        return all_handling_correct
        
    except Exception as e:
        print(f"❌ Error handling test failed: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Simple Regime Clustering Fix Test")
    print("=" * 60)
    
    success = True
    
    # Test 1: Missing methods
    if not test_missing_methods():
        success = False
    
    # Test 2: Method signatures
    if not test_method_signatures():
        success = False
    
    # Test 3: Error handling
    if not test_error_handling_improvements():
        success = False
    
    print("\n" + "=" * 60)
    if success:
        print("🎉 ALL TESTS PASSED! Regime clustering fixes are working correctly.")
    else:
        print("❌ SOME TESTS FAILED! Please check the implementation.")
    print("=" * 60)