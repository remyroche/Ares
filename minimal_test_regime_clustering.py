#!/usr/bin/env python3
"""
Minimal test for regime clustering fixes - direct import to avoid dependency issues.
"""

import sys
import os

# Add src to path
sys.path.append('src')

def test_direct_import():
    """Test direct import of the regime clustering step."""
    print("🧪 Testing Direct Import of RegimeClusteringStep")
    print("=" * 50)
    
    try:
        # Direct import to avoid circular dependencies
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "regime_clustering_step", 
            "src/training/steps/market_analysis/regime_clustering_step.py"
        )
        regime_clustering_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(regime_clustering_module)
        
        RegimeClusteringStep = regime_clustering_module.RegimeClusteringStep
        
        # Create instance
        step = RegimeClusteringStep()
        print("✅ RegimeClusteringStep imported and initialized successfully")
        
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

def test_method_functionality():
    """Test basic functionality of implemented methods."""
    print("\n🧪 Testing Method Functionality")
    print("=" * 50)
    
    try:
        # Direct import
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "regime_clustering_step", 
            "src/training/steps/market_analysis/regime_clustering_step.py"
        )
        regime_clustering_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(regime_clustering_module)
        
        RegimeClusteringStep = regime_clustering_module.RegimeClusteringStep
        step = RegimeClusteringStep()
        
        # Test configuration validation
        valid_config = {
            'symbol': 'ETHUSDT',
            'exchange': 'binance',
            'timeframe': '1h',
            'execution_mode': 'light'
        }
        
        try:
            step._validate_config(valid_config)
            print("✅ Configuration validation working")
        except Exception as e:
            print(f"❌ Configuration validation failed: {e}")
            return False
        
        # Test error handling
        error = ValueError("Test error")
        response = step._handle_execution_error(error, {})
        
        if response['success'] == False and 'error_type' in response:
            print("✅ Error handling working")
        else:
            print("❌ Error handling not working")
            return False
        
        # Test placeholder cluster creation
        try:
            placeholder = step._create_placeholder_clusters(valid_config)
            if 'artifacts' in placeholder and 'regime_clusters' in placeholder['artifacts']:
                print("✅ Placeholder cluster creation working")
            else:
                print("❌ Placeholder cluster creation not working")
                return False
        except Exception as e:
            print(f"❌ Placeholder cluster creation failed: {e}")
            return False
        
        print("✅ All functionality tests passed")
        return True
        
    except Exception as e:
        print(f"❌ Functionality test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🚀 Minimal Regime Clustering Fix Test")
    print("=" * 60)
    
    success = True
    
    # Test 1: Direct import and method existence
    if not test_direct_import():
        success = False
    
    # Test 2: Basic functionality
    if not test_method_functionality():
        success = False
    
    print("\n" + "=" * 60)
    if success:
        print("🎉 ALL TESTS PASSED! Regime clustering fixes are working correctly.")
    else:
        print("❌ SOME TESTS FAILED! Please check the implementation.")
    print("=" * 60)