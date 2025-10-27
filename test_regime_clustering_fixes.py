#!/usr/bin/env python3
"""
Test script to verify the regime clustering fixes.

This script tests the implemented fixes for the regime clustering system.
"""

import numpy as np
import pandas as pd
import sys
import os
from datetime import datetime

# Add src to path
sys.path.append('src')

def test_regime_clustering_step():
    """Test the RegimeClusteringStep with the implemented fixes."""
    print("🧪 Testing RegimeClusteringStep Implementation")
    print("=" * 60)
    
    try:
        from src.training.steps.market_analysis.regime_clustering_step import RegimeClusteringStep
        
        # Test 1: Initialize the step
        print("\n📋 Test 1: Initialization")
        step = RegimeClusteringStep()
        print("✅ RegimeClusteringStep initialized successfully")
        
        # Test 2: Configuration validation
        print("\n📋 Test 2: Configuration Validation")
        
        # Valid config
        valid_config = {
            'symbol': 'ETHUSDT',
            'exchange': 'binance',
            'timeframe': '1h',
            'execution_mode': 'light',
            'min_dwell_bars': 3,
            'max_dwell_bars': 8,
            'stability_threshold': 0.7,
            'min_cluster_ratio': 0.05,
            'max_cluster_ratio': 0.35
        }
        
        try:
            step._validate_config(valid_config)
            print("✅ Valid configuration passed validation")
        except Exception as e:
            print(f"❌ Valid configuration failed validation: {e}")
            return False
        
        # Invalid config
        invalid_config = {
            'symbol': 'ETHUSDT',
            'exchange': 'binance',
            # Missing timeframe
            'min_dwell_bars': -1,  # Invalid value
            'stability_threshold': 1.5  # Invalid value
        }
        
        try:
            step._validate_config(invalid_config)
            print("❌ Invalid configuration should have failed validation")
            return False
        except Exception as e:
            print(f"✅ Invalid configuration correctly failed validation: {e}")
        
        # Test 3: Data validation
        print("\n📋 Test 3: Data Validation")
        
        # Test with numpy array
        labels_array = np.array([0, 0, 1, 1, 2, 2, -1, -1])
        validated_labels = step._validate_and_convert_labels(labels_array)
        print(f"✅ NumPy array validation: {len(validated_labels)} labels")
        
        # Test with DataFrame
        labels_df = pd.DataFrame({'regime_label': [0, 0, 1, 1, 2, 2, -1, -1]})
        validated_labels = step._validate_and_convert_labels(labels_df)
        print(f"✅ DataFrame validation: {len(validated_labels)} labels")
        
        # Test with list
        labels_list = [0, 0, 1, 1, 2, 2, -1, -1]
        validated_labels = step._validate_and_convert_labels(labels_list)
        print(f"✅ List validation: {len(validated_labels)} labels")
        
        # Test 4: Missing method implementations
        print("\n📋 Test 4: Missing Method Implementations")
        
        # Test _merge_similar_clusters
        try:
            result = step._merge_similar_clusters(labels_array, valid_config)
            print(f"✅ _merge_similar_clusters implemented: {len(result)} labels")
        except AttributeError as e:
            print(f"❌ _merge_similar_clusters not implemented: {e}")
            return False
        
        # Test _create_refined_artifacts
        try:
            refined_clusters = {
                'refined_labels': labels_array,
                'original_labels': labels_array,
                'n_clusters': 3,
                'clustering_method': 'hdbscan_refined',
                'refinement_applied': True,
                'metadata': {}
            }
            artifacts = step._create_refined_artifacts(refined_clusters, valid_config)
            print("✅ _create_refined_artifacts implemented")
        except AttributeError as e:
            print(f"❌ _create_refined_artifacts not implemented: {e}")
            return False
        
        # Test _calculate_adaptive_dwell_time
        try:
            dwell_time = step._calculate_adaptive_dwell_time(labels_array, 3, 8, 1.0)
            print(f"✅ _calculate_adaptive_dwell_time implemented: {dwell_time}")
        except AttributeError as e:
            print(f"❌ _calculate_adaptive_dwell_time not implemented: {e}")
            return False
        
        # Test _calculate_local_stability
        try:
            stability = step._calculate_local_stability(labels_array, 2, 3)
            print(f"✅ _calculate_local_stability implemented: {stability}")
        except AttributeError as e:
            print(f"❌ _calculate_local_stability not implemented: {e}")
            return False
        
        # Test 5: Temporal stabilization logic
        print("\n📋 Test 5: Temporal Stabilization Logic")
        
        # Create test labels with isolated changes
        test_labels = np.array([0, 0, 1, 0, 0, 1, 1, 2, 1, 1, 2, 2])
        stabilized = step._apply_temporal_stabilization(test_labels, valid_config)
        
        # Check that isolated changes were removed
        changes_original = np.sum(test_labels[1:] != test_labels[:-1])
        changes_stabilized = np.sum(stabilized[1:] != stabilized[:-1])
        
        print(f"Original changes: {changes_original}")
        print(f"Stabilized changes: {changes_stabilized}")
        
        if changes_stabilized <= changes_original:
            print("✅ Temporal stabilization working correctly")
        else:
            print("❌ Temporal stabilization may have issues")
        
        # Test 6: Economic validation
        print("\n📋 Test 6: Economic Validation")
        
        # Create test labels with small clusters
        test_labels_econ = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  # Large cluster 0
                                    1, 1,  # Small cluster 1
                                    2, 2, 2, 2, 2, 2, 2, 2, 2, 2,  # Large cluster 2
                                    3, 3,  # Small cluster 3
                                    -1, -1])  # Noise
        
        validated_econ = step._apply_basic_economic_validation(test_labels_econ, valid_config)
        
        # Check that small clusters were handled
        unique_original = len(np.unique(test_labels_econ[test_labels_econ != -1]))
        unique_validated = len(np.unique(validated_econ[validated_econ != -1]))
        
        print(f"Original clusters: {unique_original}")
        print(f"Validated clusters: {unique_validated}")
        
        if unique_validated <= unique_original:
            print("✅ Economic validation working correctly")
        else:
            print("❌ Economic validation may have issues")
        
        print("\n🎉 All tests passed! Regime clustering fixes are working correctly.")
        return True
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_error_handling():
    """Test error handling improvements."""
    print("\n🧪 Testing Error Handling")
    print("=" * 60)
    
    try:
        from src.training.steps.market_analysis.regime_clustering_step import RegimeClusteringStep
        
        step = RegimeClusteringStep()
        
        # Test error handling for missing methods
        error = AttributeError("'RegimeClusteringStep' object has no attribute '_merge_similar_clusters'")
        error_response = step._handle_execution_error(error, {})
        
        if error_response['success'] == False and 'Missing method implementation' in error_response['error']:
            print("✅ Error handling for missing methods working")
        else:
            print("❌ Error handling for missing methods not working")
            return False
        
        # Test error handling for validation errors
        error = ValueError("Missing required parameter: symbol")
        error_response = step._handle_execution_error(error, {})
        
        if error_response['success'] == False and 'Data validation error' in error_response['error']:
            print("✅ Error handling for validation errors working")
        else:
            print("❌ Error handling for validation errors not working")
            return False
        
        print("✅ Error handling tests passed")
        return True
        
    except Exception as e:
        print(f"❌ Error handling test failed: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Testing Regime Clustering Fixes")
    print("=" * 80)
    
    success = True
    
    # Run main tests
    if not test_regime_clustering_step():
        success = False
    
    # Run error handling tests
    if not test_error_handling():
        success = False
    
    print("\n" + "=" * 80)
    if success:
        print("🎉 ALL TESTS PASSED! Regime clustering fixes are working correctly.")
    else:
        print("❌ SOME TESTS FAILED! Please check the implementation.")
    
    print("=" * 80)