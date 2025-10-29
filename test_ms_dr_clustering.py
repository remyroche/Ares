#!/usr/bin/env python3
"""
Test script for MS-DR clustering
"""

import numpy as np
import pandas as pd
import sys
import os
from pathlib import Path

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.training.steps.market_analysis.ms_dr_clustering.ms_dr_clusterer import (
    MSDRClusterer, MSDRConfig, create_ms_dr_clusterer
)
from src.training.steps.market_analysis.ms_dr_clustering.ms_dr_auto_tuner import (
    MSDRAutoTuner, auto_tune_ms_dr_clustering
)

def create_sample_data(n_samples=1000, n_features=10, n_regimes=3):
    """Create sample market data with regime switching."""
    np.random.seed(42)
    
    # Create regime-dependent data
    regime_lengths = np.random.multinomial(n_samples, [1/n_regimes] * n_regimes)
    regime_starts = np.cumsum([0] + regime_lengths[:-1])
    
    data = np.zeros((n_samples, n_features))
    regime_labels = np.zeros(n_samples, dtype=int)
    
    for i in range(n_regimes):
        start_idx = regime_starts[i]
        end_idx = start_idx + regime_lengths[i]
        
        # Different means and variances for each regime
        regime_mean = np.random.normal(0, 2, n_features)
        regime_std = np.random.uniform(0.5, 2.0, n_features)
        
        # Generate data for this regime
        regime_data = np.random.normal(regime_mean, regime_std, (regime_lengths[i], n_features))
        data[start_idx:end_idx] = regime_data
        regime_labels[start_idx:end_idx] = i
    
    # Add some noise
    data += np.random.normal(0, 0.1, data.shape)
    
    # Create feature names
    feature_names = [f'feature_{i}' for i in range(n_features)]
    
    return pd.DataFrame(data, columns=feature_names), regime_labels

def test_basic_clustering():
    """Test basic MS-DR clustering."""
    print("=" * 80)
    print("TESTING BASIC MS-DR CLUSTERING")
    print("=" * 80)
    
    # Create sample data
    data, true_labels = create_sample_data(n_samples=500, n_features=8, n_regimes=3)
    print(f"Created sample data: {data.shape}")
    print(f"True regimes: {len(np.unique(true_labels))}")
    
    # Test basic clustering
    try:
        clusterer = create_ms_dr_clusterer(
            n_regimes=3,
            model_type='autoregression',
            order=1,
            switching_variance=True,
            auto_select_regimes=False,
            enable_pca=True,
            pca_components=5,
            random_state=42
        )
        
        print("\n🔍 Running MS-DR clustering...")
        result = clusterer.fit_predict(data.values)
        
        if result.success:
            print("✅ MS-DR clustering successful!")
            print(f"   Discovered regimes: {result.n_clusters}")
            print(f"   Silhouette score: {result.silhouette_score:.4f}")
            print(f"   AIC: {result.aic:.2f}")
            print(f"   BIC: {result.bic:.2f}")
            print(f"   Processing time: {result.processing_time:.2f}s")
            print(f"   Memory usage: {result.memory_usage_mb:.2f} MB")
            
            # Show regime distribution
            unique_labels, counts = np.unique(result.cluster_labels, return_counts=True)
            print(f"   Regime distribution: {dict(zip(unique_labels, counts))}")
            
            # Show transition matrix if available
            if result.transition_matrix is not None:
                print(f"   Transition matrix:\n{result.transition_matrix}")
            
            return result
        else:
            print(f"❌ MS-DR clustering failed: {result.error_message}")
            return None
            
    except Exception as e:
        print(f"❌ Error during clustering: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_auto_tuning():
    """Test MS-DR auto-tuning."""
    print("\n" + "=" * 80)
    print("TESTING MS-DR AUTO-TUNING")
    print("=" * 80)
    
    # Create sample data
    data, true_labels = create_sample_data(n_samples=300, n_features=6, n_regimes=2)
    print(f"Created sample data: {data.shape}")
    
    try:
        print("\n🎯 Running MS-DR auto-tuning...")
        tuning_result = auto_tune_ms_dr_clustering(
            data=data,
            n_trials=20,  # Small number for testing
            timeout_minutes=5.0,
            enable_staged_optimization=True
        )
        
        print("✅ Auto-tuning completed!")
        print(f"   Best score: {tuning_result['best_score']:.4f}")
        print(f"   Best parameters: {tuning_result['best_params']}")
        print(f"   Total trials: {len(tuning_result['trial_history'])}")
        
        # Show trial history
        print("\n📊 Trial History:")
        for i, trial in enumerate(tuning_result['trial_history'][:5]):  # Show first 5
            if trial['success']:
                print(f"   Trial {i+1}: Score={trial['composite_score']:.4f}, "
                      f"n_regimes={trial['params']['n_regimes']}")
            else:
                print(f"   Trial {i+1}: FAILED - {trial.get('error', {}).get('error_message', 'Unknown error')}")
        
        return tuning_result
        
    except Exception as e:
        print(f"❌ Error during auto-tuning: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_hierarchical_tuning():
    """Test hierarchical MS-DR tuning."""
    print("\n" + "=" * 80)
    print("TESTING HIERARCHICAL MS-DR TUNING")
    print("=" * 80)
    
    # Create sample data
    data, true_labels = create_sample_data(n_samples=200, n_features=5, n_regimes=2)
    print(f"Created sample data: {data.shape}")
    
    try:
        tuner = MSDRAutoTuner()
        print("\n🚀 Running hierarchical MS-DR tuning...")
        
        hierarchical_result = tuner.auto_tune_hierarchical(
            data=data,
            n_trials_per_group=10,  # Small number for testing
            timeout_minutes=3.0,
            use_adaptive_bounds=True
        )
        
        print("✅ Hierarchical tuning completed!")
        print(f"   Best score: {hierarchical_result['best_score']:.4f}")
        print(f"   Best parameters: {hierarchical_result['best_params']}")
        
        return hierarchical_result
        
    except Exception as e:
        print(f"❌ Error during hierarchical tuning: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    """Main test function."""
    print("🚀 MS-DR CLUSTERING TEST SUITE")
    print("=" * 80)
    
    # Test 1: Basic clustering
    basic_result = test_basic_clustering()
    
    # Test 2: Auto-tuning
    tuning_result = test_auto_tuning()
    
    # Test 3: Hierarchical tuning
    hierarchical_result = test_hierarchical_tuning()
    
    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    
    tests_passed = 0
    total_tests = 3
    
    if basic_result and basic_result.success:
        print("✅ Basic clustering: PASSED")
        tests_passed += 1
    else:
        print("❌ Basic clustering: FAILED")
    
    if tuning_result and tuning_result.get('best_score', float('-inf')) > float('-inf'):
        print("✅ Auto-tuning: PASSED")
        tests_passed += 1
    else:
        print("❌ Auto-tuning: FAILED")
    
    if hierarchical_result and hierarchical_result.get('best_score', float('-inf')) > float('-inf'):
        print("✅ Hierarchical tuning: PASSED")
        tests_passed += 1
    else:
        print("❌ Hierarchical tuning: FAILED")
    
    print(f"\n🎯 Tests passed: {tests_passed}/{total_tests}")
    
    if tests_passed == total_tests:
        print("🎉 All tests passed! MS-DR clustering is working correctly.")
    else:
        print("⚠️ Some tests failed. Check the error messages above.")
    
    return tests_passed == total_tests

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
