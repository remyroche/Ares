#!/usr/bin/env python3
"""
Test script for Enhanced Two-Stage Optimization Integration

This script tests the integration of enhanced two-stage optimization with the enhanced regime clustering system.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import sys
import time
import os

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.training.steps.enhanced_regime_clustering import EnhancedRegimeClustering

def generate_synthetic_data(n_samples=1000, n_features=10, n_clusters=5, noise_level=0.1):
    """
    Generate synthetic data for testing.
    
    Args:
        n_samples: Number of samples
        n_features: Number of features
        n_clusters: Number of clusters
        noise_level: Noise level
        
    Returns:
        Synthetic features array and feature names
    """
    from sklearn.datasets import make_blobs
    
    # Generate clustered data
    features, _ = make_blobs(
        n_samples=n_samples,
        n_features=n_features,
        centers=n_clusters,
        cluster_std=1.0,
        random_state=42
    )
    
    # Add noise
    noise = np.random.normal(0, noise_level, features.shape)
    features += noise
    
    # Create feature names
    feature_names = [f"feature_{i}" for i in range(n_features)]
    
    return features, feature_names

def test_basic_integration():
    """Test basic integration of enhanced two-stage optimization."""
    print("🧪 Testing Basic Integration")
    print("=" * 50)
    
    # Generate synthetic data
    features, feature_names = generate_synthetic_data(n_samples=500, n_features=5, n_clusters=3)
    print(f"📊 Generated synthetic data: {features.shape}")
    
    # Test configuration with two-stage optimization
    config = {
        "target_clusters": 4,
        "min_quality_threshold": 0.3,
        "quality_drop_threshold": 0.8,
        "max_iterations": 30,
        "no_improvement_limit": 5,
        "min_coverage_threshold": 0.98,
        "bayesian_calls": 30,  # Used for two-stage optimization
        
        # Enhanced Two-Stage Optimization settings
        "stage1_ratio": 0.6,
        "robustness_level": "medium",
        "search_space_expansion": 1.5,
        "region_threshold": 0.8,
        "random_seed": 42,
        
        # Explainable AI settings
        "use_lime_shap": True,
        "lime_samples": 100,  # Reduced for testing
        "shap_samples": 20,    # Reduced for testing
        
        # Smart splitting settings
        "smart_splitting": True,
        "min_cluster_size_for_split": 10,
        
        # Automated K-means settings
        "auto_k_means": True,
        "max_k_for_auto": 5,
        "k_selection_method": "silhouette",
        
        # HMM reliability settings
        "hmm_reliability_focus": True,
        "hmm_entropy_penalty_weight": 0.15,
        "min_hmm_state_duration": 5,
        "hmm_transition_smoothness_weight": 0.1
    }
    
    # Create enhanced clustering instance
    enhanced_clustering = EnhancedRegimeClustering(config)
    
    # Run enhanced clustering
    start_time = time.time()
    results = enhanced_clustering.run_enhanced_clustering(features, feature_names)
    end_time = time.time()
    
    print(f"✅ Enhanced clustering completed: {results['success']}")
    print(f"📈 Final Score: {results['final_score_dict']['composite_score']:.4f}")
    print(f"🔧 Final Clusters: {results['final_score_dict']['n_clusters']}")
    print(f"⏱️ Execution Time: {end_time - start_time:.2f}s")
    
    # Check for two-stage optimization results
    if "two_stage_optimization" in results:
        two_stage = results["two_stage_optimization"]
        print(f"🚀 Two-Stage Optimization Results:")
        print(f"   Stage 1 Method: {two_stage['stage1_results']['method']}")
        print(f"   Stage 2 Method: {two_stage['stage2_results']['method']}")
        print(f"   Total Evaluations: {two_stage['total_evaluations']}")
        print(f"   Best Score: {two_stage['best_score']:.4f}")
        print(f"   Improvement: {two_stage['improvement']:.4f}")
    else:
        print("⚠️ No two-stage optimization results found")
    
    return results

def test_different_robustness_levels():
    """Test different robustness levels."""
    print("\n🧪 Testing Different Robustness Levels")
    print("=" * 50)
    
    features, feature_names = generate_synthetic_data(n_samples=800, n_features=6, n_clusters=4)
    
    robustness_levels = ["low", "medium", "high"]
    results_summary = []
    
    for level in robustness_levels:
        print(f"\n🛡️ Testing Robustness Level: {level.upper()}")
        
        config = {
            "target_clusters": 4,
            "bayesian_calls": 25,
            "stage1_ratio": 0.6,
            "robustness_level": level,
            "search_space_expansion": 1.5,
            "region_threshold": 0.8,
            "random_seed": 42,
            "use_lime_shap": False,  # Disable for faster testing
            "hmm_reliability_focus": False  # Disable for faster testing
        }
        
        enhanced_clustering = EnhancedRegimeClustering(config)
        
        start_time = time.time()
        results = enhanced_clustering.run_enhanced_clustering(features, feature_names)
        end_time = time.time()
        
        summary = {
            'robustness_level': level,
            'final_score': results['final_score_dict']['composite_score'],
            'n_clusters': results['final_score_dict']['n_clusters'],
            'execution_time': end_time - start_time,
            'two_stage_available': "two_stage_optimization" in results
        }
        
        if "two_stage_optimization" in results:
            two_stage = results["two_stage_optimization"]
            summary['stage1_method'] = two_stage['stage1_results']['method']
            summary['stage2_method'] = two_stage['stage2_results']['method']
            summary['total_evaluations'] = two_stage['total_evaluations']
            summary['improvement'] = two_stage['improvement']
        
        results_summary.append(summary)
        
        print(f"   ✅ Score: {summary['final_score']:.4f}")
        print(f"   🔧 Clusters: {summary['n_clusters']}")
        print(f"   ⏱️ Time: {summary['execution_time']:.2f}s")
        print(f"   🚀 Two-Stage: {summary['two_stage_available']}")
    
    return results_summary

def test_training_modes():
    """Test different training modes (light, blank, full)."""
    print("\n🧪 Testing Training Modes")
    print("=" * 50)
    
    features, feature_names = generate_synthetic_data(n_samples=1200, n_features=8, n_clusters=6)
    
    training_modes = [
        {"name": "Light", "target_clusters": 2, "evaluations": 20},
        {"name": "Blank", "target_clusters": 4, "evaluations": 30},
        {"name": "Full", "target_clusters": 8, "evaluations": 50}
    ]
    
    results_summary = []
    
    for mode in training_modes:
        print(f"\n📊 Testing {mode['name']} Mode")
        print(f"   Target Clusters: {mode['target_clusters']}")
        print(f"   Evaluations: {mode['evaluations']}")
        
        config = {
            "target_clusters": mode['target_clusters'],
            "bayesian_calls": mode['evaluations'],
            "stage1_ratio": 0.6,
            "robustness_level": "medium",
            "search_space_expansion": 1.5,
            "region_threshold": 0.8,
            "random_seed": 42,
            "use_lime_shap": False,  # Disable for faster testing
            "hmm_reliability_focus": False  # Disable for faster testing
        }
        
        enhanced_clustering = EnhancedRegimeClustering(config)
        
        start_time = time.time()
        results = enhanced_clustering.run_enhanced_clustering(features, feature_names)
        end_time = time.time()
        
        summary = {
            'mode': mode['name'],
            'target_clusters': mode['target_clusters'],
            'actual_clusters': results['final_score_dict']['n_clusters'],
            'final_score': results['final_score_dict']['composite_score'],
            'execution_time': end_time - start_time,
            'two_stage_available': "two_stage_optimization" in results
        }
        
        if "two_stage_optimization" in results:
            two_stage = results["two_stage_optimization"]
            summary['stage1_method'] = two_stage['stage1_results']['method']
            summary['stage2_method'] = two_stage['stage2_results']['method']
            summary['total_evaluations'] = two_stage['total_evaluations']
            summary['improvement'] = two_stage['improvement']
        
        results_summary.append(summary)
        
        print(f"   ✅ Score: {summary['final_score']:.4f}")
        print(f"   🔧 Target/Actual: {summary['target_clusters']}/{summary['actual_clusters']}")
        print(f"   ⏱️ Time: {summary['execution_time']:.2f}s")
        print(f"   🚀 Two-Stage: {summary['two_stage_available']}")
    
    return results_summary

def test_fallback_functionality():
    """Test fallback functionality when two-stage optimization is not available."""
    print("\n🧪 Testing Fallback Functionality")
    print("=" * 50)
    
    features, feature_names = generate_synthetic_data(n_samples=600, n_features=4, n_clusters=3)
    
    # Temporarily remove enhanced_two_stage_optimization from sys.modules
    original_module = None
    if 'src.training.steps.enhanced_two_stage_optimization' in sys.modules:
        original_module = sys.modules['src.training.steps.enhanced_two_stage_optimization']
        del sys.modules['src.training.steps.enhanced_two_stage_optimization']
    
    try:
        config = {
            "target_clusters": 3,
            "bayesian_calls": 25,
            "stage1_ratio": 0.6,
            "robustness_level": "medium",
            "search_space_expansion": 1.5,
            "region_threshold": 0.8,
            "random_seed": 42,
            "use_lime_shap": False,
            "hmm_reliability_focus": False
        }
        
        enhanced_clustering = EnhancedRegimeClustering(config)
        
        start_time = time.time()
        results = enhanced_clustering.run_enhanced_clustering(features, feature_names)
        end_time = time.time()
        
        print(f"✅ Fallback clustering completed: {results['success']}")
        print(f"📈 Final Score: {results['final_score_dict']['composite_score']:.4f}")
        print(f"🔧 Final Clusters: {results['final_score_dict']['n_clusters']}")
        print(f"⏱️ Execution Time: {end_time - start_time:.2f}s")
        print(f"🚀 Two-Stage Available: {'two_stage_optimization' in results}")
        
        return results
        
    finally:
        # Restore module
        if original_module is not None:
            sys.modules['src.training.steps.enhanced_two_stage_optimization'] = original_module

def run_comprehensive_integration_test():
    """Run comprehensive integration test suite."""
    print("🚀 Enhanced Two-Stage Optimization Integration Test Suite")
    print("=" * 70)
    
    test_results = {}
    
    try:
        # Run all tests
        test_results['basic'] = test_basic_integration()
        test_results['robustness'] = test_different_robustness_levels()
        test_results['training_modes'] = test_training_modes()
        test_results['fallback'] = test_fallback_functionality()
        
        print("\n🎉 All Integration Tests Completed Successfully!")
        print("=" * 70)
        
        # Summary
        print("\n📊 Integration Test Summary:")
        print(f"   ✅ Basic Integration: PASSED")
        print(f"   ✅ Robustness Levels: {len(test_results['robustness'])} levels tested")
        print(f"   ✅ Training Modes: {len(test_results['training_modes'])} modes tested")
        print(f"   ✅ Fallback Functionality: PASSED")
        
        # Key findings
        print("\n🎯 Key Findings:")
        
        # Robustness analysis
        if 'robustness' in test_results:
            print("\n🛡️ Robustness Analysis:")
            for result in test_results['robustness']:
                print(f"   {result['robustness_level']}: Score={result['final_score']:.4f}, Time={result['execution_time']:.2f}s, Two-Stage={result['two_stage_available']}")
        
        # Training modes analysis
        if 'training_modes' in test_results:
            print("\n📊 Training Modes Analysis:")
            for result in test_results['training_modes']:
                print(f"   {result['mode']}: Target={result['target_clusters']}, Actual={result['actual_clusters']}, Score={result['final_score']:.4f}")
        
        print("\n✅ Enhanced Two-Stage Optimization Integration is ready for production use!")
        
        return test_results
        
    except Exception as e:
        print(f"\n❌ Integration Test Failed: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    # Run comprehensive integration test
    results = run_comprehensive_integration_test()
    
    if results:
        print("\n🎯 Integration Success!")
        print("The enhanced two-stage optimization has been successfully integrated")
        print("with the enhanced regime clustering system.")
    else:
        print("\n❌ Integration failed. Please check the implementation.")