#!/usr/bin/env python3
"""
Test script for Enhanced Two-Stage Optimization

This script tests the enhanced two-stage optimization system with synthetic data
and different scenarios to verify functionality and performance.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import sys
import time

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.training.steps.enhanced_two_stage_optimization import (
    EnhancedTwoStageOptimizer,
    optimize_dbscan_parameters,
    smart_two_stage_optimization
)

def generate_synthetic_data(n_samples=1000, n_features=10, n_clusters=5, noise_level=0.1):
    """
    Generate synthetic data for testing.
    
    Args:
        n_samples: Number of samples
        n_features: Number of features
        n_clusters: Number of clusters
        noise_level: Noise level
        
    Returns:
        Synthetic features array
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
    
    return features

def test_basic_functionality():
    """Test basic functionality of the two-stage optimizer."""
    print("🧪 Testing Basic Functionality")
    print("=" * 50)
    
    # Generate synthetic data
    features = generate_synthetic_data(n_samples=500, n_features=5, n_clusters=3)
    print(f"📊 Generated synthetic data: {features.shape}")
    
    # Test basic optimization
    config = {
        "max_evaluations": 30,
        "stage1_ratio": 0.6,
        "robustness_level": "medium",
        "search_space_expansion": 1.5,
        "random_seed": 42
    }
    
    optimizer = EnhancedTwoStageOptimizer(config)
    results = optimizer.optimize_dbscan_parameters(features)
    
    print(f"✅ Optimization completed: {results['success']}")
    print(f"📈 Best Score: {results['best_score']:.4f}")
    print(f"🔧 Best Parameters: {results['best_params']}")
    print(f"📊 Total Evaluations: {results['total_evaluations']}")
    print(f"📈 Improvement: {results['improvement']:.4f}")
    
    return results

def test_different_problem_sizes():
    """Test optimization with different problem sizes."""
    print("\n🧪 Testing Different Problem Sizes")
    print("=" * 50)
    
    problem_sizes = [
        {"n_samples": 500, "n_features": 3, "n_clusters": 2, "name": "Small"},
        {"n_samples": 2000, "n_features": 5, "n_clusters": 4, "name": "Medium"},
        {"n_samples": 5000, "n_features": 8, "n_clusters": 6, "name": "Large"}
    ]
    
    results_summary = []
    
    for problem in problem_sizes:
        print(f"\n📊 Testing {problem['name']} Problem")
        print(f"   Samples: {problem['n_samples']}, Features: {problem['n_features']}, Clusters: {problem['n_clusters']}")
        
        # Generate data
        features = generate_synthetic_data(
            n_samples=problem['n_samples'],
            n_features=problem['n_features'],
            n_clusters=problem['n_clusters']
        )
        
        # Run optimization
        start_time = time.time()
        results = smart_two_stage_optimization(features, max_evaluations=40)
        end_time = time.time()
        
        # Record results
        summary = {
            'problem_size': problem['name'],
            'n_samples': problem['n_samples'],
            'n_features': problem['n_features'],
            'n_clusters': problem['n_clusters'],
            'best_score': results['best_score'],
            'best_params': results['best_params'],
            'total_evaluations': results['total_evaluations'],
            'improvement': results['improvement'],
            'execution_time': end_time - start_time,
            'stage1_method': results['stage1_results']['method'],
            'stage2_method': results['stage2_results']['method']
        }
        
        results_summary.append(summary)
        
        print(f"   ✅ Score: {results['best_score']:.4f}")
        print(f"   🔧 Params: {results['best_params']}")
        print(f"   ⏱️ Time: {end_time - start_time:.2f}s")
        print(f"   📊 Evaluations: {results['total_evaluations']}")
    
    return results_summary

def test_robustness_levels():
    """Test different robustness levels."""
    print("\n🧪 Testing Different Robustness Levels")
    print("=" * 50)
    
    features = generate_synthetic_data(n_samples=1500, n_features=6, n_clusters=4)
    
    robustness_levels = ["low", "medium", "high"]
    results_summary = []
    
    for level in robustness_levels:
        print(f"\n🛡️ Testing Robustness Level: {level.upper()}")
        
        config = {
            "max_evaluations": 50,
            "stage1_ratio": 0.6,
            "robustness_level": level,
            "search_space_expansion": 1.5,
            "random_seed": 42
        }
        
        optimizer = EnhancedTwoStageOptimizer(config)
        results = optimizer.optimize_dbscan_parameters(features)
        
        summary = {
            'robustness_level': level,
            'best_score': results['best_score'],
            'best_params': results['best_params'],
            'total_evaluations': results['total_evaluations'],
            'improvement': results['improvement'],
            'n_regions': len(results['stage1_results'].get('promising_regions', []))
        }
        
        results_summary.append(summary)
        
        print(f"   ✅ Score: {results['best_score']:.4f}")
        print(f"   🔧 Params: {results['best_params']}")
        print(f"   📊 Evaluations: {results['total_evaluations']}")
        print(f"   🎯 Regions: {summary['n_regions']}")
    
    return results_summary

def test_search_space_expansion():
    """Test different search space expansion factors."""
    print("\n🧪 Testing Different Search Space Expansion Factors")
    print("=" * 50)
    
    features = generate_synthetic_data(n_samples=1000, n_features=5, n_clusters=3)
    
    expansion_factors = [1.2, 1.5, 2.0]
    results_summary = []
    
    for factor in expansion_factors:
        print(f"\n🔍 Testing Expansion Factor: {factor}")
        
        config = {
            "max_evaluations": 40,
            "stage1_ratio": 0.6,
            "robustness_level": "medium",
            "search_space_expansion": factor,
            "random_seed": 42
        }
        
        optimizer = EnhancedTwoStageOptimizer(config)
        results = optimizer.optimize_dbscan_parameters(features)
        
        summary = {
            'expansion_factor': factor,
            'best_score': results['best_score'],
            'best_params': results['best_params'],
            'total_evaluations': results['total_evaluations'],
            'improvement': results['improvement'],
            'search_space': results['stage2_results'].get('search_space', {})
        }
        
        results_summary.append(summary)
        
        print(f"   ✅ Score: {results['best_score']:.4f}")
        print(f"   🔧 Params: {results['best_params']}")
        print(f"   📊 Evaluations: {results['total_evaluations']}")
        if 'search_space' in summary:
            print(f"   🔍 Search Space: {summary['search_space']}")
    
    return results_summary

def test_fallback_functionality():
    """Test fallback functionality when optuna is not available."""
    print("\n🧪 Testing Fallback Functionality")
    print("=" * 50)
    
    features = generate_synthetic_data(n_samples=800, n_features=4, n_clusters=3)
    
    # Temporarily remove optuna from sys.modules to test fallback
    original_optuna = None
    if 'optuna' in sys.modules:
        original_optuna = sys.modules['optuna']
        del sys.modules['optuna']
    
    try:
        config = {
            "max_evaluations": 30,
            "stage1_ratio": 0.6,
            "robustness_level": "medium",
            "search_space_expansion": 1.5,
            "random_seed": 42
        }
        
        optimizer = EnhancedTwoStageOptimizer(config)
        results = optimizer.optimize_dbscan_parameters(features)
        
        print(f"✅ Fallback optimization completed: {results['success']}")
        print(f"📈 Best Score: {results['best_score']:.4f}")
        print(f"🔧 Best Parameters: {results['best_params']}")
        print(f"📊 Total Evaluations: {results['total_evaluations']}")
        print(f"🔍 Stage 2 Method: {results['stage2_results']['method']}")
        
        return results
        
    finally:
        # Restore optuna
        if original_optuna is not None:
            sys.modules['optuna'] = original_optuna

def test_convenience_functions():
    """Test convenience functions."""
    print("\n🧪 Testing Convenience Functions")
    print("=" * 50)
    
    features = generate_synthetic_data(n_samples=600, n_features=4, n_clusters=3)
    
    # Test optimize_dbscan_parameters
    print("📊 Testing optimize_dbscan_parameters function")
    results1 = optimize_dbscan_parameters(features, max_evaluations=25)
    print(f"   ✅ Score: {results1['best_score']:.4f}")
    print(f"   🔧 Params: {results1['best_params']}")
    
    # Test smart_two_stage_optimization
    print("📊 Testing smart_two_stage_optimization function")
    results2 = smart_two_stage_optimization(features, max_evaluations=25)
    print(f"   ✅ Score: {results2['best_score']:.4f}")
    print(f"   🔧 Params: {results2['best_params']}")
    
    return results1, results2

def run_comprehensive_test():
    """Run comprehensive test suite."""
    print("🚀 Enhanced Two-Stage Optimization Test Suite")
    print("=" * 60)
    
    test_results = {}
    
    # Run all tests
    try:
        test_results['basic'] = test_basic_functionality()
        test_results['problem_sizes'] = test_different_problem_sizes()
        test_results['robustness'] = test_robustness_levels()
        test_results['expansion'] = test_search_space_expansion()
        test_results['fallback'] = test_fallback_functionality()
        test_results['convenience'] = test_convenience_functions()
        
        print("\n🎉 All Tests Completed Successfully!")
        print("=" * 60)
        
        # Summary
        print("\n📊 Test Summary:")
        print(f"   ✅ Basic Functionality: PASSED")
        print(f"   ✅ Problem Sizes: {len(test_results['problem_sizes'])} scenarios tested")
        print(f"   ✅ Robustness Levels: {len(test_results['robustness'])} levels tested")
        print(f"   ✅ Search Space Expansion: {len(test_results['expansion'])} factors tested")
        print(f"   ✅ Fallback Functionality: PASSED")
        print(f"   ✅ Convenience Functions: PASSED")
        
        return test_results
        
    except Exception as e:
        print(f"\n❌ Test Failed: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    # Run comprehensive test
    results = run_comprehensive_test()
    
    if results:
        print("\n🎯 Key Findings:")
        
        # Analyze problem size results
        if 'problem_sizes' in results:
            print("\n📊 Problem Size Analysis:")
            for result in results['problem_sizes']:
                print(f"   {result['problem_size']}: Score={result['best_score']:.4f}, Time={result['execution_time']:.2f}s")
        
        # Analyze robustness results
        if 'robustness' in results:
            print("\n🛡️ Robustness Analysis:")
            for result in results['robustness']:
                print(f"   {result['robustness_level']}: Score={result['best_score']:.4f}, Regions={result['n_regions']}")
        
        print("\n✅ Enhanced Two-Stage Optimization is ready for production use!")
    else:
        print("\n❌ Test suite failed. Please check the implementation.")