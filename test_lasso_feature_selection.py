#!/usr/bin/env python3
"""
Test script for LASSO feature selection enhancements.

This script demonstrates the new LASSO stability selection and comprehensive
feature selection methods added to the FeatureSelectionFramework.
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from utils.ml_common.feature_selection import FeatureSelectionFramework


def create_sample_data(n_samples=1000, n_features=50, n_informative=10, noise=0.1):
    """Create sample data for testing feature selection methods."""
    print("🔧 Creating sample data...")
    
    # Generate random features
    np.random.seed(42)
    X = np.random.randn(n_samples, n_features)
    
    # Create informative features (only first n_informative are relevant)
    informative_features = X[:, :n_informative]
    noise_features = X[:, n_informative:]
    
    # Create target with linear relationship to informative features
    coefficients = np.random.randn(n_informative) * 2
    y = np.dot(informative_features, coefficients) + np.random.randn(n_samples) * noise
    
    # Create feature names
    feature_names = [f"feature_{i:02d}" for i in range(n_features)]
    
    print(f"📊 Data created: {n_samples} samples, {n_features} features")
    print(f"📊 Informative features: {n_informative} (features 0-{n_informative-1})")
    print(f"📊 Target correlation with informative features: {np.corrcoef(y, np.dot(informative_features, coefficients))[0,1]:.3f}")
    
    return X, y, feature_names


def test_lasso_feature_selection():
    """Test the standard LASSO feature selection method."""
    print("\n" + "="*60)
    print("🧪 TESTING STANDARD LASSO FEATURE SELECTION")
    print("="*60)
    
    # Create sample data
    X, y, feature_names = create_sample_data()
    
    # Initialize framework
    framework = FeatureSelectionFramework({
        'enable_gpu': False,
        'enable_parallel': True,
        'random_state': 42
    })
    
    # Test LASSO with cross-validation
    print("\n🔍 Testing LASSO with cross-validation...")
    lasso_result = framework.lasso_feature_selection(
        X, y, feature_names,
        alpha=None,  # Use CV to find optimal alpha
        cv_folds=5
    )
    
    if 'error' not in lasso_result:
        print(f"✅ LASSO selection successful!")
        print(f"📊 Selected features: {len(lasso_result['selected_features'])}")
        print(f"📊 Optimal alpha: {lasso_result['selection_metadata']['optimal_alpha']:.6f}")
        print(f"📊 Model score: {lasso_result['selection_metadata']['model_score']:.3f}")
        print(f"📊 Selected features: {lasso_result['selected_features'][:10]}...")
        
        # Check if informative features were selected
        informative_selected = [f for f in lasso_result['selected_features'] 
                               if f.startswith('feature_0') or f.startswith('feature_0')]
        print(f"📊 Informative features selected: {len(informative_selected)}/10")
    else:
        print(f"❌ LASSO selection failed: {lasso_result['error']}")


def test_lasso_stability_selection():
    """Test the LASSO stability selection method."""
    print("\n" + "="*60)
    print("🧪 TESTING LASSO STABILITY SELECTION")
    print("="*60)
    
    # Create sample data
    X, y, feature_names = create_sample_data()
    
    # Initialize framework
    framework = FeatureSelectionFramework({
        'enable_gpu': False,
        'enable_parallel': True,
        'random_state': 42
    })
    
    # Test LASSO stability selection
    print("\n🔍 Testing LASSO stability selection...")
    stability_result = framework.lasso_stability_selection(
        X, y, feature_names,
        n_bootstrap=50,  # Reduced for faster testing
        bootstrap_fraction=0.8,
        stability_threshold=0.6,
        cv_folds=3
    )
    
    if 'error' not in stability_result:
        print(f"✅ LASSO stability selection successful!")
        print(f"📊 Selected features: {len(stability_result['selected_features'])}")
        print(f"📊 Bootstrap samples: {stability_result['selection_metadata']['n_bootstrap_successful']}")
        print(f"📊 Stability stats - Mean: {stability_result['selection_metadata']['stability_stats']['mean_stability']:.3f}")
        print(f"📊 Selected features: {stability_result['selected_features'][:10]}...")
        
        # Show stability scores for selected features
        print(f"\n📊 Stability scores for selected features:")
        for feature in stability_result['selected_features'][:5]:
            stability = stability_result['feature_stability_scores'][feature]
            coefficient = stability_result['feature_coefficients'][feature]
            print(f"  - {feature}: stability={stability:.3f}, coefficient={coefficient:.3f}")
    else:
        print(f"❌ LASSO stability selection failed: {stability_result['error']}")


def test_comprehensive_feature_selection():
    """Test the comprehensive feature selection method."""
    print("\n" + "="*60)
    print("🧪 TESTING COMPREHENSIVE FEATURE SELECTION")
    print("="*60)
    
    # Create sample data
    X, y, feature_names = create_sample_data()
    
    # Initialize framework
    framework = FeatureSelectionFramework({
        'enable_gpu': False,
        'enable_parallel': True,
        'random_state': 42
    })
    
    # Test comprehensive feature selection
    print("\n🔍 Testing comprehensive feature selection...")
    comprehensive_result = framework.comprehensive_feature_selection(
        X, y, feature_names,
        methods=['correlation', 'mrmr', 'lasso_stability'],
        weights={'correlation': 0.2, 'mrmr': 0.3, 'lasso_stability': 0.5},
        n_features=15
    )
    
    if 'error' not in comprehensive_result:
        print(f"✅ Comprehensive selection successful!")
        print(f"📊 Selected features: {len(comprehensive_result['selected_features'])}")
        print(f"📊 Methods successful: {comprehensive_result['selection_metadata']['n_methods_successful']}")
        print(f"📊 Selected features: {comprehensive_result['selected_features']}")
        
        # Show feature votes
        print(f"\n📊 Feature votes (top 10):")
        sorted_votes = sorted(comprehensive_result['feature_votes'].items(), 
                            key=lambda x: x[1], reverse=True)
        for feature, votes in sorted_votes[:10]:
            print(f"  - {feature}: {votes:.3f}")
    else:
        print(f"❌ Comprehensive selection failed: {comprehensive_result['error']}")


def test_method_comparison():
    """Compare different feature selection methods."""
    print("\n" + "="*60)
    print("🧪 COMPARING FEATURE SELECTION METHODS")
    print("="*60)
    
    # Create sample data
    X, y, feature_names = create_sample_data()
    
    # Initialize framework
    framework = FeatureSelectionFramework({
        'enable_gpu': False,
        'enable_parallel': True,
        'random_state': 42
    })
    
    methods = {
        'mRMR': lambda: framework.mrmr_selection(X, y, feature_names, 15),
        'LASSO': lambda: framework.lasso_feature_selection(X, y, feature_names, alpha=0.01),
        'LASSO Stability': lambda: framework.lasso_stability_selection(X, y, feature_names, n_bootstrap=30),
        'RFE': lambda: framework.recursive_feature_elimination(
            framework._get_default_model(y), X, y, feature_names, 15
        )
    }
    
    results = {}
    for method_name, method_func in methods.items():
        print(f"\n🔍 Testing {method_name}...")
        try:
            result = method_func()
            if 'error' not in result:
                results[method_name] = result
                print(f"✅ {method_name}: {len(result['selected_features'])} features selected")
            else:
                print(f"❌ {method_name}: {result['error']}")
        except Exception as e:
            print(f"❌ {method_name}: {e}")
    
    # Compare results
    if results:
        print(f"\n📊 METHOD COMPARISON:")
        print(f"{'Method':<20} {'Features':<10} {'Overlap with informative':<25}")
        print("-" * 60)
        
        for method_name, result in results.items():
            selected = set(result['selected_features'])
            informative = set([f"feature_{i:02d}" for i in range(10)])
            overlap = len(selected.intersection(informative))
            print(f"{method_name:<20} {len(selected):<10} {overlap}/10 ({overlap/10*100:.1f}%)")


def main():
    """Run all tests."""
    print("🚀 LASSO FEATURE SELECTION TESTING")
    print("="*60)
    
    try:
        # Test individual methods
        test_lasso_feature_selection()
        test_lasso_stability_selection()
        test_comprehensive_feature_selection()
        test_method_comparison()
        
        print("\n" + "="*60)
        print("✅ ALL TESTS COMPLETED SUCCESSFULLY!")
        print("="*60)
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()