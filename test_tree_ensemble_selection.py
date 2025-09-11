#!/usr/bin/env python3
from src.utils.tprint import tprint

"""
Test script for Tree-Based Ensemble Feature Selection.

This script demonstrates the new tree-based ensemble selection method that:
1. Collects candidate features from multiple methods
2. Trains a fast tree-based model on all candidates
3. Uses permutation importance to rank features
4. Cross-validates the final selection for generalization
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
    tprint("🔧 Creating sample data...")
    
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
    
    tprint(f"📊 Data created: {n_samples} samples, {n_features} features")
    tprint(f"📊 Informative features: {n_informative} (features 0-{n_informative-1})")
    tprint(f"📊 Target correlation with informative features: {np.corrcoef(y, np.dot(informative_features, coefficients))[0,1]:.3f}")
    
    return X, y, feature_names


def test_tree_ensemble_selection():
    """Test the tree-based ensemble selection method."""
    tprint("\n" + "="*60)
    tprint("🧪 TESTING TREE-BASED ENSEMBLE SELECTION")
    tprint("="*60)
    
    # Create sample data
    X, y, feature_names = create_sample_data()
    
    # Initialize framework
    framework = FeatureSelectionFramework({
        'enable_gpu': False,
        'enable_parallel': True,
        'random_state': 42
    })
    
    # Test tree-based ensemble selection
    tprint("\n🔍 Testing tree-based ensemble selection...")
    ensemble_result = framework.tree_based_ensemble_selection(
        X, y, feature_names,
        methods=['correlation', 'mrmr', 'lasso_stability'],
        n_features=15,
        cv_folds=5,
        n_estimators=50,  # Reduced for faster testing
        max_depth=8,
        permutation_importance_repeats=5
    )
    
    if 'error' not in ensemble_result:
        tprint(f"✅ Tree-based ensemble selection successful!")
        tprint(f"📊 Candidate features: {len(ensemble_result['candidate_features'])}")
        tprint(f"📊 Selected features: {len(ensemble_result['selected_features'])}")
        tprint(f"📊 Methods successful: {ensemble_result['selection_metadata']['n_methods_successful']}")
        tprint(f"📊 Baseline score: {ensemble_result['selection_metadata']['baseline_score']:.3f}")
        
        # Show permutation importance results
        tprint(f"\n📊 Permutation Importance (top 10):")
        sorted_importance = sorted(
            ensemble_result['permutation_importance'].items(),
            key=lambda x: x[1]['importance'],
            reverse=True
        )
        for feature, importance_data in sorted_importance[:10]:
            importance = importance_data['importance']
            std_importance = importance_data['std_importance']
            tprint(f"  - {feature}: {importance:.4f} ± {std_importance:.4f}")
        
        # Show CV validation results
        if 'cv_validation' in ensemble_result and 'error' not in ensemble_result['cv_validation']:
            cv_data = ensemble_result['cv_validation']
            tprint(f"\n📊 Cross-Validation Results:")
            tprint(f"  - Mean CV score: {cv_data['cv_mean']:.3f} ± {cv_data['cv_std']:.3f}")
            tprint(f"  - CV scores: {[f'{score:.3f}' for score in cv_data['cv_scores']]}")
            
            # Show feature importance stability
            tprint(f"\n📊 Feature Importance Stability (top 5):")
            stability_data = cv_data['feature_importance_stability']
            sorted_stability = sorted(
                stability_data.items(),
                key=lambda x: x[1]['stability'],
                reverse=True
            )
            for feature, stability_info in sorted_stability[:5]:
                stability = stability_info['stability']
                mean_imp = stability_info['mean_importance']
                tprint(f"  - {feature}: stability={stability:.3f}, mean_importance={mean_imp:.3f}")
        
        # Check if informative features were selected
        informative_selected = [f for f in ensemble_result['selected_features'] 
                               if f.startswith('feature_0') or f.startswith('feature_0')]
        tprint(f"\n📊 Informative features selected: {len(informative_selected)}/10")
        tprint(f"📊 Selected features: {ensemble_result['selected_features']}")
        
    else:
        tprint(f"❌ Tree-based ensemble selection failed: {ensemble_result['error']}")


def test_ensemble_vs_individual_methods():
    """Compare tree-based ensemble with individual methods."""
    tprint("\n" + "="*60)
    tprint("🧪 COMPARING ENSEMBLE VS INDIVIDUAL METHODS")
    tprint("="*60)
    
    # Create sample data
    X, y, feature_names = create_sample_data()
    
    # Initialize framework
    framework = FeatureSelectionFramework({
        'enable_gpu': False,
        'enable_parallel': True,
        'random_state': 42
    })
    
    # Test individual methods
    methods = {
        'mRMR': lambda: framework.mrmr_selection(X, y, feature_names, 15),
        'LASSO Stability': lambda: framework.lasso_stability_selection(X, y, feature_names, n_bootstrap=30),
        'Tree Ensemble': lambda: framework.tree_based_ensemble_selection(
            X, y, feature_names, 
            methods=['correlation', 'mrmr', 'lasso_stability'],
            n_features=15,
            cv_folds=3,
            n_estimators=30,
            permutation_importance_repeats=3
        )
    }
    
    results = {}
    for method_name, method_func in methods.items():
        tprint(f"\n🔍 Testing {method_name}...")
        try:
            result = method_func()
            if 'error' not in result:
                results[method_name] = result
                tprint(f"✅ {method_name}: {len(result['selected_features'])} features selected")
            else:
                tprint(f"❌ {method_name}: {result['error']}")
        except Exception as e:
            tprint(f"❌ {method_name}: {e}")
    
    # Compare results
    if results:
        tprint(f"\n📊 METHOD COMPARISON:")
        tprint(f"{'Method':<20} {'Features':<10} {'Overlap with informative':<25} {'CV Score':<15}")
        tprint("-" * 80)
        
        for method_name, result in results.items():
            selected = set(result['selected_features'])
            informative = set([f"feature_{i:02d}" for i in range(10)])
            overlap = len(selected.intersection(informative))
            
            # Get CV score if available
            cv_score = "N/A"
            if 'cv_validation' in result and 'error' not in result['cv_validation']:
                cv_score = f"{result['cv_validation']['cv_mean']:.3f}"
            elif 'selection_metadata' in result and 'baseline_score' in result['selection_metadata']:
                cv_score = f"{result['selection_metadata']['baseline_score']:.3f}"
            
            tprint(f"{method_name:<20} {len(selected):<10} {overlap}/10 ({overlap/10*100:.1f}%){'':<10} {cv_score:<15}")


def test_permutation_importance_analysis():
    """Test the permutation importance analysis in detail."""
    tprint("\n" + "="*60)
    tprint("🧪 TESTING PERMUTATION IMPORTANCE ANALYSIS")
    tprint("="*60)
    
    # Create sample data with known feature importance
    X, y, feature_names = create_sample_data(n_features=20, n_informative=5)
    
    # Initialize framework
    framework = FeatureSelectionFramework({
        'enable_gpu': False,
        'enable_parallel': True,
        'random_state': 42
    })
    
    # Test with detailed permutation importance
    tprint("\n🔍 Testing detailed permutation importance analysis...")
    result = framework.tree_based_ensemble_selection(
        X, y, feature_names,
        methods=['correlation', 'mrmr'],
        n_features=10,
        cv_folds=3,
        n_estimators=50,
        permutation_importance_repeats=20  # More repeats for better analysis
    )
    
    if 'error' not in result:
        tprint(f"✅ Permutation importance analysis successful!")
        
        # Analyze permutation importance distribution
        importance_data = result['permutation_importance']
        importances = [data['importance'] for data in importance_data.values()]
        stds = [data['std_importance'] for data in importance_data.values()]
        
        tprint(f"\n📊 Permutation Importance Statistics:")
        tprint(f"  - Mean importance: {np.mean(importances):.4f}")
        tprint(f"  - Std importance: {np.std(importances):.4f}")
        tprint(f"  - Max importance: {np.max(importances):.4f}")
        tprint(f"  - Min importance: {np.min(importances):.4f}")
        tprint(f"  - Mean std: {np.mean(stds):.4f}")
        
        # Show detailed importance for each feature
        tprint(f"\n📊 Detailed Permutation Importance:")
        sorted_importance = sorted(
            importance_data.items(),
            key=lambda x: x[1]['importance'],
            reverse=True
        )
        for feature, data in sorted_importance:
            importance = data['importance']
            std_importance = data['std_importance']
            scores = data['scores']
            tprint(f"  - {feature}: {importance:.4f} ± {std_importance:.4f} (scores: {[f'{s:.3f}' for s in scores[:5]]}...)")
        
        # Check if the most important features are the informative ones
        top_features = [feature for feature, _ in sorted_importance[:5]]
        informative_features = [f"feature_{i:02d}" for i in range(5)]
        overlap = len(set(top_features).intersection(set(informative_features)))
        tprint(f"\n📊 Top 5 features overlap with informative features: {overlap}/5")
        
    else:
        tprint(f"❌ Permutation importance analysis failed: {result['error']}")


def main():
    """Run all tests."""
    tprint("🚀 TREE-BASED ENSEMBLE FEATURE SELECTION TESTING")
    tprint("="*60)
    
    try:
        # Test individual methods
        test_tree_ensemble_selection()
        test_ensemble_vs_individual_methods()
        test_permutation_importance_analysis()
        
        tprint("\n" + "="*60)
        tprint("✅ ALL TESTS COMPLETED SUCCESSFULLY!")
        tprint("="*60)
        
    except Exception as e:
        tprint(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()