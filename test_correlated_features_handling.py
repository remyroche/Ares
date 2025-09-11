#!/usr/bin/env python3
"""
Test script for handling correlated features in permutation importance.

This script demonstrates:
1. The problem with correlated features in permutation importance
2. The solution using grouped permutation importance
3. Comparison between RandomForest and LightGBM
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from utils.ml_common.feature_selection import FeatureSelectionFramework


def create_correlated_data(n_samples=1000, n_features=20, n_informative=5, correlation_strength=0.9):
    """Create sample data with highly correlated features."""
    print("🔧 Creating correlated sample data...")
    
    # Generate random features
    np.random.seed(42)
    X = np.random.randn(n_samples, n_features)
    
    # Create highly correlated feature pairs
    for i in range(0, n_informative, 2):
        if i + 1 < n_informative:
            # Make features i and i+1 highly correlated
            X[:, i+1] = correlation_strength * X[:, i] + (1 - correlation_strength) * np.random.randn(n_samples)
    
    # Create target with linear relationship to informative features
    coefficients = np.random.randn(n_informative) * 2
    y = np.dot(X[:, :n_informative], coefficients) + np.random.randn(n_samples) * 0.1
    
    # Create feature names
    feature_names = [f"feature_{i:02d}" for i in range(n_features)]
    
    print(f"📊 Data created: {n_samples} samples, {n_features} features")
    print(f"📊 Informative features: {n_informative} (features 0-{n_informative-1})")
    print(f"📊 Correlation strength: {correlation_strength}")
    
    # Show correlation matrix for informative features
    corr_matrix = np.corrcoef(X[:, :n_informative].T)
    print(f"📊 Correlation matrix for informative features:")
    for i in range(n_informative):
        for j in range(n_informative):
            if i != j:
                print(f"  - {feature_names[i]} vs {feature_names[j]}: {corr_matrix[i,j]:.3f}")
    
    return X, y, feature_names


def test_correlated_features_problem():
    """Test the problem with correlated features in permutation importance."""
    print("\n" + "="*60)
    print("🧪 TESTING CORRELATED FEATURES PROBLEM")
    print("="*60)
    
    # Create correlated data
    X, y, feature_names = create_correlated_data(correlation_strength=0.95)
    
    # Initialize framework
    framework = FeatureSelectionFramework({
        'enable_gpu': False,
        'enable_parallel': True,
        'random_state': 42
    })
    
    # Test with correlation grouping disabled (simulate old behavior)
    print("\n🔍 Testing WITHOUT correlation grouping...")
    
    # Manually test permutation importance without grouping
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import train_test_split
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    baseline_score = model.score(X_test, y_test)
    
    # Test individual permutation importance
    individual_importance = {}
    for i, feature in enumerate(feature_names[:10]):  # Test first 10 features
        X_permuted = X_test.copy()
        np.random.shuffle(X_permuted[:, i])
        permuted_score = model.score(X_permuted, y_test)
        importance = baseline_score - permuted_score
        individual_importance[feature] = importance
    
    print(f"📊 Individual permutation importance (first 10 features):")
    for feature, importance in individual_importance.items():
        print(f"  - {feature}: {importance:.4f}")
    
    # Test grouped permutation importance
    print(f"\n🔍 Testing WITH correlation grouping...")
    
    # Group highly correlated features
    correlation_matrix = np.corrcoef(X_train.T)
    feature_groups = framework._group_correlated_features(
        feature_names[:10], correlation_matrix[:10, :10], threshold=0.8
    )
    
    print(f"📊 Feature groups identified:")
    for i, group in enumerate(feature_groups):
        if len(group) > 1:
            print(f"  - Group {i}: {group} (correlated)")
        else:
            print(f"  - Group {i}: {group[0]} (individual)")
    
    # Test grouped permutation importance
    grouped_importance = {}
    for group in feature_groups:
        X_permuted = X_test.copy()
        for feature in group:
            feature_idx = feature_names.index(feature)
            np.random.shuffle(X_permuted[:, feature_idx])
        
        permuted_score = model.score(X_permuted, y_test)
        importance = baseline_score - permuted_score
        
        for feature in group:
            grouped_importance[feature] = importance
    
    print(f"📊 Grouped permutation importance:")
    for feature, importance in grouped_importance.items():
        print(f"  - {feature}: {importance:.4f}")
    
    # Compare results
    print(f"\n📊 COMPARISON:")
    print(f"{'Feature':<15} {'Individual':<12} {'Grouped':<12} {'Difference':<12}")
    print("-" * 60)
    for feature in feature_names[:10]:
        individual = individual_importance[feature]
        grouped = grouped_importance[feature]
        difference = grouped - individual
        print(f"{feature:<15} {individual:<12.4f} {grouped:<12.4f} {difference:<12.4f}")


def test_tree_ensemble_with_correlation_grouping():
    """Test the tree-based ensemble with correlation grouping."""
    print("\n" + "="*60)
    print("🧪 TESTING TREE ENSEMBLE WITH CORRELATION GROUPING")
    print("="*60)
    
    # Create correlated data
    X, y, feature_names = create_correlated_data(correlation_strength=0.9)
    
    # Initialize framework
    framework = FeatureSelectionFramework({
        'enable_gpu': False,
        'enable_parallel': True,
        'random_state': 42
    })
    
    # Test tree-based ensemble with correlation grouping
    print("\n🔍 Testing tree-based ensemble with correlation grouping...")
    result = framework.tree_based_ensemble_selection(
        X, y, feature_names,
        methods=['correlation', 'mrmr'],
        n_features=10,
        cv_folds=3,
        n_estimators=50,
        permutation_importance_repeats=5
    )
    
    if 'error' not in result:
        print(f"✅ Tree-based ensemble with correlation grouping successful!")
        print(f"📊 Candidate features: {len(result['candidate_features'])}")
        print(f"📊 Selected features: {len(result['selected_features'])}")
        
        # Show permutation importance with grouping information
        print(f"\n📊 Permutation Importance with Grouping Info:")
        sorted_importance = sorted(
            result['permutation_importance'].items(),
            key=lambda x: x[1]['importance'],
            reverse=True
        )
        for feature, data in sorted_importance[:10]:
            importance = data['importance']
            std_importance = data['std_importance']
            group_size = data['group_size']
            is_correlated = data['is_correlated_group']
            group_info = f" (group size: {group_size})" if is_correlated else " (individual)"
            print(f"  - {feature}: {importance:.4f} ± {std_importance:.4f}{group_info}")
        
        # Show feature groups
        print(f"\n📊 Feature Groups:")
        groups_shown = set()
        for feature, data in result['permutation_importance'].items():
            group = data['group']
            group_key = tuple(sorted(group))
            if group_key not in groups_shown and len(group) > 1:
                print(f"  - Correlated group: {group}")
                groups_shown.add(group_key)
        
    else:
        print(f"❌ Tree-based ensemble failed: {result['error']}")


def test_randomforest_vs_lightgbm():
    """Compare RandomForest vs LightGBM for feature selection."""
    print("\n" + "="*60)
    print("🧪 COMPARING RANDOMFOREST VS LIGHTGBM")
    print("="*60)
    
    # Create sample data
    X, y, feature_names = create_correlated_data()
    
    # Test with RandomForest
    print("\n🔍 Testing with RandomForest...")
    framework_rf = FeatureSelectionFramework({
        'enable_gpu': False,
        'enable_parallel': True,
        'random_state': 42,
        'method_configs': {
            'tree_ensemble': {
                'use_lgbm': False,
                'n_estimators': 50,
                'max_depth': 8,
                'cv_folds': 3,
                'permutation_importance_repeats': 5
            }
        }
    })
    
    result_rf = framework_rf.tree_based_ensemble_selection(
        X, y, feature_names,
        methods=['correlation', 'mrmr'],
        n_features=10
    )
    
    # Test with LightGBM (if available)
    print("\n🔍 Testing with LightGBM...")
    framework_lgb = FeatureSelectionFramework({
        'enable_gpu': False,
        'enable_parallel': True,
        'random_state': 42,
        'method_configs': {
            'tree_ensemble': {
                'use_lgbm': True,
                'n_estimators': 50,
                'max_depth': 8,
                'cv_folds': 3,
                'permutation_importance_repeats': 5
            }
        }
    })
    
    result_lgb = framework_lgb.tree_based_ensemble_selection(
        X, y, feature_names,
        methods=['correlation', 'mrmr'],
        n_features=10
    )
    
    # Compare results
    if 'error' not in result_rf and 'error' not in result_lgb:
        print(f"\n📊 COMPARISON RESULTS:")
        print(f"{'Metric':<25} {'RandomForest':<15} {'LightGBM':<15}")
        print("-" * 60)
        
        # Compare baseline scores
        rf_score = result_rf['selection_metadata']['baseline_score']
        lgb_score = result_lgb['selection_metadata']['baseline_score']
        print(f"{'Baseline Score':<25} {rf_score:<15.3f} {lgb_score:<15.3f}")
        
        # Compare CV scores
        if 'cv_validation' in result_rf and 'error' not in result_rf['cv_validation']:
            rf_cv = result_rf['cv_validation']['cv_mean']
            lgb_cv = result_lgb['cv_validation']['cv_mean']
            print(f"{'CV Score':<25} {rf_cv:<15.3f} {lgb_cv:<15.3f}")
        
        # Compare selected features
        rf_features = set(result_rf['selected_features'])
        lgb_features = set(result_lgb['selected_features'])
        overlap = len(rf_features.intersection(lgb_features))
        print(f"{'Selected Features':<25} {len(rf_features):<15} {len(lgb_features):<15}")
        print(f"{'Feature Overlap':<25} {overlap:<15} {overlap:<15}")
        
        # Compare permutation importance
        print(f"\n📊 Top 5 Features by Importance:")
        print(f"{'Rank':<5} {'RandomForest':<15} {'LightGBM':<15}")
        print("-" * 40)
        
        rf_sorted = sorted(result_rf['permutation_importance'].items(), 
                          key=lambda x: x[1]['importance'], reverse=True)
        lgb_sorted = sorted(result_lgb['permutation_importance'].items(), 
                           key=lambda x: x[1]['importance'], reverse=True)
        
        for i in range(min(5, len(rf_sorted), len(lgb_sorted))):
            rf_feature = rf_sorted[i][0]
            lgb_feature = lgb_sorted[i][0]
            print(f"{i+1:<5} {rf_feature:<15} {lgb_feature:<15}")
    
    else:
        print(f"❌ Comparison failed:")
        if 'error' in result_rf:
            print(f"  - RandomForest error: {result_rf['error']}")
        if 'error' in result_lgb:
            print(f"  - LightGBM error: {result_lgb['error']}")


def main():
    """Run all tests."""
    print("🚀 CORRELATED FEATURES HANDLING TESTING")
    print("="*60)
    
    try:
        # Test the correlated features problem
        test_correlated_features_problem()
        
        # Test tree ensemble with correlation grouping
        test_tree_ensemble_with_correlation_grouping()
        
        # Test RandomForest vs LightGBM
        test_randomforest_vs_lightgbm()
        
        print("\n" + "="*60)
        print("✅ ALL TESTS COMPLETED SUCCESSFULLY!")
        print("="*60)
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()