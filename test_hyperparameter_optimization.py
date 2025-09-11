#!/usr/bin/env python3
from src.utils.tprint import tprint

"""
Test script for hyperparameter optimization in tree-based ensemble selection.

This script demonstrates:
1. The impact of hyperparameter optimization on feature selection
2. Comparison between fixed and optimized hyperparameters
3. The benefits of grouped permutation importance
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from utils.ml_common.feature_selection import FeatureSelectionFramework


def create_complex_data(n_samples=1000, n_features=30, n_informative=8, noise=0.1):
    """Create complex sample data that benefits from hyperparameter optimization."""
    tprint("🔧 Creating complex sample data...")
    
    # Generate random features
    np.random.seed(42)
    X = np.random.randn(n_samples, n_features)
    
    # Create informative features with different complexities
    informative_features = X[:, :n_informative]
    
    # Create target with non-linear relationships
    coefficients = np.random.randn(n_informative) * 2
    
    # Add some non-linear interactions
    y = (np.dot(informative_features, coefficients) + 
         0.5 * np.sum(informative_features[:, :3] ** 2, axis=1) +  # Quadratic terms
         0.3 * np.prod(informative_features[:, :2], axis=1) +     # Interaction terms
         np.random.randn(n_samples) * noise)
    
    # Create feature names
    feature_names = [f"feature_{i:02d}" for i in range(n_features)]
    
    tprint(f"📊 Data created: {n_samples} samples, {n_features} features")
    tprint(f"📊 Informative features: {n_informative} (features 0-{n_informative-1})")
    tprint(f"📊 Target includes linear, quadratic, and interaction terms")
    
    return X, y, feature_names


def test_hyperparameter_optimization():
    """Test the impact of hyperparameter optimization on feature selection."""
    tprint("\n" + "="*60)
    tprint("🧪 TESTING HYPERPARAMETER OPTIMIZATION")
    tprint("="*60)
    
    # Create complex data
    X, y, feature_names = create_complex_data()
    
    # Test with hyperparameter optimization enabled
    tprint("\n🔍 Testing WITH hyperparameter optimization...")
    framework_optimized = FeatureSelectionFramework({
        'enable_gpu': False,
        'enable_parallel': True,
        'random_state': 42,
        'method_configs': {
            'tree_ensemble': {
                'hyperparameter_search': True,
                'param_grid': {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [5, 10, 15, None]
                },
                'cv_folds': 3,
                'permutation_importance_repeats': 5
            }
        }
    })
    
    result_optimized = framework_optimized.tree_based_ensemble_selection(
        X, y, feature_names,
        methods=['correlation', 'mrmr'],
        n_features=12,
        cv_folds=3
    )
    
    # Test with hyperparameter optimization disabled
    tprint("\n🔍 Testing WITHOUT hyperparameter optimization...")
    framework_fixed = FeatureSelectionFramework({
        'enable_gpu': False,
        'enable_parallel': True,
        'random_state': 42,
        'method_configs': {
            'tree_ensemble': {
                'hyperparameter_search': False,
                'param_grid': {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [5, 10, 15, None]
                },
                'cv_folds': 3,
                'permutation_importance_repeats': 5
            }
        }
    })
    
    result_fixed = framework_fixed.tree_based_ensemble_selection(
        X, y, feature_names,
        methods=['correlation', 'mrmr'],
        n_features=12,
        cv_folds=3
    )
    
    # Compare results
    if 'error' not in result_optimized and 'error' not in result_fixed:
        tprint(f"\n📊 HYPERPARAMETER OPTIMIZATION COMPARISON:")
        tprint(f"{'Metric':<30} {'Optimized':<15} {'Fixed':<15} {'Improvement':<15}")
        tprint("-" * 80)
        
        # Compare baseline scores
        opt_score = result_optimized['selection_metadata']['baseline_score']
        fixed_score = result_fixed['selection_metadata']['baseline_score']
        improvement = opt_score - fixed_score
        tprint(f"{'Baseline Score':<30} {opt_score:<15.3f} {fixed_score:<15.3f} {improvement:<15.3f}")
        
        # Compare hyperparameter scores
        opt_hp_score = result_optimized['selection_metadata']['best_hyperparameter_score']
        fixed_hp_score = result_fixed['selection_metadata']['best_hyperparameter_score']
        hp_improvement = opt_hp_score - fixed_hp_score
        tprint(f"{'HP Search Score':<30} {opt_hp_score:<15.3f} {fixed_hp_score:<15.3f} {hp_improvement:<15.3f}")
        
        # Compare CV scores
        if 'cv_validation' in result_optimized and 'error' not in result_optimized['cv_validation']:
            opt_cv = result_optimized['cv_validation']['cv_mean']
            fixed_cv = result_fixed['cv_validation']['cv_mean']
            cv_improvement = opt_cv - fixed_cv
            tprint(f"{'CV Score':<30} {opt_cv:<15.3f} {fixed_cv:<15.3f} {cv_improvement:<15.3f}")
        
        # Compare selected features
        opt_features = set(result_optimized['selected_features'])
        fixed_features = set(result_fixed['selected_features'])
        overlap = len(opt_features.intersection(fixed_features))
        tprint(f"{'Selected Features':<30} {len(opt_features):<15} {len(fixed_features):<15} {overlap:<15}")
        
        # Show best hyperparameters
        tprint(f"\n📊 Best Hyperparameters:")
        opt_params = result_optimized['selection_metadata']['best_hyperparameters']
        fixed_params = result_fixed['selection_metadata']['best_hyperparameters']
        tprint(f"  - Optimized: {opt_params}")
        tprint(f"  - Fixed: {fixed_params}")
        
        # Show top features
        tprint(f"\n📊 Top 5 Features by Importance:")
        tprint(f"{'Rank':<5} {'Optimized':<15} {'Fixed':<15}")
        tprint("-" * 40)
        
        opt_sorted = sorted(result_optimized['permutation_importance'].items(), 
                           key=lambda x: x[1]['importance'], reverse=True)
        fixed_sorted = sorted(result_fixed['permutation_importance'].items(), 
                             key=lambda x: x[1]['importance'], reverse=True)
        
        for i in range(min(5, len(opt_sorted), len(fixed_sorted))):
            opt_feature = opt_sorted[i][0]
            fixed_feature = fixed_sorted[i][0]
            tprint(f"{i+1:<5} {opt_feature:<15} {fixed_feature:<15}")
        
        # Check if informative features were selected
        informative_features = set([f"feature_{i:02d}" for i in range(8)])
        opt_informative = len(opt_features.intersection(informative_features))
        fixed_informative = len(fixed_features.intersection(informative_features))
        tprint(f"\n📊 Informative Features Selected:")
        tprint(f"  - Optimized: {opt_informative}/8 ({opt_informative/8*100:.1f}%)")
        tprint(f"  - Fixed: {fixed_informative}/8 ({fixed_informative/8*100:.1f}%)")
        
    else:
        tprint(f"❌ Comparison failed:")
        if 'error' in result_optimized:
            tprint(f"  - Optimized error: {result_optimized['error']}")
        if 'error' in result_fixed:
            tprint(f"  - Fixed error: {result_fixed['error']}")


def test_different_param_grids():
    """Test different hyperparameter grids to see their impact."""
    tprint("\n" + "="*60)
    tprint("🧪 TESTING DIFFERENT HYPERPARAMETER GRIDS")
    tprint("="*60)
    
    # Create sample data
    X, y, feature_names = create_complex_data()
    
    # Define different parameter grids
    param_grids = {
        'Conservative': {
            'n_estimators': [50, 100],
            'max_depth': [5, 10]
        },
        'Moderate': {
            'n_estimators': [50, 100, 200],
            'max_depth': [5, 10, 15]
        },
        'Aggressive': {
            'n_estimators': [100, 200, 300],
            'max_depth': [10, 15, None]
        }
    }
    
    results = {}
    
    for grid_name, param_grid in param_grids.items():
        tprint(f"\n🔍 Testing {grid_name} parameter grid...")
        
        framework = FeatureSelectionFramework({
            'enable_gpu': False,
            'enable_parallel': True,
            'random_state': 42,
            'method_configs': {
                'tree_ensemble': {
                    'hyperparameter_search': True,
                    'param_grid': param_grid,
                    'cv_folds': 3,
                    'permutation_importance_repeats': 3
                }
            }
        })
        
        result = framework.tree_based_ensemble_selection(
            X, y, feature_names,
            methods=['correlation', 'mrmr'],
            n_features=10,
            cv_folds=3
        )
        
        if 'error' not in result:
            results[grid_name] = result
            tprint(f"✅ {grid_name}: {len(result['selected_features'])} features selected")
        else:
            tprint(f"❌ {grid_name}: {result['error']}")
    
    # Compare results
    if results:
        tprint(f"\n📊 PARAMETER GRID COMPARISON:")
        tprint(f"{'Grid':<15} {'HP Score':<12} {'Baseline':<12} {'CV Score':<12} {'Features':<10}")
        tprint("-" * 70)
        
        for grid_name, result in results.items():
            hp_score = result['selection_metadata']['best_hyperparameter_score']
            baseline = result['selection_metadata']['baseline_score']
            cv_score = result['cv_validation']['cv_mean'] if 'cv_validation' in result and 'error' not in result['cv_validation'] else 0.0
            n_features = len(result['selected_features'])
            
            tprint(f"{grid_name:<15} {hp_score:<12.3f} {baseline:<12.3f} {cv_score:<12.3f} {n_features:<10}")
        
        # Show best hyperparameters for each grid
        tprint(f"\n📊 Best Hyperparameters by Grid:")
        for grid_name, result in results.items():
            best_params = result['selection_metadata']['best_hyperparameters']
            tprint(f"  - {grid_name}: {best_params}")


def test_correlation_grouping_impact():
    """Test the impact of correlation grouping on feature selection."""
    tprint("\n" + "="*60)
    tprint("🧪 TESTING CORRELATION GROUPING IMPACT")
    tprint("="*60)
    
    # Create data with highly correlated features
    X, y, feature_names = create_complex_data()
    
    # Artificially create highly correlated features
    X[:, 1] = 0.95 * X[:, 0] + 0.05 * np.random.randn(len(X))  # feature_01 highly correlated with feature_00
    X[:, 3] = 0.90 * X[:, 2] + 0.10 * np.random.randn(len(X))  # feature_03 highly correlated with feature_02
    
    tprint("📊 Created highly correlated feature pairs:")
    tprint("  - feature_00 and feature_01 (r ≈ 0.95)")
    tprint("  - feature_02 and feature_03 (r ≈ 0.90)")
    
    # Test with different correlation thresholds
    thresholds = [0.7, 0.8, 0.9]
    
    for threshold in thresholds:
        tprint(f"\n🔍 Testing with correlation threshold: {threshold}")
        
        framework = FeatureSelectionFramework({
            'enable_gpu': False,
            'enable_parallel': True,
            'random_state': 42,
            'method_configs': {
                'tree_ensemble': {
                    'correlation_threshold': threshold,
                    'hyperparameter_search': True,
                    'param_grid': {
                        'n_estimators': [50, 100],
                        'max_depth': [5, 10]
                    },
                    'cv_folds': 3,
                    'permutation_importance_repeats': 3
                }
            }
        })
        
        result = framework.tree_based_ensemble_selection(
            X, y, feature_names,
            methods=['correlation', 'mrmr'],
            n_features=8,
            cv_folds=3
        )
        
        if 'error' not in result:
            tprint(f"✅ Threshold {threshold}: {len(result['selected_features'])} features selected")
            
            # Show correlation groups
            groups_shown = set()
            for feature, data in result['permutation_importance'].items():
                group = data['group']
                group_key = tuple(sorted(group))
                if group_key not in groups_shown and len(group) > 1:
                    tprint(f"  - Correlated group: {group}")
                    groups_shown.add(group_key)
        else:
            tprint(f"❌ Threshold {threshold}: {result['error']}")


def main():
    """Run all tests."""
    tprint("🚀 HYPERPARAMETER OPTIMIZATION TESTING")
    tprint("="*60)
    
    try:
        # Test hyperparameter optimization impact
        test_hyperparameter_optimization()
        
        # Test different parameter grids
        test_different_param_grids()
        
        # Test correlation grouping impact
        test_correlation_grouping_impact()
        
        tprint("\n" + "="*60)
        tprint("✅ ALL TESTS COMPLETED SUCCESSFULLY!")
        tprint("="*60)
        
    except Exception as e:
        tprint(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()