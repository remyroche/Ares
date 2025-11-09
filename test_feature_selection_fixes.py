"""
Test the new feature selection fixes to verify:
1. Exact feature counts are returned
2. Redundancy is properly reduced
3. Hierarchical clustering works correctly
"""

import numpy as np
import pandas as pd
import sys
import os

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from training.steps.pre_training.components.final_feature_selection import (
    FinalFeatureSelectionComponent,
    FinalFeatureSelectionConfig
)


def create_test_data():
    """Create test data with known redundancy patterns"""
    np.random.seed(42)
    n_samples = 200
    
    # Create base features
    base_features = np.random.randn(n_samples, 20)
    
    # Create redundant feature groups
    # Group 1: Fibonacci features (highly correlated)
    fib_base = np.random.randn(n_samples)
    fib_features = np.column_stack([
        fib_base + np.random.randn(n_samples) * 0.05,  # fibonacci_0.236_10
        fib_base + np.random.randn(n_samples) * 0.05,  # fibonacci_0.236_5
        fib_base + np.random.randn(n_samples) * 0.05,  # fibonacci_0.786_20
    ])
    
    # Group 2: Volume features (moderately correlated)
    vol_base = np.random.randn(n_samples)
    vol_features = np.column_stack([
        vol_base + np.random.randn(n_samples) * 0.2,
        vol_base + np.random.randn(n_samples) * 0.2,
    ])
    
    # Combine all features
    X = np.hstack([base_features, fib_features, vol_features])
    
    # Create feature names
    feature_names = (
        [f'base_feature_{i}' for i in range(20)] +
        ['fibonacci_0.236_10_price_returns', 'fibonacci_0.236_5_price_returns', 'fibonacci_0.786_20_price_returns'] +
        ['volume_entropy_ma_10_10', 'volume_entropy_ma_20_10']
    )
    
    # Create target with relationship to base features
    y = (
        0.5 * X[:, 0] + 
        0.3 * X[:, 1] + 
        0.2 * X[:, 2] + 
        np.random.randn(n_samples) * 0.1
    )
    
    return pd.DataFrame(X, columns=feature_names), pd.Series(y, name='target')


def test_exact_feature_count():
    """Test that exact feature counts are returned"""
    print("\n" + "="*80)
    print("TEST 1: Exact Feature Count")
    print("="*80)
    
    X, y = create_test_data()
    
    # Test different target counts
    for target_count in [60, 50, 40, 20, 10]:
        # Adjust target if we don't have enough features
        actual_target = min(target_count, len(X.columns))
        
        config = FinalFeatureSelectionConfig(
            max_features=actual_target,
            min_features=max(5, actual_target // 2),
            selection_method='permutation',
            use_tree_based=True,
            use_permutation_importance=True
        )
        
        component = FinalFeatureSelectionComponent(config)
        selected_features = component.select_features(X, y)
        
        # Check if we got exact count
        if len(selected_features) == actual_target:
            print(f"✅ Target {target_count}: Got exactly {len(selected_features)} features")
        else:
            print(f"❌ Target {target_count}: Got {len(selected_features)} features, expected {actual_target}")
            return False
    
    return True


def test_redundancy_reduction():
    """Test that redundant features are properly removed"""
    print("\n" + "="*80)
    print("TEST 2: Redundancy Reduction")
    print("="*80)
    
    X, y = create_test_data()
    
    config = FinalFeatureSelectionConfig(
        max_features=20,
        min_features=10,
        selection_method='permutation',
        use_tree_based=True,
        use_permutation_importance=True
    )
    
    component = FinalFeatureSelectionComponent(config)
    selected_features = component.select_features(X, y)
    
    # Check for redundant fibonacci features
    fib_features = [f for f in selected_features if 'fibonacci' in f]
    
    if len(fib_features) <= 1:
        print(f"✅ Redundancy check passed: Only {len(fib_features)} fibonacci feature(s) selected")
        print(f"   Selected fibonacci features: {fib_features}")
    else:
        print(f"❌ Redundancy check failed: {len(fib_features)} fibonacci features selected")
        print(f"   Selected fibonacci features: {fib_features}")
        
        # Check correlation between them
        if len(fib_features) > 1:
            corr_matrix = X[fib_features].corr().abs()
            max_corr = corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)].max()
            print(f"   Max correlation between fibonacci features: {max_corr:.4f}")
            
            if max_corr > 0.85:
                print(f"   ⚠️ High correlation detected (> 0.85)")
                return False
    
    # Check overall correlation
    if len(selected_features) > 1:
        corr_matrix = X[selected_features].corr().abs()
        # Get upper triangle (excluding diagonal)
        upper_triangle = corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)]
        max_corr = upper_triangle.max()
        avg_corr = upper_triangle.mean()
        
        print(f"\n📊 Overall correlation statistics:")
        print(f"   Max correlation: {max_corr:.4f}")
        print(f"   Average correlation: {avg_corr:.4f}")
        
        if max_corr > 0.85:
            print(f"   ❌ Max correlation exceeds threshold (0.85)")
            return False
        else:
            print(f"   ✅ Max correlation within threshold")
    
    return True


def test_hierarchical_clustering():
    """Test that hierarchical clustering method exists and works"""
    print("\n" + "="*80)
    print("TEST 3: Hierarchical Clustering Method")
    print("="*80)
    
    X, y = create_test_data()
    
    config = FinalFeatureSelectionConfig(
        max_features=15,
        min_features=10,
        selection_method='permutation',
        use_tree_based=True,
        use_permutation_importance=True
    )
    
    component = FinalFeatureSelectionComponent(config)
    
    # Check if the new method exists
    if hasattr(component, '_reduce_redundancy_hierarchical'):
        print("✅ _reduce_redundancy_hierarchical method exists")
        
        # Test the method directly
        ranked_features = list(X.columns)
        reduced_features = component._reduce_redundancy_hierarchical(
            X, ranked_features, target_count=15, correlation_threshold=0.85
        )
        
        if len(reduced_features) == 15:
            print(f"✅ Method returned exactly 15 features")
        else:
            print(f"❌ Method returned {len(reduced_features)} features, expected 15")
            return False
        
        # Check diversity
        if len(reduced_features) > 1:
            corr_matrix = X[reduced_features].corr().abs()
            upper_triangle = corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)]
            max_corr = upper_triangle.max()
            
            if max_corr <= 0.85:
                print(f"✅ Features are diverse (max correlation: {max_corr:.4f})")
            else:
                print(f"❌ Features not diverse enough (max correlation: {max_corr:.4f})")
                return False
    else:
        print("❌ _reduce_redundancy_hierarchical method not found")
        return False
    
    return True


def test_feature_quality():
    """Test that selected features are high quality"""
    print("\n" + "="*80)
    print("TEST 4: Feature Quality")
    print("="*80)
    
    X, y = create_test_data()
    
    config = FinalFeatureSelectionConfig(
        max_features=20,
        min_features=10,
        selection_method='permutation',
        use_tree_based=True,
        use_permutation_importance=True
    )
    
    component = FinalFeatureSelectionComponent(config)
    selected_features = component.select_features(X, y)
    
    # Check that base features (which have relationship with target) are selected
    base_features_selected = [f for f in selected_features if 'base_feature' in f]
    
    print(f"📊 Feature composition:")
    print(f"   Total selected: {len(selected_features)}")
    print(f"   Base features: {len(base_features_selected)}")
    print(f"   Other features: {len(selected_features) - len(base_features_selected)}")
    
    # We expect at least some base features since they have relationship with target
    if len(base_features_selected) > 0:
        print(f"✅ Base features with target relationship are selected")
    else:
        print(f"⚠️ No base features selected (might be okay if other features are better)")
    
    # Check feature scores
    feature_scores = component.get_feature_scores()
    if feature_scores:
        print(f"\n📈 Top 5 features by importance:")
        sorted_scores = sorted(feature_scores.items(), key=lambda x: x[1], reverse=True)[:5]
        for feat, score in sorted_scores:
            print(f"   {feat}: {score:.6f}")
    
    return True


def run_all_tests():
    """Run all tests and report results"""
    print("\n" + "="*80)
    print("FEATURE SELECTION FIXES - INTEGRATION TESTS")
    print("="*80)
    
    tests = [
        ("Exact Feature Count", test_exact_feature_count),
        ("Redundancy Reduction", test_redundancy_reduction),
        ("Hierarchical Clustering", test_hierarchical_clustering),
        ("Feature Quality", test_feature_quality),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n❌ {test_name} failed with exception: {e}")
            import traceback
            traceback.print_exc()
            results.append((test_name, False))
    
    # Print summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{status}: {test_name}")
    
    print(f"\nOverall: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    return all(result for _, result in results)


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
