"""
Simple test for FinalFeatureSelectionComponent key functionality

This test focuses on verifying:
1. Duplicate feature detection
2. Feature diversity constraints  
3. Permutation importance logging
4. Overall workflow integration
"""

import numpy as np
import pandas as pd
import sys
import os

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_duplicate_detection():
    """Test duplicate feature detection"""
    print("Testing duplicate feature detection...")
    
    from training.steps.pre_training.components.final_feature_selection import FinalFeatureSelectionConfig, FinalFeatureSelectionComponent
    
    # Create component
    config = FinalFeatureSelectionConfig()
    component = FinalFeatureSelectionComponent(config)
    
    # Create data with duplicates
    data = {
        'feature_1': [1, 2, 3, 4, 5],
        'feature_2': [2, 4, 6, 8, 10],
        'duplicate_1': [1, 2, 3, 4, 5],  # Exact duplicate of feature_1
        'feature_3': [3, 6, 9, 12, 15]
    }
    X = pd.DataFrame(data)
    
    # Test duplicate removal
    X_dedup = component._remove_exact_duplicates(X)
    
    # Verify duplicate was removed
    assert 'duplicate_1' not in X_dedup.columns, "Duplicate column should be removed"
    assert 'feature_1' in X_dedup.columns, "Original column should remain"
    assert len(X_dedup.columns) == 3, f"Expected 3 columns, got {len(X_dedup.columns)}"
    
    print("✅ Duplicate detection test passed")
    return True

def test_feature_diversity():
    """Test feature diversity constraints"""
    print("Testing feature diversity constraints...")
    
    from training.steps.pre_training.components.final_feature_selection import FinalFeatureSelectionConfig, FinalFeatureSelectionComponent
    
    # Create component
    config = FinalFeatureSelectionConfig()
    component = FinalFeatureSelectionComponent(config)
    
    # Create data with highly correlated features
    np.random.seed(42)
    n_samples = 100
    base_feature = np.random.randn(n_samples)
    
    data = {
        'feature_1': base_feature,
        'feature_2': base_feature * 0.95 + np.random.randn(n_samples) * 0.05,  # High correlation
        'feature_3': np.random.randn(n_samples),  # Low correlation
        'feature_4': base_feature * 0.9 + np.random.randn(n_samples) * 0.1,  # High correlation
    }
    X = pd.DataFrame(data)
    
    # Test diversity constraints
    initial_features = ['feature_1', 'feature_2', 'feature_3', 'feature_4']
    diverse_features = component._ensure_feature_diversity(initial_features, X, correlation_threshold=0.8)
    
    # Verify highly correlated features were removed
    assert len(diverse_features) < len(initial_features), "Highly correlated features should be removed"
    assert 'feature_3' in diverse_features, "Low correlation feature should remain"
    
    print("✅ Feature diversity test passed")
    return True

def test_permutation_importance_logging():
    """Test permutation importance logging"""
    print("Testing permutation importance logging...")
    
    from training.steps.pre_training.components.final_feature_selection import FinalFeatureSelectionConfig, FinalFeatureSelectionComponent
    
    # Create component with permutation importance enabled
    config = FinalFeatureSelectionConfig(use_permutation_importance=True)
    component = FinalFeatureSelectionComponent(config)
    
    # Create sample data
    np.random.seed(42)
    n_samples = 50
    X = pd.DataFrame(np.random.randn(n_samples, 10), 
                     columns=[f'feature_{i}' for i in range(10)])
    y = pd.Series(X['feature_0'] * 0.5 + X['feature_1'] * 0.3 + np.random.randn(n_samples) * 0.1)
    
    # Mock logger to capture calls
    import logging
    from unittest.mock import MagicMock
    
    mock_logger = MagicMock()
    component.logger = mock_logger
    
    try:
        # Test tree-based selection with permutation importance
        selected_features = component._apply_tree_based_selection(
            X, y, list(X.columns[:5])
        )
        
        # Verify permutation importance was used
        info_calls = [str(call) for call in mock_logger.info.call_args_list]
        debug_calls = [str(call) for call in mock_logger.debug.call_args_list]
        
        # Check for permutation importance logging
        permutation_logged = any("permutation importance" in call.lower() for call in info_calls)
        debug_logged = any("permutation importance" in call.lower() for call in debug_calls)
        
        assert permutation_logged or debug_logged, "Permutation importance should be logged"
        
        print("✅ Permutation importance logging test passed")
        return True
        
    except Exception as e:
        print(f"⚠️ Permutation importance test failed with error: {e}")
        print("This may be due to missing dependencies or insufficient data")
        return False

def test_workflow_integration():
    """Test overall workflow integration"""
    print("Testing overall workflow integration...")
    
    from training.steps.pre_training.components.final_feature_selection import FinalFeatureSelectionConfig, FinalFeatureSelectionComponent
    
    # Create component
    config = FinalFeatureSelectionConfig(
        max_features=5,
        min_features=2,
        selection_method="mutual_info",
        use_tree_based=True,
        use_permutation_importance=True
    )
    component = FinalFeatureSelectionComponent(config)
    
    # Create sample data
    np.random.seed(42)
    n_samples = 50
    X = pd.DataFrame(np.random.randn(n_samples, 15), 
                     columns=[f'feature_{i}' for i in range(15)])
    y = pd.Series(X['feature_0'] * 0.5 + X['feature_1'] * 0.3 + np.random.randn(n_samples) * 0.1)
    
    try:
        # Test complete feature selection workflow
        selected_features = component.select_features(X, y)
        
        # Verify results
        assert isinstance(selected_features, list), "Selected features should be a list"
        assert len(selected_features) > 0, "Should select at least one feature"
        assert len(selected_features) <= config.max_features, f"Should not exceed max_features ({config.max_features})"
        
        # Verify feature scores were calculated
        feature_scores = component.get_feature_scores()
        assert isinstance(feature_scores, dict), "Feature scores should be a dictionary"
        assert len(feature_scores) > 0, "Should have feature scores"
        
        print("✅ Workflow integration test passed")
        return True
        
    except Exception as e:
        print(f"⚠️ Workflow integration test failed with error: {e}")
        return False

def main():
    """Run all tests"""
    print("=" * 60)
    print("FINAL FEATURE SELECTION COMPONENT TESTS")
    print("=" * 60)
    
    tests = [
        test_duplicate_detection,
        test_feature_diversity,
        test_permutation_importance_logging,
        test_workflow_integration
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
            print()
        except Exception as e:
            print(f"❌ Test {test.__name__} failed with error: {e}")
            print()
    
    print("=" * 60)
    print(f"TEST RESULTS: {passed}/{total} tests passed")
    print("=" * 60)
    
    if passed == total:
        print("🎉 All tests passed! Final feature selection component is working correctly.")
        return True
    else:
        print(f"⚠️ {total - passed} test(s) failed. Please check the implementation.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)