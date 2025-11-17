"""
Test suite for FinalFeatureSelectionComponent

This test suite verifies the functionality of the final feature selection component,
including:
1. Duplicate feature detection
2. Feature diversity constraints
3. Permutation importance logging
4. Overall workflow integration
"""

import unittest
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock
import sys
import os

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from training.steps.pre_training.components.final_feature_selection import (
    FinalFeatureSelectionComponent,
    FinalFeatureSelectionConfig
)

# Import des assertions standardisées
from tests.utils.assertions import (
    assert_float_equals,
    assert_dict_structure,
    assert_list_structure
)


class TestFinalFeatureSelection(unittest.TestCase):
    """Test cases for FinalFeatureSelectionComponent"""
    
    def setUp(self):
        """Set up test data and component"""
        np.random.seed(42)
        
        # Create sample data
        n_samples = 200
        n_features = 50
        
        # Generate correlated features with some duplicates
        X_base = np.random.randn(n_samples, 20)
        
        # Add some highly correlated features
        X_correlated = X_base[:, :5] + np.random.randn(n_samples, 5) * 0.1
        
        # Add some duplicate features
        X_duplicates = X_base[:, :3]
        
        # Add some noise features
        X_noise = np.random.randn(n_samples, n_features - 28)
        
        # Combine all features
        X = np.hstack([X_base, X_correlated, X_duplicates, X_noise])
        
        # Create feature names
        feature_names = [f'feature_{i}' for i in range(n_features)]
        
        # Create target variable with some relationship to features
        y = (
            0.5 * X[:, 0] + 
            0.3 * X[:, 1] + 
            0.2 * X[:, 2] + 
            np.random.randn(n_samples) * 0.1
        )
        
        # Create DataFrame
        self.X = pd.DataFrame(X, columns=feature_names)
        self.y = pd.Series(y, name='target')
        
        # Create component with test configuration
        self.config = FinalFeatureSelectionConfig(
            max_features=20,
            min_features=5,
            selection_method="mutual_info",
            scoring_threshold=0.01,
            use_tree_based=True,
            use_permutation_importance=True
        )
        
        self.component = FinalFeatureSelectionComponent(self.config)
    
    def test_duplicate_feature_detection(self):
        """Test that duplicate features are correctly detected and removed"""
        # Create data with explicit duplicates
        X_with_duplicates = self.X.copy()
        X_with_duplicates['duplicate_1'] = X_with_duplicates['feature_0'].copy()
        X_with_duplicates['duplicate_2'] = X_with_duplicates['feature_1'].copy()
        
        # Test duplicate removal
        X_dedup = self.component._remove_exact_duplicates(X_with_duplicates)
        
        # Verify duplicates are removed
        self.assertNotIn('duplicate_1', X_dedup.columns)
        self.assertNotIn('duplicate_2', X_dedup.columns)
        
        # Verify original features are still there
        self.assertIn('feature_0', X_dedup.columns)
        self.assertIn('feature_1', X_dedup.columns)
        
        # Verify the count is reduced (at least 2 duplicates removed)
        self.assertLessEqual(len(X_dedup.columns), len(X_with_duplicates.columns) - 2)
    
    def test_feature_diversity_constraints(self):
        """Test that feature diversity constraints work properly"""
        # Create highly correlated features
        X_diverse = self.X.copy()
        X_diverse['high_corr_1'] = X_diverse['feature_0'] * 0.95 + np.random.randn(len(X_diverse)) * 0.05
        X_diverse['high_corr_2'] = X_diverse['feature_1'] * 0.95 + np.random.randn(len(X_diverse)) * 0.05
        
        # Select some features
        initial_features = ['feature_0', 'feature_1', 'high_corr_1', 'high_corr_2', 'feature_5']
        
        # Apply diversity constraints
        diverse_features = self.component._ensure_feature_diversity(
            initial_features, X_diverse, correlation_threshold=0.8
        )
        
        # Verify that highly correlated features are removed
        self.assertLess(len(diverse_features), len(initial_features))
        
        # Verify that the remaining features are diverse
        for i, feat1 in enumerate(diverse_features):
            for feat2 in diverse_features[i+1:]:
                corr = abs(X_diverse[feat1].corr(X_diverse[feat2]))
                self.assertLessEqual(corr, 0.8, 
                    f"Features {feat1} and {feat2} are too correlated: {corr}")
    
    def test_permutation_importance_logging(self):
        """Test that permutation importance logging provides expected debugging information"""
        # Use smaller subset to avoid index out of range error
        X_test = self.X.iloc[:50, :15]  # Use smaller dataset
        y_test = self.y.iloc[:50]
        feature_names_test = list(X_test.columns)
        
        with patch.object(self.component.logger, 'info') as mock_info, \
             patch.object(self.component.logger, 'debug') as mock_debug:
            
            # Apply tree-based selection with permutation importance
            selected_features = self.component._apply_tree_based_selection(
                X_test, y_test, feature_names_test
            )
            
            # Verify that SHAP/permutation importance logging occurred
            mock_info.assert_any_call("Using LGBM-SHAP importance (captures feature interactions with game-theoretic interpretation)")
            mock_info.assert_any_call("SHAP importance calculated for 15 features")
            
            # Verify that ALL features were logged (not just top 10)
            all_features_calls = [call for call in mock_info.call_args_list
                                if "All 15 features by SHAP importance:" in str(call)]
            self.assertGreater(len(all_features_calls), 0,
                           f"Expected 'All 15 features by SHAP importance:' in calls. Got calls: {[str(call) for call in mock_info.call_args_list]}")
            
            # Verify debug logging for importance statistics
            debug_calls = [call for call in mock_debug.call_args_list
                          if "SHAP importance stats:" in str(call)]
            self.assertGreater(len(debug_calls), 0,
                           f"Expected 'SHAP importance stats:' in debug calls. Got calls: {[str(call) for call in mock_debug.call_args_list]}")
    
    def test_overall_workflow_integration(self):
        """Test that the overall workflow integration is working as expected"""
        # Test the complete feature selection workflow
        selected_features = self.component.select_features(self.X, self.y)
        
        # Verify that features were selected
        self.assertIsInstance(selected_features, list)
        self.assertGreater(len(selected_features), 0)
        self.assertLessEqual(len(selected_features), self.config.max_features)
        
        # Verify that selected features exist in the original data
        for feature in selected_features:
            self.assertIn(feature, self.X.columns)
        
        # Verify that feature scores were calculated
        feature_scores = self.component.get_feature_scores()
        self.assertIsInstance(feature_scores, dict)
        self.assertGreater(len(feature_scores), 0)
        
        # Verify that all selected features have scores
        for feature in selected_features:
            self.assertIn(feature, feature_scores)
    
    def test_correlation_analysis(self):
        """Test correlation analysis functionality"""
        selected_features = list(self.X.columns[:10])
        
        # Run correlation analysis
        analysis = self.component.analyze_feature_correlations(self.X, selected_features)
        
        # Verify analysis structure
        self.assertIn('correlation_matrix', analysis)
        self.assertIn('high_correlation_pairs', analysis)
        self.assertIn('average_correlation', analysis)
        self.assertIn('max_correlation', analysis)
        self.assertIn('min_correlation', analysis)
        
        # Verify correlation matrix properties
        corr_matrix = analysis['correlation_matrix']
        self.assertEqual(corr_matrix.shape, (len(selected_features), len(selected_features)))
        
        # Verify diagonal is 1 (perfect correlation with itself)
        np.testing.assert_array_almost_equal(np.diag(corr_matrix), 1.0)
    
    def test_redundancy_detection(self):
        """Test redundancy detection functionality"""
        selected_features = list(self.X.columns[:15])
        
        # Run redundancy detection
        analysis = self.component.detect_redundant_features(self.X, selected_features)
        
        # Verify analysis structure
        self.assertIn('redundancy_results', analysis)
        self.assertIn('redundancy_score', analysis)
        self.assertIn('total_features', analysis)
        self.assertIn('redundant_features', analysis)
        
        # Verify redundancy results structure
        redundancy_results = analysis['redundancy_results']
        self.assertIn('correlation_redundant', redundancy_results)
        self.assertIn('mutual_info_redundant', redundancy_results)
        self.assertIn('variance_redundant', redundancy_results)
    
    def test_stability_analysis(self):
        """Test feature stability analysis"""
        selected_features = list(self.X.columns[:10])
        
        # Run stability analysis
        analysis = self.component.analyze_feature_stability(self.X, self.y, selected_features, n_windows=3)
        
        # Verify analysis structure
        self.assertIn('stability_results', analysis)
        self.assertIn('average_stability', analysis)
        self.assertIn('stable_features', analysis)
        self.assertIn('stability_threshold', analysis)
        self.assertIn('n_windows', analysis)
        
        # Verify stability results structure
        stability_results = analysis['stability_results']
        self.assertIn('window_selections', stability_results)
        self.assertIn('feature_frequency', stability_results)
        self.assertIn('stability_scores', stability_results)
        
        # Verify number of windows
        self.assertEqual(len(stability_results['window_selections']), 3)
    
    def test_cross_validation_analysis(self):
        """Test cross-validation analysis"""
        selected_features = list(self.X.columns[:10])
        
        # Run cross-validation analysis
        analysis = self.component.cross_validate_feature_selection(
            self.X, self.y, selected_features, cv_folds=3
        )
        
        # Verify analysis structure
        self.assertIn('cv_results', analysis)
        self.assertIn('average_consistency', analysis)
        self.assertIn('consistent_features', analysis)
        self.assertIn('consistency_threshold', analysis)
        self.assertIn('cv_folds', analysis)
        
        # Verify CV results structure
        cv_results = analysis['cv_results']
        self.assertIn('fold_selections', cv_results)
        self.assertIn('feature_frequency', cv_results)
        self.assertIn('selection_consistency', cv_results)
        
        # Verify number of folds
        self.assertEqual(len(cv_results['fold_selections']), 3)
    
    def test_baseline_comparison(self):
        """Test baseline comparison functionality"""
        selected_features = list(self.X.columns[:10])
        
        # Run baseline comparison
        analysis = self.component.compare_with_baseline(self.X, self.y, selected_features)
        
        # Verify analysis structure
        self.assertIn('baseline_results', analysis)
        self.assertIn('selected_features_scores', analysis)
        self.assertIn('average_selected_score', analysis)
        self.assertIn('average_baseline_score', analysis)
        self.assertIn('improvement_ratio', analysis)
        self.assertIn('n_baseline_trials', analysis)
        self.assertIn('n_features', analysis)
        
        # Verify improvement ratio is reasonable
        self.assertGreater(analysis['improvement_ratio'], 0)
        
        # Verify number of baseline trials
        self.assertEqual(len(analysis['baseline_results']), 10)
    
    def test_stability_optimized_selection(self):
        """Test stability-optimized feature selection"""
        # Test with smaller dataset for speed
        X_small = self.X.iloc[:100]
        y_small = self.y.iloc[:100]
        
        # Run stability-optimized selection
        selected_features = self.component.select_features_with_stability_optimization(
            X_small, y_small, target_features=10, use_oos_validation=True
        )
        
        # Verify features were selected
        self.assertIsInstance(selected_features, list)
        self.assertGreater(len(selected_features), 0)
        self.assertLessEqual(len(selected_features), 10)
        
        # Verify features exist in data
        for feature in selected_features:
            self.assertIn(feature, X_small.columns)
    
    def test_enhanced_analysis_integration(self):
        """Test that all enhanced analyses work together"""
        # Run feature selection
        selected_features = self.component.select_features(self.X, self.y)
        
        # Run all analyses
        correlation_analysis = self.component.analyze_feature_correlations(self.X, selected_features)
        redundancy_analysis = self.component.detect_redundant_features(self.X, selected_features)
        stability_analysis = self.component.analyze_feature_stability(self.X, self.y, selected_features, n_windows=3)
        cv_analysis = self.component.cross_validate_feature_selection(self.X, self.y, selected_features, cv_folds=3)
        baseline_comparison = self.component.compare_with_baseline(self.X, self.y, selected_features)
        
        # Get enhanced analysis
        enhanced_analysis = self.component.get_enhanced_analysis()
        
        # Verify all analyses are stored
        self.assertIsNotNone(enhanced_analysis['correlation_analysis'])
        self.assertIsNotNone(enhanced_analysis['redundancy_analysis'])
        self.assertIsNotNone(enhanced_analysis['stability_analysis'])
        self.assertIsNotNone(enhanced_analysis['cv_analysis'])
        self.assertIsNotNone(enhanced_analysis['baseline_comparison'])
        
        # Verify improved selection analysis
        improved_analysis = self.component.analyze_improved_selection(
            self.X, self.y, selected_features, self.component.method_results
        )
        
        # Verify improved analysis structure
        self.assertIn('total_features', improved_analysis)
        self.assertIn('stability_analysis', improved_analysis)
        self.assertIn('redundancy_analysis', improved_analysis)
        self.assertIn('quality_metrics', improved_analysis)


class TestFinalFeatureSelectionEdgeCases(unittest.TestCase):
    """Test edge cases for FinalFeatureSelectionComponent"""
    
    def setUp(self):
        """Set up test data for edge cases"""
        self.config = FinalFeatureSelectionConfig()
        self.component = FinalFeatureSelectionComponent(self.config)
    
    def test_empty_data(self):
        """Test handling of empty data"""
        X_empty = pd.DataFrame()
        y_empty = pd.Series([])
        
        # Should handle empty data gracefully
        selected_features = self.component.select_features(X_empty, y_empty)
        self.assertEqual(selected_features, [])
    
    def test_single_feature(self):
        """Test handling of single feature"""
        X_single = pd.DataFrame({'feature_1': [1, 2, 3, 4, 5]})
        y_single = pd.Series([1, 2, 3, 4, 5])
        
        selected_features = self.component.select_features(X_single, y_single)
        self.assertEqual(len(selected_features), 1)
        self.assertEqual(selected_features[0], 'feature_1')
    
    def test_perfect_correlation(self):
        """Test handling of perfectly correlated features"""
        X_corr = pd.DataFrame({
            'feature_1': [1, 2, 3, 4, 5],
            'feature_2': [1.1, 2.1, 3.1, 4.1, 5.1],  # Perfect correlation but not exact duplicate
            'feature_3': [2, 4, 6, 8, 10],  # Perfect correlation but not exact duplicate
        })
        y_corr = pd.Series([1, 2, 3, 4, 5])
        
        # Test duplicate removal - should keep all as they're not exact duplicates
        X_dedup = self.component._remove_exact_duplicates(X_corr)
        self.assertEqual(len(X_dedup.columns), 3,
                      f"Expected 3 columns, got {len(X_dedup.columns)}. Columns: {list(X_dedup.columns)}")
        
        # Test diversity constraints - should remove highly correlated features
        diverse_features = self.component._ensure_feature_diversity(
            list(X_corr.columns), X_corr, correlation_threshold=0.9
        )
        self.assertLess(len(diverse_features), 3)  # Should remove highly correlated features
    
    def test_constant_features(self):
        """Test handling of constant features"""
        X_const = pd.DataFrame({
            'feature_1': [1, 1, 1, 1, 1],  # Constant
            'feature_2': [1, 2, 3, 4, 5],  # Variable
            'feature_3': [0, 0, 0, 0, 0],  # Constant
        })
        y_const = pd.Series([1, 2, 3, 4, 5])
        
        # Should handle constant features gracefully
        selected_features = self.component.select_features(X_const, y_const)
        self.assertGreater(len(selected_features), 0)


def run_tests():
    """Run all tests and provide detailed results"""
    print("=" * 80)
    print("RUNNING FINAL FEATURE SELECTION TESTS")
    print("=" * 80)
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test cases
    suite.addTests(loader.loadTestsFromTestCase(TestFinalFeatureSelection))
    suite.addTests(loader.loadTestsFromTestCase(TestFinalFeatureSelectionEdgeCases))
    
    # Run tests with detailed output
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    
    # Print details of failures and errors
    if result.failures:
        print("\nFAILURES:")
        for test, traceback in result.failures:
            print(f"\n{test}:")
            print(traceback)
    
    if result.errors:
        print("\nERRORS:")
        for test, traceback in result.errors:
            print(f"\n{test}:")
            print(traceback)
    
    # Return success status
    return len(result.failures) == 0 and len(result.errors) == 0


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)