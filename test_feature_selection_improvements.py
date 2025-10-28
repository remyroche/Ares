"""
Tests for Feature Selection Improvements

This module tests the improved feature selection pipeline including:
- Unsupervised feature selection
- Feature validation
- Execution mode configurations
- Circular dependency prevention
"""

import sys
import numpy as np
import pandas as pd
from typing import List, Dict, Any

# Add project root to path
sys.path.insert(0, '/workspace')

# Import feature selection components
from src.training.steps.market_analysis.regime_feature_selector import (
    EnhancedRegimeFeatureSelector,
    EnhancedRegimeFeatureSelectorConfig
)

from src.training.steps.market_analysis.feature_selection_validation import (
    FeatureSelectionValidator,
    validate_regime_clustering_features,
    validate_hdbscan_features
)

from src.feature_generation.categories.regime_feature_categorization import (
    RegimeFeatureCategorizer,
    FeatureUseCase,
    get_regime_clustering_features,
    validate_feature_set
)

from src.training.steps.market_analysis.hdbscan_clustering.optimization.optimized_hdbscan_regime_discovery import (
    OptimizedHDBSCANRegimeDiscoveryConfig
)


def generate_test_data(n_samples: int = 1000, n_features: int = 100) -> pd.DataFrame:
    """Generate synthetic test data."""
    np.random.seed(42)
    
    # Create feature names matching categorization system
    feature_names = []
    
    # Core regime features
    feature_names.extend([
        'regime_persistence', 'vol_regime_strength', 'vol_clustering',
        'vol_regime_change', 'volume_regime_strength', 'volume_clustering',
        'statistical_persistence', 'distribution_stability'
    ])
    
    # Advanced regime features
    feature_names.extend([
        'regime_entropy', 'regime_complexity', 'regime_fractal_dimension',
        'regime_hurst_exponent', 'regime_memory_strength'
    ])
    
    # Clustering features
    feature_names.extend([
        'price_distance', 'volume_distance', 'cluster_compactness',
        'separation_strength', 'cluster_consistency', 'temporal_stability'
    ])
    
    # Fill remaining with generic features
    remaining = n_features - len(feature_names)
    if remaining > 0:
        feature_names.extend([f'feature_{i}' for i in range(remaining)])
    
    # Generate data
    data = np.random.randn(n_samples, n_features)
    df = pd.DataFrame(data, columns=feature_names[:n_features])
    
    return df


def generate_regime_labels(n_samples: int = 1000, n_regimes: int = 3) -> pd.Series:
    """Generate synthetic regime labels."""
    np.random.seed(42)
    labels = np.random.choice(n_regimes, n_samples)
    return pd.Series(labels)


class TestUnsupervisedFeatureSelection:
    """Test unsupervised feature selection mode."""
    
    def test_unsupervised_selection_without_labels(self):
        """Test that unsupervised selection works without regime labels."""
        print("\n" + "="*80)
        print("TEST: Unsupervised Selection Without Labels")
        print("="*80)
        
        # Generate test data
        features_df = generate_test_data(n_samples=1000, n_features=100)
        
        # Create selector
        selector = EnhancedRegimeFeatureSelector(step_name="test_regime_feature_selection")
        
        # Run unsupervised selection (no labels)
        result = selector.select_features(
            features_df=features_df,
            regime_labels=None,
            use_supervised=False
        )
        
        # Assertions
        assert 'selected_features' in result
        assert len(result['selected_features']) > 0
        assert len(result['selected_features']) <= selector.config.max_features
        assert result['selection_metadata']['selection_method'] == 'unsupervised_variance_correlation'
        
        print(f"✅ Selected {len(result['selected_features'])} features without labels")
        print(f"   Method: {result['selection_metadata']['selection_method']}")
        print(f"   Variance threshold: {result['selection_metadata'].get('variance_threshold', 'N/A')}")
        print(f"   Correlation threshold: {result['selection_metadata'].get('correlation_threshold', 'N/A')}")
    
    def test_supervised_fallback_to_unsupervised(self):
        """Test that supervised mode falls back to unsupervised when no labels provided."""
        print("\n" + "="*80)
        print("TEST: Supervised Fallback to Unsupervised")
        print("="*80)
        
        features_df = generate_test_data(n_samples=1000, n_features=100)
        selector = EnhancedRegimeFeatureSelector(step_name="test_regime_feature_selection")
        
        # Request supervised but provide no labels
        result = selector.select_features(
            features_df=features_df,
            regime_labels=None,
            use_supervised=True  # Request supervised
        )
        
        # Should fallback to unsupervised
        assert result['selection_metadata']['selection_method'] == 'unsupervised_variance_correlation'
        print("✅ Successfully fell back to unsupervised when labels not provided")
    
    def test_variance_filtering(self):
        """Test that low-variance features are removed."""
        print("\n" + "="*80)
        print("TEST: Variance Filtering")
        print("="*80)
        
        # Create data with some low-variance features
        n_samples = 1000
        features_df = generate_test_data(n_samples=n_samples, n_features=50)
        
        # Add low-variance features
        features_df['low_var_1'] = 1.0  # Constant
        features_df['low_var_2'] = np.random.randn(n_samples) * 0.001  # Very low variance
        
        selector = EnhancedRegimeFeatureSelector(step_name="test_regime_feature_selection")
        result = selector.select_features(
            features_df=features_df,
            regime_labels=None,
            use_supervised=False
        )
        
        # Low variance features should be filtered out
        selected = result['selected_features']
        assert 'low_var_1' not in selected or 'low_var_2' not in selected
        print(f"✅ Low-variance features filtered out")
        print(f"   Total features: {len(features_df.columns)}")
        print(f"   After filtering: {len(selected)}")
    
    def test_correlation_filtering(self):
        """Test that highly correlated features are removed."""
        print("\n" + "="*80)
        print("TEST: Correlation Filtering")
        print("="*80)
        
        n_samples = 1000
        features_df = generate_test_data(n_samples=n_samples, n_features=30)
        
        # Add highly correlated features
        base_feature = features_df.iloc[:, 0]
        features_df['corr_1'] = base_feature + np.random.randn(n_samples) * 0.01
        features_df['corr_2'] = base_feature + np.random.randn(n_samples) * 0.01
        
        selector = EnhancedRegimeFeatureSelector(step_name="test_regime_feature_selection")
        result = selector.select_features(
            features_df=features_df,
            regime_labels=None,
            use_supervised=False
        )
        
        # Only one of the correlated features should remain
        selected = result['selected_features']
        corr_features_selected = [f for f in selected if f.startswith('corr_')]
        assert len(corr_features_selected) <= 1
        print(f"✅ Highly correlated features filtered")
        print(f"   Correlated features remaining: {len(corr_features_selected)}")


class TestFeatureValidation:
    """Test feature selection validation."""
    
    def test_regime_clustering_validation(self):
        """Test validation for regime clustering features."""
        print("\n" + "="*80)
        print("TEST: Regime Clustering Validation")
        print("="*80)
        
        # Get proper regime clustering features
        categorizer = RegimeFeatureCategorizer()
        proper_features = categorizer.get_priority_features(
            FeatureUseCase.REGIME_CLUSTERING,
            max_features=50
        )
        
        # Validate
        result = validate_regime_clustering_features(proper_features[:40])
        
        assert 'valid' in result
        assert 'use_case_alignment' in result
        assert 'category_representation' in result
        
        print(f"✅ Validation completed")
        print(f"   Valid: {result['valid']}")
        print(f"   Selected: {result['selected_count']}")
        print(f"   Alignment: {result['use_case_alignment']['alignment_percentage']:.1f}%")
    
    def test_invalid_features_detection(self):
        """Test detection of invalid features for use case."""
        print("\n" + "="*80)
        print("TEST: Invalid Features Detection")
        print("="*80)
        
        # Mix of valid and invalid features
        features = [
            'regime_persistence',  # Valid
            'vol_regime_strength',  # Valid
            'invalid_feature_1',  # Invalid
            'invalid_feature_2'   # Invalid
        ]
        
        validator = FeatureSelectionValidator()
        result = validator.validate_feature_selection(
            features,
            FeatureUseCase.REGIME_CLUSTERING
        )
        
        assert result['use_case_alignment']['invalid_count'] > 0
        print(f"✅ Invalid features detected")
        print(f"   Invalid count: {result['use_case_alignment']['invalid_count']}")
        print(f"   Invalid features: {result['use_case_alignment']['invalid_features']}")
    
    def test_category_representation_validation(self):
        """Test validation of category representation."""
        print("\n" + "="*80)
        print("TEST: Category Representation Validation")
        print("="*80)
        
        # Select only from one category (poor representation)
        features = [
            'regime_entropy',
            'regime_complexity',
            'regime_fractal_dimension'
        ]
        
        validator = FeatureSelectionValidator()
        result = validator.validate_feature_selection(
            features,
            FeatureUseCase.REGIME_CLUSTERING,
            expected_categories=['core_regime', 'structural_trend']
        )
        
        # Should have warnings about underrepresented categories
        assert len(result['warnings']) > 0
        assert not result['category_representation']['sufficient_representation']
        print(f"✅ Category underrepresentation detected")
        print(f"   Warnings: {len(result['warnings'])}")
        print(f"   Underrepresented: {result['category_representation']['underrepresented']}")
    
    def test_circular_dependency_detection(self):
        """Test circular dependency detection."""
        print("\n" + "="*80)
        print("TEST: Circular Dependency Detection")
        print("="*80)
        
        validator = FeatureSelectionValidator()
        
        # Test pre-clustering with labels (circular dependency)
        result = validator.validate_circular_dependency(
            feature_selection_method='treeshap',
            has_regime_labels=True,
            clustering_stage='pre'
        )
        
        assert result['has_circular_dependency'] == True
        assert result['recommendation'] is not None
        print(f"✅ Circular dependency detected")
        print(f"   Type: {result['dependency_type']}")
        print(f"   Recommendation: {result['recommendation']}")


class TestExecutionModeConfig:
    """Test execution mode configurations."""
    
    def test_light_mode_config(self):
        """Test light mode feature limits."""
        print("\n" + "="*80)
        print("TEST: Light Mode Configuration")
        print("="*80)
        
        config = OptimizedHDBSCANRegimeDiscoveryConfig(execution_mode="light")
        
        # Check improved light mode settings
        assert config.max_features == 50, f"Expected 50 features, got {config.max_features}"
        assert config.enable_regime_features == True, "Regime features should be enabled"
        assert config.enable_entropy_features == True, "Entropy features should be enabled"
        assert config.enable_normalization_features == True, "Normalization features should be enabled"
        
        print(f"✅ Light mode config correct")
        print(f"   Max features: {config.max_features}")
        print(f"   Regime features: {config.enable_regime_features}")
        print(f"   Entropy features: {config.enable_entropy_features}")
    
    def test_blank_mode_config(self):
        """Test blank mode never disables regime features."""
        print("\n" + "="*80)
        print("TEST: Blank Mode Configuration")
        print("="*80)
        
        config = OptimizedHDBSCANRegimeDiscoveryConfig(execution_mode="blank")
        
        # Check improved blank mode settings
        assert config.max_features == 50, f"Expected 50 features, got {config.max_features}"
        assert config.enable_regime_features == True, "CRITICAL: Regime features must NEVER be disabled"
        assert config.enable_normalization_features == True, "Normalization features should be enabled"
        
        print(f"✅ Blank mode config correct")
        print(f"   Max features: {config.max_features}")
        print(f"   Regime features: {config.enable_regime_features}")
        print(f"   Normalization features: {config.enable_normalization_features}")
    
    def test_full_mode_config(self):
        """Test full mode maintains all features."""
        print("\n" + "="*80)
        print("TEST: Full Mode Configuration")
        print("="*80)
        
        config = OptimizedHDBSCANRegimeDiscoveryConfig(execution_mode="full")
        
        # Full mode should have all features enabled
        assert config.enable_regime_features == True
        assert config.enable_entropy_features == True
        assert config.enable_spectral_features == True
        assert config.enable_normalization_features == True
        
        print(f"✅ Full mode config correct")
        print(f"   Max features: {config.max_features}")
        print(f"   All features enabled: True")


class TestFeatureCategorization:
    """Test feature categorization integration."""
    
    def test_priority_features_for_regime_clustering(self):
        """Test getting priority features for regime clustering."""
        print("\n" + "="*80)
        print("TEST: Priority Features for Regime Clustering")
        print("="*80)
        
        features = get_regime_clustering_features()
        
        assert len(features) > 0
        assert len(features) <= 80
        
        # Should include core regime features
        assert any('regime' in f for f in features)
        
        print(f"✅ Priority features retrieved")
        print(f"   Count: {len(features)}")
        print(f"   Sample: {features[:5]}")
    
    def test_feature_set_validation(self):
        """Test feature set validation with categorization."""
        print("\n" + "="*80)
        print("TEST: Feature Set Validation")
        print("="*80)
        
        # Get valid features
        categorizer = RegimeFeatureCategorizer()
        valid_features = categorizer.get_priority_features(
            FeatureUseCase.REGIME_CLUSTERING,
            max_features=30
        )
        
        # Validate
        result = validate_feature_set(valid_features, FeatureUseCase.REGIME_CLUSTERING)
        
        assert result['validation_passed'] == True
        assert result['invalid_count'] == 0
        
        print(f"✅ Feature set validated")
        print(f"   Valid: {result['valid_count']}")
        print(f"   Invalid: {result['invalid_count']}")


def run_all_tests():
    """Run all tests with detailed output."""
    print("\n" + "="*80)
    print("RUNNING FEATURE SELECTION IMPROVEMENT TESTS")
    print("="*80)
    
    test_results = {
        'passed': 0,
        'failed': 0,
        'errors': []
    }
    
    # Test classes
    test_classes = [
        TestUnsupervisedFeatureSelection,
        TestFeatureValidation,
        TestExecutionModeConfig,
        TestFeatureCategorization
    ]
    
    for test_class in test_classes:
        print(f"\n{'='*80}")
        print(f"Running {test_class.__name__}")
        print(f"{'='*80}")
        
        test_instance = test_class()
        
        # Get all test methods
        test_methods = [method for method in dir(test_instance) if method.startswith('test_')]
        
        for test_method_name in test_methods:
            try:
                test_method = getattr(test_instance, test_method_name)
                test_method()
                test_results['passed'] += 1
            except Exception as e:
                test_results['failed'] += 1
                test_results['errors'].append({
                    'test': f"{test_class.__name__}.{test_method_name}",
                    'error': str(e)
                })
                print(f"❌ FAILED: {test_method_name}")
                print(f"   Error: {e}")
    
    # Print summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    print(f"✅ Passed: {test_results['passed']}")
    print(f"❌ Failed: {test_results['failed']}")
    print(f"📊 Total: {test_results['passed'] + test_results['failed']}")
    
    if test_results['errors']:
        print("\nFailed Tests:")
        for error in test_results['errors']:
            print(f"  - {error['test']}: {error['error']}")
    
    return test_results


if __name__ == "__main__":
    results = run_all_tests()
    
    # Exit with appropriate code
    exit_code = 0 if results['failed'] == 0 else 1
    sys.exit(exit_code)
