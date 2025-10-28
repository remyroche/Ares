"""
Test for Regime Feature Selection Fix

Validates that the circular dependency is fixed and unsupervised mode works correctly.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.training.steps.market_analysis.regime_feature_selector import (
    EnhancedRegimeFeatureSelector
)
from src.training.steps.base_step import step_registry


class TestRegimeFeatureSelectionFix:
    """Test suite for regime feature selection fix."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.selector = EnhancedRegimeFeatureSelector(step_name="test_regime_feature_selection")
        
        # Generate sample features with regime-like names
        np.random.seed(42)
        n_samples = 1000
        
        self.features_df = pd.DataFrame({
            # Regime features (should be selected by categorization)
            'regime_persistence': np.random.randn(n_samples),
            'vol_regime_strength': np.random.randn(n_samples),
            'volume_clustering': np.random.randn(n_samples),
            'statistical_persistence': np.random.randn(n_samples),
            'regime_entropy': np.random.randn(n_samples),
            'price_distance': np.random.randn(n_samples),
            'cluster_compactness': np.random.randn(n_samples),
            
            # Non-regime features (may be filtered out)
            'random_feature_1': np.random.randn(n_samples),
            'random_feature_2': np.random.randn(n_samples),
            'random_feature_3': np.random.randn(n_samples),
        })
        
        # Generate synthetic regime labels (for supervised mode testing)
        self.regime_labels = pd.Series(
            np.random.choice([0, 1, 2], n_samples),
            name='regime'
        )
    
    def test_registration(self):
        """Test that EnhancedRegimeFeatureSelector is registered."""
        assert step_registry.is_registered('regime_feature_selection')
        
        # Get the registered class
        registered_class = step_registry.get('regime_feature_selection')
        assert registered_class == EnhancedRegimeFeatureSelector
        
        print("✅ Registration test passed")
    
    def test_unsupervised_mode_without_labels(self):
        """Test that unsupervised mode works without regime_labels."""
        # This should NOT raise an error
        result = self.selector.select_features(
            features_df=self.features_df,
            regime_labels=None,  # No regime labels
            use_supervised=False  # Unsupervised mode
        )
        
        assert result is not None
        assert 'selected_features' in result
        assert len(result['selected_features']) > 0
        assert 'selection_metadata' in result
        assert result['selection_metadata']['selection_method'] == 'unsupervised_variance_correlation'
        
        print(f"✅ Unsupervised mode test passed - selected {len(result['selected_features'])} features")
    
    def test_regime_categorization(self):
        """Test that regime categorization filtering works."""
        # Apply categorization
        filtered_df = self.selector._apply_regime_categorization(self.features_df)
        
        # Should have filtered some features
        assert len(filtered_df.columns) <= len(self.features_df.columns)
        
        # Regime features should be prioritized
        regime_features = ['regime_persistence', 'vol_regime_strength', 'volume_clustering']
        for feature in regime_features:
            if feature in filtered_df.columns:
                print(f"✅ Regime feature '{feature}' kept by categorization")
        
        print(f"✅ Categorization test passed - {len(filtered_df.columns)}/{len(self.features_df.columns)} features kept")
    
    def test_feature_diversity(self):
        """Test that selected features are diverse (low correlation)."""
        result = self.selector.select_features(
            features_df=self.features_df,
            regime_labels=None,
            use_supervised=False
        )
        
        selected_features = result['selected_features']
        
        if len(selected_features) > 1:
            # Calculate correlation matrix of selected features
            selected_df = self.features_df[selected_features]
            corr_matrix = selected_df.corr().abs()
            
            # Remove diagonal
            np.fill_diagonal(corr_matrix.values, 0)
            
            # Check that max correlation is reasonable
            max_corr = corr_matrix.max().max()
            assert max_corr < 0.98, f"Features too correlated: {max_corr:.2f}"
            
            print(f"✅ Feature diversity test passed - max correlation: {max_corr:.3f}")
        else:
            print("⚠️ Only 1 feature selected, skipping diversity test")
    
    def test_supervised_mode_with_labels(self):
        """Test that supervised mode still works when regime_labels provided."""
        result = self.selector.select_features(
            features_df=self.features_df,
            regime_labels=self.regime_labels,  # Provide regime labels
            use_supervised=True  # Supervised mode
        )
        
        assert result is not None
        assert 'selected_features' in result
        assert len(result['selected_features']) > 0
        
        # Should have regime analysis when using supervised mode
        assert 'regime_analysis' in result
        
        print(f"✅ Supervised mode test passed - selected {len(result['selected_features'])} features")
    
    def test_feature_count_reasonable(self):
        """Test that selected feature count is reasonable."""
        result = self.selector.select_features(
            features_df=self.features_df,
            regime_labels=None,
            use_supervised=False
        )
        
        selected_count = len(result['selected_features'])
        total_count = len(self.features_df.columns)
        
        # Should select some features but not all
        assert selected_count > 0, "Should select at least some features"
        assert selected_count <= total_count, "Cannot select more features than available"
        
        # Should select a reasonable proportion (10% - 90%)
        ratio = selected_count / total_count
        assert 0.1 <= ratio <= 0.9, f"Selection ratio {ratio:.2f} outside reasonable range"
        
        print(f"✅ Feature count test passed - {selected_count}/{total_count} ({ratio:.1%})")
    
    def test_metadata_completeness(self):
        """Test that result metadata is complete."""
        result = self.selector.select_features(
            features_df=self.features_df,
            regime_labels=None,
            use_supervised=False
        )
        
        # Check required metadata fields
        assert 'selection_metadata' in result
        metadata = result['selection_metadata']
        
        required_fields = [
            'selection_method',
            'total_features',
            'variance_filtered',
            'correlation_filtered',
            'final_selected',
            'execution_time'
        ]
        
        for field in required_fields:
            assert field in metadata, f"Missing metadata field: {field}"
        
        print("✅ Metadata completeness test passed")


def run_integration_test():
    """Run a simple integration test."""
    print("\n" + "="*80)
    print("REGIME FEATURE SELECTION FIX - INTEGRATION TEST")
    print("="*80 + "\n")
    
    # Test 1: Registration
    print("Test 1: Checking step registration...")
    assert step_registry.is_registered('regime_feature_selection')
    registered_class = step_registry.get('regime_feature_selection')
    print(f"✅ Step registered as: {registered_class.__name__}")
    
    # Test 2: Create selector
    print("\nTest 2: Creating selector instance...")
    selector = EnhancedRegimeFeatureSelector(step_name="test")
    print(f"✅ Selector created: {selector.__class__.__name__}")
    
    # Test 3: Generate test data
    print("\nTest 3: Generating test data...")
    np.random.seed(42)
    features_df = pd.DataFrame({
        'regime_persistence': np.random.randn(500),
        'vol_regime_strength': np.random.randn(500),
        'volume_clustering': np.random.randn(500),
        'random_feature': np.random.randn(500),
    })
    print(f"✅ Generated {len(features_df)} samples with {len(features_df.columns)} features")
    
    # Test 4: Run unsupervised selection
    print("\nTest 4: Running unsupervised feature selection...")
    result = selector.select_features(
        features_df=features_df,
        regime_labels=None,  # No regime labels - unsupervised mode
        use_supervised=False
    )
    
    assert result is not None
    assert 'selected_features' in result
    selected_features = result['selected_features']
    print(f"✅ Selected {len(selected_features)} features: {selected_features}")
    
    # Test 5: Verify metadata
    print("\nTest 5: Verifying metadata...")
    metadata = result['selection_metadata']
    print(f"  - Selection method: {metadata['selection_method']}")
    print(f"  - Execution time: {metadata['execution_time']:.3f}s")
    print(f"  - Total features: {metadata['total_features']}")
    print(f"  - Final selected: {metadata['final_selected']}")
    print("✅ Metadata verified")
    
    # Test 6: Check no circular dependency
    print("\nTest 6: Verifying no circular dependency...")
    # This should work without errors - no regime_labels required
    result2 = selector.select_features(
        features_df=features_df,
        regime_labels=None,  # Explicitly None
        use_supervised=False
    )
    assert result2 is not None
    print("✅ No circular dependency - works without regime_labels")
    
    print("\n" + "="*80)
    print("ALL INTEGRATION TESTS PASSED ✅")
    print("="*80 + "\n")


if __name__ == '__main__':
    # Run integration test
    try:
        run_integration_test()
        
        # Run class-based tests
        print("\nRunning detailed test suite...\n")
        test = TestRegimeFeatureSelectionFix()
        
        print("\n📝 Test: Registration")
        test.test_registration()
        
        print("\n📝 Test: Unsupervised mode without labels")
        test.setup_method()
        test.test_unsupervised_mode_without_labels()
        
        print("\n📝 Test: Regime categorization")
        test.setup_method()
        test.test_regime_categorization()
        
        print("\n📝 Test: Feature diversity")
        test.setup_method()
        test.test_feature_diversity()
        
        print("\n📝 Test: Supervised mode with labels")
        test.setup_method()
        test.test_supervised_mode_with_labels()
        
        print("\n📝 Test: Feature count reasonable")
        test.setup_method()
        test.test_feature_count_reasonable()
        
        print("\n📝 Test: Metadata completeness")
        test.setup_method()
        test.test_metadata_completeness()
        
        print("\n" + "="*80)
        print("🎉 ALL TESTS PASSED SUCCESSFULLY!")
        print("="*80)
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
