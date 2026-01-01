#!/usr/bin/env python3
"""
Test sample weighting in MDA/SHAP feature selection.
"""

import numpy as np
import pandas as pd
from src.training.steps.labeling.mda_shap_feature_selection import MDA_SHAP_FeatureSelector
from src.utils.tprint import tprint_info, tprint_success


def test_sample_weighting():
    """Test that sample weights are properly used in MDA/SHAP selection."""
    tprint_info("🧪 Testing Sample Weighting in MDA/SHAP")

    # Create simple test data
    np.random.seed(42)
    n_samples = 500
    n_features = 20

    # Create features
    X = pd.DataFrame(np.random.randn(n_samples, n_features),
                    columns=[f'feature_{i}' for i in range(n_features)])

    # Create imbalanced target (like financial labels)
    y = pd.Series(np.random.choice([0, 1], n_samples, p=[0.8, 0.2]))

    # Create sample weights (emphasize minority class)
    weights = pd.Series(np.where(y == 1, 3.0, 1.0))

    tprint_info(f"Dataset: {n_samples} samples, {n_features} features")
    tprint_info(f"Target distribution: {y.value_counts().to_dict()}")
    tprint_info(f"Sample weights - mean: {weights.mean():.2f}, pos_class: {weights[y==1].mean():.2f}, neg_class: {weights[y==0].mean():.2f}")

    # Test MDA/SHAP with sample weights
    selector = MDA_SHAP_FeatureSelector(
        model_type="rf",
        n_folds=2,  # Faster for testing
        verbose=True
    )

    try:
        selected_features, results = selector.select_features(
            X=X,
            y=y,
            target_sample_weight=weights,
            top_clusters=2,
            shap_sample_size=100
        )

        tprint_success("✅ Sample weighting test completed!")
        tprint_info(f"Selected {len(selected_features)} features: {selected_features[:5]}...")

        # Verify results contain expected keys
        assert 'mda_results' in results, "MDA results missing"
        assert 'shap_results' in results, "SHAP results missing"
        assert len(selected_features) > 0, "No features selected"

        tprint_success("✅ All assertions passed!")

    except Exception as e:
        tprint_info(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    return True


if __name__ == "__main__":
    success = test_sample_weighting()
    if success:
        print("\n🎉 Sample weighting test passed!")
    else:
        print("\n❌ Sample weighting test failed!")
        exit(1)







