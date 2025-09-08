#!/usr/bin/env python3
"""
Test Step07 Integration for SR Feature Selection
"""

import pandas as pd
import numpy as np
from src.training.steps.market_analysis.step07_enhanced_matrix_operations import Step7EnhancedMatrixOperations

def test_step07_feature_selection():
    """Test step07's feature selection functionality."""
    print("🧪 Testing Step07 Feature Selection Integration...")

    # Create test data
    np.random.seed(42)
    n_samples, n_features = 1000, 50
    X = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=[f'feature_{i}' for i in range(n_features)]
    )
    y = pd.Series(np.random.choice([0, 1], n_samples))
    labels_df = pd.DataFrame({'target': y})

    # Test configuration
    config = {
        'step07_enhanced_matrix_operations': {
            'target_features': 25,
            'removal_fraction': 0.5,
            'enable_regime_selection': False,
            'enable_shap_filtering': True,
            'output_dir': 'data/matrix_operations'
        }
    }

    try:
        # Initialize step07
        step07 = Step7EnhancedMatrixOperations(config=config)
        print("✅ Step07 instance created successfully")

        # Test feature selection
        print(f"📊 Testing with {X.shape[1]} features...")
        X_selected, metadata = step07.regime_aware_initial_filtering(
            features_df=X,
            labels_df=labels_df,
            regime_labels=None
        )

        print("✅ Feature selection completed!")
        print(f"   Original features: {X.shape[1]}")
        print(f"   Selected features: {X_selected.shape[1]}")
        print(f"   Method: {metadata.get('method', 'unknown')}")
        print(".1%")

        # Show top features
        if 'top_features_by_mi' in metadata:
            print("🏆 Top 5 features by MI:")
            for i, feature in enumerate(metadata['top_features_by_mi'][:5], 1):
                print(f"   {i}. {feature}")

        return True

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_step07_feature_selection()
    if success:
        print("\n🎉 Step07 integration test PASSED!")
        print("Ready to use step07 feature selection in step02_5.")
    else:
        print("\n💥 Step07 integration test FAILED!")
        print("Check imports and dependencies.")
