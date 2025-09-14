#!/usr/bin/env python3
"""
Test script to verify that feature names are now printed directly in validation logs.
"""

import sys
import os
sys.path.append('/Users/remyroche/Documents/Ares/src')

from src.utils.tprint import tprint
from src.training.utils.feature_selection.main_framework import FeatureSelectionFramework
import numpy as np
import pandas as pd
import logging

def main():
    """Test the enhanced validation logging with feature names."""
    tprint("🧪 Testing enhanced validation logging with feature names...")

    try:
        # Create a test dataset with known issues
        np.random.seed(42)
        n_samples, n_features = 1000, 10

        # Create feature data
        X = np.random.randn(n_samples, n_features)

        # Add a constant feature (feature_0 will be constant)
        X[:, 0] = 5.0  # Constant feature

        # Add highly correlated features (feature_1 and feature_2)
        X[:, 1] = X[:, 2] + 0.01 * np.random.randn(n_samples)  # Nearly perfectly correlated

        # Create target
        y = np.random.randn(n_samples)

        # Convert to DataFrame with meaningful names
        feature_names = ['constant_feature', 'high_corr_A', 'high_corr_B', 'normal_1', 'normal_2',
                        'normal_3', 'normal_4', 'normal_5', 'normal_6', 'normal_7']
        X_df = pd.DataFrame(X, columns=feature_names)

        # Initialize framework
        framework = FeatureSelectionFramework()

        # Test the validation
        tprint("🔍 Running validation with feature names...")
        validation_result = framework.data_validator.validate_data_quality(
            X_df.values, y, feature_names
        )

        tprint("✅ Validation completed!")
        tprint(f"Valid: {validation_result.get('is_valid', False)}")
        tprint(f"Issues: {len(validation_result.get('issues', []))}")
        tprint(f"Warnings: {len(validation_result.get('warnings', []))}")

        # Print issues and warnings
        if validation_result.get('issues'):
            tprint("\n🚨 ISSUES:")
            for issue in validation_result['issues']:
                tprint(f"  - {issue}")

        if validation_result.get('warnings'):
            tprint("\n⚠️  WARNINGS:")
            for warning in validation_result['warnings']:
                tprint(f"  - {warning}")

    except Exception as e:
        tprint(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
