#!/usr/bin/env python3
"""
Test script for feature output validation with downstream compatibility checks.
Demonstrates how the validator detects issues that would affect downstream ML pipeline steps.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import asyncio

# Add the project root to the Python path
import sys
from pathlib import Path

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.utils.feature_output_validator import validate_feature_output


def create_test_features(
    feature_type: str = "good", issues: list = None
) -> pd.DataFrame:
    """Create test features with different quality levels and issues."""

    # Create base timestamp index
    base_time = datetime(2024, 1, 1)
    timestamps = [base_time + timedelta(minutes=i) for i in range(1000)]

    if feature_type == "good":
        # Good quality features
        features = pd.DataFrame(
            {
                "price_momentum": np.random.randn(1000),
                "volume_ratio": np.random.uniform(0.5, 2.0, 1000),
                "volatility_rolling": np.random.uniform(0.01, 0.05, 1000),
                "rsi": np.random.uniform(20, 80, 1000),
                "macd": np.random.randn(1000),
                "bbands_position": np.random.uniform(-1, 1, 1000),
            },
            index=timestamps,
        )

    elif feature_type == "sklearn_incompatible":
        # Features with sklearn compatibility issues
        features = pd.DataFrame(
            {
                "price_momentum": np.random.randn(1000),
                "volume_ratio": np.random.uniform(0.5, 2.0, 1000),
                "categorical_feature": [
                    "high" if x > 0.5 else "low" for x in np.random.rand(1000)
                ],  # Object dtype
                "object_feature": [
                    {"value": x} for x in np.random.rand(1000)
                ],  # Object dtype
            },
            index=timestamps,
        )

    elif feature_type == "scaling_issues":
        # Features that would cause StandardScaler issues
        features = pd.DataFrame(
            {
                "price_momentum": np.random.randn(1000),
                "zero_variance_feature": np.ones(1000),  # Zero variance
                "constant_feature": np.full(1000, 42.0),  # Constant
                "extreme_values": np.random.uniform(-1e8, 1e8, 1000),  # Extreme values
            },
            index=timestamps,
        )

    elif feature_type == "model_training_issues":
        # Features that would cause model training issues
        features = pd.DataFrame(
            {
                "price_momentum": np.random.randn(1000),
                "perfect_corr_1": np.random.randn(1000),
                "perfect_corr_2": np.random.randn(1000)
                * 2,  # Perfectly correlated with above
                "high_cardinality": np.random.rand(1000),  # High cardinality
                "zero_variance_1": np.ones(1000),
                "zero_variance_2": np.ones(1000),
                "zero_variance_3": np.ones(1000),
            },
            index=timestamps,
        )

    elif feature_type == "selection_issues":
        # Features that would cause feature selection issues
        features = pd.DataFrame(
            {
                "price_momentum": np.random.randn(1000),
                "low_variance": np.random.uniform(0, 1e-10, 1000),  # Very low variance
                "sparse_feature": np.random.choice(
                    [0, 1], 1000, p=[0.95, 0.05]
                ),  # Sparse
                "another_sparse": np.random.choice(
                    [0, 1], 1000, p=[0.98, 0.02]
                ),  # Very sparse
            },
            index=timestamps,
        )

    elif feature_type == "regime_issues":
        # Features missing regime-specific requirements
        features = pd.DataFrame(
            {
                "price_momentum": np.random.randn(1000),
                "volume_ratio": np.random.uniform(0.5, 2.0, 1000),
                # Missing temporal and volatility features
            },
            index=timestamps,
        )

    elif feature_type == "mixed_issues":
        # Features with multiple issues
        features = pd.DataFrame(
            {
                "price_momentum": np.random.randn(1000),
                "object_feature": [
                    "high" if x > 0.5 else "low" for x in np.random.rand(1000)
                ],
                "zero_variance": np.ones(1000),
                "extreme_values": np.random.uniform(-1e8, 1e8, 1000),
                "perfect_corr_1": np.random.randn(1000),
                "perfect_corr_2": np.random.randn(1000) * 2,
            },
            index=timestamps,
        )

    else:
        # Unknown type
        features = pd.DataFrame(
            {
                "unknown_feature": np.random.randn(1000),
            },
            index=timestamps,
        )

    # Add specific issues if requested
    if issues:
        for issue in issues:
            if issue == "nan_values":
                features.loc[features.index[100:200], "price_momentum"] = np.nan
            elif issue == "infinite_values":
                features.loc[features.index[300:400], "volume_ratio"] = np.inf
            elif issue == "negative_values":
                features.loc[features.index[500:600], "rsi"] = -10

    return features


async def test_feature_output_validation():
    """Test the feature output validation with different scenarios."""

    print("🧪 Testing Feature Output Validation with Downstream Compatibility")
    print("=" * 70)

    # Test scenarios
    test_scenarios = [
        ("Good Features", "good", []),
        ("Sklearn Incompatible", "sklearn_incompatible", []),
        ("Scaling Issues", "scaling_issues", []),
        ("Model Training Issues", "model_training_issues", []),
        ("Feature Selection Issues", "selection_issues", []),
        ("Regime Analysis Issues", "regime_issues", []),
        ("Mixed Issues", "mixed_issues", []),
        ("Good Features with NaN", "good", ["nan_values"]),
        ("Good Features with Infinite", "good", ["infinite_values"]),
    ]

    for scenario_name, feature_type, issues in test_scenarios:
        print(f"\n📊 Test: {scenario_name}")
        print("-" * 40)

        # Create test features
        features = create_test_features(feature_type, issues)

        # Test with different method names
        method_names = [
            "engineer_features",
            "analyze_wavelet_transforms",
            "analyze_microstructure_features",
            "calculate_price_impact",
        ]

        for method_name in method_names:
            print(f"  Method: {method_name}")

            try:
                # Validate feature output
                validation_results = validate_feature_output(
                    features=features,
                    method_name=method_name,
                    input_data_shape=(1000, 5),  # Simulate input data shape
                )

                # Display results
                if validation_results["validation_passed"]:
                    print(
                        f"    ✅ Passed (Score: {validation_results['output_quality_score']:.2f})"
                    )
                else:
                    print(
                        f"    ❌ Failed (Score: {validation_results['output_quality_score']:.2f})"
                    )

                # Show critical issues
                if validation_results["critical_issues"]:
                    print(
                        f"    Critical Issues: {len(validation_results['critical_issues'])}"
                    )
                    for issue in validation_results["critical_issues"][
                        :2
                    ]:  # Show first 2
                        print(f"      - {issue}")

                # Show warnings
                if validation_results["warnings"]:
                    print(f"    Warnings: {len(validation_results['warnings'])}")
                    for warning in validation_results["warnings"][:2]:  # Show first 2
                        print(f"      - {warning}")

                # Show recommendations
                if validation_results["recommendations"]:
                    print(
                        f"    Recommendations: {len(validation_results['recommendations'])}"
                    )
                    for rec in validation_results["recommendations"][
                        :2
                    ]:  # Show first 2
                        print(f"      - {rec}")

            except Exception as e:
                print(f"    Error: {e}")

        print()

    # Test with different data formats
    print("\n📊 Test: Different Data Formats")
    print("-" * 40)

    # Test DataFrame format
    df_features = create_test_features("good")
    df_results = validate_feature_output(df_features, "engineer_features")
    print(f"DataFrame format: {'✅' if df_results['validation_passed'] else '❌'}")

    # Test dict format with Series
    dict_features = {
        "feature1": pd.Series(np.random.randn(1000)),
        "feature2": pd.Series(np.random.randn(1000)),
    }
    dict_results = validate_feature_output(dict_features, "engineer_features")
    print(
        f"Dict with Series format: {'✅' if dict_results['validation_passed'] else '❌'}"
    )

    # Test dict format with scalars
    scalar_features = {
        "feature1": 1.0,
        "feature2": 2.0,
    }
    scalar_results = validate_feature_output(scalar_features, "engineer_features")
    print(
        f"Dict with scalars format: {'✅' if scalar_results['validation_passed'] else '❌'}"
    )

    print("\n✅ Feature output validation testing completed!")


if __name__ == "__main__":
    asyncio.run(test_feature_output_validation())
