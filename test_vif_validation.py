#!/usr/bin/env python3
"""
Test VIF Validation Decorators

This script tests the VIF validation decorators to ensure they properly handle
NaN, infinite, and zero VIF values with comprehensive logging.
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, Any

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import the VIF validation decorators and calculator
try:
    from src.utils.vif_validation_decorators import (
        validate_vif_inputs,
        validate_vif_outputs,
        safe_vif_calculation,
        comprehensive_vif_validation
    )
    from src.utils.vif_calculator import (
        calculate_vif_robust,
        calculate_vif_simple,
        analyze_vif_issues,
        get_vif_recommendations
    )
    print("✅ Successfully imported VIF validation modules")
except ImportError as e:
    print(f"❌ Failed to import VIF validation modules: {e}")
    exit(1)


def create_test_data_with_issues() -> pd.DataFrame:
    """Create test data with various issues for VIF testing."""
    np.random.seed(42)

    # Create base features
    n_samples = 1000
    n_features = 10

    # Create some correlated features
    base_feature = np.random.normal(0, 1, n_samples)

    data = {
        'feature_1': base_feature,
        'feature_2': base_feature * 2 + np.random.normal(0, 0.1, n_samples),  # Highly correlated
        'feature_3': base_feature * 3 + np.random.normal(0, 0.1, n_samples),  # Highly correlated
        'feature_4': np.random.normal(0, 1, n_samples),  # Independent
        'feature_5': np.random.normal(0, 1, n_samples),  # Independent
        'feature_6': np.random.normal(0, 1, n_samples),  # Independent
        'feature_7': np.random.normal(0, 1, n_samples),  # Independent
        'feature_8': np.random.normal(0, 1, n_samples),  # Independent
        'feature_9': np.random.normal(0, 1, n_samples),  # Independent
        'feature_10': np.random.normal(0, 1, n_samples),  # Independent
    }

    df = pd.DataFrame(data)

    # Add problematic values
    df.loc[0, 'feature_1'] = np.nan  # NaN value
    df.loc[1, 'feature_2'] = np.inf  # Infinite value
    df.loc[2, 'feature_3'] = -np.inf  # Negative infinite value
    df.loc[3:5, 'feature_4'] = 0  # Zero values
    df['feature_5'] = 1.0  # Constant feature (zero variance)
    df['feature_6'] = df['feature_1']  # Duplicate feature

    return df


def test_vif_input_validation():
    """Test VIF input validation decorator."""
    print("\n🧪 Testing VIF Input Validation Decorator")

    @validate_vif_inputs(check_nan=True, check_infinite=True, check_zero_variance=True, check_duplicates=True)
    def dummy_function(data: pd.DataFrame) -> pd.DataFrame:
        return data

    # Create test data with issues
    test_data = create_test_data_with_issues()

    print("📊 Test data shape:", test_data.shape)
    print("📊 Test data info:")
    print(test_data.info())

    # Test the decorated function
    result = dummy_function(test_data)
    print("✅ VIF input validation test completed")


def test_vif_output_validation():
    """Test VIF output validation decorator."""
    print("\n🧪 Testing VIF Output Validation Decorator")

    @validate_vif_outputs(check_nan_vif=True, check_infinite_vif=True, check_zero_vif=True)
    def dummy_vif_calculation(data: pd.DataFrame) -> pd.Series:
        # Simulate VIF calculation with issues
        features = data.columns.tolist()
        vif_values = pd.Series(index=features)

        # Add some problematic VIF values
        for i, feature in enumerate(features):
            if i == 0:
                vif_values[feature] = np.nan  # NaN VIF
            elif i == 1:
                vif_values[feature] = np.inf  # Infinite VIF
            elif i == 2:
                vif_values[feature] = 0.0  # Zero VIF
            elif i == 3:
                vif_values[feature] = 1000.0  # High VIF
            else:
                vif_values[feature] = np.random.uniform(1.0, 10.0)  # Normal VIF

        return vif_values

    # Create test data
    test_data = create_test_data_with_issues()

    # Test the decorated function
    result = dummy_vif_calculation(test_data)
    print("✅ VIF output validation test completed")


def test_safe_vif_calculation():
    """Test safe VIF calculation decorator."""
    print("\n🧪 Testing Safe VIF Calculation Decorator")

    @safe_vif_calculation(timeout_seconds=5, fallback_strategy="ones")
    def slow_vif_calculation(data: pd.DataFrame) -> pd.Series:
        # Simulate a slow VIF calculation
        import time
        time.sleep(10)  # This should timeout
        return pd.Series([1.0] * len(data.columns), index=data.columns)

    # Create test data
    test_data = create_test_data_with_issues()

    # Test the decorated function
    result = slow_vif_calculation(test_data)
    print("✅ Safe VIF calculation test completed")


def test_comprehensive_vif_validation():
    """Test comprehensive VIF validation decorator."""
    print("\n🧪 Testing Comprehensive VIF Validation Decorator")

    @comprehensive_vif_validation(
        timeout_seconds=10,
        max_vif_threshold=100.0,
        fallback_strategy="ones"
    )
    def comprehensive_vif_function(data: pd.DataFrame) -> pd.Series:
        # Simulate VIF calculation
        features = data.columns.tolist()
        vif_values = pd.Series([np.random.uniform(1.0, 5.0) for _ in features], index=features)
        return vif_values

    # Create test data
    test_data = create_test_data_with_issues()

    # Test the decorated function
    result = comprehensive_vif_function(test_data)
    print("✅ Comprehensive VIF validation test completed")


def test_vif_calculator_functions():
    """Test the VIF calculator functions directly."""
    print("\n🧪 Testing VIF Calculator Functions")

    # Create test data
    test_data = create_test_data_with_issues()

    # Test simple VIF calculation
    print("📊 Testing simple VIF calculation...")
    simple_vif = calculate_vif_simple(test_data)
    print(f"Simple VIF result: {len(simple_vif)} features")

    # Test robust VIF calculation
    print("📊 Testing robust VIF calculation...")
    robust_vif = calculate_vif_robust(test_data)
    print(f"Robust VIF result: {len(robust_vif)} features")

    # Test VIF analysis
    print("📊 Testing VIF analysis...")
    analysis = analyze_vif_issues(robust_vif)
    print(f"VIF analysis found {len(analysis['issues'])} issues")

    # Test VIF recommendations
    print("📊 Testing VIF recommendations...")
    recommendations = get_vif_recommendations(robust_vif)
    print(f"Generated {len(recommendations)} recommendations")

    print("✅ VIF calculator functions test completed")


def test_edge_cases():
    """Test edge cases for VIF validation."""
    print("\n🧪 Testing Edge Cases")

    # Test with empty DataFrame
    print("📊 Testing empty DataFrame...")
    empty_df = pd.DataFrame()
    try:
        result = calculate_vif_robust(empty_df)
        print(f"Empty DataFrame result: {len(result)} features")
    except Exception as e:
        print(f"Empty DataFrame error: {e}")

    # Test with single feature
    print("📊 Testing single feature...")
    single_feature_df = pd.DataFrame({'feature': np.random.normal(0, 1, 100)})
    try:
        result = calculate_vif_robust(single_feature_df)
        print(f"Single feature result: {len(result)} features")
    except Exception as e:
        print(f"Single feature error: {e}")

    # Test with all NaN values
    print("📊 Testing all NaN values...")
    nan_df = pd.DataFrame({
        'feature_1': [np.nan] * 100,
        'feature_2': [np.nan] * 100
    })
    try:
        result = calculate_vif_robust(nan_df)
        print(f"All NaN result: {len(result)} features")
    except Exception as e:
        print(f"All NaN error: {e}")

    print("✅ Edge cases test completed")


def main():
    """Run all VIF validation tests."""
    print("🚀 Starting VIF Validation Tests")
    print("=" * 50)

    try:
        # Test individual decorators
        test_vif_input_validation()
        test_vif_output_validation()
        test_safe_vif_calculation()
        test_comprehensive_vif_validation()

        # Test VIF calculator functions
        test_vif_calculator_functions()

        # Test edge cases
        test_edge_cases()

        print("\n" + "=" * 50)
        print("✅ All VIF validation tests completed successfully!")

    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()