#!/usr/bin/env python3
"""
Test Enhanced Validation System
Tests the enhanced data quality validator with feature-specific thresholds and market gap detection.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from src.utils.enhanced_data_quality_validator import enhanced_validate_features


def create_test_data():
    """Create test data with various issues to test the enhanced validation."""

    # Create time index
    start_date = datetime(2024, 1, 1)
    dates = [start_date + timedelta(minutes=i) for i in range(1000)]

    # Create base data
    data = pd.DataFrame(index=dates)

    # 1. Price features (should be nearly complete)
    data["close_price"] = np.random.randn(1000) * 100 + 50000
    data["volume"] = np.random.randint(1000, 10000, 1000)

    # Add some missing values to price (should trigger warnings with strict thresholds)
    data.iloc[100:106, data.columns.get_loc("close_price")] = (
        np.nan
    )  # 6 missing values (0.6%)
    data.iloc[200:211, data.columns.get_loc("volume")] = (
        np.nan
    )  # 11 missing values (1.1%)

    # 2. Technical indicators (should have moderate tolerance)
    data["rsi_14"] = np.random.uniform(0, 100, 1000)
    data["macd"] = np.random.randn(1000) * 2

    # Add some missing values to technical indicators
    data.iloc[300:321, data.columns.get_loc("rsi_14")] = (
        np.nan
    )  # 21 missing values (2.1%)

    # 3. Multi-timeframe features (should have moderate tolerance)
    data["price_1m_ratio"] = np.random.randn(1000) * 0.1 + 1
    data["volume_5m_avg"] = np.random.randint(5000, 15000, 1000)

    # Add some missing values to multi-timeframe features
    data.iloc[400:431, data.columns.get_loc("price_1m_ratio")] = (
        np.nan
    )  # 31 missing values (3.1%)

    # 4. Wavelet features (should have lenient tolerance)
    data["wavelet_coeff_1"] = np.random.randn(1000) * 1e-6  # Very low variance
    data["wavelet_coeff_2"] = np.random.randn(1000) * 1e-8  # Extremely low variance

    # Add some missing values to wavelet features (edge effects)
    data.iloc[0:51, data.columns.get_loc("wavelet_coeff_1")] = (
        np.nan
    )  # 51 missing values (5.1%)
    data.iloc[950:1000, data.columns.get_loc("wavelet_coeff_2")] = (
        np.nan
    )  # 50 missing values (5.0%)

    # 5. Add some infinite values
    data.iloc[500, data.columns.get_loc("rsi_14")] = np.inf
    data.iloc[501, data.columns.get_loc("macd")] = -np.inf

    # 6. Add some extreme values
    data.iloc[600, data.columns.get_loc("close_price")] = 1e8  # Extreme value

    # 7. Add some constant values
    data["constant_feature"] = 42  # All constant values

    # 8. Add object dtype (string) - should be fixed
    data["string_feature"] = ["test"] * 1000

    # 9. Add datetime as object - should be converted to timestamp
    data["datetime_feature"] = pd.to_datetime(dates).astype(str)

    return data


def test_enhanced_validation():
    """Test the enhanced validation system."""

    print("🧪 Testing Enhanced Data Quality Validation System")
    print("=" * 60)

    # Create test data
    test_data = create_test_data()
    print(f"📊 Created test data with shape: {test_data.shape}")
    print(f"📊 Data types: {test_data.dtypes.value_counts().to_dict()}")

    # Run enhanced validation
    print("\n🔍 Running enhanced validation...")
    validation_results = enhanced_validate_features(test_data, "test_dataset")

    # Display results
    print("\n📊 VALIDATION RESULTS:")
    print("-" * 40)

    # Summary
    summary = validation_results["summary"]
    print(f"Total issues: {summary['total_issues']}")
    print(f"Critical: {summary['critical_issues']}")
    print(f"Errors: {summary['error_issues']}")
    print(f"Warnings: {summary['warning_issues']}")
    print(f"Info: {summary['info_issues']}")

    # Feature type breakdown
    print(f"\n📊 FEATURE TYPE BREAKDOWN:")
    print("-" * 40)
    for feature_type, count in validation_results["feature_type_breakdown"].items():
        print(f"{feature_type}: {count} features")

    # Market gaps
    if validation_results["market_gaps"]["gaps_detected"]:
        print(f"\n⚠️ MARKET GAPS DETECTED:")
        print("-" * 40)
        gap_summary = validation_results["market_gaps"]["gap_summary"]
        print(f"Total gaps: {gap_summary['total_gaps']}")
        print(f"Average duration: {gap_summary['avg_gap_duration']:.1f} periods")
        print(f"Max duration: {gap_summary['max_gap_duration']} periods")
        print(f"Affected features: {gap_summary['affected_features_count']}")

    # Data type fixes
    if validation_results["data_type_fixes"]:
        print(f"\n✅ DATA TYPE FIXES APPLIED:")
        print("-" * 40)
        for fix in validation_results["data_type_fixes"]:
            print(f"  - {fix}")

    # Detailed issues
    print(f"\n🔍 DETAILED ISSUES:")
    print("-" * 40)
    for issue in validation_results["issues"][:15]:  # Show first 15 issues
        feature_type = issue.get("feature_type", "unknown")
        threshold = issue.get("threshold_applied", 0)
        print(f"  - {issue['feature']} ({feature_type}): {issue['issue_type']}")
        print(f"    {issue['description']}")
        if threshold > 0:
            print(f"    Threshold applied: {threshold*100:.1f}%")
        print()

    # Recommendations
    if validation_results["recommendations"]:
        print(f"\n💡 RECOMMENDATIONS:")
        print("-" * 40)
        for rec in validation_results["recommendations"]:
            print(f"  - {rec}")

    # Test specific scenarios
    print(f"\n🧪 TESTING SPECIFIC SCENARIOS:")
    print("-" * 40)

    # Test 1: Price features with strict thresholds
    price_features = [col for col in test_data.columns if "price" in col.lower()]
    print(f"1. Price features ({len(price_features)}): {price_features}")
    for feature in price_features:
        missing_pct = test_data[feature].isna().sum() / len(test_data) * 100
        print(f"   - {feature}: {missing_pct:.1f}% missing")

    # Test 2: Wavelet features with lenient thresholds
    wavelet_features = [col for col in test_data.columns if "wavelet" in col.lower()]
    print(f"\n2. Wavelet features ({len(wavelet_features)}): {wavelet_features}")
    for feature in wavelet_features:
        missing_pct = test_data[feature].isna().sum() / len(test_data) * 100
        variance = test_data[feature].var()
        print(f"   - {feature}: {missing_pct:.1f}% missing, variance: {variance:.2e}")

    # Test 3: Multi-timeframe features
    mtf_features = [
        col
        for col in test_data.columns
        if any(pattern in col for pattern in ["_1m_", "_5m_"])
    ]
    print(f"\n3. Multi-timeframe features ({len(mtf_features)}): {mtf_features}")
    for feature in mtf_features:
        missing_pct = test_data[feature].isna().sum() / len(test_data) * 100
        print(f"   - {feature}: {missing_pct:.1f}% missing")

    print(f"\n✅ Enhanced validation test completed successfully!")

    return validation_results


if __name__ == "__main__":
    test_enhanced_validation()
