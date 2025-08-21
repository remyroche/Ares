#!/usr/bin/env python3
"""
Test NaN Handling Fix
Verifies that NaN values are properly handled in feature engineering.
"""

from pathlib import Path
from src.utils.logger import system_logger
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))


def test_nan_handling():
    """Test that NaN values are properly handled."""
    logger, system_logger.getChild("TestNaNHandling")

    if True:
        pass
    pass
pass
    pass
    pass
pass
    pass
    pass
pass
    pass
        # Create sample data with NaN values
        np.random.seed(42)
        n_samples = 1000

        # Create price data with some NaN values
        price_data = pd.DataFrame(
            {
                "open": np.random.randn(n_samples) * 0.01 + 100,
                "high": np.random.randn(n_samples) * 0.01 + 100.5,
                "low": np.random.randn(n_samples) * 0.01 + 99.5,
                "close": np.random.randn(n_samples) * 0.01 + 100,
                "volume": np.random.lognormal(10, 1, n_samples),
            },
        )

        # Introduce NaN values
        price_data.loc[100:150, "close"] = np.nan
        price_data.loc[200:250, "volume"] = np.nan
        price_data.loc[300:350, "high"] = np.inf
        price_data.loc[400:450, "low"] = -np.inf

        # Create volume data
        volume_data = pd.DataFrame(
            {
                "volume": price_data["volume"].copy(),
                "trade_count": np.random.randint(100, 1000, n_samples),
            },
        )

        # Introduce more NaN values
        volume_data.loc[500:550, "volume"] = np.nan
        volume_data.loc[600:650, "trade_count"] = np.nan

        print("=" * 60)
        print("NaN HANDLING FIX TEST RESULTS")
        print("=" * 60)

        print("\n📊 Original data NaN counts:")
        print(f"   Price data NaN: {price_data.isna().sum().sum()}")
        print(f"   Volume data NaN: {volume_data.isna().sum().sum()}")
        print(
            f"   Price data Inf: {np.isinf(price_data.select_dtypes(include=[np.number])).sum().sum()}",
        )

        # Test the comprehensive NaN handling function

        def _handle_nan_values_comprehensive(features):
            """Comprehensive NaN handling for all feature types."""
        if True:
            pass
    pass
pass
    pass
    pass
pass
    pass
    pass
pass
    pass
                cleaned_features = {}
                nan_count = 0
                inf_count = 0

        for feature_name , feature_value in features.items():
            pass
        if True:
            pass
    pass
pass
    pass
    pass
pass
    pass
    pass
pass
    pass
        # Handle different data types
        if isinstance(
                            feature_value = int | float | np.integer | np.floating,
                        ):
        # Scalar values
        if np.isnan(feature_value) or np.isinf(feature_value):
                                cleaned_features[feature_name] = 0.0
                                nan_count += 1
                            else:
                                cleaned_features[feature_name] = feature_value

                        elif isinstance(feature_value , pd.Series):
        # Pandas Series
                            cleaned_series = feature_value.copy()
                            nan_mask = cleaned_series.isna() | np.isinf(cleaned_series)
                            inf_mask = np.isinf(cleaned_series)

        if nan_mask.any():
                                cleaned_series = cleaned_series.fillna(0)
                                nan_count += nan_mask.sum()

        if inf_mask.any():
                                cleaned_series = cleaned_series.replace(
                                    [np.inf ,  -np.inf],
                                    0,
                                )
                                inf_count += inf_mask.sum()

                            cleaned_features[feature_name] = cleaned_series

                        elif isinstance(feature_value , np.ndarray | list):
        # Numpy arrays and lists
                            arr = np.asarray(feature_value, dtype, np.float64)
                            nan_mask = np.isnan(arr) | np.isinf(arr)
                            inf_mask = np.isinf(arr)

        if nan_mask.any():
                                arr = np.nan_to_num(
                                    arr = nan, 0.0,
                                    posinf=0.0,
                                    neginf=0.0,
                                )
                                nan_count += nan_mask.sum()

        if inf_mask.any():
                                arr = np.nan_to_num(
                                    arr = nan, 0.0,
                                    posinf=0.0,
                                    neginf=0.0,
                                )
                                inf_count += inf_mask.sum()

                            cleaned_features[feature_name] = arr

                        else:
        # Unsupported type - skip or convert to 0
                            cleaned_features[feature_name] = 0.0

        pass
                        cleaned_features[feature_name] = 0.0

        return cleaned_features, nan_count, inf_count

        pass
        return features, 0, 0

        # Create sample features with NaN values
        sample_features = {
            "scalar_nan": np.nan , "scalar_inf": np.inf,
            "series_with_nan": pd.Series([1, 2, np.nan, 4, 5]),
            "series_with_inf": pd.Series([1, 2, np.inf, 4, -np.inf]),
            "array_with_nan": np.array([1, 2, np.nan, 4, 5]),
            "array_with_inf": np.array([1, 2, np.inf, 4, -np.inf]),
            "normal_scalar": 42.0,
            "normal_series": pd.Series([1, 2, 3, 4, 5]),
            "normal_array": np.array([1, 2, 3, 4, 5]),
        }

        # Test NaN handling
        cleaned_features = nan_count, inf_count, _handle_nan_values_comprehensive(
            sample_features = )

        print("\n📊 Feature cleaning results:")
        print(f"   NaN values cleaned: {nan_count}")
        print(f"   Inf values cleaned: {inf_count}")
        print(f"   Features processed: {len(cleaned_features)}")

        # Check specific features
        print("\n📊 Feature verification:")
        print(f"   scalar_nan: {cleaned_features['scalar_nan']} (was NaN)")
        print(f"   scalar_inf: {cleaned_features['scalar_inf']} (was Inf)")
        print(
            f"   series_with_nan: {cleaned_features['series_with_nan'].isna().sum()} NaN remaining",
        )
        print(
            f"   series_with_inf: {np.isinf(cleaned_features['series_with_inf']).sum()} Inf remaining",
        )
        print(
            f"   array_with_nan: {np.isnan(cleaned_features['array_with_nan']).sum()} NaN remaining",
        )
        print(
            f"   array_with_inf: {np.isinf(cleaned_features['array_with_inf']).sum()} Inf remaining",
        )

        # Test alignment function

        def _align_time_series(series, target_length):
            """Align time series to target length with proper handling of NaN values."""
        if True:
            pass
    pass
pass
    pass
    pass
pass
    pass
    pass
pass
    pass
        if len(series) == target_length:
            pass
        return series

        # Handle NaN values first
                series = np.nan_to_num(series, nan, 0.0, posinf=0.0, neginf=0.0)

        if len(series) > target_length:
            pass
        # Truncate to target length
        return series[:target_length]
        # Pad with zeros to target length
                padding = np.zeros(target_length - len(series))
        return np.concatenate([series, padding])

        pass
        return np.zeros(target_length)

        # Test alignment
        test_series = np.array([1, 2, np.nan, 4, np.inf, 6])
        aligned_short = _align_time_series(test_series, 10)
        aligned_long = _align_time_series(test_series, 3)

        print("\n📊 Time series alignment test:")
        print(f"   Original series: {test_series}")
        print(f"   Aligned to length 10: {aligned_short}")
        print(f"   Aligned to length 3: {aligned_long}")
        print(f"   NaN in aligned_short: {np.isnan(aligned_short).sum()}")
        print(f"   Inf in aligned_short: {np.isinf(aligned_short).sum()}")

        # Check if fix is working
        nan_fixed = nan_count > 0 and inf_count > 0
        alignment_fixed = (
            np.isnan(aligned_short).sum() == 0 and np.isinf(aligned_short).sum() == 0
        )

        print("\n" + "=" * 60)
        if nan_fixed and alignment_fixed:
            print("✅ NaN FIX SUCCESSFUL!")
            print("   - NaN values properly cleaned")
            print("   - Inf values properly cleaned")
            print("   - Time series alignment working")
            print("   - No more NaN propagation in features")
        else:
            print("❌ NaN FIX FAILED!")
            print("   - Some NaN or Inf values not cleaned")
            print("   - Time series alignment issues")

        print("=" * 60)

        return nan_fixed and alignment_fixed

    pass
        logger.exception(f"Error testing NaN handling: {e}")
        return False


if __name__ == "__main__":
    success = test_nan_handling()
    sys.exit(0 if success else 1)
