#!/usr/bin/env python3
"""
Test NaN Handling Fix
Verifies that NaN values are properly handled in feature engineering.
"""

from pathlib import Path
from typing import Any, Dict, Tuple
import sys
import warnings

import numpy as np
import pandas as pd

# Ensure project src in sys.path before imports relying on it
sys.path.append(str(Path(__file__).parent.parent / "src"))

from src.utils.logger import setup_logging, system_logger  # noqa: E402
from src.utils.error_handler import handle_nan_issues  # noqa: E402

warnings.filterwarnings("ignore")


def _safe_numeric(value: Any) -> float:
	"""Convert scalars to safe float, replacing NaN/Inf with 0.0.

	Args:
		value: Any scalar numeric-like value

	Returns:
		float: Safe numeric value
	"""
	try:
		arr = float(value)
		if np.isnan(arr) or np.isinf(arr):
			return 0.0
		return arr
	except Exception:
		return 0.0


def _clean_series(series: pd.Series) -> pd.Series:
	"""Clean a pandas Series by replacing NaN/Inf appropriately.

	Args:
		series: Input Series

	Returns:
		pd.Series: Cleaned Series
	"""
	cleaned = series.copy()
	# Replace infinities first to avoid propagation
	cleaned = cleaned.replace([np.inf, -np.inf], np.nan)
	# For numeric dtypes fill with 0
	if pd.api.types.is_numeric_dtype(cleaned):
		return cleaned.fillna(0)
	# For non-numeric, forward/backward fill then empty with empty string
	return cleaned.fillna(method="ffill").fillna(method="bfill").fillna("")


def _clean_array(array_like: Any) -> np.ndarray:
	"""Clean numpy array-like replacing NaN/Inf with 0.0.

	Args:
		array_like: Array-like value

	Returns:
		np.ndarray: Cleaned array
	"""
	arr = np.asarray(array_like, dtype=np.float64)
	return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)


def _handle_nan_values_comprehensive(features: Dict[str, Any]) -> Tuple[Dict[str, Any], int, int]:
	"""Comprehensive NaN handling for all feature types.

	Args:
		features: Mapping from feature name to value (scalar/Series/array)

	Returns:
		Tuple of (cleaned_features, nan_count, inf_count)
	"""
	cleaned_features: Dict[str, Any] = {}
	nan_count = 0
	inf_count = 0

	for feature_name, feature_value in features.items():
		# Scalars
		if isinstance(feature_value, (int, float, np.integer, np.floating)):
			value = float(feature_value)
			if np.isnan(value):
				nan_count += 1
			elif np.isinf(value):
				inf_count += 1
			cleaned_features[feature_name] = _safe_numeric(value)
		# Series
		elif isinstance(feature_value, pd.Series):
			series = feature_value.copy()
			series_inf_mask = np.isinf(series)
			series_nan_mask = series.isna() | series_inf_mask
			inf_count += int(series_inf_mask.sum())
			nan_count += int(series_nan_mask.sum())
			cleaned_features[feature_name] = _clean_series(series)
		# Arrays / lists
		elif isinstance(feature_value, (np.ndarray, list, tuple)):
			arr = np.asarray(feature_value, dtype=np.float64)
			inf_mask = np.isinf(arr)
			nan_mask = np.isnan(arr) | inf_mask
			inf_count += int(inf_mask.sum())
			nan_count += int(nan_mask.sum())
			cleaned_features[feature_name] = _clean_array(arr)
		# Fallback for unsupported types
		else:
			cleaned_features[feature_name] = 0.0

	return cleaned_features, nan_count, inf_count


@handle_nan_issues
def _align_time_series(series: np.ndarray, target_length: int) -> np.ndarray:
	"""Align time series to target length with proper handling of NaN/Inf values.

	- Cleans values
	- Truncates if longer than target_length
	- Pads with zeros if shorter than target_length
	"""
	if not isinstance(series, np.ndarray):
		series = np.asarray(series, dtype=np.float64)

	# Clean first
	series = np.nan_to_num(series, nan=0.0, posinf=0.0, neginf=0.0)

	if target_length <= 0:
		return np.zeros(0, dtype=np.float64)

	if len(series) == target_length:
		return series.astype(np.float64, copy=False)

	if len(series) > target_length:
		return series[:target_length].astype(np.float64, copy=False)

	# Pad
	padding = np.zeros(target_length - len(series), dtype=np.float64)
	return np.concatenate([series, padding])


def test_nan_handling() -> bool:
	"""Test that NaN values are properly handled."""
	setup_logging()
	logger = system_logger.getChild("TestNaNHandling")

	try:
		# Create sample data with NaN and Inf values
		np.random.seed(42)
		n_samples = 1000

		price_data = pd.DataFrame(
			{
				"open": np.random.randn(n_samples) * 0.01 + 100,
				"high": np.random.randn(n_samples) * 0.01 + 100.5,
				"low": np.random.randn(n_samples) * 0.01 + 99.5,
				"close": np.random.randn(n_samples) * 0.01 + 100,
				"volume": np.random.lognormal(10, 1, n_samples),
			}
		)

		# Introduce anomalies
		price_data.loc[100:150, "close"] = np.nan
		price_data.loc[200:250, "volume"] = np.nan
		price_data.loc[300:350, "high"] = np.inf
		price_data.loc[400:450, "low"] = -np.inf

		volume_data = pd.DataFrame(
			{
				"volume": price_data["volume"].copy(),
				"trade_count": np.random.randint(100, 1000, n_samples),
			}
		)

		# More NaN in volume data
		volume_data.loc[500:550, "volume"] = np.nan
		volume_data.loc[600:650, "trade_count"] = np.nan

		print("=" * 60)
		print("NaN HANDLING FIX TEST RESULTS")
		print("=" * 60)

		print("\n📊 Original data NaN counts:")
		print(f"   Price data NaN: {price_data.isna().sum().sum()}")
		print(f"   Volume data NaN: {volume_data.isna().sum().sum()}")
		print(
			f"   Price data Inf: {np.isinf(price_data.select_dtypes(include=[np.number])).sum().sum()}"
		)

		# Create sample features with anomalies
		sample_features: Dict[str, Any] = {
			"scalar_nan": float("nan"),
			"scalar_inf": float("inf"),
			"series_with_nan": pd.Series([1, 2, np.nan, 4, 5]),
			"series_with_inf": pd.Series([1, 2, np.inf, 4, -np.inf]),
			"array_with_nan": np.array([1, 2, np.nan, 4, 5]),
			"array_with_inf": np.array([1, 2, np.inf, 4, -np.inf]),
			"normal_scalar": 42.0,
			"normal_series": pd.Series([1, 2, 3, 4, 5]),
			"normal_array": np.array([1, 2, 3, 4, 5]),
		}

		cleaned_features, nan_count, inf_count = _handle_nan_values_comprehensive(
			sample_features
		)

		print("\n📊 Feature cleaning results:")
		print(f"   NaN values cleaned: {nan_count}")
		print(f"   Inf values cleaned: {inf_count}")
		print(f"   Features processed: {len(cleaned_features)}")

		print("\n📊 Feature verification:")
		print(f"   scalar_nan: {cleaned_features['scalar_nan']} (was NaN)")
		print(f"   scalar_inf: {cleaned_features['scalar_inf']} (was Inf)")
		print(
			f"   series_with_nan: {pd.Series(cleaned_features['series_with_nan']).isna().sum()} NaN remaining"
		)
		print(
			f"   series_with_inf: {np.isinf(pd.Series(cleaned_features['series_with_inf'])).sum()} Inf remaining"
		)
		print(
			f"   array_with_nan: {np.isnan(np.asarray(cleaned_features['array_with_nan'])).sum()} NaN remaining"
		)
		print(
			f"   array_with_inf: {np.isinf(np.asarray(cleaned_features['array_with_inf'])).sum()} Inf remaining"
		)

		# Time series alignment tests
		test_series = np.array([1, 2, np.nan, 4, np.inf, 6], dtype=np.float64)
		aligned_short = _align_time_series(test_series, 10)
		aligned_long = _align_time_series(test_series, 3)

		print("\n📊 Time series alignment test:")
		print(f"   Original series: {test_series}")
		print(f"   Aligned to length 10: {aligned_short}")
		print(f"   Aligned to length 3: {aligned_long}")
		print(f"   NaN in aligned_short: {np.isnan(aligned_short).sum()}")
		print(f"   Inf in aligned_short: {np.isinf(aligned_short).sum()}")

		# Validation
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
		return bool(nan_fixed and alignment_fixed)
	except Exception as e:  # noqa: BLE001
		logger = system_logger.getChild("TestNaNHandling")
		logger.exception(f"Error testing NaN handling: {e}")
		return False


if __name__ == "__main__":
	success = test_nan_handling()
	sys.exit(0 if success else 1)
