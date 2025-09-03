# src/training/steps/fractional_differentiation.py
"""Fractional Differentiation for enhanced feature engineering.

Implements fractional-order differentiation to preserve memory and maintain stationarity
while avoiding over-differencing.
"""

import copy
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.tsa.stattools import adfuller

from src.core.decorators import handles_errors
from src.utils.centralized_decorators import handle_errors, with_tracing_span
from src.utils.logger import get_logger


class FractionalDifferentiation:
    """Fractional differentiation for enhanced feature engineering.

    Replaces integer-order differentiation with fractional-order differentiation
    to preserve memory and maintain stationarity while avoiding over-differencing.

    Key benefits:
    - Preserves long-term memory better than integer differentiation
    - Maintains stationarity without over-differencing
    - Captures persistent trends more effectively
    - Reduces feature multicollinearity
    """

    def __init__(
        self,
        d: float = 0.5,
        threshold: float = 1e-5,
        window: int = 100,
        optimize_order: bool = True,
    ):
        """Initialize fractional differentiation.

        Args:
            d: Fractional order (0 < d < 1)
            threshold: Minimum value threshold for stationarity
            window: Memory window for computation
            optimize_order: Whether to automatically optimize fractional order
        """
        self.d = d
        self.threshold = threshold
        self.window = window
        self.optimize_order = optimize_order
        self.weights = self._get_fractional_weights(window)
        self.logger = get_logger("FractionalDifferentiation")

    def _get_fractional_weights(self, window: int) -> np.ndarray:
        """Generate fractional differentiation weights using binomial expansion.

        The weights follow the expansion of (1-L)^d where L is the lag operator.
        """
        weights = np.zeros(window)
        weights[0] = -self.d
        for k in range(1, window):
            weights[k] = weights[k - 1] * (k - 1 - self.d) / k
        return weights

    def fractional_diff(
        self, series: pd.Series, preserve_original: bool = True
    ) -> pd.Series:
        """Apply fractional differentiation to time series.

        Args:
            series: Input time series
            preserve_original: Whether to preserve original series name

        Returns:
            Fractionally differentiated series
        """
        if len(series) < self.window:
            # Fallback to simple differentiation for short series
            self.logger.warning(
                f"Series too short for fractional diff, using simple diff: {len(series)} < {self.window}"
            )
            return series.diff().fillna(0)

        # Apply fractional differentiation
        result = np.zeros(len(series))
        series_array = series.values

        for i in range(self.window, len(series)):
            result[i] = np.sum(self.weights * series_array[i - self.window : i])

        # Check for stationarity
        if np.std(result[self.window :]) < self.threshold:
            # Series is already stationary, return as is
            self.logger.info(
                f"Series {series.name} already stationary after fractional diff"
            )
            return pd.Series(
                result, index=series.index, name=f"{series.name}_frac_diff_{self.d}"
            )

        return pd.Series(
            result, index=series.index, name=f"{series.name}_frac_diff_{self.d}"
        )

    def optimize_fractional_order(
        self, series: pd.Series, max_d: float = 0.9, min_d: float = 0.1, steps: int = 10
    ) -> float:
        """Optimize fractional order for stationarity using ADF test.

        Args:
            series: Input time series
            max_d: Maximum fractional order to test
            min_d: Minimum fractional order to test
            steps: Number of steps to test

        Returns:
            Optimal fractional order
        """
        best_d = min_d
        best_pvalue = 1.0
        best_adf_stat = 0

        self.logger.info(f"Optimizing fractional order for series {series.name}")

        for d in np.linspace(min_d, max_d, steps):
            temp_diff = FractionalDifferentiation(
                d=d, window=self.window, optimize_order=False
            )
            diff_series = temp_diff.fractional_diff(series)

            # Remove NaN values for ADF test
            clean_series = diff_series.dropna()
            if len(clean_series) < 10:
                continue

            try:
                adf_result = adfuller(clean_series)
                pvalue = adf_result[1]
                adf_stat = adf_result[0]

                # Prefer lower p-value and more negative ADF statistic
                if pvalue < best_pvalue and adf_stat < best_adf_stat:
                    best_pvalue = pvalue
                    best_adf_stat = adf_stat
                    best_d = d
            except Exception as e:
                self.logger.warning(f"ADF test failed for d={d}: {e}")
                continue

        self.logger.info(
            f"Optimal fractional order for {series.name}: d={best_d:.3f} (p-value={best_pvalue:.4f})"
        )
        return best_d

    def apply_with_optimization(self, series: pd.Series) -> Tuple[pd.Series, float]:
        """Apply fractional differentiation with automatic order optimization.

        Args:
            series: Input time series

        Returns:
            Tuple of (differentiated_series, optimal_order)
        """
        if self.optimize_order:
            optimal_d = self.optimize_fractional_order(series)
            self.d = optimal_d
            self.weights = self._get_fractional_weights(self.window)

        result = self.fractional_diff(series)
        return result, self.d

    def batch_fractional_diff(
        self,
        data: pd.DataFrame,
        columns: Optional[list[str]] = None,
        exclude_columns: Optional[list[str]] = None,
    ) -> pd.DataFrame:
        """Apply fractional differentiation to multiple columns.

        Args:
            data: Input DataFrame
            columns: Columns to differentiate (if None, use all numeric columns)
            exclude_columns: Columns to exclude from differentiation

        Returns:
            DataFrame with additional fractional differentiation features
        """
        if columns is None:
            columns = data.select_dtypes(include=[np.number]).columns.tolist()

        if exclude_columns:
            columns = [col for col in columns if col not in exclude_columns]

        result_data = data.copy()
        optimization_results = {}

        for col in columns:
            if col in data.columns:
                try:
                    diff_series, optimal_d = self.apply_with_optimization(data[col])
                    result_data[f"{col}_frac_diff_{optimal_d:.3f}"] = diff_series
                    optimization_results[col] = optimal_d
                except Exception as e:
                    self.logger.error(f"Failed to apply fractional diff to {col}: {e}")

        self.logger.info(
            f"Applied fractional differentiation to {len(optimization_results)} columns"
        )
        return result_data, optimization_results


class FractionalFeatureGenerator:
    """High-level interface for generating fractional differentiation features."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize fractional feature generator.

        Args:
            config: Configuration dictionary
        """
        self.config = config or {
            "enable_fractional_diff": True,
            "default_d": 0.5,
            "optimize_order": True,
            "window": 100,
            "threshold": 1e-5,
            "price_columns": ["close", "high", "low", "open"],
            "volume_columns": ["volume"],
            "exclude_columns": ["timestamp", "datetime", "date"],
        }

        self.fractional_diff = FractionalDifferentiation(
            d=self.config["default_d"],
            threshold=self.config["threshold"],
            window=self.config["window"],
            optimize_order=self.config["optimize_order"],
        )

        self.logger = get_logger("FractionalFeatureGenerator")

    @handle_errors(
        exceptions=(Exception,),
        default_return=pd.DataFrame(),
        context="fractional_feature_generator.generate_features",
    )
    @with_tracing_span("FractionalFeatureGenerator.generate_features", log_args=False)
    def generate_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate fractional differentiation features.

        Args:
            data: Input DataFrame with OHLCV data

        Returns:
            DataFrame with additional fractional differentiation features
        """
        if not self.config["enable_fractional_diff"]:
            return data

        self.logger.info("Generating fractional differentiation features")

        # Apply to price columns
        price_columns = [
            col for col in self.config["price_columns"] if col in data.columns
        ]
        if price_columns:
            result_data, price_results = self.fractional_diff.batch_fractional_diff(
                data, columns=price_columns
            )
        else:
            result_data = data.copy()
            price_results = {}

        # Apply to volume columns
        volume_columns = [
            col for col in self.config["volume_columns"] if col in data.columns
        ]
        if volume_columns:
            result_data, volume_results = self.fractional_diff.batch_fractional_diff(
                result_data, columns=volume_columns
            )
        else:
            volume_results = {}

        # Log results
        total_features = len(price_results) + len(volume_results)
        self.logger.info(
            f"Generated {total_features} fractional differentiation features"
        )

        return result_data

    def get_feature_statistics(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Get statistics about fractional differentiation features."""
        frac_diff_columns = [col for col in data.columns if "frac_diff" in col]

        stats = {
            "total_frac_diff_features": len(frac_diff_columns),
            "frac_diff_columns": frac_diff_columns,
            "feature_statistics": {},
        }

        for col in frac_diff_columns:
            stats["feature_statistics"][col] = {
                "mean": data[col].mean(),
                "std": data[col].std(),
                "min": data[col].min(),
                "max": data[col].max(),
                "null_count": data[col].isnull().sum(),
            }

        return stats
