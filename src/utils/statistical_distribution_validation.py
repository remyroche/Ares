from __future__ import annotations

"""
Statistical Distribution Validation Module

This module provides comprehensive statistical validation for time series data,
including distribution checks, outlier detection, and stationarity tests.
"""

import warnings
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import anderson, jarque_bera, kstest, normaltest, shapiro
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import adfuller, kpss

from src.utils.logger import system_logger
from src.utils.pipeline_standards import (
    DataQualityLevel,
    ValidationIssue,
    ValidationResult,
)


class StatisticalValidator:
    """Validates statistical properties of time series data."""

    def __init__(self, logger=None):
        self.logger = logger or system_logger.getChild("StatisticalValidator")

        # Configuration for various tests
        self.outlier_methods = [
            "iqr",
            "zscore",
            "isolation_forest",
            "local_outlier_factor",
        ]
        self.distribution_tests = ["normality", "stationarity", "autocorrelation"]
        self.stationarity_significance = 0.05

    def validate_distribution(
        self,
        df: pd.DataFrame,
        columns: list[str] | None = None,
        expected_distribution: str | None = None,
        outlier_threshold: float = 3.0,
        check_stationarity: bool = True,
    ) -> ValidationResult:
        """
        Comprehensive distribution validation for specified columns.

        Args:
            df: DataFrame to validate
            columns: Columns to check (None = check all numeric)
            expected_distribution: Expected distribution type
            outlier_threshold: Threshold for outlier detection
            check_stationarity: Whether to check for stationarity

        Returns:
            ValidationResult with distribution findings
        """
        self.logger.info("📊 Starting statistical distribution validation")

        result = ValidationResult(passed=True)

        if df is None or df.empty:
            result.passed = False
            result.issues.append(
                ValidationIssue(
                    severity=DataQualityLevel.CRITICAL,
                    message="DataFrame is None or empty",
                )
            )
            return result

        # Determine columns to validate
        if columns is None:
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            # Exclude timestamp-like columns
            columns = [
                col
                for col in numeric_cols
                if "time" not in col.lower() and col != "timestamp"
            ]

        validation_summary = {}

        for column in columns:
            if column not in df.columns:
                result.warnings.append(
                    ValidationIssue(
                        severity=DataQualityLevel.WARNING,
                        message=f"Column '{column}' not found in DataFrame",
                    )
                )
                continue

            col_data = df[column].dropna()

            if len(col_data) < 30:  # Minimum sample size for meaningful statistics
                result.warnings.append(
                    ValidationIssue(
                        severity=DataQualityLevel.WARNING,
                        message=f"Column '{column}' has insufficient data for statistical analysis",
                        column=column,
                        details={"sample_size": len(col_data)},
                    )
                )
                continue

            col_validation = {
                "sample_size": len(col_data),
                "null_count": df[column].isnull().sum(),
                "basic_stats": self._calculate_basic_stats(col_data),
                "distribution_tests": {},
                "outlier_analysis": {},
                "issues": [],
            }

            # 1. Basic distribution shape analysis
            shape_analysis = self._analyze_distribution_shape(col_data)
            col_validation["distribution_shape"] = shape_analysis

            if shape_analysis["is_heavily_skewed"]:
                result.warnings.append(
                    ValidationIssue(
                        severity=DataQualityLevel.WARNING,
                        message=f"Column '{column}' shows heavy skewness",
                        column=column,
                        details=shape_analysis,
                    )
                )

            # 2. Normality tests
            if expected_distribution == "normal" or expected_distribution is None:
                normality_results = self._test_normality(col_data)
                col_validation["distribution_tests"]["normality"] = normality_results

                if not normality_results["is_normal"]:
                    severity = (
                        DataQualityLevel.WARNING
                        if expected_distribution is None
                        else DataQualityLevel.CRITICAL
                    )
                    result.issues.append(
                        ValidationIssue(
                            severity=severity,
                            message=f"Column '{column}' fails normality tests",
                            column=column,
                            details=normality_results,
                        )
                    )
                    if severity == DataQualityLevel.CRITICAL:
                        result.passed = False

            # 3. Outlier detection
            outlier_results = self._detect_outliers(
                col_data, method="iqr", threshold=outlier_threshold
            )
            col_validation["outlier_analysis"] = outlier_results

            if outlier_results["outlier_percentage"] > 5.0:  # More than 5% outliers
                result.warnings.append(
                    ValidationIssue(
                        severity=DataQualityLevel.WARNING,
                        message=f"Column '{column}' has high outlier percentage",
                        column=column,
                        details=outlier_results,
                    )
                )

            # 4. Stationarity tests (for time series)
            if check_stationarity and self._is_time_series_candidate(df, column):
                stationarity_results = self._test_stationarity(col_data)
                col_validation["distribution_tests"][
                    "stationarity"
                ] = stationarity_results

                if not stationarity_results["is_stationary"]:
                    result.warnings.append(
                        ValidationIssue(
                            severity=DataQualityLevel.WARNING,
                            message=f"Column '{column}' appears non-stationary",
                            column=column,
                            details=stationarity_results,
                        )
                    )

            # 5. Autocorrelation check
            autocorr_results = self._test_autocorrelation(col_data)
            col_validation["distribution_tests"]["autocorrelation"] = autocorr_results

            if autocorr_results["has_significant_autocorrelation"]:
                result.info.append(
                    ValidationIssue(
                        severity=DataQualityLevel.INFO,
                        message=f"Column '{column}' shows significant autocorrelation",
                        column=column,
                        details=autocorr_results,
                    )
                )

            # 6. Distribution shift detection (if historical data available)
            if hasattr(self, "historical_stats") and column in self.historical_stats:
                shift_results = self._detect_distribution_shift(
                    col_data,
                    self.historical_stats[column],
                )
                if shift_results["has_shifted"]:
                    result.warnings.append(
                        ValidationIssue(
                            severity=DataQualityLevel.WARNING,
                            message=f"Column '{column}' shows distribution shift",
                            column=column,
                            details=shift_results,
                        )
                    )

            validation_summary[column] = col_validation

        # Calculate overall quality score
        len(validation_summary) * 5  # 5 tests per column
        issues_count = len(
            [i for i in result.issues if i.severity == DataQualityLevel.CRITICAL]
        )
        warnings_count = len(result.warnings)

        result.quality_score = max(0, 1 - (issues_count * 0.2 + warnings_count * 0.05))
        result.metadata["validation_summary"] = validation_summary

        return result

    def _calculate_basic_stats(self, data: pd.Series) -> dict[str, float]:
        """Calculate basic statistical measures."""
        return {
            "mean": float(data.mean()),
            "median": float(data.median()),
            "std": float(data.std()),
            "variance": float(data.var()),
            "skewness": float(data.skew()),
            "kurtosis": float(data.kurtosis()),
            "min": float(data.min()),
            "max": float(data.max()),
            "q1": float(data.quantile(0.25)),
            "q3": float(data.quantile(0.75)),
            "iqr": float(data.quantile(0.75) - data.quantile(0.25)),
        }

    def _analyze_distribution_shape(self, data: pd.Series) -> dict[str, Any]:
        """Analyze the shape of the distribution."""
        skewness = float(data.skew())
        kurtosis = float(data.kurtosis())

        return {
            "skewness": skewness,
            "kurtosis": kurtosis,
            "is_symmetric": abs(skewness) < 0.5,
            "is_heavily_skewed": abs(skewness) > 2.0,
            "is_mesokurtic": abs(kurtosis) < 1.0,  # Normal-like
            "is_leptokurtic": kurtosis > 1.0,  # Heavy-tailed
            "is_platykurtic": kurtosis < -1.0,  # Light-tailed
            "distribution_type": self._classify_distribution(skewness, kurtosis),
        }

    def _classify_distribution(self, skewness: float, kurtosis: float) -> str:
        """Classify distribution based on moments."""
        if abs(skewness) < 0.5 and abs(kurtosis) < 1.0:
            return "approximately_normal"
        if skewness > 2.0:
            return "highly_right_skewed"
        if skewness < -2.0:
            return "highly_left_skewed"
        if kurtosis > 3.0:
            return "heavy_tailed"
        if kurtosis < -1.0:
            return "light_tailed"
        return "moderately_non_normal"

    def _test_normality(self, data: pd.Series) -> dict[str, Any]:
        """Perform multiple normality tests."""
        results = {
            "is_normal": True,
            "tests": {},
        }

        # Shapiro-Wilk test (best for small samples)
        if len(data) <= 5000:
            stat, p_value = shapiro(data)
            results["tests"]["shapiro_wilk"] = {
                "statistic": float(stat),
                "p_value": float(p_value),
                "is_normal": p_value > 0.05,
            }
            results["is_normal"] &= p_value > 0.05

        # Jarque-Bera test (based on skewness and kurtosis)
        stat, p_value = jarque_bera(data)
        results["tests"]["jarque_bera"] = {
            "statistic": float(stat),
            "p_value": float(p_value),
            "is_normal": p_value > 0.05,
        }
        results["is_normal"] &= p_value > 0.05

        # D'Agostino's K-squared test
        if len(data) >= 20:
            stat, p_value = normaltest(data)
            results["tests"]["dagostino"] = {
                "statistic": float(stat),
                "p_value": float(p_value),
                "is_normal": p_value > 0.05,
            }
            results["is_normal"] &= p_value > 0.05

        # Anderson-Darling test
        result = anderson(data)
        results["tests"]["anderson_darling"] = {
            "statistic": float(result.statistic),
            "critical_values": dict(
                zip(result.significance_level, result.critical_values, strict=False)
            ),
            "is_normal": result.statistic
            < result.critical_values[2],  # 5% significance
        }
        results["is_normal"] &= result.statistic < result.critical_values[2]

        return results

    def _detect_outliers(
        self,
        data: pd.Series,
        method: str = "iqr",
        threshold: float = 3.0,
    ) -> dict[str, Any]:
        """Detect outliers using specified method."""
        if method == "iqr":
            q1 = data.quantile(0.25)
            q3 = data.quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            outliers = (data < lower_bound) | (data > upper_bound)

        elif method == "zscore":
            z_scores = np.abs(stats.zscore(data))
            outliers = z_scores > threshold

        else:
            # Default to z-score
            z_scores = np.abs(stats.zscore(data))
            outliers = z_scores > threshold

        outlier_indices = np.where(outliers)[0]
        outlier_values = data.iloc[outlier_indices].values

        return {
            "method": method,
            "outlier_count": int(outliers.sum()),
            "outlier_percentage": float(outliers.sum() / len(data) * 100),
            "outlier_indices": outlier_indices.tolist()[:10],  # First 10
            "outlier_values": outlier_values.tolist()[:10],  # First 10
            "lower_bound": float(lower_bound) if method == "iqr" else None,
            "upper_bound": float(upper_bound) if method == "iqr" else None,
            "threshold": threshold if method == "zscore" else None,
        }

    def _is_time_series_candidate(self, df: pd.DataFrame, column: str) -> bool:
        """Check if column is likely a time series."""
        # Check if dataframe has a timestamp column and data is ordered
        has_timestamp = any(
            "time" in col.lower() or col == "timestamp" for col in df.columns
        )

        # Check if values show temporal patterns
        if has_timestamp and len(df) > 100:
            # Simple check: look for trend or seasonality
            rolling_mean = df[column].rolling(window=10).mean()
            rolling_mean.iloc[-1] != rolling_mean.iloc[len(rolling_mean) // 2]
            return True  # Assume time series if we have timestamp

        return False

    def _test_stationarity(self, data: pd.Series) -> dict[str, Any]:
        """Test for stationarity using ADF and KPSS tests."""
        results = {
            "is_stationary": True,
            "tests": {},
        }

        # Augmented Dickey-Fuller test (null: non-stationary)
        try:
            adf_result = adfuller(data, autolag="AIC")
            results["tests"]["adf"] = {
                "statistic": float(adf_result[0]),
                "p_value": float(adf_result[1]),
                "critical_values": adf_result[4],
                "is_stationary": adf_result[1] < self.stationarity_significance,
            }
            results["is_stationary"] &= adf_result[1] < self.stationarity_significance
        except Exception as e:
            self.logger.debug(f"ADF test failed: {e}")

        # KPSS test (null: stationary)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                kpss_result = kpss(data, regression="c", nlags="auto")
            results["tests"]["kpss"] = {
                "statistic": float(kpss_result[0]),
                "p_value": float(kpss_result[1]),
                "critical_values": kpss_result[3],
                "is_stationary": kpss_result[1] > self.stationarity_significance,
            }
            results["is_stationary"] &= kpss_result[1] > self.stationarity_significance
        except Exception as e:
            self.logger.debug(f"KPSS test failed: {e}")

        return results

    def _test_autocorrelation(self, data: pd.Series, lags: int = 10) -> dict[str, Any]:
        """Test for autocorrelation in the data."""
        try:
            # Ljung-Box test
            lb_result = acorr_ljungbox(data, lags=lags, return_df=True)

            significant_lags = lb_result[lb_result["lb_pvalue"] < 0.05]

            return {
                "has_significant_autocorrelation": len(significant_lags) > 0,
                "significant_lag_count": len(significant_lags),
                "first_significant_lag": (
                    int(significant_lags.index[0])
                    if len(significant_lags) > 0
                    else None
                ),
                "max_correlation": float(lb_result["lb_stat"].max()),
                "test_type": "ljung_box",
            }
        except Exception as e:
            self.logger.debug(f"Autocorrelation test failed: {e}")
            return {
                "has_significant_autocorrelation": False,
                "error": str(e),
            }

    def _detect_distribution_shift(
        self,
        current_data: pd.Series,
        historical_stats: dict[str, float],
    ) -> dict[str, Any]:
        """Detect if distribution has shifted from historical baseline."""
        current_stats = self._calculate_basic_stats(current_data)

        # Calculate relative changes
        mean_shift = abs(current_stats["mean"] - historical_stats["mean"]) / (
            abs(historical_stats["mean"]) + 1e-10
        )
        std_shift = abs(current_stats["std"] - historical_stats["std"]) / (
            abs(historical_stats["std"]) + 1e-10
        )

        # Kolmogorov-Smirnov test against normal distribution with historical parameters
        ks_stat, ks_pvalue = kstest(
            current_data,
            lambda x: stats.norm.cdf(
                x, loc=historical_stats["mean"], scale=historical_stats["std"]
            ),
        )

        has_shifted = mean_shift > 0.1 or std_shift > 0.2 or ks_pvalue < 0.05

        return {
            "has_shifted": has_shifted,
            "mean_shift_percentage": float(mean_shift * 100),
            "std_shift_percentage": float(std_shift * 100),
            "ks_statistic": float(ks_stat),
            "ks_pvalue": float(ks_pvalue),
            "current_stats": current_stats,
            "historical_stats": historical_stats,
        }

    def validate_correlation_stability(
        self,
        df: pd.DataFrame,
        columns: list[str],
        window_size: int = 100,
        stability_threshold: float = 0.2,
    ) -> ValidationResult:
        """Validate that correlations between columns remain stable over time."""
        result = ValidationResult(passed=True)

        if len(columns) < 2:
            result.warnings.append(
                ValidationIssue(
                    severity=DataQualityLevel.WARNING,
                    message="Need at least 2 columns for correlation analysis",
                )
            )
            return result

        # Calculate rolling correlations
        correlation_changes = {}

        for i in range(len(columns)):
            for j in range(i + 1, len(columns)):
                col1, col2 = columns[i], columns[j]

                if col1 not in df.columns or col2 not in df.columns:
                    continue

                # Calculate rolling correlation
                rolling_corr = df[col1].rolling(window=window_size).corr(df[col2])

                if rolling_corr.isna().all():
                    continue

                # Check stability
                corr_std = rolling_corr.std()
                corr_range = rolling_corr.max() - rolling_corr.min()

                if (
                    corr_std > stability_threshold
                    or corr_range > stability_threshold * 2
                ):
                    result.warnings.append(
                        ValidationIssue(
                            severity=DataQualityLevel.WARNING,
                            message=f"Unstable correlation between {col1} and {col2}",
                            details={
                                "correlation_std": float(corr_std),
                                "correlation_range": float(corr_range),
                                "mean_correlation": float(rolling_corr.mean()),
                            },
                        )
                    )

                correlation_changes[f"{col1}_vs_{col2}"] = {
                    "stable": corr_std <= stability_threshold,
                    "std": float(corr_std),
                    "range": float(corr_range),
                }

        result.metadata["correlation_analysis"] = correlation_changes
        result.quality_score = sum(
            1 for v in correlation_changes.values() if v["stable"]
        ) / max(len(correlation_changes), 1)

        return result
