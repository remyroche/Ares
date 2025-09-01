"""
Advanced ML Data Quality Validation System

This module provides comprehensive data quality validation specifically designed
for machine learning training, including statistical analysis, drift detection,
feature correlation analysis, and quality scoring.
"""

import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from collections import defaultdict
import logging

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Add project root to path
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.utils.logger import system_logger
from src.utils.pipeline_standards import PipelineStandards, pipeline_standards
from src.utils.comprehensive_file_validation import (
    ValidationSeverity,
    ValidationIssue,
    FileValidationResult
)


@dataclass
class QualityScore:
    """Represents a data quality score with components."""
    overall: float
    components: Dict[str, float]
    grade: str
    timestamp: datetime = field(default_factory=datetime.now)
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DriftReport:
    """Represents a data drift detection report."""
    issues: List[str]
    drift_metrics: Dict[str, float]
    severity: ValidationSeverity
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class MLValidationResult:
    """Result of ML-specific validation."""
    is_valid: bool
    quality_score: QualityScore
    drift_report: Optional[DriftReport] = None
    correlation_issues: List[str] = field(default_factory=list)
    target_issues: List[str] = field(default_factory=list)
    distribution_issues: List[str] = field(default_factory=list)
    outlier_issues: List[str] = field(default_factory=list)
    time_series_issues: List[str] = field(default_factory=list)
    financial_issues: List[str] = field(default_factory=list)
    summary: Dict[str, Any] = field(default_factory=dict)
    validation_timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class Alert:
    """Represents a quality alert."""
    level: str
    message: str
    timestamp: datetime
    action_required: bool
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AlertConfig:
    """Configuration for alert system."""
    slack_webhook: Optional[str] = None
    email_config: Optional[Dict[str, Any]] = None
    webhook_url: Optional[str] = None
    alert_thresholds: Dict[str, float] = field(default_factory=dict)


class StatisticalDataValidator:
    """Validates data using statistical methods."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or self._get_default_config()
        self.logger = system_logger.getChild("StatisticalDataValidator")

    def _get_default_config(self) -> Dict[str, Any]:
        return {
            "distribution_tolerance": 0.1,
            "outlier_threshold": 3.0,
            "outlier_ratio_threshold": 0.05,
            "correlation_threshold": 0.95,
            "drift_psi_threshold": 0.25,
            "class_imbalance_threshold": 0.1,
            "target_leakage_threshold": 0.9
        }

    def validate_data_distributions(
        self,
        df: pd.DataFrame,
        expected_distributions: Optional[Dict[str, Dict[str, float]]] = None
    ) -> List[str]:
        """Validate data distributions match expected patterns."""
        issues = []

        if expected_distributions is None:
        # Fallback implementation for expected_distributions
            # Use sample statistics if no expected distributions provided
            expected_distributions = self._compute_reference_distributions(df)

        for column, expected_dist in expected_distributions.items():
            if column in df.columns:
                actual_dist = df[column].describe()

                # Check for distribution shifts in mean
                if 'mean' in expected_dist:
                    mean_diff = abs(actual_dist['mean'] - expected_dist['mean'])
                    mean_tolerance = expected_dist.get('mean_tolerance', self.config['distribution_tolerance'])
                    if mean_diff > mean_tolerance:
                        issues.append(
                            f"Mean shift in {column}: {mean_diff:.3f} "
                            f"(expected: {expected_dist['mean']:.3f}, actual: {actual_dist['mean']:.3f})"
                        )

                # Check for variance changes
                if 'std' in expected_dist:
                    std_ratio = actual_dist['std'] / expected_dist['std']
                    if not (0.8 <= std_ratio <= 1.2):
                        issues.append(
                            f"Variance change in {column}: {std_ratio:.3f} "
                            f"(expected: {expected_dist['std']:.3f}, actual: {actual_dist['std']:.3f})"
                        )

                # Check for skewness changes
                if 'skew' in expected_dist:
                    actual_skew = df[column].skew()
                    skew_diff = abs(actual_skew - expected_dist['skew'])
                    if skew_diff > 0.5:
                        issues.append(
                            f"Skewness change in {column}: {skew_diff:.3f} "
                            f"(expected: {expected_dist['skew']:.3f}, actual: {actual_skew:.3f})"
                        )

        return issues

    def _compute_reference_distributions(self, df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """Compute reference distributions from the data."""
        distributions = {}

        for column in df.select_dtypes(include=[np.number]).columns:
            distributions[column] = {
                'mean': df[column].mean(),
                'std': df[column].std(),
                'skew': df[column].skew(),
                'mean_tolerance': self.config['distribution_tolerance']
            }

        return distributions

    def validate_outliers(self, df: pd.DataFrame) -> List[str]:
        """Detect and validate outliers using IQR and Z-score methods."""
        issues = []

        for column in df.select_dtypes(include=[np.number]).columns:
            # IQR method
            Q1 = df[column].quantile(0.25)
            Q3 = df[column].quantile(0.75)
            IQR = Q3 - Q1

            outlier_count = len(df[
                (df[column] < Q1 - 1.5*IQR) |
                (df[column] > Q3 + 1.5*IQR)
            ])

            outlier_ratio = outlier_count / len(df)
            if outlier_ratio > self.config['outlier_ratio_threshold']:
                issues.append(
                    f"High outlier ratio in {column}: {outlier_ratio:.2%} "
                    f"({outlier_count} outliers, threshold: {self.config['outlier_ratio_threshold']:.2%})"
                )

            # Z-score method for extreme outliers
            z_scores = np.abs(stats.zscore(df[column].dropna()))
            extreme_outliers = len(z_scores[z_scores > self.config['outlier_threshold']])

            if extreme_outliers > 0:
                issues.append(
                    f"Extreme outliers in {column}: {extreme_outliers} "
                    f"(Z-score > {self.config['outlier_threshold']})"
                )

        return issues


class TimeSeriesValidator:
    """Validates time series data quality."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or self._get_default_config()
        self.logger = system_logger.getChild("TimeSeriesValidator")

    def _get_default_config(self) -> Dict[str, Any]:
        return {
            "max_gap_multiplier": 2.0,
            "max_duplicate_ratio": 0.01,
            "future_tolerance_minutes": 5
        }

    def validate_time_series_quality(
        self,
        df: pd.DataFrame,
        timestamp_col: str,
        expected_interval: Optional[pd.Timedelta] = None
    ) -> List[str]:
        """Validate time series data quality."""
        issues = []

        if timestamp_col not in df.columns:
            issues.append(f"Timestamp column '{timestamp_col}' not found")
            return issues

        # Convert to datetime if needed
        if not pd.api.types.is_datetime64_any_dtype(df[timestamp_col]):
            try:
                df[timestamp_col] = pd.to_datetime(df[timestamp_col])
            except Exception as e:
                issues.append(f"Failed to convert {timestamp_col} to datetime: {e}")
                return issues

        # Check for time gaps
        if expected_interval is None:
        # Fallback implementation for expected_interval
            # Auto-detect interval from most common difference
            df_sorted = df.sort_values(timestamp_col)
            time_diffs = df_sorted[timestamp_col].diff().dropna()
            if len(time_diffs) > 0:
                expected_interval = time_diffs.mode().iloc[0] if len(time_diffs.mode()) > 0 else pd.Timedelta('1 minute')
            else:
                expected_interval = pd.Timedelta('1 minute')

        df_sorted = df.sort_values(timestamp_col)
        time_diffs = df_sorted[timestamp_col].diff()

        large_gaps = time_diffs[time_diffs > expected_interval * self.config['max_gap_multiplier']]
        if len(large_gaps) > 0:
            issues.append(
                f"Found {len(large_gaps)} time gaps > {expected_interval * self.config['max_gap_multiplier']}"
            )

        # Check for duplicate timestamps
        duplicate_times = df[timestamp_col].duplicated().sum()
        duplicate_ratio = duplicate_times / len(df)
        if duplicate_ratio > self.config['max_duplicate_ratio']:
            issues.append(
                f"High duplicate timestamp ratio: {duplicate_ratio:.2%} "
                f"({duplicate_times} duplicates, threshold: {self.config['max_duplicate_ratio']:.2%})"
            )

        # Check for future timestamps
        now = pd.Timestamp.now()
        future_tolerance = pd.Timedelta(minutes=self.config['future_tolerance_minutes'])
        future_times = df[df[timestamp_col] > now + future_tolerance]

        if len(future_times) > 0:
            issues.append(
                f"Found {len(future_times)} future timestamps "
                f"(beyond {self.config['future_tolerance_minutes']} minutes from now)"
            )

        # Check for very old timestamps (optional)
        old_threshold = now - pd.Timedelta(days=365)  # 1 year old
        old_times = df[df[timestamp_col] < old_threshold]
        if len(old_times) > len(df) * 0.1:  # More than 10% old data
            issues.append(
                f"High ratio of old data: {len(old_times)/len(df):.2%} "
                f"older than 1 year"
            )

        return issues


class FinancialDataValidator:
    """Validates financial data quality."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or self._get_default_config()
        self.logger = system_logger.getChild("FinancialDataValidator")

    def _get_default_config(self) -> Dict[str, Any]:
        return {
            "max_price_change_ratio": 0.5,  # 50% max price change
            "min_volume_threshold": 0.0,
            "zero_volume_ratio_threshold": 0.1
        }

    def validate_financial_data(self, df: pd.DataFrame) -> List[str]:
        """Validate financial data quality."""
        issues = []

        # Check OHLC relationships
        ohlc_cols = ['open', 'high', 'low', 'close']
        if all(col in df.columns for col in ohlc_cols):
            invalid_ohlc = df[
                (df['high'] < df['low']) |
                (df['open'] > df['high']) |
                (df['close'] > df['high']) |
                (df['open'] < df['low']) |
                (df['close'] < df['low'])
            ]

            if len(invalid_ohlc) > 0:
                issues.append(f"Found {len(invalid_ohlc)} invalid OHLC relationships")

        # Check for negative prices
        price_cols = ['open', 'high', 'low', 'close', 'price']
        for col in price_cols:
            if col in df.columns:
                negative_prices = df[df[col] < 0]
                if len(negative_prices) > 0:
                    issues.append(f"Found {len(negative_prices)} negative prices in {col}")

        # Check for zero volumes
        if 'volume' in df.columns:
            zero_volumes = df[df['volume'] == 0]
            zero_ratio = len(zero_volumes) / len(df)
            if zero_ratio > self.config['zero_volume_ratio_threshold']:
                issues.append(
                    f"High zero volume ratio: {zero_ratio:.2%} "
                    f"({len(zero_volumes)} records, threshold: {self.config['zero_volume_ratio_threshold']:.2%})"
                )

        # Check for extreme price changes
        if 'close' in df.columns and 'open' in df.columns:
            price_changes = abs(df['close'] - df['open']) / df['open']
            extreme_changes = price_changes[price_changes > self.config['max_price_change_ratio']]

            if len(extreme_changes) > 0:
                issues.append(
                    f"Found {len(extreme_changes)} extreme price changes "
                    f"(> {self.config['max_price_change_ratio']:.1%})"
                )

        # Check for missing OHLC data
        if all(col in df.columns for col in ohlc_cols):
            missing_ohlc = df[ohlc_cols].isnull().any(axis=1)
            if missing_ohlc.sum() > 0:
                issues.append(f"Found {missing_ohlc.sum()} records with missing OHLC data")

        return issues


class FeatureCorrelationValidator:
    """Validates feature correlations for ML training."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or self._get_default_config()
        self.logger = system_logger.getChild("FeatureCorrelationValidator")

    def _get_default_config(self) -> Dict[str, Any]:
        return {
            "max_correlation": 0.95,
            "max_multicollinearity_vif": 10.0,
            "min_correlation_for_removal": 0.8
        }

    def validate_feature_correlations(self, df: pd.DataFrame) -> List[str]:
        """Validate feature correlations for ML training."""
        issues = []

        # Select numeric columns only
        numeric_df = df.select_dtypes(include=[np.number])

        if len(numeric_df.columns) < 2:
            return issues

        # Calculate correlation matrix
        corr_matrix = numeric_df.corr()

        # Find highly correlated features
        high_corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_value = corr_matrix.iloc[i, j]
                if abs(corr_value) > self.config['max_correlation']:
                    high_corr_pairs.append((
                        corr_matrix.columns[i],
                        corr_matrix.columns[j],
                        corr_value
                    ))

        if high_corr_pairs:
            issues.append(f"Found {len(high_corr_pairs)} highly correlated feature pairs")
            # Show first 5 pairs
            for feat1, feat2, corr in high_corr_pairs[:5]:
                issues.append(f"  {feat1} - {feat2}: {corr:.3f}")

        # Check for multicollinearity using VIF
        vif_issues = self._check_multicollinearity(numeric_df)
        issues.extend(vif_issues)

        return issues

    def _check_multicollinearity(self, df: pd.DataFrame) -> List[str]:
        """Check for multicollinearity using Variance Inflation Factor."""
        issues = []

        if len(df.columns) < 2:
            return issues

        try:
            # Calculate VIF for each feature
            vif_data = []
            for i, col in enumerate(df.columns):
                # Use other features to predict this feature
                other_cols = [c for c in df.columns if c != col]
                if len(other_cols) > 0:
                    X = df[other_cols]
                    y = df[col]

                    # Handle missing values
                    mask = ~(X.isnull().any(axis=1) | y.isnull())
                    if mask.sum() > len(df) * 0.5:  # At least 50% of data
                        X_clean = X[mask]
                        y_clean = y[mask]

                        # Simple VIF calculation using R-squared
                        from sklearn.linear_model import LinearRegression
                        model = LinearRegression()
                        model.fit(X_clean, y_clean)
                        r_squared = model.score(X_clean, y_clean)

                        if r_squared < 1.0:  # Avoid division by zero
                            vif = 1 / (1 - r_squared)
                            vif_data.append((col, vif))

            # Check for high VIF values
            high_vif_features = [(col, vif) for col, vif in vif_data if vif > self.config['max_multicollinearity_vif']]

            if high_vif_features:
                issues.append(f"Found {len(high_vif_features)} features with high VIF")
                for col, vif in high_vif_features[:5]:  # Show first 5
                    issues.append(f"  {col}: VIF = {vif:.2f}")

        except Exception as e:
            issues.append(f"Error calculating VIF: {e}")

        return issues


class TargetVariableValidator:
    """Validates target variable for ML training."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or self._get_default_config()
        self.logger = system_logger.getChild("TargetVariableValidator")

    def _get_default_config(self) -> Dict[str, Any]:
        return {
            "class_imbalance_threshold": 0.1,
            "target_leakage_threshold": 0.9,
            "min_target_variance": 1e-6
        }

    def validate_target_variable(
        self,
        df: pd.DataFrame,
        target_col: str,
        timestamp_col: Optional[str] = None
    ) -> List[str]:
        """Validate target variable for ML training."""
        issues = []

        if target_col not in df.columns:
            issues.append(f"Target column '{target_col}' not found")
            return issues

        target = df[target_col]

        # Check for missing target values
        missing_target = target.isnull().sum()
        if missing_target > 0:
            issues.append(f"Found {missing_target} missing target values ({missing_target/len(target):.2%})")

        # Check for class imbalance (categorical targets)
        if target.dtype in ['object', 'category'] or target.nunique() < 10:
            class_counts = target.value_counts()
            min_class_ratio = class_counts.min() / class_counts.max()

            if min_class_ratio < self.config['class_imbalance_threshold']:
                issues.append(
                    f"Severe class imbalance: {min_class_ratio:.3f} "
                    f"(minority class: {class_counts.min()}, majority class: {class_counts.max()})"
                )

        # Check for target variance (regression targets)
        if target.dtype in [np.number] and target.nunique() > 10:
            target_variance = target.var()
            if target_variance < self.config['min_target_variance']:
                issues.append(
                    f"Low target variance: {target_variance:.6f} "
                    f"(threshold: {self.config['min_target_variance']})"
                )

        # Check for target leakage with time-based features
        if timestamp_col and timestamp_col in df.columns:
            time_leakage_issues = self._check_time_based_leakage(df, target_col, timestamp_col)
            issues.extend(time_leakage_issues)

        # Check for target leakage with other features
        feature_leakage_issues = self._check_feature_based_leakage(df, target_col)
        issues.extend(feature_leakage_issues)

        return issues

    def _check_time_based_leakage(
        self,
        df: pd.DataFrame,
        target_col: str,
        timestamp_col: str
    ) -> List[str]:
        """Check for target leakage with time-based features."""
        issues = []

        try:
            # Convert timestamp to datetime if needed
            if not pd.api.types.is_datetime64_any_dtype(df[timestamp_col]):
                df[timestamp_col] = pd.to_datetime(df[timestamp_col])

            # Create time-based features
            df_copy = df.copy()
            df_copy['hour'] = df_copy[timestamp_col].dt.hour
            df_copy['day_of_week'] = df_copy[timestamp_col].dt.dayofweek
            df_copy['month'] = df_copy[timestamp_col].dt.month

            # Check correlation with target
            time_features = ['hour', 'day_of_week', 'month']
            for feature in time_features:
                if feature in df_copy.columns:
                    corr = abs(df_copy[feature].corr(df_copy[target_col]))
                    if corr > self.config['target_leakage_threshold']:
                        issues.append(
                            f"Potential time-based target leakage with {feature}: corr={corr:.3f}"
                        )

        except Exception as e:
            issues.append(f"Error checking time-based leakage: {e}")

        return issues

    def _check_feature_based_leakage(self, df: pd.DataFrame, target_col: str) -> List[str]:
        """Check for target leakage with other features."""
        issues = []

        try:
            # Check for perfect or near-perfect correlations
            numeric_df = df.select_dtypes(include=[np.number])
            if target_col in numeric_df.columns:
                numeric_df = numeric_df.drop(columns=[target_col])

                for col in numeric_df.columns:
                    corr = abs(numeric_df[col].corr(df[target_col]))
                    if corr > self.config['target_leakage_threshold']:
                        issues.append(
                            f"Potential target leakage with {col}: corr={corr:.3f}"
                        )

        except Exception as e:
            issues.append(f"Error checking feature-based leakage: {e}")

        return issues


class DataDriftDetector:
    """Detects data drift between reference and current data."""

    def __init__(self, reference_data: pd.DataFrame, config: Optional[Dict[str, Any]] = None):
        self.reference_data = reference_data
        self.config = config or self._get_default_config()
        self.reference_stats = self._compute_statistics(reference_data)
        self.logger = system_logger.getChild("DataDriftDetector")

    def _get_default_config(self) -> Dict[str, Any]:
        return {
            "drift_psi_threshold": 0.25,
            "drift_ks_threshold": 0.05,
            "drift_correlation_threshold": 0.1
        }

    def detect_drift(self, current_data: pd.DataFrame) -> DriftReport:
        """Detect data drift between reference and current data."""
        issues = []
        drift_metrics = {}

        current_stats = self._compute_statistics(current_data)

        for column in self.reference_stats.keys():
            if column in current_stats:
                # Population Stability Index (PSI)
                psi = self._calculate_psi(
                    self.reference_data[column],
                    current_data[column]
                )
                drift_metrics[f"{column}_psi"] = psi

                if psi > self.config['drift_psi_threshold']:
                    issues.append(f"Drift detected in {column}: PSI={psi:.3f}")

                # Kolmogorov-Smirnov test
                ks_stat, ks_pvalue = self._calculate_ks_test(
                    self.reference_data[column],
                    current_data[column]
                )
                drift_metrics[f"{column}_ks_stat"] = ks_stat
                drift_metrics[f"{column}_ks_pvalue"] = ks_pvalue

                if ks_pvalue < self.config['drift_ks_threshold']:
                    issues.append(f"Distribution drift in {column}: KS p-value={ks_pvalue:.3f}")

        # Overall drift severity
        severity = ValidationSeverity.INFO
        if len(issues) > 5:
            severity = ValidationSeverity.CRITICAL
        elif len(issues) > 2:
            severity = ValidationSeverity.ERROR
        elif len(issues) > 0:
            severity = ValidationSeverity.WARNING

        return DriftReport(
            issues=issues,
            drift_metrics=drift_metrics,
            severity=severity
        )

    def _compute_statistics(self, df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """Compute statistics for drift detection."""
        stats = {}

        for column in df.select_dtypes(include=[np.number]).columns:
            stats[column] = {
                'mean': df[column].mean(),
                'std': df[column].std(),
                'min': df[column].min(),
                'max': df[column].max(),
                'q25': df[column].quantile(0.25),
                'q75': df[column].quantile(0.75)
            }

        return stats

    def _calculate_psi(self, reference: pd.Series, current: pd.Series) -> float:
        """Calculate Population Stability Index."""
        try:
            # Create bins for both distributions
            combined = pd.concat([reference, current])
            bins = pd.cut(combined, bins=10, duplicates='drop')

            # Calculate bin counts
            ref_counts = reference.groupby(pd.cut(reference, bins=bins.cat.categories)).count()
            curr_counts = current.groupby(pd.cut(current, bins=bins.cat.categories)).count()

            # Normalize to probabilities
            ref_probs = ref_counts / ref_counts.sum()
            curr_probs = curr_counts / curr_counts.sum()

            # Calculate PSI
            psi = 0
            for bin_name in ref_probs.index:
                if bin_name in curr_probs.index:
                    ref_p = ref_probs[bin_name]
                    curr_p = curr_probs[bin_name]

                    if ref_p > 0 and curr_p > 0:
                        psi += (curr_p - ref_p) * np.log(curr_p / ref_p)

            return psi

        except Exception:
            return 0.0

    def _calculate_ks_test(self, reference: pd.Series, current: pd.Series) -> Tuple[float, float]:
        """Calculate Kolmogorov-Smirnov test statistic and p-value."""
        try:
            # Remove NaN values
            ref_clean = reference.dropna()
            curr_clean = current.dropna()

            if len(ref_clean) > 0 and len(curr_clean) > 0:
                ks_stat, p_value = stats.ks_2samp(ref_clean, curr_clean)
                return ks_stat, p_value
            else:
                return 0.0, 1.0

        except Exception:
            return 0.0, 1.0


class DataQualityScorer:
    """Calculates overall data quality score."""

    def __init__(self, weights: Optional[Dict[str, float]] = None):
        self.weights = weights or {
            'completeness': 0.25,
            'consistency': 0.25,
            'accuracy': 0.25,
            'timeliness': 0.25
        }
        self.logger = system_logger.getChild("DataQualityScorer")

    def calculate_quality_score(self, df: pd.DataFrame, validation_result: MLValidationResult) -> QualityScore:
        """Calculate overall data quality score."""
        scores = {}

        # Completeness score
        completeness = 1 - (df.isnull().sum().sum() / (len(df) * len(df.columns)))
        scores['completeness'] = completeness

        # Consistency score
        consistency = self._calculate_consistency_score(df, validation_result)
        scores['consistency'] = consistency

        # Accuracy score
        accuracy = self._calculate_accuracy_score(df, validation_result)
        scores['accuracy'] = accuracy

        # Timeliness score
        timeliness = self._calculate_timeliness_score(df, validation_result)
        scores['timeliness'] = timeliness

        # Weighted average
        overall_score = sum(scores[metric] * self.weights[metric] for metric in scores)

        return QualityScore(
            overall=overall_score,
            components=scores,
            grade=self._get_grade(overall_score),
            details={
                'total_issues': len(validation_result.correlation_issues) +
                               len(validation_result.target_issues) +
                               len(validation_result.distribution_issues) +
                               len(validation_result.outlier_issues),
                'drift_detected': validation_result.drift_report is not None
            }
        )

    def _calculate_consistency_score(self, df: pd.DataFrame, validation_result: MLValidationResult) -> float:
        """Calculate consistency score based on validation issues."""
        base_score = 1.0

        # Penalize for correlation issues
        correlation_penalty = len(validation_result.correlation_issues) * 0.05
        base_score -= min(correlation_penalty, 0.3)

        # Penalize for distribution issues
        distribution_penalty = len(validation_result.distribution_issues) * 0.03
        base_score -= min(distribution_penalty, 0.2)

        return max(base_score, 0.0)

    def _calculate_accuracy_score(self, df: pd.DataFrame, validation_result: MLValidationResult) -> float:
        """Calculate accuracy score based on validation issues."""
        base_score = 1.0

        # Penalize for outlier issues
        outlier_penalty = len(validation_result.outlier_issues) * 0.04
        base_score -= min(outlier_penalty, 0.3)

        # Penalize for financial data issues
        financial_penalty = len(validation_result.financial_issues) * 0.05
        base_score -= min(financial_penalty, 0.3)

        return max(base_score, 0.0)

    def _calculate_timeliness_score(self, df: pd.DataFrame, validation_result: MLValidationResult) -> float:
        """Calculate timeliness score based on validation issues."""
        base_score = 1.0

        # Penalize for time series issues
        time_series_penalty = len(validation_result.time_series_issues) * 0.05
        base_score -= min(time_series_penalty, 0.4)

        return max(base_score, 0.0)

    def _get_grade(self, score: float) -> str:
        """Convert score to letter grade."""
        if score >= 0.9: return "A"
        elif score >= 0.8: return "B"
        elif score >= 0.7: return "C"
        elif score >= 0.6: return "D"
        else: return "F"


class AdvancedMLValidator:
    """Comprehensive ML data quality validator."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or self._get_default_config()
        self.logger = system_logger.getChild("AdvancedMLValidator")

        # Initialize validators
        self.statistical_validator = StatisticalDataValidator()
        self.time_series_validator = TimeSeriesValidator()
        self.financial_validator = FinancialDataValidator()
        self.correlation_validator = FeatureCorrelationValidator()
        self.target_validator = TargetVariableValidator()
        self.quality_scorer = DataQualityScorer()

        # Drift detector will be set when reference data is provided
        self.drift_detector = None

    def _get_default_config(self) -> Dict[str, Any]:
        return {
            "timestamp_column": "timestamp",
            "target_column": "target",
            "expected_interval": None,
            "reference_data": None,
            "validate_distributions": True,
            "validate_outliers": True,
            "validate_time_series": True,
            "validate_financial": True,
            "validate_correlations": True,
            "validate_target": True,
            "detect_drift": False
        }

    def set_reference_data(self, reference_data: pd.DataFrame):
        """Set reference data for drift detection."""
        self.drift_detector = DataDriftDetector(reference_data)
        self.config["reference_data"] = reference_data

    def validate_ml_data(
        self,
        df: pd.DataFrame,
        target_col: Optional[str] = None,
        timestamp_col: Optional[str] = None
    ) -> MLValidationResult:
        """Comprehensive ML data validation."""
        self.logger.info("🔍 Starting comprehensive ML data validation...")

        # Use config defaults if not provided
        target_col = target_col or self.config["target_column"]
        timestamp_col = timestamp_col or self.config["timestamp_column"]

        # Initialize result
        result = MLValidationResult(is_valid=True, quality_score=None)

        # Statistical validation
        if self.config["validate_distributions"]:
            result.distribution_issues = self.statistical_validator.validate_data_distributions(df)

        if self.config["validate_outliers"]:
            result.outlier_issues = self.statistical_validator.validate_outliers(df)

        # Time series validation
        if self.config["validate_time_series"] and timestamp_col in df.columns:
            result.time_series_issues = self.time_series_validator.validate_time_series_quality(
                df, timestamp_col, self.config["expected_interval"]
            )

        # Financial data validation
        if self.config["validate_financial"]:
            result.financial_issues = self.financial_validator.validate_financial_data(df)

        # Feature correlation validation
        if self.config["validate_correlations"]:
            result.correlation_issues = self.correlation_validator.validate_feature_correlations(df)

        # Target variable validation
        if self.config["validate_target"] and target_col:
            result.target_issues = self.target_validator.validate_target_variable(
                df, target_col, timestamp_col
            )

        # Drift detection
        if self.config["detect_drift"] and self.drift_detector:
            result.drift_report = self.drift_detector.detect_drift(df)

        # Calculate quality score
        result.quality_score = self.quality_scorer.calculate_quality_score(df, result)

        # Determine overall validity
        total_issues = (
            len(result.correlation_issues) +
            len(result.target_issues) +
            len(result.distribution_issues) +
            len(result.outlier_issues) +
            len(result.time_series_issues) +
            len(result.financial_issues)
        )

        if result.drift_report:
            total_issues += len(result.drift_report.issues)

        result.is_valid = total_issues == 0
        result.summary = {
            "total_issues": total_issues,
            "quality_score": result.quality_score.overall,
            "quality_grade": result.quality_score.grade,
            "drift_detected": result.drift_report is not None
        }

        # Log results
        if result.is_valid:
            self.logger.info(f"✅ ML data validation passed (Score: {result.quality_score.overall:.3f}, Grade: {result.quality_score.grade})")
        else:
            self.logger.warning(f"⚠️ ML data validation found {total_issues} issues (Score: {result.quality_score.overall:.3f}, Grade: {result.quality_score.grade})")

        return result


# Convenience functions for easy usage
def validate_ml_data_quality(
    df: pd.DataFrame,
    target_col: Optional[str] = None,
    timestamp_col: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None
) -> MLValidationResult:
    """Convenience function for ML data quality validation."""
    validator = AdvancedMLValidator(config)
    return validator.validate_ml_data(df, target_col, timestamp_col)


def detect_data_drift(
    reference_data: pd.DataFrame,
    current_data: pd.DataFrame
) -> DriftReport:
    """Convenience function for data drift detection."""
    detector = DataDriftDetector(reference_data)
    return detector.detect_drift(current_data)


def calculate_data_quality_score(
    df: pd.DataFrame,
    validation_result: MLValidationResult,
    weights: Optional[Dict[str, float]] = None
) -> QualityScore:
    """Convenience function for quality score calculation."""
    scorer = DataQualityScorer(weights)
    return scorer.calculate_quality_score(df, validation_result)