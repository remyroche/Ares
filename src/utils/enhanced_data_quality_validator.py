"""
Enhanced Data Quality Validator with Feature-Specific Thresholds
Provides comprehensive validation with context-aware thresholds and automatic fixes.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
from enum import Enum
import warnings
import logging
from collections import defaultdict

warnings.filterwarnings("ignore")

from src.utils.logger import system_logger


class ValidationLevel(Enum):
    """Validation severity levels."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class ValidationIssue:
    """Represents a data quality validation issue."""

    feature: str
    issue_type: str
    level: ValidationLevel
    description: str
    count: int = 0
    percentage: float = 0.0
    details: Optional[Dict[str, Any]] = None
    feature_type: str = "unknown"
    threshold_applied: float = 0.0


class EnhancedDataQualityValidator:
    """Enhanced data quality validator with feature-specific thresholds and market gap detection."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.logger = system_logger.getChild("EnhancedDataQualityValidator")
        self.config = config or self._get_default_config()
        self.issues: List[ValidationIssue] = []
        self.feature_types: Dict[str, str] = {}

    def _get_default_config(self) -> Dict[str, Any]:
        """Get default validation configuration with feature-specific thresholds."""
        return {
            # Feature-specific thresholds
            "feature_thresholds": {
                "wavelet_features": {
                    "missing_warning": 0.05,  # 5%
                    "missing_error": 0.20,  # 20%
                    "variance_threshold": 1e-12,
                    "description": "Wavelet features naturally have edge effects and low variance",
                },
                "multi_timeframe_features": {
                    "missing_warning": 0.02,  # 2%
                    "missing_error": 0.10,  # 10%
                    "variance_threshold": 1e-10,
                    "description": "Alignment issues between timeframes can cause gaps",
                },
                "technical_indicators": {
                    "missing_warning": 0.01,  # 1%
                    "missing_error": 0.05,  # 5%
                    "variance_threshold": 1e-8,
                    "description": "Technical indicators should be mostly complete",
                },
                "price_features": {
                    "missing_warning": 0.001,  # 0.1%
                    "missing_error": 0.01,  # 1%
                    "variance_threshold": 1e-6,
                    "description": "Price data should be nearly complete",
                },
            },
            # Global thresholds
            "infinite_threshold": 0.05,  # 5% infinite threshold
            "correlation_threshold": 0.95,
            "extreme_value_threshold": 1e6,
            "constant_threshold": 0.99,  # 99% same value
            "market_gap_threshold": 0.001,  # 0.1% for market gaps
            "min_gap_duration": 2,  # Minimum consecutive periods to consider as a market gap
            # Auto-fix settings
            "enable_auto_fix": True,
            "enable_market_gap_detection": True,
            "enable_data_type_fixes": True,
            "fix_strategies": {
                "nan": "drop",  # or "fill", "interpolate"
                "infinite": "clip",  # or "drop", "fill"
                "zero_variance": "drop",
                "constant": "drop",
                "extreme_values": "clip",
                "data_type_issues": "convert",
            },
        }

    def detect_feature_type(self, feature_name: str) -> str:
        """Detect feature type based on feature name patterns."""
        feature_name_lower = feature_name.lower()

        # Wavelet features
        if any(
            pattern in feature_name_lower
            for pattern in ["wavelet", "wav", "dwt", "cwt"]
        ):
            return "wavelet_features"

        # Multi-timeframe features
        if any(
            pattern in feature_name_lower
            for pattern in ["_1m_", "_5m_", "_15m_", "_1h_", "_4h_", "_1d_"]
        ):
            return "multi_timeframe_features"

        # Price features
        if any(
            pattern in feature_name_lower
            for pattern in ["price", "open", "high", "low", "close", "volume"]
        ):
            return "price_features"

        # Technical indicators
        if any(
            pattern in feature_name_lower
            for pattern in ["rsi", "macd", "bollinger", "sma", "ema", "atr", "stoch"]
        ):
            return "technical_indicators"

        # Default to technical indicators for unknown features
        return "technical_indicators"

    def get_feature_thresholds(self, feature_type: str) -> Dict[str, float]:
        """Get thresholds for specific feature type."""
        thresholds = self.config["feature_thresholds"]
        return thresholds.get(feature_type, thresholds["technical_indicators"])

    def detect_market_gaps(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Detect market gaps in price data with improved logic."""
        market_gaps = {"gaps_detected": [], "gap_summary": {}, "affected_features": []}

        # Look for price features
        price_features = [
            col
            for col in data.columns
            if self.detect_feature_type(col) == "price_features"
        ]

        if not price_features:
            return market_gaps

        # Minimum gap duration to consider as a real market gap (not just isolated missing values)
        min_gap_duration = self.config.get(
            "min_gap_duration", 2
        )  # At least 2 consecutive periods

        # Detect gaps in price data
        for feature in price_features:
            if feature in data.columns:
                # Find consecutive NaN values (gaps)
                is_na = data[feature].isna()
                gap_starts = is_na & ~is_na.shift(1).fillna(False)
                gap_ends = is_na & ~is_na.shift(-1).fillna(False)

                gap_start_indices = data.index[gap_starts]
                gap_end_indices = data.index[gap_ends]

                for start_idx, end_idx in zip(gap_start_indices, gap_end_indices):
                    gap_duration = len(data.loc[start_idx:end_idx])

                    # Only consider gaps that are long enough to be real market gaps
                    if gap_duration >= min_gap_duration:
                        gap_info = {
                            "feature": feature,
                            "start_time": start_idx,
                            "end_time": end_idx,
                            "duration": gap_duration,
                            "gap_type": "market_gap",
                        }
                        market_gaps["gaps_detected"].append(gap_info)

                        # Check if gap affects other features (only for significant gaps)
                        for other_feature in data.columns:
                            if other_feature != feature:
                                gap_data = data.loc[start_idx:end_idx, other_feature]
                                if gap_data.isna().any():
                                    market_gaps["affected_features"].append(
                                        {
                                            "primary_feature": feature,
                                            "affected_feature": other_feature,
                                            "gap_start": start_idx,
                                            "gap_end": end_idx,
                                        }
                                    )

        # Summarize gaps
        if market_gaps["gaps_detected"]:
            gap_durations = [gap["duration"] for gap in market_gaps["gaps_detected"]]
            market_gaps["gap_summary"] = {
                "total_gaps": len(market_gaps["gaps_detected"]),
                "avg_gap_duration": np.mean(gap_durations),
                "max_gap_duration": max(gap_durations),
                "min_gap_duration": min(gap_durations),
                "affected_features_count": len(
                    set(gap["feature"] for gap in market_gaps["gaps_detected"])
                ),
            }

        return market_gaps

    def fix_data_type_issues(
        self, data: pd.DataFrame
    ) -> Tuple[pd.DataFrame, List[str]]:
        """Fix data type issues in the dataset."""
        fixed_data = data.copy()
        fixes_applied = []

        for column in fixed_data.columns:
            original_dtype = fixed_data[column].dtype

            # Skip if already numeric
            if pd.api.types.is_numeric_dtype(original_dtype):
                continue

            # Handle object dtype (strings/mixed types)
            if original_dtype == "object":
                try:
                    # Try to convert to numeric
                    numeric_data = pd.to_numeric(fixed_data[column], errors="coerce")

                    # Check if conversion was successful (not all NaN)
                    if not numeric_data.isna().all():
                        fixed_data[column] = numeric_data
                        fixes_applied.append(
                            f"Converted {column} from object to numeric"
                        )
                    else:
                        # Check if it's datetime
                        try:
                            datetime_data = pd.to_datetime(
                                fixed_data[column], errors="coerce"
                            )
                            if not datetime_data.isna().all():
                                # Convert datetime to numeric (timestamp)
                                fixed_data[column] = (
                                    datetime_data.astype(np.int64) // 10**9
                                )
                                fixes_applied.append(
                                    f"Converted {column} from datetime to timestamp"
                                )
                            else:
                                fixes_applied.append(
                                    f"Could not convert {column} - keeping as object"
                                )
                        except:
                            fixes_applied.append(
                                f"Could not convert {column} - keeping as object"
                            )
                except:
                    fixes_applied.append(
                        f"Could not convert {column} - keeping as object"
                    )

        return fixed_data, fixes_applied

    def validate_dataset(
        self, data: pd.DataFrame, dataset_name: str = "unknown"
    ) -> Dict[str, Any]:
        """Enhanced dataset validation with feature-specific thresholds."""
        self.logger.info(
            f"🔍 Starting enhanced data quality validation for {dataset_name}"
        )
        self.issues.clear()

        validation_results = {
            "dataset_name": dataset_name,
            "shape": data.shape,
            "total_features": len(data.columns),
            "total_rows": len(data),
            "issues": [],
            "summary": {},
            "recommendations": [],
            "market_gaps": {},
            "data_type_fixes": [],
            "feature_type_breakdown": defaultdict(int),
        }

        # Detect feature types
        for column in data.columns:
            feature_type = self.detect_feature_type(column)
            self.feature_types[column] = feature_type
            validation_results["feature_type_breakdown"][feature_type] += 1

        # Fix data type issues first
        if self.config["enable_data_type_fixes"]:
            fixed_data, fixes_applied = self.fix_data_type_issues(data)
            validation_results["data_type_fixes"] = fixes_applied
            data = fixed_data

        # Detect market gaps
        if self.config["enable_market_gap_detection"]:
            market_gaps = self.detect_market_gaps(data)
            validation_results["market_gaps"] = market_gaps

            # Add consolidated market gap warnings
            if market_gaps["gaps_detected"]:
                # Group gaps by time period to avoid repetitive logging
                gap_groups = {}
                for gap in market_gaps["gaps_detected"]:
                    gap_key = f"{gap['start_time']}_{gap['end_time']}"
                    if gap_key not in gap_groups:
                        gap_groups[gap_key] = {
                            "start_time": gap["start_time"],
                            "end_time": gap["end_time"],
                            "duration": gap["duration"],
                            "features": [],
                        }
                    gap_groups[gap_key]["features"].append(gap["feature"])

                # Create one consolidated issue per gap period
                for gap_key, gap_info in gap_groups.items():
                    feature_list = gap_info["features"]
                    if len(feature_list) == 1:
                        feature_name = feature_list[0]
                        description = f"Market gap detected: {gap_info['duration']} periods from {gap_info['start_time']} to {gap_info['end_time']}"
                    else:
                        feature_name = f"multiple_features_{len(feature_list)}"
                        description = f"Market gap affecting {len(feature_list)} features: {gap_info['duration']} periods from {gap_info['start_time']} to {gap_info['end_time']}"

                    issue = ValidationIssue(
                        feature=feature_name,
                        issue_type="market_gap",
                        level=ValidationLevel.WARNING,
                        description=description,
                        count=gap_info["duration"],
                        percentage=gap_info["duration"] / len(data),
                        feature_type="price_features",
                        details={
                            "gap_period": gap_key,
                            "affected_features": gap_info["features"],
                            "start_time": gap_info["start_time"],
                            "end_time": gap_info["end_time"],
                            "duration": gap_info["duration"],
                        },
                    )
                    self.issues.append(issue)

        # Feature-specific validation
        self._validate_features_with_type_specific_thresholds(data)

        # Global validation checks
        self._validate_infinite_values(data)
        self._validate_extreme_values(data)
        self._validate_constant_values(data)

        # Compile results
        validation_results["issues"] = [
            {
                "feature": issue.feature,
                "issue_type": issue.issue_type,
                "level": issue.level.value,
                "description": issue.description,
                "count": issue.count,
                "percentage": issue.percentage,
                "feature_type": issue.feature_type,
                "threshold_applied": issue.threshold_applied,
                "details": issue.details,
            }
            for issue in self.issues
        ]

        # Generate summary
        validation_results["summary"] = self._generate_summary()

        # Generate recommendations
        validation_results["recommendations"] = self._generate_recommendations(
            validation_results
        )

        return validation_results

    def _validate_features_with_type_specific_thresholds(self, data: pd.DataFrame):
        """Validate features using type-specific thresholds."""
        for column in data.columns:
            feature_type = self.feature_types.get(column, "technical_indicators")
            thresholds = self.get_feature_thresholds(feature_type)

            # Calculate statistics
            total_rows = len(data)
            missing_count = data[column].isna().sum()
            missing_pct = missing_count / total_rows if total_rows > 0 else 0

            # Check infinite values only for numeric columns
            infinite_count = 0
            infinite_pct = 0
            if pd.api.types.is_numeric_dtype(data[column]):
                infinite_count = np.isinf(data[column]).sum()
                infinite_pct = infinite_count / total_rows if total_rows > 0 else 0

            # Calculate variance only for numeric columns
            feature_data = data[column].dropna()
            variance = 0
            if pd.api.types.is_numeric_dtype(data[column]) and len(feature_data) > 1:
                variance = feature_data.var()

            # Apply feature-specific thresholds
            if missing_pct > thresholds["missing_error"]:
                issue = ValidationIssue(
                    feature=column,
                    issue_type="missing_values",
                    level=ValidationLevel.ERROR,
                    description=f"🚨 {missing_pct*100:.2f}% missing values (threshold: {thresholds['missing_error']*100:.1f}%)",
                    count=missing_count,
                    percentage=missing_pct,
                    feature_type=feature_type,
                    threshold_applied=thresholds["missing_error"],
                )
                self.issues.append(issue)
            elif missing_pct > thresholds["missing_warning"]:
                issue = ValidationIssue(
                    feature=column,
                    issue_type="missing_values",
                    level=ValidationLevel.WARNING,
                    description=f"⚠️ {missing_pct*100:.2f}% missing values (threshold: {thresholds['missing_warning']*100:.1f}%)",
                    count=missing_count,
                    percentage=missing_pct,
                    feature_type=feature_type,
                    threshold_applied=thresholds["missing_warning"],
                )
                self.issues.append(issue)

            # Variance check
            if variance < thresholds["variance_threshold"]:
                issue = ValidationIssue(
                    feature=column,
                    issue_type="low_variance",
                    level=ValidationLevel.WARNING,
                    description=f"Low variance {variance:.2e} (threshold: {thresholds['variance_threshold']:.2e})",
                    count=0,
                    percentage=0,
                    feature_type=feature_type,
                    threshold_applied=thresholds["variance_threshold"],
                    details={"variance": variance},
                )
                self.issues.append(issue)

    def _validate_infinite_values(self, data: pd.DataFrame):
        """Validate infinite values."""
        for column in data.columns:
            # Only check numeric columns for infinite values
            if not pd.api.types.is_numeric_dtype(data[column]):
                continue

            infinite_count = np.isinf(data[column]).sum()
            infinite_pct = infinite_count / len(data) if len(data) > 0 else 0

            if infinite_pct > self.config["infinite_threshold"]:
                issue = ValidationIssue(
                    feature=column,
                    issue_type="infinite_values",
                    level=ValidationLevel.ERROR,
                    description=f"{infinite_pct*100:.2f}% infinite values",
                    count=infinite_count,
                    percentage=infinite_pct,
                    feature_type=self.feature_types.get(column, "unknown"),
                )
                self.issues.append(issue)

    def _validate_extreme_values(self, data: pd.DataFrame):
        """Validate extreme values."""
        for column in data.columns:
            if pd.api.types.is_numeric_dtype(data[column]):
                extreme_count = (
                    np.abs(data[column]) > self.config["extreme_value_threshold"]
                ).sum()
                extreme_pct = extreme_count / len(data) if len(data) > 0 else 0

                if extreme_pct > 0.01:  # More than 1% extreme values
                    issue = ValidationIssue(
                        feature=column,
                        issue_type="extreme_values",
                        level=ValidationLevel.WARNING,
                        description=f"{extreme_pct*100:.2f}% extreme values (> {self.config['extreme_value_threshold']})",
                        count=extreme_count,
                        percentage=extreme_pct,
                        feature_type=self.feature_types.get(column, "unknown"),
                    )
                    self.issues.append(issue)

    def _validate_constant_values(self, data: pd.DataFrame):
        """Validate constant values."""
        for column in data.columns:
            unique_values = data[column].nunique()
            constant_pct = unique_values / len(data) if len(data) > 0 else 0

            if constant_pct < (1 - self.config["constant_threshold"]):
                issue = ValidationIssue(
                    feature=column,
                    issue_type="constant_values",
                    level=ValidationLevel.WARNING,
                    description=f"🚨 Near-constant values ({unique_values} unique out of {len(data)})",
                    count=unique_values,
                    percentage=constant_pct,
                    feature_type=self.feature_types.get(column, "unknown"),
                )
                self.issues.append(issue)

    def _generate_summary(self) -> Dict[str, Any]:
        """Generate validation summary."""
        total_issues = len(self.issues)
        critical_issues = len(
            [i for i in self.issues if i.level == ValidationLevel.CRITICAL]
        )
        error_issues = len([i for i in self.issues if i.level == ValidationLevel.ERROR])
        warning_issues = len(
            [i for i in self.issues if i.level == ValidationLevel.WARNING]
        )
        info_issues = len([i for i in self.issues if i.level == ValidationLevel.SILENT])

        return {
            "total_issues": total_issues,
            "critical_issues": critical_issues,
            "error_issues": error_issues,
            "warning_issues": warning_issues,
            "info_issues": info_issues,
        }

    def _generate_recommendations(
        self, validation_results: Dict[str, Any]
    ) -> List[str]:
        """Generate recommendations based on validation results."""
        recommendations = []

        # Market gap recommendations
        if validation_results["market_gaps"]["gaps_detected"]:
            recommendations.append(
                "⚠️ Market gaps detected - consider data interpolation or gap handling"
            )

        # Data type fix recommendations
        if validation_results["data_type_fixes"]:
            recommendations.append("✅ Data type issues fixed automatically")

        # Feature-specific recommendations
        feature_issues = defaultdict(list)
        for issue in self.issues:
            feature_issues[issue.feature_type].append(issue)

        for feature_type, issues in feature_issues.items():
            if feature_type == "wavelet_features" and any(
                i.issue_type == "low_variance" for i in issues
            ):
                recommendations.append(
                    "ℹ️ Low variance in wavelet features is expected - consider adjusting thresholds"
                )

            if feature_type == "multi_timeframe_features" and any(
                i.issue_type == "missing_values" for i in issues
            ):
                recommendations.append(
                    "⚠️ Missing values in multi-timeframe features - check alignment logic"
                )

        return recommendations


def enhanced_validate_features(
    data: pd.DataFrame, dataset_name: str = "features"
) -> Dict[str, Any]:
    """
    Enhanced validation function with feature-specific thresholds.
    This is the main function to be used in the pipeline.
    """
    validator = EnhancedDataQualityValidator()
    return validator.validate_dataset(data, dataset_name)
