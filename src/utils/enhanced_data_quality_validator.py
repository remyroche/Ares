"""
Enhanced Data Quality Validator with Feature-Specific Thresholds
Provides comprehensive validation with context-aware thresholds and automatic fixes.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional

import numpy as np
import pandas as pd

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
    details: dict[str, Any] | None = None
    feature_type: str = "unknown"
    threshold_applied: float = 0.0


class EnhancedDataQualityValidator:
    """Enhanced data quality validator with feature-specific thresholds and market gap detection."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.logger = system_logger.getChild("EnhancedDataQualityValidator") if system_logger else None
        self.config: dict[str, Any] = config or self._get_default_config()
        self.issues: list[ValidationIssue] = []
        self.feature_types: dict[str, str] = {}

    def _get_default_config(self) -> dict[str, Any]:
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
            "constant_threshold": 0.99,  # 99% of same value considered near-constant
            "market_gap_threshold": 0.001,  # 0.1% for market gaps
            "min_gap_duration": 2,  # Minimum consecutive periods to consider a market gap
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
        name = feature_name.lower()
        if any(p in name for p in ["wavelet", "wav", "dwt", "cwt"]):
            return "wavelet_features"
        if any(p in name for p in ["_1m_", "_5m_", "_15m_", "_1h_", "_4h_", "_1d_"]):
            return "multi_timeframe_features"
        if any(p in name for p in ["price", "open", "high", "low", "close", "volume"]):
            return "price_features"
        if any(p in name for p in ["rsi", "macd", "bollinger", "sma", "ema", "atr", "stoch"]):
            return "technical_indicators"
        return "technical_indicators"

    def get_feature_thresholds(self, feature_type: str) -> dict[str, float]:
        """Get thresholds for specific feature type."""
        thresholds = self.config["feature_thresholds"]
        return thresholds.get(feature_type, thresholds["technical_indicators"])  # type: ignore[return-value]

    def detect_market_gaps(self, data: pd.DataFrame) -> dict[str, Any]:
        """Detect market gaps in price data with improved logic."""
        market_gaps: dict[str, Any] = {"gaps_detected": [], "gap_summary": {}, "affected_features": []}
        price_features = [col for col in data.columns if self.detect_feature_type(col) == "price_features"]
        if not price_features:
            return market_gaps

        min_gap_duration = int(self.config.get("min_gap_duration", 2))
        for feature in price_features:
            is_na = data[feature].isna()
            if not is_na.any():
                continue
            gap_starts = is_na & ~is_na.shift(1, fill_value=False)
            gap_ends = is_na & ~is_na.shift(-1, fill_value=False)
            start_indices = list(data.index[gap_starts])
            end_indices = list(data.index[gap_ends])
            # Align pairs safely
            for start_idx, end_idx in zip(start_indices, end_indices):
                gap_duration = int(len(data.loc[start_idx:end_idx]))
                if gap_duration >= min_gap_duration:
                    gap_info = {
                        "feature": feature,
                        "start_time": start_idx,
                        "end_time": end_idx,
                        "duration": gap_duration,
                        "gap_type": "market_gap",
                    }
                    market_gaps["gaps_detected"].append(gap_info)

                    # Check affected features in same period
                    for other_feature in data.columns:
                        if other_feature == feature:
                            continue
                        gap_data = data.loc[start_idx:end_idx, other_feature]
                        if gap_data.isna().any():
                            market_gaps["affected_features"].append(
                                {
                                    "primary_feature": feature,
                                    "affected_feature": other_feature,
                                    "gap_start": start_idx,
                                    "gap_end": end_idx,
                                },
                            )

        if market_gaps["gaps_detected"]:
            durations = [gap["duration"] for gap in market_gaps["gaps_detected"]]
            market_gaps["gap_summary"] = {
                "total_gaps": len(market_gaps["gaps_detected"]),
                "avg_gap_duration": float(np.mean(durations)) if durations else 0.0,
                "max_gap_duration": int(max(durations)) if durations else 0,
                "min_gap_duration": int(min(durations)) if durations else 0,
                "affected_features_count": len({gap["feature"] for gap in market_gaps["gaps_detected"]}),
            }
        return market_gaps

    def fix_data_type_issues(self, data: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
        """Fix data type issues in the dataset."""
        fixed_data = data.copy()
        fixes_applied: list[str] = []
        for column in fixed_data.columns:
            original_dtype = fixed_data[column].dtype
            if pd.api.types.is_numeric_dtype(original_dtype):
                continue
            try:
                if original_dtype == "object":
                    numeric_data = pd.to_numeric(fixed_data[column], errors="coerce")
                    if not numeric_data.isna().all():
                        fixed_data[column] = numeric_data
                        fixes_applied.append(f"Converted {column} from object to numeric")
                        continue
                    datetime_data = pd.to_datetime(fixed_data[column], errors="coerce")
                    if not datetime_data.isna().all():
                        fixed_data[column] = (datetime_data.astype("int64") // 10**9)
                        fixes_applied.append(f"Converted {column} from datetime to timestamp")
                        continue
                    fixes_applied.append(f"Could not convert {column} - keeping as object")
                else:
                    # Attempt generic conversion to numeric
                    converted = pd.to_numeric(fixed_data[column], errors="coerce")
                    if not converted.isna().all():
                        fixed_data[column] = converted
                        fixes_applied.append(f"Coerced {column} to numeric")
                    else:
                        fixes_applied.append(f"Could not convert {column}")
            except Exception:
                fixes_applied.append(f"Could not convert {column} - keeping as {original_dtype}")
        return fixed_data, fixes_applied

    def validate_dataset(self, data: pd.DataFrame, dataset_name: str = "unknown") -> dict[str, Any]:
        """Enhanced dataset validation with feature-specific thresholds."""
        log = self.logger or system_logger
        if log:
            log.info("🔍 Starting enhanced data quality validation for %s", dataset_name)
        self.issues.clear()

        results: dict[str, Any] = {
            "dataset_name": dataset_name,
            "shape": data.shape,
            "total_features": int(len(data.columns)),
            "total_rows": int(len(data)),
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
            results["feature_type_breakdown"][feature_type] += 1

        # Fix data type issues first
        if self.config.get("enable_data_type_fixes", True):
            data, fixes_applied = self.fix_data_type_issues(data)
            results["data_type_fixes"] = fixes_applied

        # Detect market gaps
        market_gaps = {"gaps_detected": []}
        if self.config.get("enable_market_gap_detection", True):
            market_gaps = self.detect_market_gaps(data)
            results["market_gaps"] = market_gaps

        # Consolidate market gap warnings
        if market_gaps.get("gaps_detected"):
            gap_groups: dict[str, dict[str, Any]] = {}
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

            for gap_key, gap_info in gap_groups.items():
                feature_list = gap_info["features"]
                if len(feature_list) == 1:
                    feature_name = feature_list[0]
                    description = (
                        f"Market gap detected: {gap_info['duration']} periods from {gap_info['start_time']} to {gap_info['end_time']}"
                    )
                else:
                    feature_name = f"multiple_features_{len(feature_list)}"
                    description = (
                        f"Market gap affecting {len(feature_list)} features: {gap_info['duration']} periods from {gap_info['start_time']} to {gap_info['end_time']}"
                    )
                issue = ValidationIssue(
                    feature=feature_name,
                    issue_type="market_gap",
                    level=ValidationLevel.WARNING,
                    description=description,
                    count=int(gap_info["duration"]),
                    percentage=(gap_info["duration"] / max(1, len(data))),
                    feature_type="price_features",
                    details={
                        "gap_period": gap_key,
                        "affected_features": feature_list,
                        "start_time": gap_info["start_time"],
                        "end_time": gap_info["end_time"],
                        "duration": gap_info["duration"],
                    },
                )
                self.issues.append(issue)

        # Feature-specific and global validations
        self._validate_features_with_type_specific_thresholds(data)
        self._validate_infinite_values(data)
        self._validate_extreme_values(data)
        self._validate_constant_values(data)

        # Compile issues
        results["issues"] = [
            {
                "feature": i.feature,
                "issue_type": i.issue_type,
                "level": i.level.value,
                "description": i.description,
                "count": i.count,
                "percentage": i.percentage,
                "feature_type": i.feature_type,
                "threshold_applied": i.threshold_applied,
                "details": i.details,
            }
            for i in self.issues
        ]

        # Summary and recommendations
        results["summary"] = self._generate_summary()
        results["recommendations"] = self._generate_recommendations(results)

        return results

    def _validate_features_with_type_specific_thresholds(self, data: pd.DataFrame) -> None:
        """Validate features using type-specific thresholds."""
        total_rows = len(data)
        for column in data.columns:
            feature_type = self.feature_types.get(column, "technical_indicators")
            thresholds = self.get_feature_thresholds(feature_type)

            missing_count = int(data[column].isna().sum())
            missing_pct = (missing_count / total_rows) if total_rows > 0 else 0.0

            # Infinite values check only for numeric columns
            infinite_count = 0
            if pd.api.types.is_numeric_dtype(data[column]):
                infinite_count = int(np.isinf(data[column]).sum())

            # Variance for numeric columns
            variance = 0.0
            if pd.api.types.is_numeric_dtype(data[column]):
                non_na = data[column].dropna()
                if len(non_na) > 1:
                    try:
                        variance = float(non_na.var())
                    except Exception:
                        variance = 0.0

            # Missing thresholds
            if missing_pct > thresholds["missing_error"]:
                self.issues.append(
                    ValidationIssue(
                        feature=column,
                        issue_type="missing_values",
                        level=ValidationLevel.ERROR,
                        description=(
                            f"{missing_pct*100:.2f}% missing values (threshold: {thresholds['missing_error']*100:.1f}%)"
                        ),
                        count=missing_count,
                        percentage=missing_pct,
                        feature_type=feature_type,
                        threshold_applied=thresholds["missing_error"],
                    ),
                )
            elif missing_pct > thresholds["missing_warning"]:
                self.issues.append(
                    ValidationIssue(
                        feature=column,
                        issue_type="missing_values",
                        level=ValidationLevel.WARNING,
                        description=(
                            f"{missing_pct*100:.2f}% missing values (threshold: {thresholds['missing_warning']*100:.1f}%)"
                        ),
                        count=missing_count,
                        percentage=missing_pct,
                        feature_type=feature_type,
                        threshold_applied=thresholds["missing_warning"],
                    ),
                )

            # Low variance check
            if pd.api.types.is_numeric_dtype(data[column]) and variance < thresholds["variance_threshold"]:
                self.issues.append(
                    ValidationIssue(
                        feature=column,
                        issue_type="low_variance",
                        level=ValidationLevel.WARNING,
                        description=(
                            f"Low variance {variance:.2e} (threshold: {thresholds['variance_threshold']:.2e})"
                        ),
                        count=0,
                        percentage=0,
                        feature_type=feature_type,
                        threshold_applied=thresholds["variance_threshold"],
                        details={"variance": variance},
                    ),
                )

    def _validate_infinite_values(self, data: pd.DataFrame) -> None:
        """Validate infinite values for numeric columns."""
        total_rows = len(data)
        for column in data.columns:
            if not pd.api.types.is_numeric_dtype(data[column]):
                continue
            infinite_count = int(np.isinf(data[column]).sum())
            infinite_pct = (infinite_count / total_rows) if total_rows > 0 else 0.0
            if infinite_pct > float(self.config["infinite_threshold"]):
                self.issues.append(
                    ValidationIssue(
                        feature=column,
                        issue_type="infinite_values",
                        level=ValidationLevel.ERROR,
                        description=f"{infinite_pct*100:.2f}% infinite values",
                        count=infinite_count,
                        percentage=infinite_pct,
                        feature_type=self.feature_types.get(column, "unknown"),
                    ),
                )

    def _validate_extreme_values(self, data: pd.DataFrame) -> None:
        """Validate extreme absolute values for numeric columns."""
        total_rows = len(data)
        threshold = float(self.config["extreme_value_threshold"])
        for column in data.columns:
            if not pd.api.types.is_numeric_dtype(data[column]):
                continue
            extreme_count = int((np.abs(data[column]) > threshold).sum())
            extreme_pct = (extreme_count / total_rows) if total_rows > 0 else 0.0
            if extreme_pct > 0.01:
                self.issues.append(
                    ValidationIssue(
                        feature=column,
                        issue_type="extreme_values",
                        level=ValidationLevel.WARNING,
                        description=f"{extreme_pct*100:.2f}% extreme values (> {threshold})",
                        count=extreme_count,
                        percentage=extreme_pct,
                        feature_type=self.feature_types.get(column, "unknown"),
                    ),
                )

    def _validate_constant_values(self, data: pd.DataFrame) -> None:
        """Validate near-constant columns."""
        total_rows = len(data)
        if total_rows == 0:
            return
        constant_threshold = float(self.config["constant_threshold"])
        for column in data.columns:
            try:
                vc = data[column].value_counts(dropna=False)
                if vc.empty:
                    continue
                top_ratio = float(vc.iloc[0]) / float(total_rows)
                if top_ratio >= constant_threshold:
                    self.issues.append(
                        ValidationIssue(
                            feature=column,
                            issue_type="constant_values",
                            level=ValidationLevel.WARNING,
                            description=(
                                f"Near-constant values: top_ratio={top_ratio:.2%} (threshold: {constant_threshold:.0%})"
                            ),
                            count=int(vc.iloc[0]),
                            percentage=top_ratio,
                            feature_type=self.feature_types.get(column, "unknown"),
                        ),
                    )
            except Exception:
                # Non-fatal
                continue

    def _generate_summary(self) -> dict[str, Any]:
        """Generate validation summary."""
        total_issues = len(self.issues)
        critical_issues = sum(1 for i in self.issues if i.level == ValidationLevel.CRITICAL)
        error_issues = sum(1 for i in self.issues if i.level == ValidationLevel.ERROR)
        warning_issues = sum(1 for i in self.issues if i.level == ValidationLevel.WARNING)
        info_issues = sum(1 for i in self.issues if i.level == ValidationLevel.INFO)
        return {
            "total_issues": total_issues,
            "critical_issues": critical_issues,
            "error_issues": error_issues,
            "warning_issues": warning_issues,
            "info_issues": info_issues,
        }

    def _generate_recommendations(self, validation_results: dict[str, Any]) -> list[str]:
        """Generate recommendations based on validation results."""
        recs: list[str] = []
        try:
            if validation_results.get("market_gaps", {}).get("gaps_detected"):
                recs.append("Market gaps detected - consider data interpolation or gap handling")
            if validation_results.get("data_type_fixes"):
                recs.append("Data type issues fixed automatically")

            feature_issues: dict[str, list[ValidationIssue]] = defaultdict(list)
            for issue in self.issues:
                feature_issues[issue.feature_type].append(issue)

            issues = feature_issues.get("wavelet_features", [])
            if any(i.issue_type == "low_variance" for i in issues):
                recs.append("Low variance in wavelet features is expected - consider adjusting thresholds")

            issues = feature_issues.get("multi_timeframe_features", [])
            if any(i.issue_type == "missing_values" for i in issues):
                recs.append("Missing values in multi-timeframe features - check alignment logic")
        except Exception:
            pass
        return recs


def enhanced_validate_features(
    data: pd.DataFrame,
    dataset_name: str = "features",
) -> dict[str, Any]:
    """
    Enhanced validation function with feature-specific thresholds.
    This is the main function to be used in the pipeline.
    """
    validator = EnhancedDataQualityValidator()
    return validator.validate_dataset(data, dataset_name)
