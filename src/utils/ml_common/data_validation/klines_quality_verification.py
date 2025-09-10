"""
Klines Data Quality Verification Module

This module provides comprehensive data quality verification specifically for klines (OHLCV) data,
including timestamp gap detection, duplicate removal, and OHLCV sanity checks.

Key Features:
- Timestamp gap detection (configurable threshold, default based on timeframe)
- True duplicate detection and removal (same timestamp + other columns)
- OHLCV sanity checks (positive values, OHLC consistency, reasonable ranges, outlier detection)
- Volume sanity checks (positive values, reasonable ranges, outlier detection)
- Comprehensive reporting and alerting
- Integration with existing validation framework
- Configurable quality thresholds and actions

Built on existing utilities:
- Uses data_quality.py for base quality assessment
- Leverages validation_utils.py for validation framework
- Integrates with math_validation.py for safe operations
- Uses structured_logging.py for comprehensive logging
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass
from enum import Enum
import warnings

from ...math_validation import safe_divide, MathValidationError
from ...validation_utils import ValidationError
from ..data_quality import DataQualityUtilities
from ...structured_logging import StructuredLogger

logger = logging.getLogger(__name__)


class QualityIssueSeverity(Enum):
    """Severity levels for data quality issues."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class QualityAction(Enum):
    """Actions to take for quality issues."""
    LOG_ONLY = "log_only"
    WARN = "warn"
    REMOVE = "remove"
    FAIL = "fail"


@dataclass
class QualityIssue:
    """Data structure for quality issues."""
    issue_type: str
    severity: QualityIssueSeverity
    message: str
    affected_rows: List[int]
    details: Dict[str, Any]
    action: QualityAction = QualityAction.LOG_ONLY


@dataclass
class QualityReport:
    """Comprehensive quality report."""
    timestamp: datetime
    total_rows: int
    issues: List[QualityIssue]
    summary: Dict[str, Any]
    recommendations: List[str]
    quality_score: float


class KlinesQualityVerifier:
    """Comprehensive klines data quality verification system."""

    def __init__(self, config: Optional[Dict[str, Any]] = None, logger: Optional[logging.Logger] = None):
        """
        Initialize klines quality verifier.

        Args:
            config: Configuration dictionary
            logger: Logger instance
        """
        self.config = config or {}
        self.logger = logger or logging.getLogger(f"{__name__}.KlinesQualityVerifier")
        self.structured_logger = StructuredLogger(self.logger)

        # Get timeframe for gap detection
        self.timeframe = self.config.get('timeframe', '1m')
        self.max_timestamp_gap_multiplier = self.config.get('max_timestamp_gap_multiplier', 2.0)
        
        # Calculate expected gap based on timeframe
        self.expected_gap_seconds = self._get_expected_gap_seconds(self.timeframe)
        self.max_timestamp_gap_seconds = self.expected_gap_seconds * self.max_timestamp_gap_multiplier

        # Quality thresholds
        self.max_duplicate_ratio = self.config.get('max_duplicate_ratio', 0.001)
        self.price_outlier_threshold = self.config.get('price_outlier_threshold', 5.0)  # Z-score
        self.volume_outlier_threshold = self.config.get('volume_outlier_threshold', 5.0)  # Z-score
        self.min_price = self.config.get('min_price', 0.000001)
        self.max_price = self.config.get('max_price', 1000000.0)
        self.min_volume = self.config.get('min_volume', 0.0)
        self.max_volume = self.config.get('max_volume', 1e12)
        self.ohlc_consistency_tolerance = self.config.get('ohlc_consistency_tolerance', 0.001)  # 0.1% tolerance

        # Actions for different issue types
        self.actions = {
            'timestamp_gap': QualityAction(self.config.get('timestamp_gap_action', 'warn')),
            'duplicate': QualityAction(self.config.get('duplicate_action', 'remove')),
            'ohlc_negative': QualityAction(self.config.get('ohlc_negative_action', 'fail')),
            'ohlc_inconsistent': QualityAction(self.config.get('ohlc_inconsistent_action', 'warn')),
            'ohlc_outlier': QualityAction(self.config.get('ohlc_outlier_action', 'warn')),
            'volume_negative': QualityAction(self.config.get('volume_negative_action', 'fail')),
            'volume_outlier': QualityAction(self.config.get('volume_outlier_action', 'warn')),
        }

        # Initialize base quality utilities
        self.base_quality = DataQualityUtilities(config)

        # Required columns for klines
        self.required_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        self.optional_columns = ['quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume']

    def _get_expected_gap_seconds(self, timeframe: str) -> float:
        """Get expected gap in seconds based on timeframe."""
        timeframe_map = {
            '1m': 60,
            '3m': 180,
            '5m': 300,
            '15m': 900,
            '30m': 1800,
            '1h': 3600,
            '2h': 7200,
            '4h': 14400,
            '6h': 21600,
            '8h': 28800,
            '12h': 43200,
            '1d': 86400,
            '3d': 259200,
            '1w': 604800,
            '1M': 2592000  # Approximate
        }
        return timeframe_map.get(timeframe, 60)  # Default to 1 minute

    def verify_klines_quality(self, data: pd.DataFrame, 
                            verify_timestamp_gaps: bool = True,
                            verify_duplicates: bool = True,
                            verify_ohlcv_sanity: bool = True,
                            verify_volume_sanity: bool = True,
                            auto_fix: bool = False) -> Tuple[pd.DataFrame, QualityReport]:
        """
        Comprehensive klines data quality verification.

        Args:
            data: Klines DataFrame to verify
            verify_timestamp_gaps: Whether to check for timestamp gaps
            verify_duplicates: Whether to check for duplicates
            verify_ohlcv_sanity: Whether to check OHLCV sanity
            verify_volume_sanity: Whether to check volume sanity
            auto_fix: Whether to automatically fix issues where possible

        Returns:
            Tuple of (cleaned_data, quality_report)

        Raises:
            ValidationError: For critical quality issues
        """
        self.logger.info("🔍 Starting comprehensive klines quality verification")
        
        # Initialize report
        report = QualityReport(
            timestamp=datetime.now(),
            total_rows=len(data),
            issues=[],
            summary={},
            recommendations=[],
            quality_score=0.0
        )

        # Make a copy for processing
        cleaned_data = data.copy()

        try:
            # 1. Basic data validation
            self._validate_basic_structure(cleaned_data, report)

            # 2. Timestamp gap verification
            if verify_timestamp_gaps:
                cleaned_data = self._verify_timestamp_gaps(cleaned_data, report, auto_fix)

            # 3. Duplicate verification
            if verify_duplicates:
                cleaned_data = self._verify_duplicates(cleaned_data, report, auto_fix)

            # 4. OHLCV sanity checks
            if verify_ohlcv_sanity:
                cleaned_data = self._verify_ohlcv_sanity(cleaned_data, report, auto_fix)

            # 5. Volume sanity checks
            if verify_volume_sanity:
                cleaned_data = self._verify_volume_sanity(cleaned_data, report, auto_fix)

            # 6. Generate summary and recommendations
            self._generate_quality_summary(cleaned_data, report)

            # 7. Calculate quality score
            report.quality_score = self._calculate_quality_score(report)

            # 8. Log results
            self._log_quality_results(report)

            # 9. Handle critical issues
            self._handle_critical_issues(report)

            self.logger.info(f"✅ Klines quality verification completed - Quality score: {report.quality_score:.2f}")
            return cleaned_data, report

        except Exception as e:
            self.logger.error(f"❌ Klines quality verification failed: {e}")
            raise ValidationError(f"Klines quality verification failed: {e}", "quality_verification") from e

    def _validate_basic_structure(self, data: pd.DataFrame, report: QualityReport) -> None:
        """Validate basic data structure and required columns."""
        self.logger.info("📋 Validating basic data structure")

        # Check for required columns
        missing_columns = [col for col in self.required_columns if col not in data.columns]
        if missing_columns:
            issue = QualityIssue(
                issue_type="missing_columns",
                severity=QualityIssueSeverity.CRITICAL,
                message=f"Missing required columns: {missing_columns}",
                affected_rows=[],
                details={"missing_columns": missing_columns},
                action=QualityAction.FAIL
            )
            report.issues.append(issue)
            raise ValidationError(f"Missing required columns: {missing_columns}", "data_structure")

        # Check data types
        if 'timestamp' in data.columns:
            if not pd.api.types.is_datetime64_any_dtype(data['timestamp']):
                try:
                    data['timestamp'] = pd.to_datetime(data['timestamp'])
                except Exception as e:
                    issue = QualityIssue(
                        issue_type="invalid_timestamp_format",
                        severity=QualityIssueSeverity.ERROR,
                        message=f"Cannot convert timestamp column to datetime: {e}",
                        affected_rows=[],
                        details={"error": str(e)},
                        action=QualityAction.FAIL
                    )
                    report.issues.append(issue)
                    raise ValidationError(f"Invalid timestamp format: {e}", "data_structure")

        # Check for empty data
        if len(data) == 0:
            issue = QualityIssue(
                issue_type="empty_data",
                severity=QualityIssueSeverity.CRITICAL,
                message="Data is empty",
                affected_rows=[],
                details={},
                action=QualityAction.FAIL
            )
            report.issues.append(issue)
            raise ValidationError("Data is empty", "data_structure")

        self.logger.info("✅ Basic structure validation passed")

    def _verify_timestamp_gaps(self, data: pd.DataFrame, report: QualityReport, auto_fix: bool) -> pd.DataFrame:
        """Verify timestamp gaps and detect large gaps."""
        self.logger.info(f"⏰ Verifying timestamp gaps (expected: {self.expected_gap_seconds}s, max: {self.max_timestamp_gap_seconds}s)")

        if 'timestamp' not in data.columns:
            self.logger.warning("⚠️ No timestamp column found, skipping gap verification")
            return data

        # Sort by timestamp if not already sorted
        if not data['timestamp'].is_monotonic_increasing:
            data = data.sort_values('timestamp').reset_index(drop=True)

        # Calculate time differences
        time_diffs = data['timestamp'].diff().dt.total_seconds()
        
        # Find large gaps
        large_gaps = time_diffs > self.max_timestamp_gap_seconds
        large_gap_count = large_gaps.sum()
        
        if large_gap_count > 0:
            large_gap_indices = data.index[large_gaps].tolist()
            large_gap_values = time_diffs[large_gaps].tolist()
            
            # Find the largest gaps
            max_gap = time_diffs.max()
            avg_gap = time_diffs.median()
            
            issue = QualityIssue(
                issue_type="timestamp_gap",
                severity=QualityIssueSeverity.WARNING if large_gap_count < len(data) * 0.01 else QualityIssueSeverity.ERROR,
                message=f"Found {large_gap_count} timestamp gaps > {self.max_timestamp_gap_seconds}s (expected: {self.expected_gap_seconds}s)",
                affected_rows=large_gap_indices,
                details={
                    "large_gap_count": int(large_gap_count),
                    "expected_gap_seconds": float(self.expected_gap_seconds),
                    "max_gap_seconds": float(max_gap),
                    "avg_gap_seconds": float(avg_gap),
                    "gap_ratio": float(large_gap_count / len(data)),
                    "largest_gaps": [
                        {"index": int(idx), "gap_seconds": float(gap)}
                        for idx, gap in zip(large_gap_indices[:10], large_gap_values[:10])
                    ]
                },
                action=self.actions['timestamp_gap']
            )
            report.issues.append(issue)
            
            self.logger.warning(f"⚠️ Found {large_gap_count} large timestamp gaps (max: {max_gap:.2f}s)")
            
            # Add recommendation
            report.recommendations.append(
                f"Consider investigating {large_gap_count} timestamp gaps > {self.max_timestamp_gap_seconds}s"
            )

        return data

    def _verify_duplicates(self, data: pd.DataFrame, report: QualityReport, auto_fix: bool) -> pd.DataFrame:
        """Verify and remove true duplicates (same timestamp + other columns)."""
        self.logger.info("🔄 Verifying duplicates")

        # Check for true duplicates (all columns match)
        duplicate_mask = data.duplicated(keep='first')
        duplicate_count = duplicate_mask.sum()
        
        if duplicate_count > 0:
            duplicate_indices = data.index[duplicate_mask].tolist()
            duplicate_ratio = duplicate_count / len(data)
            
            issue = QualityIssue(
                issue_type="duplicate",
                severity=QualityIssueSeverity.WARNING if duplicate_ratio < self.max_duplicate_ratio else QualityIssueSeverity.ERROR,
                message=f"Found {duplicate_count} true duplicates ({duplicate_ratio:.4f} of data)",
                affected_rows=duplicate_indices,
                details={
                    "duplicate_count": int(duplicate_count),
                    "duplicate_ratio": float(duplicate_ratio),
                    "max_allowed_ratio": float(self.max_duplicate_ratio)
                },
                action=self.actions['duplicate']
            )
            report.issues.append(issue)
            
            self.logger.warning(f"⚠️ Found {duplicate_count} true duplicates ({duplicate_ratio:.4f} of data)")
            
            # Auto-fix if enabled and action is remove
            if auto_fix and self.actions['duplicate'] == QualityAction.REMOVE:
                data = data.drop_duplicates(keep='first')
                self.logger.info(f"✅ Removed {duplicate_count} duplicate rows")
                report.recommendations.append(f"Removed {duplicate_count} duplicate rows")
            else:
                report.recommendations.append(f"Consider removing {duplicate_count} duplicate rows")

        # Check for timestamp-only duplicates (potential data quality issue)
        if 'timestamp' in data.columns:
            timestamp_duplicates = data['timestamp'].duplicated(keep=False)
            timestamp_duplicate_count = timestamp_duplicates.sum()
            
            if timestamp_duplicate_count > 0:
                timestamp_duplicate_indices = data.index[timestamp_duplicates].tolist()
                
                issue = QualityIssue(
                    issue_type="timestamp_duplicate",
                    severity=QualityIssueSeverity.WARNING,
                    message=f"Found {timestamp_duplicate_count} rows with duplicate timestamps",
                    affected_rows=timestamp_duplicate_indices,
                    details={
                        "timestamp_duplicate_count": int(timestamp_duplicate_count),
                        "unique_timestamps": int(data['timestamp'].nunique()),
                        "total_rows": int(len(data))
                    },
                    action=QualityAction.WARN
                )
                report.issues.append(issue)
                
                self.logger.warning(f"⚠️ Found {timestamp_duplicate_count} rows with duplicate timestamps")
                report.recommendations.append("Investigate rows with duplicate timestamps - may indicate data collection issues")

        return data

    def _verify_ohlcv_sanity(self, data: pd.DataFrame, report: QualityReport, auto_fix: bool) -> pd.DataFrame:
        """Verify OHLCV data sanity (positive values, OHLC consistency, reasonable ranges, no outliers)."""
        self.logger.info("💰 Verifying OHLCV sanity")

        ohlc_columns = ['open', 'high', 'low', 'close']
        
        # Check for negative or zero prices
        for col in ohlc_columns:
            if col in data.columns:
                invalid_prices = (data[col] <= 0) | data[col].isna()
                invalid_count = invalid_prices.sum()
                
                if invalid_count > 0:
                    invalid_indices = data.index[invalid_prices].tolist()
                    
                    issue = QualityIssue(
                        issue_type="ohlc_negative",
                        severity=QualityIssueSeverity.ERROR,
                        message=f"Found {invalid_count} invalid {col} prices (≤ 0 or NaN)",
                        affected_rows=invalid_indices,
                        details={
                            "column": col,
                            "invalid_count": int(invalid_count),
                            "invalid_ratio": float(invalid_count / len(data)),
                            "min_value": float(data[col].min()) if not data[col].empty else None,
                            "max_value": float(data[col].max()) if not data[col].empty else None
                        },
                        action=self.actions['ohlc_negative']
                    )
                    report.issues.append(issue)
                    
                    self.logger.error(f"❌ Found {invalid_count} invalid {col} prices")
                    
                    # Auto-fix if enabled and action is remove
                    if auto_fix and self.actions['ohlc_negative'] == QualityAction.REMOVE:
                        data = data[~invalid_prices]
                        self.logger.info(f"✅ Removed {invalid_count} rows with invalid {col} prices")
                        report.recommendations.append(f"Removed {invalid_count} rows with invalid {col} prices")
                    else:
                        report.recommendations.append(f"Fix {invalid_count} invalid {col} prices before proceeding")

        # OHLC consistency checks
        if all(col in data.columns for col in ohlc_columns):
            self.logger.info("📊 Verifying OHLC consistency")
            
            # Check high >= max(open, close) and low <= min(open, close)
            ohlc_errors = 0
            ohlc_error_indices = []
            
            for i, row in data.iterrows():
                open_price = row['open']
                high_price = row['high']
                low_price = row['low']
                close_price = row['close']
                
                # Check high >= max(open, close)
                if high_price < max(open_price, close_price) * (1 - self.ohlc_consistency_tolerance):
                    ohlc_errors += 1
                    ohlc_error_indices.append(i)
                
                # Check low <= min(open, close)
                if low_price > min(open_price, close_price) * (1 + self.ohlc_consistency_tolerance):
                    ohlc_errors += 1
                    ohlc_error_indices.append(i)
            
            if ohlc_errors > 0:
                issue = QualityIssue(
                    issue_type="ohlc_inconsistent",
                    severity=QualityIssueSeverity.WARNING,
                    message=f"Found {ohlc_errors} OHLC consistency violations",
                    affected_rows=ohlc_error_indices,
                    details={
                        "ohlc_error_count": int(ohlc_errors),
                        "ohlc_error_ratio": float(ohlc_errors / len(data)),
                        "tolerance": float(self.ohlc_consistency_tolerance)
                    },
                    action=self.actions['ohlc_inconsistent']
                )
                report.issues.append(issue)
                
                self.logger.warning(f"⚠️ Found {ohlc_errors} OHLC consistency violations")
                report.recommendations.append(f"Review {ohlc_errors} OHLC consistency violations")

        # Check for price outliers
        for col in ohlc_columns:
            if col in data.columns:
                valid_prices = data[col][(data[col] > 0) & data[col].notna()]
                if len(valid_prices) > 10:  # Need sufficient data for outlier detection
                    z_scores = np.abs((valid_prices - valid_prices.mean()) / valid_prices.std())
                    outliers = z_scores > self.price_outlier_threshold
                    outlier_count = outliers.sum()
                    
                    if outlier_count > 0:
                        outlier_indices = valid_prices.index[outliers].tolist()
                        
                        issue = QualityIssue(
                            issue_type="ohlc_outlier",
                            severity=QualityIssueSeverity.WARNING,
                            message=f"Found {outlier_count} {col} outliers (Z-score > {self.price_outlier_threshold})",
                            affected_rows=outlier_indices,
                            details={
                                "column": col,
                                "outlier_count": int(outlier_count),
                                "outlier_ratio": float(outlier_count / len(valid_prices)),
                                "max_z_score": float(z_scores.max()),
                                "outlier_values": valid_prices[outliers].tolist()[:10]
                            },
                            action=self.actions['ohlc_outlier']
                        )
                        report.issues.append(issue)
                        
                        self.logger.warning(f"⚠️ Found {outlier_count} {col} outliers")
                        report.recommendations.append(f"Review {outlier_count} {col} outliers for data quality")

        # Check for reasonable price ranges
        for col in ohlc_columns:
            if col in data.columns:
                valid_prices = data[col][(data[col] > 0) & data[col].notna()]
                if len(valid_prices) > 0:
                    min_price = valid_prices.min()
                    max_price = valid_prices.max()
                    
                    if min_price < self.min_price:
                        issue = QualityIssue(
                            issue_type="price_too_low",
                            severity=QualityIssueSeverity.WARNING,
                            message=f"Minimum {col} {min_price} is below threshold {self.min_price}",
                            affected_rows=[],
                            details={"column": col, "min_price": float(min_price), "threshold": float(self.min_price)},
                            action=QualityAction.WARN
                        )
                        report.issues.append(issue)
                        
                    if max_price > self.max_price:
                        issue = QualityIssue(
                            issue_type="price_too_high",
                            severity=QualityIssueSeverity.WARNING,
                            message=f"Maximum {col} {max_price} is above threshold {self.max_price}",
                            affected_rows=[],
                            details={"column": col, "max_price": float(max_price), "threshold": float(self.max_price)},
                            action=QualityAction.WARN
                        )
                        report.issues.append(issue)

        return data

    def _verify_volume_sanity(self, data: pd.DataFrame, report: QualityReport, auto_fix: bool) -> pd.DataFrame:
        """Verify volume data sanity (positive values, reasonable ranges, no outliers)."""
        self.logger.info("📊 Verifying volume sanity")

        if 'volume' not in data.columns:
            self.logger.warning("⚠️ No volume column found, skipping volume verification")
            return data

        volume_data = data['volume']
        
        # Check for negative volumes
        invalid_volumes = (volume_data < 0) | volume_data.isna()
        invalid_count = invalid_volumes.sum()
        
        if invalid_count > 0:
            invalid_indices = data.index[invalid_volumes].tolist()
            
            issue = QualityIssue(
                issue_type="volume_negative",
                severity=QualityIssueSeverity.ERROR,
                message=f"Found {invalid_count} invalid volumes (< 0 or NaN)",
                affected_rows=invalid_indices,
                details={
                    "invalid_count": int(invalid_count),
                    "invalid_ratio": float(invalid_count / len(data)),
                    "min_volume": float(volume_data.min()) if not volume_data.empty else None,
                    "max_volume": float(volume_data.max()) if not volume_data.empty else None
                },
                action=self.actions['volume_negative']
            )
            report.issues.append(issue)
            
            self.logger.error(f"❌ Found {invalid_count} invalid volumes")
            
            # Auto-fix if enabled and action is remove
            if auto_fix and self.actions['volume_negative'] == QualityAction.REMOVE:
                data = data[~invalid_volumes]
                self.logger.info(f"✅ Removed {invalid_count} rows with invalid volumes")
                report.recommendations.append(f"Removed {invalid_count} rows with invalid volumes")
            else:
                report.recommendations.append(f"Fix {invalid_count} invalid volumes before proceeding")

        # Check for volume outliers
        valid_volumes = volume_data[(volume_data >= 0) & volume_data.notna()]
        if len(valid_volumes) > 10:  # Need sufficient data for outlier detection
            z_scores = np.abs((valid_volumes - valid_volumes.mean()) / valid_volumes.std())
            outliers = z_scores > self.volume_outlier_threshold
            outlier_count = outliers.sum()
            
            if outlier_count > 0:
                outlier_indices = valid_volumes.index[outliers].tolist()
                
                issue = QualityIssue(
                    issue_type="volume_outlier",
                    severity=QualityIssueSeverity.WARNING,
                    message=f"Found {outlier_count} volume outliers (Z-score > {self.volume_outlier_threshold})",
                    affected_rows=outlier_indices,
                    details={
                        "outlier_count": int(outlier_count),
                        "outlier_ratio": float(outlier_count / len(valid_volumes)),
                        "max_z_score": float(z_scores.max()),
                        "outlier_volumes": valid_volumes[outliers].tolist()[:10]
                    },
                    action=self.actions['volume_outlier']
                )
                report.issues.append(issue)
                
                self.logger.warning(f"⚠️ Found {outlier_count} volume outliers")
                report.recommendations.append(f"Review {outlier_count} volume outliers for data quality")

        # Check for reasonable volume ranges
        if len(valid_volumes) > 0:
            min_volume = valid_volumes.min()
            max_volume = valid_volumes.max()
            
            if min_volume < self.min_volume:
                issue = QualityIssue(
                    issue_type="volume_too_low",
                    severity=QualityIssueSeverity.WARNING,
                    message=f"Minimum volume {min_volume} is below threshold {self.min_volume}",
                    affected_rows=[],
                    details={"min_volume": float(min_volume), "threshold": float(self.min_volume)},
                    action=QualityAction.WARN
                )
                report.issues.append(issue)
                
            if max_volume > self.max_volume:
                issue = QualityIssue(
                    issue_type="volume_too_high",
                    severity=QualityIssueSeverity.WARNING,
                    message=f"Maximum volume {max_volume} is above threshold {self.max_volume}",
                    affected_rows=[],
                    details={"max_volume": float(max_volume), "threshold": float(self.max_volume)},
                    action=QualityAction.WARN
                )
                report.issues.append(issue)

        return data

    def _generate_quality_summary(self, data: pd.DataFrame, report: QualityReport) -> None:
        """Generate comprehensive quality summary."""
        self.logger.info("📈 Generating quality summary")

        # Count issues by severity
        severity_counts = {}
        for severity in QualityIssueSeverity:
            severity_counts[severity.value] = sum(1 for issue in report.issues if issue.severity == severity)

        # Count issues by type
        issue_type_counts = {}
        for issue in report.issues:
            issue_type_counts[issue.issue_type] = issue_type_counts.get(issue.issue_type, 0) + 1

        # Calculate data statistics
        data_stats = {
            "total_rows": len(data),
            "columns": list(data.columns),
            "memory_usage_mb": data.memory_usage(deep=True).sum() / (1024 * 1024),
            "timeframe": self.timeframe,
            "expected_gap_seconds": self.expected_gap_seconds
        }

        if 'timestamp' in data.columns:
            data_stats.update({
                "time_range": {
                    "start": data['timestamp'].min().isoformat() if not data['timestamp'].empty else None,
                    "end": data['timestamp'].max().isoformat() if not data['timestamp'].empty else None,
                    "duration_hours": (data['timestamp'].max() - data['timestamp'].min()).total_seconds() / 3600 if len(data) > 1 else 0
                }
            })

        # OHLCV statistics
        ohlc_columns = ['open', 'high', 'low', 'close']
        for col in ohlc_columns:
            if col in data.columns:
                col_data = data[col][data[col] > 0]
                if len(col_data) > 0:
                    data_stats[f"{col}_stats"] = {
                        "mean": float(col_data.mean()),
                        "std": float(col_data.std()),
                        "min": float(col_data.min()),
                        "max": float(col_data.max()),
                        "median": float(col_data.median())
                    }

        if 'volume' in data.columns:
            volume_data = data['volume'][data['volume'] >= 0]
            if len(volume_data) > 0:
                data_stats["volume_stats"] = {
                    "mean": float(volume_data.mean()),
                    "std": float(volume_data.std()),
                    "min": float(volume_data.min()),
                    "max": float(volume_data.max()),
                    "median": float(volume_data.median())
                }

        report.summary = {
            "severity_counts": severity_counts,
            "issue_type_counts": issue_type_counts,
            "data_statistics": data_stats,
            "verification_timestamp": report.timestamp.isoformat(),
            "config_used": {
                "timeframe": self.timeframe,
                "max_timestamp_gap_seconds": self.max_timestamp_gap_seconds,
                "max_duplicate_ratio": self.max_duplicate_ratio,
                "price_outlier_threshold": self.price_outlier_threshold,
                "volume_outlier_threshold": self.volume_outlier_threshold,
                "ohlc_consistency_tolerance": self.ohlc_consistency_tolerance
            }
        }

    def _calculate_quality_score(self, report: QualityReport) -> float:
        """Calculate overall quality score (0-1)."""
        if report.total_rows == 0:
            return 0.0

        # Start with perfect score
        score = 1.0

        # Penalize based on issue severity
        severity_penalties = {
            QualityIssueSeverity.INFO: 0.01,
            QualityIssueSeverity.WARNING: 0.05,
            QualityIssueSeverity.ERROR: 0.15,
            QualityIssueSeverity.CRITICAL: 0.5
        }

        for issue in report.issues:
            penalty = severity_penalties.get(issue.severity, 0.1)
            # Scale penalty by affected rows ratio
            affected_ratio = len(issue.affected_rows) / report.total_rows if issue.affected_rows else 0.1
            score -= penalty * affected_ratio

        # Additional penalties for specific issue types
        critical_issues = [issue for issue in report.issues if issue.severity == QualityIssueSeverity.CRITICAL]
        if critical_issues:
            score -= 0.3  # Heavy penalty for critical issues

        return max(0.0, min(1.0, score))

    def _log_quality_results(self, report: QualityReport) -> None:
        """Log comprehensive quality results."""
        self.logger.info("📋 Quality verification results:")
        self.logger.info(f"   Total rows: {report.total_rows}")
        self.logger.info(f"   Quality score: {report.quality_score:.3f}")
        self.logger.info(f"   Issues found: {len(report.issues)}")
        
        # Log issues by severity
        for severity in QualityIssueSeverity:
            count = sum(1 for issue in report.issues if issue.severity == severity)
            if count > 0:
                self.logger.info(f"   {severity.value.title()}: {count}")

        # Log recommendations
        if report.recommendations:
            self.logger.info("📝 Recommendations:")
            for i, rec in enumerate(report.recommendations, 1):
                self.logger.info(f"   {i}. {rec}")

    def _handle_critical_issues(self, report: QualityReport) -> None:
        """Handle critical issues that should cause failure."""
        critical_issues = [issue for issue in report.issues if issue.severity == QualityIssueSeverity.CRITICAL]
        
        if critical_issues:
            error_messages = [issue.message for issue in critical_issues]
            error_msg = f"Critical data quality issues found: {'; '.join(error_messages)}"
            self.logger.error(f"❌ {error_msg}")
            raise ValidationError(error_msg, "critical_quality_issues", {
                "critical_issues": [issue.issue_type for issue in critical_issues],
                "quality_score": report.quality_score
            })

    def get_quality_config_template(self) -> Dict[str, Any]:
        """Get a template configuration for quality verification."""
        return {
            "timeframe": "1m",
            "max_timestamp_gap_multiplier": 2.0,
            "max_duplicate_ratio": 0.001,
            "price_outlier_threshold": 5.0,
            "volume_outlier_threshold": 5.0,
            "min_price": 0.000001,
            "max_price": 1000000.0,
            "min_volume": 0.0,
            "max_volume": 1e12,
            "ohlc_consistency_tolerance": 0.001,
            "timestamp_gap_action": "warn",
            "duplicate_action": "remove",
            "ohlc_negative_action": "fail",
            "ohlc_inconsistent_action": "warn",
            "ohlc_outlier_action": "warn",
            "volume_negative_action": "fail",
            "volume_outlier_action": "warn"
        }

    def export_quality_report(self, report: QualityReport, filepath: str) -> None:
        """Export quality report to JSON file."""
        import json
        
        # Convert report to dictionary
        report_dict = {
            "timestamp": report.timestamp.isoformat(),
            "total_rows": report.total_rows,
            "quality_score": report.quality_score,
            "issues": [
                {
                    "issue_type": issue.issue_type,
                    "severity": issue.severity.value,
                    "message": issue.message,
                    "affected_rows": issue.affected_rows,
                    "details": issue.details,
                    "action": issue.action.value
                }
                for issue in report.issues
            ],
            "summary": report.summary,
            "recommendations": report.recommendations
        }
        
        with open(filepath, 'w') as f:
            json.dump(report_dict, f, indent=2)
        
        self.logger.info(f"📄 Quality report exported to: {filepath}")


# Convenience functions
def verify_klines_quality(data: pd.DataFrame, 
                         config: Optional[Dict[str, Any]] = None,
                         auto_fix: bool = False,
                         logger: Optional[logging.Logger] = None) -> Tuple[pd.DataFrame, QualityReport]:
    """
    Convenience function for klines quality verification.

    Args:
        data: Klines DataFrame to verify
        config: Quality verification configuration
        auto_fix: Whether to automatically fix issues
        logger: Logger instance

    Returns:
        Tuple of (cleaned_data, quality_report)
    """
    verifier = KlinesQualityVerifier(config, logger)
    return verifier.verify_klines_quality(data, auto_fix=auto_fix)


def create_klines_quality_config(**kwargs) -> Dict[str, Any]:
    """
    Create klines quality verification configuration.

    Args:
        **kwargs: Configuration overrides

    Returns:
        Configuration dictionary
    """
    verifier = KlinesQualityVerifier()
    config = verifier.get_quality_config_template()
    config.update(kwargs)
    return config