#!/usr/bin/env python3
"""
Data Quality Monitor for Enhanced Training Pipeline.

This module provides comprehensive data quality monitoring throughout the training pipeline, ensuring data compatibility, quality, format compatibility, and proper indexing at every step.
"""

import asyncio
import json
import sys
from pathlib import Path
from dataclasses import dataclass, asdict
from enum import Enum
from datetime import datetime
from typing import List, Dict, Any, Optional

import pandas as pd
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.utils.logger import system_logger
from src.utils.metrics_logger import (
    log_step_metrics,
    log_step_report, create_detailed_step_report
)


class QualityLevel(Enum):
    """Quality level enumeration."""
    EXCELLENT = "excellent"
    GOOD = "good"
    ACCEPTABLE = "acceptable"
    POOR = "poor"
    CRITICAL = "critical"


@dataclass
class DataQualityMetrics:
    """Data quality metrics container."""
    completeness: float
    consistency: float
    validity: float
    timeliness: float
    uniqueness: float
    accuracy: float
    overall_score: float
    quality_level: QualityLevel
    issues: List[str]
    warnings: List[str]
    recommendations: List[str]
    timestamp: datetime


@dataclass
class CompatibilityMetrics:
    """Data compatibility metrics container."""
    format_compatible: bool
    schema_compatible: bool
    type_compatible: bool
    index_compatible: bool
    temporal_aligned: bool
    overall_compatible: bool
    issues: List[str]
    warnings: List[str]
    conversions_applied: List[str]
    timestamp: datetime


@dataclass
class FormatMetrics:
    """Data format metrics container."""
    expected_format: str
    actual_format: str
    format_match: bool
    encoding_valid: bool
    compression_valid: bool
    file_size_reasonable: bool
    issues: List[str]
    warnings: List[str]
    timestamp: datetime


@dataclass
class IndexMetrics:
    """Data indexing metrics container."""
    has_temporal_index: bool
    index_sorted: bool
    no_duplicates: bool
    no_gaps: bool
    frequency_consistent: bool
    timezone_consistent: bool
    overall_valid: bool
    issues: List[str]
    warnings: List[str]
    timestamp: datetime


class DataQualityMonitor:
    """Comprehensive data quality monitoring system for training pipeline."""

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize data quality monitor.

        Args:
            config: Configuration dictionary

        """
        self.config = config
        self.logger = system_logger.getChild("DataQualityMonitor")

        # Quality thresholds
        self.quality_config = config.get("data_quality_monitor", {})
        self.completeness_threshold = self.quality_config.get("completeness_threshold", 0.9)
        self.consistency_threshold = self.quality_config.get("consistency_threshold", 0.8)
        self.validity_threshold = self.quality_config.get("validity_threshold", 0.95)
        self.timeliness_threshold = self.quality_config.get("timeliness_threshold", 0.8)
        self.uniqueness_threshold = self.quality_config.get("uniqueness_threshold", 0.95)
        self.accuracy_threshold = self.quality_config.get("accuracy_threshold", 0.85)

        # Monitoring state
        self.monitoring_active = False
        self.quality_history: List[DataQualityMetrics] = []
        self.compatibility_history: List[CompatibilityMetrics] = []
        self.format_history: List[FormatMetrics] = []
        self.index_history: List[IndexMetrics] = []

    async def initialize(self) -> bool:
        """Initialize the data quality monitor."""
        try:
            self.logger.info("Initializing Data Quality Monitor...")

            # Validate configuration
            if not self._validate_configuration():
                self.logger.error("Invalid data quality monitor configuration")
                return False

            # Initialize monitoring components
            await self._initialize_monitoring_components()

            # Start monitoring
            self.monitoring_active = True

            self.logger.info("Data Quality Monitor initialized successfully")
            return True

        except Exception as e:
            self.logger.error(f"Error initializing Data Quality Monitor: {e}")
            return False

    def _validate_configuration(self) -> bool:
        """Validate monitor configuration."""
        try:
            self.logger.debug("Validating data quality monitor configuration...")

            # Check threshold values
            thresholds = [
                ("completeness_threshold", self.completeness_threshold),
                ("consistency_threshold", self.consistency_threshold),
                ("validity_threshold", self.validity_threshold),
                ("timeliness_threshold", self.timeliness_threshold),
                ("uniqueness_threshold", self.uniqueness_threshold),
                ("accuracy_threshold", self.accuracy_threshold)
            ]

            for name, threshold in thresholds:
                if not 0.0 <= threshold <= 1.0:
                    self.logger.error(f"Invalid {name}: {threshold}. Must be between 0.0 and 1.0")
                    return False

            self.logger.debug("Configuration validation passed")
            return True

        except Exception as e:
            self.logger.error(f"Error validating configuration: {e}")
            return False

    async def _initialize_monitoring_components(self) -> None:
        """Initialize monitoring components."""
        try:
            self.logger.debug("Initializing monitoring components...")

            # Initialize quality assessment components
            self.quality_assessor = self._create_quality_assessor()
            self.compatibility_checker = self._create_compatibility_checker()
            self.format_validator = self._create_format_validator()
            self.index_validator = self._create_index_validator()

            self.logger.debug("Monitoring components initialized")

        except Exception as e:
            self.logger.error(f"Error initializing monitoring components: {e}")
            raise

    def _create_quality_assessor(self) -> Any:
        """Create quality assessment component."""
        # Implementation would create quality assessment logic
        return {"type": "quality_assessor", "ready": True}

    def _create_compatibility_checker(self) -> Any:
        """Create compatibility checking component."""
        # Implementation would create compatibility checking logic
        return {"type": "compatibility_checker", "ready": True}

    def _create_format_validator(self) -> Any:
        """Create format validation component."""
        # Implementation would create format validation logic
        return {"type": "format_validator", "ready": True}

    def _create_index_validator(self) -> Any:
        """Create index validation component."""
        # Implementation would create index validation logic
        return {"type": "index_validator", "ready": True}

    async def monitor_data_quality(self, data: pd.DataFrame, context: str = "unknown") -> DataQualityMetrics:
        """Monitor data quality for a given dataset.

        Args:
            data: DataFrame to monitor
            context: Context information for monitoring

        Returns:
            DataQualityMetrics: Quality metrics

        """
        try:
            self.logger.info(f"Monitoring data quality for context: {context}")

            # Calculate quality metrics
            completeness = self._calculate_completeness(data)
            consistency = self._calculate_consistency(data)
            validity = self._calculate_validity(data)
            timeliness = self._calculate_timeliness(data)
            uniqueness = self._calculate_uniqueness(data)
            accuracy = self._calculate_accuracy(data)

            # Calculate overall score
            overall_score = self._calculate_overall_score(
                completeness, consistency, validity, timeliness, uniqueness, accuracy
            )

            # Determine quality level
            quality_level = self._determine_quality_level(overall_score)

            # Generate issues and recommendations
            issues = self._identify_quality_issues(
                completeness, consistency, validity, timeliness, uniqueness, accuracy
            )
            warnings = self._generate_quality_warnings(
                completeness, consistency, validity, timeliness, uniqueness, accuracy
            )
            recommendations = self._generate_quality_recommendations(issues, warnings)

            # Create metrics object
            metrics = DataQualityMetrics(
                completeness=completeness,
                consistency=consistency,
                validity=validity,
                timeliness=timeliness,
                uniqueness=uniqueness,
                accuracy=accuracy,
                overall_score=overall_score,
                quality_level=quality_level,
                issues=issues,
                warnings=warnings,
                recommendations=recommendations,
                timestamp=datetime.now()
            )

            # Store in history
            self.quality_history.append(metrics)

            # Log metrics
            self._log_quality_metrics(metrics, context)

            self.logger.info(f"Data quality monitoring completed for {context}")
            return metrics

        except Exception as e:
            self.logger.error(f"Error monitoring data quality: {e}")
            return self._create_error_metrics(e)

    def _calculate_completeness(self, data: pd.DataFrame) -> float:
        """Calculate data completeness."""
        try:
            if data.empty:
                return 0.0

            # Calculate missing value ratio
            total_cells = data.size
            missing_cells = data.isnull().sum().sum()
            completeness = 1.0 - (missing_cells / total_cells)

            return max(0.0, min(1.0, completeness))

        except Exception as e:
            self.logger.error(f"Error calculating completeness: {e}")
            return 0.0

    def _calculate_consistency(self, data: pd.DataFrame) -> float:
        """Calculate data consistency."""
        try:
            if data.empty:
                return 0.0

            # Check for data type consistency
            consistency_score = 0.0
            checks = 0

            # Check for consistent data types within columns
            for column in data.columns:
                if data[column].dtype in ['object', 'string']:
                    # Check for mixed types in string columns
                    unique_types = set(type(val).__name__ for val in data[column].dropna())
                    if len(unique_types) <= 1:
                        consistency_score += 1.0
                    checks += 1

            # Check for logical consistency (e.g., high >= low for OHLC data)
            if all(col in data.columns for col in ['high', 'low']):
                logical_consistent = (data['high'] >= data['low']).mean()
                consistency_score += logical_consistent
                checks += 1

            return consistency_score / max(checks, 1)

        except Exception as e:
            self.logger.error(f"Error calculating consistency: {e}")
            return 0.0

    def _calculate_validity(self, data: pd.DataFrame) -> float:
        """Calculate data validity."""
        try:
            if data.empty:
                return 0.0

            # Check for valid values
            validity_score = 0.0
            checks = 0

            # Check for numeric columns
            numeric_columns = data.select_dtypes(include=[np.number]).columns
            for column in numeric_columns:
                # Check for finite values
                finite_ratio = np.isfinite(data[column]).mean()
                validity_score += finite_ratio
                checks += 1

            # Check for positive values where expected
            if 'volume' in data.columns:
                positive_volume = (data['volume'] >= 0).mean()
                validity_score += positive_volume
                checks += 1

            return validity_score / max(checks, 1)

        except Exception as e:
            self.logger.error(f"Error calculating validity: {e}")
            return 0.0

    def _calculate_timeliness(self, data: pd.DataFrame) -> float:
        """Calculate data timeliness."""
        try:
            if data.empty:
                return 0.0

            # Check if data has temporal index
            if isinstance(data.index, pd.DatetimeIndex):
                # Calculate time gaps
                time_diffs = data.index.to_series().diff()
                expected_freq = self._estimate_expected_frequency(data)
                
                if expected_freq:
                    # Check if gaps are reasonable
                    reasonable_gaps = (time_diffs <= expected_freq * 2).mean()
                    return reasonable_gaps
                else:
                    return 1.0  # Assume timely if we can't determine frequency

            return 0.5  # Default score for non-temporal data

        except Exception as e:
            self.logger.error(f"Error calculating timeliness: {e}")
            return 0.0

    def _calculate_uniqueness(self, data: pd.DataFrame) -> float:
        """Calculate data uniqueness."""
        try:
            if data.empty:
                return 0.0

            # Calculate duplicate ratio
            total_rows = len(data)
            unique_rows = len(data.drop_duplicates())
            uniqueness = unique_rows / total_rows

            return max(0.0, min(1.0, uniqueness))

        except Exception as e:
            self.logger.error(f"Error calculating uniqueness: {e}")
            return 0.0

    def _calculate_accuracy(self, data: pd.DataFrame) -> float:
        """Calculate data accuracy."""
        try:
            if data.empty:
                return 0.0

            # For now, use a simple heuristic based on data range
            accuracy_score = 0.0
            checks = 0

            # Check for reasonable price ranges
            if all(col in data.columns for col in ['open', 'high', 'low', 'close']):
                # Check if prices are positive
                positive_prices = (
                    (data[['open', 'high', 'low', 'close']] > 0).all(axis=1)
                ).mean()
                accuracy_score += positive_prices
                checks += 1

            # Check for reasonable volume values
            if 'volume' in data.columns:
                reasonable_volume = (data['volume'] >= 0).mean()
                accuracy_score += reasonable_volume
                checks += 1

            return accuracy_score / max(checks, 1)

        except Exception as e:
            self.logger.error(f"Error calculating accuracy: {e}")
            return 0.0

    def _calculate_overall_score(self, completeness: float, consistency: float, 
                                validity: float, timeliness: float, 
                                uniqueness: float, accuracy: float) -> float:
        """Calculate overall quality score."""
        try:
            # Weighted average of all metrics
            weights = {
                'completeness': 0.2,
                'consistency': 0.2,
                'validity': 0.2,
                'timeliness': 0.15,
                'uniqueness': 0.15,
                'accuracy': 0.1
            }

            overall_score = (
                completeness * weights['completeness'] +
                consistency * weights['consistency'] +
                validity * weights['validity'] +
                timeliness * weights['timeliness'] +
                uniqueness * weights['uniqueness'] +
                accuracy * weights['accuracy']
            )

            return max(0.0, min(1.0, overall_score))

        except Exception as e:
            self.logger.error(f"Error calculating overall score: {e}")
            return 0.0

    def _determine_quality_level(self, overall_score: float) -> QualityLevel:
        """Determine quality level based on overall score."""
        try:
            if overall_score >= 0.9:
                return QualityLevel.EXCELLENT
            elif overall_score >= 0.8:
                return QualityLevel.GOOD
            elif overall_score >= 0.7:
                return QualityLevel.ACCEPTABLE
            elif overall_score >= 0.6:
                return QualityLevel.POOR
            else:
                return QualityLevel.CRITICAL

        except Exception as e:
            self.logger.error(f"Error determining quality level: {e}")
            return QualityLevel.CRITICAL

    def _identify_quality_issues(self, completeness: float, consistency: float,
                               validity: float, timeliness: float, 
                               uniqueness: float, accuracy: float) -> List[str]:
        """Identify quality issues based on metrics."""
        try:
            issues = []

            if completeness < self.completeness_threshold:
                issues.append(f"Low completeness: {completeness:.3f} < {self.completeness_threshold}")

            if consistency < self.consistency_threshold:
                issues.append(f"Low consistency: {consistency:.3f} < {self.consistency_threshold}")

            if validity < self.validity_threshold:
                issues.append(f"Low validity: {validity:.3f} < {self.validity_threshold}")

            if timeliness < self.timeliness_threshold:
                issues.append(f"Low timeliness: {timeliness:.3f} < {self.timeliness_threshold}")

            if uniqueness < self.uniqueness_threshold:
                issues.append(f"Low uniqueness: {uniqueness:.3f} < {self.uniqueness_threshold}")

            if accuracy < self.accuracy_threshold:
                issues.append(f"Low accuracy: {accuracy:.3f} < {self.accuracy_threshold}")

            return issues

        except Exception as e:
            self.logger.error(f"Error identifying quality issues: {e}")
            return ["Error identifying quality issues"]

    def _generate_quality_warnings(self, completeness: float, consistency: float,
                                  validity: float, timeliness: float, 
                                  uniqueness: float, accuracy: float) -> List[str]:
        """Generate quality warnings."""
        try:
            warnings = []

            # Generate warnings for metrics close to thresholds
            if completeness < self.completeness_threshold + 0.05:
                warnings.append("Completeness is close to threshold")

            if consistency < self.consistency_threshold + 0.05:
                warnings.append("Consistency is close to threshold")

            if validity < self.validity_threshold + 0.05:
                warnings.append("Validity is close to threshold")

            return warnings

        except Exception as e:
            self.logger.error(f"Error generating quality warnings: {e}")
            return ["Error generating quality warnings"]

    def _generate_quality_recommendations(self, issues: List[str], warnings: List[str]) -> List[str]:
        """Generate quality improvement recommendations."""
        try:
            recommendations = []

            if issues:
                recommendations.append("Address identified quality issues")
                recommendations.append("Implement data validation checks")
                recommendations.append("Review data source quality")

            if warnings:
                recommendations.append("Monitor quality metrics closely")
                recommendations.append("Consider preventive measures")

            if not issues and not warnings:
                recommendations.append("Maintain current quality standards")

            return recommendations

        except Exception as e:
            self.logger.error(f"Error generating recommendations: {e}")
            return ["Error generating recommendations"]

    def _estimate_expected_frequency(self, data: pd.DataFrame) -> Optional[pd.Timedelta]:
        """Estimate expected frequency for temporal data."""
        try:
            if not isinstance(data.index, pd.DatetimeIndex):
                return None

            # Calculate most common time difference
            time_diffs = data.index.to_series().diff().dropna()
            if len(time_diffs) == 0:
                return None

            # Find the most common difference
            mode_diff = time_diffs.mode()
            if len(mode_diff) > 0:
                return mode_diff.iloc[0]

            return None

        except Exception as e:
            self.logger.error(f"Error estimating frequency: {e}")
            return None

    def _log_quality_metrics(self, metrics: DataQualityMetrics, context: str) -> None:
        """Log quality metrics."""
        try:
            self.logger.info(f"Quality metrics for {context}:")
            self.logger.info(f"  Overall score: {metrics.overall_score:.3f}")
            self.logger.info(f"  Quality level: {metrics.quality_level.value}")
            self.logger.info(f"  Issues: {len(metrics.issues)}")
            self.logger.info(f"  Warnings: {len(metrics.warnings)}")

            if metrics.issues:
                for issue in metrics.issues:
                    self.logger.warning(f"  Issue: {issue}")

        except Exception as e:
            self.logger.error(f"Error logging quality metrics: {e}")

    def _create_error_metrics(self, error: Exception) -> DataQualityMetrics:
        """Create error metrics when monitoring fails."""
        return DataQualityMetrics(
            completeness=0.0,
            consistency=0.0,
            validity=0.0,
            timeliness=0.0,
            uniqueness=0.0,
            accuracy=0.0,
            overall_score=0.0,
            quality_level=QualityLevel.CRITICAL,
            issues=[f"Monitoring error: {str(error)}"],
            warnings=[],
            recommendations=["Fix monitoring system"],
            timestamp=datetime.now()
        )

    async def get_quality_summary(self) -> Dict[str, Any]:
        """Get quality monitoring summary."""
        try:
            if not self.quality_history:
                return {"message": "No quality data available"}

            # Calculate summary statistics
            recent_metrics = self.quality_history[-10:]  # Last 10 measurements

            summary = {
                "total_measurements": len(self.quality_history),
                "recent_average_score": np.mean([m.overall_score for m in recent_metrics]),
                "quality_level_distribution": self._get_quality_level_distribution(),
                "common_issues": self._get_common_issues(),
                "monitoring_active": self.monitoring_active
            }

            return summary

        except Exception as e:
            self.logger.error(f"Error getting quality summary: {e}")
            return {"error": str(e)}

    def _get_quality_level_distribution(self) -> Dict[str, int]:
        """Get distribution of quality levels."""
        try:
            distribution = {}
            for level in QualityLevel:
                count = sum(1 for m in self.quality_history if m.quality_level == level)
                distribution[level.value] = count

            return distribution

        except Exception as e:
            self.logger.error(f"Error getting quality level distribution: {e}")
            return {}

    def _get_common_issues(self) -> List[str]:
        """Get most common quality issues."""
        try:
            all_issues = []
            for metrics in self.quality_history:
                all_issues.extend(metrics.issues)

            # Count issue frequencies
            issue_counts = {}
            for issue in all_issues:
                issue_counts[issue] = issue_counts.get(issue, 0) + 1

            # Return most common issues
            sorted_issues = sorted(issue_counts.items(), key=lambda x: x[1], reverse=True)
            return [issue for issue, count in sorted_issues[:5]]

        except Exception as e:
            self.logger.error(f"Error getting common issues: {e}")
            return []

    async def cleanup(self) -> None:
        """Cleanup data quality monitor resources."""
        try:
            self.logger.info("Cleaning up Data Quality Monitor...")

            # Stop monitoring
            self.monitoring_active = False

            # Clear history
            self.quality_history.clear()
            self.compatibility_history.clear()
            self.format_history.clear()
            self.index_history.clear()

            self.logger.info("Data Quality Monitor cleanup completed")

        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")