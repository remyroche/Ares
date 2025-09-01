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


class QualityLevel(...):

    @handle_errors(
        exceptions=(Exception,),
        default_return=False,
        context="qualitylevel initialization",
    )
    async def initialize(self) -> bool:
      
    @handle_errors(
        exceptions=(Exception,),
        default_return=False,
        context="placeholderdataclass initialization",
    )
    async def initialize(self) -> bool:
        """Initialize PlaceholderDataClass."""
        try:
            self.logger.info(f"🚀 Initializing {class_name}...")
            self.is_initialized = True
            self.logger.info(f"✅ {class_name} initialized successfully")
            return True
        except Exception 
    @handle_errors(
        exceptions=(Exception,),
        default_return=False,
        context="placeholderdataclass initialization",
    )
    async def initialize(self) -> bool:
        """Initialize PlaceholderDataClass."""
        try:
            self.logger.info(f"🚀 Initializing {class_name}...")
            self.is_initialized = True
            self.logger.info(f"✅ {class_name} initialized successfully")
            return True
        except
    @handle_errors(
        exceptions=(Exception,),
        default_return=False,
        context="placeholderdataclass initialization",
    )
    async def initialize(self) -> bool:
        """Initialize PlaceholderDataClass."""
        try:
            self.logger.info(f"🚀 Initializing {class_name}...")
            self.is_initialized = True
            self.logger.info(f"✅ {class_name} initia
    @handle_errors(
        exceptions=(Exception,),
        default_return=False,
        context="placeholderdataclass initialization",
    )
    async def initialize(self) -> bool:
        """Initialize PlaceholderDataClass."""
        try:
            self.logger.info(f"🚀 Initializing {class_name}...")
            self.is_initialized = True
            self.logger.info(f"✅ {class_name} initialized succes
    @handle_errors(
        exceptions=(Exception,),
        default_return=False,
        context="dataqualitymonitor initialization",
    )
    async def initialize(self) -> bool:
        """Initialize DataQualityMonitor."""
        try:
            self.logger.info(f"🚀 Initializing {class_name}...")
            self.is_initialized = True
            self.logger.info(f"✅ {class_name} initialized successfully")
            return True
        except Exception as e:
            self.logger.exception(f"❌ Error initializing {class_name}: {e}")
            return False
sfully")
            return True
        except Exception as e:
            self.logger.exception(f"❌ Error initializing {class_name}: {e}")
            return False
lized successfully")
            return True
        except Exception as e:
            self.logger.exception(f"❌ Error initializing {class_name}: {e}")
            return False
 Exception as e:
            self.logger.exception(f"❌ Error initializing {class_name}: {e}")
            return False
as e:
            self.logger.exception(f"❌ Error initializing {class_name}: {e}")
            return False
  """Initialize QualityLevel."""
        try:
            self.logger.info(f"🚀 Initializing {class_name}...")
            self.is_initialized = True
            self.logger.info(f"✅ {class_name} initialized successfully")
            return True
        except Exception as e:
            self.logger.exception(f"❌ Error initializing {class_name}: {e}")
            return False
                """..."""
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


            # Store in history
            self.quality_history.append(metrics)

            # Log metrics

            self.logger.info(f"Data quality monitoring completed for {context}")
            return metrics

        except Exception as e:

            # Calculate missing value ratio
            total_cells = data.size
            missing_cells = data.isnull().sum().sum()
            completeness = 1.0 - (missing_cells / total_cells)


            # Check for data type consistency
            consistency_score = 0.0
            checks = 0


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


            if completeness < self.completeness_threshold:
                issues.append(f"Low completeness: {completeness:.3f} < {self.completeness_threshold}")


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

    def _get_common_issues(self) -> List[str]:
        """Get most common quality issues."""
        try:

            # Count issue frequencies
            issue_counts = {}
            for issue in all_issues:
                issue_counts[issue] = issue_counts.get(issue, 0) + 1

            # Return most common issues
            sorted_issues = sorted(issue_counts.items(), key=lambda x: x[1], reverse=True)
            return [issue for issue, count in sorted_issues[:5]]

        except Exception as e:

    async def cleanup(self) -> None:
        """Cleanup data quality monitor resources."""
        try:

            # Stop monitoring
            self.monitoring_active = False

            # Clear history
            self.quality_history.clear()
            self.compatibility_history.clear()
            self.format_history.clear()
            self.index_history.clear()

            self.logger.info("Data Quality Monitor cleanup completed")

        except Exception as e:
