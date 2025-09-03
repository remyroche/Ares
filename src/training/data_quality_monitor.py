#!/usr/bin/env python3
"""
Data Quality Monitor for Enhanced Training Pipeline.

This module provides comprehensive data quality monitoring throughout the training pipeline,
ensuring data compatibility, quality, format compatibility, and proper indexing at every step.
"""

import asyncio
import json
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum

import pandas as pd
import numpy as np
import psutil

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.utils.logger import system_logger
from src.utils.enhanced_mlflow_integration import (
    log_step_metrics,
    log_step_report,
    create_detailed_step_report
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
    """
    Comprehensive data quality monitor for the training pipeline.
    
    This class provides:
    - Real-time data quality monitoring
    - Compatibility validation
    - Format verification
    - Index validation
    - Continuous quality scoring
    - Automated issue detection and reporting
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = system_logger.getChild("DataQualityMonitor")
        self.quality_history: List[DataQualityMetrics] = []
        self.compatibility_history: List[CompatibilityMetrics] = []
        self.format_history: List[FormatMetrics] = []
        self.index_history: List[IndexMetrics] = []
        
        # Quality thresholds
        self.quality_thresholds = {
            "excellent": 0.95,
            "good": 0.85,
            "acceptable": 0.75,
            "poor": 0.60,
            "critical": 0.50
        }
        
        # Monitoring configuration
        self.monitor_config = config.get("data_quality_monitor", {})
        self.enable_real_time_monitoring = self.monitor_config.get("enable_real_time_monitoring", True)
        self.alert_threshold = self.monitor_config.get("alert_threshold", 0.8)
        self.auto_fix_enabled = self.monitor_config.get("auto_fix_enabled", False)
        
        self.logger.info("🔍 Data Quality Monitor initialized")

    async def monitor_data_quality(self, data: Any, step_name: str, context: Dict[str, Any] = None) -> DataQualityMetrics:
        """Monitor data quality for a specific step."""
        self.logger.info(f"🔍 Monitoring data quality for {step_name}")
        
        if context is None:
            context = {}
        
        try:
            if isinstance(data, pd.DataFrame):
                metrics = await self._analyze_dataframe_quality(data, step_name, context)
            elif isinstance(data, dict):
                metrics = await self._analyze_dict_quality(data, step_name, context)
            else:
                metrics = await self._analyze_generic_quality(data, step_name, context)
            
            # Store in history
            self.quality_history.append(metrics)
            
            # Log metrics
            await self._log_quality_metrics(step_name, metrics)
            
            # Check for alerts
            if metrics.overall_score < self.alert_threshold:
                await self._trigger_quality_alert(step_name, metrics)
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"❌ Error monitoring data quality for {step_name}: {e}")
            # Return critical quality metrics
            return DataQualityMetrics(
                completeness=0.0,
                consistency=0.0,
                validity=0.0,
                timeliness=0.0,
                uniqueness=0.0,
                accuracy=0.0,
                overall_score=0.0,
                quality_level=QualityLevel.CRITICAL,
                issues=[f"Quality monitoring error: {str(e)}"],
                warnings=[],
                recommendations=["Check data structure and format"],
                timestamp=datetime.now()
            )

    async def _analyze_dataframe_quality(self, data: pd.DataFrame, step_name: str, context: Dict[str, Any]) -> DataQualityMetrics:
        """Analyze quality of DataFrame data."""
        issues = []
        warnings = []
        recommendations = []
        
        # Completeness
        completeness = self._calculate_completeness(data)
        if completeness < 0.95:
            issues.append(f"Low completeness: {completeness:.3f}")
            recommendations.append("Consider data imputation or collection")
        
        # Consistency
        consistency = self._calculate_consistency(data)
        if consistency < 0.9:
            issues.append(f"Low consistency: {consistency:.3f}")
            recommendations.append("Check data relationships and constraints")
        
        # Validity
        validity = self._calculate_validity(data)
        if validity < 0.95:
            issues.append(f"Low validity: {validity:.3f}")
            recommendations.append("Validate data ranges and formats")
        
        # Timeliness
        timeliness = self._calculate_timeliness(data, context)
        if timeliness < 0.9:
            warnings.append(f"Timeliness concern: {timeliness:.3f}")
        
        # Uniqueness
        uniqueness = self._calculate_uniqueness(data)
        if uniqueness < 0.95:
            issues.append(f"Low uniqueness: {uniqueness:.3f}")
            recommendations.append("Check for duplicate records")
        
        # Accuracy
        accuracy = self._calculate_accuracy(data)
        if accuracy < 0.9:
            warnings.append(f"Accuracy concern: {accuracy:.3f}")
        
        # Overall score
        overall_score = np.mean([completeness, consistency, validity, timeliness, uniqueness, accuracy])
        
        # Determine quality level
        quality_level = self._determine_quality_level(overall_score)
        
        return DataQualityMetrics(
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

    def _calculate_completeness(self, data: pd.DataFrame) -> float:
        """Calculate data completeness score."""
        try:
            if data.empty:
                return 0.0
            
            # Calculate missing value ratio
            missing_ratio = data.isnull().sum().sum() / (len(data) * len(data.columns))
            return 1.0 - missing_ratio
            
        except Exception:
            return 0.0

    def _calculate_consistency(self, data: pd.DataFrame) -> float:
        """Calculate data consistency score."""
        try:
            consistency_checks = []
            
            # Check price relationships if OHLC data
            if all(col in data.columns for col in ["open", "high", "low", "close"]):
                price_consistency = (
                    (data["high"] >= data["low"]) &
                    (data["high"] >= data["open"]) &
                    (data["high"] >= data["close"]) &
                    (data["low"] <= data["open"]) &
                    (data["low"] <= data["close"])
                ).mean()
                consistency_checks.append(price_consistency)
            
            # Check volume consistency
            if "volume" in data.columns:
                volume_consistency = (data["volume"] >= 0).mean()
                consistency_checks.append(volume_consistency)
            
            # Check timestamp consistency
            if "timestamp" in data.columns:
                timestamp_consistency = data["timestamp"].is_monotonic_increasing
                consistency_checks.append(float(timestamp_consistency))
            
            return np.mean(consistency_checks) if consistency_checks else 1.0
            
        except Exception:
            return 0.5

    def _calculate_validity(self, data: pd.DataFrame) -> float:
        """Calculate data validity score."""
        try:
            validity_checks = []
            
            # Check for negative prices
            if all(col in data.columns for col in ["open", "high", "low", "close"]):
                price_validity = (data[["open", "high", "low", "close"]] > 0).all(axis=1).mean()
                validity_checks.append(price_validity)
            
            # Check for reasonable price ranges
            if "close" in data.columns:
                price_range_validity = (
                    (data["close"] > 0) & (data["close"] < 1e6)
                ).mean()
                validity_checks.append(price_range_validity)
            
            # Check for reasonable volumes
            if "volume" in data.columns:
                volume_validity = (
                    (data["volume"] >= 0) & (data["volume"] < 1e12)
                ).mean()
                validity_checks.append(volume_validity)
            
            return np.mean(validity_checks) if validity_checks else 1.0
            
        except Exception:
            return 0.5

    def _calculate_timeliness(self, data: pd.DataFrame, context: Dict[str, Any]) -> float:
        """Calculate data timeliness score."""
        try:
            if "timestamp" not in data.columns:
                return 0.5
            
            # Check if data is recent
            latest_timestamp = data["timestamp"].max()
            current_time = pd.Timestamp.now()
            
            # Calculate age in hours
            age_hours = (current_time - latest_timestamp).total_seconds() / 3600
            
            # Score based on age (fresher data gets higher score)
            if age_hours < 1:
                return 1.0
            elif age_hours < 24:
                return 0.9
            elif age_hours < 168:  # 1 week
                return 0.7
            else:
                return 0.5
                
        except Exception:
            return 0.5

    def _calculate_uniqueness(self, data: pd.DataFrame) -> float:
        """Calculate data uniqueness score."""
        try:
            if data.empty:
                return 0.0
            
            # Check for duplicate rows
            duplicate_ratio = data.duplicated().mean()
            return 1.0 - duplicate_ratio
            
        except Exception:
            return 0.5

    def _calculate_accuracy(self, data: pd.DataFrame) -> float:
        """Calculate data accuracy score."""
        try:
            accuracy_checks = []
            
            # Check for extreme outliers
            if "close" in data.columns:
                q1 = data["close"].quantile(0.25)
                q3 = data["close"].quantile(0.75)
                iqr = q3 - q1
                outlier_ratio = (
                    (data["close"] < q1 - 1.5 * iqr) | 
                    (data["close"] > q3 + 1.5 * iqr)
                ).mean()
                accuracy_checks.append(1.0 - outlier_ratio)
            
            # Check for reasonable price movements
            if "close" in data.columns and len(data) > 1:
                returns = data["close"].pct_change().abs()
                extreme_moves = (returns > 0.5).mean()  # 50% moves
                accuracy_checks.append(1.0 - extreme_moves)
            
            return np.mean(accuracy_checks) if accuracy_checks else 1.0
            
        except Exception:
            return 0.5

    def _determine_quality_level(self, score: float) -> QualityLevel:
        """Determine quality level based on score."""
        if score >= self.quality_thresholds["excellent"]:
            return QualityLevel.EXCELLENT
        elif score >= self.quality_thresholds["good"]:
            return QualityLevel.GOOD
        elif score >= self.quality_thresholds["acceptable"]:
            return QualityLevel.ACCEPTABLE
        elif score >= self.quality_thresholds["poor"]:
            return QualityLevel.POOR
        else:
            return QualityLevel.CRITICAL

    async def _analyze_dict_quality(self, data: Dict[str, Any], step_name: str, context: Dict[str, Any]) -> DataQualityMetrics:
        """Analyze quality of dictionary data."""
        issues = []
        warnings = []
        recommendations = []
        
        # Check for required keys
        required_keys = self._get_required_keys_for_step(step_name)
        missing_keys = [key for key in required_keys if key not in data]
        
        if missing_keys:
            issues.append(f"Missing required keys: {missing_keys}")
            recommendations.append("Ensure all required keys are present")
        
        # Calculate completeness based on key presence
        completeness = 1.0 - (len(missing_keys) / len(required_keys)) if required_keys else 1.0
        
        # Other metrics for dict data
        consistency = 1.0
        validity = 1.0
        timeliness = 1.0
        uniqueness = 1.0
        accuracy = 1.0
        
        overall_score = np.mean([completeness, consistency, validity, timeliness, uniqueness, accuracy])
        quality_level = self._determine_quality_level(overall_score)
        
        return DataQualityMetrics(
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

    async def _analyze_generic_quality(self, data: Any, step_name: str, context: Dict[str, Any]) -> DataQualityMetrics:
        """Analyze quality of generic data."""
        issues = []
        warnings = []
        recommendations = []
        
        # Basic quality checks for generic data
        completeness = 1.0 if data is not None else 0.0
        consistency = 1.0
        validity = 1.0
        timeliness = 1.0
        uniqueness = 1.0
        accuracy = 1.0
        
        if data is None:
            issues.append("Data is None")
            recommendations.append("Ensure data is properly loaded")
        
        overall_score = np.mean([completeness, consistency, validity, timeliness, uniqueness, accuracy])
        quality_level = self._determine_quality_level(overall_score)
        
        return DataQualityMetrics(
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

    def _get_required_keys_for_step(self, step_name: str) -> List[str]:
        """Get required keys for a specific step."""
        key_requirements = {
            "step01": ["symbol", "exchange", "timeframe", "data_dir"],
            "step1_5": ["symbol", "exchange", "timeframe", "data_dir"],
            "step02": ["symbol", "exchange", "timeframe", "data_dir"],
            "step03": ["symbol", "exchange", "timeframe", "data_dir"],
            "step04": ["symbol", "exchange", "timeframe", "data_dir"],
            "step05": ["symbol", "exchange", "timeframe", "data_dir"],
            "step06": ["symbol", "exchange", "timeframe", "data_dir"],
            "step07": ["symbol", "exchange", "timeframe", "data_dir"]
        }
        return key_requirements.get(step_name, [])

    async def monitor_compatibility(self, data: Any, step_name: str, expected_format: str = None) -> CompatibilityMetrics:
        """Monitor data compatibility for a specific step."""
        self.logger.info(f"🔍 Monitoring data compatibility for {step_name}")
        
        try:
            if isinstance(data, pd.DataFrame):
                metrics = self._analyze_dataframe_compatibility(data, step_name, expected_format)
            elif isinstance(data, dict):
                metrics = self._analyze_dict_compatibility(data, step_name, expected_format)
            else:
                metrics = self._analyze_generic_compatibility(data, step_name, expected_format)
            
            # Store in history
            self.compatibility_history.append(metrics)
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"❌ Error monitoring compatibility for {step_name}: {e}")
            return CompatibilityMetrics(
                format_compatible=False,
                schema_compatible=False,
                type_compatible=False,
                index_compatible=False,
                temporal_aligned=False,
                overall_compatible=False,
                issues=[f"Compatibility monitoring error: {str(e)}"],
                warnings=[],
                conversions_applied=[],
                timestamp=datetime.now()
            )

    def _analyze_dataframe_compatibility(self, data: pd.DataFrame, step_name: str, expected_format: str) -> CompatibilityMetrics:
        """Analyze DataFrame compatibility."""
        issues = []
        warnings = []
        conversions_applied = []
        
        # Check required columns
        required_columns = self._get_required_columns_for_step(step_name)
        missing_columns = [col for col in required_columns if col not in data.columns]
        schema_compatible = len(missing_columns) == 0
        
        if missing_columns:
            issues.append(f"Missing required columns: {missing_columns}")
        
        # Check data types
        type_issues = self._check_data_types(data, step_name)
        type_compatible = len(type_issues) == 0
        
        if type_issues:
            warnings.extend(type_issues)
        
        # Check indexing
        index_issues = self._check_indexing(data, step_name)
        index_compatible = len(index_issues) == 0
        
        if index_issues:
            warnings.extend(index_issues)
        
        # Check temporal alignment
        temporal_aligned = self._check_temporal_alignment(data)
        if not temporal_aligned:
            warnings.append("Temporal alignment issues detected")
        
        # Check format compatibility
        format_compatible = True  # Default for DataFrame
        
        overall_compatible = all([
            schema_compatible,
            type_compatible,
            index_compatible,
            temporal_aligned,
            format_compatible
        ])
        
        return CompatibilityMetrics(
            format_compatible=format_compatible,
            schema_compatible=schema_compatible,
            type_compatible=type_compatible,
            index_compatible=index_compatible,
            temporal_aligned=temporal_aligned,
            overall_compatible=overall_compatible,
            issues=issues,
            warnings=warnings,
            conversions_applied=conversions_applied,
            timestamp=datetime.now()
        )

    def _get_required_columns_for_step(self, step_name: str) -> List[str]:
        """Get required columns for a specific step."""
        column_requirements = {
            "step01": ["timestamp", "open", "high", "low", "close", "volume"],
            "step1_5": ["timestamp", "open", "high", "low", "close", "volume"],
            "step02": ["timestamp", "open", "high", "low", "close", "volume"],
            "step03": ["timestamp", "open", "high", "low", "close", "volume"],
            "step04": ["timestamp", "open", "high", "low", "close", "volume", "composite_cluster_id"],
            "step05": ["timestamp", "open", "high", "low", "close", "volume", "composite_cluster_id"],
            "step06": ["timestamp", "open", "high", "low", "close", "volume", "composite_cluster_id"],
            "step07": ["timestamp", "open", "high", "low", "close", "volume", "composite_cluster_id"]
        }
        return column_requirements.get(step_name, [])

    def _check_data_types(self, data: pd.DataFrame, step_name: str) -> List[str]:
        """Check data types for compatibility."""
        issues = []
        
        expected_types = {
            "timestamp": "datetime64[ns]",
            "open": "float64",
            "high": "float64",
            "low": "float64",
            "close": "float64",
            "volume": "float64"
        }
        
        for column, expected_type in expected_types.items():
            if column in data.columns:
                actual_type = str(data[column].dtype)
                if actual_type != expected_type:
                    issues.append(f"Column {column}: expected {expected_type}, got {actual_type}")
        
        return issues

    def _check_indexing(self, data: pd.DataFrame, step_name: str) -> List[str]:
        """Check indexing for compatibility."""
        issues = []
        
        if "timestamp" in data.columns:
            # Check if timestamp is sorted
            if not data["timestamp"].is_monotonic_increasing:
                issues.append("Timestamp column is not monotonically increasing")
            
            # Check for duplicate timestamps
            if data["timestamp"].duplicated().any():
                issues.append("Found duplicate timestamps")
        
        return issues

    def _check_temporal_alignment(self, data: pd.DataFrame) -> bool:
        """Check temporal alignment."""
        try:
            if "timestamp" in data.columns and len(data) > 1:
                # Check for reasonable time intervals
                time_diff = data["timestamp"].diff().dropna()
                if time_diff.std() > time_diff.mean() * 3:
                    return False
            return True
        except Exception:
            return False

    def _analyze_dict_compatibility(self, data: Dict[str, Any], step_name: str, expected_format: str) -> CompatibilityMetrics:
        """Analyze dictionary compatibility."""
        issues = []
        warnings = []
        conversions_applied = []
        
        # Check required keys
        required_keys = self._get_required_keys_for_step(step_name)
        missing_keys = [key for key in required_keys if key not in data]
        schema_compatible = len(missing_keys) == 0
        
        if missing_keys:
            issues.append(f"Missing required keys: {missing_keys}")
        
        # Other compatibility checks for dict
        type_compatible = True
        index_compatible = True
        temporal_aligned = True
        format_compatible = True
        
        overall_compatible = all([
            schema_compatible,
            type_compatible,
            index_compatible,
            temporal_aligned,
            format_compatible
        ])
        
        return CompatibilityMetrics(
            format_compatible=format_compatible,
            schema_compatible=schema_compatible,
            type_compatible=type_compatible,
            index_compatible=index_compatible,
            temporal_aligned=temporal_aligned,
            overall_compatible=overall_compatible,
            issues=issues,
            warnings=warnings,
            conversions_applied=conversions_applied,
            timestamp=datetime.now()
        )

    def _analyze_generic_compatibility(self, data: Any, step_name: str, expected_format: str) -> CompatibilityMetrics:
        """Analyze generic data compatibility."""
        issues = []
        warnings = []
        conversions_applied = []
        
        # Basic compatibility checks
        schema_compatible = data is not None
        type_compatible = True
        index_compatible = True
        temporal_aligned = True
        format_compatible = True
        
        if data is None:
            issues.append("Data is None")
        
        overall_compatible = all([
            schema_compatible,
            type_compatible,
            index_compatible,
            temporal_aligned,
            format_compatible
        ])
        
        return CompatibilityMetrics(
            format_compatible=format_compatible,
            schema_compatible=schema_compatible,
            type_compatible=type_compatible,
            index_compatible=index_compatible,
            temporal_aligned=temporal_aligned,
            overall_compatible=overall_compatible,
            issues=issues,
            warnings=warnings,
            conversions_applied=conversions_applied,
            timestamp=datetime.now()
        )

    async def monitor_format(self, data: Any, step_name: str, expected_format: str = "parquet") -> FormatMetrics:
        """Monitor data format compatibility."""
        self.logger.info(f"🔍 Monitoring data format for {step_name}")
        
        try:
            if isinstance(data, pd.DataFrame):
                metrics = self._analyze_dataframe_format(data, expected_format)
            else:
                metrics = self._analyze_generic_format(data, expected_format)
            
            # Store in history
            self.format_history.append(metrics)
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"❌ Error monitoring format for {step_name}: {e}")
            return FormatMetrics(
                expected_format=expected_format,
                actual_format="unknown",
                format_match=False,
                encoding_valid=False,
                compression_valid=False,
                file_size_reasonable=False,
                issues=[f"Format monitoring error: {str(e)}"],
                warnings=[],
                timestamp=datetime.now()
            )

    def _analyze_dataframe_format(self, data: pd.DataFrame, expected_format: str) -> FormatMetrics:
        """Analyze DataFrame format."""
        issues = []
        warnings = []
        
        actual_format = "dataframe"
        format_match = True  # DataFrame is always compatible
        encoding_valid = True
        compression_valid = True
        file_size_reasonable = True
        
        # Check DataFrame size
        memory_usage = data.memory_usage(deep=True).sum()
        if memory_usage > 1e9:  # 1GB
            warnings.append(f"Large DataFrame size: {memory_usage / 1e9:.2f}GB")
        
        return FormatMetrics(
            expected_format=expected_format,
            actual_format=actual_format,
            format_match=format_match,
            encoding_valid=encoding_valid,
            compression_valid=compression_valid,
            file_size_reasonable=file_size_reasonable,
            issues=issues,
            warnings=warnings,
            timestamp=datetime.now()
        )

    def _analyze_generic_format(self, data: Any, expected_format: str) -> FormatMetrics:
        """Analyze generic data format."""
        issues = []
        warnings = []
        
        actual_format = type(data).__name__
        format_match = True
        encoding_valid = True
        compression_valid = True
        file_size_reasonable = True
        
        if data is None:
            issues.append("Data is None")
            format_match = False
        
        return FormatMetrics(
            expected_format=expected_format,
            actual_format=actual_format,
            format_match=format_match,
            encoding_valid=encoding_valid,
            compression_valid=compression_valid,
            file_size_reasonable=file_size_reasonable,
            issues=issues,
            warnings=warnings,
            timestamp=datetime.now()
        )

    async def monitor_indexing(self, data: Any, step_name: str) -> IndexMetrics:
        """Monitor data indexing quality."""
        self.logger.info(f"🔍 Monitoring data indexing for {step_name}")
        
        try:
            if isinstance(data, pd.DataFrame):
                metrics = self._analyze_dataframe_indexing(data)
            else:
                metrics = self._analyze_generic_indexing(data)
            
            # Store in history
            self.index_history.append(metrics)
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"❌ Error monitoring indexing for {step_name}: {e}")
            return IndexMetrics(
                has_temporal_index=False,
                index_sorted=False,
                no_duplicates=False,
                no_gaps=False,
                frequency_consistent=False,
                timezone_consistent=False,
                overall_valid=False,
                issues=[f"Index monitoring error: {str(e)}"],
                warnings=[],
                timestamp=datetime.now()
            )

    def _analyze_dataframe_indexing(self, data: pd.DataFrame) -> IndexMetrics:
        """Analyze DataFrame indexing."""
        issues = []
        warnings = []
        
        # Check for temporal index
        has_temporal_index = "timestamp" in data.columns or data.index.name == "timestamp"
        
        # Check if sorted
        index_sorted = True
        if "timestamp" in data.columns:
            index_sorted = data["timestamp"].is_monotonic_increasing
        elif data.index.name == "timestamp":
            index_sorted = data.index.is_monotonic_increasing
        
        # Check for duplicates
        no_duplicates = True
        if "timestamp" in data.columns:
            no_duplicates = not data["timestamp"].duplicated().any()
        elif data.index.name == "timestamp":
            no_duplicates = not data.index.duplicated().any()
        
        # Check for gaps
        no_gaps = True
        if "timestamp" in data.columns and len(data) > 1:
            time_diff = data["timestamp"].diff().dropna()
            if time_diff.std() > time_diff.mean() * 2:
                no_gaps = False
                warnings.append("Large time gaps detected")
        
        # Check frequency consistency
        frequency_consistent = True
        if "timestamp" in data.columns and len(data) > 1:
            time_diff = data["timestamp"].diff().dropna()
            if time_diff.std() > time_diff.mean() * 0.5:
                frequency_consistent = False
                warnings.append("Inconsistent time frequency")
        
        # Check timezone consistency
        timezone_consistent = True
        if "timestamp" in data.columns:
            if data["timestamp"].dt.tz is not None:
                timezone_consistent = data["timestamp"].dt.tz == data["timestamp"].dt.tz
        
        overall_valid = all([
            has_temporal_index,
            index_sorted,
            no_duplicates,
            no_gaps,
            frequency_consistent,
            timezone_consistent
        ])
        
        return IndexMetrics(
            has_temporal_index=has_temporal_index,
            index_sorted=index_sorted,
            no_duplicates=no_duplicates,
            no_gaps=no_gaps,
            frequency_consistent=frequency_consistent,
            timezone_consistent=timezone_consistent,
            overall_valid=overall_valid,
            issues=issues,
            warnings=warnings,
            timestamp=datetime.now()
        )

    def _analyze_generic_indexing(self, data: Any) -> IndexMetrics:
        """Analyze generic data indexing."""
        issues = []
        warnings = []
        
        # Generic indexing checks
        has_temporal_index = False
        index_sorted = True
        no_duplicates = True
        no_gaps = True
        frequency_consistent = True
        timezone_consistent = True
        
        if data is None:
            issues.append("Data is None")
            overall_valid = False
        else:
            overall_valid = True
        
        return IndexMetrics(
            has_temporal_index=has_temporal_index,
            index_sorted=index_sorted,
            no_duplicates=no_duplicates,
            no_gaps=no_gaps,
            frequency_consistent=frequency_consistent,
            timezone_consistent=timezone_consistent,
            overall_valid=overall_valid,
            issues=issues,
            warnings=warnings,
            timestamp=datetime.now()
        )

    async def _log_quality_metrics(self, step_name: str, metrics: DataQualityMetrics) -> None:
        """Log quality metrics to MLflow."""
        try:
            # Convert metrics to dict for logging
            metrics_dict = asdict(metrics)
            metrics_dict["quality_level"] = metrics_dict["quality_level"].value
            metrics_dict["timestamp"] = metrics_dict["timestamp"].isoformat()
            
            # Log metrics
            log_step_metrics(
                config=self.config,
                step_name=f"{step_name}_quality_monitoring",
                metrics={
                    "overall_quality_score": metrics.overall_score,
                    "completeness": metrics.completeness,
                    "consistency": metrics.consistency,
                    "validity": metrics.validity,
                    "timeliness": metrics.timeliness,
                    "uniqueness": metrics.uniqueness,
                    "accuracy": metrics.accuracy,
                    "quality_level": metrics.quality_level.value
                },
                additional_metadata={
                    "step_name": step_name,
                    "issues_count": len(metrics.issues),
                    "warnings_count": len(metrics.warnings),
                    "recommendations_count": len(metrics.recommendations)
                }
            )
            
        except Exception as e:
            self.logger.error(f"❌ Failed to log quality metrics: {e}")

    async def _trigger_quality_alert(self, step_name: str, metrics: DataQualityMetrics) -> None:
        """Trigger quality alert for low quality data."""
        self.logger.warning(f"⚠️ QUALITY ALERT for {step_name}: Score {metrics.overall_score:.3f}")
        self.logger.warning(f"   Issues: {metrics.issues}")
        self.logger.warning(f"   Warnings: {metrics.warnings}")
        self.logger.warning(f"   Recommendations: {metrics.recommendations}")

    async def get_quality_summary(self) -> Dict[str, Any]:
        """Get comprehensive quality summary."""
        if not self.quality_history:
            return {"message": "No quality data available"}
        
        # Calculate summary statistics
        scores = [metrics.overall_score for metrics in self.quality_history]
        quality_levels = [metrics.quality_level.value for metrics in self.quality_history]
        
        summary = {
            "total_checks": len(self.quality_history),
            "average_quality_score": np.mean(scores),
            "min_quality_score": np.min(scores),
            "max_quality_score": np.max(scores),
            "quality_level_distribution": pd.Series(quality_levels).value_counts().to_dict(),
            "recent_quality_trend": scores[-10:] if len(scores) >= 10 else scores,
            "critical_issues_count": sum(1 for metrics in self.quality_history if metrics.quality_level == QualityLevel.CRITICAL),
            "poor_quality_count": sum(1 for metrics in self.quality_history if metrics.quality_level in [QualityLevel.CRITICAL, QualityLevel.POOR])
        }
        
        return summary

    async def generate_quality_report(self) -> Dict[str, Any]:
        """Generate comprehensive quality report."""
        quality_summary = await self.get_quality_summary()
        
        report = {
            "report_timestamp": datetime.now().isoformat(),
            "quality_summary": quality_summary,
            "compatibility_summary": {
                "total_checks": len(self.compatibility_history),
                "compatible_count": sum(1 for m in self.compatibility_history if m.overall_compatible),
                "incompatible_count": sum(1 for m in self.compatibility_history if not m.overall_compatible)
            },
            "format_summary": {
                "total_checks": len(self.format_history),
                "format_match_count": sum(1 for m in self.format_history if m.format_match),
                "format_mismatch_count": sum(1 for m in self.format_history if not m.format_match)
            },
            "index_summary": {
                "total_checks": len(self.index_history),
                "valid_index_count": sum(1 for m in self.index_history if m.overall_valid),
                "invalid_index_count": sum(1 for m in self.index_history if not m.overall_valid)
            },
            "recent_issues": [
                {
                    "step": f"Step {i+1}",
                    "quality_score": metrics.overall_score,
                    "issues": metrics.issues,
                    "warnings": metrics.warnings
                }
                for i, metrics in enumerate(self.quality_history[-5:])
            ]
        }
        
        return report

async def main():
    """Main execution function for testing."""
    # Example configuration
    config = {
        "SYMBOL": "ETHUSDT",
        "EXCHANGE": "BINANCE",
        "TIMEFRAME": "1m",
        "DATA_DIR": "data_cache",
        "data_quality_monitor": {
            "enable_real_time_monitoring": True,
            "alert_threshold": 0.8,
            "auto_fix_enabled": False
        }
    }
    
    # Initialize monitor
    monitor = DataQualityMonitor(config)
    
    # Create sample data for testing
    sample_data = pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=1000, freq="1min"),
        "open": np.random.uniform(100, 200, 1000),
        "high": np.random.uniform(100, 200, 1000),
        "low": np.random.uniform(100, 200, 1000),
        "close": np.random.uniform(100, 200, 1000),
        "volume": np.random.uniform(1000, 10000, 1000)
    })
    
    # Monitor data quality
    quality_metrics = await monitor.monitor_data_quality(sample_data, "step01")
    compatibility_metrics = await monitor.monitor_compatibility(sample_data, "step01")
    format_metrics = await monitor.monitor_format(sample_data, "step01")
    index_metrics = await monitor.monitor_indexing(sample_data, "step01")
    
    # Generate report
    report = await monitor.generate_quality_report()
    
    # Print results
    print("\n" + "="*80)
    print("DATA QUALITY MONITORING RESULTS")
    print("="*80)
    print(f"Quality Score: {quality_metrics.overall_score:.3f} ({quality_metrics.quality_level.value})")
    print(f"Compatibility: {'✅' if compatibility_metrics.overall_compatible else '❌'}")
    print(f"Format Match: {'✅' if format_metrics.format_match else '❌'}")
    print(f"Index Valid: {'✅' if index_metrics.overall_valid else '❌'}")
    
    print("\nQuality Summary:")
    print(json.dumps(report["quality_summary"], indent=2))

if __name__ == "__main__":
    asyncio.run( main())