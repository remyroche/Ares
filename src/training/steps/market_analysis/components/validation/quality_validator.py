"""
Quality Validator for Market Analysis Components.

This module provides comprehensive data quality validation capabilities
for market analysis pipeline steps, including data completeness,
consistency, accuracy, and reliability checks.
"""

import pandas as pd
import numpy as np
from typing import Any, Dict, List, Optional, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
import logging
from datetime import datetime, timedelta

from src.utils.tprint import tprint, tprint_info, tprint_warning, tprint_error
from src.utils.common_utilities import (
    calculate_data_quality_metrics, safe_dataframe_operation,
    validate_dataframe_columns, create_summary_statistics
)
from src.utils.math_validation import validate_finite, safe_divide, safe_log
from src.training.steps.market_analysis.components.base_component import BaseMarketAnalysisComponent, ComponentConfig

class QualityLevel(Enum):
    """Data quality levels."""
    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"
    CRITICAL = "critical"

@dataclass
class QualityValidationConfig:
    """Configuration for quality validation."""
    # Completeness thresholds
    min_completeness: float = 0.95
    max_missing_ratio: float = 0.05
    min_valid_ratio: float = 0.90
    
    # Consistency thresholds
    max_duplicate_ratio: float = 0.01
    max_inconsistency_ratio: float = 0.05
    min_temporal_consistency: float = 0.95
    
    # Accuracy thresholds
    max_outlier_ratio: float = 0.05
    max_anomaly_ratio: float = 0.02
    min_correlation_threshold: float = 0.1
    
    # Reliability thresholds
    min_data_freshness_days: int = 1
    max_data_age_days: int = 30
    min_update_frequency_hours: int = 1
    
    # Economic data specific
    check_price_consistency: bool = True
    check_volume_consistency: bool = True
    check_temporal_continuity: bool = True
    
    # Feature specific
    required_features: List[str] = field(default_factory=list)
    optional_features: List[str] = field(default_factory=list)
    feature_quality_thresholds: Dict[str, float] = field(default_factory=dict)

@dataclass
class QualityValidationResult:
    """Result of quality validation."""
    overall_quality: QualityLevel
    quality_score: float
    passed: bool
    details: Dict[str, Any] = field(default_factory=dict)
    issues: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    quality_metrics: Dict[str, float] = field(default_factory=dict)

class QualityValidator(BaseMarketAnalysisComponent):
    """
    Comprehensive quality validator for market analysis data.
    
    Provides validation for:
    - Data completeness and validity
    - Consistency and accuracy
    - Temporal and economic reliability
    - Feature-specific quality checks
    """
    
    def __init__(self, config: Optional[QualityValidationConfig] = None):
        """Initialize the quality validator."""
        super().__init__(ComponentConfig())
        self.quality_config = config or QualityValidationConfig()
        self.logger = logging.getLogger(__name__)
        
    async def validate_quality(self, 
                             data: pd.DataFrame,
                             context: str = "quality_validation") -> QualityValidationResult:
        """
        Perform comprehensive data quality validation.
        
        Args:
            data: DataFrame to validate
            context: Validation context for logging
            
        Returns:
            QualityValidationResult with quality assessment
        """
        try:
            tprint_info(f"🔍 Starting quality validation for {context}")
            
            # Initialize result
            result = QualityValidationResult(
                overall_quality=QualityLevel.EXCELLENT,
                quality_score=1.0,
                passed=True
            )
            
            # Perform quality checks
            await self._validate_completeness(data, result)
            await self._validate_consistency(data, result)
            await self._validate_accuracy(data, result)
            await self._validate_reliability(data, result)
            await self._validate_economic_consistency(data, result)
            await self._validate_feature_quality(data, result)
            
            # Calculate overall quality
            self._calculate_overall_quality(result)
            
            # Generate recommendations
            result.recommendations = self._generate_quality_recommendations(result)
            
            tprint_info(f"✅ Quality validation completed: {result.overall_quality.value} ({result.quality_score:.3f})")
            return result
            
        except Exception as e:
            tprint_error(f"❌ Quality validation failed: {str(e)}")
            return QualityValidationResult(
                overall_quality=QualityLevel.CRITICAL,
                quality_score=0.0,
                passed=False,
                issues=[str(e)]
            )
    
    async def _validate_completeness(self, data: pd.DataFrame, result: QualityValidationResult):
        """Validate data completeness."""
        try:
            # Calculate basic completeness metrics
            total_cells = data.size
            missing_cells = data.isnull().sum().sum()
            completeness = 1.0 - (missing_cells / total_cells) if total_cells > 0 else 0.0
            
            # Check per-column completeness
            column_completeness = {}
            for col in data.columns:
                col_missing = data[col].isnull().sum()
                col_total = len(data)
                col_completeness = 1.0 - (col_missing / col_total) if col_total > 0 else 0.0
                column_completeness[col] = col_completeness
                
                if col_completeness < self.quality_config.min_completeness:
                    result.issues.append(f"Column {col} completeness {col_completeness:.3f} below threshold")
            
            # Check missing data ratio
            missing_ratio = missing_cells / total_cells if total_cells > 0 else 0.0
            if missing_ratio > self.quality_config.max_missing_ratio:
                result.issues.append(f"Missing data ratio {missing_ratio:.3f} exceeds threshold")
            
            # Check valid data ratio
            valid_ratio = 1.0 - missing_ratio
            if valid_ratio < self.quality_config.min_valid_ratio:
                result.issues.append(f"Valid data ratio {valid_ratio:.3f} below threshold")
            
            result.quality_metrics['completeness'] = completeness
            result.quality_metrics['missing_ratio'] = missing_ratio
            result.quality_metrics['valid_ratio'] = valid_ratio
            result.details['column_completeness'] = column_completeness
            
        except Exception as e:
            result.issues.append(f"Completeness validation error: {str(e)}")
    
    async def _validate_consistency(self, data: pd.DataFrame, result: QualityValidationResult):
        """Validate data consistency."""
        try:
            # Check for duplicates
            duplicate_count = data.duplicated().sum()
            duplicate_ratio = duplicate_count / len(data) if len(data) > 0 else 0.0
            
            if duplicate_ratio > self.quality_config.max_duplicate_ratio:
                result.warnings.append(f"High duplicate ratio: {duplicate_ratio:.3f}")
            
            # Check for temporal consistency
            if self.quality_config.check_temporal_continuity:
                temporal_consistency = await self._check_temporal_consistency(data)
                if temporal_consistency < self.quality_config.min_temporal_consistency:
                    result.warnings.append(f"Low temporal consistency: {temporal_consistency:.3f}")
                
                result.quality_metrics['temporal_consistency'] = temporal_consistency
            
            # Check for data type consistency
            type_consistency = self._check_type_consistency(data)
            result.quality_metrics['type_consistency'] = type_consistency
            
            # Check for value consistency
            value_consistency = self._check_value_consistency(data)
            if value_consistency < (1.0 - self.quality_config.max_inconsistency_ratio):
                result.warnings.append(f"Low value consistency: {value_consistency:.3f}")
            
            result.quality_metrics['duplicate_ratio'] = duplicate_ratio
            result.quality_metrics['value_consistency'] = value_consistency
            result.details['consistency_checks'] = {
                'duplicate_count': duplicate_count,
                'type_consistency': type_consistency,
                'value_consistency': value_consistency
            }
            
        except Exception as e:
            result.issues.append(f"Consistency validation error: {str(e)}")
    
    async def _validate_accuracy(self, data: pd.DataFrame, result: QualityValidationResult):
        """Validate data accuracy."""
        try:
            # Check for outliers
            outlier_ratio = self._calculate_outlier_ratio(data)
            if outlier_ratio > self.quality_config.max_outlier_ratio:
                result.warnings.append(f"High outlier ratio: {outlier_ratio:.3f}")
            
            # Check for anomalies
            anomaly_ratio = self._calculate_anomaly_ratio(data)
            if anomaly_ratio > self.quality_config.max_anomaly_ratio:
                result.warnings.append(f"High anomaly ratio: {anomaly_ratio:.3f}")
            
            # Check correlation consistency
            correlation_consistency = self._check_correlation_consistency(data)
            if correlation_consistency < self.quality_config.min_correlation_threshold:
                result.warnings.append(f"Low correlation consistency: {correlation_consistency:.3f}")
            
            result.quality_metrics['outlier_ratio'] = outlier_ratio
            result.quality_metrics['anomaly_ratio'] = anomaly_ratio
            result.quality_metrics['correlation_consistency'] = correlation_consistency
            
        except Exception as e:
            result.issues.append(f"Accuracy validation error: {str(e)}")
    
    async def _validate_reliability(self, data: pd.DataFrame, result: QualityValidationResult):
        """Validate data reliability."""
        try:
            # Check data freshness
            freshness_score = self._check_data_freshness(data)
            if freshness_score < 0.8:
                result.warnings.append(f"Data freshness score: {freshness_score:.3f}")
            
            # Check update frequency
            update_frequency = self._check_update_frequency(data)
            if update_frequency < 0.5:
                result.warnings.append(f"Low update frequency: {update_frequency:.3f}")
            
            result.quality_metrics['freshness_score'] = freshness_score
            result.quality_metrics['update_frequency'] = update_frequency
            
        except Exception as e:
            result.issues.append(f"Reliability validation error: {str(e)}")
    
    async def _validate_economic_consistency(self, data: pd.DataFrame, result: QualityValidationResult):
        """Validate economic data consistency."""
        try:
            if self.quality_config.check_price_consistency:
                price_consistency = self._check_price_consistency(data)
                result.quality_metrics['price_consistency'] = price_consistency
                
                if price_consistency < 0.9:
                    result.warnings.append(f"Low price consistency: {price_consistency:.3f}")
            
            if self.quality_config.check_volume_consistency:
                volume_consistency = self._check_volume_consistency(data)
                result.quality_metrics['volume_consistency'] = volume_consistency
                
                if volume_consistency < 0.9:
                    result.warnings.append(f"Low volume consistency: {volume_consistency:.3f}")
            
        except Exception as e:
            result.issues.append(f"Economic consistency validation error: {str(e)}")
    
    async def _validate_feature_quality(self, data: pd.DataFrame, result: QualityValidationResult):
        """Validate feature-specific quality."""
        try:
            # Check required features
            missing_required = set(self.quality_config.required_features) - set(data.columns)
            if missing_required:
                result.issues.append(f"Missing required features: {missing_required}")
            
            # Check feature quality thresholds
            for feature, threshold in self.quality_config.feature_quality_thresholds.items():
                if feature in data.columns:
                    feature_quality = self._calculate_feature_quality(data[feature])
                    if feature_quality < threshold:
                        result.warnings.append(f"Feature {feature} quality {feature_quality:.3f} below threshold {threshold}")
                    
                    result.quality_metrics[f'{feature}_quality'] = feature_quality
            
        except Exception as e:
            result.issues.append(f"Feature quality validation error: {str(e)}")
    
    async def _check_temporal_consistency(self, data: pd.DataFrame) -> float:
        """Check temporal consistency of the data."""
        try:
            # Look for timestamp column
            timestamp_cols = ['timestamp', 'time', 'datetime', 'date']
            timestamp_col = None
            
            for col in timestamp_cols:
                if col in data.columns:
                    timestamp_col = col
                    break
            
            if timestamp_col is None:
                return 1.0  # No temporal data to check
            
            # Convert to datetime
            timestamps = pd.to_datetime(data[timestamp_col], errors='coerce')
            valid_timestamps = timestamps.dropna()
            
            if len(valid_timestamps) < 2:
                return 0.0
            
            # Check for temporal gaps
            timestamps_sorted = valid_timestamps.sort_values()
            time_diffs = timestamps_sorted.diff().dropna()
            
            if len(time_diffs) == 0:
                return 1.0
            
            # Calculate consistency based on gap distribution
            median_diff = time_diffs.median()
            large_gaps = time_diffs > median_diff * 3
            consistency = 1.0 - (large_gaps.sum() / len(time_diffs))
            
            return max(0.0, consistency)
            
        except Exception:
            return 0.0
    
    def _check_type_consistency(self, data: pd.DataFrame) -> float:
        """Check data type consistency."""
        try:
            # Check for mixed types in columns
            mixed_type_columns = 0
            for col in data.columns:
                if data[col].dtype == 'object':
                    # Check if column has mixed types
                    unique_types = data[col].apply(type).nunique()
                    if unique_types > 1:
                        mixed_type_columns += 1
            
            consistency = 1.0 - (mixed_type_columns / len(data.columns)) if len(data.columns) > 0 else 1.0
            return max(0.0, consistency)
            
        except Exception:
            return 0.0
    
    def _check_value_consistency(self, data: pd.DataFrame) -> float:
        """Check value consistency across columns."""
        try:
            # Check for impossible values
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            consistency_scores = []
            
            for col in numeric_cols:
                if col in data.columns:
                    series = data[col].dropna()
                    if len(series) > 0:
                        # Check for infinite values
                        inf_count = np.isinf(series).sum()
                        # Check for extreme values (beyond 6 standard deviations)
                        if len(series) > 1:
                            z_scores = np.abs((series - series.mean()) / series.std())
                            extreme_count = (z_scores > 6).sum()
                        else:
                            extreme_count = 0
                        
                        total_issues = inf_count + extreme_count
                        consistency = 1.0 - (total_issues / len(series))
                        consistency_scores.append(max(0.0, consistency))
            
            return np.mean(consistency_scores) if consistency_scores else 1.0
            
        except Exception:
            return 0.0
    
    def _calculate_outlier_ratio(self, data: pd.DataFrame) -> float:
        """Calculate outlier ratio in the data."""
        try:
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            outlier_counts = []
            total_values = 0
            
            for col in numeric_cols:
                if col in data.columns:
                    series = data[col].dropna()
                    if len(series) > 3:
                        Q1 = series.quantile(0.25)
                        Q3 = series.quantile(0.75)
                        IQR = Q3 - Q1
                        lower_bound = Q1 - 1.5 * IQR
                        upper_bound = Q3 + 1.5 * IQR
                        
                        outliers = ((series < lower_bound) | (series > upper_bound)).sum()
                        outlier_counts.append(outliers)
                        total_values += len(series)
            
            total_outliers = sum(outlier_counts)
            return total_outliers / total_values if total_values > 0 else 0.0
            
        except Exception:
            return 0.0
    
    def _calculate_anomaly_ratio(self, data: pd.DataFrame) -> float:
        """Calculate anomaly ratio in the data."""
        try:
            # Simple anomaly detection based on statistical properties
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            anomaly_counts = []
            total_values = 0
            
            for col in numeric_cols:
                if col in data.columns:
                    series = data[col].dropna()
                    if len(series) > 10:
                        # Use modified Z-score for anomaly detection
                        median = series.median()
                        mad = np.median(np.abs(series - median))
                        modified_z_scores = 0.6745 * (series - median) / mad
                        anomalies = (np.abs(modified_z_scores) > 3.5).sum()
                        
                        anomaly_counts.append(anomalies)
                        total_values += len(series)
            
            total_anomalies = sum(anomaly_counts)
            return total_anomalies / total_values if total_values > 0 else 0.0
            
        except Exception:
            return 0.0
    
    def _check_correlation_consistency(self, data: pd.DataFrame) -> float:
        """Check correlation consistency between related columns."""
        try:
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) < 2:
                return 1.0
            
            # Calculate correlation matrix
            corr_matrix = data[numeric_cols].corr()
            
            # Check for expected correlations (e.g., high-low, open-close)
            expected_correlations = [
                ('open', 'close'),
                ('high', 'low'),
                ('close', 'volume')
            ]
            
            correlation_scores = []
            for col1, col2 in expected_correlations:
                if col1 in corr_matrix.columns and col2 in corr_matrix.columns:
                    corr = abs(corr_matrix.loc[col1, col2])
                    if not np.isnan(corr):
                        correlation_scores.append(corr)
            
            return np.mean(correlation_scores) if correlation_scores else 1.0
            
        except Exception:
            return 0.0
    
    def _check_data_freshness(self, data: pd.DataFrame) -> float:
        """Check data freshness."""
        try:
            # Look for timestamp column
            timestamp_cols = ['timestamp', 'time', 'datetime', 'date']
            timestamp_col = None
            
            for col in timestamp_cols:
                if col in data.columns:
                    timestamp_col = col
                    break
            
            if timestamp_col is None:
                return 1.0  # No temporal data to check
            
            # Get latest timestamp
            timestamps = pd.to_datetime(data[timestamp_col], errors='coerce')
            latest_timestamp = timestamps.max()
            
            if pd.isna(latest_timestamp):
                return 0.0
            
            # Calculate age in days
            now = datetime.now()
            age_days = (now - latest_timestamp).days
            
            # Calculate freshness score
            if age_days <= self.quality_config.min_data_freshness_days:
                return 1.0
            elif age_days <= self.quality_config.max_data_age_days:
                return 1.0 - (age_days - self.quality_config.min_data_freshness_days) / (self.quality_config.max_data_age_days - self.quality_config.min_data_freshness_days)
            else:
                return 0.0
                
        except Exception:
            return 0.0
    
    def _check_update_frequency(self, data: pd.DataFrame) -> float:
        """Check data update frequency."""
        try:
            # Look for timestamp column
            timestamp_cols = ['timestamp', 'time', 'datetime', 'date']
            timestamp_col = None
            
            for col in timestamp_cols:
                if col in data.columns:
                    timestamp_col = col
                    break
            
            if timestamp_col is None:
                return 1.0  # No temporal data to check
            
            # Get timestamps
            timestamps = pd.to_datetime(data[timestamp_col], errors='coerce')
            valid_timestamps = timestamps.dropna()
            
            if len(valid_timestamps) < 2:
                return 0.0
            
            # Calculate update frequency
            timestamps_sorted = valid_timestamps.sort_values()
            time_diffs = timestamps_sorted.diff().dropna()
            
            if len(time_diffs) == 0:
                return 1.0
            
            median_diff_hours = time_diffs.median().total_seconds() / 3600
            expected_frequency_hours = self.quality_config.min_update_frequency_hours
            
            if median_diff_hours <= expected_frequency_hours:
                return 1.0
            else:
                return max(0.0, 1.0 - (median_diff_hours - expected_frequency_hours) / expected_frequency_hours)
                
        except Exception:
            return 0.0
    
    def _check_price_consistency(self, data: pd.DataFrame) -> float:
        """Check price data consistency."""
        try:
            price_cols = ['open', 'high', 'low', 'close']
            available_price_cols = [col for col in price_cols if col in data.columns]
            
            if len(available_price_cols) < 2:
                return 1.0
            
            consistency_scores = []
            
            # Check high >= low
            if 'high' in data.columns and 'low' in data.columns:
                high_low_consistent = (data['high'] >= data['low']).mean()
                consistency_scores.append(high_low_consistent)
            
            # Check high >= open, close
            for col in ['open', 'close']:
                if col in data.columns and 'high' in data.columns:
                    high_col_consistent = (data['high'] >= data[col]).mean()
                    consistency_scores.append(high_col_consistent)
            
            # Check low <= open, close
            for col in ['open', 'close']:
                if col in data.columns and 'low' in data.columns:
                    low_col_consistent = (data['low'] <= data[col]).mean()
                    consistency_scores.append(low_col_consistent)
            
            return np.mean(consistency_scores) if consistency_scores else 1.0
            
        except Exception:
            return 0.0
    
    def _check_volume_consistency(self, data: pd.DataFrame) -> float:
        """Check volume data consistency."""
        try:
            if 'volume' not in data.columns:
                return 1.0
            
            volume = data['volume'].dropna()
            if len(volume) == 0:
                return 0.0
            
            # Check for non-negative volumes
            non_negative_ratio = (volume >= 0).mean()
            
            # Check for reasonable volume values (not all zeros, not all same)
            non_zero_ratio = (volume > 0).mean()
            variance_ratio = volume.var() / (volume.mean() ** 2) if volume.mean() > 0 else 0
            
            consistency = (non_negative_ratio + non_zero_ratio + min(1.0, variance_ratio)) / 3
            return max(0.0, consistency)
            
        except Exception:
            return 0.0
    
    def _calculate_feature_quality(self, series: pd.Series) -> float:
        """Calculate quality score for a specific feature."""
        try:
            if len(series) == 0:
                return 0.0
            
            # Completeness
            completeness = 1.0 - series.isnull().sum() / len(series)
            
            # Consistency (for numeric data)
            if series.dtype in ['int64', 'float64']:
                # Check for infinite values
                inf_ratio = np.isinf(series).sum() / len(series)
                # Check for extreme outliers
                if len(series.dropna()) > 3:
                    Q1 = series.quantile(0.25)
                    Q3 = series.quantile(0.75)
                    IQR = Q3 - Q1
                    outlier_ratio = ((series < Q1 - 3 * IQR) | (series > Q3 + 3 * IQR)).sum() / len(series)
                else:
                    outlier_ratio = 0.0
                
                consistency = 1.0 - inf_ratio - outlier_ratio
            else:
                consistency = 1.0
            
            # Overall quality
            quality = (completeness + consistency) / 2
            return max(0.0, quality)
            
        except Exception:
            return 0.0
    
    def _calculate_overall_quality(self, result: QualityValidationResult):
        """Calculate overall quality score and level."""
        try:
            # Calculate weighted quality score
            weights = {
                'completeness': 0.25,
                'temporal_consistency': 0.20,
                'outlier_ratio': 0.15,
                'anomaly_ratio': 0.15,
                'price_consistency': 0.10,
                'volume_consistency': 0.10,
                'freshness_score': 0.05
            }
            
            weighted_score = 0.0
            total_weight = 0.0
            
            for metric, weight in weights.items():
                if metric in result.quality_metrics:
                    if metric in ['outlier_ratio', 'anomaly_ratio']:
                        # Invert these metrics (lower is better)
                        score = 1.0 - result.quality_metrics[metric]
                    else:
                        score = result.quality_metrics[metric]
                    
                    weighted_score += score * weight
                    total_weight += weight
            
            if total_weight > 0:
                result.quality_score = weighted_score / total_weight
            else:
                result.quality_score = 0.0
            
            # Determine quality level
            if result.quality_score >= 0.95:
                result.overall_quality = QualityLevel.EXCELLENT
            elif result.quality_score >= 0.85:
                result.overall_quality = QualityLevel.GOOD
            elif result.quality_score >= 0.70:
                result.overall_quality = QualityLevel.FAIR
            elif result.quality_score >= 0.50:
                result.overall_quality = QualityLevel.POOR
            else:
                result.overall_quality = QualityLevel.CRITICAL
            
            # Determine if passed
            result.passed = result.overall_quality in [QualityLevel.EXCELLENT, QualityLevel.GOOD, QualityLevel.FAIR]
            
        except Exception as e:
            result.issues.append(f"Overall quality calculation error: {str(e)}")
            result.quality_score = 0.0
            result.overall_quality = QualityLevel.CRITICAL
            result.passed = False
    
    def _generate_quality_recommendations(self, result: QualityValidationResult) -> List[str]:
        """Generate quality improvement recommendations."""
        recommendations = []
        
        if result.overall_quality == QualityLevel.CRITICAL:
            recommendations.append("Data quality is critical - immediate attention required")
        
        if result.quality_metrics.get('completeness', 1.0) < 0.9:
            recommendations.append("Improve data completeness by addressing missing values")
        
        if result.quality_metrics.get('outlier_ratio', 0.0) > 0.1:
            recommendations.append("Review and handle outliers in the data")
        
        if result.quality_metrics.get('temporal_consistency', 1.0) < 0.9:
            recommendations.append("Improve temporal consistency by filling gaps in time series")
        
        if result.quality_metrics.get('freshness_score', 1.0) < 0.8:
            recommendations.append("Update data to improve freshness")
        
        if len(result.issues) > 5:
            recommendations.append("Multiple quality issues detected - consider comprehensive data cleaning")
        
        return recommendations