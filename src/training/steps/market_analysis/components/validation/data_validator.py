"""
Data Validator for Market Analysis Components.

This module provides comprehensive data validation capabilities for
market analysis pipeline steps, including data quality checks,
schema validation, and statistical validation.
"""

import pandas as pd
import numpy as np
from typing import Any, Dict, List, Optional, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
import logging
from datetime import datetime

from src.utils.tprint import tprint, tprint_info, tprint_warning, tprint_error
from src.utils.common_utilities import (
    calculate_data_quality_metrics, safe_dataframe_operation,
    validate_dataframe_columns, create_summary_statistics
)
from src.utils.math_validation import validate_finite, safe_divide, safe_log
from src.training.steps.market_analysis.components.base_component import BaseMarketAnalysisComponent, ComponentConfig

class ValidationLevel(Enum):
    """Validation severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

@dataclass
class ValidationConfig:
    """Configuration for data validation."""
    # Data quality thresholds
    min_completeness: float = 0.95
    max_missing_ratio: float = 0.05
    max_outlier_ratio: float = 0.1
    min_correlation_threshold: float = 0.1
    
    # Schema validation
    required_columns: List[str] = field(default_factory=list)
    optional_columns: List[str] = field(default_factory=list)
    column_types: Dict[str, str] = field(default_factory=dict)
    
    # Statistical validation
    min_observations: int = 100
    max_skewness: float = 3.0
    max_kurtosis: float = 10.0
    
    # Temporal validation
    check_temporal_continuity: bool = True
    max_gap_ratio: float = 0.1
    min_temporal_coverage: float = 0.8
    
    # Economic validation
    check_price_consistency: bool = True
    min_price_positive: bool = True
    max_price_change_ratio: float = 0.5

@dataclass
class ValidationResult:
    """Result of data validation."""
    passed: bool
    score: float
    level: ValidationLevel
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    issues: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)

class DataValidator(BaseMarketAnalysisComponent):
    """
    Comprehensive data validator for market analysis components.
    
    Provides validation for:
    - Data quality and completeness
    - Schema compliance
    - Statistical properties
    - Temporal continuity
    - Economic consistency
    """
    
    def __init__(self, config: Optional[ValidationConfig] = None):
        """Initialize the data validator."""
        super().__init__(ComponentConfig())
        self.validation_config = config or ValidationConfig()
        self.logger = logging.getLogger(__name__)
        
    async def validate_data(self, 
                          data: pd.DataFrame, 
                          context: str = "general") -> ValidationResult:
        """
        Perform comprehensive data validation.
        
        Args:
            data: DataFrame to validate
            context: Validation context for logging
            
        Returns:
            ValidationResult with validation details
        """
        try:
            tprint_info(f"🔍 Starting data validation for {context}")
            
            # Initialize result
            result = ValidationResult(
                passed=True,
                score=1.0,
                level=ValidationLevel.INFO,
                message="Validation completed successfully"
            )
            
            # Perform validation checks
            await self._validate_schema(data, result)
            await self._validate_data_quality(data, result)
            await self._validate_statistical_properties(data, result)
            await self._validate_temporal_continuity(data, result)
            await self._validate_economic_consistency(data, result)
            
            # Calculate overall score
            result.score = self._calculate_overall_score(result)
            result.passed = result.score >= 0.8 and result.level != ValidationLevel.CRITICAL
            
            # Generate recommendations
            result.recommendations = self._generate_recommendations(result)
            
            tprint_info(f"✅ Data validation completed: score={result.score:.3f}, passed={result.passed}")
            return result
            
        except Exception as e:
            tprint_error(f"❌ Data validation failed: {str(e)}")
            return ValidationResult(
                passed=False,
                score=0.0,
                level=ValidationLevel.CRITICAL,
                message=f"Validation failed with error: {str(e)}",
                issues=[str(e)]
            )
    
    async def _validate_schema(self, data: pd.DataFrame, result: ValidationResult):
        """Validate data schema compliance."""
        try:
            # Check required columns
            missing_required = set(self.validation_config.required_columns) - set(data.columns)
            if missing_required:
                result.issues.append(f"Missing required columns: {missing_required}")
                result.level = ValidationLevel.ERROR
            
            # Check column types
            for col, expected_type in self.validation_config.column_types.items():
                if col in data.columns:
                    actual_type = str(data[col].dtype)
                    if expected_type not in actual_type:
                        result.warnings.append(f"Column {col} has type {actual_type}, expected {expected_type}")
            
            # Check for unexpected columns
            unexpected = set(data.columns) - set(self.validation_config.required_columns) - set(self.validation_config.optional_columns)
            if unexpected:
                result.warnings.append(f"Unexpected columns found: {unexpected}")
            
        except Exception as e:
            result.issues.append(f"Schema validation error: {str(e)}")
            result.level = ValidationLevel.ERROR
    
    async def _validate_data_quality(self, data: pd.DataFrame, result: ValidationResult):
        """Validate data quality metrics."""
        try:
            # Calculate quality metrics
            quality_metrics = calculate_data_quality_metrics(data)
            
            # Check completeness
            completeness = quality_metrics.get('completeness', 0.0)
            if completeness < self.validation_config.min_completeness:
                result.issues.append(f"Data completeness {completeness:.3f} below threshold {self.validation_config.min_completeness}")
                result.level = ValidationLevel.ERROR
            
            # Check missing data ratio
            missing_ratio = quality_metrics.get('missing_ratio', 0.0)
            if missing_ratio > self.validation_config.max_missing_ratio:
                result.issues.append(f"Missing data ratio {missing_ratio:.3f} exceeds threshold {self.validation_config.max_missing_ratio}")
                result.level = ValidationLevel.WARNING
            
            # Check outlier ratio
            outlier_ratio = quality_metrics.get('outlier_ratio', 0.0)
            if outlier_ratio > self.validation_config.max_outlier_ratio:
                result.warnings.append(f"High outlier ratio: {outlier_ratio:.3f}")
            
            result.details['quality_metrics'] = quality_metrics
            
        except Exception as e:
            result.issues.append(f"Data quality validation error: {str(e)}")
            result.level = ValidationLevel.ERROR
    
    async def _validate_statistical_properties(self, data: pd.DataFrame, result: ValidationResult):
        """Validate statistical properties of the data."""
        try:
            # Check minimum observations
            if len(data) < self.validation_config.min_observations:
                result.issues.append(f"Insufficient observations: {len(data)} < {self.validation_config.min_observations}")
                result.level = ValidationLevel.ERROR
            
            # Check numerical columns for statistical properties
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            
            for col in numeric_cols:
                if col in data.columns and not data[col].isna().all():
                    series = data[col].dropna()
                    
                    # Check skewness
                    skewness = abs(series.skew())
                    if skewness > self.validation_config.max_skewness:
                        result.warnings.append(f"Column {col} has high skewness: {skewness:.3f}")
                    
                    # Check kurtosis
                    kurtosis = abs(series.kurtosis())
                    if kurtosis > self.validation_config.max_kurtosis:
                        result.warnings.append(f"Column {col} has high kurtosis: {kurtosis:.3f}")
            
            result.details['statistical_properties'] = {
                'n_observations': len(data),
                'n_numeric_columns': len(numeric_cols),
                'skewness_checks': len(numeric_cols),
                'kurtosis_checks': len(numeric_cols)
            }
            
        except Exception as e:
            result.issues.append(f"Statistical validation error: {str(e)}")
            result.level = ValidationLevel.ERROR
    
    async def _validate_temporal_continuity(self, data: pd.DataFrame, result: ValidationResult):
        """Validate temporal continuity of the data."""
        try:
            if not self.validation_config.check_temporal_continuity:
                return
            
            # Check for timestamp column
            timestamp_cols = ['timestamp', 'time', 'datetime', 'date']
            timestamp_col = None
            
            for col in timestamp_cols:
                if col in data.columns:
                    timestamp_col = col
                    break
            
            if timestamp_col is None:
                result.warnings.append("No timestamp column found for temporal validation")
                return
            
            # Convert to datetime if needed
            timestamps = pd.to_datetime(data[timestamp_col], errors='coerce')
            valid_timestamps = timestamps.dropna()
            
            if len(valid_timestamps) == 0:
                result.issues.append("No valid timestamps found")
                result.level = ValidationLevel.ERROR
                return
            
            # Check for gaps
            timestamps_sorted = valid_timestamps.sort_values()
            time_diffs = timestamps_sorted.diff().dropna()
            
            if len(time_diffs) > 0:
                median_diff = time_diffs.median()
                large_gaps = time_diffs > median_diff * 3
                gap_ratio = large_gaps.sum() / len(time_diffs)
                
                if gap_ratio > self.validation_config.max_gap_ratio:
                    result.warnings.append(f"High gap ratio in temporal data: {gap_ratio:.3f}")
                
                # Check temporal coverage
                total_span = (timestamps_sorted.max() - timestamps_sorted.min()).total_seconds()
                expected_span = len(valid_timestamps) * median_diff.total_seconds()
                coverage = min(1.0, expected_span / total_span) if total_span > 0 else 0.0
                
                if coverage < self.validation_config.min_temporal_coverage:
                    result.warnings.append(f"Low temporal coverage: {coverage:.3f}")
            
            result.details['temporal_validation'] = {
                'timestamp_column': timestamp_col,
                'valid_timestamps': len(valid_timestamps),
                'total_observations': len(data),
                'gap_ratio': gap_ratio if len(time_diffs) > 0 else 0.0,
                'temporal_coverage': coverage if len(time_diffs) > 0 else 1.0
            }
            
        except Exception as e:
            result.issues.append(f"Temporal validation error: {str(e)}")
            result.level = ValidationLevel.ERROR
    
    async def _validate_economic_consistency(self, data: pd.DataFrame, result: ValidationResult):
        """Validate economic consistency of market data."""
        try:
            if not self.validation_config.check_price_consistency:
                return
            
            # Check for price columns
            price_cols = ['close', 'price', 'close_price', 'last_price']
            price_col = None
            
            for col in price_cols:
                if col in data.columns:
                    price_col = col
                    break
            
            if price_col is None:
                result.warnings.append("No price column found for economic validation")
                return
            
            prices = data[price_col].dropna()
            
            if len(prices) == 0:
                result.issues.append("No valid price data found")
                result.level = ValidationLevel.ERROR
                return
            
            # Check for positive prices
            if self.validation_config.min_price_positive:
                negative_prices = (prices <= 0).sum()
                if negative_prices > 0:
                    result.issues.append(f"Found {negative_prices} non-positive prices")
                    result.level = ValidationLevel.ERROR
            
            # Check for extreme price changes
            if len(prices) > 1:
                price_changes = prices.pct_change().dropna()
                extreme_changes = abs(price_changes) > self.validation_config.max_price_change_ratio
                extreme_ratio = extreme_changes.sum() / len(price_changes)
                
                if extreme_ratio > 0.05:  # More than 5% extreme changes
                    result.warnings.append(f"High ratio of extreme price changes: {extreme_ratio:.3f}")
            
            result.details['economic_validation'] = {
                'price_column': price_col,
                'valid_prices': len(prices),
                'negative_prices': negative_prices if self.validation_config.min_price_positive else 0,
                'extreme_changes_ratio': extreme_ratio if len(prices) > 1 else 0.0
            }
            
        except Exception as e:
            result.issues.append(f"Economic validation error: {str(e)}")
            result.level = ValidationLevel.ERROR
    
    def _calculate_overall_score(self, result: ValidationResult) -> float:
        """Calculate overall validation score."""
        base_score = 1.0
        
        # Deduct for issues
        issue_penalty = len(result.issues) * 0.1
        warning_penalty = len(result.warnings) * 0.05
        
        # Level-based penalties
        level_penalty = {
            ValidationLevel.INFO: 0.0,
            ValidationLevel.WARNING: 0.1,
            ValidationLevel.ERROR: 0.3,
            ValidationLevel.CRITICAL: 0.5
        }.get(result.level, 0.0)
        
        score = max(0.0, base_score - issue_penalty - warning_penalty - level_penalty)
        return min(1.0, score)
    
    def _generate_recommendations(self, result: ValidationResult) -> List[str]:
        """Generate recommendations based on validation results."""
        recommendations = []
        
        if result.issues:
            recommendations.append("Address critical issues before proceeding with analysis")
        
        if result.warnings:
            recommendations.append("Review warnings and consider data preprocessing")
        
        if result.score < 0.8:
            recommendations.append("Consider improving data quality before analysis")
        
        if len(result.issues) > 5:
            recommendations.append("High number of issues detected - consider data cleaning")
        
        return recommendations