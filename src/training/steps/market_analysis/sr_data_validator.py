"""
Comprehensive data validation pipeline for SR detection.

This module provides robust data validation for OHLCV data used in SR detection,
including quality checks, outlier detection, and data integrity validation.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum

from src.utils.logger import system_logger

class ValidationLevel(Enum):
    """Validation levels for different use cases."""
    BASIC = "basic"
    STANDARD = "standard"
    STRICT = "strict"
    DEBUG = "debug"

@dataclass
class ValidationResult:
    """Result of data validation."""
    is_valid: bool
    issues: List[str]
    warnings: List[str]
    quality_score: float
    recommendations: List[str]
    data_quality_metrics: Dict[str, Any]

class SRDataValidator:
    """Comprehensive data validator for SR detection."""
    
    def __init__(self, validation_level: ValidationLevel = ValidationLevel.STANDARD):
        self.validation_level = validation_level
        self.logger = system_logger.getChild('SRDataValidator')
        
        # Define validation thresholds based on level
        self.thresholds = self._get_thresholds()
        
        # Required columns for OHLCV data
        self.required_columns = ['open', 'high', 'low', 'close', 'volume']
        
        # Optional columns that enhance validation
        self.optional_columns = ['timestamp', 'datetime', 'time']
    
    def _get_thresholds(self) -> Dict[str, Dict[str, float]]:
        """Get validation thresholds based on validation level."""
        base_thresholds = {
            'basic': {
                'min_rows': 10,
                'max_missing_pct': 0.1,
                'max_outlier_pct': 0.2,
                'min_volume': 0.0,
                'max_price_change_pct': 0.5,
                'min_correlation': 0.3
            },
            'standard': {
                'min_rows': 50,
                'max_missing_pct': 0.05,
                'max_outlier_pct': 0.1,
                'min_volume': 1.0,
                'max_price_change_pct': 0.2,
                'min_correlation': 0.5
            },
            'strict': {
                'min_rows': 100,
                'max_missing_pct': 0.01,
                'max_outlier_pct': 0.05,
                'min_volume': 10.0,
                'max_price_change_pct': 0.1,
                'min_correlation': 0.7
            },
            'debug': {
                'min_rows': 1,
                'max_missing_pct': 0.0,
                'max_outlier_pct': 0.0,
                'min_volume': 0.0,
                'max_price_change_pct': 0.0,
                'min_correlation': 0.0
            }
        }
        return base_thresholds[self.validation_level.value]
    
    def validate_ohlcv_data(self, data: pd.DataFrame) -> ValidationResult:
        """
        Comprehensive validation of OHLCV data for SR detection.
        
        Args:
            data: DataFrame containing OHLCV data
            
        Returns:
            ValidationResult with validation details
        """
        issues = []
        warnings = []
        recommendations = []
        quality_metrics = {}
        
        try:
            # Basic structure validation
            structure_result = self._validate_data_structure(data)
            issues.extend(structure_result['issues'])
            warnings.extend(structure_result['warnings'])
            quality_metrics.update(structure_result['metrics'])
            
            if structure_result['is_valid']:
                # Data type validation
                type_result = self._validate_data_types(data)
                issues.extend(type_result['issues'])
                warnings.extend(type_result['warnings'])
                quality_metrics.update(type_result['metrics'])
                
                # Data quality validation
                quality_result = self._validate_data_quality(data)
                issues.extend(quality_result['issues'])
                warnings.extend(quality_result['warnings'])
                quality_metrics.update(quality_result['metrics'])
                
                # Price relationship validation
                price_result = self._validate_price_relationships(data)
                issues.extend(price_result['issues'])
                warnings.extend(price_result['warnings'])
                quality_metrics.update(price_result['metrics'])
                
                # Outlier detection
                outlier_result = self._detect_outliers(data)
                issues.extend(outlier_result['issues'])
                warnings.extend(outlier_result['warnings'])
                quality_metrics.update(outlier_result['metrics'])
                
                # Time series validation
                ts_result = self._validate_time_series(data)
                issues.extend(ts_result['issues'])
                warnings.extend(ts_result['warnings'])
                quality_metrics.update(ts_result['metrics'])
                
                # Generate recommendations
                recommendations = self._generate_recommendations(issues, warnings, quality_metrics)
            
            # Calculate overall quality score
            quality_score = self._calculate_quality_score(issues, warnings, quality_metrics)
            
            is_valid = len(issues) == 0 or self.validation_level == ValidationLevel.DEBUG
            
            return ValidationResult(
                is_valid=is_valid,
                issues=issues,
                warnings=warnings,
                quality_score=quality_score,
                recommendations=recommendations,
                data_quality_metrics=quality_metrics
            )
            
        except Exception as e:
            self.logger.error(f"Data validation failed: {e}")
            return ValidationResult(
                is_valid=False,
                issues=[f"Validation error: {str(e)}"],
                warnings=[],
                quality_score=0.0,
                recommendations=["Check data format and try again"],
                data_quality_metrics={}
            )
    
    def _validate_data_structure(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Validate basic data structure."""
        issues = []
        warnings = []
        metrics = {}
        
        # Check if data is empty
        if data is None or len(data) == 0:
            issues.append("Data is empty or None")
            return {'is_valid': False, 'issues': issues, 'warnings': warnings, 'metrics': metrics}
        
        # Check minimum rows
        min_rows = self.thresholds['min_rows']
        if len(data) < min_rows:
            issues.append(f"Insufficient data: {len(data)} rows, minimum {min_rows} required")
        
        # Check required columns
        missing_columns = [col for col in self.required_columns if col not in data.columns]
        if missing_columns:
            issues.append(f"Missing required columns: {missing_columns}")
        
        # Check for duplicate columns
        duplicate_columns = data.columns[data.columns.duplicated()].tolist()
        if duplicate_columns:
            issues.append(f"Duplicate columns found: {duplicate_columns}")
        
        # Check for duplicate index
        if data.index.duplicated().any():
            issues.append("Duplicate index values found")
        
        metrics.update({
            'total_rows': len(data),
            'total_columns': len(data.columns),
            'missing_required_columns': len(missing_columns),
            'duplicate_columns': len(duplicate_columns),
            'duplicate_index_count': data.index.duplicated().sum()
        })
        
        return {
            'is_valid': len(issues) == 0,
            'issues': issues,
            'warnings': warnings,
            'metrics': metrics
        }
    
    def _validate_data_types(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Validate data types for OHLCV columns."""
        issues = []
        warnings = []
        metrics = {}
        
        numeric_columns = ['open', 'high', 'low', 'close', 'volume']
        
        for col in numeric_columns:
            if col in data.columns:
                # Check if column is numeric
                if not pd.api.types.is_numeric_dtype(data[col]):
                    issues.append(f"Column '{col}' is not numeric")
                    continue
                
                # Check for non-finite values
                non_finite = ~np.isfinite(data[col])
                non_finite_count = non_finite.sum()
                if non_finite_count > 0:
                    issues.append(f"Column '{col}' has {non_finite_count} non-finite values")
                
                # Check for negative values (except volume which can be 0)
                if col == 'volume':
                    negative_count = (data[col] < 0).sum()
                    if negative_count > 0:
                        issues.append(f"Column '{col}' has {negative_count} negative values")
                else:
                    non_positive_count = (data[col] <= 0).sum()
                    if non_positive_count > 0:
                        issues.append(f"Column '{col}' has {non_positive_count} non-positive values")
                
                metrics[f'{col}_non_finite'] = non_finite_count
                metrics[f'{col}_negative'] = (data[col] < 0).sum() if col == 'volume' else (data[col] <= 0).sum()
        
        return {
            'is_valid': len(issues) == 0,
            'issues': issues,
            'warnings': warnings,
            'metrics': metrics
        }
    
    def _validate_data_quality(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Validate data quality metrics."""
        issues = []
        warnings = []
        metrics = {}
        
        # Check missing values
        missing_pct = data.isnull().sum().sum() / (len(data) * len(data.columns))
        max_missing_pct = self.thresholds['max_missing_pct']
        
        if missing_pct > max_missing_pct:
            issues.append(f"Too many missing values: {missing_pct:.2%} (max: {max_missing_pct:.2%})")
        
        # Check for constant columns
        constant_columns = []
        for col in data.columns:
            if data[col].nunique() <= 1:
                constant_columns.append(col)
        
        if constant_columns:
            warnings.append(f"Constant columns found: {constant_columns}")
        
        # Check volume quality
        if 'volume' in data.columns:
            volume_stats = data['volume'].describe()
            zero_volume_pct = (data['volume'] == 0).mean()
            if zero_volume_pct > 0.1:
                warnings.append(f"High percentage of zero volume: {zero_volume_pct:.2%}")
            
            metrics.update({
                'zero_volume_pct': zero_volume_pct,
                'volume_mean': volume_stats['mean'],
                'volume_std': volume_stats['std']
            })
        
        metrics.update({
            'missing_pct': missing_pct,
            'constant_columns': len(constant_columns),
            'total_missing': data.isnull().sum().sum()
        })
        
        return {
            'is_valid': len(issues) == 0,
            'issues': issues,
            'warnings': warnings,
            'metrics': metrics
        }
    
    def _validate_price_relationships(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Validate OHLC price relationships."""
        issues = []
        warnings = []
        metrics = {}
        
        if not all(col in data.columns for col in ['open', 'high', 'low', 'close']):
            return {'is_valid': True, 'issues': issues, 'warnings': warnings, 'metrics': metrics}
        
        # Check high >= low
        high_low_violations = (data['high'] < data['low']).sum()
        if high_low_violations > 0:
            issues.append(f"High < Low violations: {high_low_violations}")
        
        # Check high >= open
        high_open_violations = (data['high'] < data['open']).sum()
        if high_open_violations > 0:
            issues.append(f"High < Open violations: {high_open_violations}")
        
        # Check high >= close
        high_close_violations = (data['high'] < data['close']).sum()
        if high_close_violations > 0:
            issues.append(f"High < Close violations: {high_close_violations}")
        
        # Check low <= open
        low_open_violations = (data['low'] > data['open']).sum()
        if low_open_violations > 0:
            issues.append(f"Low > Open violations: {low_open_violations}")
        
        # Check low <= close
        low_close_violations = (data['low'] > data['close']).sum()
        if low_close_violations > 0:
            issues.append(f"Low > Close violations: {low_close_violations}")
        
        # Calculate price change statistics
        price_changes = data['close'].pct_change().dropna()
        extreme_changes = (abs(price_changes) > self.thresholds['max_price_change_pct']).sum()
        
        if extreme_changes > 0:
            warnings.append(f"Extreme price changes detected: {extreme_changes}")
        
        metrics.update({
            'high_low_violations': high_low_violations,
            'high_open_violations': high_open_violations,
            'high_close_violations': high_close_violations,
            'low_open_violations': low_open_violations,
            'low_close_violations': low_close_violations,
            'extreme_changes': extreme_changes,
            'max_price_change': price_changes.abs().max() if len(price_changes) > 0 else 0,
            'avg_price_change': price_changes.abs().mean() if len(price_changes) > 0 else 0
        })
        
        return {
            'is_valid': len(issues) == 0,
            'issues': issues,
            'warnings': warnings,
            'metrics': metrics
        }
    
    def _detect_outliers(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Detect outliers in OHLCV data."""
        issues = []
        warnings = []
        metrics = {}
        
        outlier_columns = ['open', 'high', 'low', 'close', 'volume']
        total_outliers = 0
        
        for col in outlier_columns:
            if col not in data.columns:
                continue
            
            # Use IQR method for outlier detection
            Q1 = data[col].quantile(0.25)
            Q3 = data[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = ((data[col] < lower_bound) | (data[col] > upper_bound)).sum()
            outlier_pct = outliers / len(data)
            
            if outlier_pct > self.thresholds['max_outlier_pct']:
                warnings.append(f"High outlier percentage in '{col}': {outlier_pct:.2%}")
            
            total_outliers += outliers
            metrics[f'{col}_outliers'] = outliers
            metrics[f'{col}_outlier_pct'] = outlier_pct
        
        metrics['total_outliers'] = total_outliers
        metrics['total_outlier_pct'] = total_outliers / (len(data) * len(outlier_columns))
        
        return {
            'is_valid': True,
            'issues': issues,
            'warnings': warnings,
            'metrics': metrics
        }
    
    def _validate_time_series(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Validate time series properties."""
        issues = []
        warnings = []
        metrics = {}
        
        # Check if index is datetime
        if not isinstance(data.index, pd.DatetimeIndex):
            warnings.append("Index is not datetime - time series analysis may be limited")
        
        # Check for gaps in time series
        if isinstance(data.index, pd.DatetimeIndex) and len(data) > 1:
            time_diffs = data.index.to_series().diff().dropna()
            if len(time_diffs) > 0:
                median_diff = time_diffs.median()
                large_gaps = (time_diffs > 3 * median_diff).sum()
                
                if large_gaps > 0:
                    warnings.append(f"Large time gaps detected: {large_gaps}")
                
                metrics['median_time_diff'] = median_diff.total_seconds()
                metrics['large_gaps'] = large_gaps
        
        return {
            'is_valid': True,
            'issues': issues,
            'warnings': warnings,
            'metrics': metrics
        }
    
    def _generate_recommendations(self, issues: List[str], warnings: List[str], metrics: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on validation results."""
        recommendations = []
        
        if issues:
            recommendations.append("Fix critical issues before proceeding with SR detection")
        
        if warnings:
            recommendations.append("Review warnings and consider data preprocessing")
        
        if metrics.get('missing_pct', 0) > 0.01:
            recommendations.append("Consider imputing missing values")
        
        if metrics.get('total_outlier_pct', 0) > 0.05:
            recommendations.append("Consider outlier treatment or robust detection methods")
        
        if metrics.get('zero_volume_pct', 0) > 0.1:
            recommendations.append("Consider filtering or imputing zero volume periods")
        
        if metrics.get('extreme_changes', 0) > 0:
            recommendations.append("Review extreme price changes for data quality issues")
        
        return recommendations
    
    def _calculate_quality_score(self, issues: List[str], warnings: List[str], metrics: Dict[str, Any]) -> float:
        """Calculate overall data quality score (0-1)."""
        base_score = 1.0
        
        # Deduct for issues
        issue_penalty = len(issues) * 0.2
        base_score -= issue_penalty
        
        # Deduct for warnings
        warning_penalty = len(warnings) * 0.05
        base_score -= warning_penalty
        
        # Deduct for data quality metrics
        if metrics.get('missing_pct', 0) > 0:
            base_score -= min(metrics['missing_pct'] * 2, 0.3)
        
        if metrics.get('total_outlier_pct', 0) > 0:
            base_score -= min(metrics['total_outlier_pct'] * 2, 0.2)
        
        if metrics.get('zero_volume_pct', 0) > 0.1:
            base_score -= min((metrics['zero_volume_pct'] - 0.1) * 2, 0.1)
        
        return max(0.0, min(1.0, base_score))
    
    def get_validation_summary(self, result: ValidationResult) -> str:
        """Get a human-readable validation summary."""
        summary = f"Data Validation Summary (Level: {self.validation_level.value})\n"
        summary += f"Valid: {'✅' if result.is_valid else '❌'}\n"
        summary += f"Quality Score: {result.quality_score:.2f}\n"
        summary += f"Issues: {len(result.issues)}\n"
        summary += f"Warnings: {len(result.warnings)}\n"
        
        if result.issues:
            summary += "\nIssues:\n"
            for issue in result.issues:
                summary += f"  - {issue}\n"
        
        if result.warnings:
            summary += "\nWarnings:\n"
            for warning in result.warnings:
                summary += f"  - {warning}\n"
        
        if result.recommendations:
            summary += "\nRecommendations:\n"
            for rec in result.recommendations:
                summary += f"  - {rec}\n"
        
        return summary