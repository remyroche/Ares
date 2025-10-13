"""
Data validation utilities for feature selection.

This module provides comprehensive data validation capabilities
for feature selection operations including data quality checks,
schema validation, and temporal alignment verification.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass
import warnings

from src.utils.tprint import tprint_debug, tprint_info, tprint_warning, tprint_success


@dataclass
class DataValidationResult:
    """Result of data validation operation."""
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    quality_metrics: Dict[str, Any]
    validated_data: Optional[pd.DataFrame] = None


@dataclass
class DataQualityMetrics:
    """Data quality metrics for validation."""
    n_samples: int
    n_features: int
    missing_ratio: float
    duplicate_ratio: float
    variance_ratio: float
    correlation_ratio: float
    temporal_consistency: float
    data_quality_score: float


class DataValidator:
    """Data validator for feature selection operations."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.logger = get_logger("DataValidator")
        
        # Validation thresholds
        self.thresholds = {
            'max_missing_ratio': self.config.get('max_missing_ratio', 0.1),
            'max_duplicate_ratio': self.config.get('max_duplicate_ratio', 0.05),
            'min_variance_ratio': self.config.get('min_variance_ratio', 0.01),
            'max_correlation_ratio': self.config.get('max_correlation_ratio', 0.95),
            'min_temporal_consistency': self.config.get('min_temporal_consistency', 0.8)
        }
    
    def validate_data(self, data: pd.DataFrame, target: Optional[pd.Series] = None) -> DataValidationResult:
        """
        Validate data for feature selection.
        
        Args:
            data: Feature matrix to validate
            target: Optional target variable
            
        Returns:
            DataValidationResult with validation results
        """
        tprint_info(f"🔍 Validating data: {data.shape}")
        
        errors = []
        warnings = []
        quality_metrics = {}
        
        try:
            # Basic data structure validation
            structure_errors = self._validate_data_structure(data)
            errors.extend(structure_errors)
            
            if structure_errors:
                return DataValidationResult(
                    is_valid=False,
                    errors=errors,
                    warnings=warnings,
                    quality_metrics=quality_metrics
                )
            
            # Calculate quality metrics
            quality_metrics = self._calculate_quality_metrics(data, target)
            
            # Validate data quality
            quality_errors, quality_warnings = self._validate_data_quality(data, quality_metrics)
            errors.extend(quality_errors)
            warnings.extend(quality_warnings)
            
            # Validate target if provided
            if target is not None:
                target_errors, target_warnings = self._validate_target(target, data)
                errors.extend(target_errors)
                warnings.extend(target_warnings)
            
            # Validate temporal consistency if applicable
            temporal_errors, temporal_warnings = self._validate_temporal_consistency(data)
            errors.extend(temporal_errors)
            warnings.extend(temporal_warnings)
            
            # Clean data if validation passes
            cleaned_data = None
            if not errors:
                cleaned_data = self._clean_data(data)
            
            is_valid = len(errors) == 0
            
            if is_valid:
                tprint_success(f"   ✅ Data validation passed")
                if warnings:
                    tprint_warning(f"   ⚠️ {len(warnings)} warnings generated")
            else:
                tprint_warning(f"   ⚠️ Data validation failed: {len(errors)} errors")
            
            return DataValidationResult(
                is_valid=is_valid,
                errors=errors,
                warnings=warnings,
                quality_metrics=quality_metrics,
                validated_data=cleaned_data
            )
            
        except Exception as e:
            error_msg = f"Data validation failed with exception: {e}"
            tprint_warning(f"   ⚠️ {error_msg}")
            return DataValidationResult(
                is_valid=False,
                errors=[error_msg],
                warnings=warnings,
                quality_metrics=quality_metrics
            )
    
    def _validate_data_structure(self, data: pd.DataFrame) -> List[str]:
        """Validate basic data structure."""
        errors = []
        
        # Check if data is empty
        if data.empty:
            errors.append("Data is empty")
            return errors
        
        # Check if data has features
        if data.shape[1] == 0:
            errors.append("Data has no features")
        
        # Check if data has samples
        if data.shape[0] == 0:
            errors.append("Data has no samples")
        
        # Check for numeric data
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) == 0:
            errors.append("Data contains no numeric features")
        elif len(numeric_cols) < data.shape[1] * 0.8:  # Less than 80% numeric
            errors.append("Data contains too many non-numeric features")
        
        # Check for infinite values
        if np.isinf(data.select_dtypes(include=[np.number])).any().any():
            errors.append("Data contains infinite values")
        
        return errors
    
    def _calculate_quality_metrics(self, data: pd.DataFrame, target: Optional[pd.Series] = None) -> Dict[str, Any]:
        """Calculate data quality metrics."""
        tprint_debug("📊 Calculating data quality metrics")
        
        # Basic metrics
        n_samples, n_features = data.shape
        missing_ratio = data.isnull().sum().sum() / (n_samples * n_features)
        duplicate_ratio = data.duplicated().sum() / n_samples
        
        # Variance metrics
        numeric_data = data.select_dtypes(include=[np.number])
        if not numeric_data.empty:
            variances = numeric_data.var()
            variance_ratio = variances.mean() / (variances.std() + 1e-10)
        else:
            variance_ratio = 0.0
        
        # Correlation metrics
        if not numeric_data.empty and numeric_data.shape[1] > 1:
            corr_matrix = numeric_data.corr().abs()
            # Remove diagonal and get upper triangle
            upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
            high_corr_ratio = (upper_triangle > 0.95).sum().sum() / (n_features * (n_features - 1) / 2)
        else:
            high_corr_ratio = 0.0
        
        # Temporal consistency (simplified)
        temporal_consistency = 1.0  # Default to perfect if no temporal validation needed
        
        # Calculate overall quality score
        quality_score = self._calculate_quality_score(
            missing_ratio, duplicate_ratio, variance_ratio, high_corr_ratio, temporal_consistency
        )
        
        metrics = {
            'n_samples': n_samples,
            'n_features': n_features,
            'missing_ratio': missing_ratio,
            'duplicate_ratio': duplicate_ratio,
            'variance_ratio': variance_ratio,
            'correlation_ratio': high_corr_ratio,
            'temporal_consistency': temporal_consistency,
            'data_quality_score': quality_score
        }
        
        tprint_debug(f"   📊 Quality score: {quality_score:.3f}")
        return metrics
    
    def _calculate_quality_score(
        self, 
        missing_ratio: float, 
        duplicate_ratio: float, 
        variance_ratio: float, 
        correlation_ratio: float, 
        temporal_consistency: float
    ) -> float:
        """Calculate overall data quality score."""
        # Weighted combination of quality metrics
        weights = {
            'missing': 0.3,
            'duplicate': 0.2,
            'variance': 0.2,
            'correlation': 0.2,
            'temporal': 0.1
        }
        
        # Convert ratios to scores (higher is better)
        missing_score = max(0, 1 - missing_ratio / self.thresholds['max_missing_ratio'])
        duplicate_score = max(0, 1 - duplicate_ratio / self.thresholds['max_duplicate_ratio'])
        variance_score = min(1, variance_ratio / self.thresholds['min_variance_ratio'])
        correlation_score = max(0, 1 - correlation_ratio / self.thresholds['max_correlation_ratio'])
        temporal_score = temporal_consistency
        
        # Calculate weighted average
        quality_score = (
            weights['missing'] * missing_score +
            weights['duplicate'] * duplicate_score +
            weights['variance'] * variance_score +
            weights['correlation'] * correlation_score +
            weights['temporal'] * temporal_score
        )
        
        return quality_score
    
    def _validate_data_quality(self, data: pd.DataFrame, metrics: Dict[str, Any]) -> Tuple[List[str], List[str]]:
        """Validate data quality based on metrics."""
        errors = []
        warnings = []
        
        # Check missing data
        if metrics['missing_ratio'] > self.thresholds['max_missing_ratio']:
            errors.append(f"Missing data ratio ({metrics['missing_ratio']:.3f}) exceeds threshold ({self.thresholds['max_missing_ratio']})")
        elif metrics['missing_ratio'] > self.thresholds['max_missing_ratio'] * 0.5:
            warnings.append(f"Missing data ratio ({metrics['missing_ratio']:.3f}) is high")
        
        # Check duplicate data
        if metrics['duplicate_ratio'] > self.thresholds['max_duplicate_ratio']:
            errors.append(f"Duplicate data ratio ({metrics['duplicate_ratio']:.3f}) exceeds threshold ({self.thresholds['max_duplicate_ratio']})")
        elif metrics['duplicate_ratio'] > self.thresholds['max_duplicate_ratio'] * 0.5:
            warnings.append(f"Duplicate data ratio ({metrics['duplicate_ratio']:.3f}) is high")
        
        # Check variance
        if metrics['variance_ratio'] < self.thresholds['min_variance_ratio']:
            errors.append(f"Variance ratio ({metrics['variance_ratio']:.3f}) below threshold ({self.thresholds['min_variance_ratio']})")
        elif metrics['variance_ratio'] < self.thresholds['min_variance_ratio'] * 2:
            warnings.append(f"Variance ratio ({metrics['variance_ratio']:.3f}) is low")
        
        # Check correlation
        if metrics['correlation_ratio'] > self.thresholds['max_correlation_ratio']:
            errors.append(f"High correlation ratio ({metrics['correlation_ratio']:.3f}) exceeds threshold ({self.thresholds['max_correlation_ratio']})")
        elif metrics['correlation_ratio'] > self.thresholds['max_correlation_ratio'] * 0.8:
            warnings.append(f"Correlation ratio ({metrics['correlation_ratio']:.3f}) is high")
        
        # Check overall quality score
        if metrics['data_quality_score'] < 0.5:
            errors.append(f"Overall data quality score ({metrics['data_quality_score']:.3f}) is too low")
        elif metrics['data_quality_score'] < 0.7:
            warnings.append(f"Overall data quality score ({metrics['data_quality_score']:.3f}) is moderate")
        
        return errors, warnings
    
    def _validate_target(self, target: pd.Series, data: pd.DataFrame) -> Tuple[List[str], List[str]]:
        """Validate target variable."""
        errors = []
        warnings = []
        
        # Check length match
        if len(target) != len(data):
            errors.append(f"Target length ({len(target)}) doesn't match data length ({len(data)})")
            return errors, warnings
        
        # Check for missing values
        if target.isnull().any():
            missing_count = target.isnull().sum()
            errors.append(f"Target contains {missing_count} missing values")
        
        # Check for infinite values
        if np.isinf(target).any():
            errors.append("Target contains infinite values")
        
        # Check data type
        if not pd.api.types.is_numeric_dtype(target):
            warnings.append("Target is not numeric, may need encoding")
        
        # Check variance
        if target.var() == 0:
            errors.append("Target has zero variance")
        elif target.var() < 1e-10:
            warnings.append("Target has very low variance")
        
        return errors, warnings
    
    def _validate_temporal_consistency(self, data: pd.DataFrame) -> Tuple[List[str], List[str]]:
        """Validate temporal consistency if applicable."""
        errors = []
        warnings = []
        
        # Check if data has datetime index
        if isinstance(data.index, pd.DatetimeIndex):
            # Check for gaps in time series
            time_diffs = data.index.to_series().diff()
            if time_diffs.isnull().sum() > 1:  # More than one gap
                warnings.append("Time series has gaps")
            
            # Check for duplicate timestamps
            if data.index.duplicated().any():
                errors.append("Time series has duplicate timestamps")
            
            # Check for time order
            if not data.index.is_monotonic_increasing:
                warnings.append("Time series is not in chronological order")
        
        return errors, warnings
    
    def _clean_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Clean data based on validation results."""
        tprint_debug("🧹 Cleaning data")
        
        cleaned_data = data.copy()
        
        # Remove rows with all NaN values
        cleaned_data = cleaned_data.dropna(how='all')
        
        # Remove columns with all NaN values
        cleaned_data = cleaned_data.dropna(axis=1, how='all')
        
        # Fill remaining NaN values with median for numeric columns
        numeric_cols = cleaned_data.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if cleaned_data[col].isnull().any():
                median_val = cleaned_data[col].median()
                cleaned_data[col] = cleaned_data[col].fillna(median_val)
        
        # Remove duplicate rows
        cleaned_data = cleaned_data.drop_duplicates()
        
        tprint_debug(f"   🧹 Cleaned data: {data.shape} -> {cleaned_data.shape}")
        return cleaned_data
    
    def validate_feature_selection_input(
        self, 
        X: pd.DataFrame, 
        y: pd.Series, 
        feature_names: Optional[List[str]] = None
    ) -> DataValidationResult:
        """Validate input for feature selection operations."""
        tprint_info("🔍 Validating feature selection input")
        
        # Validate feature matrix
        X_result = self.validate_data(X)
        if not X_result.is_valid:
            return X_result
        
        # Validate target
        y_result = self.validate_data(pd.DataFrame({'target': y}))
        if not y_result.is_valid:
            return DataValidationResult(
                is_valid=False,
                errors=y_result.errors,
                warnings=y_result.warnings,
                quality_metrics={}
            )
        
        # Validate feature names if provided
        if feature_names is not None:
            if len(feature_names) != X.shape[1]:
                return DataValidationResult(
                    is_valid=False,
                    errors=[f"Feature names length ({len(feature_names)}) doesn't match data columns ({X.shape[1]})"],
                    warnings=[],
                    quality_metrics={}
                )
        
        # Combine results
        combined_errors = X_result.errors + y_result.errors
        combined_warnings = X_result.warnings + y_result.warnings
        combined_metrics = {**X_result.quality_metrics, **y_result.quality_metrics}
        
        return DataValidationResult(
            is_valid=len(combined_errors) == 0,
            errors=combined_errors,
            warnings=combined_warnings,
            quality_metrics=combined_metrics,
            validated_data=X_result.validated_data
        )
    
    def get_validation_summary(self, result: DataValidationResult) -> Dict[str, Any]:
        """Get a summary of validation results."""
        return {
            'is_valid': result.is_valid,
            'error_count': len(result.errors),
            'warning_count': len(result.warnings),
            'quality_score': result.quality_metrics.get('data_quality_score', 0.0),
            'data_shape': result.validated_data.shape if result.validated_data is not None else None,
            'has_cleaned_data': result.validated_data is not None
        }