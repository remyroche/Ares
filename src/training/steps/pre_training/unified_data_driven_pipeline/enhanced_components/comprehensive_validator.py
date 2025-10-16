"""
Comprehensive Validation Framework

This module implements a comprehensive validation system that includes:
- Input validation with schema checking
- Temporal alignment validation
- Target column validation and selection
- Data quality assessment
- Performance monitoring integration
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass
from enum import Enum
import logging
import time
from pathlib import Path
import warnings

try:
    from src.utils.tprint import (
        tprint, tprint_info, tprint_success, tprint_warning, tprint_error, tprint_debug
    )
    TPRINT_AVAILABLE = True
except ImportError:
    TPRINT_AVAILABLE = False
    def tprint(*args, **kwargs): print("TPRINT:", *args, **kwargs)
    def tprint_info(*args, **kwargs): print("INFO:", *args, **kwargs)
    def tprint_success(*args, **kwargs): print("SUCCESS:", *args, **kwargs)
    def tprint_warning(*args, **kwargs): print("WARNING:", *args, **kwargs)
    def tprint_error(*args, **kwargs): print("ERROR:", *args, **kwargs)
    def tprint_debug(*args, **kwargs): print("DEBUG:", *args, **kwargs)

logger = logging.getLogger(__name__)

class ValidationLevel(Enum):
    """Validation levels."""
    BASIC = "basic"
    STANDARD = "standard"
    STRICT = "strict"
    COMPREHENSIVE = "comprehensive"

class ValidationStatus(Enum):
    """Validation status."""
    PASSED = "passed"
    FAILED = "failed"
    WARNING = "warning"
    SKIPPED = "skipped"

class ErrorSeverity(Enum):
    """Error severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class ErrorCategory(Enum):
    """Error categories."""
    DATA_QUALITY = "data_quality"
    TEMPORAL_ALIGNMENT = "temporal_alignment"
    SCHEMA_VALIDATION = "schema_validation"
    TARGET_VALIDATION = "target_validation"
    PERFORMANCE = "performance"
    MEMORY = "memory"

@dataclass
class ValidationResult:
    """Result of a validation operation."""
    is_valid: bool
    status: ValidationStatus
    quality_score: float
    errors: List[str]
    warnings: List[str]
    recommendations: List[str]
    metadata: Dict[str, Any]
    validation_time: float

@dataclass
class ValidationSummary:
    """Summary of validation results."""
    overall_status: ValidationStatus
    quality_score: float
    total_validations: int
    passed_validations: int
    failed_validations: int
    warning_validations: int
    validation_results: Dict[str, ValidationResult]
    recommendations: List[str]
    execution_time: float

@dataclass
class ValidationConfig:
    """Configuration for validation framework."""
    validation_level: ValidationLevel = ValidationLevel.STANDARD
    enable_temporal_validation: bool = True
    enable_schema_validation: bool = True
    enable_target_validation: bool = True
    enable_performance_validation: bool = True
    quality_threshold: float = 0.7
    temporal_tolerance_seconds: int = 3600  # 1 hour
    memory_warning_threshold_mb: float = 1000.0
    memory_critical_threshold_mb: float = 2000.0
    max_missing_data_ratio: float = 0.1
    min_data_points: int = 100
    required_columns: List[str] = None
    target_column_patterns: List[str] = None

class ComprehensiveValidator:
    """
    Comprehensive validation framework for the unified pipeline.

    This validator provides multiple levels of validation including:
    - Data quality assessment
    - Schema validation
    - Temporal alignment checking
    - Target column validation
    - Performance monitoring
    """

    def __init__(self, config: Optional[ValidationConfig] = None):
        """Initialize the comprehensive validator."""
        self.config = config or ValidationConfig()

        # Set default required columns
        if self.config.required_columns is None:
            self.config.required_columns = ['open', 'high', 'low', 'close', 'volume']

        # Set default target column patterns
        if self.config.target_column_patterns is None:
            self.config.target_column_patterns = [
                'target', 'label', 'return', 'profit', 'pnl', 'y'
            ]

        # Performance tracking
        self.performance_stats = {
            'total_validations': 0,
            'successful_validations': 0,
            'failed_validations': 0,
            'warning_validations': 0,
            'total_execution_time': 0.0,
            'validation_breakdown': {}
        }

        tprint_success("✅ Comprehensive Validator initialized")

    def validate_data(
        self,
        data: pd.DataFrame,
        required_columns: Optional[List[str]] = None,
        validation_level: Optional[ValidationLevel] = None
    ) -> Tuple[bool, ValidationSummary, pd.DataFrame]:
        """
        Comprehensive data validation.

        Args:
            data: Input data to validate
            required_columns: Optional list of required columns
            validation_level: Optional validation level override

        Returns:
            Tuple of (is_valid, validation_summary, cleaned_data)
        """
        tprint_info("🔍 Starting comprehensive data validation")
        start_time = time.time()

        try:
            # Use provided parameters or defaults
            req_columns = required_columns or self.config.required_columns
            val_level = validation_level or self.config.validation_level

            validation_results = {}

            # Basic data structure validation
            basic_result = self._validate_basic_structure(data, req_columns)
            validation_results['basic_structure'] = basic_result

            if not basic_result.is_valid:
                return False, self._create_validation_summary(validation_results, start_time), data

            # Schema validation
            if self.config.enable_schema_validation:
                schema_result = self._validate_schema(data, val_level)
                validation_results['schema'] = schema_result

            # Data quality validation
            quality_result = self._validate_data_quality(data, val_level)
            validation_results['data_quality'] = quality_result

            # Temporal validation
            if self.config.enable_temporal_validation:
                temporal_result = self._validate_temporal_alignment(data, val_level)
                validation_results['temporal'] = temporal_result

            # Target validation
            if self.config.enable_target_validation:
                target_result = self._validate_target_columns(data, val_level)
                validation_results['targets'] = target_result

            # Performance validation
            if self.config.enable_performance_validation:
                perf_result = self._validate_performance_requirements(data, val_level)
                validation_results['performance'] = perf_result

            # Clean data based on validation results
            cleaned_data = self._clean_data(data, validation_results)

            # Create summary
            summary = self._create_validation_summary(validation_results, start_time)

            # Update performance stats
            self._update_performance_stats(validation_results, start_time)

            is_valid = summary.overall_status in [ValidationStatus.PASSED, ValidationStatus.WARNING]

            if is_valid:
                tprint_success("✅ Data validation passed")
            else:
                tprint_error("❌ Data validation failed")

            return is_valid, summary, cleaned_data

        except Exception as e:
            tprint_error(f"❌ Validation failed with error: {e}")
            return False, self._create_error_summary(str(e), start_time), data

    def _validate_basic_structure(
        self,
        data: pd.DataFrame,
        required_columns: List[str]
    ) -> ValidationResult:
        """Validate basic data structure."""
        start_time = time.time()
        errors = []
        warnings = []

        try:
            # Check if data is DataFrame
            if not isinstance(data, pd.DataFrame):
                errors.append("Data must be a pandas DataFrame")
                return ValidationResult(
                    is_valid=False,
                    status=ValidationStatus.FAILED,
                    quality_score=0.0,
                    errors=errors,
                    warnings=warnings,
                    recommendations=[],
                    metadata={'data_type': type(data).__name__},
                    validation_time=time.time() - start_time
                )

            # Check if data is empty
            if data.empty:
                errors.append("Data is empty")
                return ValidationResult(
                    is_valid=False,
                    status=ValidationStatus.FAILED,
                    quality_score=0.0,
                    errors=errors,
                    warnings=warnings,
                    recommendations=[],
                    metadata={'shape': data.shape},
                    validation_time=time.time() - start_time
                )

            # Check minimum data points
            if len(data) < self.config.min_data_points:
                errors.append(f"Insufficient data points: {len(data)} < {self.config.min_data_points}")

            # Check required columns
            missing_columns = [col for col in required_columns if col not in data.columns]
            if missing_columns:
                errors.append(f"Missing required columns: {missing_columns}")

            # Check for duplicate columns
            duplicate_columns = data.columns[data.columns.duplicated()].tolist()
            if duplicate_columns:
                warnings.append(f"Duplicate columns found: {duplicate_columns}")

            is_valid = len(errors) == 0
            quality_score = 1.0 if is_valid else max(0.0, 1.0 - len(errors) * 0.2)

            return ValidationResult(
                is_valid=is_valid,
                status=ValidationStatus.PASSED if is_valid else ValidationStatus.FAILED,
                quality_score=quality_score,
                errors=errors,
                warnings=warnings,
                recommendations=[],
                metadata={
                    'shape': data.shape,
                    'columns': list(data.columns),
                    'missing_columns': missing_columns,
                    'duplicate_columns': duplicate_columns
                },
                validation_time=time.time() - start_time
            )

        except Exception as e:
            return ValidationResult(
                is_valid=False,
                status=ValidationStatus.FAILED,
                quality_score=0.0,
                errors=[f"Basic structure validation failed: {str(e)}"],
                warnings=warnings,
                recommendations=[],
                metadata={'error': str(e)},
                validation_time=time.time() - start_time
            )

    def _validate_schema(
        self,
        data: pd.DataFrame,
        validation_level: ValidationLevel
    ) -> ValidationResult:
        """Validate data schema."""
        start_time = time.time()
        errors = []
        warnings = []

        try:
            # Check data types
            numeric_columns = data.select_dtypes(include=[np.number]).columns
            non_numeric_columns = data.select_dtypes(exclude=[np.number]).columns

            if len(non_numeric_columns) > 0 and validation_level in [ValidationLevel.STRICT, ValidationLevel.COMPREHENSIVE]:
                warnings.append(f"Non-numeric columns found: {list(non_numeric_columns)}")

            # Check for infinite values
            inf_columns = []
            for col in numeric_columns:
                if np.isinf(data[col]).any():
                    inf_columns.append(col)

            if inf_columns:
                errors.append(f"Columns with infinite values: {inf_columns}")

            # Check for complex numbers
            complex_columns = []
            for col in numeric_columns:
                if np.iscomplexobj(data[col]):
                    complex_columns.append(col)

            if complex_columns:
                errors.append(f"Columns with complex numbers: {complex_columns}")

            is_valid = len(errors) == 0
            quality_score = 1.0 if is_valid else max(0.0, 1.0 - len(errors) * 0.3)

            return ValidationResult(
                is_valid=is_valid,
                status=ValidationStatus.PASSED if is_valid else ValidationStatus.FAILED,
                quality_score=quality_score,
                errors=errors,
                warnings=warnings,
                recommendations=[],
                metadata={
                    'numeric_columns': len(numeric_columns),
                    'non_numeric_columns': len(non_numeric_columns),
                    'inf_columns': inf_columns,
                    'complex_columns': complex_columns
                },
                validation_time=time.time() - start_time
            )

        except Exception as e:
            return ValidationResult(
                is_valid=False,
                status=ValidationStatus.FAILED,
                quality_score=0.0,
                errors=[f"Schema validation failed: {str(e)}"],
                warnings=warnings,
                recommendations=[],
                metadata={'error': str(e)},
                validation_time=time.time() - start_time
            )

    def _validate_data_quality(
        self,
        data: pd.DataFrame,
        validation_level: ValidationLevel
    ) -> ValidationResult:
        """Validate data quality."""
        start_time = time.time()
        errors = []
        warnings = []

        try:
            # Check missing data ratio
            missing_ratio = data.isnull().sum().sum() / (len(data) * len(data.columns))

            if missing_ratio > self.config.max_missing_data_ratio:
                errors.append(f"Too much missing data: {missing_ratio:.2%} > {self.config.max_missing_data_ratio:.2%}")
            elif missing_ratio > self.config.max_missing_data_ratio / 2:
                warnings.append(f"High missing data ratio: {missing_ratio:.2%}")

            # Check for constant columns
            constant_columns = []
            for col in data.columns:
                if data[col].nunique() <= 1:
                    constant_columns.append(col)

            if constant_columns:
                warnings.append(f"Constant columns found: {constant_columns}")

            # Check for highly correlated columns
            if validation_level in [ValidationLevel.STRICT, ValidationLevel.COMPREHENSIVE]:
                numeric_data = data.select_dtypes(include=[np.number])
                if len(numeric_data.columns) > 1:
                    corr_matrix = numeric_data.corr().abs()
                    high_corr_pairs = []
                    for i in range(len(corr_matrix.columns)):
                        for j in range(i+1, len(corr_matrix.columns)):
                            if corr_matrix.iloc[i, j] > 0.95:
                                high_corr_pairs.append((corr_matrix.columns[i], corr_matrix.columns[j]))

                    if high_corr_pairs:
                        warnings.append(f"Highly correlated columns: {high_corr_pairs[:5]}")  # Show first 5

            # Calculate quality score
            quality_score = 1.0
            quality_score -= missing_ratio * 0.5  # Penalty for missing data
            quality_score -= len(constant_columns) * 0.1  # Penalty for constant columns
            quality_score = max(0.0, quality_score)

            is_valid = len(errors) == 0 and quality_score >= self.config.quality_threshold

            return ValidationResult(
                is_valid=is_valid,
                status=ValidationStatus.PASSED if is_valid else ValidationStatus.FAILED,
                quality_score=quality_score,
                errors=errors,
                warnings=warnings,
                recommendations=[],
                metadata={
                    'missing_ratio': missing_ratio,
                    'constant_columns': constant_columns,
                    'total_columns': len(data.columns),
                    'numeric_columns': len(data.select_dtypes(include=[np.number]).columns)
                },
                validation_time=time.time() - start_time
            )

        except Exception as e:
            return ValidationResult(
                is_valid=False,
                status=ValidationStatus.FAILED,
                quality_score=0.0,
                errors=[f"Data quality validation failed: {str(e)}"],
                warnings=warnings,
                recommendations=[],
                metadata={'error': str(e)},
                validation_time=time.time() - start_time
            )

    def _validate_temporal_alignment(
        self,
        data: pd.DataFrame,
        validation_level: ValidationLevel
    ) -> ValidationResult:
        """Validate temporal alignment."""
        start_time = time.time()
        errors = []
        warnings = []

        try:
            # Check if index is datetime
            if not isinstance(data.index, pd.DatetimeIndex):
                warnings.append("Index is not datetime, temporal validation skipped")
                return ValidationResult(
                    is_valid=True,
                    status=ValidationStatus.WARNING,
                    quality_score=0.8,
                    errors=errors,
                    warnings=warnings,
                    recommendations=[],
                    metadata={'index_type': type(data.index).__name__},
                    validation_time=time.time() - start_time
                )

            # Check for duplicate timestamps
            duplicate_timestamps = data.index.duplicated().sum()
            if duplicate_timestamps > 0:
                errors.append(f"Duplicate timestamps found: {duplicate_timestamps}")

            # Check for gaps in time series
            if len(data) > 1:
                time_diffs = data.index.to_series().diff().dropna()
                if len(time_diffs) > 0:
                    median_diff = time_diffs.median()
                    large_gaps = time_diffs > median_diff * 3
                    if large_gaps.any():
                        warnings.append(f"Large time gaps detected: {large_gaps.sum()} gaps")

            # Check for future data (if applicable)
            now = pd.Timestamp.now()
            future_data = data.index > now
            if future_data.any():
                warnings.append(f"Future timestamps found: {future_data.sum()}")

            is_valid = len(errors) == 0
            quality_score = 1.0 if is_valid else max(0.0, 1.0 - len(errors) * 0.3)

            return ValidationResult(
                is_valid=is_valid,
                status=ValidationStatus.PASSED if is_valid else ValidationStatus.FAILED,
                quality_score=quality_score,
                errors=errors,
                warnings=warnings,
                recommendations=[],
                metadata={
                    'index_type': 'DatetimeIndex',
                    'duplicate_timestamps': duplicate_timestamps,
                    'time_range': (data.index.min(), data.index.max()) if len(data) > 0 else None
                },
                validation_time=time.time() - start_time
            )

        except Exception as e:
            return ValidationResult(
                is_valid=False,
                status=ValidationStatus.FAILED,
                quality_score=0.0,
                errors=[f"Temporal validation failed: {str(e)}"],
                warnings=warnings,
                recommendations=[],
                metadata={'error': str(e)},
                validation_time=time.time() - start_time
            )

    def _validate_target_columns(
        self,
        data: pd.DataFrame,
        validation_level: ValidationLevel
    ) -> ValidationResult:
        """Validate target columns."""
        start_time = time.time()
        errors = []
        warnings = []

        try:
            # Find target columns
            target_columns = []
            for pattern in self.config.target_column_patterns:
                matching_cols = [col for col in data.columns if pattern.lower() in col.lower()]
                target_columns.extend(matching_cols)

            target_columns = list(set(target_columns))  # Remove duplicates

            if not target_columns:
                warnings.append("No target columns found")
                return ValidationResult(
                    is_valid=True,
                    status=ValidationStatus.WARNING,
                    quality_score=0.5,
                    errors=errors,
                    warnings=warnings,
                    recommendations=[],
                    metadata={'target_columns': target_columns},
                    validation_time=time.time() - start_time
                )

            # Validate target columns
            valid_targets = []
            for col in target_columns:
                if col in data.columns:
                    series = data[col]

                    # Check for missing values
                    missing_ratio = series.isnull().sum() / len(series)
                    if missing_ratio > 0.5:
                        warnings.append(f"Target column {col} has high missing ratio: {missing_ratio:.2%}")

                    # Check for constant values
                    if series.nunique() <= 1:
                        warnings.append(f"Target column {col} is constant")

                    # Check for infinite values
                    if np.isinf(series).any():
                        errors.append(f"Target column {col} contains infinite values")

                    valid_targets.append(col)

            is_valid = len(errors) == 0
            quality_score = len(valid_targets) / max(1, len(target_columns)) if target_columns else 0.0

            return ValidationResult(
                is_valid=is_valid,
                status=ValidationStatus.PASSED if is_valid else ValidationStatus.FAILED,
                quality_score=quality_score,
                errors=errors,
                warnings=warnings,
                recommendations=[],
                metadata={
                    'target_columns': target_columns,
                    'valid_targets': valid_targets,
                    'total_targets': len(target_columns)
                },
                validation_time=time.time() - start_time
            )

        except Exception as e:
            return ValidationResult(
                is_valid=False,
                status=ValidationStatus.FAILED,
                quality_score=0.0,
                errors=[f"Target validation failed: {str(e)}"],
                warnings=warnings,
                recommendations=[],
                metadata={'error': str(e)},
                validation_time=time.time() - start_time
            )

    def _validate_performance_requirements(
        self,
        data: pd.DataFrame,
        validation_level: ValidationLevel
    ) -> ValidationResult:
        """Validate performance requirements."""
        start_time = time.time()
        errors = []
        warnings = []

        try:
            # Check memory usage
            memory_usage_mb = data.memory_usage(deep=True).sum() / 1024 / 1024

            if memory_usage_mb > self.config.memory_critical_threshold_mb:
                errors.append(f"Memory usage too high: {memory_usage_mb:.1f}MB > {self.config.memory_critical_threshold_mb}MB")
            elif memory_usage_mb > self.config.memory_warning_threshold_mb:
                warnings.append(f"High memory usage: {memory_usage_mb:.1f}MB")

            # Check data size
            if len(data) > 1000000:  # 1M rows
                warnings.append(f"Large dataset: {len(data):,} rows")

            if len(data.columns) > 1000:  # 1K columns
                warnings.append(f"Many columns: {len(data.columns)} columns")

            is_valid = len(errors) == 0
            quality_score = 1.0 if is_valid else max(0.0, 1.0 - len(errors) * 0.5)

            return ValidationResult(
                is_valid=is_valid,
                status=ValidationStatus.PASSED if is_valid else ValidationStatus.FAILED,
                quality_score=quality_score,
                errors=errors,
                warnings=warnings,
                recommendations=[],
                metadata={
                    'memory_usage_mb': memory_usage_mb,
                    'data_shape': data.shape,
                    'memory_threshold': self.config.memory_warning_threshold_mb
                },
                validation_time=time.time() - start_time
            )

        except Exception as e:
            return ValidationResult(
                is_valid=False,
                status=ValidationStatus.FAILED,
                quality_score=0.0,
                errors=[f"Performance validation failed: {str(e)}"],
                warnings=warnings,
                recommendations=[],
                metadata={'error': str(e)},
                validation_time=time.time() - start_time
            )

    def _clean_data(
        self,
        data: pd.DataFrame,
        validation_results: Dict[str, ValidationResult]
    ) -> pd.DataFrame:
        """Clean data based on validation results."""
        try:
            cleaned_data = data.copy()

            # Remove infinite values
            cleaned_data = cleaned_data.replace([np.inf, -np.inf], np.nan)

            # Remove constant columns if basic structure validation passed
            if 'basic_structure' in validation_results:
                basic_result = validation_results['basic_structure']
                if basic_result.is_valid:
                    constant_columns = []
                    for col in cleaned_data.columns:
                        if cleaned_data[col].nunique() <= 1:
                            constant_columns.append(col)

                    if constant_columns:
                        cleaned_data = cleaned_data.drop(columns=constant_columns)
                        tprint_debug(f"Removed {len(constant_columns)} constant columns")

            return cleaned_data

        except Exception as e:
            tprint_warning(f"⚠️ Data cleaning failed: {e}")
            return data

    def _create_validation_summary(
        self,
        validation_results: Dict[str, ValidationResult],
        start_time: float
    ) -> ValidationSummary:
        """Create validation summary."""
        try:
            total_validations = len(validation_results)
            passed_validations = sum(1 for r in validation_results.values() if r.status == ValidationStatus.PASSED)
            failed_validations = sum(1 for r in validation_results.values() if r.status == ValidationStatus.FAILED)
            warning_validations = sum(1 for r in validation_results.values() if r.status == ValidationStatus.WARNING)

            # Calculate overall quality score
            quality_scores = [r.quality_score for r in validation_results.values()]
            overall_quality_score = np.mean(quality_scores) if quality_scores else 0.0

            # Determine overall status
            if failed_validations > 0:
                overall_status = ValidationStatus.FAILED
            elif warning_validations > 0:
                overall_status = ValidationStatus.WARNING
            else:
                overall_status = ValidationStatus.PASSED

            # Collect recommendations
            recommendations = []
            for result in validation_results.values():
                recommendations.extend(result.recommendations)

            return ValidationSummary(
                overall_status=overall_status,
                quality_score=overall_quality_score,
                total_validations=total_validations,
                passed_validations=passed_validations,
                failed_validations=failed_validations,
                warning_validations=warning_validations,
                validation_results=validation_results,
                recommendations=recommendations,
                execution_time=time.time() - start_time
            )

        except Exception as e:
            tprint_error(f"❌ Failed to create validation summary: {e}")
            return self._create_error_summary(str(e), start_time)

    def _create_error_summary(self, error_message: str, start_time: float) -> ValidationSummary:
        """Create error summary."""
        return ValidationSummary(
            overall_status=ValidationStatus.FAILED,
            quality_score=0.0,
            total_validations=0,
            passed_validations=0,
            failed_validations=1,
            warning_validations=0,
            validation_results={},
            recommendations=[],
            execution_time=time.time() - start_time
        )

    def _update_performance_stats(
        self,
        validation_results: Dict[str, ValidationResult],
        start_time: float
    ):
        """Update performance statistics."""
        try:
            self.performance_stats['total_validations'] += 1
            self.performance_stats['total_execution_time'] += time.time() - start_time

            for name, result in validation_results.items():
                if name not in self.performance_stats['validation_breakdown']:
                    self.performance_stats['validation_breakdown'][name] = {
                        'total': 0, 'passed': 0, 'failed': 0, 'warning': 0
                    }

                breakdown = self.performance_stats['validation_breakdown'][name]
                breakdown['total'] += 1

                if result.status == ValidationStatus.PASSED:
                    breakdown['passed'] += 1
                    self.performance_stats['successful_validations'] += 1
                elif result.status == ValidationStatus.FAILED:
                    breakdown['failed'] += 1
                    self.performance_stats['failed_validations'] += 1
                elif result.status == ValidationStatus.WARNING:
                    breakdown['warning'] += 1
                    self.performance_stats['warning_validations'] += 1

        except Exception as e:
            tprint_warning(f"⚠️ Failed to update performance stats: {e}")

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        return self.performance_stats.copy()

def create_comprehensive_validator(
    config: Optional[ValidationConfig] = None
) -> ComprehensiveValidator:
    """Create a comprehensive validator with default configuration."""
    return ComprehensiveValidator(config)
