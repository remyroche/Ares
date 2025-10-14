"""
Comprehensive Validation System for UnifiedDataDrivenPipeline

This module provides comprehensive validation functionality from FeatureLookbackOptimizationComponent
integrated into the UnifiedDataDrivenPipeline, including:
- Multi-level validation (basic, standard, strict, exhaustive)
- Advanced data validation with stationarity tests
- Sophisticated error categorization and handling
- Memory monitoring and optimization
- Performance validation and metrics
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass
from enum import Enum
import time
import logging
import psutil
import gc
from pathlib import Path
import json
import pickle

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

# Import stationarity tests
try:
    from statsmodels.tsa.stattools import adfuller, kpss
    STATIONARITY_TESTS_AVAILABLE = True
except ImportError:
    STATIONARITY_TESTS_AVAILABLE = False
    adfuller = None
    kpss = None

logger = logging.getLogger(__name__)


class ValidationLevel(Enum):
    """Validation levels for input data."""
    BASIC = "basic"
    STANDARD = "standard"
    STRICT = "strict"
    EXHAUSTIVE = "exhaustive"


class ErrorSeverity(Enum):
    """Error severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ErrorCategory(Enum):
    """Error categories."""
    VALIDATION = "validation"
    COMPUTATION = "computation"
    MEMORY = "memory"
    IO = "io"
    NETWORK = "network"
    SYSTEM = "system"
    UNKNOWN = "unknown"


@dataclass
class ValidationSummary:
    """Summary of validation results."""
    is_valid: bool
    validation_level: ValidationLevel
    errors: List[str]
    warnings: List[str]
    recommendations: List[str]
    validation_time: float
    n_checks_performed: int
    memory_usage_mb: float
    data_quality_score: float


@dataclass
class ErrorInfo:
    """Information about an error."""
    error_id: str
    category: ErrorCategory
    severity: ErrorSeverity
    message: str
    component: str
    timestamp: float
    context: Dict[str, Any]
    stack_trace: Optional[str] = None


@dataclass
class PerformanceValidationResult:
    """Result of performance validation."""
    memory_usage_mb: float
    peak_memory_usage_mb: float
    cpu_usage_percent: float
    execution_time: float
    data_size_mb: float
    validation_passed: bool
    recommendations: List[str]


class ComprehensiveValidator:
    """
    Comprehensive validator with multi-level validation and advanced checks.
    
    Features:
    - Multi-level validation (basic, standard, strict, exhaustive)
    - Advanced data validation with stationarity tests
    - Sophisticated error categorization and handling
    - Memory monitoring and optimization
    - Performance validation and metrics
    """
    
    def __init__(self, component_name: str = "ComprehensiveValidator"):
        """Initialize the comprehensive validator."""
        self.component_name = component_name
        self.logger = logging.getLogger(component_name)
        
        # Validation statistics
        self.validation_stats = {
            'total_validations': 0,
            'successful_validations': 0,
            'failed_validations': 0,
            'validation_time': 0.0,
            'errors_by_category': {category.value: 0 for category in ErrorCategory},
            'errors_by_severity': {severity.value: 0 for severity in ErrorSeverity},
            'recent_errors': []
        }
        
        # Memory monitoring
        self.memory_stats = {
            'peak_memory_usage': 0.0,
            'current_memory_usage': 0.0,
            'memory_warnings': 0,
            'memory_critical': 0
        }
        
        tprint_success("✅ Comprehensive Validator initialized")
    
    def validate_data_comprehensive(
        self,
        data: Any,
        required_columns: List[str],
        validation_level: ValidationLevel = ValidationLevel.STANDARD,
        check_stationarity: bool = True,
        check_memory: bool = True
    ) -> Tuple[bool, ValidationSummary, Any]:
        """
        Perform comprehensive validation with specified level.
        
        Args:
            data: Input data to validate
            required_columns: List of required column names
            validation_level: Level of validation to perform
            check_stationarity: Whether to check stationarity
            check_memory: Whether to check memory usage
            
        Returns:
            Tuple of (is_valid, validation_summary, cleaned_data)
        """
        start_time = time.time()
        
        try:
            self.validation_stats['total_validations'] += 1
            
            # Basic validation
            if not self._basic_validation(data, required_columns):
                return False, self._create_validation_summary(
                    False, validation_level, ["Basic validation failed"], [], [], 
                    time.time() - start_time, 1, 0.0, 0.0
                ), None
            
            # Level-specific validation
            if validation_level == ValidationLevel.BASIC:
                return True, self._create_validation_summary(
                    True, validation_level, [], [], [], 
                    time.time() - start_time, 1, self._get_memory_usage(), 1.0
                ), data
            
            # Standard validation
            errors, warnings, recommendations = self._standard_validation(data, required_columns)
            if validation_level == ValidationLevel.STANDARD:
                is_valid = len(errors) == 0
                return is_valid, self._create_validation_summary(
                    is_valid, validation_level, errors, warnings, recommendations,
                    time.time() - start_time, 2, self._get_memory_usage(), self._calculate_data_quality_score(data)
                ), data
            
            # Strict validation
            errors, warnings, recommendations = self._strict_validation(data, required_columns)
            if validation_level == ValidationLevel.STRICT:
                is_valid = len(errors) == 0
                return is_valid, self._create_validation_summary(
                    is_valid, validation_level, errors, warnings, recommendations,
                    time.time() - start_time, 3, self._get_memory_usage(), self._calculate_data_quality_score(data)
                ), data
            
            # Exhaustive validation
            errors, warnings, recommendations = self._exhaustive_validation(
                data, required_columns, check_stationarity, check_memory
            )
            is_valid = len(errors) == 0
            
            validation_time = time.time() - start_time
            self.validation_stats['validation_time'] += validation_time
            
            if is_valid:
                self.validation_stats['successful_validations'] += 1
            else:
                self.validation_stats['failed_validations'] += 1
            
            return is_valid, self._create_validation_summary(
                is_valid, validation_level, errors, warnings, recommendations,
                validation_time, 4, self._get_memory_usage(), self._calculate_data_quality_score(data)
            ), data
            
        except Exception as e:
            self.logger.error(f"Validation failed: {e}")
            return False, self._create_validation_summary(
                False, validation_level, [f"Validation error: {e}"], [], [],
                time.time() - start_time, 0, self._get_memory_usage(), 0.0
            ), None
    
    def _basic_validation(self, data: Any, required_columns: List[str]) -> bool:
        """Perform basic validation."""
        try:
            if data is None:
                return False
            
            if not hasattr(data, 'columns'):
                return False
            
            if not hasattr(data, 'shape'):
                return False
            
            if data.empty:
                return False
            
            return True
        except:
            return False
    
    def _standard_validation(
        self, 
        data: pd.DataFrame, 
        required_columns: List[str]
    ) -> Tuple[List[str], List[str], List[str]]:
        """Perform standard validation."""
        errors = []
        warnings = []
        recommendations = []
        
        try:
            # Check required columns
            missing_columns = [col for col in required_columns if col not in data.columns]
            if missing_columns:
                errors.append(f"Missing required columns: {missing_columns}")
            
            # Check data types
            for col in required_columns:
                if col in data.columns:
                    if not pd.api.types.is_numeric_dtype(data[col]):
                        warnings.append(f"Column {col} is not numeric")
                        recommendations.append(f"Consider converting {col} to numeric")
            
            # Check for excessive missing values
            missing_ratio = data.isnull().sum().sum() / (data.shape[0] * data.shape[1])
            if missing_ratio > 0.5:
                warnings.append(f"High missing value ratio: {missing_ratio:.2%}")
                recommendations.append("Consider data imputation or removal of rows/columns")
            
            # Check data size
            if data.shape[0] < 10:
                warnings.append("Very small dataset")
                recommendations.append("Consider using more data for reliable results")
            
        except Exception as e:
            errors.append(f"Standard validation error: {e}")
        
        return errors, warnings, recommendations
    
    def _strict_validation(
        self, 
        data: pd.DataFrame, 
        required_columns: List[str]
    ) -> Tuple[List[str], List[str], List[str]]:
        """Perform strict validation."""
        errors, warnings, recommendations = self._standard_validation(data, required_columns)
        
        try:
            # Check for infinite values
            inf_count = np.isinf(data.select_dtypes(include=[np.number])).sum().sum()
            if inf_count > 0:
                errors.append(f"Found {inf_count} infinite values")
                recommendations.append("Remove or replace infinite values")
            
            # Check for extreme outliers
            for col in data.select_dtypes(include=[np.number]).columns:
                if col in required_columns:
                    q99 = data[col].quantile(0.99)
                    q01 = data[col].quantile(0.01)
                    iqr = q99 - q01
                    outliers = ((data[col] > q99 + 3 * iqr) | (data[col] < q01 - 3 * iqr)).sum()
                    if outliers > len(data) * 0.05:  # More than 5% outliers
                        warnings.append(f"Column {col} has {outliers} extreme outliers")
                        recommendations.append(f"Consider outlier treatment for {col}")
            
            # Check for constant columns
            constant_columns = data.columns[data.nunique() <= 1]
            if len(constant_columns) > 0:
                warnings.append(f"Constant columns found: {constant_columns.tolist()}")
                recommendations.append("Remove constant columns")
            
        except Exception as e:
            errors.append(f"Strict validation error: {e}")
        
        return errors, warnings, recommendations
    
    def _exhaustive_validation(
        self, 
        data: pd.DataFrame, 
        required_columns: List[str],
        check_stationarity: bool = True,
        check_memory: bool = True
    ) -> Tuple[List[str], List[str], List[str]]:
        """Perform exhaustive validation."""
        errors, warnings, recommendations = self._strict_validation(data, required_columns)
        
        try:
            # Check for duplicate rows
            duplicate_rows = data.duplicated().sum()
            if duplicate_rows > 0:
                warnings.append(f"Found {duplicate_rows} duplicate rows")
                recommendations.append("Consider removing duplicate rows")
            
            # Check for perfect correlations
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 1:
                corr_matrix = data[numeric_cols].corr()
                perfect_corr_pairs = []
                for i in range(len(corr_matrix.columns)):
                    for j in range(i+1, len(corr_matrix.columns)):
                        if abs(corr_matrix.iloc[i, j]) > 0.99:
                            perfect_corr_pairs.append((corr_matrix.columns[i], corr_matrix.columns[j]))
                
                if perfect_corr_pairs:
                    warnings.append(f"Perfect correlations found: {perfect_corr_pairs}")
                    recommendations.append("Consider removing highly correlated features")
            
            # Check memory usage
            if check_memory:
                memory_usage = data.memory_usage(deep=True).sum() / 1024 / 1024  # MB
                if memory_usage > 1000:  # More than 1GB
                    warnings.append(f"High memory usage: {memory_usage:.1f} MB")
                    recommendations.append("Consider data optimization or chunking")
                
                # Update memory stats
                self.memory_stats['current_memory_usage'] = memory_usage
                self.memory_stats['peak_memory_usage'] = max(
                    self.memory_stats['peak_memory_usage'], memory_usage
                )
            
            # Check stationarity
            if check_stationarity and STATIONARITY_TESTS_AVAILABLE:
                for col in required_columns:
                    if col in data.columns and pd.api.types.is_numeric_dtype(data[col]):
                        series = data[col].dropna()
                        if len(series) > 50:  # Minimum length for stationarity tests
                            try:
                                # ADF test
                                adf_stat, adf_pvalue, _, _, adf_critical, _ = adfuller(series)
                                if adf_pvalue > 0.05:
                                    warnings.append(f"Column {col} may be non-stationary (ADF p-value: {adf_pvalue:.4f})")
                                    recommendations.append(f"Consider differencing or detrending {col}")
                                
                                # KPSS test
                                kpss_stat, kpss_pvalue, _, kpss_critical = kpss(series, regression='c')
                                if kpss_pvalue < 0.05:
                                    warnings.append(f"Column {col} may be non-stationary (KPSS p-value: {kpss_pvalue:.4f})")
                                    recommendations.append(f"Consider differencing or detrending {col}")
                                
                            except Exception as e:
                                warnings.append(f"Stationarity test failed for {col}: {e}")
            
        except Exception as e:
            errors.append(f"Exhaustive validation error: {e}")
        
        return errors, warnings, recommendations
    
    def validate_performance(
        self,
        data: pd.DataFrame,
        max_memory_mb: float = 1000.0,
        max_execution_time: float = 300.0
    ) -> PerformanceValidationResult:
        """
        Validate performance characteristics of the data and system.
        
        Args:
            data: Input data
            max_memory_mb: Maximum allowed memory usage in MB
            max_execution_time: Maximum allowed execution time in seconds
            
        Returns:
            PerformanceValidationResult with performance metrics
        """
        start_time = time.time()
        
        try:
            # Get current memory usage
            current_memory = self._get_memory_usage()
            peak_memory = self.memory_stats['peak_memory_usage']
            
            # Get CPU usage
            cpu_usage = psutil.cpu_percent()
            
            # Calculate data size
            data_size = data.memory_usage(deep=True).sum() / 1024 / 1024  # MB
            
            # Check performance constraints
            memory_passed = current_memory <= max_memory_mb
            execution_time = time.time() - start_time
            time_passed = execution_time <= max_execution_time
            
            validation_passed = memory_passed and time_passed
            
            # Generate recommendations
            recommendations = []
            if not memory_passed:
                recommendations.append(f"Memory usage ({current_memory:.1f} MB) exceeds limit ({max_memory_mb} MB)")
                recommendations.append("Consider data chunking or optimization")
            
            if not time_passed:
                recommendations.append(f"Execution time ({execution_time:.1f}s) exceeds limit ({max_execution_time}s)")
                recommendations.append("Consider parallel processing or optimization")
            
            if data_size > max_memory_mb * 0.8:
                recommendations.append("Data size is approaching memory limit")
                recommendations.append("Consider data compression or sampling")
            
            return PerformanceValidationResult(
                memory_usage_mb=current_memory,
                peak_memory_usage_mb=peak_memory,
                cpu_usage_percent=cpu_usage,
                execution_time=execution_time,
                data_size_mb=data_size,
                validation_passed=validation_passed,
                recommendations=recommendations
            )
            
        except Exception as e:
            self.logger.error(f"Performance validation failed: {e}")
            return PerformanceValidationResult(
                memory_usage_mb=0.0,
                peak_memory_usage_mb=0.0,
                cpu_usage_percent=0.0,
                execution_time=time.time() - start_time,
                data_size_mb=0.0,
                validation_passed=False,
                recommendations=[f"Performance validation error: {e}"]
            )
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        try:
            process = psutil.Process()
            memory_mb = process.memory_info().rss / 1024 / 1024
            return memory_mb
        except:
            return 0.0
    
    def _calculate_data_quality_score(self, data: pd.DataFrame) -> float:
        """Calculate data quality score (0-1)."""
        try:
            if data.empty:
                return 0.0
            
            # Calculate various quality metrics
            completeness = 1.0 - (data.isnull().sum().sum() / (data.shape[0] * data.shape[1]))
            
            # Check for constant columns
            constant_ratio = (data.nunique() <= 1).sum() / len(data.columns)
            variability = 1.0 - constant_ratio
            
            # Check for infinite values
            inf_ratio = np.isinf(data.select_dtypes(include=[np.number])).sum().sum() / (data.shape[0] * data.shape[1])
            finiteness = 1.0 - inf_ratio
            
            # Combine metrics
            quality_score = (completeness + variability + finiteness) / 3.0
            return max(0.0, min(1.0, quality_score))
            
        except:
            return 0.0
    
    def _create_validation_summary(
        self,
        is_valid: bool,
        validation_level: ValidationLevel,
        errors: List[str],
        warnings: List[str],
        recommendations: List[str],
        validation_time: float,
        n_checks: int,
        memory_usage: float,
        data_quality_score: float
    ) -> ValidationSummary:
        """Create validation summary."""
        return ValidationSummary(
            is_valid=is_valid,
            validation_level=validation_level,
            errors=errors,
            warnings=warnings,
            recommendations=recommendations,
            validation_time=validation_time,
            n_checks_performed=n_checks,
            memory_usage_mb=memory_usage,
            data_quality_score=data_quality_score
        )
    
    def handle_error(
        self,
        error: Exception,
        category: ErrorCategory = ErrorCategory.UNKNOWN,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        context: Optional[Dict[str, Any]] = None
    ) -> ErrorInfo:
        """Handle an error with standardized categorization."""
        try:
            error_id = f"{self.component_name}_{int(time.time() * 1000)}"
            timestamp = time.time()
            
            # Create error info
            error_info = ErrorInfo(
                error_id=error_id,
                category=category,
                severity=severity,
                message=str(error),
                component=self.component_name,
                timestamp=timestamp,
                context=context or {},
                stack_trace=self._get_stack_trace(error)
            )
            
            # Update statistics
            self.validation_stats['errors_by_category'][category.value] += 1
            self.validation_stats['errors_by_severity'][severity.value] += 1
            
            # Add to recent errors (keep last 100)
            self.validation_stats['recent_errors'].append(error_info)
            if len(self.validation_stats['recent_errors']) > 100:
                self.validation_stats['recent_errors'] = self.validation_stats['recent_errors'][-100:]
            
            # Log based on severity
            if severity == ErrorSeverity.CRITICAL:
                self.logger.critical(f"CRITICAL ERROR [{error_id}]: {error}")
            elif severity == ErrorSeverity.HIGH:
                self.logger.error(f"HIGH ERROR [{error_id}]: {error}")
            elif severity == ErrorSeverity.MEDIUM:
                self.logger.warning(f"MEDIUM ERROR [{error_id}]: {error}")
            else:
                self.logger.info(f"LOW ERROR [{error_id}]: {error}")
            
            return error_info
            
        except Exception as e:
            self.logger.error(f"Error handling failed: {e}")
            return ErrorInfo(
                error_id="error_handler_failed",
                category=ErrorCategory.SYSTEM,
                severity=ErrorSeverity.CRITICAL,
                message=f"Error handler failed: {e}",
                component=self.component_name,
                timestamp=time.time(),
                context={}
            )
    
    def _get_stack_trace(self, error: Exception) -> str:
        """Get stack trace for an error."""
        try:
            import traceback
            return traceback.format_exc()
        except:
            return str(error)
    
    def get_validation_stats(self) -> Dict[str, Any]:
        """Get validation statistics."""
        return self.validation_stats.copy()
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory statistics."""
        return self.memory_stats.copy()
    
    def reset_stats(self):
        """Reset validation statistics."""
        self.validation_stats = {
            'total_validations': 0,
            'successful_validations': 0,
            'failed_validations': 0,
            'validation_time': 0.0,
            'errors_by_category': {category.value: 0 for category in ErrorCategory},
            'errors_by_severity': {severity.value: 0 for severity in ErrorSeverity},
            'recent_errors': []
        }
        self.memory_stats = {
            'peak_memory_usage': 0.0,
            'current_memory_usage': 0.0,
            'memory_warnings': 0,
            'memory_critical': 0
        }


def create_comprehensive_validator(component_name: str = "ComprehensiveValidator") -> ComprehensiveValidator:
    """Create a comprehensive validator with default configuration."""
    return ComprehensiveValidator(component_name)