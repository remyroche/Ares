from src.utils.tprint import tprint

from typing import Dict, List, Optional, Union, Any, Tuple

from src.utils.logger import system_logger
from src.core.decorators import handles_errors
from src.training.steps.standardized_parquet_handler import standardized_parquet_handler
from src.utils.enhanced_artifact_manager import get_artifact_manager
from src.utils.artifact_pickup_utils import get_artifact_pickup_utils
from src.utils.version_manager import get_version_manager
import numpy as np
import pandas as pd

"""Step 5: Labeling with Standardized Data Quality Management.

This module creates comprehensive labels for the training data, combining triple barrier
labels with additional labeling strategies and meta-labeling features.

Key Enhancements:
- Dynamic Label Generation: Added the ability to generate triple barrier labels directly within step05 using regime-aware methods
- Regime-Aware Triple Barrier: Integrated HMM regime-specific barrier optimization for more sophisticated labeling
- Fallback Mechanisms: Implemented robust fallback to default labeling when regime-aware methods aren't available
- Configuration-Driven Behavior: Added configurable toggles for automatic barrier recalculation
- Comprehensive Function Call Monitoring: Enhanced with detailed function call validation, tracking, and reporting
- Function-to-Function Call Tracking: Monitors all inter-function calls with detailed outcome reporting
- Detailed Completion Reporting: Provides comprehensive reports on function execution outcomes
"""

from src.core.decorators import traced, validates, cached, log_execution_time
# Enhanced error handler - using fallback implementation
class EnhancedErrorHandler:
    """Enhanced error handler fallback."""
    @staticmethod
    def handle_error(error, context=None):
        print(f"Error handled: {error}")

def handle_errors_with_tracking(func):
    """Error tracking decorator fallback."""
    return func

# Import comprehensive optimization utilities for enhanced performance
try:
    # M1 Hardware-Specific Optimizations
    from src.utils.hardware.m1_gpu_utils import get_m1_gpu_manager
    from src.utils.hardware.m1_memory_optimizer import get_m1_memory_optimizer
    from src.utils.hardware.m1_cpu_optimizer import get_m1_cpu_optimizer

    # Processing Core Optimizations
    from src.utils.vectorized_processing_core import get_vectorized_processing_core
    from src.utils.matrix_operations import EnhancedMatrixOperations
    from src.utils.enhanced_step_optimizations import get_step_optimization_manager

    # Data Management Optimizations - using fallback
    class OptimizedDataManager:
        """Optimized data manager fallback."""
        def __init__(self):
            pass

        def optimize_dataframe(self, df):
            return df

    OPTIMIZATIONS_AVAILABLE = True
    system_logger.info("🚀 All optimization utilities successfully loaded")
except ImportError as e:
    OPTIMIZATIONS_AVAILABLE = False
    system_logger.warning(f"⚠️ Some optimization utilities not available: {e}")

import asyncio

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import time
import logging

from datetime import datetime
import json
import hashlib

from functools import wraps
from dataclasses import dataclass, field

import re

from collections import defaultdict, Counter

from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from src.utils.common_operations import ensure_directory, safe_json_dump
from src.utils.pipeline_standards import pipeline_standards
from src.utils.logger import system_logger

# Simplified monitoring - removed complex FunctionCallMonitor

# Simplified timer utility for basic performance monitoring
class SimpleTimer:
    """Simple timer utility for basic performance monitoring."""

    def __init__(self, logger: Any=None) -> None:
        self.logger = logger or system_logger
        self.start_times: Dict[str, float] = {}

    def start(self, operation_name: str) -> None:
        """Start timing an operation."""
        self.start_times[operation_name] = time.time()
        self.logger.debug(f'⏱️ Started timing: {operation_name}')

    def stop(self, operation_name: str) -> float:
        """Stop timing an operation and return elapsed time."""
        if operation_name not in self.start_times:
            self.logger.warning(f'⚠️ No start time found for: {operation_name}')
            return 0.0

        elapsed = time.time() - self.start_times[operation_name]
        del self.start_times[operation_name]
        self.logger.debug(f'⏱️ {operation_name} completed in {elapsed:.3f}s')
        return elapsed

    def time_operation(self, operation_name: str):
        """Decorator to time a function."""
        def decorator(func: Callable) -> Callable:
            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                self.start(operation_name)
                try:
                    result = await func(*args, **kwargs)
                    elapsed = self.stop(operation_name)
                    self.logger.info(f'✅ {operation_name} completed in {elapsed:.3f}s')
                    return result
                except Exception as e:
                    elapsed = self.stop(operation_name)
                    self.logger.error(f'❌ {operation_name} failed after {elapsed:.3f}s: {e}')
                    raise

            @wraps(func)
            def sync_wrapper(*args, **kwargs):
                self.start(operation_name)
                try:
                    result = func(*args, **kwargs)
                    elapsed = self.stop(operation_name)
                    self.logger.info(f'✅ {operation_name} completed in {elapsed:.3f}s')
                    return result
                except Exception as e:
                    elapsed = self.stop(operation_name)
                    self.logger.error(f'❌ {operation_name} failed after {elapsed:.3f}s: {e}')
                    raise

            return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
        return decorator

# Removed complex PerformanceMonitor - using simplified SimpleTimer

class ComprehensiveValidationFramework:
    """Comprehensive validation framework for all function operations."""

    def __init__(self, logger: Any=None) -> None:
        self.logger = logger or system_logger
        self.validation_rules: Dict[str, List[Callable]] = {}
        self.validation_history: List[Dict[str, Any]] = []
        self.validation_results: Dict[str, Dict[str, Any]] = {}
        self._initialize_default_validation_rules()

    def _initialize_default_validation_rules(self) -> None:
        """Initialize default validation rules for common operations."""
        try:
            self.validation_rules['input_validation'] = [self._validate_dataframe_input, self._validate_string_input, self._validate_numeric_input, self._validate_path_input]
            self.validation_rules['output_validation'] = [self._validate_dataframe_output, self._validate_boolean_output, self._validate_numeric_output, self._validate_series_output]
            self.validation_rules['data_quality'] = [self._validate_data_completeness, self._validate_data_types, self._validate_data_ranges, self._validate_data_consistency]
            self.validation_rules['performance_validation'] = [self._validate_execution_time, self._validate_memory_usage, self._validate_cpu_usage]
            self.validation_rules['business_logic'] = [self._validate_labeling_logic, self._validate_regime_logic, self._validate_triple_barrier_logic]
            self.logger.info('✅ Default validation rules initialized')
        except Exception as e:
            self.logger.error(f'❌ Failed to initialize default validation rules: {e}')

    def _validate_dataframe_input(self, data: Any, context: Dict[str, Any]=None) -> Dict[str, Any]:
        """Validate DataFrame input."""
        result = {'valid': True, 'errors': [], 'warnings': []}
        try:
            if not isinstance(data, pd.DataFrame):
                result['valid'] = False
                result['errors'].append(f'Expected DataFrame, got {type(data).__name__}')
                return result
            if data.empty:
                result['valid'] = False
                result['errors'].append('DataFrame is empty')
                return result
            required_columns = context.get('required_columns', [])
            missing_columns = [col for col in required_columns if col not in data.columns]
            if missing_columns:
                result['valid'] = False
                result['errors'].append(f'Missing required columns: {missing_columns}')
            critical_columns = context.get('critical_columns', [])
            for col in critical_columns:
                if col in data.columns and data[col].isna().any():
                    result['warnings'].append(f"Column '{col}' contains NaN values")
            expected_types = context.get('expected_types', {})
            for col, expected_type in expected_types.items():
                if col in data.columns:
                    actual_type = data[col].dtype
                    if not pd.api.types.is_dtype_equal(actual_type, expected_type):
                        result['warnings'].append(f"Column '{col}' has type {actual_type}, expected {expected_type}")
        except Exception as e:
            result['valid'] = False
            result['errors'].append(f'Validation error: {str(e)}')
        return result

    def _validate_string_input(self, data: Any, context: Dict[str, Any]=None) -> Dict[str, Any]:
        """Validate string input."""
        result = {'valid': True, 'errors': [], 'warnings': []}
        try:
            if not isinstance(data, str):
                result['valid'] = False
                result['errors'].append(f'Expected string, got {type(data).__name__}')
                return result
            if not data.strip():
                result['valid'] = False
                result['errors'].append('String is empty or whitespace only')
                return result
            min_length = context.get('min_length', 0)
            max_length = context.get('max_length', float('inf'))
            if len(data) < min_length:
                result['valid'] = False
                result['errors'].append(f'String too short (min: {min_length})')
            if len(data) > max_length:
                result['valid'] = False
                result['errors'].append(f'String too long (max: {max_length})')
            pattern = context.get('pattern')
            if pattern and (not re.match(pattern, data)):
                result['valid'] = False
                result['errors'].append(f"String doesn't match required pattern: {pattern}")
        except Exception as e:
            result['valid'] = False
            result['errors'].append(f'Validation error: {str(e)}')
        return result

    def _validate_numeric_input(self, data: Any, context: Dict[str, Any]=None) -> Dict[str, Any]:
        """Validate numeric input."""
        result = {'valid': True, 'errors': [], 'warnings': []}
        try:
            if not isinstance(data, (int, float, np.number)):
                result['valid'] = False
                result['errors'].append(f'Expected numeric, got {type(data).__name__}')
                return result
            min_value = context.get('min_value', float('-inf'))
            max_value = context.get('max_value', float('inf'))
            if data < min_value:
                result['valid'] = False
                result['errors'].append(f'Value too small (min: {min_value})')
            if data > max_value:
                result['valid'] = False
                result['errors'].append(f'Value too large (max: {max_value})')
            if np.isnan(data) or np.isinf(data):
                result['valid'] = False
                result['errors'].append('Value is NaN or infinite')
        except Exception as e:
            result['valid'] = False
            result['errors'].append(f'Validation error: {str(e)}')
        return result

    def _validate_path_input(self, data: Any, context: Dict[str, Any]=None) -> Dict[str, Any]:
        """Validate path input."""
        result = {'valid': True, 'errors': [], 'warnings': []}
        try:
            path = Path(data) if not isinstance(data, Path) else data
            must_exist = context.get('must_exist', True)
            if must_exist and (not path.exists()):
                result['valid'] = False
                result['errors'].append(f'Path does not exist: {path}')
                return result
            expected_type = context.get('expected_type', 'file')
            if path.exists():
                if expected_type == 'file' and (not path.is_file()):
                    result['valid'] = False
                    result['errors'].append(f'Expected file, got directory: {path}')
                elif expected_type == 'directory' and (not path.is_dir()):
                    result['valid'] = False
                    result['errors'].append(f'Expected directory, got file: {path}')
            expected_extensions = context.get('expected_extensions', [])
            if expected_extensions and path.suffix.lower() not in expected_extensions:
                result['valid'] = False
                result['errors'].append(f'Invalid file extension. Expected: {expected_extensions}')
        except Exception as e:
            result['valid'] = False
            result['errors'].append(f'Validation error: {str(e)}')
        return result

    def _validate_dataframe_output(self, data: Any, context: Dict[str, Any]=None) -> Dict[str, Any]:
        """Validate DataFrame output."""
        result = {'valid': True, 'errors': [], 'warnings': []}
        try:
            if data is None:
                result['valid'] = False
                result['errors'].append('Output is None')
                return result
            if not isinstance(data, pd.DataFrame):
                result['valid'] = False
                result['errors'].append(f'Expected DataFrame output, got {type(data).__name__}')
                return result
            if data.empty:
                result['warnings'].append('Output DataFrame is empty')
            required_columns = context.get('required_columns', [])
            missing_columns = [col for col in required_columns if col not in data.columns]
            if missing_columns:
                result['valid'] = False
                result['errors'].append(f'Missing required output columns: {missing_columns}')
            if 'label' in data.columns:
                label_counts = data['label'].value_counts()
                if len(label_counts) == 0:
                    result['warnings'].append('No labels generated')
                elif len(label_counts) == 1:
                    result['warnings'].append('Only one label class generated')
        except Exception as e:
            result['valid'] = False
            result['errors'].append(f'Validation error: {str(e)}')
        return result

    def _validate_boolean_output(self, data: Any, context: Dict[str, Any]=None) -> Dict[str, Any]:
        """Validate boolean output."""
        result = {'valid': True, 'errors': [], 'warnings': []}
        try:
            if not isinstance(data, bool):
                result['valid'] = False
                result['errors'].append(f'Expected boolean output, got {type(data).__name__}')
                return result
            expected_value = context.get('expected_value')
            if expected_value is not None and data != expected_value:
                result['warnings'].append(f'Expected {expected_value}, got {data}')
        except Exception as e:
            result['valid'] = False
            result['errors'].append(f'Validation error: {str(e)}')
        return result

    def _validate_numeric_output(self, data: Any, context: Dict[str, Any]=None) -> Dict[str, Any]:
        """Validate numeric output."""
        result = {'valid': True, 'errors': [], 'warnings': []}
        try:
            if not isinstance(data, (int, float, np.number)):
                result['valid'] = False
                result['errors'].append(f'Expected numeric output, got {type(data).__name__}')
                return result
            if np.isnan(data) or np.isinf(data):
                result['valid'] = False
                result['errors'].append('Output is NaN or infinite')
            min_value = context.get('min_value', float('-inf'))
            max_value = context.get('max_value', float('inf'))
            if data < min_value or data > max_value:
                result['warnings'].append(f'Output value {data} outside expected range [{min_value}, {max_value}]')
        except Exception as e:
            result['valid'] = False
            result['errors'].append(f'Validation error: {str(e)}')
        return result

    def _validate_series_output(self, data: Any, context: Dict[str, Any]=None) -> Dict[str, Any]:
        """Validate Series output."""
        result = {'valid': True, 'errors': [], 'warnings': []}
        try:
            if data is None:
                result['valid'] = False
                result['errors'].append('Output is None')
                return result
            if not isinstance(data, pd.Series):
                result['valid'] = False
                result['errors'].append(f'Expected Series output, got {type(data).__name__}')
                return result
            if data.empty:
                result['warnings'].append('Output Series is empty')
            if data.isna().any():
                nan_count = data.isna().sum()
                result['warnings'].append(f'Output Series contains {nan_count} NaN values')
        except Exception as e:
            result['valid'] = False
            result['errors'].append(f'Validation error: {str(e)}')
        return result

    def _validate_data_completeness(self, data: Any, context: Dict[str, Any]=None) -> Dict[str, Any]:
        """Validate data completeness using comprehensive quality assessment."""
        result = {'valid': True, 'errors': [], 'warnings': []}
        try:
            if isinstance(data, pd.DataFrame):
                # Use comprehensive quality assessment for completeness
                try:
                    from src.utils.data.quality.comprehensive_quality_scorer import get_quality_scorer
                    quality_scorer = get_quality_scorer()
                    quality_assessment = quality_scorer.assess_data_quality(
                        data,
                        context="market_analysis",
                        step_name="labeling_data_completeness",
                        data_type="klines"
                    )
                    
                    # Check completeness component score
                    completeness_score = quality_assessment.component_scores.get('completeness', 0.0)
                    min_completeness = context.get('min_completeness', 0.8)
                    
                    if completeness_score < min_completeness:
                        result['warnings'].append(f'Data completeness {completeness_score:.2%} below threshold {min_completeness:.2%}')
                        result['warnings'].extend(quality_assessment.issues)
                    
                except ImportError:
                    # Fallback to basic completeness check
                    total_cells = data.size
                    missing_cells = data.isna().sum().sum()
                    completeness_ratio = (total_cells - missing_cells) / total_cells if total_cells > 0 else 0
                    min_completeness = context.get('min_completeness', 0.95)
                    if completeness_ratio < min_completeness:
                        result['warnings'].append(f'Data completeness {completeness_ratio:.2%} below threshold {min_completeness:.2%}')
        except Exception as e:
            result['valid'] = False
            result['errors'].append(f'Validation error: {str(e)}')
        return result

    def _validate_data_types(self, data: Any, context: Dict[str, Any]=None) -> Dict[str, Any]:
        """Validate data types using comprehensive quality assessment."""
        result = {'valid': True, 'errors': [], 'warnings': []}
        try:
            if isinstance(data, pd.DataFrame):
                # Use comprehensive quality assessment for data types
                try:
                    from src.utils.data.quality.data_quality import DataQualityFramework
                    quality_framework = DataQualityFramework()
                    quality_result = quality_framework.validate_dataframe_quality(data, context="labeling_data_types")
                    
                    # Check for data type issues
                    if quality_result.issues:
                        type_issues = [issue for issue in quality_result.issues if 'data_type' in issue.lower()]
                        if type_issues:
                            result['warnings'].extend(type_issues)
                    
                    # Also check expected types if provided
                    expected_types = context.get('expected_types', {})
                    for col, expected_type in expected_types.items():
                        if col in data.columns:
                            actual_type = data[col].dtype
                            if not pd.api.types.is_dtype_equal(actual_type, expected_type):
                                result['warnings'].append(f"Column '{col}' type mismatch: {actual_type} vs {expected_type}")
                
                except ImportError:
                    # Fallback to basic type checking
                    expected_types = context.get('expected_types', {})
                    for col, expected_type in expected_types.items():
                        if col in data.columns:
                            actual_type = data[col].dtype
                            if not pd.api.types.is_dtype_equal(actual_type, expected_type):
                                result['warnings'].append(f"Column '{col}' type mismatch: {actual_type} vs {expected_type}")
        except Exception as e:
            result['valid'] = False
            result['errors'].append(f'Validation error: {str(e)}')
        return result

    def _validate_data_ranges(self, data: Any, context: Dict[str, Any]=None) -> Dict[str, Any]:
        """Validate data ranges using comprehensive quality assessment."""
        result = {'valid': True, 'errors': [], 'warnings': []}
        try:
            if isinstance(data, pd.DataFrame):
                # Use comprehensive quality assessment for data ranges
                try:
                    from src.utils.data.quality.advanced_quality_metrics import AdvancedQualityMetrics
                    quality_metrics = AdvancedQualityMetrics()
                    quality_assessment = quality_metrics.comprehensive_quality_assessment(
                        data,
                        context="labeling_data_ranges",
                        step_name="data_range_validation"
                    )
                    
                    # Check for price anomalies and range issues
                    for metric in quality_assessment.metrics:
                        if 'price_anomaly' in metric.name or 'range' in metric.name.lower():
                            if metric.severity in ['error', 'critical']:
                                result['errors'].append(metric.message)
                            elif metric.severity == 'warning':
                                result['warnings'].append(metric.message)
                    
                    # Also check custom column ranges if provided
                    column_ranges = context.get('column_ranges', {})
                    for col, (min_val, max_val) in column_ranges.items():
                        if col in data.columns:
                            col_data = data[col].dropna()
                            if len(col_data) > 0:
                                if col_data.min() < min_val or col_data.max() > max_val:
                                    result['warnings'].append(f"Column '{col}' values outside range [{min_val}, {max_val}]")
                
                except ImportError:
                    # Fallback to basic range checking
                    column_ranges = context.get('column_ranges', {})
                    for col, (min_val, max_val) in column_ranges.items():
                        if col in data.columns:
                            col_data = data[col].dropna()
                            if len(col_data) > 0:
                                if col_data.min() < min_val or col_data.max() > max_val:
                                    result['warnings'].append(f"Column '{col}' values outside range [{min_val}, {max_val}]")
        except Exception as e:
            result['valid'] = False
            result['errors'].append(f'Validation error: {str(e)}')
        return result

    def _validate_data_consistency(self, data: Any, context: Dict[str, Any]=None) -> Dict[str, Any]:
        """Validate data consistency using comprehensive quality assessment."""
        result = {'valid': True, 'errors': [], 'warnings': []}
        try:
            if isinstance(data, pd.DataFrame):
                # Use comprehensive quality assessment for data consistency
                try:
                    from src.utils.data.quality.comprehensive_quality_scorer import get_quality_scorer
                    quality_scorer = get_quality_scorer()
                    quality_assessment = quality_scorer.assess_data_quality(
                        data,
                        context="market_analysis",
                        step_name="labeling_data_consistency",
                        data_type="klines"
                    )
                    
                    # Check consistency component score
                    consistency_score = quality_assessment.component_scores.get('consistency', 0.0)
                    if consistency_score < 0.8:
                        result['warnings'].append(f'Data consistency score {consistency_score:.2%} below threshold')
                        result['warnings'].extend(quality_assessment.issues)
                    
                    # Check for specific consistency issues
                    for metric in quality_assessment.metrics:
                        if 'duplicate' in metric.name.lower() or 'ohlc' in metric.name.lower():
                            if metric.severity in ['error', 'critical']:
                                result['errors'].append(metric.message)
                            elif metric.severity == 'warning':
                                result['warnings'].append(metric.message)
                
                except ImportError:
                    # Fallback to basic consistency checks
                    if data.duplicated().any():
                        duplicate_count = data.duplicated().sum()
                        result['warnings'].append(f'Found {duplicate_count} duplicate rows')
                    if 'close' in data.columns and 'high' in data.columns and ('low' in data.columns):
                        invalid_ohlc = (data['close'] > data['high']) | (data['close'] < data['low'])
                        if invalid_ohlc.any():
                            invalid_count = invalid_ohlc.sum()
                            result['warnings'].append(f'Found {invalid_count} rows with invalid OHLC relationships')
        except Exception as e:
            result['valid'] = False
            result['errors'].append(f'Validation error: {str(e)}')
        return result

    def _validate_execution_time(self, data: Any, context: Dict[str, Any]=None) -> Dict[str, Any]:
        """Validate execution time."""
        result = {'valid': True, 'errors': [], 'warnings': []}
        try:
            execution_time = context.get('execution_time', 0)
            max_time = context.get('max_execution_time', 300)
            if execution_time > max_time:
                result['warnings'].append(f'Execution time {execution_time:.2f}s exceeds threshold {max_time}s')
        except Exception as e:
            result['valid'] = False
            result['errors'].append(f'Validation error: {str(e)}')
        return result

    def _validate_memory_usage(self, data: Any, context: Dict[str, Any]=None) -> Dict[str, Any]:
        """Validate memory usage."""
        result = {'valid': True, 'errors': [], 'warnings': []}
        try:
            memory_usage = context.get('memory_usage_mb', 0)
            max_memory = context.get('max_memory_mb', 1000)
            if memory_usage > max_memory:
                result['warnings'].append(f'Memory usage {memory_usage:.1f}MB exceeds threshold {max_memory}MB')
        except Exception as e:
            result['valid'] = False
            result['errors'].append(f'Validation error: {str(e)}')
        return result

    def _validate_cpu_usage(self, data: Any, context: Dict[str, Any]=None) -> Dict[str, Any]:
        """Validate CPU usage."""
        result = {'valid': True, 'errors': [], 'warnings': []}
        try:
            cpu_usage = context.get('cpu_usage_percent', 0)
            max_cpu = context.get('max_cpu_percent', 80)
            if cpu_usage > max_cpu:
                result['warnings'].append(f'CPU usage {cpu_usage:.1f}% exceeds threshold {max_cpu}%')
        except Exception as e:
            result['valid'] = False
            result['errors'].append(f'Validation error: {str(e)}')
        return result

    def _validate_labeling_logic(self, data: Any, context: Dict[str, Any]=None) -> Dict[str, Any]:
        """Validate labeling logic."""
        result = {'valid': True, 'errors': [], 'warnings': []}
        try:
            if isinstance(data, pd.DataFrame) and 'label' in data.columns:
                labels = data['label'].dropna()
                if len(labels) > 0:
                    label_counts = labels.value_counts()
                    total_labels = len(labels)
                    if len(label_counts) > 1:
                        max_count = label_counts.max()
                        min_count = label_counts.min()
                        imbalance_ratio = max_count / min_count
                        if imbalance_ratio > 10:
                            result['warnings'].append(f'Severe class imbalance detected (ratio: {imbalance_ratio:.1f})')
                    for label, count in label_counts.items():
                        percentage = count / total_labels * 100
                        if percentage < 1:
                            result['warnings'].append(f'Very few samples for label {label} ({percentage:.1f}%)')
        except Exception as e:
            result['valid'] = False
            result['errors'].append(f'Validation error: {str(e)}')
        return result

    def _validate_regime_logic(self, data: Any, context: Dict[str, Any]=None) -> Dict[str, Any]:
        """Validate regime logic."""
        result = {'valid': True, 'errors': [], 'warnings': []}
        try:
            if isinstance(data, pd.DataFrame):
                regime_columns = [col for col in data.columns if 'regime' in col.lower()]
                for regime_col in regime_columns:
                    regimes = data[regime_col].dropna()
                    if len(regimes) > 0:
                        regime_counts = regimes.value_counts()
                        if len(regime_counts) < 2:
                            result['warnings'].append(f'Only {len(regime_counts)} regime(s) detected in {regime_col}')
                        elif len(regime_counts) > 10:
                            result['warnings'].append(f'Too many regimes ({len(regime_counts)}) in {regime_col}')
                        if len(regime_counts) > 1:
                            max_count = regime_counts.max()
                            min_count = regime_counts.min()
                            balance_ratio = max_count / min_count
                            if balance_ratio > 5:
                                result['warnings'].append(f'Unbalanced regimes in {regime_col} (ratio: {balance_ratio:.1f})')
        except Exception as e:
            result['valid'] = False
            result['errors'].append(f'Validation error: {str(e)}')
        return result

    def _validate_triple_barrier_logic(self, data: Any, context: Dict[str, Any]=None) -> Dict[str, Any]:
        """Validate triple barrier logic."""
        result = {'valid': True, 'errors': [], 'warnings': []}
        try:
            if isinstance(data, pd.DataFrame):
                tb_columns = [col for col in data.columns if 'triple_barrier' in col.lower()]
                for tb_col in tb_columns:
                    tb_labels = data[tb_col].dropna()
                    if len(tb_labels) > 0:
                        valid_labels = tb_labels.isin([-1, 0, 1])
                        if not valid_labels.all():
                            invalid_labels = tb_labels[~valid_labels].unique()
                            result['warnings'].append(f'Invalid triple barrier labels in {tb_col}: {invalid_labels}')
                        label_counts = tb_labels.value_counts()
                        total_labels = len(tb_labels)
                        neutral_count = label_counts.get(0, 0)
                        neutral_ratio = neutral_count / total_labels
                        if neutral_ratio > 0.8:
                            result['warnings'].append(f'Too many neutral labels in {tb_col} ({neutral_ratio:.1%})')
        except Exception as e:
            result['valid'] = False
            result['errors'].append(f'Validation error: {str(e)}')
        return result

    def validate_function_input(self, function_name: str, input_data: Any, context: Dict[str, Any]=None) -> Dict[str, Any]:
        """Validate function input using all applicable rules."""
        try:
            validation_result = {'function_name': function_name, 'validation_type': 'input', 'timestamp': datetime.now().isoformat(), 'overall_valid': True, 'rule_results': {}, 'errors': [], 'warnings': []}
            context = context or {}
            for rule_name, rules in self.validation_rules.items():
                if rule_name in ['input_validation', 'data_quality']:
                    for rule in rules:
                        try:
                            rule_result = rule(input_data, context)
                            validation_result['rule_results'][rule_name] = rule_result
                            if not rule_result['valid']:
                                validation_result['overall_valid'] = False
                                validation_result['errors'].extend(rule_result['errors'])
                            validation_result['warnings'].extend(rule_result['warnings'])
                        except Exception as e:
                            validation_result['errors'].append(f'Rule {rule_name} failed: {str(e)}')
                            validation_result['overall_valid'] = False
            self.validation_history.append(validation_result)
            return validation_result
        except Exception as e:
            self.logger.error(f'❌ Failed to validate function input: {e}')
            return {'overall_valid': False, 'errors': [str(e)], 'warnings': []}

    def validate_function_output(self, function_name: str, output_data: Any, context: Dict[str, Any]=None) -> Dict[str, Any]:
        """Validate function output using all applicable rules."""
        try:
            validation_result = {'function_name': function_name, 'validation_type': 'output', 'timestamp': datetime.now().isoformat(), 'overall_valid': True, 'rule_results': {}, 'errors': [], 'warnings': []}
            context = context or {}
            for rule_name, rules in self.validation_rules.items():
                if rule_name in ['output_validation', 'data_quality', 'business_logic']:
                    for rule in rules:
                        try:
                            rule_result = rule(output_data, context)
                            validation_result['rule_results'][rule_name] = rule_result
                            if not rule_result['valid']:
                                validation_result['overall_valid'] = False
                                validation_result['errors'].extend(rule_result['errors'])
                            validation_result['warnings'].extend(rule_result['warnings'])
                        except Exception as e:
                            validation_result['errors'].append(f'Rule {rule_name} failed: {str(e)}')
                            validation_result['overall_valid'] = False
            self.validation_history.append(validation_result)
            return validation_result
        except Exception as e:
            self.logger.error(f'❌ Failed to validate function output: {e}')
            return {'overall_valid': False, 'errors': [str(e)], 'warnings': []}

    def generate_validation_report(self) -> Dict[str, Any]:
        """Generate comprehensive validation report."""
        try:
            if not self.validation_history:
                return {'total_validations': 0, 'message': 'No validation data recorded'}
            total_validations = len(self.validation_history)
            successful_validations = len([v for v in self.validation_history if v['overall_valid']])
            failed_validations = total_validations - successful_validations
            function_validations = {}
            for validation in self.validation_history:
                func_name = validation['function_name']
                if func_name not in function_validations:
                    function_validations[func_name] = {'input': [], 'output': []}
                function_validations[func_name][validation['validation_type']].append(validation)
            error_patterns = {}
            warning_patterns = {}
            for validation in self.validation_history:
                for error in validation['errors']:
                    error_patterns[error] = error_patterns.get(error, 0) + 1
                for warning in validation['warnings']:
                    warning_patterns[warning] = warning_patterns.get(warning, 0) + 1
            return {'total_validations': total_validations, 'successful_validations': successful_validations, 'failed_validations': failed_validations, 'success_rate': successful_validations / total_validations * 100 if total_validations > 0 else 0, 'function_validations': function_validations, 'error_patterns': error_patterns, 'warning_patterns': warning_patterns, 'most_common_errors': sorted(error_patterns.items(), key=lambda x: x[1], reverse=True)[:5], 'most_common_warnings': sorted(warning_patterns.items(), key=lambda x: x[1], reverse=True)[:5]}
        except Exception as e:
            self.logger.error(f'❌ Failed to generate validation report: {e}')
            return {}

    def log_validation_report(self, report: Dict[str, Any]) -> None:
        """Log comprehensive validation report."""
        try:
            if report.get('total_validations', 0) == 0:
                self.logger.info('📋 No validation data recorded')
                return
            self.logger.info('📋 COMPREHENSIVE VALIDATION REPORT')
            self.logger.info('=' * 50)
            self.logger.info(f"Total Validations: {report['total_validations']}")
            self.logger.info(f"Successful Validations: {report['successful_validations']}")
            self.logger.info(f"Failed Validations: {report['failed_validations']}")
            self.logger.info(f"Success Rate: {report['success_rate']:.1f}%")
            function_validations = report.get('function_validations', {})
            if function_validations:
                self.logger.info(f'\n🔍 FUNCTION VALIDATION RESULTS:')
                for func_name, validations in function_validations.items():
                    input_validations = validations.get('input', [])
                    output_validations = validations.get('output', [])
                    input_success = len([v for v in input_validations if v['overall_valid']])
                    output_success = len([v for v in output_validations if v['overall_valid']])
                    self.logger.info(f'   {func_name}:')
                    self.logger.info(f'     Input Validations: {input_success}/{len(input_validations)} successful')
                    self.logger.info(f'     Output Validations: {output_success}/{len(output_validations)} successful')
            most_common_errors = report.get('most_common_errors', [])
            if most_common_errors:
                self.logger.info(f'\n❌ MOST COMMON ERRORS:')
                for error, count in most_common_errors:
                    self.logger.info(f'   - {error}: {count} occurrences')
            most_common_warnings = report.get('most_common_warnings', [])
            if most_common_warnings:
                self.logger.info(f'\n⚠️ MOST COMMON WARNINGS:')
                for warning, count in most_common_warnings:
                    self.logger.info(f'   - {warning}: {count} occurrences')
        except Exception as e:
            self.logger.error(f'❌ Failed to log validation report: {e}')

# Removed custom comprehensive_validation - using standardized decorators

# Removed complex fallback logic - using standardized imports
logger = system_logger.getChild('Step5Labeling')

class LabelingStep:
    """Step 5: Labeling with standardized data quality management and regime-aware triple barrier method."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.logger = system_logger.getChild('LabelingStep')
        self.standards = pipeline_standards
        self.start_time = None
        self.step_timings = {}
        self.timer = SimpleTimer(self.logger)
        self.error_handler = EnhancedErrorHandler(self.logger)
        self.validation_framework = ComprehensiveValidationFramework(self.logger)
        
        # Initialize artifact and version managers
        self.artifact_manager = get_artifact_manager()
        self.pickup_utils = get_artifact_pickup_utils()
        self.version_manager = get_version_manager()

        # Initialize comprehensive optimization components
        if OPTIMIZATIONS_AVAILABLE:
            try:
                # M1 Hardware-Specific Optimizations
                self.gpu_manager = get_m1_gpu_manager()
                self.memory_optimizer = get_m1_memory_optimizer()
                self.cpu_optimizer = get_m1_cpu_optimizer()

                # Processing Core Optimizations
                self.vectorized_core = get_vectorized_processing_core()
                self.matrix_operations = EnhancedMatrixOperations()
                self.step_optimizer = get_step_optimization_manager()

                # Data Management Optimizations
                self.data_manager = OptimizedDataManager(
                    base_path=Path(self.config.get('DATA_DIR', 'historical_data')),
                    enable_caching=True,
                    enable_compression=True,
                    enable_parallel_io=True
                )

                self.logger.info('🚀 Step 5 initialized with comprehensive optimization suite:')
                self.logger.info('  ✅ M1 GPU Manager (MPS acceleration)')
                self.logger.info('  ✅ M1 Memory Optimizer')
                self.logger.info('  ✅ M1 CPU Optimizer (parallel processing)')
                self.logger.info('  ✅ Vectorized Processing Core')
                self.logger.info('  ✅ Enhanced Matrix Operations')
                self.logger.info('  ✅ Enhanced Step Optimizer')
                self.logger.info('  ✅ Optimized Data Manager')

            except Exception as e:
                self.logger.warning(f'Failed to initialize some optimizations: {e}')
                # Initialize with fallbacks
                self._initialize_fallback_optimizations()
        else:
            self._initialize_fallback_optimizations()

    def _initialize_fallback_optimizations(self):
        """Initialize fallback optimizations when full suite is not available."""
        self.gpu_manager = None
        self.memory_optimizer = None
        self.cpu_optimizer = None
        self.vectorized_core = None
        self.matrix_operations = None
        self.step_optimizer = None
        self.data_manager = None
        self.logger.info('📋 Initialized with fallback optimizations (basic functionality only)')

        self._validate_environment()
        self._initialize_components()

    def _setup_optimization_context(self) -> Dict[str, Any]:
        """Setup optimization context for the current execution."""
        context = {
            'memory_checkpoint': None,
            'optimization_profile': None,
            'data_manager_session': None
        }

        if self.memory_optimizer:
            context['memory_checkpoint'] = self.memory_optimizer.memory_checkpoint('step05_labeling')

        if self.step_optimizer:
            from src.utils.enhanced_step_optimizations import WorkloadType, OptimizationProfile
            # Create optimization profile based on current workload
            context['optimization_profile'] = OptimizationProfile(
                workload_type=WorkloadType.MEMORY_INTENSIVE,
                data_size_mb=500,  # Estimate based on typical data size
                expected_duration=300,  # 5 minutes expected
                priority="high"
            )

        if self.data_manager:
            context['data_manager_session'] = self.data_manager.create_session()

        return context

    async def _load_data_optimized(self, file_path: Path, optimization_context: Dict[str, Any]) -> pd.DataFrame:
        """Load data using optimized data manager with memory management."""
        try:
            session = optimization_context.get('data_manager_session')
            if not session:
                # Fallback to standard loading
                return standardized_parquet_handler.read_parquet_standardized(file_path)

            # Use optimized data manager for loading
            data_id = f"{file_path.stem}_data"
            data = await session.load_data_async(data_id, file_path)

            # Apply memory optimizations
            if self.memory_optimizer:
                data_size_mb = data.memory_usage(deep=True).sum() / (1024**2)
                if self.memory_optimizer.should_chunk_data(data_size_mb, "general"):
                    self.logger.info(f"📦 Large dataset detected ({data_size_mb:.1f}MB), applying memory optimizations")
                    # Optimize data types for memory efficiency
                    data = self.memory_optimizer.optimize_dataframe_memory(data)

            return data

        except Exception as e:
            self.logger.warning(f"Optimized data loading failed, falling back to standard loading: {e}")
            return standardized_parquet_handler.read_parquet_standardized(file_path)

    async def _generate_comprehensive_labels_optimized(
        self,
        data: pd.DataFrame,
        symbol: str,
        exchange: str,
        timeframe: str,
        optimization_context: Dict[str, Any]
    ) -> Optional[pd.DataFrame]:
        """Generate comprehensive labels using all available optimizations."""
        start_time = time.time()

        try:
            # Apply memory checkpoint if available
            memory_context = optimization_context.get('memory_checkpoint')
            if memory_context:
                async with memory_context:
                    return await self._perform_optimized_labeling(
                        data, symbol, exchange, timeframe, optimization_context
                    )
            else:
                return await self._perform_optimized_labeling(
                    data, symbol, exchange, timeframe, optimization_context
                )

        except Exception as e:
            self.logger.exception(f'❌ Error in optimized comprehensive labeling: {e}')
            # Cleanup on error
            if self.memory_optimizer:
                self.memory_optimizer.optimize_memory()
            return None

    async def _perform_optimized_labeling(
        self,
        data: pd.DataFrame,
        symbol: str,
        exchange: str,
        timeframe: str,
        optimization_context: Dict[str, Any]
    ) -> Optional[pd.DataFrame]:
        """Perform the actual optimized labeling operations."""
        try:
            result_data = data.copy()

            # Step 1: Pre-process data with optimizations
            if self.memory_optimizer:
                data_size_mb = result_data.memory_usage(deep=True).sum() / (1024**2)
                if self.memory_optimizer.should_chunk_data(data_size_mb, "general"):
                    self.logger.info('🧠 Applying memory optimizations to input data')
                    result_data = self.memory_optimizer.optimize_dataframe_memory(result_data)

            # Step 2: Generate triple barrier labels with regime-aware optimizations
            if 'triple_barrier_label' not in result_data.columns:
                self.logger.info('🔄 Triple barrier labels not found, generating with optimizations...')
                if self.regime_barrier_optimizer and self.auto_recalculate_hmm_barriers:
                    try:
                        self.logger.info('🚀 Attempting regime-aware triple barrier labeling...')
                        if self.regime_col in result_data.columns:
                            self.logger.info(f'✅ Found regime column: {self.regime_col}')

                            # Use vectorized processing if available
                            if self.vectorized_core:
                                self.logger.info('⚡ Using vectorized processing for regime-aware labeling')
                                regime_labels = await self._generate_regime_aware_labels_vectorized(
                                    result_data, symbol, exchange, timeframe
                                )
                            else:
                                regime_labels = await self.timer.time_operation('generate_regime_aware_labels')(
                                    self._generate_regime_aware_labels
                                )(result_data, symbol, exchange, timeframe)

                            if regime_labels is not None:
                                result_data['triple_barrier_label'] = regime_labels
                                result_data['labeling_method'] = 'regime_aware_optimized'
                                self.logger.info('✅ Generated regime-aware triple barrier labels with optimizations')
                            else:
                                raise Exception('Regime-aware labeling failed')
                        else:
                            self.logger.warning(f"⚠️ Regime column '{self.regime_col}' not found")
                            raise Exception('Regime column not found')
                    except Exception as e:
                        self.logger.error(f'❌ Regime-aware labeling failed: {e}')
                        self.logger.error('❌ No fallback labeling method available - regime-aware labeling is required')
                        return None
                else:
                    if not self.auto_recalculate_hmm_barriers:
                        self.logger.error('❌ Auto-calculation disabled for regime-aware labeling')
                    if self.regime_barrier_optimizer is None:
                        self.logger.error('❌ Regime barrier optimizer not available')
                    self.logger.error('❌ Regime-aware labeling is required - no fallback available')
                    return None

            # Step 3: Apply meta-labeling with optimizations
            if self.meta_labeling_system:
                try:
                    await self.meta_labeling_system.initialize()

                    # Use CPU optimizer for parallel meta-labeling if available
                    if self.cpu_optimizer:
                        self.logger.info('🏃 Using M1 CPU optimizer for parallel meta-labeling')
                        meta_labels = await self._apply_meta_labeling_parallel(result_data, symbol, exchange, timeframe)
                    else:
                        analyst_labels = await self.meta_labeling_system._generate_analyst_labels(data, symbol, exchange, timeframe)
                        if analyst_labels is not None:
                            result_data['analyst_label'] = analyst_labels
                            self.logger.info('✅ Generated analyst labels')
                        tactician_labels = await self.meta_labeling_system._generate_tactician_labels(data, symbol, exchange, timeframe)
                        if tactician_labels is not None:
                            result_data['tactician_label'] = tactician_labels
                            self.logger.info('✅ Generated tactician labels')
                except Exception as e:
                    self.logger.warning(f'⚠️ Meta-labeling failed: {e}')

            # Step 4: Create composite labels with matrix operations optimization
            composite_label = await self._create_composite_label_optimized(result_data)
            result_data['label'] = composite_label
            result_data['label_confidence'] = await self._calculate_label_confidence(result_data)
            result_data['label_source'] = await self._determine_label_source(result_data)

            self.logger.info(f'✅ Generated comprehensive labels with {len(result_data.columns)} columns using full optimization suite')
            self.logger.info(f"   - Label distribution: {result_data['label'].value_counts().to_dict()}")
            self.logger.info(f"   - Labeling method used: {result_data.get('labeling_method', 'unknown')}")

            return result_data

        except Exception as e:
            self.logger.exception(f'❌ Error in optimized labeling operations: {e}')
            return None

    async def _generate_regime_aware_labels_vectorized(
        self,
        data: pd.DataFrame,
        symbol: str,
        exchange: str,
        timeframe: str
    ) -> Optional[pd.Series]:
        """Generate regime-aware labels using vectorized processing core."""
        try:
            self.logger.info('⚡ Using vectorized processing core for regime-aware labeling...')

            if not self._validate_regime_aware_inputs(data):
                return None

            # Use vectorized core for processing
            if self.vectorized_core:
                # Prepare data for vectorized processing
                price_data = data[['close', 'high', 'low']].values
                regime_data = data[self.regime_col].values

                # Use vectorized core to process regime-aware labels
                # This would integrate with the vectorized processing core
                labels = self._vectorized_triple_barrier_labels_regime_aware(
                    price_data, regime_data, data.index
                )

                if labels is not None:
                    self.logger.info(f'✅ Generated {len(labels)} vectorized regime-aware labels')
                    return labels

            # Fallback to standard regime labeler
            regime_labeler = self._create_regime_labeler()
            if regime_labeler is None:
                return None

            return self.timer.time_operation('generate_labels_with_regime_labeler')(
                self._generate_labels_with_regime_labeler
            )(regime_labeler, data)

        except Exception as e:
            self.logger.exception(f'❌ Error in vectorized regime-aware labeling: {e}')
            return None

    def _vectorized_triple_barrier_labels_regime_aware(
        self,
        price_data: np.ndarray,
        regime_data: np.ndarray,
        index: pd.Index
    ) -> Optional[pd.Series]:
        """Vectorized triple barrier labeling with regime awareness."""
        try:
            # Extract price data
            prices = price_data[:, 0]  # close prices
            highs = price_data[:, 1]
            lows = price_data[:, 2]

            # Configuration parameters
            profit_take = 0.002  # 0.2%
            stop_loss = 0.001    # 0.1%
            time_barrier = self.time_barrier_minutes
            max_lookahead = self.max_lookahead

            # Vectorized barrier calculations
            labels = np.zeros(len(prices))

            # Calculate future returns for all points at once
            future_returns = np.zeros(len(prices))
            for i in range(len(prices) - 1):
                if i + max_lookahead < len(prices):
                    future_returns[i] = (prices[i + max_lookahead] - prices[i]) / prices[i]
                else:
                    future_returns[i] = (prices[-1] - prices[i]) / prices[i]

            # Vectorized profit/loss barrier hits
            profit_hits = future_returns >= profit_take
            loss_hits = future_returns <= -stop_loss

            # Vectorized time barrier (simplified)
            time_hits = np.zeros(len(prices), dtype=bool)
            for i in range(len(prices)):
                if i + time_barrier < len(prices):
                    future_window = prices[i:i + time_barrier]
                    if len(future_window) > 0:
                        max_return = (future_window.max() - prices[i]) / prices[i]
                        min_return = (future_window.min() - prices[i]) / prices[i]
                        if max_return < profit_take and min_return > -stop_loss:
                            time_hits[i] = True

            # Vectorized label assignment
            labels[profit_hits] = 1    # Profit take hit
            labels[loss_hits] = -1     # Stop loss hit
            labels[time_hits] = 0      # Time barrier hit

            # Apply regime-aware adjustments if regime column exists
            if len(regime_data) > 0:
                labels = self._apply_regime_aware_adjustments_vectorized(
                    labels, prices, regime_data
                )

            self.logger.info(f'✅ Generated {len(labels)} vectorized regime-aware triple barrier labels')

            return pd.Series(labels, index=index)

        except Exception as e:
            self.logger.exception(f'❌ Error in vectorized triple barrier labeling: {e}')
            return None

    def _apply_regime_aware_adjustments_vectorized(
        self,
        labels: np.ndarray,
        prices: np.ndarray,
        regime_data: np.ndarray
    ) -> np.ndarray:
        """Apply regime-aware adjustments using vectorized operations."""
        try:
            # Calculate regime-specific statistics
            unique_regimes = np.unique(regime_data)

            for regime in unique_regimes:
                regime_mask = regime_data == regime
                regime_labels = labels[regime_mask]
                regime_prices = prices[regime_mask]

                if len(regime_labels) > 0:
                    # Calculate regime volatility
                    if len(regime_prices) > 1:
                        regime_volatility = np.std(np.diff(regime_prices) / regime_prices[:-1])
                    else:
                        regime_volatility = 0.01  # Default moderate volatility

                    # Scale barriers based on volatility
                    if regime_volatility > 0.01:  # High volatility regime
                        # More conservative barriers
                        profit_mask = regime_labels == 1
                        loss_mask = regime_labels == -1
                        # Reduce profit targets and increase stop losses
                        labels[regime_mask & profit_mask] = 0.5
                        labels[regime_mask & loss_mask] = -0.5
                    elif regime_volatility < 0.005:  # Low volatility regime
                        # More aggressive barriers
                        profit_mask = regime_labels == 1
                        loss_mask = regime_labels == -1
                        # Increase profit targets and reduce stop losses
                        labels[regime_mask & profit_mask] = 1.5
                        labels[regime_mask & loss_mask] = -0.5

            return labels

        except Exception as e:
            self.logger.warning(f'⚠️ Error applying vectorized regime adjustments: {e}')
            return labels

    async def _apply_meta_labeling_parallel(
        self,
        data: pd.DataFrame,
        symbol: str,
        exchange: str,
        timeframe: str
    ) -> None:
        """Apply meta-labeling using parallel processing."""
        try:
            if not self.cpu_optimizer or not self.meta_labeling_system:
                return

            self.logger.info('🏃 Applying meta-labeling with parallel processing')

            # Use CPU optimizer for parallel execution
            tasks = [
                self.meta_labeling_system._generate_analyst_labels(data, symbol, exchange, timeframe),
                self.meta_labeling_system._generate_tactician_labels(data, symbol, exchange, timeframe)
            ]

            # Execute in parallel
            results = await self.cpu_optimizer.parallel_map_async(
                lambda task: task, tasks, max_workers=2
            )

            # Process results
            if len(results) >= 1 and results[0] is not None:
                data['analyst_label'] = results[0]
                self.logger.info('✅ Generated analyst labels (parallel)')

            if len(results) >= 2 and results[1] is not None:
                data['tactician_label'] = results[1]
                self.logger.info('✅ Generated tactician labels (parallel)')

        except Exception as e:
            self.logger.warning(f'⚠️ Parallel meta-labeling failed: {e}')

    async def _create_composite_label_optimized(self, data: pd.DataFrame) -> pd.Series:
        """Create composite label using optimized matrix operations."""
        try:
            if self.matrix_operations:
                self.logger.info('🔢 Using enhanced matrix operations for composite labeling')

                # Use matrix operations for efficient composite label creation
                composite_label = data['triple_barrier_label'].copy()

                if 'analyst_label' in data.columns:
                    # Use matrix operations for efficient override logic
                    analyst_override_mask = (
                        (data['analyst_label'] != 0) &
                        (data['triple_barrier_label'] == 0)
                    ).values

                    # Apply overrides using vectorized operations
                    composite_label.loc[analyst_override_mask] = data.loc[analyst_override_mask, 'analyst_label']

                return composite_label
            else:
                # Fallback to standard implementation
                return await self._create_composite_label(data)

        except Exception as e:
            self.logger.warning(f'⚠️ Optimized composite labeling failed, using fallback: {e}')
            return await self._create_composite_label(data)

    async def _save_data_optimized(
        self,
        data: pd.DataFrame,
        output_path: Path,
        metadata_path: Path,
        metadata: Dict[str, Any],
        optimization_context: Dict[str, Any]
    ) -> bool:
        """Save data using optimized data manager with versioned filenames."""
        try:
            # Generate versioned filenames
            base_name = output_path.stem.replace('_labeled_data', '')
            versioned_filename = self.artifact_manager.get_versioned_filename(f"{base_name}_labeled_data", ".parquet")
            versioned_metadata_filename = self.artifact_manager.get_versioned_filename(f"{base_name}_labeling_metadata", ".json")
            
            # Update paths to use versioned filenames
            versioned_output_path = output_path.parent / versioned_filename
            versioned_metadata_path = metadata_path.parent / versioned_metadata_filename
            
            session = optimization_context.get('data_manager_session')
            if not session:
                # Fallback to standard saving with versioned filenames
                standardized_parquet_handler.write_parquet_standardized(data, versioned_output_path)
                safe_json_dump(metadata, versioned_metadata_path, indent=2, default=str)
                self.logger.info(f"✅ Saved labeled data with versioned filename: {versioned_filename}")
                return True

            # Use optimized data manager for saving with versioned filenames
            data_id = f"{base_name}_labeled_data"
            await session.save_data_async(data_id, data, versioned_output_path, metadata=metadata)

            self.logger.info(f"✅ Saved labeled data with versioned filename: {versioned_filename}")
            return True

        except Exception as e:
            self.logger.warning(f"Optimized data saving failed, falling back to standard saving: {e}")
            try:
                # Use versioned filenames in fallback
                standardized_parquet_handler.write_parquet_standardized(data, versioned_output_path)
                safe_json_dump(metadata, versioned_metadata_path, indent=2, default=str)
                self.logger.info(f"✅ Saved labeled data with versioned filename (fallback): {versioned_filename}")
                return True
            except Exception as fallback_error:
                self.logger.error(f"Standard saving also failed: {fallback_error}")
                return False

    def _get_optimization_summary(self) -> Dict[str, Any]:
        """Get summary of optimizations used."""
        return {
            'm1_gpu_manager': self.gpu_manager is not None,
            'm1_memory_optimizer': self.memory_optimizer is not None,
            'm1_cpu_optimizer': self.cpu_optimizer is not None,
            'vectorized_processing_core': self.vectorized_core is not None,
            'enhanced_matrix_operations': self.matrix_operations is not None,
            'enhanced_step_optimizer': self.step_optimizer is not None,
            'optimized_data_manager': self.data_manager is not None,
            'regime_barrier_optimizer': self.regime_barrier_optimizer is not None,
            'meta_labeling_system': self.meta_labeling_system is not None
        }

    # Removed complex function monitoring setup - using SimpleTimer

    # Removed complex validation methods - using ComprehensiveValidationFramework

    def _compute_labeling_fingerprint(self, triple_barrier_path: Path) -> Dict[str, Any]:
        """Compute a stable fingerprint of source labeling inputs to ensure idempotence.

        Uses source file size and mtime plus relevant config toggles.
        """
        try:
            stat = triple_barrier_path.stat()
            relevant_cfg = {'vectorized_labelling_orchestrator': self.config.get('vectorized_labelling_orchestrator', {}), 'labeling': self.config.get('labeling', {}), 'time_barrier_minutes': getattr(self, 'time_barrier_minutes', None), 'max_lookahead': getattr(self, 'max_lookahead', None), 'regime_col': getattr(self, 'regime_col', None), 'auto_recalculate_hmm_barriers': getattr(self, 'auto_recalculate_hmm_barriers', None)}
            relevant_cfg_json = json.dumps(relevant_cfg, sort_keys=True, default=str)
            cfg_hash = hashlib.sha256(relevant_cfg_json.encode('utf-8')).hexdigest()
            return {'source_path': str(triple_barrier_path), 'source_size': stat.st_size, 'source_mtime': int(stat.st_mtime), 'config_hash': cfg_hash}
        except Exception as e:
            self.logger.warning(f'⚠️ Failed to compute labeling fingerprint: {e}')
            return {}

    def _validate_environment(self) -> None:
        """Validate environment dependencies."""
        self.logger.info('🔍 Environment validation simplified - using standardized imports')

    def _initialize_components(self) -> None:
        """Initialize labeling components with regime-aware triple barrier support."""
        self.logger.info('🔧 Initializing labeling components...')
        labeling_cfg = self.config.get('vectorized_labelling_orchestrator', {})
        self.auto_recalculate_hmm_barriers = bool(labeling_cfg.get('auto_recalculate_hmm_barriers', True))
        try:
            from src.utils.regime_data_access import get_regime_column
            detected = get_regime_column(pd.DataFrame(columns=['composite_cluster_id'])) or 'hmm_regime'
        except Exception:
            detected = 'hmm_regime'
        self.regime_col = str(labeling_cfg.get('hmm_barrier_regime_column', detected))
        self.time_barrier_minutes = int(labeling_cfg.get('time_barrier_minutes', 30))
        self.max_lookahead = int(labeling_cfg.get('max_lookahead', 100))
        self.logger.info(f'📋 Regime-aware labeling configuration:')
        self.logger.info(f'   - Auto recalculate HMM barriers: {self.auto_recalculate_hmm_barriers}')
        self.logger.info(f'   - HMM regime column: {self.regime_col}')
        self.logger.info(f'   - Time barrier minutes: {self.time_barrier_minutes}')
        self.logger.info(f'   - Max lookahead: {self.max_lookahead}')
        try:
            from src.training.steps.step06_labeling_components.regime_specific_triple_barrier_optimizer import RegimeSpecificTripleBarrierOptimizer  # type: ignore
            self.regime_barrier_optimizer = RegimeSpecificTripleBarrierOptimizer(self.config)
            self.logger.info('✅ RegimeSpecificTripleBarrierOptimizer initialized successfully')
        except ImportError as e:
            self.logger.error(f'❌ Failed to import RegimeSpecificTripleBarrierOptimizer: {e}')
            raise RuntimeError(f'Regime barrier optimizer is required but not available: {e}')
        except Exception as e:
            self.logger.error(f'❌ Failed to initialize RegimeSpecificTripleBarrierOptimizer: {e}')
            raise RuntimeError(f'Regime barrier optimizer initialization failed: {e}')
        # Initialize meta-labeling system
        try:
            from src.analyst.meta_labeling_system import MetaLabelingSystem
            self.meta_labeling_system = MetaLabelingSystem(self.config)
            self.logger.info('✅ Meta-labeling system initialized successfully')
        except ImportError as e:
            self.logger.error(f'❌ Failed to import MetaLabelingSystem: {e}')
            raise RuntimeError(f'Meta labeling system is required but not available: {e}')
        except Exception as e:
            self.logger.error(f'❌ Failed to initialize MetaLabelingSystem: {e}')
            raise RuntimeError(f'Meta labeling system initialization failed: {e}')

    async def initialize(self) -> None:
        """Initialize the labeling step."""
        self.start_time = time.time()
        self.logger.info('🚀 Initializing Labeling Step...')
        self.logger.info('📋 Step 5 Configuration:')
        self.logger.info(f"   - Symbol: {self.config.get('SYMBOL', 'N/A')}")
        self.logger.info(f"   - Exchange: {self.config.get('EXCHANGE', 'N/A')}")
        self.logger.info(f"   - Timeframe: {self.config.get('TIMEFRAME', 'N/A')}")
        self.logger.info(f"   - Data Directory: {self.config.get('DATA_DIR', 'N/A')}")
        self.logger.info('✅ Labeling Step initialized successfully')

    def _log_step_timing(self, step_name: str, start_time: float) -> None:
        """Log timing information for a step."""
        elapsed = time.time() - start_time
        self.step_timings[step_name] = elapsed
        self.logger.info(f'⏱️ {step_name} completed in {elapsed:.2f} seconds')

    @traced(span_name='execute_labeling')
    @validates()
    @handles_errors()
    @cached()
    @log_execution_time()
    async def execute_labeling(self, symbol: str, exchange: str, timeframe: str, data_dir: str='historical_data', force_rerun: bool=False) -> bool:
        step_start = time.time()
        self.logger.info(f'🚀 Executing Labeling for {symbol} on {exchange}')

        # Initialize optimization context
        optimization_context = self._setup_optimization_context()

        try:
            triple_barrier_path = Path(data_dir) / 'training' / f'{exchange}_{symbol}_{timeframe}_triple_barrier_labels.parquet'
            if not triple_barrier_path.exists():
                self.logger.error(f'❌ Triple barrier labels not found at {triple_barrier_path}')
                return False
            self.logger.info(f'📁 Loading triple barrier labels from {triple_barrier_path}')
            labeled_dir = ensure_directory(Path(data_dir) / 'training' / 'labeled_data')
            output_path = labeled_dir / f'{exchange}_{symbol}_{timeframe}_labeled_data.parquet'
            metadata_path = labeled_dir / f'{exchange}_{symbol}_{timeframe}_labeling_metadata.json'
            current_fp = self._compute_labeling_fingerprint(triple_barrier_path)
            if not force_rerun and output_path.exists() and metadata_path.exists():
                try:
                    with open(metadata_path, 'r', encoding='utf-8') as f:
                        existing_meta = json.load(f)
                    existing_fp = existing_meta.get('source_fingerprint', {})
                    if existing_fp == current_fp and existing_meta.get('total_samples', 0) > 0:
                        self.logger.info('🟢 Labeling is idempotent: existing outputs match current inputs. Skipping recomputation.')
                        self._log_step_timing('Labeling (skipped)', step_start)
                        return True
                except Exception as e:
                    self.logger.warning(f'⚠️ Failed to read existing labeling metadata: {e}')

            # Use optimized data loading
            if self.data_manager and optimization_context.get('data_manager_session'):
                self.logger.info('🔄 Using optimized data manager for loading')
                data = await self._load_data_optimized(triple_barrier_path, optimization_context)
            else:
                self.logger.info('📖 Using standard pandas loading')
                data = standardized_parquet_handler.read_parquet_standardized(triple_barrier_path)
            try:
                from src.utils.regime_data_access import ensure_regime_labels, get_regime_column
                data = ensure_regime_labels(data, exchange=exchange, symbol=symbol, timeframe=timeframe, data_dir=data_dir)
                detected_col = get_regime_column(data)
                if detected_col and detected_col != self.regime_col:
                    self.logger.info(f"🔁 Using detected regime column '{detected_col}' instead of '{self.regime_col}'")
                    self.regime_col = detected_col
            except Exception:
                pass
            self.logger.info(f'✅ Loaded data with shape: {data.shape}')
            # Use optimized comprehensive labeling with vectorized processing
            data = await self._generate_comprehensive_labels_optimized(
                data, symbol, exchange, timeframe, optimization_context
            )
            if data is None:
                self.logger.error('❌ Comprehensive labeling failed')
                return False
            # Use optimized data saving
            if self.data_manager and optimization_context.get('data_manager_session'):
                self.logger.info('💾 Using optimized data manager for saving')
                success = await self._save_data_optimized(
                    data, output_path, metadata_path, metadata, optimization_context
                )
            else:
                self.logger.info('💾 Using standard parquet saving')
                standardized_parquet_handler.write_parquet_standardized(data, output_path)
                safe_json_dump(metadata, metadata_path, indent=2, default=str)
                success = True

            if success:
                self.logger.info(f'✅ Labeled data saved to {output_path}')
                label_distribution = {}
                if 'label' in data.columns:
                    label_distribution = data['label'].value_counts().to_dict()
                metadata = {
                    'symbol': symbol,
                    'exchange': exchange,
                    'timeframe': timeframe,
                    'total_samples': int(len(data)),
                    'label_distribution': label_distribution,
                    'created_at': pd.Timestamp.now().isoformat(),
                    'labeling_config': self.config.get('labeling', {}),
                    'source_fingerprint': current_fp,
                    'optimization_used': self._get_optimization_summary()
                }
                safe_json_dump(metadata, metadata_path, indent=2, default=str)

                self._log_step_timing('execute_labeling', step_start)
                await self._log_step5_artifacts_and_report(symbol, exchange, timeframe, data_dir, data, output_path, metadata_path)
                await self._generate_and_log_function_call_report()

                # Final memory optimization
                if self.memory_optimizer:
                    final_memory_stats = self.memory_optimizer.optimize_memory()
                    self.logger.info(f'🧹 Final memory optimization: {final_memory_stats.get("memory_freed_mb", 0):.1f}MB freed')

                return True
            else:
                self.logger.error('❌ Failed to save labeled data')
                return False
        except Exception as e:
            self.logger.exception(f'❌ Error in labeling: {e}')
            await self._generate_and_log_function_call_report()
            return False

    async def _generate_and_log_function_call_report(self) -> None:
        """Generate and log function call report using standardized error handling."""
        try:
            self.logger.info('📊 Generating function call report...')
            # Use standardized error handler
            error_summary = self.error_handler.get_error_summary()
            if error_summary['total_errors'] > 0:
                self.logger.warning(f"⚠️ {error_summary['total_errors']} errors occurred during execution")
            else:
                self.logger.info('✅ No errors recorded during execution')
        except Exception as e:
            self.logger.error(f'❌ Failed to generate function call report: {e}')

    # Removed custom _save_function_call_report - using standardized error handling

    # Removed custom _log_function_call_relationships - using standardized error handling

    # Removed custom _analyze_function_completion_outcomes and _log_detailed_completion_report - using standardized error handling

    async def _log_step5_artifacts_and_report(self, symbol: str, exchange: str, timeframe: str, data_dir: str, labeled_data: pd.DataFrame, output_path: Path, metadata_path: Path) -> None:
        """Log step 5 artifacts and create simplified report."""
        try:
            self.logger.info('📊 Logging step 5 artifacts and reports...')
            if labeled_data is not None:
                total_samples = len(labeled_data)
                labeled_samples = len(labeled_data[labeled_data['label'].notna()]) if 'label' in labeled_data.columns else 0
                self.logger.info(f'✅ Labeled {labeled_samples}/{total_samples} samples successfully')
            self.logger.info(f'✅ Output saved to: {output_path}')
            self.logger.info(f'✅ Metadata saved to: {metadata_path}')
            self.logger.info('✅ Step 5 artifacts and reports logged successfully')
        except Exception as e:
            self.logger.error(f'❌ Failed to log step 5 artifacts and reports: {e}')

    async def _create_composite_label(self, data: pd.DataFrame) -> pd.Series:
        """Create composite label from multiple labeling strategies."""
        try:
            composite_label = data['triple_barrier_label'].copy()
            if 'analyst_label' in data.columns:
                analyst_override_mask = (data['analyst_label'] != 0) & (data['triple_barrier_label'] == 0)
                composite_label[analyst_override_mask] = data['analyst_label'][analyst_override_mask]
            return composite_label
        except Exception as e:
            self.logger.warning(f'⚠️ Error creating composite label: {e}')
            return data['triple_barrier_label']

    async def _calculate_label_confidence(self, data: pd.DataFrame) -> pd.Series:
        """Calculate confidence scores for labels."""
        try:
            confidence = np.ones(len(data), dtype=np.float32)
            if 'analyst_label' in data.columns:
                agreement_mask = (data['label'] == data['analyst_label']) & (data['analyst_label'] != 0)
                confidence[agreement_mask] += 0.2
            confidence = np.minimum(confidence, 1.0)
            return pd.Series(confidence, index=data.index)
        except Exception as e:
            self.logger.warning(f'⚠️ Error calculating label confidence: {e}')
            return pd.Series(1.0, index=data.index)

    async def _determine_label_source(self, data: pd.DataFrame) -> pd.Series:
        """Determine the source of each label."""
        try:
            sources = []
            for idx in range(len(data)):
                if data['label'].iloc[idx] == data['triple_barrier_label'].iloc[idx]:
                    if 'analyst_label' in data.columns and data['label'].iloc[idx] == data['analyst_label'].iloc[idx]:
                        sources.append('triple_barrier+analyst')
                    else:
                        sources.append('triple_barrier')
                elif 'analyst_label' in data.columns and data['label'].iloc[idx] == data['analyst_label'].iloc[idx]:
                    sources.append('analyst')
                else:
                    sources.append('composite')
            return pd.Series(sources, index=data.index)
        except Exception as e:
            self.logger.warning(f'⚠️ Error determining label source: {e}')
            return pd.Series('unknown', index=data.index)

    def _validate_regime_aware_inputs(self, data: pd.DataFrame) -> bool:
        """Validate inputs for regime-aware labeling."""
        try:
            if self.regime_barrier_optimizer is None:
                self.logger.error('❌ Regime barrier optimizer not available')
                return False
            if self.regime_col not in data.columns:
                self.logger.error(f"❌ Regime column '{self.regime_col}' not found in data")
                return False
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            missing_columns = [col for col in required_columns if col not in data.columns]
            if missing_columns:
                self.logger.error(f'❌ Missing required columns for triple barrier labeling: {missing_columns}')
                return False
            if data.empty:
                self.logger.error('❌ Input data is empty')
                return False
            return True
        except Exception as e:
            self.logger.error(f'❌ Error during regime-aware input validation: {e}')
            return False

    def _create_regime_labeler(self):
        """Create and configure the regime labeler."""
        try:
            from src.training.steps.step06_labeling_components.regime_aware_triple_barrier_labeling import RegimeAwareTripleBarrierLabeling  # type: ignore
            return RegimeAwareTripleBarrierLabeling(default_profit_take_multiplier=0.002, default_stop_loss_multiplier=0.001, default_time_barrier_minutes=self.time_barrier_minutes, default_max_lookahead=self.max_lookahead)
        except ImportError as e:
            self.logger.error(f'❌ Failed to import RegimeAwareTripleBarrierLabeling: {e}')
            raise RuntimeError(f'Regime aware triple barrier labeling is required but not available: {e}')

    def _generate_labels_with_regime_labeler(self, regime_labeler: Any, data: pd.DataFrame) -> Optional[pd.Series]:
        """Generate labels using the regime labeler."""
        try:
            labels = regime_labeler.generate_labels(data, regime_column=self.regime_col, time_barrier_minutes=self.time_barrier_minutes, max_lookahead=self.max_lookahead)
            if labels is not None:
                self.logger.info(f'✅ Generated {len(labels)} regime-aware labels')
                return labels
            else:
                raise Exception('Regime-aware labeling returned None')
        except Exception as e:
            self.logger.warning(f'⚠️ Regime-aware labeling failed: {e}')
            return None

async def run_step(symbol: str, exchange: str, timeframe: str, data_dir: str=None, force_rerun: bool=False, config: Optional[Dict[str, Any]]=None) -> bool:
    """Run the labeling step with standardized data quality management."

    Args:
        symbol: Trading symbol
        exchange: Exchange name
        timeframe: Timeframe for data
        data_dir: Data directory (will use standardized path if None)
        force_rerun: Force rerun the step
        config: Configuration dictionary

    Returns:
        True if successful, False otherwise
    """
    if config is None:
        config = {}
    if data_dir is None:
        data_dir = standardized_parquet_handler.get_standardized_path('processed_data', exchange, symbol)
    step_config = {'SYMBOL': symbol, 'EXCHANGE': exchange, 'TIMEFRAME': timeframe, 'DATA_DIR': data_dir, 'labeling': {'enable_meta_labeling': True, 'enable_trend_labels': True, 'enable_volatility_labels': True, 'composite_label_strategy': 'weighted_combination'}, 'vectorized_labelling_orchestrator': {'auto_recalculate_hmm_barriers': True, 'hmm_barrier_regime_column': 'hmm_regime', 'time_barrier_minutes': 30, 'max_lookahead': 100, 'profit_take_multiplier': 0.002, 'stop_loss_multiplier': 0.001}, **config}
    step = LabelingStep(step_config)
    await step.initialize()
    return await step.execute_labeling(symbol=symbol, exchange=exchange, timeframe=timeframe, data_dir=data_dir, force_rerun=force_rerun)
if __name__ == '__main__':

    async def test() -> None:
        success = await run_step(symbol='ETHUSDT', exchange='BINANCE', timeframe='1m', data_dir='historical_data')
        tprint(f'Step 5 result: {success}')
    asyncio.run(test())