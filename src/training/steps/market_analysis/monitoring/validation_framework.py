from ..standardized_parquet_handler import standardized_parquet_handler
"""Comprehensive Validation Framework.

This module provides comprehensive validation for all function operations.
"""
import logging
import re
from datetime import datetime
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Dict, List

import asyncio
import numpy as np
import pandas as pd

from src.utils.comprehensive_function_logger import log_step_functions, log_important_calls, log_all_calls, log_internal_call, log_step_progress, log_data_operation
import time

class ComprehensiveValidationFramework:
    """Comprehensive validation framework for all function operations."""

    @log_important_calls
    def __init__(self, logger: Any = None):
        self.logger = logger or logging.getLogger(__name__)
        self.validation_rules: Dict[str, List[Callable]] = {}
        self.validation_history: List[Dict[str, Any]] = []
        self.validation_results: Dict[str, Dict[str, Any]] = {}
        
        # Initialize default validation rules
        self._initialize_default_validation_rules()
    @log_all_calls
    
    def _initialize_default_validation_rules(self) -> None:
        """Initialize default validation rules for common operations."""
        try:
            # Input validation rules
            self.validation_rules['input_validation'] = [
                self._validate_dataframe_input,
                self._validate_string_input,
                self._validate_numeric_input,
                self._validate_path_input
            ]
            
            # Output validation rules
            self.validation_rules['output_validation'] = [
                self._validate_dataframe_output,
                self._validate_boolean_output,
                self._validate_numeric_output,
                self._validate_series_output
            ]
            
            # Data quality validation rules
            self.validation_rules['data_quality'] = [
                self._validate_data_completeness,
                self._validate_data_types,
                self._validate_data_ranges,
                self._validate_data_consistency
            ]
            
            # Performance validation rules
            self.validation_rules['performance_validation'] = [
                self._validate_execution_time,
                self._validate_memory_usage,
                self._validate_cpu_usage
            ]
            
            # Business logic validation rules
            self.validation_rules['business_logic'] = [
                self._validate_labeling_logic,
                self._validate_regime_logic,
                self._validate_triple_barrier_logic
            ]
            
            self.logger.info('✅ Default validation rules initialized')
            
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize default validation rules: {e}")
    @log_all_calls
    
    def _validate_dataframe_input(self, data: Any, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Validate DataFrame input."""
        result = {'valid': True, 'errors': [], 'warnings': []}
        
        try:
            if not isinstance(data, pd.DataFrame):
                result['valid'] = False
                result['errors'].append(f"Expected DataFrame, got {type(data).__name__}")
                return result
            
            if data.empty:
                result['valid'] = False
                result['errors'].append("DataFrame is empty")
                return result
            
            # Check for required columns
            required_columns = context.get('required_columns', [])
            missing_columns = [col for col in required_columns if col not in data.columns]
            if missing_columns:
                result['valid'] = False
                result['errors'].append(f"Missing required columns: {missing_columns}")
            
            # Check for NaN values in critical columns
            critical_columns = context.get('critical_columns', [])
            for col in critical_columns:
                if col in data.columns and data[col].isna().any():
                    result['warnings'].append(f"Column '{col}' contains NaN values")
            
            # Check data types
            expected_types = context.get('expected_types', {})
            for col, expected_type in expected_types.items():
                if col in data.columns:
                    actual_type = data[col].dtype
                    if not pd.api.types.is_dtype_equal(actual_type, expected_type):
                        result['warnings'].append(f"Column '{col}' has type {actual_type}, expected {expected_type}")
            
        except Exception as e:
            result['valid'] = False
            result['errors'].append(f"Validation error: {str(e)}")
        
        return result
    @log_all_calls
    
    def _validate_string_input(self, data: Any, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Validate string input."""
        result = {'valid': True, 'errors': [], 'warnings': []}
        
        try:
            if not isinstance(data, str):
                result['valid'] = False
                result['errors'].append(f"Expected string, got {type(data).__name__}")
                return result
            
            if not data.strip():
                result['valid'] = False
                result['errors'].append("String is empty or whitespace only")
                return result
            
            # Check length constraints
            min_length = context.get('min_length', 0)
            max_length = context.get('max_length', float('inf'))
            
            if len(data) < min_length:
                result['valid'] = False
                result['errors'].append(f"String too short (min: {min_length})")
            
            if len(data) > max_length:
                result['valid'] = False
                result['errors'].append(f"String too long (max: {max_length})")
            
            # Check pattern constraints
            pattern = context.get('pattern')
            if pattern and not re.match(pattern, data):
                result['valid'] = False
                result['errors'].append(f"String doesn't match required pattern: {pattern}")
            
        except Exception as e:
            result['valid'] = False
            result['errors'].append(f"Validation error: {str(e)}")
        
        return result
    @log_all_calls
    
    def _validate_numeric_input(self, data: Any, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Validate numeric input."""
        result = {'valid': True, 'errors': [], 'warnings': []}
        
        try:
            if not isinstance(data, (int, float, np.number)):
                result['valid'] = False
                result['errors'].append(f"Expected numeric, got {type(data).__name__}")
                return result
            
            # Check range constraints
            min_value = context.get('min_value', float('-inf'))
            max_value = context.get('max_value', float('inf'))
            
            if data < min_value:
                result['valid'] = False
                result['errors'].append(f"Value too small (min: {min_value})")
            
            if data > max_value:
                result['valid'] = False
                result['errors'].append(f"Value too large (max: {max_value})")
            
            # Check for NaN or infinite values
            if np.isnan(data) or np.isinf(data):
                result['valid'] = False
                result['errors'].append("Value is NaN or infinite")
            
        except Exception as e:
            result['valid'] = False
            result['errors'].append(f"Validation error: {str(e)}")
        
        return result
    @log_all_calls
    
    def _validate_path_input(self, data: Any, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Validate path input."""
        result = {'valid': True, 'errors': [], 'warnings': []}
        
        try:
            path = Path(data) if not isinstance(data, Path) else data
            
            # Check if path exists
            must_exist = context.get('must_exist', True)
            if must_exist and not path.exists():
                result['valid'] = False
                result['errors'].append(f"Path does not exist: {path}")
                return result
            
            # Check if it's a file or directory
            expected_type = context.get('expected_type', 'file')  # 'file' or 'directory'
            if path.exists():
                if expected_type == 'file' and not path.is_file():
                    result['valid'] = False
                    result['errors'].append(f"Expected file, got directory: {path}")
                elif expected_type == 'directory' and not path.is_dir():
                    result['valid'] = False
                    result['errors'].append(f"Expected directory, got file: {path}")
            
            # Check file extension
            expected_extensions = context.get('expected_extensions', [])
            if expected_extensions and path.suffix.lower() not in expected_extensions:
                result['valid'] = False
                result['errors'].append(f"Invalid file extension. Expected: {expected_extensions}")
            
        except Exception as e:
            result['valid'] = False
            result['errors'].append(f"Validation error: {str(e)}")
        
        return result
    @log_all_calls
    
    def _validate_dataframe_output(self, data: Any, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Validate DataFrame output."""
        result = {'valid': True, 'errors': [], 'warnings': []}
        
        try:
            if data is None:
                result['valid'] = False
                result['errors'].append("Output is None")
                return result
            
            if not isinstance(data, pd.DataFrame):
                result['valid'] = False
                result['errors'].append(f"Expected DataFrame output, got {type(data).__name__}")
                return result
            
            if data.empty:
                result['warnings'].append("Output DataFrame is empty")
            
            # Check for required output columns
            required_columns = context.get('required_columns', [])
            missing_columns = [col for col in required_columns if col not in data.columns]
            if missing_columns:
                result['valid'] = False
                result['errors'].append(f"Missing required output columns: {missing_columns}")
            
            # Check data quality
            if 'label' in data.columns:
                label_counts = data['label'].value_counts()
                if len(label_counts) == 0:
                    result['warnings'].append("No labels generated")
                elif len(label_counts) == 1:
                    result['warnings'].append("Only one label class generated")
            
        except Exception as e:
            result['valid'] = False
            result['errors'].append(f"Validation error: {str(e)}")
        
        return result
    @log_all_calls
    
    def _validate_boolean_output(self, data: Any, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Validate boolean output."""
        result = {'valid': True, 'errors': [], 'warnings': []}
        
        try:
            if not isinstance(data, bool):
                result['valid'] = False
                result['errors'].append(f"Expected boolean output, got {type(data).__name__}")
                return result
            
            # Check expected value
            expected_value = context.get('expected_value')
            if expected_value is not None and data != expected_value:
                result['warnings'].append(f"Expected {expected_value}, got {data}")
            
        except Exception as e:
            result['valid'] = False
            result['errors'].append(f"Validation error: {str(e)}")
        
        return result
    @log_all_calls
    
    def _validate_numeric_output(self, data: Any, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Validate numeric output."""
        result = {'valid': True, 'errors': [], 'warnings': []}
        
        try:
            if not isinstance(data, (int, float, np.number)):
                result['valid'] = False
                result['errors'].append(f"Expected numeric output, got {type(data).__name__}")
                return result
            
            # Check for NaN or infinite values
            if np.isnan(data) or np.isinf(data):
                result['valid'] = False
                result['errors'].append("Output is NaN or infinite")
            
            # Check range constraints
            min_value = context.get('min_value', float('-inf'))
            max_value = context.get('max_value', float('inf'))
            
            if data < min_value or data > max_value:
                result['warnings'].append(f"Output value {data} outside expected range [{min_value}, {max_value}]")
            
        except Exception as e:
            result['valid'] = False
            result['errors'].append(f"Validation error: {str(e)}")
        
        return result
    @log_all_calls
    
    def _validate_series_output(self, data: Any, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Validate Series output."""
        result = {'valid': True, 'errors': [], 'warnings': []}
        
        try:
            if data is None:
                result['valid'] = False
                result['errors'].append("Output is None")
                return result
            
            if not isinstance(data, pd.Series):
                result['valid'] = False
                result['errors'].append(f"Expected Series output, got {type(data).__name__}")
                return result
            
            if data.empty:
                result['warnings'].append("Output Series is empty")
            
            # Check for NaN values
            if data.isna().any():
                nan_count = data.isna().sum()
                result['warnings'].append(f"Output Series contains {nan_count} NaN values")
            
        except Exception as e:
            result['valid'] = False
            result['errors'].append(f"Validation error: {str(e)}")
        
        return result
    @log_all_calls
    
    def _validate_data_completeness(self, data: Any, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Validate data completeness."""
        result = {'valid': True, 'errors': [], 'warnings': []}
        
        try:
            if isinstance(data, pd.DataFrame):
                total_cells = data.size
                missing_cells = data.isna().sum().sum()
                completeness_ratio = (total_cells - missing_cells) / total_cells if total_cells > 0 else 0
                
                min_completeness = context.get('min_completeness', 0.95)
                if completeness_ratio < min_completeness:
                    result['warnings'].append(f"Data completeness {completeness_ratio:.2%} below threshold {min_completeness:.2%}")
            
        except Exception as e:
            result['valid'] = False
            result['errors'].append(f"Validation error: {str(e)}")
        
        return result
    @log_all_calls
    
    def _validate_data_types(self, data: Any, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Validate data types."""
        result = {'valid': True, 'errors': [], 'warnings': []}
        
        try:
            if isinstance(data, pd.DataFrame):
                expected_types = context.get('expected_types', {})
                for col, expected_type in expected_types.items():
                    if col in data.columns:
                        actual_type = data[col].dtype
                        if not pd.api.types.is_dtype_equal(actual_type, expected_type):
                            result['warnings'].append(f"Column '{col}' type mismatch: {actual_type} vs {expected_type}")
            
        except Exception as e:
            result['valid'] = False
            result['errors'].append(f"Validation error: {str(e)}")
        
        return result
    @log_all_calls
    
    def _validate_data_ranges(self, data: Any, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Validate data ranges."""
        result = {'valid': True, 'errors': [], 'warnings': []}
        
        try:
            if isinstance(data, pd.DataFrame):
                column_ranges = context.get('column_ranges', {})
                for col, (min_val, max_val) in column_ranges.items():
                    if col in data.columns:
                        col_data = data[col].dropna()
                        if len(col_data) > 0:
                            if col_data.min() < min_val or col_data.max() > max_val:
                                result['warnings'].append(f"Column '{col}' values outside range [{min_val}, {max_val}]")
            
        except Exception as e:
            result['valid'] = False
            result['errors'].append(f"Validation error: {str(e)}")
        
        return result
    @log_all_calls
    
    def _validate_data_consistency(self, data: Any, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Validate data consistency."""
        result = {'valid': True, 'errors': [], 'warnings': []}
        
        try:
            if isinstance(data, pd.DataFrame):
                # Check for duplicate rows
                if data.duplicated().any():
                    duplicate_count = data.duplicated().sum()
                    result['warnings'].append(f"Found {duplicate_count} duplicate rows")
                
                # Check for inconsistent data patterns
                if 'close' in data.columns and 'high' in data.columns and 'low' in data.columns:
                    invalid_ohlc = (data['close'] > data['high']) | (data['close'] < data['low'])
                    if invalid_ohlc.any():
                        invalid_count = invalid_ohlc.sum()
                        result['warnings'].append(f"Found {invalid_count} rows with invalid OHLC relationships")
            
        except Exception as e:
            result['valid'] = False
            result['errors'].append(f"Validation error: {str(e)}")
        
        return result
    @log_all_calls
    
    def _validate_execution_time(self, data: Any, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Validate execution time."""
        result = {'valid': True, 'errors': [], 'warnings': []}
        
        try:
            execution_time = context.get('execution_time', 0)
            max_time = context.get('max_execution_time', 300)  # 5 minutes default
            
            if execution_time > max_time:
                result['warnings'].append(f"Execution time {execution_time:.2f}s exceeds threshold {max_time}s")
            
        except Exception as e:
            result['valid'] = False
            result['errors'].append(f"Validation error: {str(e)}")
        
        return result
    @log_all_calls
    
    def _validate_memory_usage(self, data: Any, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Validate memory usage."""
        result = {'valid': True, 'errors': [], 'warnings': []}
        
        try:
            memory_usage = context.get('memory_usage_mb', 0)
            max_memory = context.get('max_memory_mb', 1000)  # 1GB default
            
            if memory_usage > max_memory:
                result['warnings'].append(f"Memory usage {memory_usage:.1f}MB exceeds threshold {max_memory}MB")
            
        except Exception as e:
            result['valid'] = False
            result['errors'].append(f"Validation error: {str(e)}")
        
        return result
    @log_all_calls
    
    def _validate_cpu_usage(self, data: Any, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Validate CPU usage."""
        result = {'valid': True, 'errors': [], 'warnings': []}
        
        try:
            cpu_usage = context.get('cpu_usage_percent', 0)
            max_cpu = context.get('max_cpu_percent', 80)  # 80% default
            
            if cpu_usage > max_cpu:
                result['warnings'].append(f"CPU usage {cpu_usage:.1f}% exceeds threshold {max_cpu}%")
            
        except Exception as e:
            result['valid'] = False
            result['errors'].append(f"Validation error: {str(e)}")
        
        return result
    @log_all_calls
    
    def _validate_labeling_logic(self, data: Any, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Validate labeling logic."""
        result = {'valid': True, 'errors': [], 'warnings': []}
        
        try:
            if isinstance(data, pd.DataFrame) and 'label' in data.columns:
                labels = data['label'].dropna()
                
                # Check label distribution
                if len(labels) > 0:
                    label_counts = labels.value_counts()
                    total_labels = len(labels)
                    
                    # Check for extreme class imbalance
                    if len(label_counts) > 1:
                        max_count = label_counts.max()
                        min_count = label_counts.min()
                        imbalance_ratio = max_count / min_count
                        
                        if imbalance_ratio > 10:
                            result['warnings'].append(f"Severe class imbalance detected (ratio: {imbalance_ratio:.1f})")
                    
                    # Check for reasonable label distribution
                    for label, count in label_counts.items():
                        percentage = count / total_labels * 100
                        if percentage < 1:
                            result['warnings'].append(f"Very few samples for label {label} ({percentage:.1f}%)")
            
        except Exception as e:
            result['valid'] = False
            result['errors'].append(f"Validation error: {str(e)}")
        
        return result
    @log_all_calls
    
    def _validate_regime_logic(self, data: Any, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Validate regime logic."""
        result = {'valid': True, 'errors': [], 'warnings': []}
        
        try:
            if isinstance(data, pd.DataFrame):
                regime_columns = [col for col in data.columns if 'regime' in col.lower()]
                
                for regime_col in regime_columns:
                    regimes = data[regime_col].dropna()
                    if len(regimes) > 0:
                        regime_counts = regimes.value_counts()
                        
                        # Check for reasonable number of regimes
                        if len(regime_counts) < 2:
                            result['warnings'].append(f"Only {len(regime_counts)} regime(s) detected in {regime_col}")
                        elif len(regime_counts) > 10:
                            result['warnings'].append(f"Too many regimes ({len(regime_counts)}) in {regime_col}")
                        
                        # Check for regime balance
                        if len(regime_counts) > 1:
                            max_count = regime_counts.max()
                            min_count = regime_counts.min()
                            balance_ratio = max_count / min_count
                            
                            if balance_ratio > 5:
                                result['warnings'].append(f"Unbalanced regimes in {regime_col} (ratio: {balance_ratio:.1f})")
            
        except Exception as e:
            result['valid'] = False
            result['errors'].append(f"Validation error: {str(e)}")
        
        return result
    @log_all_calls
    
    def _validate_triple_barrier_logic(self, data: Any, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Validate triple barrier logic."""
        result = {'valid': True, 'errors': [], 'warnings': []}
        
        try:
            if isinstance(data, pd.DataFrame):
                tb_columns = [col for col in data.columns if 'triple_barrier' in col.lower()]
                
                for tb_col in tb_columns:
                    tb_labels = data[tb_col].dropna()
                    if len(tb_labels) > 0:
                        # Check for valid triple barrier labels (-1, 0, 1)
                        valid_labels = tb_labels.isin([-1, 0, 1])
                        if not valid_labels.all():
                            invalid_labels = tb_labels[~valid_labels].unique()
                            result['warnings'].append(f"Invalid triple barrier labels in {tb_col}: {invalid_labels}")
                        
                        # Check label distribution
                        label_counts = tb_labels.value_counts()
                        total_labels = len(tb_labels)
                        
                        # Check for too many neutral labels
                        neutral_count = label_counts.get(0, 0)
                        neutral_ratio = neutral_count / total_labels
                        
                        if neutral_ratio > 0.8:
                            result['warnings'].append(f"Too many neutral labels in {tb_col} ({neutral_ratio:.1%})")
            
        except Exception as e:
            result['valid'] = False
            result['errors'].append(f"Validation error: {str(e)}")
        
        return result
    
    def validate_function_input(self, function_name: str, input_data: Any, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Validate function input using all applicable rules."""
        try:
            validation_result = {
                'function_name': function_name,
                'validation_type': 'input',
                'timestamp': datetime.now().isoformat(),
                'overall_valid': True,
                'rule_results': {},
                'errors': [],
                'warnings': []
            }
            
            context = context or {}
            
            # Run input validation rules
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
                            validation_result['errors'].append(f"Rule {rule_name} failed: {str(e)}")
                            validation_result['overall_valid'] = False
            
            # Store validation result
            self.validation_history.append(validation_result)
            
            return validation_result
            
        except Exception as e:
            self.logger.error(f"❌ Failed to validate function input: {e}")
            return {'overall_valid': False, 'errors': [str(e)], 'warnings': []}
    
    def validate_function_output(self, function_name: str, output_data: Any, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Validate function output using all applicable rules."""
        try:
            validation_result = {
                'function_name': function_name,
                'validation_type': 'output',
                'timestamp': datetime.now().isoformat(),
                'overall_valid': True,
                'rule_results': {},
                'errors': [],
                'warnings': []
            }
            
            context = context or {}
            
            # Run output validation rules
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
                            validation_result['errors'].append(f"Rule {rule_name} failed: {str(e)}")
                            validation_result['overall_valid'] = False
            
            # Store validation result
            self.validation_history.append(validation_result)
            
            return validation_result
            
        except Exception as e:
            self.logger.error(f"❌ Failed to validate function output: {e}")
            return {'overall_valid': False, 'errors': [str(e)], 'warnings': []}
    
    def generate_validation_report(self) -> Dict[str, Any]:
        """Generate comprehensive validation report."""
        try:
            if not self.validation_history:
                return {'total_validations': 0, 'message': 'No validation data recorded'}
            
            # Analyze validation results
            total_validations = len(self.validation_history)
            successful_validations = len([v for v in self.validation_history if v['overall_valid']])
            failed_validations = total_validations - successful_validations
            
            # Group by function
            function_validations = {}
            for validation in self.validation_history:
                func_name = validation['function_name']
                if func_name not in function_validations:
                    function_validations[func_name] = {'input': [], 'output': []}
                function_validations[func_name][validation['validation_type']].append(validation)
            
            # Analyze error patterns
            error_patterns = {}
            warning_patterns = {}
            
            for validation in self.validation_history:
                for error in validation['errors']:
                    error_patterns[error] = error_patterns.get(error, 0) + 1
                
                for warning in validation['warnings']:
                    warning_patterns[warning] = warning_patterns.get(warning, 0) + 1
            
            return {
                'total_validations': total_validations,
                'successful_validations': successful_validations,
                'failed_validations': failed_validations,
                'success_rate': (successful_validations / total_validations * 100) if total_validations > 0 else 0,
                'function_validations': function_validations,
                'error_patterns': error_patterns,
                'warning_patterns': warning_patterns,
                'most_common_errors': sorted(error_patterns.items(), key = lambda x: x[1], reverse = True)[:5],
                'most_common_warnings': sorted(warning_patterns.items(), key = lambda x: x[1], reverse = True)[:5]
            }
            
        except Exception as e:
            self.logger.error(f"❌ Failed to generate validation report: {e}")
            return {}
    
    def log_validation_report(self, report: Dict[str, Any]) -> None:
        """Log comprehensive validation report."""
        try:
            if report.get('total_validations', 0) == 0:
                self.logger.info("📋 No validation data recorded")
                return
            
            self.logger.info("📋 COMPREHENSIVE VALIDATION REPORT")
            self.logger.info("=" * 50)
            self.logger.info(f"Total Validations: {report['total_validations']}")
            self.logger.info(f"Successful Validations: {report['successful_validations']}")
            self.logger.info(f"Failed Validations: {report['failed_validations']}")
            self.logger.info(f"Success Rate: {report['success_rate']:.1f}%")
            
            # Function-specific validation results
            function_validations = report.get('function_validations', {})
            if function_validations:
                self.logger.info(f"\n🔍 FUNCTION VALIDATION RESULTS:")
                for func_name, validations in function_validations.items():
                    input_validations = validations.get('input', [])
                    output_validations = validations.get('output', [])
                    
                    input_success = len([v for v in input_validations if v['overall_valid']])
                    output_success = len([v for v in output_validations if v['overall_valid']])
                    
                    self.logger.info(f"   {func_name}:")
                    self.logger.info(f"     Input Validations: {input_success}/{len(input_validations)} successful")
                    self.logger.info(f"     Output Validations: {output_success}/{len(output_validations)} successful")
            
            # Most common errors
            most_common_errors = report.get('most_common_errors', [])
            if most_common_errors:
                self.logger.info(f"\n❌ MOST COMMON ERRORS:")
                for error, count in most_common_errors:
                    self.logger.info(f"   - {error}: {count} occurrences")
            
            # Most common warnings
            most_common_warnings = report.get('most_common_warnings', [])
            if most_common_warnings:
                self.logger.info(f"\n⚠️ MOST COMMON WARNINGS:")
                for warning, count in most_common_warnings:
                    self.logger.info(f"   - {warning}: {count} occurrences")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to log validation report: {e}")

def comprehensive_validation(validator: ComprehensiveValidationFramework):
    """Decorator for comprehensive validation."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            # Validate inputs
            input_context = {
                'function_name': func.__name__,
                'args_count': len(args),
                'kwargs_keys': list(kwargs.keys())
            }
            
            # Validate first argument if it's a DataFrame
            if args and isinstance(args[0], pd.DataFrame):
                input_validation = validator.validate_function_input(func.__name__, args[0], input_context)
                if not input_validation['overall_valid']:
                    validator.logger.warning(f"⚠️ Input validation failed for {func.__name__}: {input_validation['errors']}")
            
            try:
                result = await func(*args, **kwargs)
                
                # Validate output
                output_context = {
                    'function_name': func.__name__,
                    'execution_time': getattr(func, '_execution_time', 0)
                }
                
                output_validation = validator.validate_function_output(func.__name__, result, output_context)
                if not output_validation['overall_valid']:
                    validator.logger.warning(f"⚠️ Output validation failed for {func.__name__}: {output_validation['errors']}")
                
                return result
                
            except Exception as e:
                # Log validation failure
                validator.logger.error(f"❌ Function {func.__name__} failed with error: {e}")
                raise
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            # Validate inputs
            input_context = {
                'function_name': func.__name__,
                'args_count': len(args),
                'kwargs_keys': list(kwargs.keys())
            }
            
            # Validate first argument if it's a DataFrame
            if args and isinstance(args[0], pd.DataFrame):
                input_validation = validator.validate_function_input(func.__name__, args[0], input_context)
                if not input_validation['overall_valid']:
                    validator.logger.warning(f"⚠️ Input validation failed for {func.__name__}: {input_validation['errors']}")
            
            try:
                result = func(*args, **kwargs)
                
                # Validate output
                output_context = {
                    'function_name': func.__name__,
                    'execution_time': getattr(func, '_execution_time', 0)
                }
                
                output_validation = validator.validate_function_output(func.__name__, result, output_context)
                if not output_validation['overall_valid']:
                    validator.logger.warning(f"⚠️ Output validation failed for {func.__name__}: {output_validation['errors']}")
                
                return result
                
            except Exception as e:
                # Log validation failure
                validator.logger.error(f"❌ Function {func.__name__} failed with error: {e}")
                raise
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    return decorator