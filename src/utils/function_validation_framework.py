#!/usr/bin/env python3
"""
Comprehensive Function Validation Framework

This module provides detailed function validation including:
- Function entry parameter validation with type checking
- Input sanitization and security validation
- Business logic validation for domain-specific functions
- Output validation and return value checking
- Side effect detection and validation
- Comprehensive validation reporting
"""

import inspect
import logging
import re
from typing import Any, Callable, Dict, List, Optional, Union, Tuple, Type
from dataclasses import dataclass, field
from enum import Enum
import json
from datetime import datetime
import os
import pathlib


class ValidationSeverity(Enum):
    """Severity levels for validation issues."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class ValidationCategory(Enum):
    """Categories of validation checks."""
    TYPE_CHECK = "type_check"
    VALUE_CHECK = "value_check"
    SECURITY_CHECK = "security_check"
    BUSINESS_LOGIC = "business_logic"
    PERFORMANCE_CHECK = "performance_check"
    SIDE_EFFECT_CHECK = "side_effect_check"


@dataclass
class ValidationIssue:
    """Represents a validation issue."""
    category: ValidationCategory
    severity: ValidationSeverity
    message: str
    parameter_name: Optional[str] = None
    expected_value: Optional[Any] = None
    actual_value: Optional[Any] = None
    recommendation: Optional[str] = None
    details: Optional[Dict[str, Any]] = None


@dataclass
class ValidationResult:
    """Result of function validation."""
    is_valid: bool
    issues: List[ValidationIssue] = field(default_factory=list)
    warnings: List[ValidationIssue] = field(default_factory=list)
    errors: List[ValidationIssue] = field(default_factory=list)
    critical_issues: List[ValidationIssue] = field(default_factory=list)
    validation_score: float = 1.0
    validation_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class FunctionValidator:
    """Comprehensive function validator."""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        self.validation_rules = self._initialize_validation_rules()
    
    def _initialize_validation_rules(self) -> Dict[str, Dict[str, Any]]:
        """Initialize validation rules for different function types."""
        return {
            'data_collection': {
                'required_params': ['symbol', 'exchange'],
                'param_types': {
                    'symbol': str,
                    'exchange': str,
                    'timeframe': str,
                    'data_dir': (str, type(None))
                },
                'param_patterns': {
                    'symbol': r'^[A-Z]{2,10}USDT?$',
                    'exchange': r'^[A-Z_]+$',
                    'timeframe': r'^[0-9]+[mhdw]$'
                },
                'param_ranges': {
                    'timeframe': ['1m', '5m', '15m', '30m', '1h', '4h', '1d']
                },
                'security_checks': ['path_traversal', 'injection']
            },
            'data_validation': {
                'required_params': ['data', 'schema'],
                'param_types': {
                    'data': (list, dict),
                    'schema': str
                },
                'business_logic': ['data_integrity', 'schema_compliance']
            },
            'file_operations': {
                'required_params': ['file_path'],
                'param_types': {
                    'file_path': str
                },
                'security_checks': ['path_traversal', 'file_permissions'],
                'business_logic': ['file_existence', 'file_format']
            }
        }
    
    def validate_function_entry(self, func: Callable, args: tuple, kwargs: dict, 
                            function_type: str = 'generic') -> ValidationResult:
        """Validate function entry with comprehensive checks."""
        start_time = datetime.now()
        result = ValidationResult(is_valid=True)
        
        try:
            # Get function signature
            sig = inspect.signature(func)
            bound_args = sig.bind(*args, **kwargs)
            bound_args.apply_defaults()
            
            # Basic parameter validation
            self._validate_parameter_count(func, bound_args, result)
            self._validate_required_parameters(func, bound_args, result, function_type)
            self._validate_parameter_types(func, bound_args, result, function_type)
            self._validate_parameter_values(func, bound_args, result, function_type)
            
            # Security validation
            self._validate_security(func, bound_args, result, function_type)
            
            # Business logic validation
            self._validate_business_logic(func, bound_args, result, function_type)
            
            # Performance validation
            self._validate_performance_concerns(func, bound_args, result, function_type)
            
            # Calculate validation score
            result.validation_score = self._calculate_validation_score(result)
            result.is_valid = len(result.critical_issues) == 0 and len(result.errors) == 0
            
        except Exception as e:
            result.is_valid = False
            result.critical_issues.append(ValidationIssue(
                category=ValidationCategory.TYPE_CHECK,
                severity=ValidationSeverity.CRITICAL,
                message=f"Validation error: {str(e)}",
                details={'exception': str(e)}
            ))
        
        finally:
            result.validation_time = (datetime.now() - start_time).total_seconds()
        
        return result
    
    def _validate_parameter_count(self, func: Callable, bound_args: inspect.BoundArguments, 
                                result: ValidationResult) -> None:
        """Validate parameter count."""
        sig = inspect.signature(func)
        expected_count = len(sig.parameters)
        actual_count = len(bound_args.arguments)
        
        if actual_count != expected_count:
            result.warnings.append(ValidationIssue(
                category=ValidationCategory.TYPE_CHECK,
                severity=ValidationSeverity.WARNING,
                message=f"Parameter count mismatch: expected {expected_count}, got {actual_count}",
                details={'expected': expected_count, 'actual': actual_count}
            ))
    
    def _validate_required_parameters(self, func: Callable, bound_args: inspect.BoundArguments, 
                                    result: ValidationResult, function_type: str) -> None:
        """Validate required parameters."""
        rules = self.validation_rules.get(function_type, {})
        required_params = rules.get('required_params', [])
        
        for param_name in required_params:
            if param_name not in bound_args.arguments:
                result.errors.append(ValidationIssue(
                    category=ValidationCategory.TYPE_CHECK,
                    severity=ValidationSeverity.ERROR,
                    message=f"Required parameter '{param_name}' is missing",
                    parameter_name=param_name,
                    recommendation=f"Provide the required parameter: {param_name}"
                ))
    
    def _validate_parameter_types(self, func: Callable, bound_args: inspect.BoundArguments, 
                                result: ValidationResult, function_type: str) -> None:
        """Validate parameter types."""
        rules = self.validation_rules.get(function_type, {})
        param_types = rules.get('param_types', {})
        
        for param_name, value in bound_args.arguments.items():
            if param_name in param_types:
                expected_type = param_types[param_name]
                
                # Handle union types
                if isinstance(expected_type, tuple):
                    if not isinstance(value, expected_type):
                        result.errors.append(ValidationIssue(
                            category=ValidationCategory.TYPE_CHECK,
                            severity=ValidationSeverity.ERROR,
                            message=f"Parameter '{param_name}' type mismatch: expected one of {expected_type}, got {type(value)}",
                            parameter_name=param_name,
                            expected_value=expected_type,
                            actual_value=type(value),
                            recommendation=f"Ensure parameter '{param_name}' is of type {expected_type}"
                        ))
                else:
                    if not isinstance(value, expected_type):
                        result.errors.append(ValidationIssue(
                            category=ValidationCategory.TYPE_CHECK,
                            severity=ValidationSeverity.ERROR,
                            message=f"Parameter '{param_name}' type mismatch: expected {expected_type}, got {type(value)}",
                            parameter_name=param_name,
                            expected_value=expected_type,
                            actual_value=type(value),
                            recommendation=f"Ensure parameter '{param_name}' is of type {expected_type}"
                        ))
    
    def _validate_parameter_values(self, func: Callable, bound_args: inspect.BoundArguments, 
                                result: ValidationResult, function_type: str) -> None:
        """Validate parameter values."""
        rules = self.validation_rules.get(function_type, {})
        param_patterns = rules.get('param_patterns', {})
        param_ranges = rules.get('param_ranges', {})
        
        for param_name, value in bound_args.arguments.items():
            # Pattern validation
            if param_name in param_patterns and isinstance(value, str):
                pattern = param_patterns[param_name]
                if not re.match(pattern, value):
                    result.warnings.append(ValidationIssue(
                        category=ValidationCategory.VALUE_CHECK,
                        severity=ValidationSeverity.WARNING,
                        message=f"Parameter '{param_name}' value '{value}' does not match expected pattern",
                        parameter_name=param_name,
                        expected_value=pattern,
                        actual_value=value,
                        recommendation=f"Ensure parameter '{param_name}' matches pattern: {pattern}"
                    ))
            
            # Range validation
            if param_name in param_ranges and isinstance(value, str):
                allowed_values = param_ranges[param_name]
                if value not in allowed_values:
                    result.warnings.append(ValidationIssue(
                        category=ValidationCategory.VALUE_CHECK,
                        severity=ValidationSeverity.WARNING,
                        message=f"Parameter '{param_name}' value '{value}' is not in allowed range",
                        parameter_name=param_name,
                        expected_value=allowed_values,
                        actual_value=value,
                        recommendation=f"Use one of the allowed values: {allowed_values}"
                    ))
            
            # Null/empty validation
            if value is None and param_name in ['symbol', 'exchange', 'data_dir']:
                result.critical_issues.append(ValidationIssue(
                    category=ValidationCategory.VALUE_CHECK,
                    severity=ValidationSeverity.CRITICAL,
                    message=f"Critical parameter '{param_name}' is None",
                    parameter_name=param_name,
                    actual_value=value,
                    recommendation=f"Provide a valid value for parameter '{param_name}'"
                ))
            
            if isinstance(value, str) and value.strip() == "":
                result.warnings.append(ValidationIssue(
                    category=ValidationCategory.VALUE_CHECK,
                    severity=ValidationSeverity.WARNING,
                    message=f"Parameter '{param_name}' is empty string",
                    parameter_name=param_name,
                    actual_value=value,
                    recommendation=f"Provide a non-empty value for parameter '{param_name}'"
                ))
    
    def _validate_security(self, func: Callable, bound_args: inspect.BoundArguments, 
                        result: ValidationResult, function_type: str) -> None:
        """Validate security concerns."""
        rules = self.validation_rules.get(function_type, {})
        security_checks = rules.get('security_checks', [])
        
        for param_name, value in bound_args.arguments.items():
            if isinstance(value, str):
                # Path traversal check
                if 'path_traversal' in security_checks:
                    if '..' in value or value.startswith('/'):
                        result.critical_issues.append(ValidationIssue(
                            category=ValidationCategory.SECURITY_CHECK,
                            severity=ValidationSeverity.CRITICAL,
                            message=f"Potential path traversal detected in parameter '{param_name}'",
                            parameter_name=param_name,
                            actual_value=value,
                            recommendation="Use relative paths and avoid '..' sequences"
                        ))
                
                # Injection check
                if 'injection' in security_checks:
                    dangerous_patterns = [';', '|', '&', '`', '$', '$(', '${']
                    for pattern in dangerous_patterns:
                        if pattern in value:
                            result.warnings.append(ValidationIssue(
                                category=ValidationCategory.SECURITY_CHECK,
                                severity=ValidationSeverity.WARNING,
                                message=f"Potential injection pattern '{pattern}' detected in parameter '{param_name}'",
                                parameter_name=param_name,
                                actual_value=value,
                                recommendation="Sanitize input to prevent injection attacks"
                            ))
    
    def _validate_business_logic(self, func: Callable, bound_args: inspect.BoundArguments, 
                            result: ValidationResult, function_type: str) -> None:
        """Validate business logic constraints."""
        rules = self.validation_rules.get(function_type, {})
        business_logic_checks = rules.get('business_logic', [])
        
        for check in business_logic_checks:
            if check == 'data_integrity':
                self._validate_data_integrity(func, bound_args, result)
            elif check == 'schema_compliance':
                self._validate_schema_compliance(func, bound_args, result)
            elif check == 'file_existence':
                self._validate_file_existence(func, bound_args, result)
            elif check == 'file_format':
                self._validate_file_format(func, bound_args, result)
    
    def _validate_data_integrity(self, func: Callable, bound_args: inspect.BoundArguments, 
                            result: ValidationResult) -> None:
        """Validate data integrity."""
        for param_name, value in bound_args.arguments.items():
            if isinstance(value, (list, dict)) and len(value) == 0:
                result.warnings.append(ValidationIssue(
                    category=ValidationCategory.BUSINESS_LOGIC,
                    severity=ValidationSeverity.WARNING,
                    message=f"Parameter '{param_name}' contains empty data",
                    parameter_name=param_name,
                    actual_value=value,
                    recommendation="Ensure data contains valid entries"
                ))
    
    def _validate_schema_compliance(self, func: Callable, bound_args: inspect.BoundArguments, 
                                result: ValidationResult) -> None:
        """Validate schema compliance."""
        schema_param = bound_args.arguments.get('schema')
        data_param = bound_args.arguments.get('data')
        
        if schema_param and data_param:
            # Basic schema validation logic
            if isinstance(data_param, dict) and schema_param not in ['klines', 'aggtrades', 'unified']:
                result.warnings.append(ValidationIssue(
                    category=ValidationCategory.BUSINESS_LOGIC,
                    severity=ValidationSeverity.WARNING,
                    message=f"Unknown schema '{schema_param}'",
                    parameter_name='schema',
                    actual_value=schema_param,
                    recommendation="Use one of: klines, aggtrades, unified"
                ))
    
    def _validate_file_existence(self, func: Callable, bound_args: inspect.BoundArguments, 
                            result: ValidationResult) -> None:
        """Validate file existence."""
        file_path_param = bound_args.arguments.get('file_path')
        if file_path_param and isinstance(file_path_param, str):
            if not os.path.exists(file_path_param):
                result.warnings.append(ValidationIssue(
                    category=ValidationCategory.BUSINESS_LOGIC,
                    severity=ValidationSeverity.WARNING,
                    message=f"File does not exist: {file_path_param}",
                    parameter_name='file_path',
                    actual_value=file_path_param,
                    recommendation="Ensure file exists before processing"
                ))
    
    def _validate_file_format(self, func: Callable, bound_args: inspect.BoundArguments, 
                            result: ValidationResult) -> None:
        """Validate file format."""
        file_path_param = bound_args.arguments.get('file_path')
        if file_path_param and isinstance(file_path_param, str):
            if not file_path_param.endswith(('.parquet', '.csv', '.json')):
                result.warnings.append(ValidationIssue(
                    category=ValidationCategory.BUSINESS_LOGIC,
                    severity=ValidationSeverity.WARNING,
                    message=f"Unsupported file format: {file_path_param}",
                    parameter_name='file_path',
                    actual_value=file_path_param,
                    recommendation="Use supported formats: .parquet, .csv, .json"
                ))
    
    def _validate_performance_concerns(self, func: Callable, bound_args: inspect.BoundArguments, 
                                    result: ValidationResult, function_type: str) -> None:
        """Validate performance concerns."""
        # Check for large data parameters
        for param_name, value in bound_args.arguments.items():
            if isinstance(value, (list, dict)):
                size = len(value)
                if size > 10000:
                    result.warnings.append(ValidationIssue(
                        category=ValidationCategory.PERFORMANCE_CHECK,
                        severity=ValidationSeverity.WARNING,
                        message=f"Large data parameter '{param_name}' with {size} items",
                        parameter_name=param_name,
                        actual_value=size,
                        recommendation="Consider data chunking or streaming for large datasets"
                    ))
    
    def _calculate_validation_score(self, result: ValidationResult) -> float:
        """Calculate validation score based on issues."""
        score = 1.0
        
        # Deduct points for issues
        score -= len(result.critical_issues) * 0.4
        score -= len(result.errors) * 0.3
        score -= len(result.warnings) * 0.1
        
        return max(0.0, min(1.0, score))
    
    def validate_function_output(self, func: Callable, return_value: Any, 
                            function_type: str = 'generic') -> ValidationResult:
        """Validate function output."""
        start_time = datetime.now()
        result = ValidationResult(is_valid=True)
        
        try:
            # Check return type annotation
            if hasattr(func, '__annotations__') and 'return' in func.__annotations__:
                expected_type = func.__annotations__['return']
                if not isinstance(return_value, expected_type):
                    result.errors.append(ValidationIssue(
                        category=ValidationCategory.TYPE_CHECK,
                        severity=ValidationSeverity.ERROR,
                        message=f"Return type mismatch: expected {expected_type}, got {type(return_value)}",
                        expected_value=expected_type,
                        actual_value=type(return_value),
                        recommendation=f"Ensure function returns {expected_type}"
                    ))
            
            # Check for None returns on critical functions
            if return_value is None and func.__name__ in ['execute', 'run_step', 'initialize']:
                result.critical_issues.append(ValidationIssue(
                    category=ValidationCategory.VALUE_CHECK,
                    severity=ValidationSeverity.CRITICAL,
                    message="Critical function returned None",
                    actual_value=return_value,
                    recommendation="Ensure critical functions return valid results"
                ))
            
            # Check for empty results
            if isinstance(return_value, (list, dict)) and len(return_value) == 0:
                result.warnings.append(ValidationIssue(
                    category=ValidationCategory.VALUE_CHECK,
                    severity=ValidationSeverity.WARNING,
                    message="Function returned empty collection",
                    actual_value=return_value,
                    recommendation="Consider returning meaningful data or None"
                ))
            
            # Calculate validation score
            result.validation_score = self._calculate_validation_score(result)
            result.is_valid = len(result.critical_issues) == 0 and len(result.errors) == 0
            
        except Exception as e:
            result.is_valid = False
            result.critical_issues.append(ValidationIssue(
                category=ValidationCategory.TYPE_CHECK,
                severity=ValidationSeverity.CRITICAL,
                message=f"Output validation error: {str(e)}",
                details={'exception': str(e)}
            ))
        
        finally:
            result.validation_time = (datetime.now() - start_time).total_seconds()
        
        return result
    
    def generate_validation_report(self, result: ValidationResult, func_name: str) -> Dict[str, Any]:
        """Generate comprehensive validation report."""
        return {
            'function_name': func_name,
            'timestamp': datetime.now().isoformat(),
            'is_valid': result.is_valid,
            'validation_score': result.validation_score,
            'validation_time': result.validation_time,
            'summary': {
                'total_issues': len(result.issues),
                'critical_issues': len(result.critical_issues),
                'errors': len(result.errors),
                'warnings': len(result.warnings)
            },
            'issues_by_category': self._group_issues_by_category(result.issues),
            'issues_by_severity': self._group_issues_by_severity(result.issues),
            'detailed_issues': [
                {
                    'category': issue.category.value,
                    'severity': issue.severity.value,
                    'message': issue.message,
                    'parameter_name': issue.parameter_name,
                    'expected_value': str(issue.expected_value) if issue.expected_value is not None else None,
                    'actual_value': str(issue.actual_value) if issue.actual_value is not None else None,
                    'recommendation': issue.recommendation,
                    'details': issue.details
                }
                for issue in result.issues
            ],
            'metadata': result.metadata
        }
    
    def _group_issues_by_category(self, issues: List[ValidationIssue]) -> Dict[str, int]:
        """Group issues by category."""
        categories = {}
        for issue in issues:
            category = issue.category.value
            categories[category] = categories.get(category, 0) + 1
        return categories
    
    def _group_issues_by_severity(self, issues: List[ValidationIssue]) -> Dict[str, int]:
        """Group issues by severity."""
        severities = {}
        for issue in issues:
            severity = issue.severity.value
            severities[severity] = severities.get(severity, 0) + 1
        return severities


# Global validator instance
_global_validator = FunctionValidator()


def validate_function_entry(function_type: str = 'generic'):
    """Decorator for validating function entry."""
    def decorator(func: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            # Validate function entry
            validation_result = _global_validator.validate_function_entry(func, args, kwargs, function_type)
            
            # Log validation results
            if not validation_result.is_valid:
                logger = logging.getLogger(func.__module__)
                logger.error(f"❌ Function entry validation failed for {func.__name__}")
                for issue in validation_result.critical_issues + validation_result.errors:
                    logger.error(f"   {issue.severity.value.upper()}: {issue.message}")
            
            # Execute function
            return func(*args, **kwargs)
        
        return wrapper
    return decorator


def validate_function_output(function_type: str = 'generic'):
    """Decorator for validating function output."""
    def decorator(func: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            # Execute function
            result = func(*args, **kwargs)
            
            # Validate function output
            validation_result = _global_validator.validate_function_output(func, result, function_type)
            
            # Log validation results
            if not validation_result.is_valid:
                logger = logging.getLogger(func.__module__)
                logger.error(f"❌ Function output validation failed for {func.__name__}")
                for issue in validation_result.critical_issues + validation_result.errors:
                    logger.error(f"   {issue.severity.value.upper()}: {issue.message}")
            
            return result
        
        return wrapper
    return decorator


def get_function_validator() -> FunctionValidator:
    """Get the global function validator instance."""
    return _global_validator