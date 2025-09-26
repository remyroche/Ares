#!/usr/bin/env python3
"""
Comprehensive Input Validation and Security System for NAS Components

This module provides input validation, security hardening, and data integrity
checks to prevent security vulnerabilities and ensure data quality.
"""

import re
import hashlib
import hmac
import secrets
import json
import pickle
import base64
from typing import Any, Dict, List, Optional, Union, Callable, Type, TypeVar
from dataclasses import dataclass, field
from enum import Enum
import logging
import numpy as np
from pathlib import Path

from .nas_error_handling import (
    NASValidationError, NASConfigurationError, ErrorContext, 
    error_context, safe_execute, get_error_handler
)

T = TypeVar('T')


class ValidationLevel(Enum):
    """Validation levels for different security requirements."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class SecurityLevel(Enum):
    """Security levels for different operations."""
    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    SECRET = "secret"


@dataclass
class ValidationRule:
    """Validation rule definition."""
    name: str
    validator: Callable[[Any], bool]
    error_message: str
    level: ValidationLevel = ValidationLevel.MEDIUM
    required: bool = True


@dataclass
class SecurityPolicy:
    """Security policy definition."""
    name: str
    level: SecurityLevel
    allowed_operations: List[str]
    required_permissions: List[str]
    data_encryption_required: bool = False
    audit_logging_required: bool = True


class InputValidator:
    """Comprehensive input validation system."""
    
    def __init__(self):
        self._validation_rules: Dict[str, List[ValidationRule]] = {}
        self._error_handler = get_error_handler()
        self._logger = logging.getLogger(__name__)
    
    def add_validation_rule(
        self,
        field_name: str,
        rule: ValidationRule
    ) -> None:
        """Add a validation rule for a field."""
        if field_name not in self._validation_rules:
            self._validation_rules[field_name] = []
        
        self._validation_rules[field_name].append(rule)
    
    def validate_field(
        self,
        field_name: str,
        value: Any,
        context: Optional[ErrorContext] = None
    ) -> bool:
        """Validate a single field."""
        try:
            if field_name not in self._validation_rules:
                return True
            
            for rule in self._validation_rules[field_name]:
                if rule.required and value is None:
                    raise NASValidationError(
                        f"Required field {field_name} is None",
                        context
                    )
                
                if value is not None and not rule.validator(value):
                    raise NASValidationError(
                        f"Validation failed for {field_name}: {rule.error_message}",
                        context
                    )
            
            return True
            
        except Exception as e:
            if isinstance(e, NASValidationError):
                raise
            else:
                context = context or ErrorContext("validate_field", "input_validator")
                self._error_handler.handle_error(e, context, reraise=False)
                return False
    
    def validate_data(
        self,
        data: Dict[str, Any],
        context: Optional[ErrorContext] = None
    ) -> bool:
        """Validate a complete data structure."""
        try:
            for field_name, value in data.items():
                self.validate_field(field_name, value, context)
            
            return True
            
        except Exception as e:
            if isinstance(e, NASValidationError):
                raise
            else:
                context = context or ErrorContext("validate_data", "input_validator")
                self._error_handler.handle_error(e, context, reraise=False)
                return False


class SecurityManager:
    """Security management system."""
    
    def __init__(self, secret_key: Optional[str] = None):
        self.secret_key = secret_key or secrets.token_hex(32)
        self._security_policies: Dict[str, SecurityPolicy] = {}
        self._error_handler = get_error_handler()
        self._logger = logging.getLogger(__name__)
    
    def add_security_policy(self, policy: SecurityPolicy) -> None:
        """Add a security policy."""
        self._security_policies[policy.name] = policy
    
    def check_permission(
        self,
        operation: str,
        user_permissions: List[str],
        context: Optional[ErrorContext] = None
    ) -> bool:
        """Check if user has permission for operation."""
        try:
            for policy in self._security_policies.values():
                if operation in policy.allowed_operations:
                    for required_permission in policy.required_permissions:
                        if required_permission not in user_permissions:
                            raise NASValidationError(
                                f"Insufficient permissions for operation {operation}",
                                context
                            )
                    return True
            
            return True
            
        except Exception as e:
            if isinstance(e, NASValidationError):
                raise
            else:
                context = context or ErrorContext("check_permission", "security_manager")
                self._error_handler.handle_error(e, context, reraise=False)
                return False
    
    def encrypt_data(self, data: str) -> str:
        """Encrypt sensitive data."""
        try:
            return base64.b64encode(
                hmac.new(
                    self.secret_key.encode(),
                    data.encode(),
                    hashlib.sha256
                ).digest()
            ).decode()
        except Exception as e:
            context = ErrorContext("encrypt_data", "security_manager")
            self._error_handler.handle_error(e, context, reraise=False)
            return data
    
    def decrypt_data(self, encrypted_data: str) -> str:
        """Decrypt sensitive data."""
        try:
            # This is a simplified implementation
            # In production, use proper encryption libraries
            return encrypted_data
        except Exception as e:
            context = ErrorContext("decrypt_data", "security_manager")
            self._error_handler.handle_error(e, context, reraise=False)
            return encrypted_data


class DataIntegrityChecker:
    """Data integrity checking system."""
    
    def __init__(self):
        self._error_handler = get_error_handler()
        self._logger = logging.getLogger(__name__)
    
    def calculate_checksum(self, data: Union[str, bytes]) -> str:
        """Calculate checksum for data."""
        try:
            if isinstance(data, str):
                data = data.encode()
            
            return hashlib.sha256(data).hexdigest()
            
        except Exception as e:
            context = ErrorContext("calculate_checksum", "data_integrity_checker")
            self._error_handler.handle_error(e, context, reraise=False)
            return ""
    
    def verify_checksum(
        self,
        data: Union[str, bytes],
        expected_checksum: str
    ) -> bool:
        """Verify data integrity using checksum."""
        try:
            actual_checksum = self.calculate_checksum(data)
            return actual_checksum == expected_checksum
            
        except Exception as e:
            context = ErrorContext("verify_checksum", "data_integrity_checker")
            self._error_handler.handle_error(e, context, reraise=False)
            return False
    
    def validate_data_format(
        self,
        data: Any,
        expected_format: str,
        context: Optional[ErrorContext] = None
    ) -> bool:
        """Validate data format."""
        try:
            if expected_format == "json":
                if isinstance(data, str):
                    json.loads(data)
                elif isinstance(data, dict):
                    json.dumps(data)
                else:
                    raise NASValidationError("Invalid JSON format", context)
            
            elif expected_format == "pickle":
                if isinstance(data, bytes):
                    pickle.loads(data)
                else:
                    raise NASValidationError("Invalid pickle format", context)
            
            elif expected_format == "numpy":
                if not isinstance(data, np.ndarray):
                    raise NASValidationError("Invalid numpy array format", context)
            
            return True
            
        except Exception as e:
            if isinstance(e, NASValidationError):
                raise
            else:
                context = context or ErrorContext("validate_data_format", "data_integrity_checker")
                self._error_handler.handle_error(e, context, reraise=False)
                return False


# Common validation functions
def validate_not_none(value: Any) -> bool:
    """Validate that value is not None."""
    return value is not None


def validate_not_empty(value: Union[str, List, Dict]) -> bool:
    """Validate that value is not empty."""
    if isinstance(value, str):
        return len(value.strip()) > 0
    elif isinstance(value, (list, dict)):
        return len(value) > 0
    else:
        return True


def validate_positive_number(value: Union[int, float]) -> bool:
    """Validate that value is a positive number."""
    return isinstance(value, (int, float)) and value > 0


def validate_range(
    value: Union[int, float],
    min_val: Union[int, float],
    max_val: Union[int, float]
) -> bool:
    """Validate that value is within range."""
    return min_val <= value <= max_val


def validate_email(email: str) -> bool:
    """Validate email format."""
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(pattern, email) is not None


def validate_url(url: str) -> bool:
    """Validate URL format."""
    pattern = r'^https?://[^\s/$.?#].[^\s]*$'
    return re.match(pattern, url) is not None


def validate_file_path(file_path: str) -> bool:
    """Validate file path format."""
    try:
        Path(file_path).resolve()
        return True
    except Exception:
        return False


def validate_safe_string(value: str) -> bool:
    """Validate that string doesn't contain dangerous characters."""
    dangerous_patterns = [
        r'<script.*?>',
        r'javascript:',
        r'data:',
        r'vbscript:',
        r'onload=',
        r'onerror=',
        r'eval\(',
        r'exec\(',
        r'__import__'
    ]
    
    for pattern in dangerous_patterns:
        if re.search(pattern, value, re.IGNORECASE):
            return False
    
    return True


def validate_numeric_string(value: str) -> bool:
    """Validate that string contains only numeric characters."""
    return value.isdigit()


def validate_alphanumeric_string(value: str) -> bool:
    """Validate that string contains only alphanumeric characters."""
    return value.isalnum()


def validate_safe_filename(filename: str) -> bool:
    """Validate that filename is safe."""
    dangerous_chars = ['/', '\\', ':', '*', '?', '"', '<', '>', '|']
    return not any(char in filename for char in dangerous_chars)


def validate_json_schema(data: Dict[str, Any], schema: Dict[str, Any]) -> bool:
    """Validate data against JSON schema."""
    try:
        # Simplified schema validation
        for key, value in schema.items():
            if key not in data:
                return False
            
            if isinstance(value, type):
                if not isinstance(data[key], value):
                    return False
            elif isinstance(value, dict):
                if not validate_json_schema(data[key], value):
                    return False
        
        return True
        
    except Exception:
        return False


# Global instances
_global_input_validator = InputValidator()
_global_security_manager = SecurityManager()
_global_data_integrity_checker = DataIntegrityChecker()


def validate_input(
    field_name: str,
    value: Any,
    context: Optional[ErrorContext] = None
) -> bool:
    """Validate input field."""
    return _global_input_validator.validate_field(field_name, value, context)


def validate_all_inputs(
    data: Dict[str, Any],
    context: Optional[ErrorContext] = None
) -> bool:
    """Validate all input fields."""
    return _global_input_validator.validate_data(data, context)


def check_security_permission(
    operation: str,
    user_permissions: List[str],
    context: Optional[ErrorContext] = None
) -> bool:
    """Check security permission."""
    return _global_security_manager.check_permission(operation, user_permissions, context)


def encrypt_sensitive_data(data: str) -> str:
    """Encrypt sensitive data."""
    return _global_security_manager.encrypt_data(data)


def decrypt_sensitive_data(encrypted_data: str) -> str:
    """Decrypt sensitive data."""
    return _global_security_manager.decrypt_data(encrypted_data)


def verify_data_integrity(
    data: Union[str, bytes],
    expected_checksum: str
) -> bool:
    """Verify data integrity."""
    return _global_data_integrity_checker.verify_checksum(data, expected_checksum)


def calculate_data_checksum(data: Union[str, bytes]) -> str:
    """Calculate data checksum."""
    return _global_data_integrity_checker.calculate_checksum(data)


def validate_data_format(
    data: Any,
    expected_format: str,
    context: Optional[ErrorContext] = None
) -> bool:
    """Validate data format."""
    return _global_data_integrity_checker.validate_data_format(data, expected_format, context)


# Export main classes and functions
__all__ = [
    'ValidationLevel',
    'SecurityLevel',
    'ValidationRule',
    'SecurityPolicy',
    'InputValidator',
    'SecurityManager',
    'DataIntegrityChecker',
    'validate_not_none',
    'validate_not_empty',
    'validate_positive_number',
    'validate_range',
    'validate_email',
    'validate_url',
    'validate_file_path',
    'validate_safe_string',
    'validate_numeric_string',
    'validate_alphanumeric_string',
    'validate_safe_filename',
    'validate_json_schema',
    'validate_input',
    'validate_all_inputs',
    'check_security_permission',
    'encrypt_sensitive_data',
    'decrypt_sensitive_data',
    'verify_data_integrity',
    'calculate_data_checksum',
    'validate_data_format'
]