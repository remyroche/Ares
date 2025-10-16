"""
Comprehensive Input Validation and Sanitization Utilities

This module provides robust input validation, sanitization, and type checking
utilities for the Ares project with enhanced error handling and security features.
"""

import re
import logging
from typing import Any, Dict, List, Optional, Union, Callable, Type, Pattern, Tuple
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import json
import ipaddress
import urllib.parse

logger = logging.getLogger(__name__)

class ValidationSeverity(Enum):
    """Validation severity levels."""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

class ValidationResult:
    """Result of input validation operation."""

    def __init__(self,
                 is_valid: bool,
                 value: Any = None,
                 errors: List[str] = None,
                 warnings: List[str] = None,
                 sanitized_value: Any = None,
                 severity: ValidationSeverity = ValidationSeverity.INFO):
        """Initialize validation result.

        Args:
            is_valid: Whether validation passed
            value: Original value that was validated
            errors: List of error messages
            warnings: List of warning messages
            sanitized_value: Sanitized version of the value
            severity: Severity level of validation issues
        """
        self.is_valid = is_valid
        self.value = value
        self.errors = errors or []
        self.warnings = warnings or []
        self.sanitized_value = sanitized_value
        self.severity = severity

    def add_error(self, message: str, severity: ValidationSeverity = ValidationSeverity.ERROR):
        """Add an error message."""
        self.errors.append(message)
        if severity.value > self.severity.value:
            self.severity = severity

    def add_warning(self, message: str):
        """Add a warning message."""
        self.warnings.append(message)
        if self.severity == ValidationSeverity.DEBUG:
            self.severity = ValidationSeverity.INFO

    def has_issues(self) -> bool:
        """Check if validation has any issues."""
        return bool(self.errors or self.warnings)

    def get_highest_severity(self) -> ValidationSeverity:
        """Get the highest severity level."""
        return self.severity

@dataclass
class ValidationRule:
    """Configuration for a validation rule."""
    name: str
    rule_type: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    error_message: Optional[str] = None
    warning_message: Optional[str] = None
    severity: ValidationSeverity = ValidationSeverity.ERROR
    enabled: bool = True

class InputValidator:
    """Comprehensive input validation and sanitization system."""

    def __init__(self, strict_mode: bool = False, enable_logging: bool = True):
        """Initialize input validator.

        Args:
            strict_mode: Whether to use strict validation (fail on warnings)
            enable_logging: Whether to log validation activities
        """
        self.strict_mode = strict_mode
        self.enable_logging = enable_logging
        self.validation_rules: Dict[str, List[ValidationRule]] = {}
        self.logger = logging.getLogger(f"{__name__}.InputValidator")

        # Register default validation rules
        self._register_default_rules()

    def _register_default_rules(self):
        """Register default validation rules."""
        # String validation rules
        self.register_rule("string", ValidationRule(
            name="non_empty",
            rule_type="required",
            error_message="String cannot be empty"
        ))

        self.register_rule("string", ValidationRule(
            name="max_length",
            rule_type="max_length",
            parameters={"max_length": 1000},
            error_message="String too long"
        ))

        self.register_rule("string", ValidationRule(
            name="min_length",
            rule_type="min_length",
            parameters={"min_length": 1},
            error_message="String too short"
        ))

        # Numeric validation rules
        self.register_rule("numeric", ValidationRule(
            name="finite",
            rule_type="finite",
            error_message="Value must be finite"
        ))

        self.register_rule("numeric", ValidationRule(
            name="positive",
            rule_type="positive",
            error_message="Value must be positive"
        ))

        # File path validation rules
        self.register_rule("filepath", ValidationRule(
            name="exists",
            rule_type="exists",
            error_message="File path does not exist"
        ))

        self.register_rule("filepath", ValidationRule(
            name="readable",
            rule_type="readable",
            error_message="File path is not readable"
        ))

        self.register_rule("filepath", ValidationRule(
            name="writable",
            rule_type="writable",
            error_message="File path is not writable"
        ))

    def register_rule(self, data_type: str, rule: ValidationRule):
        """Register a validation rule for a data type.

        Args:
            data_type: Type of data to validate ('string', 'numeric', 'filepath', etc.)
            rule: Validation rule configuration
        """
        if data_type not in self.validation_rules:
            self.validation_rules[data_type] = []

        self.validation_rules[data_type].append(rule)

        if self.enable_logging:
            self.logger.debug(f"Registered validation rule '{rule.name}' for type '{data_type}'")

    def validate_string(self,
                       value: Any,
                       max_length: Optional[int] = None,
                       min_length: Optional[int] = None,
                       pattern: Optional[Union[str, Pattern]] = None,
                       allow_unicode: bool = True,
                       strip_whitespace: bool = True) -> ValidationResult:
        """Validate and sanitize string input.

        Args:
            value: Value to validate
            max_length: Maximum allowed length
            min_length: Minimum allowed length
            pattern: Regex pattern to match
            allow_unicode: Whether to allow unicode characters
            strip_whitespace: Whether to strip leading/trailing whitespace

        Returns:
            ValidationResult with validation status and sanitized value
        """
        result = ValidationResult(is_valid=True, value=value)

        # Convert to string if not already
        if value is None:
            result.add_error("Value cannot be None", ValidationSeverity.ERROR)
            result.is_valid = False
            return result

        try:
            string_value = str(value)
        except Exception as e:
            result.add_error(f"Cannot convert value to string: {e}", ValidationSeverity.ERROR)
            result.is_valid = False
            return result

        # Sanitize string
        original_value = string_value

        if strip_whitespace:
            string_value = string_value.strip()

        if not allow_unicode:
            # Remove non-ASCII characters
            string_value = string_value.encode('ascii', 'ignore').decode('ascii')

        result.sanitized_value = string_value

        # Apply validation rules
        if min_length is not None and len(string_value) < min_length:
            result.add_error(
                f"String length {len(string_value)} is below minimum {min_length}",
                ValidationSeverity.ERROR
            )
            result.is_valid = False

        if max_length is not None and len(string_value) > max_length:
            result.add_error(
                f"String length {len(string_value)} exceeds maximum {max_length}",
                ValidationSeverity.ERROR
            )
            result.is_valid = False

        if pattern is not None:
            if isinstance(pattern, str):
                pattern = re.compile(pattern)

            if not pattern.match(string_value):
                result.add_error(f"String does not match required pattern", ValidationSeverity.ERROR)
                result.is_valid = False

        # Log changes if value was modified
        if string_value != original_value and self.enable_logging:
            self.logger.debug(f"Sanitized string: '{original_value}' -> '{string_value}'")

        return result

    def validate_numeric(self,
                        value: Any,
                        min_val: Optional[float] = None,
                        max_val: Optional[float] = None,
                        allow_nan: bool = False,
                        allow_inf: bool = False,
                        integer_only: bool = False) -> ValidationResult:
        """Validate and sanitize numeric input.

        Args:
            value: Value to validate
            min_val: Minimum allowed value
            max_val: Maximum allowed value
            allow_nan: Whether to allow NaN values
            allow_inf: Whether to allow infinite values
            integer_only: Whether value must be an integer

        Returns:
            ValidationResult with validation status and sanitized value
        """
        result = ValidationResult(is_valid=True, value=value)

        if value is None:
            result.add_error("Value cannot be None", ValidationSeverity.ERROR)
            result.is_valid = False
            return result

        try:
            # Try to convert to numeric
            if integer_only:
                numeric_value = int(float(value))
                result.sanitized_value = numeric_value
            else:
                numeric_value = float(value)
                result.sanitized_value = numeric_value

        except (ValueError, TypeError) as e:
            result.add_error(f"Cannot convert value to numeric: {e}", ValidationSeverity.ERROR)
            result.is_valid = False
            return result

        # Check for special values
        if not allow_nan and (numeric_value != numeric_value):  # NaN check
            result.add_error("NaN values not allowed", ValidationSeverity.ERROR)
            result.is_valid = False

        if not allow_inf:
            if numeric_value == float('inf'):
                result.add_error("Positive infinity not allowed", ValidationSeverity.ERROR)
                result.is_valid = False
            elif numeric_value == float('-inf'):
                result.add_error("Negative infinity not allowed", ValidationSeverity.ERROR)
                result.is_valid = False

        # Range checks
        if min_val is not None and numeric_value < min_val:
            result.add_error(f"Value {numeric_value} is below minimum {min_val}", ValidationSeverity.ERROR)
            result.is_valid = False

        if max_val is not None and numeric_value > max_val:
            result.add_error(f"Value {numeric_value} exceeds maximum {max_val}", ValidationSeverity.ERROR)
            result.is_valid = False

        return result

    def validate_filepath(self,
                         value: Any,
                         must_exist: bool = False,
                         must_be_readable: bool = False,
                         must_be_writable: bool = False,
                         must_be_file: bool = False,
                         must_be_directory: bool = False) -> ValidationResult:
        """Validate and sanitize file path input.

        Args:
            value: Path value to validate
            must_exist: Whether path must exist
            must_be_readable: Whether path must be readable
            must_be_writable: Whether path must be writable
            must_be_file: Whether path must be a file
            must_be_directory: Whether path must be a directory

        Returns:
            ValidationResult with validation status and sanitized path
        """
        result = ValidationResult(is_valid=True, value=value)

        if value is None:
            result.add_error("Path cannot be None", ValidationSeverity.ERROR)
            result.is_valid = False
            return result

        try:
            path = Path(str(value)).resolve()
            result.sanitized_value = path
        except Exception as e:
            result.add_error(f"Invalid path format: {e}", ValidationSeverity.ERROR)
            result.is_valid = False
            return result

        # Existence checks
        if must_exist and not path.exists():
            result.add_error(f"Path does not exist: {path}", ValidationSeverity.ERROR)
            result.is_valid = False

        if path.exists():
            # File/directory type checks
            if must_be_file and not path.is_file():
                result.add_error(f"Path is not a file: {path}", ValidationSeverity.ERROR)
                result.is_valid = False

            if must_be_directory and not path.is_dir():
                result.add_error(f"Path is not a directory: {path}", ValidationSeverity.ERROR)
                result.is_valid = False

            # Permission checks
            if must_be_readable and not os.access(path, os.R_OK):
                result.add_error(f"Path is not readable: {path}", ValidationSeverity.ERROR)
                result.is_valid = False

            if must_be_writable and not os.access(path, os.W_OK):
                result.add_error(f"Path is not writable: {path}", ValidationSeverity.ERROR)
                result.is_valid = False

        return result

    def validate_json(self,
                     value: Any,
                     required_keys: Optional[List[str]] = None,
                     schema: Optional[Dict[str, Any]] = None) -> ValidationResult:
        """Validate JSON data structure.

        Args:
            value: JSON data to validate
            required_keys: List of required keys
            schema: Schema definition for validation

        Returns:
            ValidationResult with validation status and parsed data
        """
        result = ValidationResult(is_valid=True, value=value)

        if value is None:
            result.add_error("JSON value cannot be None", ValidationSeverity.ERROR)
            result.is_valid = False
            return result

        try:
            # Parse JSON if it's a string
            if isinstance(value, str):
                json_data = json.loads(value)
            else:
                json_data = value

            result.sanitized_value = json_data

        except json.JSONDecodeError as e:
            result.add_error(f"Invalid JSON format: {e}", ValidationSeverity.ERROR)
            result.is_valid = False
            return result
        except Exception as e:
            result.add_error(f"Error parsing JSON: {e}", ValidationSeverity.ERROR)
            result.is_valid = False
            return result

        # Validate required keys
        if required_keys and isinstance(json_data, dict):
            missing_keys = set(required_keys) - set(json_data.keys())
            if missing_keys:
                result.add_error(f"Missing required keys: {sorted(missing_keys)}", ValidationSeverity.ERROR)
                result.is_valid = False

        # Schema validation (basic implementation)
        if schema and isinstance(json_data, dict):
            for key, expected_type in schema.items():
                if key in json_data:
                    actual_value = json_data[key]
                    if not isinstance(actual_value, expected_type):
                        result.add_warning(f"Key '{key}' has type {type(actual_value).__name__}, expected {expected_type.__name__}")

        return result

    def validate_email(self, value: Any) -> ValidationResult:
        """Validate email address format.

        Args:
            value: Email address to validate

        Returns:
            ValidationResult with validation status
        """
        result = ValidationResult(is_valid=True, value=value)

        if value is None:
            result.add_error("Email cannot be None", ValidationSeverity.ERROR)
            result.is_valid = False
            return result

        # Basic email regex pattern
        email_pattern = re.compile(
            r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        )

        string_result = self.validate_string(value, max_length=254)
        if not string_result.is_valid:
            result.errors.extend(string_result.errors)
            result.is_valid = False
            return result

        email = string_result.sanitized_value

        if not email_pattern.match(email):
            result.add_error(f"Invalid email format: {email}", ValidationSeverity.ERROR)
            result.is_valid = False

        result.sanitized_value = email.lower()  # Normalize to lowercase

        return result

    def validate_url(self, value: Any) -> ValidationResult:
        """Validate URL format.

        Args:
            value: URL to validate

        Returns:
            ValidationResult with validation status and parsed URL
        """
        result = ValidationResult(is_valid=True, value=value)

        if value is None:
            result.add_error("URL cannot be None", ValidationSeverity.ERROR)
            result.is_valid = False
            return result

        string_result = self.validate_string(value, max_length=2048)
        if not string_result.is_valid:
            result.errors.extend(string_result.errors)
            result.is_valid = False
            return result

        url_string = string_result.sanitized_value

        try:
            parsed_url = urllib.parse.urlparse(url_string)
            if not parsed_url.scheme or not parsed_url.netloc:
                result.add_error(f"Invalid URL format: {url_string}", ValidationSeverity.ERROR)
                result.is_valid = False
            else:
                result.sanitized_value = parsed_url

        except Exception as e:
            result.add_error(f"Error parsing URL: {e}", ValidationSeverity.ERROR)
            result.is_valid = False

        return result

    def validate_ip_address(self, value: Any) -> ValidationResult:
        """Validate IP address format.

        Args:
            value: IP address to validate

        Returns:
            ValidationResult with validation status and parsed IP
        """
        result = ValidationResult(is_valid=True, value=value)

        if value is None:
            result.add_error("IP address cannot be None", ValidationSeverity.ERROR)
            result.is_valid = False
            return result

        string_result = self.validate_string(value)
        if not string_result.is_valid:
            result.errors.extend(string_result.errors)
            result.is_valid = False
            return result

        ip_string = string_result.sanitized_value

        try:
            # Try IPv4 first, then IPv6
            try:
                ip_addr = ipaddress.IPv4Address(ip_string)
            except ipaddress.AddressValueError:
                ip_addr = ipaddress.IPv6Address(ip_string)

            result.sanitized_value = ip_addr

        except ipaddress.AddressValueError as e:
            result.add_error(f"Invalid IP address format: {e}", ValidationSeverity.ERROR)
            result.is_valid = False

        return result

    def validate_dataframe(self,
                          value: Any,
                          required_columns: Optional[List[str]] = None,
                          min_rows: Optional[int] = None,
                          max_rows: Optional[int] = None) -> ValidationResult:
        """Validate pandas DataFrame.

        Args:
            value: DataFrame to validate
            required_columns: List of required column names
            min_rows: Minimum number of rows required
            max_rows: Maximum number of rows allowed

        Returns:
            ValidationResult with validation status
        """
        result = ValidationResult(is_valid=True, value=value)

        if value is None:
            result.add_error("DataFrame cannot be None", ValidationSeverity.ERROR)
            result.is_valid = False
            return result

        try:
            import pandas as pd

            if not isinstance(value, pd.DataFrame):
                result.add_error(f"Value is not a DataFrame, got {type(value)}", ValidationSeverity.ERROR)
                result.is_valid = False
                return result

            df = value

            # Check for empty DataFrame
            if df.empty:
                result.add_warning("DataFrame is empty")

            # Row count validation
            if min_rows is not None and len(df) < min_rows:
                result.add_error(f"DataFrame has {len(df)} rows, minimum required: {min_rows}", ValidationSeverity.ERROR)
                result.is_valid = False

            if max_rows is not None and len(df) > max_rows:
                result.add_error(f"DataFrame has {len(df)} rows, maximum allowed: {max_rows}", ValidationSeverity.ERROR)
                result.is_valid = False

            # Column validation
            if required_columns:
                missing_columns = set(required_columns) - set(df.columns)
                if missing_columns:
                    result.add_error(f"Missing required columns: {sorted(missing_columns)}", ValidationSeverity.ERROR)
                    result.is_valid = False

            result.sanitized_value = df

        except ImportError:
            result.add_error("pandas not available for DataFrame validation", ValidationSeverity.ERROR)
            result.is_valid = False
        except Exception as e:
            result.add_error(f"Error validating DataFrame: {e}", ValidationSeverity.ERROR)
            result.is_valid = False

        return result

    def validate_custom(self,
                       value: Any,
                       validator_func: Callable[[Any], bool],
                       error_message: str = "Custom validation failed") -> ValidationResult:
        """Apply custom validation function.

        Args:
            value: Value to validate
            validator_func: Custom validation function
            error_message: Error message if validation fails

        Returns:
            ValidationResult with validation status
        """
        result = ValidationResult(is_valid=True, value=value)

        try:
            if not validator_func(value):
                result.add_error(error_message, ValidationSeverity.ERROR)
                result.is_valid = False
        except Exception as e:
            result.add_error(f"Error in custom validator: {e}", ValidationSeverity.ERROR)
            result.is_valid = False

        return result

class InputSanitizer:
    """Comprehensive input sanitization utilities."""

    @staticmethod
    def sanitize_string(value: str,
                       max_length: Optional[int] = None,
                       allow_unicode: bool = True,
                       strip_whitespace: bool = True,
                       remove_html: bool = False) -> str:
        """Sanitize string input with multiple cleaning options.

        Args:
            value: String to sanitize
            max_length: Maximum length to truncate to
            allow_unicode: Whether to preserve unicode characters
            strip_whitespace: Whether to strip whitespace
            remove_html: Whether to remove HTML tags

        Returns:
            Sanitized string
        """
        if not isinstance(value, str):
            value = str(value)

        # Strip whitespace
        if strip_whitespace:
            value = value.strip()

        # Remove HTML tags if requested
        if remove_html:
            value = re.sub(r'<[^>]+>', '', value)

        # Truncate if too long
        if max_length and len(value) > max_length:
            value = value[:max_length]

        # Remove non-ASCII if not allowed
        if not allow_unicode:
            value = value.encode('ascii', 'ignore').decode('ascii')

        return value

    @staticmethod
    def sanitize_numeric(value: Any,
                        default: float = 0.0,
                        min_val: Optional[float] = None,
                        max_val: Optional[float] = None) -> float:
        """Sanitize numeric input with range constraints.

        Args:
            value: Value to sanitize
            default: Default value if conversion fails
            min_val: Minimum allowed value
            max_val: Maximum allowed value

        Returns:
            Sanitized numeric value
        """
        try:
            numeric_value = float(value)

            if min_val is not None and numeric_value < min_val:
                return min_val

            if max_val is not None and numeric_value > max_val:
                return max_val

            return numeric_value

        except (ValueError, TypeError):
            return default

    @staticmethod
    def sanitize_filepath(value: Any, must_exist: bool = False) -> Optional[Path]:
        """Sanitize file path input.

        Args:
            value: Path to sanitize
            must_exist: Whether path must exist

        Returns:
            Sanitized Path object or None if invalid
        """
        try:
            if value is None:
                return None

            path = Path(str(value)).resolve()

            if must_exist and not path.exists():
                return None

            return path

        except Exception:
            return None

    @staticmethod
    def sanitize_json(value: Any, default: Any = None) -> Any:
        """Sanitize JSON input.

        Args:
            value: JSON data to sanitize
            default: Default value if parsing fails

        Returns:
            Parsed JSON data or default value
        """
        try:
            if isinstance(value, str):
                return json.loads(value)
            return value
        except (json.JSONDecodeError, TypeError):
            return default

# Global validator and sanitizer instances
input_validator = InputValidator()
input_sanitizer = InputSanitizer()

def validate_and_sanitize(data_type: str, value: Any, **kwargs) -> ValidationResult:
    """Convenience function to validate and sanitize input in one call.

    Args:
        data_type: Type of validation to perform
        value: Value to validate and sanitize
        **kwargs: Validation parameters

    Returns:
        ValidationResult with validation status and sanitized value
    """
    if data_type == "string":
        return input_validator.validate_string(value, **kwargs)
    elif data_type == "numeric":
        return input_validator.validate_numeric(value, **kwargs)
    elif data_type == "filepath":
        return input_validator.validate_filepath(value, **kwargs)
    elif data_type == "json":
        return input_validator.validate_json(value, **kwargs)
    elif data_type == "email":
        return input_validator.validate_email(value)
    elif data_type == "url":
        return input_validator.validate_url(value)
    elif data_type == "ip":
        return input_validator.validate_ip_address(value)
    elif data_type == "dataframe":
        return input_validator.validate_dataframe(value, **kwargs)
    else:
        return ValidationResult(
            is_valid=False,
            value=value,
            errors=[f"Unknown data type for validation: {data_type}"],
            severity=ValidationSeverity.ERROR
        )
