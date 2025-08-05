"""
Test validation utilities for addressing EM101/EM102 and TRY003 issues.

This module provides utilities for proper assertion message formatting
and test validation patterns that comply with linting rules.
"""

from typing import Any

from src.utils.error_handler import format_assertion_message, safe_assertion


class TestValidator:
    """Utility class for test validation with proper error message formatting."""

    @staticmethod
    def assert_validation_result(
        is_valid: bool,
        errors: list[str],
        context: str = "",
        *,
        expected_valid: bool = True,
    ) -> None:
        """
        Assert validation result with proper message formatting.

        Args:
            is_valid: Whether the validation passed
            errors: List of validation errors
            context: Context for the validation
            expected_valid: Whether the validation should have passed

        Raises:
            AssertionError: If validation result doesn't match expectation
        """
        if expected_valid and not is_valid:
            # Assign message to variable to address EM101/EM102
            error_message = format_assertion_message(
                expected="valid",
                actual="invalid",
                context=context,
                message_template="Should be valid: {actual}",
            )
            safe_assertion(False, error_message, context=context)

        if not expected_valid and is_valid:
            # Assign message to variable to address EM101/EM102
            error_message = format_assertion_message(
                expected="invalid",
                actual="valid",
                context=context,
                message_template="Should not be valid",
            )
            safe_assertion(False, error_message, context=context)

    @staticmethod
    def assert_error_count(
        errors: list[str],
        expected_count: int,
        context: str = "",
    ) -> None:
        """
        Assert error count with proper message formatting.

        Args:
            errors: List of errors
            expected_count: Expected number of errors
            context: Context for the assertion

        Raises:
            AssertionError: If error count doesn't match expectation
        """
        actual_count = len(errors)
        if actual_count != expected_count:
            # Assign message to variable to address EM101/EM102
            error_message = format_assertion_message(
                expected=expected_count,
                actual=actual_count,
                context=context,
                message_template="Expected {expected} errors, got {actual}",
            )
            safe_assertion(False, error_message, context=context)

    @staticmethod
    def assert_threshold_value(
        actual_value: Any,
        expected_value: Any,
        threshold_name: str,
        context: str = "",
    ) -> None:
        """
        Assert threshold value with proper message formatting.

        Args:
            actual_value: Actual threshold value
            expected_value: Expected threshold value
            threshold_name: Name of the threshold being checked
            context: Context for the assertion

        Raises:
            AssertionError: If threshold value doesn't match expectation
        """
        if actual_value != expected_value:
            # Assign message to variable to address EM101/EM102
            error_message = format_assertion_message(
                expected=expected_value,
                actual=actual_value,
                context=f"{context}.{threshold_name}",
            )
            safe_assertion(False, error_message, context=context)

    @staticmethod
    def assert_required_field(
        data: dict,
        field_name: str,
        context: str = "",
    ) -> None:
        """
        Assert required field exists with proper message formatting.

        Args:
            data: Dictionary to check
            field_name: Name of required field
            context: Context for the assertion

        Raises:
            AssertionError: If required field is missing
        """
        if field_name not in data:
            # Assign message to variable to address EM101/EM102
            error_message = f"'{field_name}' not found in {context}"
            safe_assertion(False, error_message, context=context)

    @staticmethod
    def assert_step_rules(
        rules: dict,
        expected_can_skip: bool,
        expected_failure_action: str,
        context: str = "",
    ) -> None:
        """
        Assert step rules with proper message formatting.

        Args:
            rules: Step rules dictionary
            expected_can_skip: Expected can_skip value
            expected_failure_action: Expected failure_action value
            context: Context for the assertion

        Raises:
            AssertionError: If step rules don't match expectations
        """
        actual_can_skip = rules.get("can_skip")
        if actual_can_skip != expected_can_skip:
            # Assign message to variable to address EM101/EM102
            error_message = format_assertion_message(
                expected=expected_can_skip,
                actual=actual_can_skip,
                context=f"{context}.can_skip",
                message_template="Expected can_skip to be {expected}, got {actual}",
            )
            safe_assertion(False, error_message, context=context)

        actual_failure_action = rules.get("failure_action", "")
        if actual_failure_action != expected_failure_action:
            # Assign message to variable to address EM101/EM102
            error_message = format_assertion_message(
                expected=expected_failure_action,
                actual=actual_failure_action,
                context=f"{context}.failure_action",
                message_template="Expected failure_action to be {expected}, got {actual}",
            )
            safe_assertion(False, error_message, context=context)


def create_validation_message(
    operation: str,
    errors: list[str],
    context: str = "",
) -> str:
    """
    Create validation message with proper formatting.

    Args:
        operation: Operation that was validated
        errors: List of validation errors
        context: Context for the validation

    Returns:
        Formatted validation message
    """
    # Assign message to variable to address EM101/EM102
    base_message = f"{operation} validation failed"

    if errors:
        error_details = "; ".join(errors)
        message = f"{base_message}: {error_details}"
    else:
        message = base_message

    if context:
        return f"{context}: {message}"
    return message


def create_threshold_message(
    expected: Any,
    actual: Any,
    threshold_name: str,
    context: str = "",
) -> str:
    """
    Create threshold comparison message with proper formatting.

    Args:
        expected: Expected threshold value
        actual: Actual threshold value
        threshold_name: Name of the threshold
        context: Context for the comparison

    Returns:
        Formatted threshold message
    """
    # Assign message to variable to address EM101/EM102
    message = format_assertion_message(
        expected=expected,
        actual=actual,
        context=f"{context}.{threshold_name}",
    )
    return message


def validate_imports() -> tuple[bool, list[str]]:
    """
    Validate imports with proper error handling.

    Returns:
        Tuple of (is_valid, errors)
    """
    errors = []

    try:
        # Import validation logic here
        # This is a placeholder - actual implementation would check imports
        return True, errors
    except Exception as e:
        # Assign error message to variable to address EM101/EM102
        error_message = f"Import validation failed: {str(e)}"
        errors.append(error_message)
        return False, errors


def validate_data_format(data: Any) -> tuple[bool, list[str]]:
    """
    Validate data format with proper error handling.

    Args:
        data: Data to validate

    Returns:
        Tuple of (is_valid, errors)
    """
    errors = []

    try:
        # Data format validation logic here
        # This is a placeholder - actual implementation would validate data
        return True, errors
    except Exception as e:
        # Assign error message to variable to address EM101/EM102
        error_message = f"Data format validation failed: {str(e)}"
        errors.append(error_message)
        return False, errors


def validate_data_quality(data: Any) -> tuple[bool, list[str]]:
    """
    Validate data quality with proper error handling.

    Args:
        data: Data to validate

    Returns:
        Tuple of (is_valid, errors)
    """
    errors = []

    try:
        # Data quality validation logic here
        # This is a placeholder - actual implementation would validate quality
        return True, errors
    except Exception as e:
        # Assign error message to variable to address EM101/EM102
        error_message = f"Data quality validation failed: {str(e)}"
        errors.append(error_message)
        return False, errors


def validate_file_paths(path: str) -> tuple[bool, list[str]]:
    """
    Validate file paths with proper error handling.

    Args:
        path: Path to validate

    Returns:
        Tuple of (is_valid, errors)
    """
    errors = []

    try:
        # File path validation logic here
        # This is a placeholder - actual implementation would validate paths
        return True, errors
    except Exception as e:
        # Assign error message to variable to address EM101/EM102
        error_message = f"File path validation failed: {str(e)}"
        errors.append(error_message)
        return False, errors
