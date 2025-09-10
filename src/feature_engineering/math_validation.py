"""
Math Validation Functions

This module provides common math validation functions used across the feature engineering package.
"""

import numpy as np

def safe_divide(a, b, default=0.0):
    """Safely divide two numbers, returning default if division by zero or error occurs."""
    try:
        return a / b if b != 0 else default
    except:
        return default

def safe_log(x, default=0.0):
    """Safely calculate logarithm, returning default if x <= 0 or error occurs."""
    try:
        return np.log(x) if x > 0 else default
    except:
        return default

def safe_sqrt(x, default=0.0):
    """Safely calculate square root, returning default if x < 0 or error occurs."""
    try:
        return np.sqrt(x) if x >= 0 else default
    except:
        return default

def validate_positive(value, name="value"):
    """Validate that a value is positive."""
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")
    return value

def validate_range(value, min_val, max_val, name="value"):
    """Validate that a value is within a range."""
    if not (min_val <= value <= max_val):
        raise ValueError(f"{name} must be between {min_val} and {max_val}, got {value}")
    return value

class MathValidationError(Exception):
    """Math validation error."""
    pass
