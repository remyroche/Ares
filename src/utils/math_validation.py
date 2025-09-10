"""
Mathematical validation utilities to prevent division by zero and other mathematical errors.
"""

import math
import numpy as np
from typing import Any, Union, Optional
from functools import wraps
import logging

logger = logging.getLogger(__name__)


class MathValidationError(Exception):
    """Custom exception for mathematical validation errors."""
    pass


def safe_divide(numerator: Union[float, int], denominator: Union[float, int], 
                default: float = 0.0, epsilon: float = 1e-10) -> float:
    """
    Safely divide two numbers, preventing division by zero.
    
    Args:
        numerator: The numerator
        denominator: The denominator
        default: Default value to return if denominator is zero or very small
        epsilon: Minimum threshold for denominator to be considered valid
        
    Returns:
        float: Safe division result or default value
    """
    if abs(denominator) < epsilon:
        logger.warning(f"Division by zero or very small number prevented: {numerator} / {denominator}")
        return default
    
    try:
        result = numerator / denominator
        if not math.isfinite(result):
            logger.warning(f"Non-finite result from division: {numerator} / {denominator} = {result}")
            return default
        return result
    except (ZeroDivisionError, OverflowError, ValueError) as e:
        logger.warning(f"Mathematical error in division: {e}")
        return default


def safe_log(value: Union[float, int], base: Union[float, int] = math.e, 
             default: float = 0.0, epsilon: float = 1e-10) -> float:
    """
    Safely calculate logarithm, preventing log of zero or negative numbers.
    
    Args:
        value: The value to take logarithm of
        base: The logarithm base (default: e)
        default: Default value to return if value is invalid
        epsilon: Minimum threshold for value to be considered valid
        
    Returns:
        float: Safe logarithm result or default value
    """
    if value <= epsilon:
        logger.warning(f"Logarithm of zero or negative number prevented: log_{base}({value})")
        return default
    
    try:
        if base == math.e:
            result = math.log(value)
        else:
            result = math.log(value) / math.log(base)
        
        if not math.isfinite(result):
            logger.warning(f"Non-finite result from logarithm: log_{base}({value}) = {result}")
            return default
        return result
    except (ValueError, OverflowError) as e:
        logger.warning(f"Mathematical error in logarithm: {e}")
        return default


def safe_sqrt(value: Union[float, int], default: float = 0.0) -> float:
    """
    Safely calculate square root, preventing sqrt of negative numbers.
    
    Args:
        value: The value to take square root of
        default: Default value to return if value is negative
        
    Returns:
        float: Safe square root result or default value
    """
    if value < 0:
        logger.warning(f"Square root of negative number prevented: sqrt({value})")
        return default
    
    try:
        result = math.sqrt(value)
        if not math.isfinite(result):
            logger.warning(f"Non-finite result from square root: sqrt({value}) = {result}")
            return default
        return result
    except (ValueError, OverflowError) as e:
        logger.warning(f"Mathematical error in square root: {e}")
        return default


def safe_power(base: Union[float, int], exponent: Union[float, int], 
               default: float = 1.0) -> float:
    """
    Safely calculate power, preventing overflow and invalid operations.
    
    Args:
        base: The base number
        exponent: The exponent
        default: Default value to return if operation is invalid
        
    Returns:
        float: Safe power result or default value
    """
    try:
        result = math.pow(base, exponent)
        if not math.isfinite(result):
            logger.warning(f"Non-finite result from power: {base}^{exponent} = {result}")
            return default
        return result
    except (ValueError, OverflowError) as e:
        logger.warning(f"Mathematical error in power: {e}")
        return default


def validate_finite(value: Any, name: str = "value") -> float:
    """
    Validate that a value is finite (not NaN or infinite).
    
    Args:
        value: The value to validate
        name: Name of the value for error messages
        
    Returns:
        float: The validated finite value
        
    Raises:
        MathValidationError: If value is not finite
    """
    try:
        float_val = float(value)
        if not math.isfinite(float_val):
            raise MathValidationError(f"{name} is not finite: {value}")
        return float_val
    except (ValueError, TypeError) as e:
        raise MathValidationError(f"{name} cannot be converted to float: {value}") from e


def validate_positive(value: Any, name: str = "value", epsilon: float = 1e-10) -> float:
    """
    Validate that a value is positive.
    
    Args:
        value: The value to validate
        name: Name of the value for error messages
        epsilon: Minimum threshold for value to be considered positive
        
    Returns:
        float: The validated positive value
        
    Raises:
        MathValidationError: If value is not positive
    """
    float_val = validate_finite(value, name)
    if float_val < epsilon:
        raise MathValidationError(f"{name} must be positive: {value}")
    return float_val


def validate_range(value: Any, min_val: float, max_val: float, 
                  name: str = "value") -> float:
    """
    Validate that a value is within a specified range.
    
    Args:
        value: The value to validate
        min_val: Minimum allowed value
        max_val: Maximum allowed value
        name: Name of the value for error messages
        
    Returns:
        float: The validated value within range
        
    Raises:
        MathValidationError: If value is outside range
    """
    float_val = validate_finite(value, name)
    if not (min_val <= float_val <= max_val):
        raise MathValidationError(f"{name} must be between {min_val} and {max_val}: {value}")
    return float_val


def safe_kelly_calculation(win_probability: float, win_amount: float, 
                          loss_amount: float, kelly_multiplier: float = 1.0) -> float:
    """
    Safely calculate Kelly criterion position size.
    
    Args:
        win_probability: Probability of winning (0-1)
        win_amount: Amount won on successful trade
        loss_amount: Amount lost on failed trade
        kelly_multiplier: Multiplier to apply to Kelly fraction
        
    Returns:
        float: Safe Kelly position size
    """
    try:
        # Validate inputs
        win_prob = validate_range(win_probability, 0.0, 1.0, "win_probability")
        win_amt = validate_positive(win_amount, "win_amount")
        loss_amt = validate_positive(loss_amount, "loss_amount")
        multiplier = validate_positive(kelly_multiplier, "kelly_multiplier")
        
        # Calculate Kelly fraction
        expected_value = win_prob * win_amt - (1 - win_prob) * loss_amt
        if expected_value <= 0:
            logger.warning("Negative expected value in Kelly calculation, returning 0")
            return 0.0
        
        kelly_fraction = expected_value / win_amt
        kelly_fraction = max(0.0, min(1.0, kelly_fraction))  # Clamp to [0, 1]
        
        return kelly_fraction * multiplier
        
    except MathValidationError as e:
        logger.warning(f"Kelly calculation validation error: {e}")
        return 0.0
    except Exception as e:
        logger.warning(f"Unexpected error in Kelly calculation: {e}")
        return 0.0


def safe_weighted_average(values: list[float], weights: list[float], 
                         default: float = 0.0) -> float:
    """
    Safely calculate weighted average, preventing division by zero.
    
    Args:
        values: List of values
        weights: List of weights
        default: Default value if calculation fails
        
    Returns:
        float: Safe weighted average
    """
    try:
        if len(values) != len(weights):
            raise MathValidationError("Values and weights must have same length")
        
        if not values:
            return default
        
        # Convert to numpy arrays for easier handling
        values_array = np.array(values, dtype=float)
        weights_array = np.array(weights, dtype=float)
        
        # Check for finite values
        if not np.all(np.isfinite(values_array)) or not np.all(np.isfinite(weights_array)):
            logger.warning("Non-finite values in weighted average calculation")
            return default
        
        # Calculate weighted sum and total weight
        weighted_sum = np.sum(values_array * weights_array)
        total_weight = np.sum(weights_array)
        
        return safe_divide(weighted_sum, total_weight, default)
        
    except Exception as e:
        logger.warning(f"Error in weighted average calculation: {e}")
        return default


def safe_percentage_change(old_value: float, new_value: float, 
                          default: float = 0.0) -> float:
    """
    Safely calculate percentage change, preventing division by zero.
    
    Args:
        old_value: Original value
        new_value: New value
        default: Default value if calculation fails
        
    Returns:
        float: Safe percentage change
    """
    try:
        old_val = validate_finite(old_value, "old_value")
        new_val = validate_finite(new_value, "new_value")
        
        return safe_divide(new_val - old_val, old_val, default) * 100
        
    except MathValidationError as e:
        logger.warning(f"Percentage change validation error: {e}")
        return default
    except Exception as e:
        logger.warning(f"Unexpected error in percentage change: {e}")
        return default


def validate_correlation_matrix(matrix: np.ndarray, name: str = "correlation_matrix") -> np.ndarray:
    """
    Validate that a matrix is a valid correlation matrix.
    
    Args:
        matrix: The matrix to validate
        name: Name of the matrix for error messages
        
    Returns:
        np.ndarray: The validated correlation matrix
        
    Raises:
        MathValidationError: If matrix is not a valid correlation matrix
    """
    if not isinstance(matrix, np.ndarray):
        raise MathValidationError(f"{name} must be a numpy array")
    
    if matrix.ndim != 2:
        raise MathValidationError(f"{name} must be 2-dimensional")
    
    if matrix.shape[0] != matrix.shape[1]:
        raise MathValidationError(f"{name} must be square")
    
    if not np.all(np.isfinite(matrix)):
        raise MathValidationError(f"{name} contains non-finite values")
    
    # Check diagonal elements are 1
    if not np.allclose(np.diag(matrix), 1.0):
        raise MathValidationError(f"{name} diagonal elements must be 1")
    
    # Check symmetry
    if not np.allclose(matrix, matrix.T):
        raise MathValidationError(f"{name} must be symmetric")
    
    # Check eigenvalues are non-negative (positive semi-definite)
    # Use enhanced matrix operations if available
    try:
        from .ml_common.matrix_operations import get_enhanced_matrix_operations
        enhanced_ops = get_enhanced_matrix_operations()
        eigenvalues, _ = enhanced_ops.eigendecomposition(matrix, use_gpu=False)
    except ImportError:
        eigenvalues = np.linalg.eigvals(matrix)
    
    if np.any(eigenvalues < -1e-10):  # Small tolerance for numerical errors
        raise MathValidationError(f"{name} is not positive semi-definite")
    
    return matrix


def safe_matrix_inverse(matrix: np.ndarray, default: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Safely calculate matrix inverse, handling singular matrices.
    
    Args:
        matrix: The matrix to invert
        default: Default matrix to return if inversion fails
        
    Returns:
        np.ndarray: Safe matrix inverse or default
    """
    try:
        if not isinstance(matrix, np.ndarray):
            raise MathValidationError("Matrix must be a numpy array")
        
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            raise MathValidationError("Matrix must be square")
        
        if not np.all(np.isfinite(matrix)):
            raise MathValidationError("Matrix contains non-finite values")
        
        # Use enhanced matrix operations if available
        try:
            from .ml_common.matrix_operations import get_enhanced_matrix_operations
            enhanced_ops = get_enhanced_matrix_operations()
            
            # Check condition number using enhanced operations
            cond_num = enhanced_ops.condition_number(matrix, use_gpu=False)
            if cond_num > 1e12:  # Very ill-conditioned
                logger.warning(f"Matrix is ill-conditioned (condition number: {cond_num})")
                if default is not None:
                    return default
                else:
                    raise MathValidationError("Matrix is too ill-conditioned to invert safely")
            
            inverse = enhanced_ops.matrix_inverse(matrix, use_gpu=False)
            
        except ImportError:
            # Fallback to standard numpy operations
            cond_num = np.linalg.cond(matrix)
            if cond_num > 1e12:  # Very ill-conditioned
                logger.warning(f"Matrix is ill-conditioned (condition number: {cond_num})")
                if default is not None:
                    return default
                else:
                    raise MathValidationError("Matrix is too ill-conditioned to invert safely")
            
            inverse = np.linalg.inv(matrix)
        
        if not np.all(np.isfinite(inverse)):
            logger.warning("Matrix inverse contains non-finite values")
            if default is not None:
                return default
            else:
                raise MathValidationError("Matrix inverse contains non-finite values")
        
        return inverse
        
    except np.linalg.LinAlgError as e:
        logger.warning(f"Linear algebra error in matrix inversion: {e}")
        if default is not None:
            return default
        else:
            raise MathValidationError(f"Cannot invert matrix: {e}") from e
    except Exception as e:
        logger.warning(f"Unexpected error in matrix inversion: {e}")
        if default is not None:
            return default
        else:
            raise MathValidationError(f"Unexpected error in matrix inversion: {e}") from e


def math_safe(func):
    """
    Decorator to make mathematical functions safe by catching and handling errors.
    
    Usage:
        @math_safe
        def risky_calculation(x, y):
            return x / y
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except (ZeroDivisionError, OverflowError, ValueError, MathValidationError) as e:
            logger.warning(f"Mathematical error in {func.__name__}: {e}")
            return 0.0  # Default safe return value
        except Exception as e:
            logger.error(f"Unexpected error in {func.__name__}: {e}")
            return 0.0
    
    return wrapper


# Example usage and testing
if __name__ == "__main__":
    # Test safe division
    print(f"Safe division: {safe_divide(10, 0)}")  # Should return 0.0
    print(f"Safe division: {safe_divide(10, 2)}")  # Should return 5.0
    
    # Test safe logarithm
    print(f"Safe log: {safe_log(0)}")  # Should return 0.0
    print(f"Safe log: {safe_log(10)}")  # Should return ~2.3
    
    # Test Kelly calculation
    print(f"Kelly: {safe_kelly_calculation(0.6, 100, 50)}")  # Should return positive value
    print(f"Kelly: {safe_kelly_calculation(0.3, 100, 50)}")  # Should return 0.0 (negative EV)
    
    # Test weighted average
    print(f"Weighted avg: {safe_weighted_average([1, 2, 3], [1, 1, 1])}")  # Should return 2.0
    print(f"Weighted avg: {safe_weighted_average([1, 2, 3], [0, 0, 0])}")  # Should return 0.0