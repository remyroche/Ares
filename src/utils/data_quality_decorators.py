"""
Data Quality Decorators

This module provides decorators for automatic data quality validation
at each pipeline step, with special attention to NaN, infinite, and constant values.
"""

import functools
import logging
from typing import Any, Callable, Dict, Optional, Union
import numpy as np
import pandas as pd

try:
    from src.utils.logger import system_logger
except ImportError:
    system_logger = logging.getLogger("DataQualityDecorators")


def validate_data_quality_at_step(
    step_name: str,
    validate_input: bool = True,
    validate_output: bool = True,
    check_nan: bool = True,
    check_infinite: bool = True,
    check_constant: bool = True,
    check_correlation: bool = True,
    max_nan_ratio: float = 0.0,    # 0% NaN (zero tolerance)
    max_infinite_count: int = 0,   # 0 infinite values (zero tolerance)
    min_unique_values: int = 2,
    max_correlation_threshold: float = 0.95,
    fail_on_issues: bool = False,
    log_issues: bool = True
):
    """
    Decorator to validate data quality at each pipeline step.
    
    Args:
        step_name: Name of the step for logging
        validate_input: Whether to validate input data
        validate_output: Whether to validate output data
        check_nan: Whether to check for NaN values
        check_infinite: Whether to check for infinite values
        check_constant: Whether to check for constant features
        check_correlation: Whether to check for high correlations
        max_nan_ratio: Maximum allowed ratio of NaN values
        max_infinite_ratio: Maximum allowed ratio of infinite values
        min_unique_values: Minimum unique values for non-constant features
        max_correlation_threshold: Maximum correlation threshold
        fail_on_issues: Whether to fail the step on quality issues
        log_issues: Whether to log quality issues
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            logger = system_logger.getChild(f"DataQuality.{step_name}")
            
            # Validate input data if requested
            if validate_input:
                logger.info(f"🔍 Validating input data quality for {step_name}...")
                input_issues = _validate_data_quality(
                    args, kwargs, "input", logger,
                    check_nan, check_infinite, check_constant, check_correlation,
                    max_nan_ratio, max_infinite_count, min_unique_values, max_correlation_threshold
                )
                
                if input_issues and log_issues:
                    logger.warning(f"⚠️ Input data quality issues found in {step_name}:")
                    for issue in input_issues[:5]:  # Show first 5 issues
                        logger.warning(f"   - {issue}")
                    if len(input_issues) > 5:
                        logger.warning(f"   ... and {len(input_issues) - 5} more issues")
                
                if input_issues and fail_on_issues:
                    raise ValueError(f"Input data quality validation failed for {step_name}: {input_issues}")
            
            # Execute the function
            try:
                result = await func(*args, **kwargs)
            except Exception as e:
                logger.error(f"❌ {step_name} execution failed: {e}")
                raise
            
            # Validate output data if requested
            if validate_output and result is not None:
                logger.info(f"🔍 Validating output data quality for {step_name}...")
                output_issues = _validate_data_quality(
                    [result], {}, "output", logger,
                    check_nan, check_infinite, check_constant, check_correlation,
                    max_nan_ratio, max_infinite_count, min_unique_values, max_correlation_threshold
                )
                
                if output_issues and log_issues:
                    logger.warning(f"⚠️ Output data quality issues found in {step_name}:")
                    for issue in output_issues[:5]:  # Show first 5 issues
                        logger.warning(f"   - {issue}")
                    if len(output_issues) > 5:
                        logger.warning(f"   ... and {len(output_issues) - 5} more issues")
                
                if output_issues and fail_on_issues:
                    raise ValueError(f"Output data quality validation failed for {step_name}: {output_issues}")
            
            return result
        
        return wrapper
    return decorator


def _validate_data_quality(
    args: tuple,
    kwargs: dict,
    data_type: str,
    logger: logging.Logger,
    check_nan: bool,
    check_infinite: bool,
    check_constant: bool,
    check_correlation: bool,
    max_nan_ratio: float,
    max_infinite_count: int,
    min_unique_values: int,
    max_correlation_threshold: float
) -> list[str]:
    """
    Validate data quality for given arguments and keyword arguments.
    
    Returns:
        List of quality issues found
    """
    issues = []
    
    # Check all arguments for DataFrames
    for i, arg in enumerate(args):
        if isinstance(arg, pd.DataFrame):
            df_issues = _validate_dataframe_quality(
                arg, f"{data_type}_arg_{i}", logger,
                check_nan, check_infinite, check_constant, check_correlation,
                max_nan_ratio, max_infinite_count, min_unique_values, max_correlation_threshold
            )
            issues.extend(df_issues)
    
    # Check all keyword arguments for DataFrames
    for key, value in kwargs.items():
        if isinstance(value, pd.DataFrame):
            df_issues = _validate_dataframe_quality(
                value, f"{data_type}_kwarg_{key}", logger,
                check_nan, check_infinite, check_constant, check_correlation,
                max_nan_ratio, max_infinite_count, min_unique_values, max_correlation_threshold
            )
            issues.extend(df_issues)
    
    return issues


def _validate_dataframe_quality(
    df: pd.DataFrame,
    df_name: str,
    logger: logging.Logger,
    check_nan: bool,
    check_infinite: bool,
    check_constant: bool,
    check_correlation: bool,
    max_nan_ratio: float,
    max_infinite_count: int,
    min_unique_values: int,
    max_correlation_threshold: float
) -> list[str]:
    """
    Validate DataFrame quality with specific checks.
    
    Returns:
        List of quality issues found
    """
    issues = []
    
    if df.empty:
        issues.append(f"{df_name}: DataFrame is empty")
        return issues
    
    # Check for NaN values (zero tolerance)
    if check_nan:
        nan_counts = df.isnull().sum()
        nan_features = nan_counts[nan_counts > 0].index.tolist()  # Any NaN values
        if nan_features:
            issues.append(f"{df_name}: Features with NaN values (zero tolerance): {nan_features}")
    
    # Check for infinite values (zero tolerance)
    if check_infinite:
        infinite_features = []
        for col in df.select_dtypes(include=[np.number]).columns:
            infinite_count = np.isinf(df[col]).sum()
            if infinite_count > 0:  # Any infinite values
                infinite_features.append(col)
        
        if infinite_features:
            issues.append(f"{df_name}: Features with infinite values (zero tolerance): {infinite_features}")
    
    # Check for constant features (2+ unique values, except boolean)
    if check_constant:
        constant_features = []
        for col in df.columns:
            unique_count = df[col].nunique()
            # Allow boolean features (2 unique values) and binary features
            if unique_count < min_unique_values and not _is_boolean_feature(df[col]):
                constant_features.append(col)
        
        if constant_features:
            issues.append(f"{df_name}: Constant features found: {constant_features}")
    
    # Check for high correlations
    if check_correlation:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 1:
            corr_matrix = df[numeric_cols].corr().abs()
            high_corr_pairs = []
            
            for i in range(len(corr_matrix.columns)):
                for j in range(i + 1, len(corr_matrix.columns)):
                    if corr_matrix.iloc[i, j] > max_correlation_threshold:
                        high_corr_pairs.append((corr_matrix.columns[i], corr_matrix.columns[j]))
            
            if high_corr_pairs:
                issues.append(f"{df_name}: Highly correlated feature pairs: {high_corr_pairs}")
    
    return issues


def _is_boolean_feature(series: pd.Series) -> bool:
    """
    Check if a series represents a boolean feature.
    
    Args:
        series: Pandas series to check
        
    Returns:
        True if the series represents a boolean feature
    """
    # Check if it's already boolean dtype
    if pd.api.types.is_bool_dtype(series):
        return True
    
    # Check if it has exactly 2 unique values that could be boolean
    unique_values = series.dropna().unique()
    if len(unique_values) == 2:
        # Check if values are typical boolean patterns
        unique_set = set(unique_values)
        boolean_patterns = [
            {True, False},
            {1, 0},
            {1.0, 0.0},
            {'True', 'False'},
            {'true', 'false'},
            {'1', '0'},
            {'yes', 'no'},
            {'Y', 'N'},
            {'y', 'n'}
        ]
        
        for pattern in boolean_patterns:
            if unique_set == pattern:
                return True
    
    return False


def validate_step1_quality(func: Callable) -> Callable:
    """Decorator specifically for Step1 data quality validation."""
    return validate_data_quality_at_step(
        "step1_data_collection",
        validate_input=True,
        validate_output=True,
        check_nan=True,
        check_infinite=True,
        check_constant=False,  # Raw data can be constant
        check_correlation=False,  # Raw data correlation is not relevant
        fail_on_issues=False,
        log_issues=True
    )(func)


def validate_step1_5_quality(func: Callable) -> Callable:
    """Decorator specifically for Step1.5 data quality validation."""
    return validate_data_quality_at_step(
        "step1_5_data_converter",
        validate_input=True,
        validate_output=True,
        check_nan=True,
        check_infinite=True,
        check_constant=False,  # Unified data can be constant
        check_correlation=False,  # Unified data correlation is not relevant
        fail_on_issues=False,
        log_issues=True
    )(func)


def validate_step2_quality(func: Callable) -> Callable:
    """Decorator specifically for Step2 data quality validation with special attention to features."""
    return validate_data_quality_at_step(
        "step2_feature_engineering",
        validate_input=True,
        validate_output=True,
        check_nan=True,
        check_infinite=True,
        check_constant=True,  # Features should not be constant
        check_correlation=True,  # Feature correlation is important
        max_nan_ratio=0.0,  # 0% NaN (zero tolerance)
        max_infinite_count=0,  # 0 infinite values (zero tolerance)
        min_unique_values=2,  # 2+ unique values (except boolean)
        max_correlation_threshold=0.95,
        fail_on_issues=False,
        log_issues=True
    )(func)


def log_feature_quality_issues(df: pd.DataFrame, df_name: str, logger: Optional[logging.Logger] = None) -> None:
    """
    Log detailed feature quality issues for a DataFrame.
    
    Args:
        df: DataFrame to check
        df_name: Name of the DataFrame for logging
        logger: Logger to use (defaults to system_logger)
    """
    if logger is None:
        logger = system_logger.getChild("FeatureQualityLogger")
    
    logger.info(f"🔍 Checking feature quality for {df_name}...")
    
    # Check for NaN values (zero tolerance)
    nan_counts = df.isnull().sum()
    nan_features = nan_counts[nan_counts > 0].index.tolist()  # Any NaN values
    if nan_features:
        logger.warning(f"⚠️ {df_name}: Features with NaN values (zero tolerance) ({len(nan_features)}):")
        for feature in nan_features[:10]:  # Show first 10
            nan_count = nan_counts[feature]
            nan_ratio = nan_count / len(df) * 100
            logger.warning(f"   - {feature}: {nan_count} NaN values ({nan_ratio:.1f}%)")
        if len(nan_features) > 10:
            logger.warning(f"   ... and {len(nan_features) - 10} more features with NaN values")
    
    # Check for infinite values (zero tolerance)
    infinite_features = []
    for col in df.select_dtypes(include=[np.number]).columns:
        infinite_count = np.isinf(df[col]).sum()
        if infinite_count > 0:  # Any infinite values
            infinite_features.append((col, infinite_count))
    
    if infinite_features:
        logger.warning(f"⚠️ {df_name}: Features with infinite values (zero tolerance) ({len(infinite_features)}):")
        for feature, count in infinite_features[:10]:  # Show first 10
            logger.warning(f"   - {feature}: {count} infinite values")
        if len(infinite_features) > 10:
            logger.warning(f"   ... and {len(infinite_features) - 10} more features with infinite values")
    
    # Check for constant features (2+ unique values, except boolean)
    constant_features = []
    for col in df.columns:
        unique_count = df[col].nunique()
        if unique_count < 2 and not _is_boolean_feature(df[col]):
            constant_features.append((col, unique_count))
    
    if constant_features:
        logger.warning(f"⚠️ {df_name}: Constant or near-constant features ({len(constant_features)}):")
        for feature, unique_count in constant_features[:10]:  # Show first 10
            logger.warning(f"   - {feature}: {unique_count} unique values")
        if len(constant_features) > 10:
            logger.warning(f"   ... and {len(constant_features) - 10} more constant features")
    
    # Check for high correlations
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 1:
        corr_matrix = df[numeric_cols].corr().abs()
        high_corr_pairs = []
        
        for i in range(len(corr_matrix.columns)):
            for j in range(i + 1, len(corr_matrix.columns)):
                corr_value = corr_matrix.iloc[i, j]
                if corr_value > 0.95:
                    high_corr_pairs.append((corr_matrix.columns[i], corr_matrix.columns[j], corr_value))
        
        if high_corr_pairs:
            logger.warning(f"⚠️ {df_name}: Highly correlated feature pairs ({len(high_corr_pairs)}):")
            for feat1, feat2, corr_value in high_corr_pairs[:5]:  # Show first 5
                logger.warning(f"   - {feat1} ↔ {feat2}: {corr_value:.3f}")
            if len(high_corr_pairs) > 5:
                logger.warning(f"   ... and {len(high_corr_pairs) - 5} more highly correlated pairs")
    
    # Summary
    total_issues = len(nan_features) + len(infinite_features) + len(constant_features) + len(high_corr_pairs)
    if total_issues == 0:
        logger.info(f"✅ {df_name}: No feature quality issues detected")
    else:
        logger.warning(f"⚠️ {df_name}: Total feature quality issues: {total_issues}")


# Convenience function for quick validation
def quick_validate_features(df: pd.DataFrame, df_name: str = "DataFrame") -> Dict[str, Any]:
    """
    Quick validation of features with summary statistics.
    
    Args:
        df: DataFrame to validate
        df_name: Name of the DataFrame
        
    Returns:
        Dictionary with validation results
    """
    results = {
        "df_name": df_name,
        "shape": df.shape,
        "total_features": len(df.columns),
        "total_samples": len(df),
        "issues": {
            "nan_features": [],
            "infinite_features": [],
            "constant_features": [],
            "high_correlation_pairs": []
        },
        "summary": {
            "nan_count": 0,
            "infinite_count": 0,
            "constant_count": 0,
            "high_correlation_count": 0
        }
    }
    
    # Check for NaN values (zero tolerance)
    nan_counts = df.isnull().sum()
    nan_features = nan_counts[nan_counts > 0].index.tolist()  # Any NaN values
    results["issues"]["nan_features"] = nan_features
    results["summary"]["nan_count"] = len(nan_features)
    
    # Check for infinite values (zero tolerance)
    infinite_features = []
    for col in df.select_dtypes(include=[np.number]).columns:
        infinite_count = np.isinf(df[col]).sum()
        if infinite_count > 0:  # Any infinite values
            infinite_features.append(col)
    results["issues"]["infinite_features"] = infinite_features
    results["summary"]["infinite_count"] = len(infinite_features)
    
    # Check for constant features (2+ unique values, except boolean)
    constant_features = []
    for col in df.columns:
        unique_count = df[col].nunique()
        if unique_count < 2 and not _is_boolean_feature(df[col]):
            constant_features.append(col)
    results["issues"]["constant_features"] = constant_features
    results["summary"]["constant_count"] = len(constant_features)
    
    # Check for high correlations
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    high_corr_pairs = []
    if len(numeric_cols) > 1:
        corr_matrix = df[numeric_cols].corr().abs()
        for i in range(len(corr_matrix.columns)):
            for j in range(i + 1, len(corr_matrix.columns)):
                if corr_matrix.iloc[i, j] > 0.95:
                    high_corr_pairs.append((corr_matrix.columns[i], corr_matrix.columns[j]))
    results["issues"]["high_correlation_pairs"] = high_corr_pairs
    results["summary"]["high_correlation_count"] = len(high_corr_pairs)
    
    return results
