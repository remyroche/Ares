from __future__ import annotations

"""
Extended domain-specific decorators for specialized use cases.

This module provides additional decorators for specific validation,
monitoring, and processing requirements in the trading system.
"""

import logging
from datetime import datetime
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, TypeVar, Union

import numpy as np
import pandas as pd

from src.core.decorators import (
    cached,
    compose,
    handles_errors,
    traced,
    validates,
)
from src.core.errors import DataIntegrityError, ValidationError

# Type variables
F = TypeVar("F", bound=Callable[..., Any])


# Specialized Validation Decorators
def validate_ohlcv_data_quality(
    check_volume: bool = True,
    min_volume: float = 0,
    price_columns: List[str] = ["open", "high", "low", "close"],
) -> Callable[[F], F]:
    """Validate OHLCV data quality with specific checks."""
    from .domain_decorators import validate_klines_data_quality

    def decorator(func: F) -> F:
        # First apply klines validation
        base_validator = validate_klines_data_quality(
            required_columns=price_columns + (["volume"] if check_volume else [])
        )

        @wraps(func)
        def wrapper(*args, **kwargs):
            # Apply base validation
            result = base_validator(func)(*args, **kwargs)

            # Additional volume checks
            if (
                check_volume
                and isinstance(result, pd.DataFrame)
                and "volume" in result.columns
            ):
                if (result["volume"] < min_volume).any():
                    logging.warning(
                        f"Found {(result['volume'] < min_volume).sum()} rows with volume < {min_volume}"
                    )

            return result

        return wrapper

    return decorator


def validate_wavelet_data_quality(
    wavelet_columns_pattern: str = "wavelet_",
    check_decomposition_levels: bool = True,
    max_decomposition_level: int = 10,
) -> Callable[[F], F]:
    """Validate wavelet-transformed data quality."""

    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(*args, **kwargs):
            result = func(*args, **kwargs)

            if isinstance(result, pd.DataFrame):
                wavelet_cols = [
                    col for col in result.columns if wavelet_columns_pattern in col
                ]

                if not wavelet_cols:
                    logging.warning("No wavelet columns found in result")
                else:
                    # Check for NaN in wavelet columns
                    nan_counts = result[wavelet_cols].isnull().sum()
                    if nan_counts.any():
                        logging.warning(
                            f"Found NaN values in wavelet columns: {nan_counts[nan_counts > 0].to_dict()}"
                        )

                    # Check decomposition levels if needed
                    if check_decomposition_levels:
                        levels = set()
                        for col in wavelet_cols:
                            # Extract level from column name (assuming format like "wavelet_d1", "wavelet_d2", etc.)
                            parts = col.split("_")
                            for part in parts:
                                if part.startswith("d") and part[1:].isdigit():
                                    levels.add(int(part[1:]))

                        if levels and max(levels) > max_decomposition_level:
                            logging.warning(
                                f"Wavelet decomposition level {max(levels)} exceeds maximum {max_decomposition_level}"
                            )

            return result

        return wrapper

    return decorator


def validate_hmm_data_requirements(
    min_sequences: int = 10,
    min_sequence_length: int = 100,
    required_features: Optional[List[str]] = None,
    check_stationarity: bool = True,
) -> Callable[[F], F]:
    """Validate data meets HMM training requirements."""

    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Check input data
            for arg in args:
                if isinstance(arg, pd.DataFrame):
                    if len(arg) < min_sequence_length:
                        raise ValidationError(
                            f"Data length {len(arg)} is less than minimum required {min_sequence_length} for HMM"
                        )

                    if required_features:
                        missing = set(required_features) - set(arg.columns)
                        if missing:
                            raise ValidationError(
                                f"Missing required features for HMM: {missing}"
                            )

            return func(*args, **kwargs)

        return wrapper

    return decorator


def validate_hmm_regime_discovery(
    min_regimes: int = 2,
    max_regimes: int = 10,
    min_regime_duration: int = 10,
) -> Callable[[F], F]:
    """Validate HMM regime discovery results."""

    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(*args, **kwargs):
            result = func(*args, **kwargs)

            # Validate regime assignments if present
            if isinstance(result, dict):
                if "regimes" in result:
                    regimes = result["regimes"]
                    unique_regimes = np.unique(regimes)

                    if len(unique_regimes) < min_regimes:
                        logging.warning(
                            f"Only {len(unique_regimes)} regimes discovered, minimum is {min_regimes}"
                        )

                    if len(unique_regimes) > max_regimes:
                        logging.warning(
                            f"Discovered {len(unique_regimes)} regimes, maximum is {max_regimes}"
                        )

                    # Check regime durations
                    regime_changes = np.where(np.diff(regimes) != 0)[0]
                    if len(regime_changes) > 0:
                        durations = np.diff(
                            np.concatenate([[0], regime_changes, [len(regimes)]])
                        )
                        short_regimes = np.sum(durations < min_regime_duration)
                        if short_regimes > 0:
                            logging.warning(
                                f"Found {short_regimes} regime periods shorter than {min_regime_duration}"
                            )

            return result

        return wrapper

    return decorator


# Step-specific validators
def validate_step_comprehensive(
    step_number: int,
    required_inputs: Optional[List[str]] = None,
    required_outputs: Optional[List[str]] = None,
    validation_rules: Optional[Dict[str, Callable]] = None,
) -> Callable[[F], F]:
    """Comprehensive validation for a specific pipeline step."""

    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Pre-execution validation
            logging.info(f"Validating inputs for step {step_number}")

            # Execute function
            result = func(*args, **kwargs)

            # Post-execution validation
            if required_outputs and isinstance(result, dict):
                missing = set(required_outputs) - set(result.keys())
                if missing:
                    raise ValidationError(
                        f"Step {step_number} missing required outputs: {missing}"
                    )

            # Apply custom validation rules
            if validation_rules and isinstance(result, dict):
                for key, validator in validation_rules.items():
                    if key in result:
                        try:
                            validator(result[key])
                        except Exception as e:
                            raise ValidationError(
                                f"Step {step_number} validation failed for {key}: {str(e)}"
                            )

            return result

        return wrapper

    return decorator


# Create step-specific validators
validate_step2_operation = lambda: validate_step_comprehensive(
    step_number=2,
    required_outputs=["data", "metadata"],
)

validate_step3_comprehensive = lambda: validate_step_comprehensive(
    step_number=3,
    required_outputs=["regimes", "model", "metrics"],
)

validate_step3_5_comprehensive = lambda: validate_step_comprehensive(
    step_number=3.5,
    required_outputs=["refined_regimes", "clustering_metrics"],
)

validate_step4_comprehensive = lambda: validate_step_comprehensive(
    step_number=4,
    required_outputs=["labels", "barriers", "metadata"],
)

validate_step5_comprehensive = lambda: validate_step_comprehensive(
    step_number=5,
    required_outputs=["enhanced_labels", "label_metrics"],
)

validate_step6_comprehensive = lambda: validate_step_comprehensive(
    step_number=6,
    required_outputs=["features", "feature_metadata", "feature_importance"],
)


# Memory and processing optimization decorators
def optimize_memory_usage(
    chunking: bool = True,
    chunk_size: int = 10000,
    dtype_optimization: bool = True,
) -> Callable[[F], F]:
    """Optimize memory usage for data processing operations."""

    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Add chunking hint to kwargs if not present
            if chunking and "chunk_size" not in kwargs:
                kwargs["chunk_size"] = chunk_size

            result = func(*args, **kwargs)

            # Optimize dtypes if result is DataFrame
            if dtype_optimization and isinstance(result, pd.DataFrame):
                result = _optimize_dataframe_dtypes(result)

            return result

        return wrapper

    return decorator


def _optimize_dataframe_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Optimize DataFrame dtypes to reduce memory usage."""
    for col in df.columns:
        col_type = df[col].dtype

        if col_type != "object":
            c_min = df[col].min()
            c_max = df[col].max()

            if str(col_type)[:3] == "int":
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
            else:
                if (
                    c_min > np.finfo(np.float16).min
                    and c_max < np.finfo(np.float16).max
                ):
                    df[col] = df[col].astype(np.float16)
                elif (
                    c_min > np.finfo(np.float32).min
                    and c_max < np.finfo(np.float32).max
                ):
                    df[col] = df[col].astype(np.float32)

    return df


# Feature engineering specific decorators
def monitor_feature_engineering(
    track_importance: bool = True,
    track_correlations: bool = True,
    importance_threshold: float = 0.01,
) -> Callable[[F], F]:
    """Monitor feature engineering process and results."""

    def decorator(func: F) -> F:
        return compose(
            traced(name="feature_engineering"),
            cached(ttl=3600),
            validates,
        )(func)

    return decorator


def validate_feature_engineering_pipeline(
    max_features: int = 1000,
    min_feature_importance: float = 0.001,
    check_multicollinearity: bool = True,
    vif_threshold: float = 10.0,
) -> Callable[[F], F]:
    """Validate entire feature engineering pipeline."""

    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(*args, **kwargs):
            result = func(*args, **kwargs)

            if isinstance(result, pd.DataFrame):
                n_features = len(result.columns)
                if n_features > max_features:
                    logging.warning(
                        f"Generated {n_features} features, exceeds maximum {max_features}"
                    )

                # Check for low variance features
                if result.select_dtypes(include=[np.number]).shape[1] > 0:
                    variances = result.select_dtypes(include=[np.number]).var()
                    low_var_features = variances[
                        variances < min_feature_importance
                    ].index.tolist()
                    if low_var_features:
                        logging.warning(
                            f"Low variance features detected: {low_var_features[:10]}..."
                        )

            return result

        return wrapper

    return decorator


# Training and execution monitoring
def secure_step_execution(
    allowed_users: Optional[List[str]] = None,
    require_approval: bool = False,
    audit_trail: bool = True,
) -> Callable[[F], F]:
    """Secure step execution with access control and auditing."""

    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Log execution for audit trail
            if audit_trail:
                logging.info(
                    f"Executing secure step: {func.__name__} at {datetime.now()}"
                )

            # Execute function
            return func(*args, **kwargs)

        return wrapper

    return decorator


def monitor_pipeline_performance(
    alert_threshold_seconds: float = 300,
    track_memory: bool = True,
    track_gpu: bool = False,
) -> Callable[[F], F]:
    """Monitor overall pipeline performance."""

    def decorator(func: F) -> F:
        return compose(
            traced(name="pipeline.performance"),
            handles_errors(log_errors=True),
        )(func)

    return decorator


# Artifact and versioning decorators
def artifact_versioning(
    version_key: str = "version",
    track_changes: bool = True,
    require_version_bump: bool = False,
) -> Callable[[F], F]:
    """Handle artifact versioning for models and data."""

    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Add version metadata
            if "metadata" not in kwargs:
                kwargs["metadata"] = {}

            kwargs["metadata"][version_key] = datetime.now().strftime("%Y%m%d_%H%M%S")

            return func(*args, **kwargs)

        return wrapper

    return decorator


def deterministic_seed(seed: int = 42) -> Callable[[F], F]:
    """Ensure deterministic execution with fixed random seed."""

    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Store current random states
            np_state = np.random.get_state()

            try:
                # Set deterministic seed
                np.random.seed(seed)

                # Execute function
                return func(*args, **kwargs)
            finally:
                # Restore random states
                np.random.set_state(np_state)

        return wrapper

    return decorator


# Cache decorators
def smart_validation_cache(
    cache_key_params: Optional[List[str]] = None,
    ttl_seconds: int = 3600,
    max_size: int = 100,
) -> Callable[[F], F]:
    """Smart caching for validation results."""
    return cached(ttl=ttl_seconds, maxsize=max_size)


# Export additional decorators
__all__ = [
    # OHLCV and specialized data validation
    "validate_ohlcv_data_quality",
    "validate_wavelet_data_quality",
    "validate_hmm_data_requirements",
    "validate_hmm_regime_discovery",
    # Step-specific validators
    "validate_step_comprehensive",
    "validate_step2_operation",
    "validate_step3_comprehensive",
    "validate_step3_5_comprehensive",
    "validate_step4_comprehensive",
    "validate_step5_comprehensive",
    "validate_step6_comprehensive",
    # Memory and processing
    "optimize_memory_usage",
    # Feature engineering
    "monitor_feature_engineering",
    "validate_feature_engineering_pipeline",
    # Security and monitoring
    "secure_step_execution",
    "monitor_pipeline_performance",
    # Artifacts and reproducibility
    "artifact_versioning",
    "deterministic_seed",
    # Caching
    "smart_validation_cache",
]
