"""
Decorators and utilities for temporal alignment validation.

This module provides decorators and utilities to ensure temporal consistency
when working with financial time series data across the pipeline.
"""

import functools
import pandas as pd
from typing import Callable, Any
import logging

logger = logging.getLogger(__name__)


def ensure_datetime_index(func: Callable) -> Callable:
    """
    Decorator to ensure function returns DataFrame with DatetimeIndex.

    This decorator validates that any DataFrame returned by the decorated
    function has a DatetimeIndex. If not, it logs a warning.

    Usage:
        @ensure_datetime_index
        def load_data(self) -> pd.DataFrame:
            return self._get_artifact("data")

    Args:
        func: Function to decorate

    Returns:
        Decorated function
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        result = func(*args, **kwargs)

        if isinstance(result, pd.DataFrame):
            if not isinstance(result.index, pd.DatetimeIndex):
                logger.warning(
                    f"⚠️ Function {func.__name__} returned DataFrame without DatetimeIndex. "
                    f"Found: {type(result.index).__name__}. "
                    f"This may cause temporal alignment issues."
                )
            else:
                logger.debug(
                    f"✅ Function {func.__name__} returned DataFrame with DatetimeIndex "
                    f"({len(result)} rows, {result.index.min()} to {result.index.max()})"
                )

        return result

    return wrapper


def validate_temporal_alignment(func: Callable) -> Callable:
    """
    Decorator to validate temporal alignment of DataFrames passed to function.

    This decorator checks that all DataFrame arguments have DatetimeIndex and
    that their indices match exactly. If not, it raises a ValueError with
    detailed information about the misalignment.

    Usage:
        @validate_temporal_alignment
        def combine_features(self, df1: pd.DataFrame, df2: pd.DataFrame):
            return pd.concat([df1, df2], axis=1)

    Args:
        func: Function to decorate

    Returns:
        Decorated function

    Raises:
        ValueError: If DataFrames are not temporally aligned
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        # Extract DataFrame arguments
        dataframes = [
            arg for arg in args if isinstance(arg, pd.DataFrame)
        ] + [
            val for val in kwargs.values() if isinstance(val, pd.DataFrame)
        ]

        if len(dataframes) >= 2:
            # Check all have DatetimeIndex
            for i, df in enumerate(dataframes):
                if not isinstance(df.index, pd.DatetimeIndex):
                    raise ValueError(
                        f"Argument {i} in {func.__name__} must have DatetimeIndex. "
                        f"Found: {type(df.index).__name__}. "
                        f"All DataFrames must have DatetimeIndex for temporal alignment."
                    )

            # Check indices match
            reference_idx = dataframes[0].index
            for i, df in enumerate(dataframes[1:], start=1):
                if not reference_idx.equals(df.index):
                    common = len(reference_idx.intersection(df.index))
                    only_ref = len(reference_idx.difference(df.index))
                    only_other = len(df.index.difference(reference_idx))

                    raise ValueError(
                        f"⚠️ Temporal misalignment in {func.__name__}!\n"
                        f"DataFrame 0: {len(dataframes[0])} rows, "
                        f"range: {dataframes[0].index.min()} to {dataframes[0].index.max()}\n"
                        f"DataFrame {i}: {len(df)} rows, "
                        f"range: {df.index.min()} to {df.index.max()}\n"
                        f"Common timestamps: {common}\n"
                        f"Only in DataFrame 0: {only_ref}\n"
                        f"Only in DataFrame {i}: {only_other}\n"
                        f"Solution: Use BaseStep._align_to_reference() or .reindex() "
                        f"before passing to {func.__name__}"
                    )

            logger.debug(
                f"✅ Temporal alignment validated for {len(dataframes)} DataFrames "
                f"in {func.__name__}"
            )

        return func(*args, **kwargs)

    return wrapper


def log_temporal_info(func: Callable) -> Callable:
    """
    Decorator to log temporal information about DataFrame inputs and outputs.

    This decorator logs useful information about DataFrames before and after
    function execution, including row counts, date ranges, and index types.

    Usage:
        @log_temporal_info
        def process_data(self, df: pd.DataFrame) -> pd.DataFrame:
            return df.dropna()

    Args:
        func: Function to decorate

    Returns:
        Decorated function
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        # Log input DataFrames
        input_dataframes = [
            arg for arg in args if isinstance(arg, pd.DataFrame)
        ] + [
            val for val in kwargs.values() if isinstance(val, pd.DataFrame)
        ]

        if input_dataframes:
            logger.info(f"📊 Function {func.__name__} - Input DataFrames:")
            for i, df in enumerate(input_dataframes):
                if isinstance(df.index, pd.DatetimeIndex):
                    logger.info(
                        f"  DataFrame {i}: {len(df)} rows, "
                        f"DatetimeIndex {df.index.min()} to {df.index.max()}"
                    )
                else:
                    logger.info(
                        f"  DataFrame {i}: {len(df)} rows, "
                        f"Index type: {type(df.index).__name__}"
                    )

        # Execute function
        result = func(*args, **kwargs)

        # Log output DataFrame
        if isinstance(result, pd.DataFrame):
            if isinstance(result.index, pd.DatetimeIndex):
                logger.info(
                    f"📊 Function {func.__name__} - Output: "
                    f"{len(result)} rows, "
                    f"DatetimeIndex {result.index.min()} to {result.index.max()}"
                )
            else:
                logger.info(
                    f"📊 Function {func.__name__} - Output: "
                    f"{len(result)} rows, "
                    f"Index type: {type(result.index).__name__}"
                )

        return result

    return wrapper


def require_datetime_index(func: Callable) -> Callable:
    """
    Decorator to require that all DataFrame arguments have DatetimeIndex.

    Unlike validate_temporal_alignment, this only checks for DatetimeIndex
    presence, not alignment between DataFrames.

    Usage:
        @require_datetime_index
        def calculate_returns(self, df: pd.DataFrame) -> pd.DataFrame:
            return df.pct_change()

    Args:
        func: Function to decorate

    Returns:
        Decorated function

    Raises:
        ValueError: If any DataFrame argument lacks DatetimeIndex
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        # Extract DataFrame arguments
        dataframes = [
            (i, arg) for i, arg in enumerate(args) if isinstance(arg, pd.DataFrame)
        ] + [
            (key, val) for key, val in kwargs.items() if isinstance(val, pd.DataFrame)
        ]

        # Check all have DatetimeIndex
        for identifier, df in dataframes:
            if not isinstance(df.index, pd.DatetimeIndex):
                raise ValueError(
                    f"Argument '{identifier}' in {func.__name__} must have DatetimeIndex. "
                    f"Found: {type(df.index).__name__}. "
                    f"Use ArtifactManager._ensure_temporal_index() to convert."
                )

        return func(*args, **kwargs)

    return wrapper


# Utility functions for manual validation

def check_temporal_alignment(*dataframes: pd.DataFrame) -> dict:
    """
    Check temporal alignment of multiple DataFrames and return diagnostic info.

    This is a utility function (not a decorator) for programmatic checking
    of temporal alignment.

    Args:
        *dataframes: DataFrames to check

    Returns:
        Dictionary with alignment information:
        - aligned: bool - Whether all DataFrames are aligned
        - issues: list - List of alignment issues found
        - common_index_length: int - Number of common timestamps
        - stats: list - Per-DataFrame statistics

    Example:
        result = check_temporal_alignment(df1, df2, df3)
        if not result['aligned']:
            logger.warning(f"Alignment issues: {result['issues']}")
    """
    if len(dataframes) < 2:
        return {
            'aligned': True,
            'issues': [],
            'common_index_length': len(dataframes[0]) if dataframes else 0,
            'stats': []
        }

    issues = []
    stats = []

    # Check for DatetimeIndex
    for i, df in enumerate(dataframes):
        df_stats = {
            'index': i,
            'length': len(df),
            'has_datetime_index': isinstance(df.index, pd.DatetimeIndex),
            'index_type': type(df.index).__name__
        }

        if isinstance(df.index, pd.DatetimeIndex):
            df_stats['start'] = str(df.index.min())
            df_stats['end'] = str(df.index.max())
            df_stats['frequency'] = str(df.index.inferred_freq)

        stats.append(df_stats)

        if not isinstance(df.index, pd.DatetimeIndex):
            issues.append(
                f"DataFrame {i} does not have DatetimeIndex "
                f"(found: {type(df.index).__name__})"
            )

    # Check index alignment
    if not issues:
        reference_idx = dataframes[0].index
        common_idx = reference_idx

        for i, df in enumerate(dataframes[1:], start=1):
            common_idx = common_idx.intersection(df.index)

            if not reference_idx.equals(df.index):
                only_ref = len(reference_idx.difference(df.index))
                only_other = len(df.index.difference(reference_idx))
                issues.append(
                    f"DataFrame 0 and {i} have different indices: "
                    f"{only_ref} unique to df0, {only_other} unique to df{i}"
                )

        common_index_length = len(common_idx)
    else:
        common_index_length = 0

    return {
        'aligned': len(issues) == 0,
        'issues': issues,
        'common_index_length': common_index_length,
        'stats': stats
    }


def get_common_temporal_index(*dataframes: pd.DataFrame) -> pd.DatetimeIndex:
    """
    Get the common temporal index across multiple DataFrames.

    This finds the intersection of all DatetimeIndex values.

    Args:
        *dataframes: DataFrames with DatetimeIndex

    Returns:
        DatetimeIndex containing only timestamps present in all DataFrames

    Raises:
        ValueError: If any DataFrame lacks DatetimeIndex
    """
    if not dataframes:
        return pd.DatetimeIndex([])

    # Validate all have DatetimeIndex
    for i, df in enumerate(dataframes):
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError(
                f"DataFrame {i} must have DatetimeIndex. "
                f"Found: {type(df.index).__name__}"
            )

    # Find intersection
    common_idx = dataframes[0].index
    for df in dataframes[1:]:
        common_idx = common_idx.intersection(df.index)

    logger.info(
        f"Common temporal index: {len(common_idx)} timestamps "
        f"from {len(dataframes)} DataFrames"
    )

    return common_idx
