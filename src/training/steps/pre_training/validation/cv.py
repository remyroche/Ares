"""Cross-validation utilities for pre-training steps."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Iterator, Mapping, Optional

import pandas as pd

# Import core utilities
try:
    from ....utils.tprint import tprint, tprint_debug, tprint_error, tprint_info
    from ....utils.common_operations import (
        validate_dataframe, validate_dataframe_columns, safe_divide,
        validate_positive, validate_range, format_bytes, timed_operation
    )
    from ....utils.ml_common.optimization.grid_utils import generate_parameter_grid
    from ....utils.hardware.m1_memory_optimizer import get_m1_memory_optimizer
    from ....utils.hardware.m1_cpu_optimizer import get_m1_cpu_optimizer
    # Import matrix operations for correlation analysis and batch processing
    from ....utils.matrix_operations import (
        safe_correlation_matrix, matrix_correlation_analysis, batch_matrix_multiply,
        optimize_dataframe, get_unified_matrix_operations, get_vectorized_processing_core
    )
    MATRIX_OPERATIONS_AVAILABLE = True
except ImportError as e:
    # Fallback imports if utils are not available
    MATRIX_OPERATIONS_AVAILABLE = False
    def tprint(*args, **kwargs): pass
    def tprint_debug(*args, **kwargs): pass
    def tprint_error(*args, **kwargs): pass
    def tprint_info(*args, **kwargs): pass
    def validate_dataframe(df): return isinstance(df, pd.DataFrame) and not df.empty
    def validate_dataframe_columns(df, cols): return set(cols).issubset(set(df.columns))
    def safe_divide(a, b, default=0.0): return a / b if b != 0 else default
    def validate_positive(value, name="value"): return value if value > 0 else 0.0
    def validate_range(value, min_val=None, max_val=None, name="value"): return value
    def format_bytes(bytes_value): return f"{bytes_value}B"
    def timed_operation(func): return func
    def generate_parameter_grid(params): return [{}]
    def get_m1_memory_optimizer(): return None
    def get_m1_cpu_optimizer(): return None
    # Matrix operations fallbacks
    def safe_correlation_matrix(df): return df.corr() if hasattr(df, 'corr') else None
    def matrix_correlation_analysis(*args, **kwargs): return {}
    def batch_matrix_multiply(*args, **kwargs): return None
    def optimize_dataframe(df): return df
    def get_unified_matrix_operations(): return None
    def get_vectorized_processing_core(): return None


@dataclass
class WalkForwardFold:
    """Container describing a single walk-forward split."""

    fold: int
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame

    def to_mapping(self) -> Dict[str, pd.DataFrame]:
        """Return a lightweight mapping used by downstream validation helpers."""

        return {"train": self.train, "validation": self.validation, "test": self.test}


def _ensure_datetime_index(df: pd.DataFrame, *, name: str) -> pd.DataFrame:
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError(f"{name} must have a DatetimeIndex for walk-forward CV")
    if not df.index.is_monotonic_increasing:
        df = df.sort_index()
    return df


@timed_operation
def purged_walk_forward_cv(
    data: pd.DataFrame,
    *,
    n_splits: int,
    train_ratio: float,
    validation_ratio: float,
    test_ratio: float,
    purge_window_hours: float,
    embargo_window_hours: float,
) -> Iterator[WalkForwardFold]:
    """Yield purged, embargoed walk-forward splits.

    The function creates an expanding-window walk-forward plan. Each fold uses all
    observations prior to the validation window for training while respecting the
    configured purge and embargo windows.
    """

    tprint_info(f"Starting purged walk-forward CV with {n_splits} splits")

    # Validate inputs using core utilities
    n_splits = validate_positive(n_splits, "n_splits")
    train_ratio = validate_range(train_ratio, 0.0, 1.0, "train_ratio")
    validation_ratio = validate_range(validation_ratio, 0.0, 1.0, "validation_ratio")
    test_ratio = validate_range(test_ratio, 0.0, 1.0, "test_ratio")

    if not validate_dataframe(data):
        raise ValueError("Input data must be a valid pandas DataFrame")

    working = _ensure_datetime_index(data.copy(), name="data")
    n_samples = len(working)
    if n_samples == 0:
        tprint_warning("No samples in data for CV")
        return iter(())

    purge_delta = pd.Timedelta(hours=max(purge_window_hours, 0.0))
    embargo_delta = pd.Timedelta(hours=max(embargo_window_hours, 0.0))

    min_train_size = max(1, int(round(n_samples * max(train_ratio, 0.0))))
    tprint_debug(f"CV setup: {n_samples} samples, min_train_size={min_train_size}, "
                 f"purge={purge_delta}, embargo={embargo_delta}")

    if min_train_size >= n_samples:
        # Not enough data to create validation windows – yield a single fold
        tprint_warning("Insufficient data for validation windows, yielding single fold")
        yield WalkForwardFold(
            fold=0,
            train=working.copy(),
            validation=working.iloc[0:0].copy(),
            test=working.iloc[0:0].copy(),
        )
        return

    remaining = n_samples - min_train_size
    per_fold_window = max(1, -(-remaining // n_splits))  # ceil division

    ratio_denominator = max(validation_ratio + test_ratio, 1e-6)
    validation_fraction = safe_divide(validation_ratio, ratio_denominator, 0.5)

    start_idx = min_train_size
    memory_optimizer = get_m1_memory_optimizer()
    cpu_optimizer = get_m1_cpu_optimizer()

    # Initialize matrix operations for correlation analysis and optimization
    matrix_ops = get_unified_matrix_operations() if MATRIX_OPERATIONS_AVAILABLE else None
    vectorized_core = get_vectorized_processing_core() if MATRIX_OPERATIONS_AVAILABLE else None

    tprint_debug(f"Matrix operations available: {MATRIX_OPERATIONS_AVAILABLE}")

    for fold in range(n_splits):
        if start_idx >= n_samples:
            tprint_debug(f"Fold {fold}: reached end of data at index {start_idx}")
            break

        window_end = min(n_samples, start_idx + per_fold_window)
        if window_end - start_idx <= 0:
            tprint_debug(f"Fold {fold}: empty window")
            break

        window_length = window_end - start_idx
        validation_length = max(1, int(round(window_length * validation_fraction)))
        if validation_length >= window_length:
            validation_length = window_length - 1

        validation_slice = working.iloc[start_idx : start_idx + validation_length]
        if validation_slice.empty:
            tprint_debug(f"Fold {fold}: empty validation slice")
            start_idx = window_end
            continue

        test_slice = working.iloc[start_idx + validation_length : window_end]

        validation_start = validation_slice.index[0]
        train_end_time = validation_start - purge_delta
        train_slice = working.loc[:train_end_time]

        if len(train_slice) < min_train_size:
            tprint_debug(f"Fold {fold}: insufficient training data ({len(train_slice)} < {min_train_size})")
            start_idx = window_end
            continue

        # Optimize memory usage for large datasets
        if memory_optimizer:
            memory_optimizer.optimize_dataframe_memory(train_slice)

        # Use matrix operations for correlation analysis if available
        fold_correlations = None
        if matrix_ops and not train_slice.empty and not validation_slice.empty:
            try:
                # Analyze correlations between train and validation sets
                combined_data = pd.concat([train_slice, validation_slice], axis=0)
                if len(combined_data.columns) > 1:  # Need at least 2 columns for correlation
                    fold_correlations = safe_correlation_matrix(combined_data)
                    tprint_debug(f"Fold {fold}: computed correlation matrix for {len(combined_data.columns)} features")
            except Exception as e:
                tprint_debug(f"Fold {fold}: correlation analysis failed: {e}")

        # Optimize DataFrames using vectorized core if available
        if vectorized_core:
            try:
                train_slice = optimize_dataframe(train_slice)
                validation_slice = optimize_dataframe(validation_slice)
                test_slice = optimize_dataframe(test_slice)
                tprint_debug(f"Fold {fold}: optimized DataFrames using vectorized core")
            except Exception as e:
                tprint_debug(f"Fold {fold}: DataFrame optimization failed: {e}")

        fold_obj = WalkForwardFold(
            fold=fold,
            train=train_slice.copy(),
            validation=validation_slice.copy(),
            test=test_slice.copy(),
        )

        tprint_debug(f"Fold {fold}: train={len(fold_obj.train)}, val={len(fold_obj.validation)}, test={len(fold_obj.test)}")
        if fold_correlations is not None:
            tprint_debug(f"Fold {fold}: correlation analysis completed")

        yield fold_obj

        evaluation_end = test_slice.index[-1] if not test_slice.empty else validation_slice.index[-1]
        embargo_start = evaluation_end + embargo_delta
        next_start = working.index.searchsorted(embargo_start, side="left")
        start_idx = max(window_end, next_start)

    tprint_info(f"Generated {fold + 1} CV folds from {n_samples} samples")
    if matrix_ops:
        tprint_debug("Matrix operations used for correlation analysis during CV generation")


@timed_operation
def analyze_cv_fold_correlations(folds: Iterable[Mapping[str, pd.DataFrame]]) -> Dict[str, Any]:
    """Analyze correlations across CV folds for data leakage detection.

    Args:
        folds: Iterable of CV fold mappings

    Returns:
        Dictionary containing correlation analysis results
    """
    tprint_info("Starting CV fold correlation analysis")

    if not MATRIX_OPERATIONS_AVAILABLE:
        tprint_warning("Matrix operations not available for correlation analysis")
        return {}

    matrix_ops = get_unified_matrix_operations()
    if not matrix_ops:
        tprint_warning("Matrix operations instance not available")
        return {}

    fold_correlations = []
    fold_sizes = []

    for fold_idx, fold in enumerate(folds):
        try:
            # Extract numeric data from each fold
            fold_data = []
            for split_name in ['train', 'validation', 'test']:
                df = fold.get(split_name)
                if df is not None and not df.empty and len(df.columns) > 0:
                    numeric_df = df.select_dtypes(include=[np.number])
                    if not numeric_df.empty:
                        fold_data.append(numeric_df.values)

            if len(fold_data) >= 2:  # Need at least 2 splits for correlation
                # Compute correlation between different splits
                combined_data = np.concatenate(fold_data, axis=0)
                if combined_data.shape[1] > 1:  # Need multiple features
                    corr_matrix = safe_correlation_matrix(pd.DataFrame(combined_data))
                    if corr_matrix is not None:
                        fold_correlations.append(corr_matrix.values)
                        fold_sizes.append(combined_data.shape[0])
                        tprint_debug(f"Fold {fold_idx}: correlation matrix computed for {combined_data.shape[1]} features")

        except Exception as e:
            tprint_debug(f"Fold {fold_idx}: correlation analysis failed: {e}")
            continue

    if not fold_correlations:
        tprint_warning("No correlation data available for analysis")
        return {}

    # Aggregate correlation results
    try:
        avg_correlation = np.mean(fold_correlations, axis=0)
        max_correlation = np.max(fold_correlations, axis=0)
        min_correlation = np.min(fold_correlations, axis=0)

        results = {
            'average_correlation_matrix': avg_correlation.tolist(),
            'max_correlation_matrix': max_correlation.tolist(),
            'min_correlation_matrix': min_correlation.tolist(),
            'num_folds_analyzed': len(fold_correlations),
            'total_samples_analyzed': sum(fold_sizes),
            'features_analyzed': len(fold_correlations[0]) if fold_correlations else 0
        }

        tprint_info(f"CV correlation analysis completed: {len(fold_correlations)} folds, {results['features_analyzed']} features")
        return results

    except Exception as e:
        tprint_error(f"CV correlation analysis aggregation failed: {e}")
        return {}


def _get_fold_df(fold: Mapping[str, pd.DataFrame], key: str) -> pd.DataFrame:
    value = fold.get(key)
    if value is None:
        return pd.DataFrame()
    if not isinstance(value, pd.DataFrame):
        raise TypeError(f"Fold entry '{key}' must be a pandas DataFrame")
    return _ensure_datetime_index(value, name=f"fold[{key}]")


def validate_cv_no_leakage(
    folds: Iterable[Mapping[str, pd.DataFrame]],
    *,
    purge_window_hours: float,
    embargo_window_hours: float,
) -> None:
    """Ensure walk-forward splits respect purge/embargo gaps and chronology."""

    tprint_info("Validating CV folds for temporal leakage")

    purge_delta = pd.Timedelta(hours=max(purge_window_hours, 0.0))
    embargo_delta = pd.Timedelta(hours=max(embargo_window_hours, 0.0))

    last_evaluation_end: Optional[pd.Timestamp] = None

    for idx, fold in enumerate(folds):
        train_df = _get_fold_df(fold, "train")
        val_df = _get_fold_df(fold, "validation")
        test_df = _get_fold_df(fold, "test")

        if val_df.empty and test_df.empty:
            raise ValueError(f"Fold {idx} must contain validation or test samples")

        if not train_df.empty and not val_df.empty:
            if train_df.index.max() >= val_df.index.min():
                tprint_error(f"Fold {idx}: train/validation overlap detected")
                raise ValueError(f"Fold {idx} has overlapping train/validation windows")
            purge_gap = val_df.index.min() - train_df.index.max()
            if purge_gap < purge_delta:
                tprint_error(f"Fold {idx}: purge window violation (gap: {purge_gap}, required: {purge_delta})")
                raise ValueError(f"Fold {idx} violates purge window requirements")

        if not val_df.empty and not test_df.empty:
            if val_df.index.max() >= test_df.index.min():
                tprint_error(f"Fold {idx}: validation/test overlap detected")
                raise ValueError(f"Fold {idx} validation overlaps with test window")

        evaluation_start = val_df.index.min() if not val_df.empty else test_df.index.min()
        evaluation_end = test_df.index.max() if not test_df.empty else val_df.index.max()

        if last_evaluation_end is not None:
            if evaluation_start <= last_evaluation_end:
                tprint_error(f"Fold {idx}: non-increasing fold windows")
                raise ValueError("Fold windows are not strictly increasing")
            embargo_gap = evaluation_start - last_evaluation_end
            if embargo_gap < embargo_delta:
                tprint_error(f"Fold {idx}: embargo window violation (gap: {embargo_gap}, required: {embargo_delta})")
                raise ValueError(f"Fold {idx} violates embargo window requirements")

        last_evaluation_end = evaluation_end

    tprint_info(f"CV validation completed successfully for {idx + 1} folds")

