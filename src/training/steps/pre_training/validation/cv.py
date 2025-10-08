"""Cross-validation utilities for pre-training steps."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Iterator, Mapping, Optional

import pandas as pd


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

    if n_splits <= 0:
        raise ValueError("n_splits must be positive")

    working = _ensure_datetime_index(data.copy(), name="data")
    n_samples = len(working)
    if n_samples == 0:
        return iter(())

    purge_delta = pd.Timedelta(hours=max(purge_window_hours, 0.0))
    embargo_delta = pd.Timedelta(hours=max(embargo_window_hours, 0.0))

    min_train_size = max(1, int(round(n_samples * max(train_ratio, 0.0))))
    if min_train_size >= n_samples:
        # Not enough data to create validation windows – yield a single fold
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
    validation_fraction = validation_ratio / ratio_denominator

    start_idx = min_train_size
    for fold in range(n_splits):
        if start_idx >= n_samples:
            break

        window_end = min(n_samples, start_idx + per_fold_window)
        if window_end - start_idx <= 0:
            break

        window_length = window_end - start_idx
        validation_length = max(1, int(round(window_length * validation_fraction)))
        if validation_length >= window_length:
            validation_length = window_length - 1

        validation_slice = working.iloc[start_idx : start_idx + validation_length]
        if validation_slice.empty:
            start_idx = window_end
            continue

        test_slice = working.iloc[start_idx + validation_length : window_end]

        validation_start = validation_slice.index[0]
        train_end_time = validation_start - purge_delta
        train_slice = working.loc[:train_end_time]

        if len(train_slice) < min_train_size:
            start_idx = window_end
            continue

        yield WalkForwardFold(
            fold=fold,
            train=train_slice.copy(),
            validation=validation_slice.copy(),
            test=test_slice.copy(),
        )

        evaluation_end = test_slice.index[-1] if not test_slice.empty else validation_slice.index[-1]
        embargo_start = evaluation_end + embargo_delta
        next_start = working.index.searchsorted(embargo_start, side="left")
        start_idx = max(window_end, next_start)


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
                raise ValueError(f"Fold {idx} has overlapping train/validation windows")
            if (val_df.index.min() - train_df.index.max()) < purge_delta:
                raise ValueError(f"Fold {idx} violates purge window requirements")

        if not val_df.empty and not test_df.empty:
            if val_df.index.max() >= test_df.index.min():
                raise ValueError(f"Fold {idx} validation overlaps with test window")

        evaluation_start = val_df.index.min() if not val_df.empty else test_df.index.min()
        evaluation_end = test_df.index.max() if not test_df.empty else val_df.index.max()

        if last_evaluation_end is not None:
            if evaluation_start <= last_evaluation_end:
                raise ValueError("Fold windows are not strictly increasing")
            if (evaluation_start - last_evaluation_end) < embargo_delta:
                raise ValueError(f"Fold {idx} violates embargo window requirements")

        last_evaluation_end = evaluation_end

