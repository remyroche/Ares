"""Data access utilities for NAS/TAS regime analysis."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence, Tuple

import numpy as np
import pandas as pd

from src.utils.common_operations import safe_read_parquet
from src.utils.math_validation import validate_finite
from src.utils.tprint import (
    tprint_info,
    tprint_structured,
    tprint_success,
    tprint_timer,
)


class RegimeDataError(FileNotFoundError):
    """Raised when regime data cannot be located."""


def get_clustering_directory(data_cache_path: Path, symbol: str) -> Path:
    """Return the directory containing cached clustering data for a symbol."""
    clustering_dir = data_cache_path / "nas_tas_clustering" / symbol
    if not clustering_dir.exists():
        raise RegimeDataError(f"Clustering directory not found: {clustering_dir}")
    return clustering_dir


def find_latest_regime_file(clustering_dir: Path) -> Path:
    """Locate the most recent regime assignment parquet file in a directory."""
    regime_files = list(clustering_dir.glob("nas_tas_regime_assignments_*.parquet"))
    if not regime_files:
        raise RegimeDataError(f"No regime assignment files found in {clustering_dir}")
    return max(regime_files, key=lambda path: path.stat().st_mtime)


def load_regime_assignments(regime_file: Path) -> pd.DataFrame:
    """Load the parquet file containing regime assignments and cached features."""
    with tprint_timer(f"Loading regime assignments from {regime_file}"):
        if not regime_file.exists():
            raise RegimeDataError(f"Regime file not accessible: {regime_file}")

        frame = safe_read_parquet(regime_file)
        if frame is None:
            raise RegimeDataError(f"Failed to read regime assignments from {regime_file}")

        if "regime_id" not in frame.columns:
            raise RegimeDataError("regime_id column missing from regime assignments")

        # Ensure we do not carry an unexpected index level that hides the timestamp column.
        if isinstance(frame.index, pd.MultiIndex):
            frame = frame.copy()
            frame.reset_index(inplace=True)
        else:
            frame = frame.copy()

        feature_columns = {
            "nas": _candidate_feature_columns(frame.columns, "nas"),
            "tas": _candidate_feature_columns(frame.columns, "tas"),
        }
        tprint_structured(
            {
                "rows": len(frame),
                "columns": list(frame.columns),
                "feature_columns": feature_columns,
            }
        )

        return frame


def extract_regime_labels(regime_frame: pd.DataFrame) -> np.ndarray:
    """Extract regime labels from the cached dataframe with validation."""
    labels = regime_frame["regime_id"].to_numpy()
    labels = validate_finite(labels, "regime_labels")
    return labels.astype(int, copy=False)


def _candidate_feature_columns(columns: Iterable[str], feature_set: str) -> Sequence[str]:
    prefix = f"{feature_set}_feature_"
    return [name for name in columns if name.startswith(prefix)]


def _extract_feature_matrix(
    regime_frame: pd.DataFrame,
    feature_set: str,
) -> Tuple[np.ndarray, Sequence[str]]:
    """Extract a raw feature matrix for the requested feature set."""
    prefix = f"{feature_set}_feature_"
    array_column = f"{feature_set}_features"

    direct_columns = _candidate_feature_columns(regime_frame.columns, feature_set)
    if direct_columns:
        feature_df = regime_frame[direct_columns].apply(pd.to_numeric, errors="coerce")
        if feature_df.isnull().any().any():
            raise RegimeDataError(
                f"Non-numeric values encountered in {feature_set.upper()} feature columns"
            )
        matrix = feature_df.to_numpy(dtype=float)
        return matrix, list(direct_columns)

    if array_column in regime_frame.columns:
        series = regime_frame[array_column]
        matrix = np.asarray(series.apply(np.asarray).tolist(), dtype=float)
        if matrix.ndim != 2:
            raise RegimeDataError(
                f"Expected 2D feature arrays in column '{array_column}' but found shape {matrix.shape}"
            )
        column_names = [f"{prefix}{idx}" for idx in range(matrix.shape[1])]
        return matrix, column_names

    raise RegimeDataError(
        f"No cached features found for '{feature_set.upper()}' in regime assignments"
    )


def _standardize_features(
    features: np.ndarray,
    labels: np.ndarray,
    feature_set: str,
) -> np.ndarray:
    """Standardize features using observed statistics and log regime offsets."""
    if features.ndim != 2:
        raise ValueError("Features must be a 2D array")

    mean = features.mean(axis=0)
    std = features.std(axis=0, ddof=0)
    std[std == 0.0] = 1.0

    standardized = (features - mean) / std
    standardized = validate_finite(standardized, f"{feature_set}_features")

    regime_offsets = {}
    for regime_id in np.unique(labels):
        regime_mask = labels == regime_id
        if not regime_mask.any():
            continue
        regime_mean = standardized[regime_mask].mean(axis=0)
        regime_offsets[int(regime_id)] = regime_mean.round(6).tolist()

    tprint_structured(
        {
            f"{feature_set}_feature_stats": {
                "sample_count": int(features.shape[0]),
                "feature_count": int(features.shape[1]),
                "global_mean": mean.round(6).tolist(),
                "global_std": std.round(6).tolist(),
                "regime_offsets": regime_offsets,
            }
        }
    )

    return standardized


def load_nas_dataset(regime_frame: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """Load NAS regime features and labels from the cached assignments."""
    labels = extract_regime_labels(regime_frame)
    features, _ = _extract_feature_matrix(regime_frame, "nas")
    standardized = _standardize_features(features, labels, "nas")
    return standardized, labels


def load_tas_dataset(regime_frame: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """Load TAS regime features and labels from the cached assignments."""
    labels = extract_regime_labels(regime_frame)
    features, _ = _extract_feature_matrix(regime_frame, "tas")
    standardized = _standardize_features(features, labels, "tas")
    return standardized, labels


def load_regime_datasets(
    data_cache_path: Path,
    symbol: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load NAS and TAS datasets for a symbol from the cached clustering outputs."""
    with tprint_timer(f"Loading regime datasets for {symbol}"):
        clustering_dir = get_clustering_directory(data_cache_path, symbol)
        tprint_info(f"Using clustering directory: {clustering_dir}")

        latest_file = find_latest_regime_file(clustering_dir)
        tprint_info(f"Loading regime file: {latest_file}")

        regime_frame = load_regime_assignments(latest_file)

        nas_features, nas_labels = load_nas_dataset(regime_frame)
        tas_features, tas_labels = load_tas_dataset(regime_frame)

        if len(nas_labels) != len(tas_labels):
            raise ValueError(
                f"Label length mismatch: NAS={len(nas_labels)}, TAS={len(tas_labels)}"
            )

        if nas_features.shape[0] != tas_features.shape[0]:
            raise ValueError(
                "Feature length mismatch between NAS and TAS datasets"
            )

        tprint_success(
            f"Loaded NAS ({nas_features.shape}) and TAS ({tas_features.shape}) datasets for {symbol}"
        )

        return nas_features, nas_labels, tas_features, tas_labels
