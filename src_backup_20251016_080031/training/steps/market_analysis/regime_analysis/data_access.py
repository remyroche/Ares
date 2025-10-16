"""Data access utilities for NAS/TAS regime analysis."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from src.utils.common_operations import safe_read_parquet
from src.utils.math_validation import validate_finite, validate_numeric_array
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
    labels = validate_numeric_array(labels, "regime_labels")
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
        try:
            # Convert columns to numeric, handling errors gracefully
            feature_df = regime_frame[direct_columns].copy()

            # Check what data types we have
            for col in direct_columns:
                dtype = feature_df[col].dtype
                sample_values = feature_df[col].head(3).tolist()
                print(f"Column {col}: dtype={dtype}, sample_values={sample_values}")

                # Try to convert to numeric
                feature_df[col] = pd.to_numeric(feature_df[col], errors='coerce')

            # Check for any NaN values after conversion
            nan_count = feature_df.isnull().sum().sum()
            if nan_count > 0:
                nan_cols = feature_df.columns[feature_df.isnull().any()].tolist()
                raise RegimeDataError(
                    f"Non-numeric values encountered in {feature_set.upper()} feature columns: {nan_cols}. "
                    f"Total NaN count: {nan_count}"
                )

            matrix = feature_df.to_numpy(dtype=float)
            return matrix, list(direct_columns)
        except Exception as e:
            # Debug: Show what's in the columns
            print(f"DEBUG: Available columns: {list(regime_frame.columns)}")
            print(f"DEBUG: Direct columns for {feature_set}: {direct_columns}")
            for col in direct_columns[:3]:  # Show first 3 columns
                print(f"DEBUG: Column {col} dtype: {regime_frame[col].dtype}")
                print(f"DEBUG: Column {col} sample: {regime_frame[col].head(3).tolist()}")

            raise RegimeDataError(
                f"Failed to process {feature_set.upper()} feature columns: {e}"
            )

    if array_column in regime_frame.columns:
        try:
            series = regime_frame[array_column]
            # Handle different array formats
            if series.dtype == object:
                # Convert each element to numpy array
                arrays = []
                for item in series:
                    if isinstance(item, (list, np.ndarray)):
                        arrays.append(np.asarray(item))
                    else:
                        arrays.append(np.array([item]))

                if arrays:
                    matrix = np.stack(arrays)
                    if matrix.ndim != 2:
                        raise RegimeDataError(
                            f"Expected 2D feature arrays in column '{array_column}' but found shape {matrix.shape}"
                        )
                else:
                    raise RegimeDataError(f"Empty feature arrays in column '{array_column}'")
            else:
                matrix = series.to_numpy()

            column_names = [f"{prefix}{idx}" for idx in range(matrix.shape[1])]
            return matrix, column_names
        except Exception as e:
            raise RegimeDataError(
                f"Failed to process {feature_set.upper()} feature arrays in column '{array_column}': {e}"
            )

    # 🚨 CRITICAL ERROR: No features found!
    from src.utils.tprint import tprint_error
    tprint_error(
        f"🚨 CRITICAL: No {feature_set.upper()} features found in regime assignments!\n"
        f"   Available columns: {list(regime_frame.columns)}\n"
        f"   Looking for: columns starting with '{prefix}' or column '{array_column}'\n"
        f"\n"
        f"   This means:\n"
        f"   1. Features were never added to regime_assignments parquet file\n"
        f"   2. The clustering pipeline is incomplete\n"
        f"   3. You cannot train models without features!\n"
        f"\n"
        f"   FIX: Update the clustering pipeline to include features in the parquet file.\n"
        f"   The file should have columns like: nas_feature_0, nas_feature_1, ...\n"
        f"   or a column 'nas_features' containing arrays."
    )
    
    raise RegimeDataError(
        f"No cached features found for '{feature_set.upper()}' in regime assignments. "
        f"Available columns: {list(regime_frame.columns)}. "
        f"The clustering pipeline must be updated to save features!"
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
    standardized = validate_numeric_array(standardized, f"{feature_set}_features")

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
    """
    Load NAS regime features and labels from the cached assignments.

    Fast fails if features are missing - clustering pipeline must be fixed first.
    """
    labels = extract_regime_labels(regime_frame)

    try:
        features, _ = _extract_feature_matrix(regime_frame, "nas")
        standardized = _standardize_features(features, labels, "nas")
        return standardized, labels
    except RegimeDataError as e:
        # Fast fail - clustering pipeline must be fixed first
        from src.utils.tprint import tprint_error
        tprint_error(
            f"🚨 CRITICAL: No NAS features found in regime_assignments file!\n"
            f"   Available columns: {list(regime_frame.columns)}\n"
            f"   Expected: columns starting with 'nas_feature_' or column 'nas_features'\n"
            f"\n"
            f"   SOLUTION: Re-run the clustering step (nas_tas_clustering).\n"
            f"   The clustering component now saves features with regime assignments.\n"
            f"   Run: python3 src/launcher/ares_launcher.py step05 nas_tas_clustering\n"
            f"\n"
            f"   After fixing clustering, regime analysis will work properly."
        )
        raise ValueError(
            "No NAS features found in regime_assignments file. "
            "Re-run clustering step to generate features. "
            f"Available columns: {list(regime_frame.columns)}"
        ) from e


def load_tas_dataset(regime_frame: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load TAS regime features and labels from the cached assignments.

    Fast fails if features are missing - clustering pipeline must be fixed first.
    """
    labels = extract_regime_labels(regime_frame)

    try:
        features, _ = _extract_feature_matrix(regime_frame, "tas")
        standardized = _standardize_features(features, labels, "tas")
        return standardized, labels
    except RegimeDataError as e:
        # Fast fail - clustering pipeline must be fixed first
        from src.utils.tprint import tprint_error
        tprint_error(
            f"🚨 CRITICAL: No TAS features found in regime_assignments file!\n"
            f"   Available columns: {list(regime_frame.columns)}\n"
            f"   Expected: columns starting with 'tas_feature_' or column 'tas_features'\n"
            f"\n"
            f"   SOLUTION: Re-run the clustering step (nas_tas_clustering).\n"
            f"   The clustering component now saves features with regime assignments.\n"
            f"   Run: python3 src/launcher/ares_launcher.py step05 nas_tas_clustering\n"
            f"\n"
            f"   After fixing clustering, regime analysis will work properly."
        )
        raise ValueError(
            "No TAS features found in regime_assignments file. "
            "Re-run clustering step to generate features. "
            f"Available columns: {list(regime_frame.columns)}"
        ) from e


def load_regime_datasets(
    data_cache_path: Path,
    symbol: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load NAS and TAS datasets for a symbol from the cached clustering outputs.

    Fast fails if features are missing - clustering pipeline must be fixed first.
    Returns (nas_features, nas_labels, tas_features, tas_labels).
    """
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

        if nas_features is not None and tas_features is not None:
            if nas_features.shape[0] != tas_features.shape[0]:
                raise ValueError(
                    "Feature length mismatch between NAS and TAS datasets"
                )

        # Build success message
        nas_shape = nas_features.shape if nas_features is not None else "no features"
        tas_shape = tas_features.shape if tas_features is not None else "no features"
        tprint_success(
            f"Loaded NAS ({nas_shape}, {len(nas_labels)} labels) and "
            f"TAS ({tas_shape}, {len(tas_labels)} labels) datasets for {symbol}"
        )

        return nas_features, nas_labels, tas_features, tas_labels
