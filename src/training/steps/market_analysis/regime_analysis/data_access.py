"""Data access utilities for NAS/TAS regime analysis."""
from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd


DEFAULT_FEATURE_COUNT = 10
NAS_FEATURE_OFFSET = 0.5
TAS_FEATURE_OFFSET = 0.3


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
    """Load the parquet file containing regime assignments."""
    try:
        return pd.read_parquet(regime_file)
    except Exception as exc:  # pragma: no cover - passthrough for pandas errors
        raise RegimeDataError(f"Failed to read regime assignments from {regime_file}") from exc


def extract_regime_labels(regime_frame: pd.DataFrame) -> np.ndarray:
    """Extract regime labels from the cached dataframe."""
    if "regime_id" not in regime_frame:
        raise KeyError("regime_id column missing from regime assignments")
    return regime_frame["regime_id"].to_numpy()


def create_synthetic_features(
    labels: np.ndarray,
    *,
    seed: int,
    feature_count: int,
    regime_offset: float,
) -> np.ndarray:
    """Create deterministic synthetic features mirroring original script behaviour."""
    rng = np.random.default_rng(seed)
    features = rng.standard_normal((labels.shape[0], feature_count))
    unique_regimes = np.unique(labels)
    for regime_id in unique_regimes:
        mask = labels == regime_id
        features[mask] += regime_id * regime_offset
    return features


def load_nas_dataset(regime_frame: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """Load NAS regime features and labels."""
    labels = extract_regime_labels(regime_frame)
    features = create_synthetic_features(
        labels,
        seed=42,
        feature_count=DEFAULT_FEATURE_COUNT,
        regime_offset=NAS_FEATURE_OFFSET,
    )
    return features, labels


def load_tas_dataset(regime_frame: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """Load TAS regime features and labels."""
    labels = extract_regime_labels(regime_frame)
    features = create_synthetic_features(
        labels,
        seed=99,
        feature_count=DEFAULT_FEATURE_COUNT,
        regime_offset=TAS_FEATURE_OFFSET,
    )
    return features, labels


def load_regime_datasets(data_cache_path: Path, symbol: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load NAS and TAS datasets for a symbol from the cached clustering outputs."""
    clustering_dir = get_clustering_directory(data_cache_path, symbol)
    latest_file = find_latest_regime_file(clustering_dir)
    regime_frame = load_regime_assignments(latest_file)

    nas_features, nas_labels = load_nas_dataset(regime_frame)
    tas_features, tas_labels = load_tas_dataset(regime_frame)
    return nas_features, nas_labels, tas_features, tas_labels
