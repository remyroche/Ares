"""Data access utilities for NAS/TAS regime analysis."""
from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

# Enhanced utility imports
from src.utils.common_operations import (
    safe_dataframe_operation, validate_dataframe_columns, safe_convert_dtypes,
    calculate_data_quality_metrics, optimize_dataframe_memory, safe_numpy_operation,
    validate_numpy_array, safe_array_operation, validate_file_path, safe_file_operation
)
from src.utils.common_utilities import (
    safe_dataframe_operation as safe_df_op, validate_dataframe_columns as validate_df_cols,
    safe_convert_dtypes as safe_convert, calculate_data_quality_metrics as calc_quality,
    optimize_dataframe_performance, safe_apply_function, validate_data_consistency,
    calculate_statistical_metrics, safe_aggregation, validate_data_types
)
from src.utils.math_validation import (
    safe_divide, safe_log, safe_sqrt, safe_power, validate_finite, safe_exp,
    safe_sin, safe_cos, safe_tan, validate_positive, validate_range, safe_abs,
    safe_min, safe_max, safe_mean, safe_std, safe_correlation, safe_covariance,
    validate_matrix_operations, safe_matrix_multiply, safe_matrix_inverse,
    safe_eigenvalues, safe_svd, validate_numerical_stability
)
from src.utils.data.unified_data_utils import (
    load_market_data, validate_market_data, preprocess_market_data,
    calculate_returns, calculate_volatility, calculate_volume_metrics,
    detect_market_regimes, validate_data_quality, optimize_data_storage
)
from src.utils.data.quality.advanced_quality_metrics import (
    calculate_comprehensive_quality_score, detect_data_drift, validate_statistical_properties,
    assess_data_completeness, evaluate_data_consistency, generate_quality_report
)
from src.utils.tprint import tprint, tprint_info, tprint_success, tprint_warning, tprint_error


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
    """Load the parquet file containing regime assignments with enhanced validation."""
    try:
        tprint(f"🔄 Loading regime assignments from {regime_file}", "INFO")
        
        # Validate file path
        if not validate_file_path(regime_file):
            raise RegimeDataError(f"Invalid file path: {regime_file}")
        
        # Load data with enhanced error handling
        data = safe_file_operation(
            lambda: pd.read_parquet(regime_file),
            f"Failed to read regime assignments from {regime_file}"
        )
        
        # Validate DataFrame
        if not validate_dataframe_columns(data, ['regime_id']):
            tprint("⚠️ Missing required columns, attempting to standardize", "WARNING")
            data = safe_convert_dtypes(data, {'regime_id': 'int64'})
        
        # Calculate data quality metrics
        quality_metrics = calculate_data_quality_metrics(data)
        tprint(f"📈 Data quality score: {quality_metrics.get('quality_score', 0):.3f}", "INFO")
        
        # Optimize DataFrame memory usage
        data = optimize_dataframe_memory(data)
        tprint("✅ Regime assignments loaded and optimized", "SUCCESS")
        
        return data
        
    except Exception as exc:  # pragma: no cover - passthrough for pandas errors
        tprint(f"❌ Failed to read regime assignments: {exc}", "ERROR")
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
    """Create deterministic synthetic features with enhanced validation and optimization."""
    try:
        tprint(f"🔄 Creating synthetic features (seed={seed}, features={feature_count})", "INFO")
        
        # Validate inputs using math validation utilities
        if not validate_positive(feature_count, "feature_count"):
            raise ValueError("Feature count must be positive")
        if not validate_range(regime_offset, 0.0, 1.0, "regime_offset"):
            tprint("⚠️ Regime offset outside recommended range, applying correction", "WARNING")
            regime_offset = safe_abs(regime_offset) % 1.0
        
        # Validate labels
        labels = validate_numpy_array(labels, "labels")
        if not validate_finite(labels.max(), "max_label"):
            raise ValueError("Invalid labels detected")
        
        # Create features with enhanced random generation
        rng = np.random.default_rng(seed)
        features = safe_array_operation(
            lambda: rng.standard_normal((labels.shape[0], feature_count)),
            "Failed to generate random features"
        )
        
        # Apply regime-specific offsets with enhanced validation
        unique_regimes = np.unique(labels)
        tprint(f"📊 Processing {len(unique_regimes)} unique regimes", "INFO")
        
        for regime_id in unique_regimes:
            mask = labels == regime_id
            if np.any(mask):
                # Calculate safe offset
                offset = safe_multiply(regime_id, regime_offset)
                features[mask] = safe_array_operation(
                    lambda: features[mask] + offset,
                    f"Failed to apply offset for regime {regime_id}"
                )
        
        # Validate numerical stability
        if not validate_numerical_stability(features):
            tprint("⚠️ Numerical stability issues detected, applying corrections", "WARNING")
            features = safe_array_operation(
                lambda: np.nan_to_num(features, nan=0.0, posinf=1e6, neginf=-1e6),
                "Failed to correct numerical stability"
            )
        
        # Calculate feature statistics
        feature_stats = calculate_statistical_metrics(features)
        tprint(f"📈 Feature statistics: mean={feature_stats.get('mean', 0):.3f}, std={feature_stats.get('std', 0):.3f}", "INFO")
        
        tprint("✅ Synthetic features created successfully", "SUCCESS")
        return features
        
    except Exception as e:
        tprint(f"❌ Synthetic feature creation failed: {e}", "ERROR")
        raise ValueError(f"Synthetic feature creation failed: {e}")


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
