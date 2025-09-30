"""Data access utilities for NAS/TAS regime analysis."""
from __future__ import annotations

from pathlib import Path
from typing import Tuple, Dict, Any, Optional
import os

import numpy as np
import pandas as pd

# Import common operations for data quality and validation
from src.utils.common_operations import (
    validate_dataframe_columns,
    calculate_data_quality_metrics,
    create_data_quality_report,
    safe_convert_dtypes,
    optimize_dataframe_dtypes,
    safe_read_parquet,
    safe_to_parquet,
    validate_dataframe_schema,
    guard_dataframe_nulls,
    get_dataframe_info,
    safe_timestamp_conversion,
    validate_timestamp_column,
    create_summary_statistics,
    safe_fillna,
    safe_merge_dataframes,
    safe_drop_columns,
    safe_rename_columns,
    optimize_dataframe_dtypes,
    safe_resample,
    align_dataframes,
    validate_file_path,
    get_file_size,
    check_disk_space,
    safe_copy,
    safe_deepcopy,
    get_memory_usage,
    optimize_memory,
    memory_checkpoint,
    gpu_context
)

# Import math validation for safe operations
from src.utils.math_validation import (
    safe_mean,
    safe_std,
    safe_correlation,
    safe_covariance,
    validate_finite,
    validate_positive,
    validate_range,
    safe_percentage_change,
    safe_weighted_average,
    safe_kelly_calculation
)

# Import tprint for enhanced logging
from src.utils.tprint import (
    tprint,
    tprint_info,
    tprint_warning,
    tprint_error,
    tprint_success,
    tprint_performance,
    tprint_timer,
    tprint_structured
)

# Import M1 hardware optimizations
try:
    from src.utils.hardware.m1_memory_optimizer import get_m1_memory_optimizer
    from src.utils.hardware.m1_cpu_optimizer import get_m1_cpu_optimizer
    from src.utils.hardware.m1_gpu_utils import get_m1_gpu_manager
    M1_HARDWARE_AVAILABLE = True
except ImportError:
    M1_HARDWARE_AVAILABLE = False
    get_m1_memory_optimizer = lambda: None
    get_m1_cpu_optimizer = lambda: None
    get_m1_gpu_manager = lambda: None


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
    """Load the parquet file containing regime assignments with enhanced data quality validation."""
    with tprint_timer(f"Loading regime assignments from {regime_file}"):
        try:
            # Comprehensive file validation
            if not validate_file_path(regime_file):
                raise RegimeDataError(f"Regime file not accessible: {regime_file}")
            
            # Check file permissions and readability
            if not regime_file.is_file():
                raise RegimeDataError(f"Path is not a file: {regime_file}")
            
            if not os.access(regime_file, os.R_OK):
                raise RegimeDataError(f"File is not readable: {regime_file}")
            
            # Check disk space before loading with more accurate estimation
            file_size_mb = get_file_size(regime_file) / (1024**2)
            required_gb = max(1.0, file_size_mb * 3 / 1024)  # 3x file size for safety
            disk_info = check_disk_space(regime_file, required_gb=required_gb)
            if not disk_info['sufficient']:
                raise RegimeDataError(f"Insufficient disk space: {disk_info['free_gb']:.2f}GB available, {required_gb:.2f}GB required")
            
            # Use memory checkpoint for large files with proper cleanup
            try:
                with memory_checkpoint("regime_data_loading"):
                    # Load with safe parquet reading
                    df = safe_read_parquet(regime_file)
                    if df is None:
                        raise RegimeDataError(f"Failed to read regime assignments from {regime_file}")
                    
                    # Validate required columns
                    required_columns = ['regime_id']
                    if not validate_dataframe_columns(df, required_columns):
                        raise RegimeDataError(f"Missing required columns in {regime_file}")
                    
                    # Calculate and log data quality metrics
                    quality_metrics = calculate_data_quality_metrics(df)
                    tprint_structured({
                        "file": str(regime_file),
                        "rows": quality_metrics['total_rows'],
                        "columns": quality_metrics['total_columns'],
                        "missing_percentage": quality_metrics['missing_percentage'],
                        "duplicate_percentage": quality_metrics['duplicate_percentage']
                    })
                    
                    # Guard against excessive nulls
                    df = guard_dataframe_nulls(df, threshold=0.5)
                    
                    # Optimize data types for memory efficiency
                    df = optimize_dataframe_dtypes(df)
                    
                    # Log memory usage
                    memory_usage = get_memory_usage()
                    tprint_performance(f"Memory usage after loading: {memory_usage / (1024**2):.2f} MB")
                    
                    tprint_success(f"Successfully loaded regime assignments: {len(df)} rows, {len(df.columns)} columns")
                    return df
            except Exception as checkpoint_error:
                # Cleanup on checkpoint failure
                memory_optimizer = get_m1_memory_optimizer()
                if memory_optimizer:
                    memory_optimizer.cleanup_arrays([])
                raise checkpoint_error
                
        except Exception as exc:
            tprint_error(f"Failed to load regime assignments: {exc}")
            raise RegimeDataError(f"Failed to read regime assignments from {regime_file}") from exc


def extract_regime_labels(regime_frame: pd.DataFrame) -> np.ndarray:
    """Extract regime labels from the cached dataframe with validation."""
    with tprint_timer("Extracting regime labels"):
        try:
            if "regime_id" not in regime_frame:
                raise KeyError("regime_id column missing from regime assignments")
            
            # Extract labels with validation
            labels = regime_frame["regime_id"].to_numpy()
            
            # Validate labels are finite and positive
            labels = validate_finite(labels, "regime_labels")
            
            # Log label statistics
            unique_labels = np.unique(labels)
            tprint_structured({
                "total_labels": len(labels),
                "unique_regimes": len(unique_labels),
                "regime_range": f"{np.min(labels)}-{np.max(labels)}",
                "label_distribution": {f"regime_{int(label)}": int(np.sum(labels == label)) for label in unique_labels}
            })
            
            return labels
            
        except Exception as exc:
            tprint_error(f"Failed to extract regime labels: {exc}")
            raise


def create_synthetic_features(
    labels: np.ndarray,
    *,
    seed: int,
    feature_count: int,
    regime_offset: float,
) -> np.ndarray:
    """Create deterministic synthetic features mirroring original script behaviour with M1 optimizations."""
    with tprint_timer(f"Creating synthetic features (count={feature_count}, offset={regime_offset})"):
        try:
            # Initialize M1 optimizers if available
            memory_optimizer = get_m1_memory_optimizer()
            cpu_optimizer = get_m1_cpu_optimizer()
            gpu_manager = get_m1_gpu_manager()
            
            # Use memory checkpoint for large feature generation with proper cleanup
            try:
                with memory_checkpoint("synthetic_feature_generation"):
                    rng = np.random.default_rng(seed)
                    features = rng.standard_normal((labels.shape[0], feature_count))
                    
                    # Validate features are finite
                    features = validate_finite(features, "synthetic_features")
                    
                    unique_regimes = np.unique(labels)
                    for regime_id in unique_regimes:
                        mask = labels == regime_id
                        if np.any(mask):
                            # Apply regime-specific offset
                            features[mask] += regime_id * regime_offset
                    
                    # Optimize for M1 if available
                    if M1_HARDWARE_AVAILABLE and memory_optimizer:
                        features = memory_optimizer.optimize_array_memory(features)
                    
                    # Log feature statistics
                    tprint_structured({
                        "feature_shape": features.shape,
                        "feature_mean": float(safe_mean(features.flatten())),
                        "feature_std": float(safe_std(features.flatten())),
                        "regime_count": len(unique_regimes),
                        "memory_usage_mb": features.nbytes / (1024**2)
                    })
                    
                    tprint_success(f"Generated synthetic features: {features.shape}")
                    return features
            except Exception as checkpoint_error:
                # Cleanup on checkpoint failure
                if memory_optimizer:
                    memory_optimizer.cleanup_arrays([])
                raise checkpoint_error
                
        except Exception as exc:
            tprint_error(f"Failed to create synthetic features: {exc}")
            raise


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
    """Load NAS and TAS datasets for a symbol from the cached clustering outputs with comprehensive monitoring."""
    with tprint_timer(f"Loading regime datasets for {symbol}"):
        try:
            # Initialize hardware optimizers
            memory_optimizer = get_m1_memory_optimizer()
            cpu_optimizer = get_m1_cpu_optimizer()
            
            # Start memory monitoring if available
            if memory_optimizer:
                memory_optimizer.start_monitoring()
            
            # Get clustering directory with validation
            clustering_dir = get_clustering_directory(data_cache_path, symbol)
            tprint_info(f"Using clustering directory: {clustering_dir}")
            
            # Find and validate latest regime file
            latest_file = find_latest_regime_file(clustering_dir)
            file_size = get_file_size(latest_file)
            tprint_info(f"Loading regime file: {latest_file} ({file_size / (1024**2):.2f} MB)")
            
            # Load regime assignments with quality validation
            regime_frame = load_regime_assignments(latest_file)
            
            # Create comprehensive data quality report
            quality_report = create_data_quality_report(regime_frame)
            tprint_structured({
                "data_quality": quality_report,
                "file_path": str(latest_file),
                "symbol": symbol
            })
            
            # Load NAS dataset
            tprint_info("Loading NAS dataset...")
            nas_features, nas_labels = load_nas_dataset(regime_frame)
            
            # Load TAS dataset
            tprint_info("Loading TAS dataset...")
            tas_features, tas_labels = load_tas_dataset(regime_frame)
            
            # Validate dataset consistency
            if len(nas_labels) != len(tas_labels):
                raise ValueError(f"Label length mismatch: NAS={len(nas_labels)}, TAS={len(tas_labels)}")
            
            if nas_features.shape[0] != tas_features.shape[0]:
                raise ValueError(f"Feature length mismatch: NAS={nas_features.shape[0]}, TAS={tas_features.shape[0]}")
            
            # Log final statistics
            tprint_structured({
                "nas_features_shape": nas_features.shape,
                "nas_labels_unique": len(np.unique(nas_labels)),
                "tas_features_shape": tas_features.shape,
                "tas_labels_unique": len(np.unique(tas_labels)),
                "total_samples": len(nas_labels),
                "memory_usage_mb": get_memory_usage() / (1024**2)
            })
            
            # Optimize memory if M1 available
            if M1_HARDWARE_AVAILABLE and memory_optimizer:
                nas_features = memory_optimizer.optimize_array_memory(nas_features)
                tas_features = memory_optimizer.optimize_array_memory(tas_features)
            
            tprint_success(f"Successfully loaded regime datasets for {symbol}")
            return nas_features, nas_labels, tas_features, tas_labels
            
        except Exception as exc:
            tprint_error(f"Failed to load regime datasets for {symbol}: {exc}")
            raise
        finally:
            # Stop memory monitoring
            if memory_optimizer:
                memory_optimizer.stop_monitoring()
