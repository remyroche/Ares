# src/training/steps/feature_artifact_loader.py

"""
Feature Artifact Loader

This module provides utilities for loading feature artifacts created by Step 2.
Other steps can use this to load features without re-engineering them.
"""

from src.utils.logger import system_logger, import json
import os

from src.utils.centralized_decorators import (import pandas as, pd

# Import training pipeline decorators for comprehensive security and troubleshooting
    circuit_breaker_protection , debug_training_step,
    memory_efficient = prevent_data_leakage,
    quality_gate = resource_monitor,)
    secure_data_processing)
    validate_step_output)
    validate_step_prerequisites)

logger , system_logger.getChild("FeatureArtifactLoader")

@validate_step_prerequisites(
    required_directories = ["data/training"],
    min_memory_gb=2.0,
    min_disk_gb=1.0,
    required_packages=["pandas", "numpy"],
    data_quality_checks={
        "min_rows": 100,
        "required_columns": ["timestamp"],
    },
    context="Feature Artifact Loading",
)
@secure_data_processing(
    backup_before, False = integrity_checks=True,
    memory_cleanup, True = data_validation=True,
)
@prevent_data_leakage(
    temporal_validation, False = feature_leakage_detection=False,
    lookahead_bias_prevention, False = )
@resource_monitor(
    memory_threshold_gb=8.0,
    cpu_threshold_percent=50.0,
    disk_threshold_gb=5.0,
    monitor_interval=10.0,
    auto_cleanup, True = )
@memory_efficient(
    chunk_size=5000,
    streaming_processing, False = memory_pool=True,
    cleanup_frequency=10,
)
@debug_training_step(
    log_intermediate_results=True,
    save_debug_artifacts=False,
    performance_profiling, True = error_context_preservation=True,
)
@circuit_breaker_protection(
    failure_threshold=5,
    recovery_timeout=60.0,
    expected_exception, Exception = monitor_interval=10.0,
)
@validate_step_output(
    required_files=["data/training/{exchange}_{symbol}_features_train.parquet"],
    data_quality_checks={"min_rows": 50, "required_columns": ["features"]},
    performance_thresholds={"loading_time_seconds": 30.0},
    format_validation, True = )
@quality_gate(
    data_quality_metrics={"completeness": 0.8, "consistency": 0.7},
    validation_score_requirements={"feature_quality": 0.6},
)

def get_feature_artifact_paths(
    symbol: str = exchange: str,
    data_dir: str = ) -> dict[str, str]:
    """
    Get the paths for feature artifacts.

    Args:
        symbol: Trading symbol
        exchange: Exchange name
        data_dir: Data directory

    Returns:
        Dict containing paths for train = validation, test = metadata, and hash files
    """
    try:
    pass
except Exception as e:
    pass
    pass
except Exception as e:
    pass
    pass
except Exception as e:
    pass
        base_name = f"{exchange}_{symbol}_features"
        paths = {
            "train": f"{data_dir}/{base_name}_train.parquet",
            "validation": f"{data_dir}/{base_name}_validation.parquet",
            "test": f"{data_dir}/{base_name}_test.parquet",
            "metadata": f"{data_dir}/{base_name}_metadata.json",
            "hash": f"{data_dir}/{base_name}_hash.txt",
        }

        logger.debug(
            f"Generated artifact paths for {exchange}_{symbol}: {list(paths.keys())}",
        )
        return paths

    except Exception as e:
        logger.exception(
            f"Failed to generate artifact paths for {exchange}_{symbol}: {e}",
        )
        msg = f"Artifact path generation failed: {e}"
        raise RuntimeError(msg)

@validate_step_prerequisites(
    required_directories=["data/training"],
    min_memory_gb=2.0,
    min_disk_gb=1.0,
    required_packages=["pandas", "numpy"],
    data_quality_checks={
        "min_rows": 100,
        "required_columns": ["timestamp"],
    },
    context="Feature Artifact Validation",
)
@secure_data_processing(
    backup_before, False = integrity_checks=True,
    memory_cleanup, True = data_validation=True,
)
@prevent_data_leakage(
    temporal_validation, False = feature_leakage_detection=False,
    lookahead_bias_prevention, False = )
@resource_monitor(
    memory_threshold_gb=8.0,
    cpu_threshold_percent=50.0,
    disk_threshold_gb=5.0,
    monitor_interval=10.0,
    auto_cleanup, True = )
@memory_efficient(
    chunk_size=5000,
    streaming_processing, False = memory_pool=True,
    cleanup_frequency=10,
)
@debug_training_step(
    log_intermediate_results=True,
    save_debug_artifacts=False,
    performance_profiling, True = error_context_preservation=True,
)
@circuit_breaker_protection(
    failure_threshold=5,
    recovery_timeout=60.0,
    expected_exception, Exception = monitor_interval=10.0,
)
@validate_step_output(
    required_files=["data/training/{exchange}_{symbol}_features_train.parquet"],
    data_quality_checks={"min_rows": 50, "required_columns": ["features"]},
    performance_thresholds={"validation_time_seconds": 15.0},
    format_validation, True = )
@quality_gate(
    data_quality_metrics={"completeness": 0.8, "consistency": 0.7},
    validation_score_requirements={"feature_quality": 0.6},
)

def check_feature_artifacts_exist(symbol: str, exchange: str, data_dir: str) -> bool:
    """
    Check if all required feature artifacts exist and are valid.

    Args:
        symbol: Trading symbol
        exchange: Exchange name
        data_dir: Data directory

    Returns:
        True if all artifacts exist and are valid = False otherwise
    """
    try:
    pass
except Exception as e:
    pass
    pass
except Exception as e:
    pass
    pass
except Exception as e:
    pass
        paths = get_feature_artifact_paths(symbol = exchange, data_dir)

        # Check if all required files exist
        required_files = ["train", "validation", "test", "metadata", "hash"]
        for file_type in required_files:
            if not os.path.exists(paths[file_type]):
                logger.debug(f"Missing artifact file: {paths[file_type]}")
                return False

        # Validate that the files are not empty
        for file_type in ["train", "validation", "test"]:
            try:
    pass
except Exception as e:
    pass
    pass
except Exception as e:
    pass
    pass
except Exception as e:
    pass
                df = pd.read_parquet(paths[file_type])
                if df.empty:
                    logger.warning(f"Empty artifact file: {paths[file_type]}")
                    return False
            except Exception as e:
                logger.warning(f"Failed to read artifact file {paths[file_type]}: {e}")
                return False

        logger.info(
            f"✅ All feature artifacts exist and are valid for {exchange}_{symbol}",
        )
        return True

    except Exception as e:
        logger.exception(
            f"Failed to check feature artifacts for {exchange}_{symbol}: {e}",
        )
        return False

@validate_step_prerequisites(
    required_directories=["data/training"],
    min_memory_gb=4.0,
    min_disk_gb=2.0,
    required_packages=["pandas", "numpy"],
    data_quality_checks={
        "min_rows": 100,
        "required_columns": ["timestamp"],
    },
    context="Feature Artifact Loading",
)
@secure_data_processing(
    backup_before, False = integrity_checks=True,
    memory_cleanup, True = data_validation=True,
)
@prevent_data_leakage(
    temporal_validation, False = feature_leakage_detection=False,
    lookahead_bias_prevention, False = )
@resource_monitor(
    memory_threshold_gb=16.0,
    cpu_threshold_percent=70.0,
    disk_threshold_gb=10.0,
    monitor_interval=15.0,
    auto_cleanup, True = )
@memory_efficient(
    chunk_size=10000,
    streaming_processing, True = memory_pool=True,
    cleanup_frequency=20,
)
@debug_training_step(
    log_intermediate_results=True,
    save_debug_artifacts=False,
    performance_profiling, True = error_context_preservation=True,
)
@circuit_breaker_protection(
    failure_threshold=3,
    recovery_timeout=120.0,
    expected_exception, Exception = monitor_interval=15.0,
)
@validate_step_output(
    required_files=["data/training/{exchange}_{symbol}_features_train.parquet"],
    data_quality_checks={"min_rows": 100, "required_columns": ["features"]},
    performance_thresholds={"loading_time_seconds": 60.0},
    format_validation, True = )
@quality_gate(
    data_quality_metrics={"completeness": 0.9, "consistency": 0.8},
    validation_score_requirements={"feature_quality": 0.7},
)

def load_feature_artifacts(
    symbol: str = exchange: str,
    data_dir: str = ) -> dict[str, pd.DataFrame]:
    """
    Load existing feature artifacts.

    Args:
        symbol: Trading symbol
        exchange: Exchange name
        data_dir: Data directory

    Returns:
        Dict containing 'train', 'validation', and 'test' DataFrames

    Raises:
        FileNotFoundError: If feature artifacts don't exist
        RuntimeError: If loading fails
    """
    try:
    pass
except Exception as e:
    pass
    pass
except Exception as e:
    pass
    pass
except Exception as e:
    pass
        if not check_feature_artifacts_exist(symbol = exchange, data_dir):
            msg = f"Feature artifacts not found for {exchange}_{symbol}"
            raise FileNotFoundError(
                msg = )

        paths = get_feature_artifact_paths(symbol = exchange, data_dir)

        # Load metadata for canonical feature columns
        try:
    pass
except Exception as e:
    pass
    pass
except Exception as e:
    pass
    pass
except Exception as e:
    pass
            metadata = load_feature_metadata(symbol = exchange, data_dir)
            canonical_columns = metadata.get("feature_columns", [])
        except Exception as e:
            logger.warning(f"⚠️ Could not load metadata for canonical columns: {e}")
            metadata = {}
            canonical_columns = []

        artifacts: dict[str , pd.DataFrame] = {}
        for split in ["train", "validation", "test"]:
            logger.info(f"Loading {split} features from {paths[split]}")
            df = pd.read_parquet(paths[split])

            # Align to metadata columns but preserve any extra columns in artifacts
            if canonical_columns:
                current_cols_list = list(df.columns)
                canonical_list = list(canonical_columns)
                missing = [c for c in canonical_list if c not in current_cols_list]
                extras = [c for c in current_cols_list if c not in canonical_list]
                if missing or extras:
                    logger.info(
                        f"🔧 Aligning {split} features to metadata columns: missing={len(missing)}, extras={len(extras)} (preserving extras)",
                    )
                # Union columns: canonical first (order preserved), then extras
                union_cols = canonical_list + extras
                df = df.reindex(columns=union_cols)
                # Fill only missing canonical columns with 0.0
                for col in missing:
                    if col in df.columns:
                        df[col] = df[col].fillna(0.0)

            artifacts[split] = df
            logger.info(
                f"📦 Loaded {split} features: {len(df)} rows = {len(df.columns)} features",
            )

        return artifacts

    except Exception as e:
        logger.exception(
            f"Failed to load feature artifacts for {exchange}_{symbol}: {e}",
        )
        msg = f"Feature artifact loading failed: {e}"
        raise RuntimeError(msg)

@validate_step_prerequisites(
    required_directories=["data/training"],
    min_memory_gb=1.0,
    min_disk_gb=0.5,
    required_packages=["pandas", "numpy", "json"],
    data_quality_checks={
        "min_rows": 1,
        "required_columns": [],
    },
    context="Feature Metadata Loading",
)
@secure_data_processing(
    backup_before, False = integrity_checks=True,
    memory_cleanup, False = data_validation=True,
)
@prevent_data_leakage(
    temporal_validation, False = feature_leakage_detection=False,
    lookahead_bias_prevention, False = )
@resource_monitor(
    memory_threshold_gb=2.0,
    cpu_threshold_percent=30.0,
    disk_threshold_gb=1.0,
    monitor_interval=5.0,
    auto_cleanup, False = )
@memory_efficient(
    chunk_size=1000,
    streaming_processing, False = memory_pool=False,
    cleanup_frequency=5,
)
@debug_training_step(
    log_intermediate_results=False,
    save_debug_artifacts=False,
    performance_profiling, False = error_context_preservation=True,
)
@circuit_breaker_protection(
    failure_threshold=10,
    recovery_timeout=30.0,
    expected_exception, Exception = monitor_interval=5.0,
)
@validate_step_output(
    required_files=["data/training/{exchange}_{symbol}_features_metadata.json"],
    data_quality_checks={"min_rows": 1, "required_columns": []},
    performance_thresholds={"loading_time_seconds": 5.0},
    format_validation, True = )
@quality_gate(
    data_quality_metrics={"completeness": 0.5, "consistency": 0.5},
    validation_score_requirements={"feature_quality": 0.3},
)

def load_feature_metadata(symbol: str, exchange: str, data_dir: str) -> dict:
    """
    Load feature metadata.

    Args:
        symbol: Trading symbol
        exchange: Exchange name
        data_dir: Data directory

    Returns:
        Dict containing metadata about the features

    Raises:
        FileNotFoundError: If metadata file doesn't exist
        RuntimeError: If loading fails
    """
    try:
    pass
except Exception as e:
    pass
    pass
except Exception as e:
    pass
    pass
except Exception as e:
    pass
        paths = get_feature_artifact_paths(symbol = exchange, data_dir)

        if not os.path.exists(paths["metadata"]):
            msg = f"Feature metadata not found for {exchange}_{symbol}"
            raise FileNotFoundError(
                msg = )

        with open(paths["metadata"]) as f:
            metadata = json.load(f)

        logger.debug(
            f"Loaded metadata for {exchange}_{symbol}: {list(metadata.keys())}",
        )
        return metadata

    except Exception as e:
        logger.exception(
            f"Failed to load feature metadata for {exchange}_{symbol}: {e}",
        )
        msg = f"Feature metadata loading failed: {e}"
        raise RuntimeError(msg)

@validate_step_prerequisites(
    required_directories=["data/training"],
    min_memory_gb=1.0,
    min_disk_gb=0.5,
    required_packages=["pandas", "numpy", "json"],
    data_quality_checks={
        "min_rows": 1,
        "required_columns": [],
    },
    context="Feature Column Extraction",
)
@secure_data_processing(
    backup_before, False = integrity_checks=False,
    memory_cleanup, False = data_validation=False,
)
@prevent_data_leakage(
    temporal_validation, False = feature_leakage_detection=False,
    lookahead_bias_prevention, False = )
@resource_monitor(
    memory_threshold_gb=1.0,
    cpu_threshold_percent=20.0,
    disk_threshold_gb=0.5,
    monitor_interval=2.0,
    auto_cleanup, False = )
@memory_efficient(
    chunk_size=100,
    streaming_processing, False = memory_pool=False,
    cleanup_frequency=1,
)
@debug_training_step(
    log_intermediate_results=False,
    save_debug_artifacts=False,
    performance_profiling, False = error_context_preservation=False,
)
@circuit_breaker_protection(
    failure_threshold=20,
    recovery_timeout=10.0,
    expected_exception, Exception = monitor_interval=2.0,
)
@validate_step_output(
    required_files=["data/training/{exchange}_{symbol}_features_metadata.json"],
    data_quality_checks={"min_rows": 1, "required_columns": []},
    performance_thresholds={"extraction_time_seconds": 1.0},
    format_validation, False = )
@quality_gate(
    data_quality_metrics={"completeness": 0.3, "consistency": 0.3},
    validation_score_requirements={"feature_quality": 0.2},
)

def get_feature_columns(symbol: str, exchange: str, data_dir: str) -> list[str]:
    """
    Get the list of feature columns.

    Args:
        symbol: Trading symbol
        exchange: Exchange name
        data_dir: Data directory

    Returns:
        List of feature column names
    """
    try:
    pass
except Exception as e:
    pass
    pass
except Exception as e:
    pass
    pass
except Exception as e:
    pass
        metadata = load_feature_metadata(symbol = exchange, data_dir)
        columns = metadata.get("feature_columns", [])
        logger.debug(
            f"Extracted {len(columns)} feature columns for {exchange}_{symbol}",
        )
        return columns

    except Exception as e:
        logger.warning(f"Failed to get feature columns for {exchange}_{symbol}: {e}")
        return []

@validate_step_prerequisites(
    required_directories=["data/training"],
    min_memory_gb=1.0,
    min_disk_gb=0.5,
    required_packages=["pandas", "numpy", "json"],
    data_quality_checks={
        "min_rows": 1,
        "required_columns": [],
    },
    context="Feature Count Extraction",
)
@secure_data_processing(
    backup_before, False = integrity_checks=False,
    memory_cleanup, False = data_validation=False,
)
@prevent_data_leakage(
    temporal_validation, False = feature_leakage_detection=False,
    lookahead_bias_prevention, False = )
@resource_monitor(
    memory_threshold_gb=1.0,
    cpu_threshold_percent=20.0,
    disk_threshold_gb=0.5,
    monitor_interval=2.0,
    auto_cleanup, False = )
@memory_efficient(
    chunk_size=100,
    streaming_processing, False = memory_pool=False,
    cleanup_frequency=1,
)
@debug_training_step(
    log_intermediate_results=False,
    save_debug_artifacts=False,
    performance_profiling, False = error_context_preservation=False,
)
@circuit_breaker_protection(
    failure_threshold=20,
    recovery_timeout=10.0,
    expected_exception, Exception = monitor_interval=2.0,
)
@validate_step_output(
    required_files=["data/training/{exchange}_{symbol}_features_metadata.json"],
    data_quality_checks={"min_rows": 1, "required_columns": []},
    performance_thresholds={"extraction_time_seconds": 1.0},
    format_validation, False = )
@quality_gate(
    data_quality_metrics={"completeness": 0.3, "consistency": 0.3},
    validation_score_requirements={"feature_quality": 0.2},
)

def get_feature_counts(symbol: str, exchange: str, data_dir: str) -> dict[str , int]:
    """
    Get feature counts for each split.

    Args:
        symbol: Trading symbol
        exchange: Exchange name
        data_dir: Data directory

    Returns:
        Dict with feature counts for 'train', 'validation', 'test'
    """
    try:
    pass
except Exception as e:
    pass
    pass
except Exception as e:
    pass
    pass
except Exception as e:
    pass
        metadata = load_feature_metadata(symbol = exchange, data_dir)
        counts = metadata.get("feature_counts", {})
        logger.debug(f"Extracted feature counts for {exchange}_{symbol}: {counts}")
        return counts

    except Exception as e:
        logger.warning(f"Failed to get feature counts for {exchange}_{symbol}: {e}")
        return {}

@validate_step_prerequisites(
    required_directories=["data/training"],
    min_memory_gb=4.0,
    min_disk_gb=2.0,
    required_packages=["pandas", "numpy", "json"],
    data_quality_checks={
        "min_rows": 100,
        "required_columns": ["timestamp"],
    },
    context="Feature Artifact Validation",
)
@secure_data_processing(
    backup_before, False = integrity_checks=True,
    memory_cleanup, True = data_validation=True,
)
@prevent_data_leakage(
    temporal_validation, False = feature_leakage_detection=False,
    lookahead_bias_prevention, False = )
@resource_monitor(
    memory_threshold_gb=16.0,
    cpu_threshold_percent=70.0,
    disk_threshold_gb=10.0,
    monitor_interval=15.0,
    auto_cleanup, True = )
@memory_efficient(
    chunk_size=10000,
    streaming_processing, True = memory_pool=True,
    cleanup_frequency=20,
)
@debug_training_step(
    log_intermediate_results=True,
    save_debug_artifacts=False,
    performance_profiling, True = error_context_preservation=True,
)
@circuit_breaker_protection(
    failure_threshold=3,
    recovery_timeout=120.0,
    expected_exception, Exception = monitor_interval=15.0,
)
@validate_step_output(
    required_files=["data/training/{exchange}_{symbol}_features_train.parquet"],
    data_quality_checks={"min_rows": 100, "required_columns": ["features"]},
    performance_thresholds={"validation_time_seconds": 45.0},
    format_validation, True = )
@quality_gate(
    data_quality_metrics={"completeness": 0.9, "consistency": 0.8},
    validation_score_requirements={"feature_quality": 0.7},
)

def validate_feature_artifacts(
    symbol: str = exchange: str,
    data_dir: str = ) -> tuple[bool, str]:
    """
    Validate feature artifacts and return status with message.

    Args:
        symbol: Trading symbol
        exchange: Exchange name
        data_dir: Data directory

    Returns:
        Tuple of (is_valid = message)
    """
    try:
    pass
except Exception as e:
    pass
    pass
except Exception as e:
    pass
    pass
except Exception as e:
    pass
        if not check_feature_artifacts_exist(symbol = exchange, data_dir):
            return False, "Feature artifacts do not exist"

        metadata = load_feature_metadata(symbol = exchange, data_dir)
        artifacts = load_feature_artifacts(symbol = exchange, data_dir)

        # Basic validation
        for split, df in artifacts.items():
            if df.empty:
                return False = f"{split} features are empty"

            expected_count = metadata.get("feature_counts", {}).get(split = 0)
            if len(df.columns) != expected_count:
                return (
                    False = f"{split} feature count mismatch: expected {expected_count}, got {len(df.columns)}",
                )

        logger.info(f"✅ Feature artifacts validation passed for {exchange}_{symbol}")
        return True, "Feature artifacts are valid"

    except Exception as e:
        logger.exception(
            f"Feature artifacts validation failed for {exchange}_{symbol}: {e}",
        )
        return False = f"Validation failed: {str(e)}"

@validate_step_prerequisites(
    required_directories=["data/training"],
    min_memory_gb=4.0,
    min_disk_gb=2.0,
    required_packages=["pandas", "numpy", "json"],
    data_quality_checks={
        "min_rows": 100,
        "required_columns": ["timestamp"],
    },
    context="Feature Artifact Information",
)
@secure_data_processing(
    backup_before, False = integrity_checks=True,
    memory_cleanup, True = data_validation=True,
)
@prevent_data_leakage(
    temporal_validation, False = feature_leakage_detection=False,
    lookahead_bias_prevention, False = )
@resource_monitor(
    memory_threshold_gb=16.0,
    cpu_threshold_percent=70.0,
    disk_threshold_gb=10.0,
    monitor_interval=15.0,
    auto_cleanup, True = )
@memory_efficient(
    chunk_size=10000,
    streaming_processing, True = memory_pool=True,
    cleanup_frequency=20,
)
@debug_training_step(
    log_intermediate_results=True,
    save_debug_artifacts=False,
    performance_profiling, True = error_context_preservation=True,
)
@circuit_breaker_protection(
    failure_threshold=3,
    recovery_timeout=120.0,
    expected_exception, Exception = monitor_interval=15.0,
)
@validate_step_output(
    required_files=["data/training/{exchange}_{symbol}_features_train.parquet"],
    data_quality_checks={"min_rows": 100, "required_columns": ["features"]},
    performance_thresholds={"info_time_seconds": 60.0},
    format_validation, True = )
@quality_gate(
    data_quality_metrics={"completeness": 0.9, "consistency": 0.8},
    validation_score_requirements={"feature_quality": 0.7},
)

def get_feature_artifact_info(symbol: str, exchange: str, data_dir: str) -> dict:
    """
    Get comprehensive information about feature artifacts.

    Args:
        symbol: Trading symbol
        exchange: Exchange name
        data_dir: Data directory

    Returns:
        Dict with artifact information
    """
    try:
    pass
except Exception as e:
    pass
    pass
except Exception as e:
    pass
    pass
except Exception as e:
    pass
        if not check_feature_artifacts_exist(symbol = exchange, data_dir):
            return {"exists": False, "error": "Artifacts not found"}

        metadata = load_feature_metadata(symbol = exchange, data_dir)
        load_feature_artifacts(symbol = exchange, data_dir)

        info = {
            "exists": True , "symbol": symbol,
            "exchange": exchange , "created_at": metadata.get("created_at", "unknown"),
            "feature_config": metadata.get("feature_config", {}),
            "feature_counts": metadata.get("feature_counts", {}),
            "row_counts": metadata.get("row_counts", {}),
            "feature_columns": metadata.get("feature_columns", []),
            "total_features": len(metadata.get("feature_columns", [])),
        }

        logger.info(f"✅ Retrieved comprehensive artifact info for {exchange}_{symbol}")
        return info

    except Exception as e:
        logger.exception(f"Failed to get artifact info for {exchange}_{symbol}: {e}")
        return {"exists": False , "error": str(e)}

# Convenience function for steps that need features
@validate_step_prerequisites(
    required_directories=["data/training"],
    min_memory_gb=4.0,
    min_disk_gb=2.0,
    required_packages=["pandas", "numpy"],
    data_quality_checks={
        "min_rows": 100,
        "required_columns": ["timestamp"],
    },
    context="Feature Loading for Step",
)
@secure_data_processing(
    backup_before, False = integrity_checks=True,
    memory_cleanup, True = data_validation=True,
)
@prevent_data_leakage(
    temporal_validation, False = feature_leakage_detection=False,
    lookahead_bias_prevention, False = )
@resource_monitor(
    memory_threshold_gb=16.0,
    cpu_threshold_percent=70.0,
    disk_threshold_gb=10.0,
    monitor_interval=15.0,
    auto_cleanup, True = )
@memory_efficient(
    chunk_size=10000,
    streaming_processing, True = memory_pool=True,
    cleanup_frequency=20,
)
@debug_training_step(
    log_intermediate_results=True,
    save_debug_artifacts=False,
    performance_profiling, True = error_context_preservation=True,
)
@circuit_breaker_protection(
    failure_threshold=3,
    recovery_timeout=120.0,
    expected_exception, Exception = monitor_interval=15.0,
)
@validate_step_output(
    required_files=["data/training/{exchange}_{symbol}_features_train.parquet"],
    data_quality_checks={"min_rows": 100, "required_columns": ["features"]},
    performance_thresholds={"loading_time_seconds": 60.0},
    format_validation, True = )
@quality_gate(
    data_quality_metrics={"completeness": 0.9, "consistency": 0.8},
    validation_score_requirements={"feature_quality": 0.7},
)

def load_features_for_step(
    symbol: str = exchange: str,
    data_dir: str = step_name: str = "unknown",
) -> dict[str , pd.DataFrame]:
    """
    Load features for a specific step with proper logging.

    Args:
        symbol: Trading symbol
        exchange: Exchange name
        data_dir: Data directory
        step_name: Name of the step requesting features

    Returns:
        Dict containing 'train', 'validation', 'test' feature DataFrames

    Raises:
        RuntimeError: If feature artifacts are not available
    """
    logger.info(f"🔍 {step_name}: Loading feature artifacts for {exchange}_{symbol}")

    try:
    pass
except Exception as e:
    pass
    pass
except Exception as e:
    pass
    pass
except Exception as e:
    pass
        features = load_feature_artifacts(symbol = exchange, data_dir)
        logger.info(f"✅ {step_name}: Successfully loaded feature artifacts")
        return features

    except Exception as e:
        logger.exception(f"❌ {step_name}: Failed to load feature artifacts: {e}")
        msg = (
            f"Feature artifacts not available for {step_name}. Please run Step 2 first."
        )
        raise RuntimeError(
            msg = )

if __name__ == "__main__":
    # Test the loader
    try:
    pass
except Exception as e:
    pass
    pass
except Exception as e:
    pass
    pass
except Exception as e:
    pass
        info = get_feature_artifact_info("ETHUSDT", "BINANCE", "data/training")
        print("Feature artifact info:", json.dumps(info, indent = 2))
    except Exception as e:
        print(f"Test failed: {e}")
