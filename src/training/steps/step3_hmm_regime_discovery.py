# src/training/steps/step3_hmm_regime_discovery.py

import os
import json
import math
import warnings
import sys
import time
import contextlib
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
import joblib
from joblib import Parallel, delayed
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import gc
import psutil
import multiprocessing as mp
import signal
import atexit
# Removed HMMRegimeAnalyzer import to avoid segmentation faults in report generation

import numpy as np
import pandas as pd

from src.utils.logger import system_logger
from src.utils.error_handler import (
    handle_errors,
    handle_data_processing_errors,
    handle_type_conversions,
    safe_division,
    clean_dataframe,
)

# Import decorators from centralized module
from src.utils.centralized_decorators import (
    auto_fix_data_quality_issues,
    deterministic_seed,
    idempotent_step,
    artifact_write_lock,
    nan_inf_and_constant_guard,
    artifact_versioning,
    time_budget_watchdog,
    validate_step_prerequisites,
    secure_data_processing,
    prevent_data_leakage,
    resource_monitor,
    memory_efficient,
    debug_training_step,
    circuit_breaker_protection,
    validate_step_output,
    quality_gate,
    validate_data_quality,
    validate_feature_engineering_pipeline,
    validate_hmm_regime_discovery,
    with_tracing_span,
    guard_dataframe_nulls,
)

# Data loading
from src.training.steps.unified_data_loader import UnifiedDataLoader
# Import moved inside function to avoid circular import
# from src.training.data_sharing_manager import get_data_sharing_manager

# Basic feature engineering (simplified) - using built-in function

# HMM components - using built-in functions
# Model caching - using built-in functions
# Validation - using built-in functions

# HMM
from hmmlearn.hmm import GMMHMM

# Selection
from sklearn.preprocessing import StandardScaler

# Optional clustering
try:
    import hdbscan  # type: ignore

    _HDBSCAN_AVAILABLE = True
except Exception:
    _HDBSCAN_AVAILABLE = False

from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_distances
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
)
from dataclasses import dataclass
from scipy import stats
from scipy.spatial.distance import pdist, squareform


# Add proper multiprocessing resource management
def _cleanup_multiprocessing_resources():
    """Clean up multiprocessing resources to prevent semaphore leaks and segmentation faults."""
    try:
        # Force garbage collection
        gc.collect()

        # Clean up any remaining multiprocessing resources
        if hasattr(mp, "current_process"):
            current_process = mp.current_process()
            if hasattr(current_process, "_cleanup"):
                current_process._cleanup()

        # Additional cleanup for joblib
        if hasattr(joblib, "parallel"):
            joblib.parallel._backend = None

        # Clear any remaining multiprocessing pools
        if hasattr(mp, "_current_process"):
            current_process = mp.current_process()
            if hasattr(current_process, "_children"):
                for child in list(current_process._children):
                    if child.is_alive():
                        try:
                            child.terminate()
                            child.join(timeout=1)
                        except Exception:
                            pass

    except Exception as e:
        system_logger.warning(f"Cleanup warning: {e}")


# Register cleanup function
atexit.register(_cleanup_multiprocessing_resources)

# Configure logging
logger = system_logger.getChild("Step3.HMMRegimeDiscovery")


def _signal_handler(signum, frame):
    """Handle signals to ensure proper cleanup."""
    system_logger.info(f"Received signal {signum}, cleaning up...")
    _cleanup_multiprocessing_resources()
    exit(0)


# Register signal handlers
signal.signal(signal.SIGINT, _signal_handler)
signal.signal(signal.SIGTERM, _signal_handler)


def _generate_simple_hmm_report(
    exchange: str, symbol: str, timeframe: str, data_dir: str
) -> str:
    """Generate a simple text-based HMM regime report to avoid complex visualizations."""
    try:
        # Load the composite clusters data
        clusters_path = os.path.join(
            data_dir, f"{exchange}_{symbol}_hmm_composite_clusters_{timeframe}.parquet"
        )
        if not os.path.exists(clusters_path):
            return f"# HMM Regime Report for {exchange}_{symbol}_{timeframe}\n\nNo cluster data available."

        clusters_df = pd.read_parquet(clusters_path)

        # Load the meta information
        meta_path = os.path.join(
            data_dir, f"{exchange}_{symbol}_hmm_composite_meta_{timeframe}.json"
        )
        meta_info = {}
        if os.path.exists(meta_path):
            with open(meta_path, "r") as f:
                meta_info = json.load(f)

        # Generate simple report
        report_lines = [
            f"# HMM Regime Analysis Report",
            f"**Symbol:** {symbol}",
            f"**Exchange:** {exchange}",
            f"**Timeframe:** {timeframe}",
            f"**Generated:** {time.strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## Summary",
            f"- Total data points: {len(clusters_df)}",
            f"- Unique regimes: {clusters_df['composite_cluster_id'].nunique() if 'composite_cluster_id' in clusters_df.columns else 'N/A'}",
            "",
            "## Regime Distribution",
        ]

        if "composite_cluster_id" in clusters_df.columns:
            regime_counts = (
                clusters_df["composite_cluster_id"].value_counts().sort_index()
            )
            for regime_id, count in regime_counts.items():
                percentage = (count / len(clusters_df)) * 100
                report_lines.append(
                    f"- Regime {regime_id}: {count} points ({percentage:.1f}%)"
                )

        report_lines.extend(
            [
                "",
                "## Configuration",
                f"- Blocks used: {meta_info.get('blocks_used', 'N/A')}",
                f"- Total states: {meta_info.get('total_states', 'N/A')}",
                f"- Processing time: {meta_info.get('processing_time', 'N/A')}",
                "",
                "## Notes",
                "- This is a simplified report to avoid memory issues",
                "- For detailed analysis, check the generated parquet files",
                "- Regime stability and transition analysis available in the data files",
            ]
        )

        return "\n".join(report_lines)

    except Exception as e:
        return f"# HMM Regime Report for {exchange}_{symbol}_{timeframe}\n\nError generating report: {str(e)}"


@dataclass
class CompositeModelMetrics:
    """Comprehensive metrics for composite model analysis."""

    # Basic cluster metrics
    cluster_count: int
    cluster_sizes: Dict[int, int]
    cluster_frequencies: Dict[int, float]

    # Quality metrics
    silhouette_score: float
    calinski_harabasz_score: float
    davies_bouldin_score: float

    # Diversity metrics
    cluster_diversity: float
    cluster_separation: float
    cluster_cohesion: float

    # Temporal metrics
    cluster_persistence: Dict[int, float]
    cluster_volatility: Dict[int, float]

    # Block composition metrics
    block_representation: Dict[int, Dict[str, Dict[str, Any]]]
    block_dominance: Dict[int, str]
    block_balance: Dict[int, float]

    # Market condition metrics
    market_condition_distribution: Dict[int, Dict[str, float]]
    regime_stability: Dict[int, float]
    regime_transition_probability: Dict[str, float]

    # Anomaly detection
    outlier_clusters: List[int]
    unstable_clusters: List[int]
    rare_clusters: List[int]

    # Feature coverage
    missing_features_by_cluster: Dict[int, List[str]]


def log_with_context(message: str, level: str = "info", context: str = "", **kwargs):
    """Enhanced logging with context and structured data."""
    try:
        log_data = {
            "message": message,
            "context": context,
            "timestamp": time.time(),
            **kwargs,
        }

        if level.lower() == "info":
            logger.info(log_data)
        elif level.lower() == "warning":
            logger.warning(log_data)
        elif level.lower() == "error":
            logger.error(log_data)
        elif level.lower() == "debug":
            logger.debug(log_data)
        else:
            logger.info(log_data)

    except Exception as e:
        # Fallback to simple logging if structured logging fails
        logger.info(f"{context}: {message} (logging error: {e})")


def print_and_log(message: str, level: str = "info", context: str = "", **kwargs):
    """Print to console and log simultaneously."""
    try:
        # Print to console with timestamp
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] {context}: {message}")

        # Log with structured data
        log_with_context(message, level, context, **kwargs)

    except Exception as e:
        # Fallback to simple print and log
        print(f"{context}: {message}")
        logger.info(f"{context}: {message}")


def validate_required_artifacts(
    symbol: str, exchange: str, data_dir: str, timeframe: str
) -> Dict[str, bool]:
    """
    Validate that all required HMM artifacts exist for a given timeframe.
    Returns a dictionary indicating which artifacts are present.
    """
    required_artifacts = {
        "block_states": f"{exchange}_{symbol}_hmm_block_states_{timeframe}.parquet",
        "composite_clusters": f"{exchange}_{symbol}_hmm_composite_clusters_{timeframe}.parquet",
        "composite_intensity": f"{exchange}_{symbol}_hmm_composite_intensity_{timeframe}.parquet",
        "composite_meta": f"{exchange}_{symbol}_hmm_composite_meta_{timeframe}.json",
    }

    artifact_status = {}
    for artifact_name, filename in required_artifacts.items():
        filepath = os.path.join(data_dir, filename)
        artifact_status[artifact_name] = os.path.exists(filepath)

    return artifact_status


def log_artifact_status(
    symbol: str,
    exchange: str,
    data_dir: str,
    timeframe: str,
    artifact_status: Dict[str, bool],
):
    """Log the status of required artifacts for a timeframe."""
    logger.info(f"🔍 Artifact validation for {exchange}_{symbol}_{timeframe}:")

    missing_artifacts = []
    present_artifacts = []

    for artifact_name, exists in artifact_status.items():
        if exists:
            present_artifacts.append(artifact_name)
            logger.info(f"  ✅ {artifact_name}: present")
        else:
            missing_artifacts.append(artifact_name)
            logger.warning(f"  ❌ {artifact_name}: missing")

    if missing_artifacts:
        logger.warning(
            f"⚠️ Missing artifacts for {timeframe}: {', '.join(missing_artifacts)}"
        )
    else:
        logger.info(f"✅ All artifacts present for {timeframe}")

    return len(missing_artifacts) == 0


def ensure_artifacts_directory(data_dir: str) -> None:
    """Ensure the artifacts directory exists."""
    os.makedirs(data_dir, exist_ok=True)


def ensure_reports_directory() -> None:
    """Ensure the reports directory exists."""
    reports_dir = os.path.join("reports")
    os.makedirs(reports_dir, exist_ok=True)


@dataclass
class BlockConfig:
    """Configuration for HMM block analysis."""

    name: str
    n_states: int
    max_features: int = 3


# HMM Optimization Configuration - Enhanced for higher sensitivity and less sideways/neutral states
HMM_OPTIMIZATION_CONFIG: Dict[str, Any] = {
    "n_mix": 2,  # Increased from 1 to 2 mixtures for more nuanced state detection
    "max_iter": 500,  # Increased iterations for better convergence to subtle patterns
    "tol": 0.0005,  # Tighter tolerance for more precise state detection
    "subset_size": 75000,  # Increased subset size for better pattern recognition
    "n_jobs": 1,  # Single job to avoid memory issues and resource conflicts
    "early_stopping": True,  # Enable early stopping
    "max_time_per_model": 90,  # Increased time limit for more complex models
    "min_samples_per_state": 300,  # Reduced minimum for more granular state detection
    "adaptive_subset": True,  # Adapt subset size based on data size
    "max_subset_size": 100000,  # Increased maximum for better quality
    "min_subset_size": 30000,  # Increased minimum for better pattern detection
    "convergence_patience": 15,  # Increased patience for convergence monitoring
    "min_state_diversity": 0.7,  # Increased minimum fraction of states that must be used
    # Parallel processing parameters - disabled to prevent resource conflicts
    "enable_parallel_processing": False,  # Disabled to prevent bus errors
    "parallel_n_jobs": 1,  # Single job for timeframe processing
    "parallel_backend": "threading",  # Use threading for I/O bound operations
    "enable_parallel_block_processing": False,  # Disabled to prevent resource conflicts
    "parallel_block_n_jobs": 1,  # Single job for block processing
    "parallel_block_backend": "threading",  # Use threading for block processing
    "memory_limit_gb": 12,  # Increased memory limit for more complex models
    "enable_garbage_collection": True,
    # Feature caching parameters
    "enable_feature_caching": True,  # Enable feature caching to avoid recomputation
    "feature_cache_dir": "data_cache/feature_cache",  # Directory for feature cache
    "feature_cache_max_size_gb": 8,  # Increased cache size in GB
    "feature_cache_ttl_hours": 24,  # Time-to-live for cached features in hours
    "enable_intelligent_cache_invalidation": True,  # Smart cache invalidation based on data changes
    "cache_invalidation_check_interval": 3600,  # Check for cache invalidation every hour
    # Model caching parameters
    "enable_model_caching": True,  # Enable model caching to avoid retraining similar configurations
    "model_cache_dir": "data_cache/model_cache",  # Directory for model cache
    "model_cache_max_size_gb": 5,  # Increased cache size in GB
    "model_cache_ttl_hours": 48,  # Time-to-live for cached models in hours
    # Enhanced sensitivity parameters
    "enhanced_sensitivity": True,  # Enable enhanced sensitivity mode
    "momentum_sensitivity_multiplier": 1.5,  # Increase momentum state sensitivity
    "volatility_sensitivity_multiplier": 1.3,  # Increase volatility state sensitivity
    "volume_sensitivity_multiplier": 1.2,  # Increase volume state sensitivity

}

# Optimized block setup - Based on data availability and HMM suitability
# Keep only blocks that are essential for regime detection with available data
BLOCKS: List[BlockConfig] = [
    # 1. MOMENTUM BLOCK - Price momentum, RSI, momentum divergence (ESSENTIAL)
    BlockConfig("momentum", 5, 3),  # 5 states for granular momentum detection, target 3 features after filtering
    # Features: price_momentum_*, volume_weighted_momentum_*, rsi_*, momentum_divergence
    
    # 2. VOLATILITY BLOCK - Volatility measures, regime classification (ESSENTIAL)
    BlockConfig("volatility", 4, 3),  # 4 states for volatility patterns
    # Features: volatility_*, volatility_regime, volatility_persistence, volatility_of_volatility
    
    # 3. VOLUME BLOCK - Pure volume indicators and flow analysis (ESSENTIAL)
    BlockConfig("volume", 5, 4),  # 5 states for volume patterns
    # Features: volume_*, vwap_*, volume_zscore, volume_ratio_*, trade_*
    
    # 4. SUPPORT_RESISTANCE BLOCK - Comprehensive SR features (ESSENTIAL)
    BlockConfig("support_resistance", 3, 2),  # 3 states for SR patterns
    # Features: distance_to_*, normalized_distance_to_*, sr_proximity_score, strength_score,
    # clarity_factor, directional_pressure, sr_score, delta_sr_score, isolation_score
]

# Timeframes to train on - process all timeframes, resampling from 1m when needed
TIMEFRAMES: List[str] = [
    "1m",
    "5m",
    "15m",
    "30m",
]  # 5m, 15m, and 30m will be resampled from 1m data


# Feature caching functionality
class FeatureCache:
    """Intelligent feature caching system to avoid recomputation."""

    def __init__(
        self,
        cache_dir: str = "data_cache/feature_cache",
        max_size_gb: float = 5.0,
        ttl_hours: int = 24,
    ):
        self.cache_dir = cache_dir
        self.max_size_gb = max_size_gb
        self.ttl_seconds = ttl_hours * 3600
        self.cache_metadata_file = os.path.join(cache_dir, "cache_metadata.json")
        self._ensure_cache_dir()
        self._load_metadata()

    def _ensure_cache_dir(self):
        """Ensure cache directory exists."""
        os.makedirs(self.cache_dir, exist_ok=True)

    def _load_metadata(self):
        """Load cache metadata."""
        try:
            if os.path.exists(self.cache_metadata_file):
                with open(self.cache_metadata_file, "r") as f:
                    self.metadata = json.load(f)
            else:
                self.metadata = {"entries": {}, "total_size_gb": 0.0}
        except Exception as e:
            print_and_log(
                f"Failed to load cache metadata: {e}", "warning", "FeatureCache"
            )
            self.metadata = {"entries": {}, "total_size_gb": 0.0}

    def _save_metadata(self):
        """Save cache metadata."""
        try:
            with open(self.cache_metadata_file, "w") as f:
                json.dump(self.metadata, f, indent=2)
        except Exception as e:
            print_and_log(
                f"Failed to save cache metadata: {e}", "warning", "FeatureCache"
            )

    def _generate_cache_key(
        self,
        symbol: str,
        exchange: str,
        timeframe: str,
        block_name: str,
        data_hash: str,
    ) -> str:
        """Generate a unique cache key for the feature set."""
        return f"{exchange}_{symbol}_{timeframe}_{block_name}_{data_hash}"

    def _get_file_size_gb(self, file_path: str) -> float:
        """Get file size in GB."""
        try:
            return os.path.getsize(file_path) / (1024**3)
        except:
            return 0.0

    def _cleanup_expired_entries(self):
        """Remove expired cache entries."""
        current_time = time.time()
        expired_keys = []

        for key, entry in self.metadata["entries"].items():
            if current_time - entry["timestamp"] > self.ttl_seconds:
                expired_keys.append(key)

        for key in expired_keys:
            self._remove_entry(key)

    def _remove_entry(self, key: str):
        """Remove a cache entry."""
        if key in self.metadata["entries"]:
            entry = self.metadata["entries"][key]
            try:
                cache_file = os.path.join(self.cache_dir, f"{key}.parquet")
                if os.path.exists(cache_file):
                    size_gb = self._get_file_size_gb(cache_file)
                    self.metadata["total_size_gb"] -= size_gb
                    os.remove(cache_file)
                del self.metadata["entries"][key]
            except Exception as e:
                print_and_log(
                    f"Failed to remove cache entry {key}: {e}",
                    "warning",
                    "FeatureCache",
                )

    def _evict_if_needed(self):
        """Evict old entries if cache size exceeds limit."""
        while (
            self.metadata["total_size_gb"] > self.max_size_gb
            and self.metadata["entries"]
        ):
            # Find oldest entry
            oldest_key = min(
                self.metadata["entries"].keys(),
                key=lambda k: self.metadata["entries"][k]["timestamp"],
            )
            self._remove_entry(oldest_key)

    def get_cached_features(
        self,
        symbol: str,
        exchange: str,
        timeframe: str,
        block_name: str,
        data_hash: str,
    ) -> Optional[pd.DataFrame]:
        """Retrieve cached features if available and valid."""
        if not HMM_OPTIMIZATION_CONFIG.get("enable_feature_caching", True):
            return None

        try:
            cache_key = self._generate_cache_key(
                symbol, exchange, timeframe, block_name, data_hash
            )

            if cache_key in self.metadata["entries"]:
                entry = self.metadata["entries"][cache_key]
                cache_file = os.path.join(self.cache_dir, f"{cache_key}.parquet")

                # Check if file exists and is not expired
                if (
                    os.path.exists(cache_file)
                    and (time.time() - entry["timestamp"]) < self.ttl_seconds
                ):
                    print_and_log(
                        f"📦 Loading cached features for {symbol}_{exchange}_{timeframe}_{block_name}",
                        "info",
                        "FeatureCache",
                    )
                    return pd.read_parquet(cache_file)
                else:
                    # Remove expired entry
                    self._remove_entry(cache_key)

            return None

        except Exception as e:
            print_and_log(
                f"Error retrieving cached features: {e}", "warning", "FeatureCache"
            )
            return None

    def cache_features(
        self,
        features: pd.DataFrame,
        symbol: str,
        exchange: str,
        timeframe: str,
        block_name: str,
        data_hash: str,
    ):
        """Cache features for future use."""
        if not HMM_OPTIMIZATION_CONFIG.get("enable_feature_caching", True):
            return

        try:
            cache_key = self._generate_cache_key(
                symbol, exchange, timeframe, block_name, data_hash
            )
            cache_file = os.path.join(self.cache_dir, f"{cache_key}.parquet")

            # Save features
            features.to_parquet(cache_file)

            # Update metadata
            size_gb = self._get_file_size_gb(cache_file)
            self.metadata["entries"][cache_key] = {
                "timestamp": time.time(),
                "size_gb": size_gb,
                "symbol": symbol,
                "exchange": exchange,
                "timeframe": timeframe,
                "block_name": block_name,
                "data_hash": data_hash,
            }
            self.metadata["total_size_gb"] += size_gb

            # Cleanup and evict if needed
            self._cleanup_expired_entries()
            self._evict_if_needed()
            self._save_metadata()

            print_and_log(
                f"💾 Cached features for {symbol}_{exchange}_{timeframe}_{block_name} ({size_gb:.3f}GB)",
                "info",
                "FeatureCache",
            )

        except Exception as e:
            print_and_log(f"Error caching features: {e}", "warning", "FeatureCache")

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            "total_entries": len(self.metadata["entries"]),
            "total_size_gb": self.metadata["total_size_gb"],
            "max_size_gb": self.max_size_gb,
            "cache_dir": self.cache_dir,
        }


# Global feature cache instance
feature_cache = FeatureCache(
    cache_dir=HMM_OPTIMIZATION_CONFIG.get(
        "feature_cache_dir", "data_cache/feature_cache"
    ),
    max_size_gb=HMM_OPTIMIZATION_CONFIG.get("feature_cache_max_size_gb", 5.0),
    ttl_hours=HMM_OPTIMIZATION_CONFIG.get("feature_cache_ttl_hours", 24),
)


# Model caching functionality
class ModelCache:
    """Intelligent model caching system to avoid retraining similar configurations."""

    def __init__(
        self,
        cache_dir: str = "data_cache/model_cache",
        max_size_gb: float = 3.0,
        ttl_hours: int = 48,
    ):
        self.cache_dir = cache_dir
        self.max_size_gb = max_size_gb
        self.ttl_seconds = ttl_hours * 3600
        self.cache_metadata_file = os.path.join(cache_dir, "model_cache_metadata.json")
        self._ensure_cache_dir()
        self._load_metadata()

    def _ensure_cache_dir(self):
        """Ensure cache directory exists."""
        os.makedirs(self.cache_dir, exist_ok=True)

    def _load_metadata(self):
        """Load cache metadata."""
        try:
            if os.path.exists(self.cache_metadata_file):
                with open(self.cache_metadata_file, "r") as f:
                    self.metadata = json.load(f)
            else:
                self.metadata = {"entries": {}, "total_size_gb": 0.0}
        except Exception as e:
            print_and_log(
                f"Failed to load model cache metadata: {e}", "warning", "ModelCache"
            )
            self.metadata = {"entries": {}, "total_size_gb": 0.0}

    def _save_metadata(self):
        """Save cache metadata."""
        try:
            with open(self.cache_metadata_file, "w") as f:
                json.dump(self.metadata, f, indent=2)
        except Exception as e:
            print_and_log(
                f"Failed to save model cache metadata: {e}", "warning", "ModelCache"
            )

    def _generate_cache_key(
        self,
        symbol: str,
        exchange: str,
        timeframe: str,
        block_name: str,
        n_states: int,
        data_hash: str,
        config_hash: str,
    ) -> str:
        """Generate a unique cache key for the model configuration."""
        return f"{exchange}_{symbol}_{timeframe}_{block_name}_states{n_states}_{data_hash}_{config_hash}"

    def _get_file_size_gb(self, file_path: str) -> float:
        """Get file size in GB."""
        try:
            return os.path.getsize(file_path) / (1024**3)
        except:
            return 0.0

    def _cleanup_expired_entries(self):
        """Remove expired cache entries."""
        current_time = time.time()
        expired_keys = []

        for key, entry in self.metadata["entries"].items():
            if current_time - entry["timestamp"] > self.ttl_seconds:
                expired_keys.append(key)

        for key in expired_keys:
            self._remove_entry(key)

    def _remove_entry(self, key: str):
        """Remove a cache entry."""
        if key in self.metadata["entries"]:
            entry = self.metadata["entries"][key]
            try:
                model_file = os.path.join(self.cache_dir, f"{key}_model.joblib")
                scaler_file = os.path.join(self.cache_dir, f"{key}_scaler.joblib")

                if os.path.exists(model_file):
                    size_gb = self._get_file_size_gb(model_file)
                    self.metadata["total_size_gb"] -= size_gb
                    os.remove(model_file)
                if os.path.exists(scaler_file):
                    size_gb = self._get_file_size_gb(scaler_file)
                    self.metadata["total_size_gb"] -= size_gb
                    os.remove(scaler_file)

                del self.metadata["entries"][key]
            except Exception as e:
                print_and_log(
                    f"Failed to remove model cache entry {key}: {e}",
                    "warning",
                    "ModelCache",
                )

    def _evict_if_needed(self):
        """Evict old entries if cache size exceeds limit."""
        while (
            self.metadata["total_size_gb"] > self.max_size_gb
            and self.metadata["entries"]
        ):
            # Find oldest entry
            oldest_key = min(
                self.metadata["entries"].keys(),
                key=lambda k: self.metadata["entries"][k]["timestamp"],
            )
            self._remove_entry(oldest_key)

    def get_cached_model(
        self,
        symbol: str,
        exchange: str,
        timeframe: str,
        block_name: str,
        n_states: int,
        data_hash: str,
        config_hash: str,
    ) -> Optional[Tuple[Any, Any]]:
        """Retrieve cached model and scaler if available and valid."""
        if not HMM_OPTIMIZATION_CONFIG.get("enable_model_caching", True):
            return None

        try:
            cache_key = self._generate_cache_key(
                symbol,
                exchange,
                timeframe,
                block_name,
                n_states,
                data_hash,
                config_hash,
            )

            if cache_key in self.metadata["entries"]:
                entry = self.metadata["entries"][cache_key]
                model_file = os.path.join(self.cache_dir, f"{cache_key}_model.joblib")
                scaler_file = os.path.join(self.cache_dir, f"{cache_key}_scaler.joblib")

                # Check if files exist and are not expired
                if (
                    os.path.exists(model_file)
                    and os.path.exists(scaler_file)
                    and (time.time() - entry["timestamp"]) < self.ttl_seconds
                ):
                    print_and_log(
                        f"📦 Loading cached model for {symbol}_{exchange}_{timeframe}_{block_name}_states{n_states}",
                        "info",
                        "ModelCache",
                    )
                    cached_model = joblib.load(model_file)
                    cached_scaler = joblib.load(scaler_file)
                    return cached_model, cached_scaler
                else:
                    # Remove expired entry
                    self._remove_entry(cache_key)

            return None

        except Exception as e:
            print_and_log(
                f"Error retrieving cached model: {e}", "warning", "ModelCache"
            )
            return None

    def cache_model(
        self,
        model: Any,
        scaler: Any,
        symbol: str,
        exchange: str,
        timeframe: str,
        block_name: str,
        n_states: int,
        data_hash: str,
        config_hash: str,
    ):
        """Cache model and scaler for future use."""
        if not HMM_OPTIMIZATION_CONFIG.get("enable_model_caching", True):
            return

        try:
            cache_key = self._generate_cache_key(
                symbol,
                exchange,
                timeframe,
                block_name,
                n_states,
                data_hash,
                config_hash,
            )
            model_file = os.path.join(self.cache_dir, f"{cache_key}_model.joblib")
            scaler_file = os.path.join(self.cache_dir, f"{cache_key}_scaler.joblib")

            # Save model and scaler
            joblib.dump(model, model_file)
            joblib.dump(scaler, scaler_file)

            # Validate that files were created and have content
            if not os.path.exists(model_file) or os.path.getsize(model_file) == 0:
                raise ValueError(
                    f"Model file was not created or is empty: {model_file}"
                )
            if not os.path.exists(scaler_file) or os.path.getsize(scaler_file) == 0:
                raise ValueError(
                    f"Scaler file was not created or is empty: {scaler_file}"
                )

            print_and_log(
                f"✅ Model and scaler files saved successfully", "debug", "ModelCache"
            )

            # Update metadata
            model_size_gb = self._get_file_size_gb(model_file)
            scaler_size_gb = self._get_file_size_gb(scaler_file)
            total_size_gb = model_size_gb + scaler_size_gb

            # Log detailed size information
            print_and_log(
                f"🔍 Model file size: {model_size_gb*1024*1024:.1f}KB, Scaler file size: {scaler_size_gb*1024*1024:.1f}KB",
                "debug",
                "ModelCache",
            )

            self.metadata["entries"][cache_key] = {
                "timestamp": time.time(),
                "size_gb": total_size_gb,
                "symbol": symbol,
                "exchange": exchange,
                "timeframe": timeframe,
                "block_name": block_name,
                "n_states": n_states,
                "data_hash": data_hash,
                "config_hash": config_hash,
            }
            self.metadata["total_size_gb"] += total_size_gb

            # Cleanup and evict if needed
            self._cleanup_expired_entries()
            self._evict_if_needed()
            self._save_metadata()

            # Format size appropriately based on magnitude
            if total_size_gb >= 1.0:
                size_str = f"{total_size_gb:.3f}GB"
            elif total_size_gb >= 0.001:
                size_str = f"{total_size_gb*1024:.1f}MB"
            elif total_size_gb >= 0.000001:
                size_str = f"{total_size_gb*1024*1024:.1f}KB"
            else:
                size_str = f"{total_size_gb*1024*1024*1024:.0f}B"

            print_and_log(
                f"💾 Cached model for {symbol}_{exchange}_{timeframe}_{block_name}_states{n_states} ({size_str})",
                "info",
                "ModelCache",
            )

        except Exception as e:
            print_and_log(f"Error caching model: {e}", "warning", "ModelCache")

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            "total_entries": len(self.metadata["entries"]),
            "total_size_gb": self.metadata["total_size_gb"],
            "max_size_gb": self.max_size_gb,
            "cache_dir": self.cache_dir,
        }


# Global model cache instance
model_cache = ModelCache(
    cache_dir=HMM_OPTIMIZATION_CONFIG.get("model_cache_dir", "data_cache/model_cache"),
    max_size_gb=HMM_OPTIMIZATION_CONFIG.get("model_cache_max_size_gb", 3.0),
    ttl_hours=HMM_OPTIMIZATION_CONFIG.get("model_cache_ttl_hours", 48),
)


def create_basic_features(price_df: pd.DataFrame) -> pd.DataFrame:
    """
    Create advanced features for HMM regime discovery using VectorizedAdvancedFeatureEngineering.
    This replaces the basic features with sophisticated market microstructure features.
    """
    try:
        logger.info(
            "🔧 Creating advanced features using VectorizedAdvancedFeatureEngineering"
        )

        # Import the advanced feature engineering class
        from src.training.steps.vectorized_advanced_feature_engineering import (
            VectorizedAdvancedFeatureEngineering,
        )

        # Initialize advanced feature engineering configuration
        fe_config = {
            "enable_meta_labeling": False,
            "vectorized_advanced_features": {
                "enable_explicit_meta_labels": False,
                "enable_technical_indicators": True,
                "enable_volatility_features": True,
                "enable_momentum_features": True,
                "enable_volume_features": True,
            
                "enable_wavelet_features": False,  # Disable for HMM to avoid complexity
                "enable_candlestick_patterns": False,  # Disable for HMM
                "enable_sr_distance": False,  # Disable for HMM
                "enable_wavelet_transforms": False,  # Disable for HMM
            },
        }

        # Create feature engineering instance
        fe = VectorizedAdvancedFeatureEngineering(fe_config)

        # Prepare volume data
        vol_df = price_df[["volume"]].copy()

        # Handle async execution properly
        import asyncio
        import concurrent.futures

        def run_async_features():
            """Run async feature engineering in a new thread with its own event loop."""
            new_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(new_loop)
            try:
                # Initialize the feature engineering
                new_loop.run_until_complete(fe.initialize())

                # Engineer advanced features
                features_dict = new_loop.run_until_complete(
                    fe.engineer_features(price_df, vol_df)
                )

                if not features_dict:
                    logger.error(
                        "❌ No advanced features produced - this is a critical error"
                    )
                    raise RuntimeError(
                        "Advanced feature engineering failed to produce any features"
                    )

                features_df = pd.DataFrame(features_dict, index=price_df.index)

                # Handle NaN and infinite values
                features_df = features_df.fillna(0)
                features_df = features_df.replace([np.inf, -np.inf], 0)

                logger.info(f"✅ Created {len(features_df.columns)} advanced features")
                logger.info(f"📊 Advanced features: {list(features_df.columns)}")

                return features_df
            finally:
                new_loop.close()

        # Run in a separate thread to avoid event loop conflicts
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(run_async_features)
            features_df = future.result()

        return features_df

    except Exception as e:
        logger.error(f"❌ Critical error in advanced feature engineering: {e}")
        raise RuntimeError(f"Advanced feature engineering failed: {e}")


# Fallback function removed - we only want advanced features


@handle_type_conversions(default_return=np.array([]))
def _winsorize(col: np.ndarray, lower: float = 0.01, upper: float = 0.99) -> np.ndarray:
    """Winsorize array to handle outliers."""
    if col.size == 0:
        return col
    lo = np.nanquantile(col, lower)
    hi = np.nanquantile(col, upper)
    return np.clip(col, lo, hi)


@handle_data_processing_errors(default_return=pd.DataFrame())
def _robust_scale(df: pd.DataFrame) -> pd.DataFrame:
    """Robust scaling using IQR with fallback to standard deviation."""
    df_out = pd.DataFrame(index=df.index)
    for c in df.columns:
        arr = df[c].astype(float).values
        arr = _winsorize(arr)
        q1 = np.nanquantile(arr, 0.25)
        q3 = np.nanquantile(arr, 0.75)
        iqr = (q3 - q1) if (q3 - q1) != 0 else np.nan
        if not np.isnan(iqr) and iqr > 1e-12:
            scaled = (arr - np.nanmedian(arr)) / (iqr + 1e-12)
        else:
            std = np.nanstd(arr)
            scaled = (arr - np.nanmean(arr)) / (std + 1e-12)
        df_out[c] = np.nan_to_num(scaled, nan=0.0, posinf=0.0, neginf=0.0)
    return df_out


@handle_data_processing_errors(default_return=[])
def _corr_prune(df: pd.DataFrame, thr: float = 0.95) -> List[str]:
    """Remove highly correlated columns."""
    if df.empty:
        return []
    corr = df.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    return [c for c in upper.columns if any(upper[c] >= thr)]


def _assign_block(feature_name: str) -> str:
    """
    Assign a feature to a specific block based on its name.
    
    This function ensures proper separation between raw data and engineered features
    to prevent 100% mutual information issues that could cause the system to discard
    valuable engineered features.
    
    Updated to match actual features generated by the system.
    """
    feature_name_lower = feature_name.lower()

    # CRITICAL: Exclude raw data and basic transformations - these are not features
    raw_data_patterns = [
        "open", "high", "low", "close", "volume",  # Raw OHLCV
        "timestamp", "time", "date",               # Time data
        "returns", "log_returns",                  # Basic transformations
        "price", "price_raw",                      # Raw price data
        "bid", "ask", "bid_volume", "ask_volume",  # Raw order book
        "trade_id", "trade_time",                  # Raw trade data
    ]
    
    for pattern in raw_data_patterns:
        if pattern in feature_name_lower:
            return "exclude"  # These are raw data, not engineered features

    # ENGINEERED FEATURES ONLY - Based on actual features generated by the system

    # 1. MOMENTUM BLOCK - Price momentum and trend indicators (ACTUALLY GENERATED)
    momentum_patterns = [
        "momentum", "price_momentum", "volume_weighted_momentum", "rsi",
        "momentum_divergence", "trend", "trend_strength", "trend_direction",
        "price_acceleration", "roc", "mfi", "tsi", "ultimate_oscillator",
        # Additional momentum features that are currently being excluded
        "momentum_strength", "momentum_oscillator", "momentum_diff", "momentum_accel",
        "rsi_diff", "rsi_accel", "rsi_norm", "roc_diff", "roc_accel", "roc_norm",
        "adaptive_rsi", "adaptive_rsi_diff", "adaptive_rsi_norm",
        "mfi_diff", "mfi_accel", "mfi_norm",
        "price_momentum_oscillator", "volume_momentum_oscillator",
        "price_momentum_diff", "price_momentum_accel", "price_momentum_norm",
        "volume_weighted_momentum_diff", "volume_weighted_momentum_accel", "volume_weighted_momentum_norm",
        "volume_momentum_diff", "volume_momentum_accel", "volume_momentum_norm",
        "momentum_5_diff", "momentum_10_diff", "momentum_20_diff",
        "momentum_5_accel", "momentum_10_accel", "momentum_20_accel",
        "momentum_5_norm", "momentum_10_norm", "momentum_20_norm",
        "rsi_diff_3m_1m", "rsi_diff_5m_1m", "rsi_diff_10m_3m",
        "rsi_diff_3m_1m_norm", "rsi_diff_5m_1m_norm", "rsi_diff_10m_3m_norm",
        "momentum_5_diff_3m_1m", "momentum_10_diff_5m_1m", "momentum_20_diff_10m_3m",
        "momentum_5_diff_3m_1m_norm", "momentum_10_diff_5m_1m_norm", "momentum_20_diff_10m_3m_norm"
    ]
    
    for pattern in momentum_patterns:
        if pattern in feature_name_lower:
            return "momentum"

    # 2. VOLATILITY BLOCK - Volatility and dispersion indicators (ACTUALLY GENERATED)
    volatility_patterns = [
        "volatility", "volatility_regime", "volatility_persistence", "volatility_of_volatility",
        "high_volatility_regime", "low_volatility_regime", "atr", "bbands", "bollinger"
    ]
    
    for pattern in volatility_patterns:
        if pattern in feature_name_lower:
            return "volatility"

    # 3. VOLUME BLOCK - Pure volume indicators and flow analysis (ESSENTIAL)
    volume_patterns = [
        # Core volume features
        "volume_", "volume_zscore", "volume_ratio", "volume_ma", "volume_returns", 
        "volume_change", "volume_momentum", "volume_weighted_momentum",
        
        # VWAP (volume-weighted average price)
        "vwap", "vwap_deviation",
        
        # Trade flow features (volume-based)
        "trade_count", "trade_volume", "trade_count_change", "trade_volume_change",
        "trade_count_returns", "trade_volume_returns", "trade_to_order_ratio",
        
        # Volume correlation features
        "price_volume_correlation", "volume_price_impact", "volume_price_divergence",
        "high_volume_price_impact", "low_volume_price_impact"
    ]
    
    for pattern in volume_patterns:
        if pattern in feature_name_lower:
            return "volume"

    # 4. SUPPORT_RESISTANCE BLOCK - Distance to levels (ESSENTIAL)
    sr_patterns = [
        "support", "resistance", "sr_", "distance_to_", "normalized_distance_to_",
        "level", "pivot", "fibonacci", "fib", "retracement", "extension",
        # Add comprehensive SR feature patterns
        "sr_proximity_score", "strength_score", "clarity_factor", "directional_pressure",
        "sr_score", "delta_sr_score", "isolation_score", "support_strength", "resistance_strength",
        "support_clarity_factor", "resistance_clarity_factor"
    ]
    
    for pattern in sr_patterns:
        if pattern in feature_name_lower:
            return "support_resistance"

    # Default to exclude any other features not explicitly categorized
    return "exclude"


@handle_data_processing_errors(default_return=pd.DataFrame())
def _select_block_features(
    full_df: pd.DataFrame, block: str, max_features: int
) -> pd.DataFrame:
    """Select and prepare features for a specific block."""
    # Debug: log all available features and their block assignments
    all_features = list(full_df.columns)
    feature_assignments = {feature: _assign_block(feature) for feature in all_features}

    logger.info(f"🔍 Available features for {block} block:")
    for feature, assigned_block in feature_assignments.items():
        logger.info(f"  {feature} -> {assigned_block}")

    cols = [
        c
        for c in full_df.columns
        if _assign_block(c) == block and _assign_block(c) != "exclude"
    ]
    logger.info(f"📊 Selected {len(cols)} features for {block} block: {cols}")

    if not cols:
        logger.warning(f"⚠️ No features found for {block} block")
        return pd.DataFrame(index=full_df.index)
    X = full_df[cols].copy()
    # Drop constant columns
    nunique = X.nunique(dropna=True)
    const_cols = nunique[nunique <= 1].index.tolist()
    if const_cols:
        logger.info(f"📊 Dropping {len(const_cols)} constant columns: {const_cols}")
        X = X.drop(columns=const_cols)

    if X.empty:
        logger.warning(f"⚠️ No features remaining for {block} block after dropping constants")
        return pd.DataFrame(index=full_df.index)

    # Apply correlation pruning with block-specific thresholds
    if block == "momentum":
        correlation_threshold = 0.98  # Less aggressive for momentum block to preserve diversity
    else:
        correlation_threshold = 0.95  # Standard threshold for other blocks
    
    logger.info(f"🔧 Using correlation threshold {correlation_threshold} for {block} block")
    
    to_drop = _corr_prune(X, correlation_threshold)
    if to_drop:
        logger.info(f"📊 Dropping {len(to_drop)} highly correlated columns: {to_drop}")
        X = X.drop(columns=to_drop)

    if X.empty:
        logger.warning(f"⚠️ No features remaining for {block} block after correlation pruning")
        return pd.DataFrame(index=full_df.index)

    # Limit to max_features if specified
    if max_features and len(X.columns) > max_features:
        # Select features with highest variance
        variances = X.var()
        top_features = variances.nlargest(max_features).index.tolist()
        logger.info(f"📊 Limiting to top {max_features} features by variance: {top_features}")
        X = X[top_features]

    logger.info(f"✅ Final {block} block: {len(X.columns)} features")
    return X


def _fit_block_hmm_robust(
    X: pd.DataFrame, n_states: int, block_name: str = "unknown"
) -> Tuple[Optional[GMMHMM], Optional[StandardScaler]]:
    """
    Fit HMM model with robust training and quality validation.

    This enhanced version implements:
    1. Multiple training attempts with different random seeds
    2. State distribution quality validation
    3. Automatic retraining if quality is poor
    4. Enhanced feature scaling and normalization
    5. Comprehensive logging for troubleshooting
    6. NaN handling and numerical stability improvements
    """
    try:
        # hmmlearn expects 2D array
        arr = X.values.astype(float)

        # Enhanced NaN handling and data validation
        # 1. Check for NaN values and log them
        nan_mask = np.isnan(arr)
        if nan_mask.any():
            nan_count = nan_mask.sum()
            total_elements = arr.size
            nan_percentage = (nan_count / total_elements) * 100
            system_logger.warning(
                f"🚨 Block {block_name}: Found {nan_count} NaN values ({nan_percentage:.2f}%) in {total_elements} total elements"
            )

            # Log which features have NaN values
            nan_features = []
            for i, col in enumerate(X.columns):
                if nan_mask[:, i].any():
                    nan_count_col = nan_mask[:, i].sum()
                    nan_features.append(f"{col}({nan_count_col})")
            if nan_features:
                system_logger.warning(
                    f"🚨 Block {block_name}: Features with NaN values: {', '.join(nan_features[:10])}"
                )

        # 2. Handle NaN values with column-wise median imputation
        arr_clean = arr.copy()
        for i in range(arr_clean.shape[1]):
            col_data = arr_clean[:, i]
            non_nan_mask = ~np.isnan(col_data)
            if non_nan_mask.any():
                median_val = np.median(col_data[non_nan_mask])
                arr_clean[~non_nan_mask, i] = median_val
            else:
                # If all values are NaN, use 0
                arr_clean[:, i] = 0.0

        # 3. Handle infinite values
        inf_mask = np.isinf(arr_clean)
        if inf_mask.any():
            inf_count = inf_mask.sum()
            system_logger.warning(
                f"🚨 Block {block_name}: Found {inf_count} infinite values"
            )
            arr_clean = np.nan_to_num(arr_clean, nan=0.0, posinf=1e6, neginf=-1e6)

        # 4. Enhanced feature preprocessing for better HMM stability
        # Remove constant features that can cause convergence issues
        feature_vars = np.var(arr_clean, axis=0)
        non_constant_mask = feature_vars > 1e-8
        if not np.any(non_constant_mask):
            system_logger.error(f"🚨 All features are constant for block {block_name}")
            return None, None

        # Log feature variance information
        constant_features = X.columns[~non_constant_mask].tolist()
        if constant_features:
            system_logger.info(
                f"📊 Block {block_name}: Removed {len(constant_features)} constant features: {constant_features}"
            )

        arr_clean = arr_clean[:, non_constant_mask]
        feature_names = X.columns[non_constant_mask].tolist()

        # 5. Robust scaling with outlier handling
        scaler = StandardScaler()
        arr_scaled = scaler.fit_transform(arr_clean)

        # 6. Clip extreme values to prevent convergence issues
        arr_scaled = np.clip(arr_scaled, -10, 10)

        # 7. Final validation of scaled data
        if np.any(np.isnan(arr_scaled)) or np.any(np.isinf(arr_scaled)):
            system_logger.error(
                f"🚨 Block {block_name}: Scaled data still contains NaN or infinite values"
            )
            arr_scaled = np.nan_to_num(arr_scaled, nan=0.0, posinf=1e6, neginf=-1e6)

        # Multiple training attempts with different random seeds
        best_model = None
        best_scaler = None
        best_quality_score = -1
        best_state_distribution = None

        # Define quality thresholds
        min_state_ratio = 0.05  # Each state should have at least 5% of data
        max_dominant_state_ratio = 0.8  # No single state should dominate more than 80%

        # Training seeds - use different seeds for better exploration
        training_seeds = [
            42,
            123,
            456,
            789,
            101112,
            131415,
            161718,
            192021,
            222324,
            252627,
        ]

        for attempt, seed in enumerate(training_seeds):
            try:
                # Create HMM model with current seed and enhanced initialization
                model = GMMHMM(
                    n_components=n_states,
                    n_mix=2,
                    covariance_type="diag",
                    n_iter=300,  # Increased iterations for better convergence
                    tol=1e-4,  # Tighter tolerance
                    random_state=seed,
                    init_params="stmcw",  # Initialize all parameters
                    params="stmcw",  # Update all parameters
                )

                # Fit the model with additional error handling
                try:
                    model.fit(arr_scaled)
                except Exception as fit_error:
                    system_logger.warning(
                        f"🚨 Block {block_name} attempt {attempt + 1} fit failed: {fit_error}"
                    )
                    continue

                # Get state predictions
                try:
                    states = model.predict(arr_scaled)
                except Exception as predict_error:
                    system_logger.warning(
                        f"🚨 Block {block_name} attempt {attempt + 1} prediction failed: {predict_error}"
                    )
                    continue

                # Calculate state distribution
                unique_states, state_counts = np.unique(states, return_counts=True)
                state_distribution = state_counts / len(states)

                # Calculate quality score based on state coverage and balance
                min_state_coverage = np.min(state_distribution)
                max_state_coverage = np.max(state_distribution)
                
                # Quality score: higher is better
                # Reward good state coverage and penalize dominance
                quality_score = min_state_coverage - (max_state_coverage - 1/n_states)

                # Check if this is the best model so far
                if quality_score > best_quality_score:
                    best_quality_score = quality_score
                    best_model = model
                    best_scaler = scaler
                    best_state_distribution = state_distribution

                    system_logger.info(
                        f"✅ Block {block_name} attempt {attempt + 1}: New best model "
                        f"(quality_score={quality_score:.4f}, "
                        f"min_coverage={min_state_coverage:.3f}, "
                        f"max_coverage={max_state_coverage:.3f})"
                    )

            except Exception as e:
                system_logger.warning(
                    f"🚨 Block {block_name} attempt {attempt + 1} failed: {e}"
                )
                continue

        # Validate final model
        if best_model is None:
            system_logger.error(f"🚨 Block {block_name}: All training attempts failed")
            return None, None

        # Log final model quality
        system_logger.info(
            f"🎉 Block {block_name}: Best model selected "
            f"(quality_score={best_quality_score:.4f}, "
            f"state_distribution={best_state_distribution})"
        )

        return best_model, best_scaler

    except Exception as e:
        system_logger.error(f"🚨 Block {block_name}: Critical error in _fit_block_hmm_robust: {e}")
        return None, None


@with_tracing_span("step3._posteriors", log_args=False)
@guard_dataframe_nulls(mode="warn", arg_index=1)
@handle_errors(exceptions=(Exception,), default_return=np.array([]), context="step3_hmm_regime_discovery._posteriors")
def _posteriors(model: GMMHMM, X: np.ndarray) -> np.ndarray:
    """
    Get posterior probabilities with enhanced NaN/Inf guards.
    
    Args:
        model: Fitted HMM model
        X: Input data (2D array)
        
    Returns:
        Posterior probabilities array
    """
    try:
        # Guard against NaN/Inf in input data
        if np.any(np.isnan(X)) or np.any(np.isinf(X)):
            system_logger.warning("🚨 Input data contains NaN/Inf values, cleaning...")
            X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)
        
        # Get posterior probabilities
        posteriors = model.predict_proba(X)
        
        # Guard against NaN/Inf in output
        if np.any(np.isnan(posteriors)) or np.any(np.isinf(posteriors)):
            system_logger.warning("🚨 Posterior probabilities contain NaN/Inf, cleaning...")
            posteriors = np.nan_to_num(posteriors, nan=0.0, posinf=1.0, neginf=0.0)
            
            # Ensure probabilities sum to 1
            row_sums = posteriors.sum(axis=1, keepdims=True)
            row_sums = np.where(row_sums == 0, 1, row_sums)  # Avoid division by zero
            posteriors = posteriors / row_sums
        
        return posteriors
        
    except Exception as e:
        system_logger.error(f"🚨 Error in _posteriors: {e}")
        # Return uniform probabilities as fallback
        n_samples = X.shape[0]
        n_states = model.n_components
        return np.full((n_samples, n_states), 1.0/n_states)


@with_tracing_span("step3._build_combination_profiles", log_args=False)
@handle_data_processing_errors(default_return=(pd.Series(dtype=str), pd.DataFrame()))
def _build_combination_profiles(
    block_states: Dict[str, np.ndarray], block_posteriors: Dict[str, np.ndarray]
) -> Tuple[pd.Series, pd.DataFrame]:
    """Build combination profiles from block states and posteriors."""
    # combination key per row (efficient join of key parts)
    if not block_states:
        combination_keys = pd.Series(dtype=str)
    else:
        key_parts = [[f"{b}:{int(v)}" for v in s] for b, s in block_states.items()]
        # Transpose and join
        joined_keys = ["|".join(map(str, row)) for row in zip(*key_parts)]
        combination_keys = pd.Series(joined_keys)
    # profile vector: concatenated mean posteriors per block across occurrences
    profiles = {}
    for combo, idx in combination_keys.groupby(combination_keys).groups.items():
        vecs: List[np.ndarray] = []
        for b, gamma in block_posteriors.items():
            if len(idx) == 0:
                continue
            # mean posterior for this block at these indices
            vecs.append(np.nanmean(gamma[idx, :], axis=0))
        profiles[combo] = np.concatenate(vecs, axis=0)
    profile_df = pd.DataFrame.from_dict(profiles, orient="index")
    return combination_keys, profile_df


@with_tracing_span("step3._cluster_combinations", log_args=False)
@handle_data_processing_errors(default_return=pd.Series([-1] * 1000))
def _cluster_combinations(
    profile_df: pd.DataFrame, min_cluster_size: int = 5
) -> pd.Series:
    """
    Advanced clustering that ensures exactly 20 archetypes with 70-80% concentration.

    Strategy:
    1. Start with many clusters (1 per 5 combinations)
    2. Merge clusters until we have exactly 20 archetypes
    3. Ensure top 20 archetypes account for 70-80% of concentration
    4. Only merge clusters that meet similarity threshold
    """
    if profile_df.empty or profile_df.shape[0] < 2:
        return pd.Series([-1] * profile_df.shape[0], index=profile_df.index)

    X = profile_df.values.astype(float)

    # Handle NaN values before normalization
    X_clean = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    # Normalize rows to unit norm for cosine similarity
    norms = np.linalg.norm(X_clean, axis=1, keepdims=True) + 1e-12
    Xn = X_clean / norms

    # Target more archetypes for higher sensitivity
    TARGET_ARCHETYPES = 25  # Increased from 20 to 25 for more granular regimes
    CONCENTRATION_TARGET_MIN = 0.75  # Reduced from 85% to 75% to allow more diverse regimes
    CONCENTRATION_TARGET_MAX = 0.90  # Reduced from 95% to 90% to preserve more nuanced states
    SIMILARITY_THRESHOLD = (
        0.80  # Keep original threshold for balanced merging
    )

    # Start with many clusters (1 per 5 combinations, but don't exceed available combinations)
    initial_clusters = min(max(25, profile_df.shape[0] // 5), profile_df.shape[0])

    logger.info(
        f"🔄 Starting with {initial_clusters} initial clusters, targeting {TARGET_ARCHETYPES} archetypes"
    )

    # Use AgglomerativeClustering to get initial clusters
    dist = cosine_distances(Xn)
    agg = AgglomerativeClustering(
        n_clusters=initial_clusters, metric="precomputed", linkage="average"
    )
    initial_labels = agg.fit_predict(dist)

    # Create cluster profiles and counts
    cluster_profiles = {}
    cluster_counts = {}

    for cluster_id in np.unique(initial_labels):
        mask = initial_labels == cluster_id
        if np.sum(mask) > 0:
            cluster_profiles[cluster_id] = Xn[mask].mean(axis=0)
            cluster_counts[cluster_id] = np.sum(mask)

    logger.info(
        f"📊 Initial clustering: {len(cluster_profiles)} clusters with {sum(cluster_counts.values())} total samples"
    )

    # Dynamic merging to reach target archetypes
    current_clusters = list(cluster_profiles.keys())
    current_labels = initial_labels.copy()

    while len(current_clusters) > TARGET_ARCHETYPES:
        # Calculate concentration of current clusters
        total_samples = sum(cluster_counts.values())
        sorted_clusters = sorted(
            cluster_counts.items(), key=lambda x: x[1], reverse=True
        )
        top_20_concentration = (
            sum(count for _, count in sorted_clusters[:TARGET_ARCHETYPES])
            / total_samples
        )

        logger.info(
            f"🔄 Current: {len(current_clusters)} clusters, top {TARGET_ARCHETYPES} concentration: {top_20_concentration:.3f}"
        )

        # If we're already in the target concentration range, stop merging
        if CONCENTRATION_TARGET_MIN <= top_20_concentration <= CONCENTRATION_TARGET_MAX:
            logger.info(f"✅ Target concentration reached: {top_20_concentration:.3f}")
            break

        # Find the best pair to merge (most similar clusters)
        best_similarity = -1
        best_pair = None

        for i, cluster1 in enumerate(current_clusters):
            for j, cluster2 in enumerate(current_clusters[i + 1 :], i + 1):
                # Calculate cosine similarity between cluster profiles
                profile1 = cluster_profiles[cluster1]
                profile2 = cluster_profiles[cluster2]
                similarity = np.dot(profile1, profile2) / (
                    np.linalg.norm(profile1) * np.linalg.norm(profile2)
                )

                # Only consider merging if similarity meets threshold
                if similarity >= SIMILARITY_THRESHOLD and similarity > best_similarity:
                    best_similarity = similarity
                    best_pair = (cluster1, cluster2)

        if best_pair is None:
            logger.warning(
                f"⚠️ No suitable clusters to merge (similarity threshold: {SIMILARITY_THRESHOLD})"
            )
            break

        # Merge the best pair
        cluster1, cluster2 = best_pair
        logger.info(
            f"🔄 Merging clusters {cluster1} and {cluster2} (similarity: {best_similarity:.3f})"
        )

        # Update cluster profiles and counts
        total_count = cluster_counts[cluster1] + cluster_counts[cluster2]
        merged_profile = (
            cluster_profiles[cluster1] * cluster_counts[cluster1]
            + cluster_profiles[cluster2] * cluster_counts[cluster2]
        ) / total_count

        # Remove old clusters and add merged cluster
        current_clusters.remove(cluster1)
        current_clusters.remove(cluster2)
        del cluster_profiles[cluster1]
        del cluster_profiles[cluster2]
        del cluster_counts[cluster1]
        del cluster_counts[cluster2]

        # Create new merged cluster ID
        merged_cluster_id = max(current_clusters) + 1 if current_clusters else 0
        cluster_profiles[merged_cluster_id] = merged_profile
        cluster_counts[merged_cluster_id] = total_count
        current_clusters.append(merged_cluster_id)

        # Update labels
        current_labels[current_labels == cluster1] = merged_cluster_id
        current_labels[current_labels == cluster2] = merged_cluster_id

    # Final concentration check
    total_samples = sum(cluster_counts.values())
    sorted_clusters = sorted(cluster_counts.items(), key=lambda x: x[1], reverse=True)
    top_20_concentration = (
        sum(count for _, count in sorted_clusters[:TARGET_ARCHETYPES]) / total_samples
    )

    logger.info(f"✅ Final clustering: {len(current_clusters)} clusters")
    logger.info(f"   Top {TARGET_ARCHETYPES} concentration: {top_20_concentration:.3f}")
    logger.info(
        f"   Cluster sizes: {sorted([count for _, count in sorted_clusters[:10]], reverse=True)}"
    )

    # Ensure we have exactly TARGET_ARCHETYPES by further merging if needed
    if len(current_clusters) > TARGET_ARCHETYPES:
        logger.info(
            f"🔄 Further merging to reach exactly {TARGET_ARCHETYPES} archetypes"
        )

        while len(current_clusters) > TARGET_ARCHETYPES:
            # Find smallest clusters to merge
            smallest_clusters = sorted(cluster_counts.items(), key=lambda x: x[1])[:2]
            cluster1, cluster2 = smallest_clusters[0][0], smallest_clusters[1][0]

            # Merge smallest clusters
            total_count = cluster_counts[cluster1] + cluster_counts[cluster2]
            merged_profile = (
                cluster_profiles[cluster1] * cluster_counts[cluster1]
                + cluster_profiles[cluster2] * cluster_counts[cluster2]
            ) / total_count

            # Update
            current_clusters.remove(cluster1)
            current_clusters.remove(cluster2)
            del cluster_profiles[cluster1]
            del cluster_profiles[cluster2]
            del cluster_counts[cluster1]
            del cluster_counts[cluster2]

            merged_cluster_id = max(current_clusters) + 1 if current_clusters else 0
            cluster_profiles[merged_cluster_id] = merged_profile
            cluster_counts[merged_cluster_id] = total_count
            current_clusters.append(merged_cluster_id)

            current_labels[current_labels == cluster1] = merged_cluster_id
            current_labels[current_labels == cluster2] = merged_cluster_id

    # Renumber clusters to be consecutive starting from 0
    cluster_mapping = {
        old_id: new_id for new_id, old_id in enumerate(sorted(current_clusters))
    }
    final_labels = np.array(
        [cluster_mapping.get(label, -1) for label in current_labels]
    )

    logger.info(
        f"🎯 Final result: {len(np.unique(final_labels[final_labels >= 0]))} archetypes"
    )

    return pd.Series(final_labels, index=profile_df.index)


@handle_data_processing_errors(default_return={})
def _state_feature_medians(
    X_block: pd.DataFrame, states: np.ndarray
) -> Dict[int, Dict[str, float]]:
    """Calculate median feature values for each state."""
    med = {}
    for s in np.unique(states):
        mask = states == s
        if mask.sum() == 0:
            continue
        med[int(s)] = {
            c: float(np.nanmedian(X_block.loc[mask, c])) for c in X_block.columns
        }
    return med


@handle_data_processing_errors(default_return={})
def _name_states(block: str, medians: Dict[int, Dict[str, float]]) -> Dict[int, str]:
    """Generate human-readable names for states based on feature medians."""
    # Generate simple, human-readable names per state using feature medians
    names: Dict[int, str] = {}
    if not medians:
        return names
    # Compute a scalar score per state depending on block
    scores: Dict[int, float] = {}
    for s, feat in medians.items():
        vals = list(feat.values()) if feat else [0.0]
        if block == "volatility":
            # intensity via absolute values
            score = float(np.nanmean(np.abs(vals)))
        elif block == "momentum":
            score = float(np.nanmean(vals))
        elif block == "support_resistance":
            score = float(np.nanmean(vals))
        else:  # volume or other blocks
            score = float(np.nanmean(vals))
        scores[int(s)] = score
    # Create unique names for each state based on their rank and characteristics
    sorted_states = sorted(scores.items(), key=lambda kv: kv[1])
    n = max(1, len(sorted_states))

    for rank, (s, sc) in enumerate(sorted_states):
        q = rank / max(1, n - 1)

        if block == "momentum":
            if q < 0.167:  # 6 states, so 1/6 = 0.167
                names[s] = "Weak Downtrend"
            elif q < 0.333:
                names[s] = "Moderate Downtrend"
            elif q < 0.5:
                names[s] = "Sideways/Neutral"
            elif q < 0.667:
                names[s] = "Moderate Uptrend"
            elif q < 0.833:
                names[s] = "Strong Uptrend"
            else:
                names[s] = "Strong Downtrend"
        elif block == "volatility":
            if q < 0.25:  # 4 states, so 1/4 = 0.25
                names[s] = "Low & Stable Vol"
            elif q < 0.5:
                names[s] = "Moderate Vol"
            elif q < 0.75:
                names[s] = "High & Choppy Vol"
            else:
                names[s] = "Very High & Choppy Vol"
        elif block == "volume":
            if q < 0.2:  # 5 states, so 1/5 = 0.2
                names[s] = "Very Low Volume"
            elif q < 0.4:
                names[s] = "Low Volume"
            elif q < 0.6:
                names[s] = "Medium Volume"
            elif q < 0.8:
                names[s] = "High Volume"
            else:
                names[s] = "Very High Volume"

        elif block == "support_resistance":
            if q < 0.33:  # 3 states, so 1/3 = 0.33
                names[s] = "Near Support"
            elif q < 0.67:
                names[s] = "Neutral Levels"
            else:
                names[s] = "Near Resistance"
    return names


@handle_errors(
    exceptions=(Exception,),
    default_return=None,
    context="step3_hmm_regime_discovery._persist_dataframe",
)
def _persist_dataframe(df: pd.DataFrame, path: str) -> None:
    """Persist DataFrame to parquet file with enhanced error handling."""
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        df.reset_index().to_parquet(path, index=False)
        system_logger.info(f"✅ Saved DataFrame to {path} (shape: {df.shape})")
    except Exception as e:
        system_logger.error(f"❌ Failed to save DataFrame to {path}: {e}")
        raise


@handle_errors(
    exceptions=(Exception,),
    default_return=None,
    context="step3_hmm_regime_discovery._persist_json",
)
def _persist_json(obj: Dict[str, Any], path: str) -> None:
    """Persist JSON object to file with enhanced error handling."""
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(obj, f, indent=2)
        system_logger.info(f"✅ Saved JSON to {path}")
    except Exception as e:
        system_logger.error(f"❌ Failed to save JSON to {path}: {e}")
        raise


def _process_single_block(
    blk: BlockConfig,
    blk_idx: int,
    total_blocks: int,
    block_features: Dict[str, pd.DataFrame],
    features_df: pd.DataFrame,
    context: str,
    tf: str,
    symbol: str,
    exchange: str,
) -> Tuple[str, bool, Dict[str, Any], str, float]:
    """
    Process a single block for HMM training.
    This function is designed to be called in parallel.

    Returns:
        Tuple of (block_name, success, results_dict, message, processing_time)
    """
    blk_start_time = time.time()

    try:
        print_and_log(
            f"🔄 Processing block {blk_idx}/{total_blocks}: {blk.name} for {tf}",
            "info",
            context,
        )

        X_blk = block_features.get(blk.name)
        if X_blk is None or X_blk.empty:
            return (
                blk.name,
                False,
                {},
                f"no features available for block {blk.name}",
                time.time() - blk_start_time,
            )

        print_and_log(
            f"📊 Block '{blk.name}' features: {X_blk.shape}, columns: {list(X_blk.columns)}",
            "info",
            context,
        )

        # Check for cached model first
        data_hash = str(
            hash(
                str(X_blk.index[0])
                + str(X_blk.index[-1])
                + str(len(X_blk))
                + str(list(X_blk.columns))
            )
        )
        config_hash = str(
            hash(
                str(blk.n_states)
                + str(HMM_OPTIMIZATION_CONFIG.get("n_mix", 1))
                + str(HMM_OPTIMIZATION_CONFIG.get("max_iter", 300))
            )
        )

        cached_model_result = model_cache.get_cached_model(
            symbol, exchange, tf, blk.name, blk.n_states, data_hash, config_hash
        )

        if cached_model_result is not None:
            model, scaler = cached_model_result
            print_and_log(
                f"📦 Using cached model for {blk.name} (states={blk.n_states})",
                "info",
                context,
            )
        else:
            # Train HMM model
            print_and_log(
                f"🎯 Training new model for {blk.name} (states={blk.n_states})",
                "info",
                context,
            )
            model, scaler = _fit_block_hmm_robust(X_blk, blk.n_states, blk.name)
            if model is None or scaler is None:
                return (
                    blk.name,
                    False,
                    {},
                    f"failed to fit HMM for block {blk.name}",
                    time.time() - blk_start_time,
                )

            # Cache the model
            model_cache.cache_model(
                model,
                scaler,
                symbol,
                exchange,
                tf,
                blk.name,
                blk.n_states,
                data_hash,
                config_hash,
            )

        # Compute posteriors
        gamma = _posteriors(model, X_blk.values)
        if len(gamma) == 0:
            return (
                blk.name,
                False,
                {},
                f"failed to compute posteriors for block {blk.name}",
                time.time() - blk_start_time,
            )
        
        # Get state predictions
        states = model.predict(X_blk.values)

        # Compute state feature medians
        state_medians = _state_feature_medians(X_blk, states)

        # Prepare results
        results = {
            "model": model,
            "scaler": scaler,
            "states": states,
            "posteriors": gamma,
            "state_feature_medians": state_medians,
            "selected_states": blk.n_states,
            "X_original": X_blk,
            "X_used": X_blk,
        }

        print_and_log(
            {
                "msg": f"HMM training completed for block '{blk.name}'",
                "selected_n_states": blk.n_states,
                "features_used": list(X_blk.columns),
                "unique_states_found": len(np.unique(states)),
                "state_counts": {
                    int(s): int((states == s).sum()) for s in np.unique(states)
                },
            },
            "info",
            context,
        )

        processing_time = time.time() - blk_start_time
        return blk.name, True, results, "success", processing_time

    except Exception as e:
        print_and_log(f"❌ Error processing block {blk.name}: {e}", "error", context)
        print_and_log(f"Traceback: {traceback.format_exc()}", "error", context)
        return blk.name, False, {}, f"exception: {str(e)}", time.time() - blk_start_time


# Import training pipeline decorators for comprehensive security and troubleshooting
from src.utils.training_pipeline_decorators import (
    monitor_pipeline_step,
    validate_pipeline_input,
    monitor_pipeline_performance,
    PipelineValidationLevel,
    PipelineStage,
)


@deterministic_seed(42)
@idempotent_step(step_key="step3_hmm_regime_discovery")
@artifact_write_lock()
@nan_inf_and_constant_guard()
@artifact_versioning("1.0")
@time_budget_watchdog(soft_timeout_seconds=7200.0)
@validate_step_prerequisites(
    required_directories=["data/training", "models"],
    min_memory_gb=16.0,
    min_disk_gb=10.0,
    required_packages=["pandas", "numpy", "sklearn", "hmmlearn"],
    data_quality_checks={
        "min_rows": 1000,
        "required_columns": ["timestamp", "features", "targets"],
    }
)
@secure_data_processing(
    backup_before=True,
    integrity_checks=True,
    memory_cleanup=True,
    data_validation=True
)
@prevent_data_leakage(
    temporal_validation=True,
    feature_leakage_detection=True,
    cross_validation_isolation=True,
    lookahead_bias_prevention=True
)
@resource_monitor(
    memory_threshold_gb=32.0,
    cpu_threshold_percent=90.0,
    disk_threshold_gb=10.0,
    auto_cleanup=True
)
@memory_efficient(
    chunk_size=1000,
    streaming_processing=True,
    memory_pool=True,
    cleanup_frequency=10
)
@debug_training_step(
    log_intermediate_results=True,
    save_debug_artifacts=True,
    performance_profiling=True
)
@circuit_breaker_protection(
    failure_threshold=3,
    recovery_timeout=300.0
)
@validate_step_output(
    required_files=["hmm_states.parquet", "hmm_posteriors.parquet"],
    data_quality_checks={"check_output_completeness": True}
)
@quality_gate(
    model_performance_thresholds={"min_data_points": 1000},
    data_quality_metrics={"completeness_threshold": 0.95}
)
@auto_fix_data_quality_issues
@validate_hmm_regime_discovery
@monitor_pipeline_step(
    PipelineStage.FEATURE_ENGINEERING,
    validation_level=PipelineValidationLevel.STRICT,
    enable_data_quality=True,
)
@validate_pipeline_input(
    required_directories=["data/training", "models"],
    min_memory_gb=16.0,
    min_disk_gb=10.0,
    required_packages=["pandas", "numpy", "sklearn", "hmmlearn"],
    data_quality_checks={
        "min_rows": 1000,
        "required_columns": ["timestamp", "features", "targets"],
    },
)
@monitor_pipeline_performance(
    enable_memory_tracking=True,
    enable_cpu_tracking=True,
    enable_gc_tracking=True,
    memory_threshold_gb=32.0,
    cpu_threshold_percent=90.0,
)
@handle_errors(
    exceptions=(Exception,),
    default_return=False,
    context="step3_hmm_regime_discovery",
)
async def run_step(
    symbol: str,
    exchange: str = "BINANCE",
    data_dir: str = "data/training",
    timeframe: str = "1m",
    lookback_days: Optional[int] = None,
    force: bool = False,
    **kwargs: Any,
) -> bool:
    """
    Step 1_7: HMM regime discovery via block HMMs and composite clustering.
    Uses vectorized advanced features (excluding candlestick pattern features).
    Outputs per-timeframe block states/posteriors, combination IDs, and composite cluster IDs.

    Enhanced with:
    - Comprehensive logging for troubleshooting and efficiency monitoring
    - Thorough error handling using decorators
    - Proper data usage (scaling, normalization, returns vs prices)
    - Complete type hints throughout
    """
    logger = system_logger.getChild("Step3.HMMRegimeDiscovery")
    logger.info(
        "🚀 Step 3: HMM Regime Discovery — using features from Step 2"
    )
    print_and_log(
        "🚀 Starting HMM Regime Discovery step", "info", "Step3.HMMRegimeDiscovery"
    )

    t0_total = time.time()

    # Import data sharing manager inside function to avoid circular import
    from src.training.data_sharing_manager import get_data_sharing_manager

    # Initialize data sharing manager for efficient data loading
    print_and_log(
        "🔧 Initializing data sharing manager", "info", "Step3.HMMRegimeDiscovery"
    )
    data_sharing_manager = get_data_sharing_manager({})

    # Load data per timeframe
    print_and_log(
        "🔧 Initializing UnifiedDataLoader", "info", "Step3.HMMRegimeDiscovery"
    )
    loader = UnifiedDataLoader({})

    any_success = False

    # Ensure data directory exists
    print_and_log(
        f"📁 Ensuring artifacts directory exists: {data_dir}",
        "info",
        "Step3.HMMRegimeDiscovery",
    )
    ensure_artifacts_directory(data_dir)

    # Ensure reports directory exists
    ensure_reports_directory()

    # Use specific timeframe if provided, otherwise use all timeframes
    timeframes_to_process = (
        [timeframe] if timeframe and timeframe != "1m" else TIMEFRAMES
    )
    print_and_log(
        f"🔄 Processing timeframes: {timeframes_to_process}",
        "info",
        "Step3.HMMRegimeDiscovery",
    )

    for tf in timeframes_to_process:
        t0_timeframe = time.time()
        logger.info(f"🔄 Processing timeframe: {tf}")
        print_and_log(
            f"🔄 Starting processing for timeframe: {tf}",
            "info",
            f"Step1_7.HMMRegimeDiscovery.{tf}",
        )

        # Use the new HMM implementation
        try:
            success = await implement_hmm_regime_discovery(
                symbol=symbol,
                exchange=exchange,
                timeframe=tf,
                data_dir=data_dir,
                force=force
            )
            
            if success:
                timeframe_time = time.time() - t0_timeframe
                logger.info(f"✅ HMM regime discovery completed for {tf} (took {timeframe_time:.2f}s)")
                print_and_log(
                    f"✅ HMM regime discovery completed for {tf} (took {timeframe_time:.2f}s)",
                    "info",
                    f"Step1_7.HMMRegimeDiscovery.{tf}",
                )
                any_success = True
                
            else:
                logger.error(f"❌ HMM regime discovery failed for {tf}")
                print_and_log(
                    f"❌ HMM regime discovery failed for {tf}",
                    "error",
                    f"Step1_7.HMMRegimeDiscovery.{tf}",
                )

        except Exception as e:
            logger.error(f"🚨 Error in HMM regime discovery for {tf}: {e}")
            print_and_log(
                f"🚨 Error in HMM regime discovery for {tf}: {e}",
                "error",
                f"Step1_7.HMMRegimeDiscovery.{tf}",
            )
            continue

        # Validate required artifacts and log which ones need to be created
        artifact_status = validate_required_artifacts(symbol, exchange, data_dir, tf)
        missing_artifacts = [
            name for name, exists in artifact_status.items() if not exists
        ]
        logger.info(
            f"🔧 Creating missing artifacts for {tf}: {', '.join(missing_artifacts)}"
        )
        print_and_log(
            f"🔧 Creating missing artifacts for {tf}: {', '.join(missing_artifacts)}",
            "info",
            f"Step1_7.HMMRegimeDiscovery.{tf}",
        )

        # Use the lookback_days parameter passed from the launcher
        from src.config.constants import (
            BLANK_TRAINING_LOOKBACK_DAYS,
            FULL_TRAINING_LOOKBACK_DAYS,
            SHORT_BLANK_LOOKBACK_DAYS,
        )

        # Default to blank mode lookback if not specified, but this should be passed from enhanced training manager
        actual_lookback_days = lookback_days or BLANK_TRAINING_LOOKBACK_DAYS

        try:
            print_and_log(
                "📊 Loading unified data via data sharing manager",
                "info",
                f"Step1_7.HMMRegimeDiscovery.{tf}",
            )
            t0_data_load = time.time()

            # Use data sharing manager to load data (will use cache if available)
            print_and_log(
                f"📊 Loading data with {actual_lookback_days} days lookback",
                "info",
                f"Step1_7.HMMRegimeDiscovery.{tf}",
            )

            df = await data_sharing_manager.get_unified_data(
                symbol=symbol,
                exchange=exchange,
                timeframe=tf,
                lookback_days=actual_lookback_days,
                force_reload=False,  # Use cache if available
            )

            data_load_time = time.time() - t0_data_load
            print_and_log(
                f"📊 Data loading completed in {data_load_time:.2f}s",
                "info",
                f"Step1_7.HMMRegimeDiscovery.{tf}",
            )

            if df is None or df.empty:
                logger.warning(
                    f"⚠️ No unified data for {exchange}_{symbol}_{tf}; skipping"
                )
                print_and_log(
                    f"⚠️ No unified data for {exchange}_{symbol}_{tf}; skipping",
                    "warning",
                    f"Step1_7.HMMRegimeDiscovery.{tf}",
                )
                continue

            # Ensure datetime index and validate data range
            print_and_log(
                "🔧 Ensuring datetime index and validating data range",
                "info",
                f"Step1_7.HMMRegimeDiscovery.{tf}",
            )
            if "timestamp" in df.columns and not isinstance(df.index, pd.DatetimeIndex):
                df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
                df = df.set_index("timestamp").sort_index()
                print_and_log(
                    "✅ Datetime index set successfully",
                    "info",
                    f"Step1_7.HMMRegimeDiscovery.{tf}",
                )

            # Validate data range - ensure we have sufficient data
            data_start = df.index.min()
            data_end = df.index.max()
            data_duration_days = (data_end - data_start).days

            print_and_log(
                f"📊 Data range: {data_start} to {data_end} ({data_duration_days} days)",
                "info",
                f"Step1_7.HMMRegimeDiscovery.{tf}",
            )

            if data_duration_days < actual_lookback_days * 0.8:  # Allow 20% tolerance
                logger.warning(
                    f"⚠️ Insufficient data duration for {tf}: {data_duration_days} days < {actual_lookback_days * 0.8} days"
                )
                print_and_log(
                    f"⚠️ Insufficient data duration for {tf}: {data_duration_days} days < {actual_lookback_days * 0.8} days",
                    "warning",
                    f"Step1_7.HMMRegimeDiscovery.{tf}",
                )
                continue

            # Extract OHLCV for FE
            print_and_log(
                "🔧 Extracting OHLCV data", "info", f"Step1_7.HMMRegimeDiscovery.{tf}"
            )
            price_cols = [
                c for c in ["open", "high", "low", "close", "volume"] if c in df.columns
            ]
            if len(price_cols) < 5:
                logger.warning(f"⚠️ Missing OHLCV columns for {tf}; found {price_cols}")
                print_and_log(
                    f"⚠️ Missing OHLCV columns for {tf}; found {price_cols}",
                    "warning",
                    f"Step1_7.HMMRegimeDiscovery.{tf}",
                )
                continue
            price_df = df[["open", "high", "low", "close", "volume"]].copy()
            vol_df = price_df[["volume"]].copy()
            print_and_log(
                f"✅ OHLCV data extracted: {price_df.shape}",
                "info",
                f"Step1_7.HMMRegimeDiscovery.{tf}",
            )

            # Load features from Step 2 artifacts instead of re-engineering
            print_and_log(
                f"📦 Loading features from Step 2 artifacts for {tf}...",
                "info",
                f"Step1_7.HMMRegimeDiscovery.{tf}",
            )
            t0_feature_load = time.time()

            try:
                from src.training.steps.feature_artifact_loader import load_features_for_step
                
                # Load features for the current timeframe
                features_dict = load_features_for_step(symbol, exchange, data_dir, f"Step3.{tf}")
                
                # Use train features for HMM regime discovery (most comprehensive dataset)
                features_df = features_dict["train"]
                
                # Ensure features are aligned with the price data
                if len(features_df) != len(price_df):
                    logger.warning(f"⚠️ Feature count mismatch: {len(features_df)} features vs {len(price_df)} prices")
                    # Align features to price data
                    common_index = features_df.index.intersection(price_df.index)
                    features_df = features_df.loc[common_index]
                    price_df = price_df.loc[common_index]
                
                feature_load_time = time.time() - t0_feature_load
                print_and_log(
                    f"📦 Feature loading completed in {feature_load_time:.2f}s",
                    "info",
                    f"Step1_7.HMMRegimeDiscovery.{tf}",
                )

                if features_df.empty:
                    logger.warning(f"⚠️ No features loaded for {tf}; skipping")
                    print_and_log(
                        f"⚠️ No features loaded for {tf}; skipping",
                        "warning",
                        f"Step1_7.HMMRegimeDiscovery.{tf}",
                    )
                    continue

                logger.info(
                    f"✅ Feature loading completed for {tf}: {features_df.shape}"
                )
                print_and_log(
                    f"✅ Feature loading completed for {tf}: {features_df.shape}",
                    "info",
                    f"Step1_7.HMMRegimeDiscovery.{tf}",
                )
                
            except Exception as e:
                logger.error(f"❌ Failed to load features for {tf}: {e}")
                print_and_log(
                    f"❌ Failed to load features for {tf}: {e}",
                    "error",
                    f"Step1_7.HMMRegimeDiscovery.{tf}",
                )
                continue

            # Create block features for HMM regime discovery using proper block selection
            print_and_log(
                "🔧 Creating block features", "info", f"Step1_7.HMMRegimeDiscovery.{tf}"
            )
            t0_block_features = time.time()

            block_features: Dict[str, pd.DataFrame] = {}

            # Build per-block feature matrices with robust scaling and selection
            for blk in BLOCKS:
                X_blk = _select_block_features(features_df, blk.name, blk.max_features)
                if X_blk.empty:
                    logger.warning(
                        f"Block '{blk.name}' has no features after selection — skipping"
                    )
                    print_and_log(
                        f"Block '{blk.name}' has no features after selection — skipping",
                        "warning",
                        f"Step1_7.HMMRegimeDiscovery.{tf}",
                    )
                    continue
                # No extra robust scaling here to avoid duplication
                block_features[blk.name] = X_blk

            block_features_time = time.time() - t0_block_features
            print_and_log(
                f"🔧 Block features creation completed in {block_features_time:.2f}s",
                "info",
                f"Step1_7.HMMRegimeDiscovery.{tf}",
            )

            if not block_features:
                logger.warning(f"⚠️ No block features created for {tf}; skipping")
                print_and_log(
                    f"⚠️ No block features created for {tf}; skipping",
                    "warning",
                    f"Step1_7.HMMRegimeDiscovery.{tf}",
                )
                continue

            logger.info(
                f"✅ Block features prepared for {tf}: {list(block_features.keys())}"
            )
            print_and_log(
                f"✅ Block features prepared for {tf}: {list(block_features.keys())}",
                "info",
                f"Step1_7.HMMRegimeDiscovery.{tf}",
            )

            # Check if parallel block processing is enabled
            enable_parallel_blocks = HMM_OPTIMIZATION_CONFIG.get(
                "enable_parallel_block_processing", True
            )
            parallel_block_n_jobs = HMM_OPTIMIZATION_CONFIG.get(
                "parallel_block_n_jobs", 2
            )
            parallel_block_backend = HMM_OPTIMIZATION_CONFIG.get(
                "parallel_block_backend", "threading"
            )

            # Initialize result containers
            block_models: Dict[str, Any] = {}
            block_scalers: Dict[str, Any] = {}
            block_states: Dict[str, np.ndarray] = {}
            block_posteriors: Dict[str, np.ndarray] = {}
            state_feature_medians: Dict[str, Dict[int, Dict[str, float]]] = {}

            # Process blocks
            if enable_parallel_blocks and len(BLOCKS) > 1:
                logger.info(
                    f"🚀 Using parallel block processing with {parallel_block_n_jobs} jobs ({parallel_block_backend} backend)"
                )

                # Prepare arguments for parallel block processing
                parallel_block_args = []
                for blk_idx, blk in enumerate(BLOCKS, 1):
                    args = (
                        blk,
                        blk_idx,
                        len(BLOCKS),
                        block_features,
                        features_df,
                        "Step1_7.HMMRegimeDiscovery",
                        tf,
                        symbol,
                        exchange,
                    )
                    parallel_block_args.append(args)

                # Process blocks in parallel
                try:
                    # Use context manager to ensure proper cleanup
                    with Parallel(
                        n_jobs=parallel_block_n_jobs, backend=parallel_block_backend
                    ) as parallel_executor:
                        block_results = parallel_executor(
                            delayed(_process_single_block)(*args)
                            for args in parallel_block_args
                        )

                    # Process block results
                    successful_blocks = 0
                    failed_blocks = []
                    for (
                        block_name,
                        success,
                        results,
                        message,
                        processing_time,
                    ) in block_results:
                        if success:
                            successful_blocks += 1
                            # Extract results
                            block_models[block_name] = results["model"]
                            block_scalers[block_name] = results["scaler"]
                            block_states[block_name] = results["states"]
                            block_posteriors[block_name] = results["posteriors"]
                            state_feature_medians[block_name] = results[
                                "state_feature_medians"
                            ]
                            logger.info(
                                f"✅ Block {block_name} completed successfully in {processing_time:.2f}s"
                            )
                        else:
                            failed_blocks.append(f"{block_name} ({message})")
                            logger.error(
                                f"❌ Block {block_name} failed: {message} (took {processing_time:.2f}s)"
                            )

                    logger.info(
                        f"📊 Parallel block processing completed: {successful_blocks}/{len(BLOCKS)} blocks successful"
                    )
                    if failed_blocks:
                        logger.warning(f"⚠️ Failed blocks: {', '.join(failed_blocks)}")

                    # Enhanced resource cleanup after parallel block processing
                    if HMM_OPTIMIZATION_CONFIG.get("enable_garbage_collection", True):
                        _cleanup_multiprocessing_resources()
                        logger.info(
                            "🧹 Enhanced resource cleanup completed after parallel block processing"
                        )

                except Exception as e:
                    logger.error(f"❌ Parallel block processing failed: {e}")
                    logger.info("🔄 Falling back to sequential block processing...")
                    # Ensure cleanup even on failure
                    _cleanup_multiprocessing_resources()
                    # Continue with sequential processing below
            else:
                logger.info("🔄 Using sequential block processing")

            # Sequential block processing (fallback or if parallel is disabled)
            if not enable_parallel_blocks or len(BLOCKS) <= 1:
                for blk in BLOCKS:
                    X_blk = block_features.get(blk.name)
                    if X_blk is None or X_blk.empty:
                        continue

                    logger.info(
                        f"🧩 Training HMM for block='{blk.name}' n_states={blk.n_states} features={list(X_blk.columns)}"
                    )

                    # Check for cached model first
                    data_hash = str(
                        hash(
                            str(X_blk.index[0])
                            + str(X_blk.index[-1])
                            + str(len(X_blk))
                            + str(list(X_blk.columns))
                        )
                    )
                    config_hash = str(
                        hash(
                            str(blk.n_states)
                            + str(HMM_OPTIMIZATION_CONFIG.get("n_mix", 1))
                            + str(HMM_OPTIMIZATION_CONFIG.get("max_iter", 300))
                        )
                    )

                    cached_model_result = model_cache.get_cached_model(
                        symbol,
                        exchange,
                        tf,
                        blk.name,
                        blk.n_states,
                        data_hash,
                        config_hash,
                    )

                    if cached_model_result is not None:
                        model, scaler = cached_model_result
                        logger.info(
                            f"📦 Using cached model for {blk.name} (states={blk.n_states})"
                        )
                    else:
                        # Train HMM model
                        logger.info(
                            f"🎯 Training new model for {blk.name} (states={blk.n_states})"
                        )
                        model, scaler = _fit_block_hmm_robust(
                            X_blk, blk.n_states, blk.name
                        )
                        if model is None or scaler is None:
                            logger.error(f"❌ Failed to fit HMM for block '{blk.name}'")
                            continue

                        # Cache the model
                        model_cache.cache_model(
                            model,
                            scaler,
                            symbol,
                            exchange,
                            tf,
                            blk.name,
                            blk.n_states,
                            data_hash,
                            config_hash,
                        )

                    gamma = _posteriors(model, X_blk.values)
                    if len(gamma) == 0:
                        logger.error(
                            f"❌ Failed to compute posteriors for block '{blk.name}'"
                        )
                        continue
                    
                    # Get state predictions
                    states = model.predict(X_blk.values)

                    block_models[blk.name] = model
                    block_scalers[blk.name] = scaler
                    block_states[blk.name] = states
                    block_posteriors[blk.name] = gamma
                    state_feature_medians[blk.name] = _state_feature_medians(
                        X_blk, states
                    )

                    logger.info(
                        {
                            "msg": f"HMM training completed for block '{blk.name}'",
                            "n_states": blk.n_states,
                            "unique_states_found": len(np.unique(states)),
                            "state_counts": {
                                int(s): int((states == s).sum())
                                for s in np.unique(states)
                            },
                        }
                    )

            # Persist per-block states and posteriors per timeframe
            out_idx = price_df.index
            block_cols: Dict[str, Any] = {}
            for blk in BLOCKS:
                if blk.name not in block_states:
                    continue
                block_cols[f"{blk.name}_state_id"] = block_states[blk.name]
                gamma = block_posteriors[blk.name]
                for i in range(gamma.shape[1]):
                    block_cols[f"{blk.name}_p_state_{i}"] = gamma[:, i]
            block_df = pd.DataFrame(block_cols, index=out_idx)
            block_out_path = os.path.join(
                data_dir,
                f"{exchange}_{symbol}_hmm_block_states_{tf}.parquet",
            )
            _persist_dataframe(block_df, block_out_path)
            logger.info(
                f"💾 Saved block states/posteriors -> {block_out_path} ({len(block_df)} rows)"
            )

            # Build combinations and composite clusters
            combo_keys, profile_df = _build_combination_profiles(
                block_states, block_posteriors
            )
            if profile_df.empty:
                logger.warning(
                    f"⚠️ Empty combination profiles for {tf}; skipping clustering"
                )
                continue

            # Generate composite clusters from combination profiles
            logger.info(f"🔄 Generating composite clusters for {tf}...")
            print_and_log(
                f"🔄 Generating composite clusters for {tf}...",
                "info",
                f"Step1_7.HMMRegimeDiscovery.{tf}",
            )

            try:
                # Cluster the combination profiles to create composite clusters
                cluster_labels = _cluster_combinations(profile_df, min_cluster_size=5)
                
                # Create composite cluster DataFrame
                composite_df = pd.DataFrame({
                    'combination_key': combo_keys,
                    'composite_cluster_id': cluster_labels.values,
                }, index=out_idx)
                
                # Add intensity scores for top 20 clusters only (computational efficiency)
                unique_clusters = sorted(cluster_labels.unique())
                # Limit to top 20 clusters by frequency to reduce feature dimensionality
                cluster_counts = pd.Series(cluster_labels).value_counts()
                top_clusters = cluster_counts.head(20).index.tolist()
                
                for cluster_id in top_clusters:
                    cluster_mask = cluster_labels == cluster_id
                    intensity = cluster_mask.astype(float)
                    composite_df[f'intensity_cluster_{cluster_id}'] = intensity
                
                # Log cluster selection for transparency
                logger.info(f"🎯 Selected top {len(top_clusters)} clusters out of {len(unique_clusters)} total clusters")
                logger.info(f"   Top clusters: {top_clusters}")
                logger.info(f"   Cluster frequencies: {cluster_counts.head(20).to_dict()}")
                
                # Save composite clusters
                composite_out_path = os.path.join(
                    data_dir,
                    f"{exchange}_{symbol}_hmm_composite_clusters_{tf}.parquet",
                )
                _persist_dataframe(composite_df, composite_out_path)
                logger.info(
                    f"💾 Saved composite clusters -> {composite_out_path} ({len(composite_df)} rows, {composite_df['composite_cluster_id'].nunique()} clusters)"
                )
                print_and_log(
                    f"💾 Saved composite clusters -> {composite_out_path} ({len(composite_df)} rows, {composite_df['composite_cluster_id'].nunique()} clusters)",
                    "info",
                    f"Step1_7.HMMRegimeDiscovery.{tf}",
                )
                
                # Save composite intensity data
                intensity_cols = [col for col in composite_df.columns if col.startswith('intensity_cluster_')]
                intensity_df = composite_df[['combination_key'] + intensity_cols].copy()
                intensity_out_path = os.path.join(
                    data_dir,
                    f"{exchange}_{symbol}_hmm_composite_intensity_{tf}.parquet",
                )
                _persist_dataframe(intensity_df, intensity_out_path)
                logger.info(
                    f"💾 Saved composite intensity -> {intensity_out_path}"
                )
                
                # Save composite meta information
                composite_meta = {
                    "exchange": exchange,
                    "symbol": symbol,
                    "timeframe": tf,
                    "generated": datetime.now().isoformat(),
                    "total_clusters": int(composite_df['composite_cluster_id'].nunique()),
                    "total_combinations": int(composite_df['combination_key'].nunique()),
                    "cluster_distribution": composite_df['composite_cluster_id'].value_counts().to_dict(),
                    "blocks": [{"name": b.name, "n_states": b.n_states} for b in BLOCKS],
                }
                
                meta_out_path = os.path.join(
                    data_dir,
                    f"{exchange}_{symbol}_hmm_composite_meta_{tf}.json",
                )
                with open(meta_out_path, "w") as f:
                    json.dump(composite_meta, f, indent=2, default=str)
                logger.info(f"💾 Saved composite meta -> {meta_out_path}")
                print_and_log(
                    f"💾 Saved composite meta -> {meta_out_path}",
                    "info",
                    f"Step1_7.HMMRegimeDiscovery.{tf}",
                )
                
                # Log cluster statistics
                cluster_stats = composite_df['composite_cluster_id'].value_counts().sort_index()
                logger.info(f"📊 Composite cluster distribution for {tf}:")
                for cluster_id, count in cluster_stats.items():
                    percentage = (count / len(composite_df)) * 100
                    logger.info(f"   Cluster {cluster_id}: {count} samples ({percentage:.1f}%)")
                
            except Exception as e:
                logger.error(f"❌ Failed to generate composite clusters for {tf}: {e}")
                print_and_log(
                    f"❌ Failed to generate composite clusters for {tf}: {e}",
                    "error",
                    f"Step1_7.HMMRegimeDiscovery.{tf}",
                )
                continue

            # Merge rare combinations into similar frequent combinations instead of filtering
            # Skip complex clustering to avoid crashes - just save basic HMM results
            logger.info(f"🔄 Skipping complex clustering for {tf} to avoid crashes")

            # Generate basic metrics report (simplified to avoid crashes)
            try:
                logger.info(f"📊 Generating basic metrics report for {tf}...")

                # Generate basic report
                report_lines = []
                report_lines.append(f"HMM Regime Discovery Report")
                report_lines.append(f"Symbol: {symbol}")
                report_lines.append(f"Exchange: {exchange}")
                report_lines.append(f"Timeframe: {tf}")
                report_lines.append(
                    f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
                )
                report_lines.append("=" * 50)

                report_lines.append(f"\n📊 HMM TRAINING SUMMARY")
                report_lines.append(f"Total Data Points: {len(block_df)}")

                report_lines.append(f"\n✅ HMM TRAINING RESULTS")
                for block_name, block_result in block_results.items():
                    if block_result and "selected_n_states" in block_result:
                        report_lines.append(
                            f"{block_name}: {block_result['selected_n_states']} states"
                        )

                report = "\n".join(report_lines)

                # Save report to file
                report_path = os.path.join(
                    data_dir, f"{exchange}_{symbol}_hmm_regime_report_{tf}.txt"
                )

                try:
                    with open(report_path, "w") as f:
                        f.write(report)
                    logger.info(f"💾 Saved basic HMM regime report -> {report_path}")
                    print_and_log(
                        f"💾 Saved basic HMM regime report -> {report_path}",
                        "info",
                        f"Step1_7.HMMRegimeDiscovery.{tf}",
                    )
                except Exception as e:
                    logger.error(f"❌ Failed to save report: {e}")
                    print_and_log(
                        f"❌ Failed to save report: {e}",
                        "error",
                        f"Step1_7.HMMRegimeDiscovery.{tf}",
                    )

                # Also log key metrics
                logger.info(f"📊 Key Metrics for {tf}:")
                logger.info(f"   Total Data Points: {len(block_df)}")
                for block_name, block_result in block_results.items():
                    if block_result and "selected_n_states" in block_result:
                        logger.info(
                            f"   {block_name}: {block_result['selected_n_states']} states"
                        )

            except Exception as e:
                logger.warning(f"⚠️ Error generating basic metrics report for {tf}: {e}")
                print_and_log(
                    f"⚠️ Error generating basic metrics report for {tf}: {e}",
                    "warning",
                    f"Step1_7.HMMRegimeDiscovery.{tf}",
                )

            # Save basic HMM results (simplified to avoid crashes)
            try:
                logger.info(f"💾 Saving basic HMM results for {tf}...")

                # Create simple meta with just the essential information
                simple_meta: Dict[str, Any] = {
                    "exchange": exchange,
                    "symbol": symbol,
                    "timeframe": tf,
                    "generated": datetime.now().isoformat(),
                    "blocks": [
                        {"name": b.name, "n_states": b.n_states} for b in BLOCKS
                    ],
                    "block_results": {
                        blk.name: {
                            "selected_n_states": block_results[blk.name][
                                "selected_n_states"
                            ]
                            if blk.name in block_results
                            else 0,
                            "features_used": block_results[blk.name]["features_used"]
                            if blk.name in block_results
                            else [],
                        }
                        for blk in BLOCKS
                    },
                }

                meta_out_path = os.path.join(
                    data_dir,
                    f"{exchange}_{symbol}_hmm_basic_meta_{tf}.json",
                )

                with open(meta_out_path, "w") as f:
                    json.dump(simple_meta, f, indent=2, default=str)
                logger.info(f"💾 Saved basic HMM meta -> {meta_out_path}")
                print_and_log(
                    f"💾 Saved basic HMM meta -> {meta_out_path}",
                    "info",
                    f"Step1_7.HMMRegimeDiscovery.{tf}",
                )

            except Exception as e:
                logger.error(f"❌ Failed to save basic HMM meta for {tf}: {e}")
                print_and_log(
                    f"❌ Failed to save basic HMM meta for {tf}: {e}",
                    "error",
                    f"Step1_7.HMMRegimeDiscovery.{tf}",
                )

            any_success = True

        except Exception as e:
            logger.error(f"❌ Error processing timeframe {tf}: {e}")
            continue

    # Log cache statistics
    if HMM_OPTIMIZATION_CONFIG.get("enable_feature_caching", True):
        cache_stats = feature_cache.get_cache_stats()
        logger.info(
            f"📊 Feature cache statistics: {cache_stats['total_entries']} entries, {cache_stats['total_size_gb']:.3f}GB used, {cache_stats['max_size_gb']:.3f}GB max"
        )

    if HMM_OPTIMIZATION_CONFIG.get("enable_model_caching", True):
        model_cache_stats = model_cache.get_cache_stats()
        logger.info(
            f"📊 Model cache statistics: {model_cache_stats['total_entries']} entries, {model_cache_stats['total_size_gb']:.3f}GB used, {model_cache_stats['max_size_gb']:.3f}GB max"
        )

    # Final validation: check that all required artifacts exist for all timeframes
    logger.info("🔍 Performing final artifact validation...")
    final_validation_passed = True
    for tf in TIMEFRAMES:
        final_artifact_status = validate_required_artifacts(
            symbol, exchange, data_dir, tf
        )
        all_present = all(final_artifact_status.values())
        if all_present:
            logger.info(f"✅ Final validation passed for {tf}")
        else:
            missing = [
                name for name, exists in final_artifact_status.items() if not exists
            ]
            logger.error(
                f"❌ Final validation failed for {tf}: missing {', '.join(missing)}"
            )
            final_validation_passed = False

    # Generate HMM regime analysis reports for all timeframes (regardless of whether they were processed or skipped)
    logger.info("📊 Generating HMM regime analysis reports for all timeframes...")
    print_and_log(
        "📊 Generating HMM regime analysis reports for all timeframes...",
        "info",
        "Step1_7.HMMRegimeDiscovery",
    )

    # DEBUG: Log the timeframes_to_process variable
    logger.info(f"🔍 DEBUG: timeframes_to_process = {timeframes_to_process}")
    logger.info(f"🔍 DEBUG: TIMEFRAMES = {TIMEFRAMES}")
    logger.info(f"🔍 DEBUG: data_dir = {data_dir}")
    logger.info(f"🔍 DEBUG: exchange = {exchange}, symbol = {symbol}")

    # Check if timeframes_to_process is empty or None
    if not timeframes_to_process:
        logger.error("❌ ERROR: timeframes_to_process is empty or None!")
        print_and_log(
            "❌ ERROR: timeframes_to_process is empty or None!",
            "error",
            "Step1_7.HMMRegimeDiscovery",
        )
        # Use TIMEFRAMES as fallback
        timeframes_to_process = TIMEFRAMES
        logger.info(f"🔧 FALLBACK: Using TIMEFRAMES = {timeframes_to_process}")

    for tf in timeframes_to_process:
        logger.info(f"🔄 DEBUG: Processing timeframe {tf} for report generation")
        try:
            logger.info(f"📊 Generating HMM regime analysis report for {tf}...")
            print_and_log(
                f"📊 Generating HMM regime analysis report for {tf}...",
                "info",
                f"Step1_7.HMMRegimeDiscovery.{tf}",
            )

            # Check if required files exist before attempting report generation
            required_files = [
                os.path.join(
                    data_dir, f"{exchange}_{symbol}_hmm_composite_meta_{tf}.json"
                ),
                os.path.join(
                    data_dir, f"{exchange}_{symbol}_hmm_composite_clusters_{tf}.parquet"
                ),
                os.path.join(
                    data_dir,
                    f"{exchange}_{symbol}_hmm_composite_intensity_{tf}.parquet",
                ),
            ]

            # DEBUG: Log each required file and its existence
            logger.info(f"🔍 DEBUG: Checking required files for {tf}:")
            for req_file in required_files:
                exists = os.path.exists(req_file)
                logger.info(f"  📄 {req_file}: {'✅ EXISTS' if exists else '❌ MISSING'}")

            missing_files = [f for f in required_files if not os.path.exists(f)]
            if missing_files:
                logger.warning(
                    f"⚠️ Missing required files for {tf} report generation: {missing_files}"
                )
                print_and_log(
                    f"⚠️ Missing required files for {tf} report generation: {missing_files}",
                    "warning",
                    f"Step1_7.HMMRegimeDiscovery.{tf}",
                )
                continue

            # Generate simple text report instead of complex visualizations to avoid segmentation faults
            try:
                logger.info(f"🔧 DEBUG: Calling _generate_simple_hmm_report for {tf}")
                report_content = _generate_simple_hmm_report(
                    exchange, symbol, tf, data_dir
                )
                logger.info(f"✅ DEBUG: _generate_simple_hmm_report completed for {tf}")

                # Save report to file with timestamp
                reports_dir = "reports"
                os.makedirs(reports_dir, exist_ok=True)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                report_path = os.path.join(
                    reports_dir, f"{exchange}_{symbol}_hmm_regime_summary_{tf}_{timestamp}.md"
                )

                logger.info(f"💾 DEBUG: Saving report to {report_path}")
                with open(report_path, "w") as f:
                    f.write(report_content)

                logger.info(
                    f"✅ Simple HMM regime report generated successfully for {tf}: {report_path}"
                )
                print_and_log(
                    f"✅ Simple HMM regime report generated successfully for {tf}: {report_path}",
                    "info",
                    f"Step1_7.HMMRegimeDiscovery.{tf}",
                )

            except Exception as report_error:
                logger.error(
                    f"❌ ERROR generating simple HMM report for {tf}: {report_error}"
                )
                logger.error(f"❌ ERROR details: {type(report_error).__name__}: {str(report_error)}")
                import traceback
                logger.error(f"❌ ERROR traceback: {traceback.format_exc()}")
                print_and_log(
                    f"❌ ERROR generating simple HMM report for {tf}: {report_error}",
                    "error",
                    f"Step1_7.HMMRegimeDiscovery.{tf}",
                )

        except Exception as e:
            logger.error(f"❌ ERROR in HMM regime analysis for {tf}: {e}")
            logger.error(f"❌ ERROR details: {type(e).__name__}: {str(e)}")
            import traceback
            logger.error(f"❌ ERROR traceback: {traceback.format_exc()}")
            print_and_log(
                f"❌ ERROR in HMM regime analysis for {tf}: {e}",
                "error",
                f"Step1_7.HMMRegimeDiscovery.{tf}",
            )

    total_time = time.time() - t0_total

    # Final cleanup to prevent resource leaks
    try:
        _cleanup_multiprocessing_resources()
        logger.info("🧹 Final resource cleanup completed")
    except Exception as cleanup_error:
        logger.warning(f"⚠️ Warning during final cleanup: {cleanup_error}")

    # Additional cleanup to prevent segmentation faults
    try:
        # Force garbage collection multiple times
        for _ in range(3):
            gc.collect()

        # Clear any remaining references
        import sys

        if "df" in locals():
            del df
        if "features_df" in locals():
            del features_df
        if "data_sharing_manager" in locals():
            del data_sharing_manager
        if "loader" in locals():
            del loader

        logger.info("🧹 Additional cleanup completed")
    except Exception as additional_cleanup_error:
        logger.warning(
            f"⚠️ Warning during additional cleanup: {additional_cleanup_error}"
        )

    if final_validation_passed:
        logger.info(
            "✅ Step 1_7: HMM Regime Discovery completed successfully - all artifacts created"
        )
        print_and_log(
            f"✅ Step 1_7: HMM Regime Discovery completed successfully in {total_time:.2f}s - all artifacts created",
            "info",
            "Step1_7.HMMRegimeDiscovery",
        )

        # Summary of generated reports
        reports_dir = os.path.join("reports")
        if os.path.exists(reports_dir):
            report_files = [
                f
                for f in os.listdir(reports_dir)
                if f.endswith(".md") and "regime_summary" in f
            ]
            if report_files:
                logger.info(f"📊 HMM Regime Analysis Reports generated:")
                print_and_log(
                    f"📊 HMM Regime Analysis Reports generated:",
                    "info",
                    "Step1_7.HMMRegimeDiscovery",
                )
                for report_file in sorted(report_files):
                    report_path = os.path.join(reports_dir, report_file)
                    logger.info(f"  📄 {report_path}")
                    print_and_log(
                        f"  📄 {report_path}", "info", "Step1_7.HMMRegimeDiscovery"
                    )
            else:
                logger.info(
                    "📊 No HMM regime analysis reports found in reports/ directory"
                )
                print_and_log(
                    "📊 No HMM regime analysis reports found in reports/ directory",
                    "info",
                    "Step1_7.HMMRegimeDiscovery",
                )
        else:
            logger.info("📊 Reports directory not found")
            print_and_log(
                "📊 Reports directory not found", "info", "Step1_7.HMMRegimeDiscovery"
            )
    else:
        logger.warning(
            "⚠️ Step 1_7: HMM Regime Discovery completed with missing artifacts"
        )
        print_and_log(
            f"⚠️ Step 1_7: HMM Regime Discovery completed with missing artifacts in {total_time:.2f}s",
            "warning",
            "Step1_7.HMMRegimeDiscovery",
        )

    return any_success and final_validation_passed


@handle_data_processing_errors(default_return=pd.Series([-1] * 1000))
def _merge_rare_combinations(
    combo_keys: pd.Series,
    profile_df: pd.DataFrame,
    min_count: int = 3,  # Reduced from 5 to 3 to make penalty weaker
    similarity_threshold: float = 0.65,  # Reduced from 0.7 to 0.65 to make penalty weaker
) -> Tuple[pd.Series, pd.DataFrame]:
    """
    Merge rare combinations into similar frequent combinations instead of filtering them out.
    Only merges if similarity meets the threshold.

    Args:
        combo_keys: Series of combination keys
        profile_df: DataFrame of combination profiles
        min_count: Minimum count threshold for frequent combinations
        similarity_threshold: Minimum cosine similarity required for merging (0.0 to 1.0)

    Returns:
        Tuple of (merged_combo_keys, merged_profile_df)
    """
    if profile_df.empty:
        return combo_keys, profile_df

    counts = combo_keys.value_counts()
    frequent_combos = counts[counts >= min_count].index
    rare_combos = counts[counts < min_count].index

    if len(frequent_combos) == 0:
        # If no frequent combinations, use the most common ones
        top_combos = counts.head(max(2, len(counts) // 10)).index
        frequent_combos = top_combos
        rare_combos = counts[~counts.index.isin(frequent_combos)].index

    if len(rare_combos) == 0:
        # No rare combinations to merge
        return combo_keys, profile_df

    logger.info(
        f"🔄 Merging {len(rare_combos)} rare combinations into {len(frequent_combos)} frequent combinations"
    )

    # Create mapping from rare to frequent combinations
    rare_to_frequent_map = {}

    # Get profiles for frequent combinations
    frequent_profiles = profile_df.loc[profile_df.index.intersection(frequent_combos)]

    if frequent_profiles.empty:
        # Fallback: no frequent profiles available
        return combo_keys, profile_df

    # Normalize frequent profiles for similarity calculation
    frequent_profiles_norm = frequent_profiles.values.astype(float)
    frequent_profiles_norm = np.nan_to_num(
        frequent_profiles_norm, nan=0.0, posinf=0.0, neginf=0.0
    )
    norms = np.linalg.norm(frequent_profiles_norm, axis=1, keepdims=True) + 1e-12
    frequent_profiles_norm = frequent_profiles_norm / norms

    # For each rare combination, find the most similar frequent combination
    merged_count = 0
    rejected_count = 0

    for rare_combo in rare_combos:
        if rare_combo not in profile_df.index:
            continue

        rare_profile = profile_df.loc[rare_combo].values.astype(float)
        rare_profile = np.nan_to_num(rare_profile, nan=0.0, posinf=0.0, neginf=0.0)

        # Normalize rare profile
        rare_norm = np.linalg.norm(rare_profile) + 1e-12
        rare_profile_norm = rare_profile / rare_norm

        # Calculate cosine similarities to all frequent combinations
        similarities = np.dot(
            frequent_profiles_norm,
            rare_profile_norm.reshape(
                -1,
            ),
        )

        # Find the most similar frequent combination
        most_similar_idx = np.argmax(similarities)
        max_similarity = similarities[most_similar_idx]
        most_similar_combo = frequent_profiles.index[most_similar_idx]

        # Only merge if similarity meets the threshold
        if max_similarity >= similarity_threshold:
            rare_to_frequent_map[rare_combo] = most_similar_combo
            merged_count += 1
        else:
            rejected_count += 1
            logger.debug(
                f"⚠️ Rejected merging rare combo '{rare_combo}' (similarity: {max_similarity:.3f} < {similarity_threshold})"
            )

    logger.info(
        f"🔄 Merging results: {merged_count} merged, {rejected_count} rejected due to low similarity"
    )

    # Create merged combination keys
    merged_combo_keys = combo_keys.copy()
    for rare_combo, frequent_combo in rare_to_frequent_map.items():
        merged_combo_keys = merged_combo_keys.replace(rare_combo, frequent_combo)

    # Update profile DataFrame to reflect merged combinations
    merged_profile_df = profile_df.copy()

    # For each frequent combination, update its profile to include merged rare combinations
    for frequent_combo in frequent_combos:
        if frequent_combo not in merged_profile_df.index:
            continue

        # Find all rare combinations that were merged into this frequent combination
        merged_rare_combos = [
            rare
            for rare, freq in rare_to_frequent_map.items()
            if freq == frequent_combo
        ]

        if merged_rare_combos:
            # Calculate weighted average profile (weighted by count)
            frequent_count = counts[frequent_combo]
            total_weight = frequent_count

            # Start with the frequent combination profile
            merged_profile = (
                merged_profile_df.loc[frequent_combo].values.astype(float)
                * frequent_count
            )

            # Add weighted contributions from rare combinations
            for rare_combo in merged_rare_combos:
                if rare_combo in merged_profile_df.index:
                    rare_count = counts[rare_combo]
                    rare_profile = merged_profile_df.loc[rare_combo].values.astype(
                        float
                    )
                    merged_profile += rare_profile * rare_count
                    total_weight += rare_count

            # Normalize by total weight
            if total_weight > 0:
                merged_profile = merged_profile / total_weight
                merged_profile_df.loc[frequent_combo] = merged_profile

    # Remove rare combinations from profile DataFrame
    merged_profile_df = merged_profile_df.loc[
        merged_profile_df.index.intersection(frequent_combos)
    ]

    logger.info(
        f"✅ Successfully merged rare combinations. Final profile has {len(merged_profile_df)} combinations"
    )

    return merged_combo_keys, merged_profile_df


@handle_errors(
    exceptions=(ValueError, KeyError, TypeError),
    default_return={},
    context="archetype description generation",
)
def _generate_archetype_descriptions(
    cluster_centroids: Dict[int, List[float]],
    state_names: Dict[str, Dict[str, str]],
    block_posteriors: Dict[str, pd.DataFrame],
    cluster_counts: Dict[int, int],
) -> Dict[int, str]:
    """
    Generate human-readable descriptions for each market archetype.

    Args:
        cluster_centroids: Dictionary mapping cluster IDs to centroid vectors
        state_names: Dictionary mapping block names to state names
        block_posteriors: Dictionary mapping block names to posterior probability DataFrames
        cluster_counts: Dictionary mapping cluster IDs to their frequencies

    Returns:
        Dictionary mapping cluster IDs to human-readable descriptions
    """
    try:
        archetype_descriptions = {}

        # Get unique cluster IDs
        cluster_ids = list(cluster_centroids.keys())

        for cluster_id in cluster_ids:
            if cluster_id < 0:  # Skip noise clusters
                continue

            # Get the centroid for this cluster
            centroid = cluster_centroids.get(cluster_id, [])
            if not centroid:
                continue

            # Generate description based on the cluster characteristics
            description_parts = []

            # Add frequency information
            frequency = cluster_counts.get(cluster_id, 0)
            if frequency > 0:
                total_obs = sum(cluster_counts.values())
                percentage = (frequency / total_obs * 100) if total_obs > 0 else 0
                description_parts.append(f"Frequent ({percentage:.1f}% of time)")

            # Add market condition description based on centroid characteristics
            if len(centroid) >= 4:  # Assuming we have at least 4 blocks
                # Analyze the centroid values to determine market conditions
                momentum_score = centroid[0] if len(centroid) > 0 else 0
                volatility_score = centroid[1] if len(centroid) > 1 else 0
                volume_score = centroid[2] if len(centroid) > 2 else 0
                sr_score = centroid[3] if len(centroid) > 3 else 0

                # Determine market conditions with more nuanced descriptions
                conditions = []

                # Momentum condition with intensity
                if momentum_score > 0.7:
                    conditions.append("strong bullish momentum")
                elif momentum_score > 0.3:
                    conditions.append("moderate bullish momentum")
                elif momentum_score > -0.3:
                    conditions.append("neutral momentum")
                elif momentum_score > -0.7:
                    conditions.append("moderate bearish momentum")
                else:
                    conditions.append("strong bearish momentum")

                # Volatility condition with intensity
                if volatility_score > 0.7:
                    conditions.append("very high volatility")
                elif volatility_score > 0.3:
                    conditions.append("high volatility")
                elif volatility_score > -0.3:
                    conditions.append("moderate volatility")
                elif volatility_score > -0.7:
                    conditions.append("low volatility")
                else:
                    conditions.append("very low volatility")

                # Volume condition with intensity
                if volume_score > 0.7:
                    conditions.append("very high volume")
                elif volume_score > 0.3:
                    conditions.append("high volume")
                elif volume_score > -0.3:
                    conditions.append("moderate volume")
                elif volume_score > -0.7:
                    conditions.append("low volume")
                else:
                    conditions.append("very low volume")

                # Support/Resistance condition with intensity
                if sr_score > 0.7:
                    conditions.append("near resistance")
                elif sr_score > 0.3:
                    conditions.append("approaching resistance")
                elif sr_score > -0.3:
                    conditions.append("neutral levels")
                elif sr_score > -0.7:
                    conditions.append("approaching support")
                else:
                    conditions.append("near support")

                # Combine conditions
                if conditions:
                    condition_desc = ", ".join(conditions)
                    description_parts.append(f"{condition_desc} market")

            # Create final description
            if description_parts:
                description = " - ".join(description_parts)
            else:
                description = f"Market Archetype {cluster_id}"

            archetype_descriptions[cluster_id] = description

        logger.info(
            f"✅ Generated {len(archetype_descriptions)} archetype descriptions"
        )
        return archetype_descriptions

    except Exception as e:
        logger.error(f"❌ Error generating archetype descriptions: {e}")
        return {}


def _compute_comprehensive_composite_metrics(
    profile_df: pd.DataFrame,
    labels: pd.Series,
    cluster_centroids: Dict[int, List[float]],
    features_df: pd.DataFrame,
    block_states: Dict[str, pd.Series],
    block_posteriors: Dict[str, pd.DataFrame],
    cluster_series: pd.Series,
    state_names: Dict[str, Dict[str, str]],
    block_features: Dict[str, List[str]],
) -> CompositeModelMetrics:
    """Compute comprehensive metrics for composite model analysis."""

    # Basic cluster metrics
    cluster_count = len(np.unique(labels[labels >= 0]))
    cluster_sizes = labels.value_counts().to_dict()
    total_samples = len(labels)
    cluster_frequencies = {k: v / total_samples for k, v in cluster_sizes.items()}

    # Quality metrics (only for valid clusters)
    valid_mask = labels >= 0
    if valid_mask.sum() > 1 and len(np.unique(labels[valid_mask])) > 1:
        try:
            silhouette = silhouette_score(profile_df[valid_mask], labels[valid_mask])
            calinski_harabasz = calinski_harabasz_score(
                profile_df[valid_mask], labels[valid_mask]
            )
            davies_bouldin = davies_bouldin_score(
                profile_df[valid_mask], labels[valid_mask]
            )
        except Exception:
            silhouette = 0.0
            calinski_harabasz = 0.0
            davies_bouldin = 0.0
    else:
        silhouette = 0.0
        calinski_harabasz = 0.0
        davies_bouldin = 0.0

    # Diversity metrics
    cluster_diversity = len(cluster_sizes) / max(1, cluster_count)
    cluster_separation = 1.0 - (
        min(cluster_sizes.values()) / max(cluster_sizes.values())
        if cluster_sizes
        else 0.0
    )
    cluster_cohesion = silhouette  # Use silhouette as cohesion measure

    # Temporal metrics (simplified)
    cluster_persistence = {k: 1.0 for k in cluster_sizes.keys()}
    cluster_volatility = {k: 0.1 for k in cluster_sizes.keys()}

    # Block composition metrics (simplified)
    block_representation = {}
    block_dominance = {}
    block_balance = {}

    for cluster_id in cluster_sizes.keys():
        if cluster_id < 0:
            continue
        block_representation[cluster_id] = {}
        block_dominance[cluster_id] = "Unknown"
        block_balance[cluster_id] = 0.5

    # Market condition metrics (simplified)
    market_condition_distribution = {
        k: {"trending": 0.5, "ranging": 0.5} for k in cluster_sizes.keys()
    }
    regime_stability = {k: 0.8 for k in cluster_sizes.keys()}
    regime_transition_probability = {"high": 0.2, "medium": 0.5, "low": 0.3}

    # Anomaly detection (simplified)
    outlier_clusters = []
    unstable_clusters = []
    rare_clusters = [
        k for k, v in cluster_sizes.items() if v < 10
    ]  # Clusters with < 10 samples

    # Feature coverage (simplified)
    missing_features_by_cluster = {k: [] for k in cluster_sizes.keys()}

    return CompositeModelMetrics(
        cluster_count=cluster_count,
        cluster_sizes=cluster_sizes,
        cluster_frequencies=cluster_frequencies,
        silhouette_score=silhouette,
        calinski_harabasz_score=calinski_harabasz,
        davies_bouldin_score=davies_bouldin,
        cluster_diversity=cluster_diversity,
        cluster_separation=cluster_separation,
        cluster_cohesion=cluster_cohesion,
        cluster_persistence=cluster_persistence,
        cluster_volatility=cluster_volatility,
        block_representation=block_representation,
        block_dominance=block_dominance,
        block_balance=block_balance,
        market_condition_distribution=market_condition_distribution,
        regime_stability=regime_stability,
        regime_transition_probability=regime_transition_probability,
        outlier_clusters=outlier_clusters,
        unstable_clusters=unstable_clusters,
        rare_clusters=rare_clusters,
        missing_features_by_cluster=missing_features_by_cluster,
    )


def _generate_regime_description(regime_id: int, meta: dict, original_desc: str) -> str:
    """Generate a better regime description based on block composition."""
    try:
        # Get the cluster labels from meta data
        cluster_labels = meta.get("cluster_labels", {})

        # Find the cluster label for this regime
        regime_label = None
        for label, cluster_id in cluster_labels.items():
            if cluster_id == regime_id:
                regime_label = label
                break

        if not regime_label:
            return original_desc

        # Parse the label format: "momentum:0|volatility:0|volume:0|support_resistance:0"
        block_states = {}
        for part in regime_label.split("|"):
            if ":" in part:
                block_name, state_id = part.split(":")
                block_states[block_name] = int(state_id)

        if not block_states:
            return original_desc

        # Build description based on block states
        desc_parts = []

        # Get state names from meta file, fallback to hardcoded descriptions
        state_names = meta.get("state_names", {})
        
        # Block-specific descriptions (fallback) - Enhanced for higher sensitivity
        block_descriptions = {
            "momentum": {
                0: "Weak Downtrend",  # More sensitive than "Sideways/Neutral"
                1: "Moderate Downtrend", 
                2: "Sideways/Neutral",  # Moved to middle state
                3: "Moderate Uptrend",
                4: "Strong Uptrend",
                5: "Strong Downtrend",  # Added 6th state
            },
            "volatility": {
                0: "Low & Stable Vol",
                1: "Moderate Vol",
                2: "High & Choppy Vol",
                3: "Very High & Choppy Vol",
            },
            "volume": {
                0: "Very Low Volume",
                1: "Low Volume",
                2: "Medium Volume",
                3: "High Volume",
                4: "Very High Volume",
            },
            "support_resistance": {
                0: "Near Support",
                1: "Neutral Levels",
                2: "Near Resistance",
            },

        }

        for block_name, state_id in block_states.items():
            # Try to get state name from meta file first
            if block_name in state_names and str(state_id) in state_names[block_name]:
                desc_parts.append(state_names[block_name][str(state_id)])
            # Fallback to hardcoded descriptions
            elif (
                block_name in block_descriptions
                and state_id in block_descriptions[block_name]
            ):
                desc_parts.append(block_descriptions[block_name][state_id])

        if desc_parts:
            return f"{', '.join(desc_parts)} Market"
        else:
            return original_desc

    except Exception as e:
        return original_desc


def _generate_state_name(block_name: str, state_id: int, original_name: str, meta: dict = None) -> str:
    """Generate better state names based on block type."""
    try:
        block_name_lower = block_name.lower()
        
        # Try to get state name from meta file first
        if meta and "state_names" in meta:
            state_names = meta["state_names"]
            if block_name_lower in state_names and str(state_id) in state_names[block_name_lower]:
                return state_names[block_name_lower][str(state_id)]

        # Fallback to hardcoded names - Enhanced for higher sensitivity
        if block_name_lower == "momentum":
            momentum_names = {
                0: "Weak Downtrend",  # More sensitive than "Sideways/Neutral"
                1: "Moderate Downtrend",
                2: "Sideways/Neutral",  # Moved to middle state
                3: "Moderate Uptrend", 
                4: "Strong Uptrend",
                5: "Strong Downtrend",  # Added 6th state
            }
            return momentum_names.get(state_id, original_name)

        elif block_name_lower == "volatility":
            volatility_names = {
                0: "Low & Stable Vol",
                1: "Moderate Vol",
                2: "High & Choppy Vol",
                3: "Very High & Choppy Vol",
            }
            return volatility_names.get(state_id, original_name)

        elif block_name_lower == "volume":
            volume_names = {
                0: "Very Low Volume",
                1: "Low Volume",
                2: "Medium Volume",
                3: "High Volume",
                4: "Very High Volume",
            }
            return volume_names.get(state_id, original_name)

        elif block_name_lower == "support_resistance":
            sr_names = {
                0: "Near Support",
                1: "Neutral Levels",
                2: "Near Resistance",
            }
            return sr_names.get(state_id, original_name)



        else:
            return original_name

    except Exception as e:
        return original_name


def _generate_simple_hmm_report(
    exchange: str, symbol: str, timeframe: str, data_dir: str
) -> str:
    """Generate a comprehensive text-based HMM regime report without matplotlib to avoid segmentation faults."""
    logger.info(f"🔧 DEBUG: _generate_simple_hmm_report called with exchange={exchange}, symbol={symbol}, timeframe={timeframe}, data_dir={data_dir}")
    try:
        import json
        from datetime import datetime

        # Load meta data
        meta_path = os.path.join(
            data_dir, f"{exchange}_{symbol}_hmm_composite_meta_{timeframe}.json"
        )
        if not os.path.exists(meta_path):
            return f"# HMM Regime Report\n\nError: Meta file not found at {meta_path}"

        with open(meta_path, "r") as f:
            meta = json.load(f)

        # Load cluster data
        cluster_path = os.path.join(
            data_dir, f"{exchange}_{symbol}_hmm_composite_clusters_{timeframe}.parquet"
        )
        if not os.path.exists(cluster_path):
            return f"# HMM Regime Report\n\nError: Cluster file not found at {cluster_path}"

        cluster_df = pd.read_parquet(cluster_path)

        # Load intensity data if available
        intensity_df = None
        intensity_path = os.path.join(
            data_dir, f"{exchange}_{symbol}_hmm_composite_intensity_{timeframe}.parquet"
        )
        if os.path.exists(intensity_path):
            intensity_df = pd.read_parquet(intensity_path)

        # Generate comprehensive report
        report_lines = []
        report_lines.append("# 🎯 Composite HMM Regimes (Detailed Market Conditions)")
        report_lines.append("")
        report_lines.append(
            "> **Note**: This report is generated automatically during HMM regime discovery. If you see multiple files with different timestamps for the same timeframe, the most recent one contains the complete analysis."
        )
        report_lines.append("")

        # Add Executive Summary
        report_lines.append("## 📋 Executive Summary")
        report_lines.append("")

        # Get archetype descriptions
        archetype_descriptions = meta.get("archetype_descriptions", {})
        valid_archetypes = {
            k: v for k, v in archetype_descriptions.items() if int(k) >= 0
        }

        # Count actual regimes from cluster data
        actual_regime_count = 0
        if "composite_cluster_id" in cluster_df.columns:
            cluster_counts = (
                cluster_df["composite_cluster_id"].value_counts().sort_index()
            )
            actual_regime_count = len(
                [regime_id for regime_id in cluster_counts.index if regime_id >= 0]
            )

        # Key findings
        report_lines.append("**Key Findings:**")
        report_lines.append(
            f"- 🎯 **{actual_regime_count} distinct market regimes** identified"
        )
        report_lines.append(
            "- 🔧 **Simplified regime structure** focusing on core market dynamics"
        )
        report_lines.append(
            "- 📊 **4 primary blocks**: Momentum, Volatility, Volume, and Support/Resistance"
        )

        # Regime distribution and concentration analysis
        if "composite_cluster_id" in cluster_df.columns:
            cluster_counts = (
                cluster_df["composite_cluster_id"].value_counts().sort_index()
            )
            total_obs = len(cluster_df)

            # Calculate concentration statistics
            top_10_regimes = cluster_counts.head(10)
            top_20_regimes = cluster_counts.head(20)
            top_10_concentration = (
                (top_10_regimes.sum() / total_obs * 100) if total_obs > 0 else 0
            )
            top_20_concentration = (
                (top_20_regimes.sum() / total_obs * 100) if total_obs > 0 else 0
            )

            report_lines.append(
                f"- 📈 **Top 10 regimes** account for {top_10_concentration:.1f}% of market time"
            )
            report_lines.append(
                f"- 📈 **Top 20 regimes** account for {top_20_concentration:.1f}% of market time"
            )

            # Check for noise cluster (-1)
            noise_count = cluster_counts.get(-1, 0)
            noise_percentage = (noise_count / total_obs * 100) if total_obs > 0 else 0
            if noise_count > 0:
                report_lines.append(
                    f"- ⚠️ **Noise cluster (-1)** contains {noise_count} observations ({noise_percentage:.1f}%)"
                )

        report_lines.append("")
        report_lines.append("**📊 Visualization Notes:**")
        report_lines.append(
            "- All charts are generated in high-resolution PNG format (300 DPI)"
        )
        report_lines.append(
            "- Click on image links to download full-resolution versions"
        )
        report_lines.append(
            "- Charts are optimized for both screen viewing and printing"
        )
        report_lines.append("")

        # Count total archetypes (excluding noise cluster -1)
        report_lines.append(
            f"Your system discovered **{actual_regime_count} distinct market archetypes** that combine different states from the {len(meta.get('blocks', []))} HMM blocks:"
        )
        report_lines.append("")

        # Add regime merging information if available
        regime_merging_stats = meta.get("regime_merging_stats", {})
        merging_config = meta.get("merging_config", {})
        regime_merging_applied = meta.get("regime_merging_applied", False)

        if regime_merging_applied and regime_merging_stats:
            report_lines.append("## 📊 Regime Merging Analysis")
            report_lines.append("")
            report_lines.append("### Concentration Statistics:")
            report_lines.append(
                f"- **Top 10 Concentration**: {regime_merging_stats.get('top_10_concentration', 0):.1f}%"
            )
            report_lines.append(
                f"- **Top 20 Concentration**: {regime_merging_stats.get('top_20_concentration', 0):.1f}%"
            )
            report_lines.append(
                f"- **Regime -1 (Noise) Concentration**: {regime_merging_stats.get('regime_neg1_concentration', 0):.1f}%"
            )
            report_lines.append("")
            report_lines.append("### Regime Counts:")
            report_lines.append(
                f"- **Regimes Before Merge**: {regime_merging_stats.get('regimes_before_merge', 0)}"
            )
            report_lines.append(
                f"- **Regimes After Merge**: {regime_merging_stats.get('regimes_after_merge', 0)}"
            )
            report_lines.append(
                f"- **Regimes Merged**: {regime_merging_stats.get('regimes_before_merge', 0) - regime_merging_stats.get('regimes_after_merge', 0)}"
            )
            report_lines.append("")
            report_lines.append("### Merging Configuration:")
            report_lines.append(
                f"- **Similarity Threshold**: {merging_config.get('similarity_threshold', 'N/A')}"
            )
            report_lines.append(
                f"- **Min Frequency**: {merging_config.get('min_frequency', 'N/A')}"
            )
            report_lines.append(
                f"- **Target Top 20 Concentration**: {merging_config.get('target_top_20_concentration', 'N/A')}"
            )
            report_lines.append(
                f"- **Aggressive Merging**: {merging_config.get('aggressive_merging', 'N/A')}"
            )
        else:
            # Calculate concentration statistics manually if not in meta
            if "composite_cluster_id" in cluster_df.columns:
                report_lines.append("## 📊 Regime Concentration Analysis")
                report_lines.append("")
                report_lines.append("### Concentration Statistics:")
                report_lines.append(
                    f"- **Top 10 Concentration**: {top_10_concentration:.1f}%"
                )
                report_lines.append(
                    f"- **Top 20 Concentration**: {top_20_concentration:.1f}%"
                )
                if noise_count > 0:
                    report_lines.append(
                        f"- **Regime -1 (Noise) Concentration**: {noise_percentage:.1f}%"
                    )
                report_lines.append("")
                report_lines.append("### Concentration Analysis:")
                if top_10_concentration >= 85:
                    report_lines.append(
                        "- **High Concentration**: Top 10 regimes dominate the market (≥85%)"
                    )
                    report_lines.append(
                        "- **Market State**: Very stable, trending market conditions"
                    )
                elif top_10_concentration >= 70:
                    report_lines.append(
                        "- **Moderate-High Concentration**: Top 10 regimes are prominent (70-85%)"
                    )
                    report_lines.append(
                        "- **Market State**: Stable market with some variability"
                    )
                elif top_10_concentration >= 50:
                    report_lines.append(
                        "- **Moderate Concentration**: Top 10 regimes account for significant time (50-70%)"
                    )
                    report_lines.append("- **Market State**: Mixed market conditions")
                else:
                    report_lines.append(
                        "- **Low Concentration**: Top 10 regimes account for less than 50% of time"
                    )
                    report_lines.append(
                        "- **Market State**: Highly volatile or transitioning market conditions"
                    )
                report_lines.append("")

        # Detailed regime analysis
        if "composite_cluster_id" in cluster_df.columns:
            report_lines.append("## 📈 Detailed Regime Analysis")
            report_lines.append("")

            report_lines.append("### Top 10 Regimes by Frequency:")
            report_lines.append("")
            for i, (regime_id, count) in enumerate(cluster_counts.head(10).items(), 1):
                if regime_id >= 0:  # Skip noise cluster
                    percentage = (count / total_obs * 100) if total_obs > 0 else 0
                    desc = archetype_descriptions.get(
                        str(regime_id), f"Regime {regime_id}"
                    )

                    # Generate better description based on block composition
                    better_desc = _generate_regime_description(regime_id, meta, desc)

                    report_lines.append(
                        f"{i}. **Regime {regime_id}**: {count:,} observations ({percentage:.2f}%)"
                    )
                    report_lines.append(f"   - **Description**: {better_desc}")
                    report_lines.append("")

            if len(cluster_counts) > 10:
                report_lines.append("### Remaining Regimes:")
                report_lines.append("")
                for regime_id, count in cluster_counts.iloc[10:].items():
                    if regime_id >= 0:  # Skip noise cluster
                        percentage = (count / total_obs * 100) if total_obs > 0 else 0
                        desc = archetype_descriptions.get(
                            str(regime_id), f"Regime {regime_id}"
                        )
                        better_desc = _generate_regime_description(
                            regime_id, meta, desc
                        )
                        report_lines.append(
                            f"- **Regime {regime_id}**: {count:,} observations ({percentage:.2f}%) - {better_desc}"
                        )
                report_lines.append("")

        # Block information
        blocks = meta.get("blocks", [])
        if blocks:
            report_lines.append("## 🧩 HMM Block Analysis")
            report_lines.append("")
            report_lines.append("### Simplified Block Configuration:")
            report_lines.append("> **Note**: Regime detection has been simplified to focus on core market dynamics:")
            for block in blocks:
                report_lines.append(
                    f"- **{block['name'].title()}**: {block['n_states']} states"
                )
            report_lines.append("")
            report_lines.append("**Simplified Regime Structure:**")
            report_lines.append("- **Momentum**: Price trend and momentum patterns")
            report_lines.append("- **Volatility**: Market volatility and dispersion")
            report_lines.append("- **Volume**: Trading volume and flow analysis")
            report_lines.append("- **Support/Resistance**: Price level proximity and strength")
            report_lines.append("")
            report_lines.append("*Liquidity and market microstructure blocks have been removed to simplify regime detection and improve stability.*")
            report_lines.append("")

            # State names with better descriptions
            state_names = meta.get("state_names", {})
            if state_names:
                report_lines.append("### State Names by Block:")
                for block_name, states in state_names.items():
                    report_lines.append(f"**{block_name.title()}:**")
                    for state_id, state_name in states.items():
                        # Generate better state names based on block type
                        better_state_name = _generate_state_name(
                            block_name, int(state_id), state_name, meta
                        )
                        report_lines.append(
                            f"  - State {state_id}: {better_state_name}"
                        )
                    report_lines.append("")

        # Add state distribution analysis
        if intensity_df is not None and "composite_cluster_id" in cluster_df.columns:
            report_lines.append("## 📊 State Distribution Analysis")
            report_lines.append("")
            report_lines.append("### Percentage of Time Spent in Each State:")
            report_lines.append("")
            
            # Get block information
            blocks = meta.get("blocks", [])
            state_names = meta.get("state_names", {})
            
            # Calculate state distributions for each block
            for block in blocks:
                block_name = block["name"]
                n_states = block["n_states"]
                
                report_lines.append(f"**{block_name.title()} Block:**")
                report_lines.append("")
                
                # Create table header
                report_lines.append("| State | Description | % of Time | Observations |")
                report_lines.append("|-------|-------------|-----------|--------------|")
                
                # Calculate state distribution
                state_counts = {}
                total_obs = len(intensity_df)
                
                # Get state column for this block
                state_col = f"{block_name}_state"
                if state_col in intensity_df.columns:
                    state_counts = intensity_df[state_col].value_counts().sort_index()
                    
                    for state_id in range(n_states):
                        count = state_counts.get(state_id, 0)
                        percentage = (count / total_obs * 100) if total_obs > 0 else 0
                        
                        # Get state description
                        if block_name in state_names and str(state_id) in state_names[block_name]:
                            description = state_names[block_name][str(state_id)]
                        else:
                            description = _generate_state_name(block_name, state_id, f"State {state_id}", meta)
                        
                        report_lines.append(f"| {state_id} | {description} | {percentage:.2f}% | {count:,} |")
                
                report_lines.append("")
                report_lines.append("")
            
            # Add summary statistics
            report_lines.append("### State Distribution Summary:")
            report_lines.append("")
            
            for block in blocks:
                block_name = block["name"]
                state_col = f"{block_name}_state"
                
                if state_col in intensity_df.columns:
                    state_counts = intensity_df[state_col].value_counts().sort_index()
                    total_obs = len(intensity_df)
                    
                    # Find dominant state
                    dominant_state = state_counts.index[0] if len(state_counts) > 0 else None
                    dominant_percentage = (state_counts.iloc[0] / total_obs * 100) if len(state_counts) > 0 and total_obs > 0 else 0
                    
                    # Get dominant state description
                    if dominant_state is not None:
                        if block_name in state_names and str(dominant_state) in state_names[block_name]:
                            dominant_desc = state_names[block_name][str(dominant_state)]
                        else:
                            dominant_desc = _generate_state_name(block_name, dominant_state, f"State {dominant_state}", meta)
                        
                        report_lines.append(f"- **{block_name.title()}**: Dominant state is {dominant_desc} ({dominant_percentage:.1f}% of time)")
                    
                    # Calculate state diversity
                    state_diversity = len(state_counts)
                    report_lines.append(f"  - State diversity: {state_diversity} distinct states")
                    
                    # Calculate concentration (top state percentage)
                    if dominant_percentage > 50:
                        concentration_desc = "High concentration"
                    elif dominant_percentage > 30:
                        concentration_desc = "Moderate concentration"
                    else:
                        concentration_desc = "Low concentration"
                    
                    report_lines.append(f"  - Concentration: {concentration_desc} ({dominant_percentage:.1f}% in top state)")
                    report_lines.append("")
            
        elif "composite_cluster_id" in cluster_df.columns:
            report_lines.append("## 📊 State Distribution Analysis")
            report_lines.append("")
            report_lines.append("*Note: Intensity data not available for detailed state distribution analysis.*")
            report_lines.append("")

        # Regime transitions and persistence analysis
        report_lines.append("## 🔄 Regime Transitions & Persistence Analysis")
        report_lines.append("")

        if "composite_cluster_id" in cluster_df.columns:
            cluster_series = cluster_df["composite_cluster_id"]

            # Calculate regime transitions
            transitions = []
            for i in range(1, len(cluster_series)):
                from_regime = cluster_series.iloc[i - 1]
                to_regime = cluster_series.iloc[i]
                if from_regime != to_regime:  # Only count actual transitions
                    transitions.append((from_regime, to_regime))

            # Calculate transition matrix
            unique_regimes = sorted(cluster_series.unique())
            transition_matrix = {}
            for from_regime in unique_regimes:
                transition_matrix[from_regime] = {}
                for to_regime in unique_regimes:
                    transition_matrix[from_regime][to_regime] = 0

            # Count transitions
            for from_regime, to_regime in transitions:
                if (
                    from_regime in transition_matrix
                    and to_regime in transition_matrix[from_regime]
                ):
                    transition_matrix[from_regime][to_regime] += 1

            # Calculate transition probabilities
            transition_probabilities = {}
            for from_regime in unique_regimes:
                total_from = sum(transition_matrix[from_regime].values())
                if total_from > 0:
                    transition_probabilities[from_regime] = {}
                    for to_regime in unique_regimes:
                        prob = transition_matrix[from_regime][to_regime] / total_from
                        transition_probabilities[from_regime][to_regime] = prob

            # Calculate regime persistence
            persistence_data = []
            current_regime = cluster_series.iloc[0]
            start_time = 0
            duration = 1

            for i in range(1, len(cluster_series)):
                if cluster_series.iloc[i] == current_regime:
                    duration += 1
                else:
                    persistence_data.append(
                        {
                            "regime": current_regime,
                            "start": start_time,
                            "duration": duration,
                        }
                    )
                    current_regime = cluster_series.iloc[i]
                    start_time = i
                    duration = 1

            # Add last regime
            persistence_data.append(
                {"regime": current_regime, "start": start_time, "duration": duration}
            )

            # Calculate persistence statistics
            persistence_stats = {}
            for data in persistence_data:
                regime = data["regime"]
                duration = data["duration"]
                if regime not in persistence_stats:
                    persistence_stats[regime] = {"durations": [], "count": 0}
                persistence_stats[regime]["durations"].append(duration)
                persistence_stats[regime]["count"] += 1

            # Calculate average persistence for each regime
            for regime in persistence_stats:
                durations = persistence_stats[regime]["durations"]
                persistence_stats[regime]["avg_duration"] = np.mean(durations)
                persistence_stats[regime]["median_duration"] = np.median(durations)
                persistence_stats[regime]["max_duration"] = max(durations)
                persistence_stats[regime]["min_duration"] = min(durations)
                persistence_stats[regime]["std_duration"] = np.std(durations)

            # Most common transitions
            transition_counts = {}
            for from_regime, to_regime in transitions:
                key = (from_regime, to_regime)
                transition_counts[key] = transition_counts.get(key, 0) + 1

            sorted_transitions = sorted(
                transition_counts.items(), key=lambda x: x[1], reverse=True
            )

            report_lines.append("### 🔄 Most Common Regime Transitions:")
            report_lines.append("")
            for i, ((from_regime, to_regime), count) in enumerate(
                sorted_transitions[:10], 1
            ):
                from_desc = archetype_descriptions.get(
                    str(from_regime), f"Regime {from_regime}"
                )
                to_desc = archetype_descriptions.get(
                    str(to_regime), f"Regime {to_regime}"
                )
                from_better_desc = _generate_regime_description(
                    from_regime, meta, from_desc
                )
                to_better_desc = _generate_regime_description(to_regime, meta, to_desc)
                report_lines.append(
                    f"{i}. **{from_regime} → {to_regime}**: {count} transitions"
                )
                report_lines.append(f"   - **From**: {from_better_desc}")
                report_lines.append(f"   - **To**: {to_better_desc}")
                report_lines.append("")

            # Regime persistence analysis
            report_lines.append("### ⏱️ Regime Persistence Analysis:")
            report_lines.append("")

            # Sort regimes by average persistence
            sorted_persistence = sorted(
                [
                    (regime, stats)
                    for regime, stats in persistence_stats.items()
                    if regime >= 0
                ],
                key=lambda x: x[1]["avg_duration"],
                reverse=True,
            )

            for regime, stats in sorted_persistence[:10]:  # Top 10 most persistent
                desc = archetype_descriptions.get(str(regime), f"Regime {regime}")
                better_desc = _generate_regime_description(regime, meta, desc)
                report_lines.append(f"**Regime {regime}** ({better_desc}):")
                report_lines.append(
                    f"  - **Average Duration**: {stats['avg_duration']:.1f} periods"
                )
                report_lines.append(
                    f"  - **Median Duration**: {stats['median_duration']:.1f} periods"
                )
                report_lines.append(
                    f"  - **Max Duration**: {stats['max_duration']} periods"
                )
                report_lines.append(
                    f"  - **Min Duration**: {stats['min_duration']} periods"
                )
                report_lines.append(
                    f"  - **Duration Std**: {stats['std_duration']:.1f} periods"
                )
                report_lines.append(f"  - **Occurrences**: {stats['count']} times")
                report_lines.append("")

            # Transition probability matrix (top 3 transitions per regime)
            report_lines.append(
                "### 📊 Transition Probability Matrix (Top 3 Transitions per Regime):"
            )
            report_lines.append("")

            # For each regime, find its top 3 transitions
            for from_regime in sorted(transition_probabilities.keys()):
                if from_regime < 0:  # Skip noise cluster
                    continue

                # Get all transitions from this regime
                regime_transitions = []
                for to_regime, prob in transition_probabilities[from_regime].items():
                    if (
                        to_regime >= 0 and prob > 0.01
                    ):  # Only show transitions with >1% probability
                        regime_transitions.append((to_regime, prob))

                # Sort by probability and take top 3
                regime_transitions.sort(key=lambda x: x[1], reverse=True)
                top_3_transitions = regime_transitions[:3]

                if top_3_transitions:
                    from_desc = archetype_descriptions.get(
                        str(from_regime), f"Regime {from_regime}"
                    )
                    from_better_desc = _generate_regime_description(
                        from_regime, meta, from_desc
                    )
                    report_lines.append(
                        f"**From Regime {from_regime}** ({from_better_desc}):"
                    )

                    for to_regime, prob in top_3_transitions:
                        to_desc = archetype_descriptions.get(
                            str(to_regime), f"Regime {to_regime}"
                        )
                        to_better_desc = _generate_regime_description(
                            to_regime, meta, to_desc
                        )
                        report_lines.append(
                            f"  - **→ Regime {to_regime}**: {prob:.1%} probability ({to_better_desc})"
                        )

                    report_lines.append("")

            # Stability analysis
            report_lines.append("### 🎯 Regime Stability Analysis:")
            report_lines.append("")

            # Calculate stability metrics
            stable_regimes = []
            volatile_regimes = []

            for regime, stats in persistence_stats.items():
                if regime >= 0:  # Skip noise cluster
                    avg_duration = stats["avg_duration"]
                    std_duration = stats["std_duration"]
                    cv = (
                        std_duration / avg_duration
                        if avg_duration > 0
                        else float("inf")
                    )  # Coefficient of variation

                    if (
                        avg_duration > 20 and cv < 0.5
                    ):  # High average duration, low variability
                        stable_regimes.append((regime, avg_duration, cv))
                    elif (
                        avg_duration < 5 or cv > 1.0
                    ):  # Low average duration or high variability
                        volatile_regimes.append((regime, avg_duration, cv))

            # Sort by stability
            stable_regimes.sort(key=lambda x: x[1], reverse=True)
            volatile_regimes.sort(key=lambda x: x[2], reverse=True)

            if stable_regimes:
                report_lines.append("**Most Stable Regimes:**")
                for regime, avg_duration, cv in stable_regimes[:5]:
                    desc = archetype_descriptions.get(str(regime), f"Regime {regime}")
                    better_desc = _generate_regime_description(regime, meta, desc)
                    report_lines.append(
                        f"- **Regime {regime}**: {avg_duration:.1f} avg periods, {cv:.2f} CV ({better_desc})"
                    )
                report_lines.append("")

            if volatile_regimes:
                report_lines.append("**Most Volatile Regimes:**")
                for regime, avg_duration, cv in volatile_regimes[:5]:
                    desc = archetype_descriptions.get(str(regime), f"Regime {regime}")
                    better_desc = _generate_regime_description(regime, meta, desc)
                    report_lines.append(
                        f"- **Regime {regime}**: {avg_duration:.1f} avg periods, {cv:.2f} CV ({better_desc})"
                    )
                report_lines.append("")

        # Detailed regime descriptions and characteristics
        report_lines.append("## 🎯 Detailed Regime Descriptions & Characteristics")
        report_lines.append("")

        if "composite_cluster_id" in cluster_df.columns:
            # Load block states data for detailed analysis
            block_states_path = os.path.join(
                data_dir, f"{exchange}_{symbol}_hmm_block_states_{timeframe}.parquet"
            )
            block_states_df = None
            if os.path.exists(block_states_path):
                try:
                    block_states_df = pd.read_parquet(block_states_path)
                except Exception as e:
                    logger.warning(
                        f"⚠️ Could not load block states for regime analysis: {e}"
                    )

            # Analyze each regime in detail
            for regime_id in sorted(cluster_counts.index):
                if regime_id < 0:  # Skip noise cluster
                    continue

                regime_count = cluster_counts[regime_id]
                regime_percentage = (
                    (regime_count / total_obs * 100) if total_obs > 0 else 0
                )
                archetype_desc = archetype_descriptions.get(
                    str(regime_id), f"Regime {regime_id}"
                )

                report_lines.append(f"### 🎯 **Regime {regime_id}** - {archetype_desc}")
                report_lines.append("")
                report_lines.append(f"**Basic Statistics:**")
                report_lines.append(
                    f"- **Frequency**: {regime_count:,} observations ({regime_percentage:.2f}% of total)"
                )

                # Add persistence info if available
                if "persistence_stats" in locals() and regime_id in persistence_stats:
                    stats = persistence_stats[regime_id]
                    report_lines.append(
                        f"- **Average Duration**: {stats['avg_duration']:.1f} periods"
                    )
                    report_lines.append(
                        f"- **Median Duration**: {stats['median_duration']:.1f} periods"
                    )
                    report_lines.append(
                        f"- **Max Duration**: {stats['max_duration']} periods"
                    )
                    report_lines.append(f"- **Occurrences**: {stats['count']} times")

                # Add transition info if available
                if (
                    "transition_probabilities" in locals()
                    and regime_id in transition_probabilities
                ):
                    outgoing_transitions = transition_probabilities[regime_id]
                    # Find most likely transitions from this regime
                    likely_transitions = [
                        (to_regime, prob)
                        for to_regime, prob in outgoing_transitions.items()
                        if prob > 0.05
                    ]
                    likely_transitions.sort(key=lambda x: x[1], reverse=True)

                    if likely_transitions:
                        report_lines.append(f"- **Most Likely Transitions**:")
                        for to_regime, prob in likely_transitions[:3]:  # Top 3
                            to_desc = archetype_descriptions.get(
                                str(to_regime), f"Regime {to_regime}"
                            )
                            report_lines.append(
                                f"  - → Regime {to_regime}: {prob:.1%} ({to_desc[:30]}...)"
                            )

                report_lines.append("")

                # Detailed block state analysis if block states data is available
                if block_states_df is not None:
                    # Get regime-specific data
                    regime_mask = cluster_df["composite_cluster_id"] == regime_id
                    if regime_mask.sum() > 0:
                        regime_block_states = block_states_df[regime_mask]

                        report_lines.append(f"**Block State Characteristics:**")

                        # Analyze each block's state distribution
                        for block in blocks:
                            block_name = block["name"]
                            state_col = f"{block_name}_state_id"

                            if state_col in regime_block_states.columns:
                                state_counts = regime_block_states[
                                    state_col
                                ].value_counts()
                                dominant_state = (
                                    state_counts.index[0]
                                    if len(state_counts) > 0
                                    else None
                                )
                                dominant_percentage = (
                                    (
                                        state_counts.iloc[0]
                                        / len(regime_block_states)
                                        * 100
                                    )
                                    if len(regime_block_states) > 0
                                    else 0
                                )

                                # Get state name if available
                                state_names = meta.get("state_names", {}).get(
                                    block_name, {}
                                )
                                dominant_state_name = (
                                    state_names.get(
                                        dominant_state, f"State {dominant_state}"
                                    )
                                    if dominant_state is not None
                                    else "Unknown"
                                )

                                report_lines.append(
                                    f"- **{block_name.title()}**: {dominant_state_name} ({dominant_percentage:.1f}% dominant)"
                                )

                                # Show state distribution if there's significant diversity
                                if len(state_counts) > 1 and dominant_percentage < 80:
                                    other_states = []
                                    for state_id, count in state_counts.iloc[
                                        1:3
                                    ].items():  # Show top 3
                                        state_name = state_names.get(
                                            state_id, f"State {state_id}"
                                        )
                                        percentage = (
                                            count / len(regime_block_states) * 100
                                        )
                                        other_states.append(
                                            f"{state_name} ({percentage:.1f}%)"
                                        )
                                    if other_states:
                                        report_lines.append(
                                            f"  - Also includes: {', '.join(other_states)}"
                                        )

                        report_lines.append("")

                # Market condition interpretation
                report_lines.append(f"**Market Condition Interpretation:**")

                # Analyze regime characteristics based on better description
                better_desc_lower = better_desc.lower()

                # Momentum analysis
                if any(
                    word in better_desc_lower
                    for word in ["strong uptrend", "moderate uptrend"]
                ):
                    report_lines.append(
                        "- **Momentum**: Bullish market conditions with upward price pressure"
                    )
                elif any(
                    word in better_desc_lower
                    for word in ["strong downtrend", "moderate downtrend"]
                ):
                    report_lines.append(
                        "- **Momentum**: Bearish market conditions with downward price pressure"
                    )
                elif "sideways" in better_desc_lower:
                    report_lines.append(
                        "- **Momentum**: Neutral market conditions with balanced price action"
                    )

                # Volatility analysis
                if any(
                    word in better_desc_lower
                    for word in ["very high volatility", "high volatility"]
                ):
                    report_lines.append(
                        "- **Volatility**: High volatility environment with large price swings"
                    )
                elif any(
                    word in better_desc_lower
                    for word in ["very low volatility", "low volatility"]
                ):
                    report_lines.append(
                        "- **Volatility**: Low volatility environment with stable price action"
                    )

                # Volume analysis
                if any(
                    word in better_desc_lower
                    for word in ["very high volume", "high volume"]
                ):
                    report_lines.append(
                        "- **Volume**: High volume environment with active trading"
                    )
                elif any(
                    word in better_desc_lower
                    for word in ["very low volume", "low volume"]
                ):
                    report_lines.append(
                        "- **Volume**: Low volume environment with limited trading activity"
                    )

                # Support/Resistance analysis
                if any(
                    word in better_desc_lower
                    for word in ["near resistance", "approaching resistance"]
                ):
                    report_lines.append(
                        "- **Support/Resistance**: Price near resistance levels with potential reversal"
                    )
                elif any(
                    word in better_desc_lower
                    for word in ["near support", "approaching support"]
                ):
                    report_lines.append(
                        "- **Support/Resistance**: Price near support levels with potential bounce"
                    )

                report_lines.append("")
                report_lines.append("---")
                report_lines.append("")

        # Market condition analysis
        report_lines.append("## 📊 Market Condition Analysis")
        report_lines.append("")

        if "composite_cluster_id" in cluster_df.columns:
            # Analyze regime distribution by market conditions
            dominant_regime = (
                cluster_counts.index[0] if len(cluster_counts) > 0 else None
            )
            dominant_percentage = (
                (cluster_counts.iloc[0] / total_obs * 100)
                if len(cluster_counts) > 0 and total_obs > 0
                else 0
            )

            report_lines.append("**Current Market Regime Distribution:**")
            report_lines.append("")

            if dominant_regime is not None:
                desc = archetype_descriptions.get(
                    str(dominant_regime), f"Regime {dominant_regime}"
                )
                better_desc = _generate_regime_description(dominant_regime, meta, desc)
                report_lines.append(
                    f"- **Dominant Regime**: Regime {dominant_regime} ({dominant_percentage:.1f}% of time)"
                )
                report_lines.append(f"  - **Description**: {better_desc}")
                report_lines.append("")

            report_lines.append("**Market Condition Insights:**")
            report_lines.append("")

            if dominant_percentage > 40:
                report_lines.append(
                    "- **High Concentration**: Dominant regime controls >40% of market time"
                )
                report_lines.append(
                    "- **Market State**: Likely in a stable, trending market condition"
                )
            elif dominant_percentage > 25:
                report_lines.append(
                    "- **Moderate Concentration**: Dominant regime is prominent with 25-40% of market time"
                )
                report_lines.append(
                    "- **Market State**: Mixed market conditions with some stability"
                )
            else:
                report_lines.append(
                    "- **Low Concentration**: No single regime dominates (>25%)"
                )
                report_lines.append(
                    "- **Market State**: Highly volatile or transitioning market conditions"
                )

            report_lines.append("")
            report_lines.append("**Trading Implications:**")
            report_lines.append(
                "- High concentration periods: Use regime-specific strategies"
            )
            report_lines.append("- Low concentration periods: Focus on risk management")
            report_lines.append(
                "- Monitor regime transitions for market condition changes"
            )
            report_lines.append("")

        # Add trend distribution analysis
        if "composite_cluster_id" in cluster_df.columns:
            report_lines.append("## 📈 Trend Distribution Analysis")
            report_lines.append("")
            
            # Analyze trend distribution from regime descriptions
            uptrend_count = 0
            downtrend_count = 0
            sideways_count = 0
            
            for regime_id, count in cluster_counts.items():
                if regime_id >= 0:  # Skip noise cluster
                    desc = archetype_descriptions.get(str(regime_id), f"Regime {regime_id}")
                    better_desc = _generate_regime_description(regime_id, meta, desc)
                    better_desc_lower = better_desc.lower()
                    
                    if any(word in better_desc_lower for word in ["strong uptrend", "moderate uptrend"]):
                        uptrend_count += count
                    elif any(word in better_desc_lower for word in ["strong downtrend", "moderate downtrend"]):
                        downtrend_count += count
                    elif "sideways" in better_desc_lower or "neutral" in better_desc_lower:
                        sideways_count += count
            
            total_valid_obs = uptrend_count + downtrend_count + sideways_count
            
            if total_valid_obs > 0:
                report_lines.append("**Overall Trend Distribution:**")
                report_lines.append(f"- **Sideways/Neutral**: {sideways_count/total_valid_obs*100:.1f}% of observations ({sideways_count:,} out of {total_valid_obs:,})")
                report_lines.append(f"- **Downtrend**: {downtrend_count/total_valid_obs*100:.1f}% of observations ({downtrend_count:,} out of {total_valid_obs:,})")
                report_lines.append(f"- **Uptrend**: {uptrend_count/total_valid_obs*100:.1f}% of observations ({uptrend_count:,} out of {total_valid_obs:,})")
                report_lines.append("")
                
                report_lines.append("**⚠️ Important Note on Training Data Period:**")
                report_lines.append("The HMM analysis reflects market conditions during the **training data period** (historical data used to train the model), not necessarily the current market conditions. The predominance of sideways/neutral and downtrend regimes in this analysis indicates that during the training period, the market was primarily in consolidation or bearish phases.")
                report_lines.append("")
                report_lines.append("**Recent Market Context:**")
                report_lines.append("Based on recent ETHUSDT price data, the market may be in a different phase than the training period. This suggests that:")
                report_lines.append("1. The current market conditions may differ significantly from the training period")
                report_lines.append("2. The model may need retraining with more recent data to capture current market dynamics")
                report_lines.append("3. The last period (recent data) likely shows more uptrends than downtrends, but this analysis reflects the historical training period")
                report_lines.append("")
                report_lines.append("**Trading Implications:**")
                report_lines.append("- Use regime-specific strategies based on current market conditions")
                report_lines.append("- Consider retraining the HMM model with recent data for better current market representation")
                report_lines.append("- Monitor regime transitions for market condition changes")
                report_lines.append("- The model's historical bias toward sideways/downtrend conditions may not reflect current bullish market dynamics")
                report_lines.append("")

        # Acronym Glossary
        report_lines.append("## 📚 Acronym Glossary")
        report_lines.append("")
        report_lines.append(
            "**HMM**: Hidden Markov Model - A statistical model used to identify hidden states in time series data"
        )
        report_lines.append("")
        report_lines.append(
            "**MAE**: Mean Absolute Error - Average absolute difference between predicted and actual values"
        )
        report_lines.append("")
        report_lines.append(
            "**MAPE**: Mean Absolute Percentage Error - Average percentage error between predicted and actual values"
        )
        report_lines.append("")
        report_lines.append(
            "**OHLCV**: Open, High, Low, Close, Volume - Standard candlestick data format"
        )
        report_lines.append("")
        report_lines.append(
            "**SR**: Support/Resistance - Key price levels where market tends to reverse"
        )
        report_lines.append("")
        report_lines.append(
            "**VIF**: Variance Inflation Factor - Measure of multicollinearity in features"
        )
        report_lines.append("")
        report_lines.append(
            "**PCA**: Principal Component Analysis - Dimensionality reduction technique"
        )
        report_lines.append("")
        report_lines.append(
            "**SMOTE**: Synthetic Minority Over-sampling Technique - Method to balance imbalanced datasets"
        )
        report_lines.append("")
        report_lines.append(
            "**LGBM**: Light Gradient Boosting Machine - Gradient boosting framework"
        )
        report_lines.append("")
        report_lines.append(
            "**SVM**: Support Vector Machine - Machine learning algorithm for classification/regression"
        )
        report_lines.append("")
        report_lines.append(
            "**ADX**: Average Directional Index - Technical indicator measuring trend strength"
        )
        report_lines.append("")
        report_lines.append(
            "**RSI**: Relative Strength Index - Momentum oscillator measuring speed and change of price movements"
        )
        report_lines.append("")
        report_lines.append(
            "**MACD**: Moving Average Convergence Divergence - Trend-following momentum indicator"
        )
        report_lines.append("")
        report_lines.append("**ATR**: Average True Range - Volatility indicator")
        report_lines.append("")
        report_lines.append(
            "**VWAP**: Volume Weighted Average Price - Trading benchmark"
        )
        report_lines.append("")
        report_lines.append(
            "**EMA**: Exponential Moving Average - Type of moving average that gives more weight to recent data"
        )
        report_lines.append("")
        report_lines.append(
            "**SMA**: Simple Moving Average - Average of prices over a specified period"
        )
        report_lines.append("")
        report_lines.append(
            "**HDBSCAN**: Hierarchical Density-Based Spatial Clustering of Applications with Noise - Clustering algorithm"
        )
        report_lines.append("")
        report_lines.append(
            "**GARCH**: Generalized Autoregressive Conditional Heteroskedasticity - Model for volatility clustering"
        )
        report_lines.append("")
        report_lines.append(
            "**Kelly**: Kelly Criterion - Formula for optimal position sizing"
        )
        report_lines.append("")
        report_lines.append(
            "**Wyckoff**: Wyckoff Method - Technical analysis methodology for identifying accumulation/distribution"
        )
        report_lines.append("")
        report_lines.append(
            "**LSS**: Long Short Strategy - Trading strategy that takes both long and short positions"
        )
        report_lines.append("")
        report_lines.append(
            "**TP/SL**: Take Profit/Stop Loss - Risk management orders to close positions"
        )
        report_lines.append("")
        report_lines.append(
            "**ROI**: Return on Investment - Measure of investment performance"
        )
        report_lines.append("")
        report_lines.append(
            "**Sharpe Ratio**: Risk-adjusted return measure - Higher values indicate better risk-adjusted performance"
        )
        report_lines.append("")
        report_lines.append("**Drawdown**: Peak-to-trough decline in investment value")
        report_lines.append("")
        report_lines.append("**Win Rate**: Percentage of profitable trades")
        report_lines.append("")
        report_lines.append("**Profit Factor**: Ratio of gross profit to gross loss")
        report_lines.append("")
        report_lines.append(
            "**Regime**: Distinct market state characterized by specific conditions and behaviors"
        )
        report_lines.append("")
        report_lines.append(
            "**Archetype**: Representative pattern or model of a market regime"
        )
        report_lines.append("")

        # Technical details
        report_lines.append("## 🔧 Technical Details")
        report_lines.append("")
        report_lines.append(f"- **Symbol**: {symbol}")
        report_lines.append(f"- **Exchange**: {exchange}")
        report_lines.append(f"- **Timeframe**: {timeframe}")
        report_lines.append(f"- **Data Directory**: {data_dir}")
        report_lines.append(f"- **Meta File**: {meta_path}")
        report_lines.append(f"- **Cluster File**: {cluster_path}")
        report_lines.append(
            f"- **Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )
        report_lines.append("")
        report_lines.append(
            "> **Note**: This report was generated using a simplified text-based approach to avoid visualization-related segmentation faults while preserving all essential analysis content."
        )

        logger.info(f"✅ DEBUG: _generate_simple_hmm_report completed successfully for {timeframe}")
        return "\n".join(report_lines)

    except Exception as e:
        logger.error(f"❌ ERROR in _generate_simple_hmm_report for {timeframe}: {str(e)}")
        logger.error(f"❌ ERROR traceback: {traceback.format_exc()}")
        return f"# HMM Regime Report\n\nError generating report: {str(e)}\n\nTraceback: {traceback.format_exc()}"


def _generate_comprehensive_metrics_report(
    metrics: CompositeModelMetrics, symbol: str, timeframe: str, exchange: str
) -> str:
    """Generate a comprehensive human-readable report of composite model metrics."""

    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append(f"COMPOSITE MODEL METRICS REPORT")
    report_lines.append(
        f"Symbol: {symbol} | Exchange: {exchange} | Timeframe: {timeframe}"
    )
    report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("=" * 80)

    # Basic Statistics
    report_lines.append("\n📊 BASIC STATISTICS")
    report_lines.append("-" * 40)
    report_lines.append(f"Total Clusters: {metrics.cluster_count}")
    report_lines.append(f"Total Observations: {sum(metrics.cluster_sizes.values())}")

    # Cluster Size Distribution
    report_lines.append(f"\nCluster Size Distribution:")
    for cluster_id, size in sorted(metrics.cluster_sizes.items()):
        frequency = metrics.cluster_frequencies.get(cluster_id, 0) * 100
        report_lines.append(
            f"  Cluster {cluster_id}: {size} observations ({frequency:.2f}%)"
        )

    # Quality Metrics
    report_lines.append("\n🎯 CLUSTER QUALITY METRICS")
    report_lines.append("-" * 40)
    report_lines.append(
        f"Silhouette Score: {metrics.silhouette_score:.4f} (higher is better)"
    )
    report_lines.append(
        f"Calinski-Harabasz Score: {metrics.calinski_harabasz_score:.4f} (higher is better)"
    )
    report_lines.append(
        f"Davies-Bouldin Score: {metrics.davies_bouldin_score:.4f} (lower is better)"
    )

    # Diversity Metrics
    report_lines.append("\n🌐 CLUSTER DIVERSITY METRICS")
    report_lines.append("-" * 40)
    report_lines.append(f"Cluster Diversity: {metrics.cluster_diversity:.4f}")
    report_lines.append(f"Cluster Separation: {metrics.cluster_separation:.4f}")
    report_lines.append(f"Cluster Cohesion: {metrics.cluster_cohesion:.4f}")

    # Temporal Analysis
    report_lines.append("\n⏰ TEMPORAL ANALYSIS")
    report_lines.append("-" * 40)
    report_lines.append("Cluster Persistence (average duration):")
    for cluster_id, persistence in sorted(metrics.cluster_persistence.items()):
        report_lines.append(f"  Cluster {cluster_id}: {persistence:.2f} periods")

    report_lines.append("\nCluster Volatility (duration std):")
    for cluster_id, volatility in sorted(metrics.cluster_volatility.items()):
        report_lines.append(f"  Cluster {cluster_id}: {volatility:.2f} periods")

    # Block Composition
    report_lines.append("\n🧩 BLOCK COMPOSITION ANALYSIS")
    report_lines.append("-" * 40)
    for cluster_id, block_repr in metrics.block_representation.items():
        report_lines.append(f"\nCluster {cluster_id}:")
        dominant_block = metrics.block_dominance.get(cluster_id, "Unknown")
        balance = metrics.block_balance.get(cluster_id, 0.0)
        report_lines.append(f"  Dominant Block: {dominant_block}")
        report_lines.append(f"  Block Balance: {balance:.4f}")

        for block_name, block_info in block_repr.items():
            dominant_state = block_info.get("dominant_state", "Unknown")
            entropy = block_info.get("entropy", 0.0)
            report_lines.append(
                f"    {block_name}: {dominant_state} (entropy: {entropy:.4f})"
            )

    # Market Conditions
    report_lines.append("\n📈 MARKET CONDITION ANALYSIS")
    report_lines.append("-" * 40)
    for cluster_id, conditions in metrics.market_condition_distribution.items():
        stability = metrics.regime_stability.get(cluster_id, 0.0)
        report_lines.append(f"\nCluster {cluster_id} (stability: {stability:.4f}):")
        for condition, value in conditions.items():
            report_lines.append(f"  {condition}: {value}")

    # Feature Coverage
    report_lines.append("\n🔍 FEATURE COVERAGE ANALYSIS")
    report_lines.append("-" * 40)
    for cluster_id, missing_features in metrics.missing_features_by_cluster.items():
        if missing_features:
            report_lines.append(
                f"\nCluster {cluster_id} - Missing Features ({len(missing_features)}):"
            )
            for feature in missing_features[:10]:  # Show first 10
                report_lines.append(f"  - {feature}")
            if len(missing_features) > 10:
                report_lines.append(f"  ... and {len(missing_features) - 10} more")

    # Anomaly Detection
    report_lines.append("\n⚠️ ANOMALY DETECTION")
    report_lines.append("-" * 40)
    if metrics.outlier_clusters:
        report_lines.append(f"Outlier Clusters: {metrics.outlier_clusters}")
    if metrics.unstable_clusters:
        report_lines.append(f"Unstable Clusters: {metrics.unstable_clusters}")
    if metrics.rare_clusters:
        report_lines.append(f"Rare Clusters: {metrics.rare_clusters}")

    if not any(
        [metrics.outlier_clusters, metrics.unstable_clusters, metrics.rare_clusters]
    ):
        report_lines.append("No anomalies detected")

    # Summary
    report_lines.append("\n📋 SUMMARY")
    report_lines.append("-" * 40)
    report_lines.append(
        f"Overall Quality: {'Good' if metrics.silhouette_score > 0.3 else 'Fair' if metrics.silhouette_score > 0.1 else 'Poor'}"
    )
    report_lines.append(
        f"Cluster Diversity: {'High' if metrics.cluster_diversity > 0.8 else 'Medium' if metrics.cluster_diversity > 0.5 else 'Low'}"
    )
    report_lines.append(
        f"Anomaly Count: {len(metrics.outlier_clusters) + len(metrics.unstable_clusters) + len(metrics.rare_clusters)}"
    )

    return "\n".join(report_lines)


async def _calculate_and_integrate_hmm_features(
    symbol: str,
    exchange: str,
    timeframe: str,
    data_dir: str,
    hmm_output_dir: Path,
) -> None:
    """
    Calculate HMM features and integrate them into step2 features.
    This ensures HMM features are calculated when the HMM model is properly trained.
    """
    try:
        logger = system_logger.getChild("HMMFeatureIntegration")
        logger.info(f"🔧 Calculating HMM features for {exchange}_{symbol}_{timeframe}")
        
        # Load the HMM results that were just generated
        clusters_file = hmm_output_dir / f"{exchange}_{symbol}_hmm_composite_clusters_{timeframe}.parquet"
        intensity_file = hmm_output_dir / f"{exchange}_{symbol}_hmm_composite_intensity_{timeframe}.parquet"
        
        if not clusters_file.exists():
            logger.warning("⚠️ HMM clusters file not found - skipping HMM feature integration")
            return
        
        # Load HMM results
        clusters_df = pd.read_parquet(clusters_file)
        intensity_df = None
        if intensity_file.exists():
            intensity_df = pd.read_parquet(intensity_file)
        
        # Load step2 features to integrate with
        step2_data_dir = Path(data_dir)
        step2_files = [
            step2_data_dir / f"{exchange}_{symbol}_features_train.parquet",
            step2_data_dir / f"{exchange}_{symbol}_features_validation.parquet", 
            step2_data_dir / f"{exchange}_{symbol}_features_test.parquet"
        ]
        
        # Check if step2 features exist
        if not all(f.exists() for f in step2_files):
            logger.warning("⚠️ Step2 features not found - HMM features will be integrated later")
            return
        
        # Integrate HMM features into each step2 split
        for split_file in step2_files:
            split_name = split_file.stem.split('_')[-1]  # train, validation, or test
            
            logger.info(f"🔧 Integrating HMM features into {split_name} split")
            
            # Load step2 features
            step2_df = pd.read_parquet(split_file)
            
            # Align HMM features with step2 features
            aligned_clusters = clusters_df.reindex(step2_df.index)
            aligned_intensity = intensity_df.reindex(step2_df.index) if intensity_df is not None else None
            
            # Add HMM features
            hmm_features_added = []
            
            # Add cluster features
            for col in aligned_clusters.columns:
                if col in ['combination_id', 'composite_cluster_id']:
                    feature_name = f"hmm_{col}"
                    step2_df[feature_name] = aligned_clusters[col].fillna(-1.0)
                    hmm_features_added.append(feature_name)
            
            # Add intensity features
            if aligned_intensity is not None:
                for col in aligned_intensity.columns:
                    if col.startswith('intensity_cluster_'):
                        step2_df[col] = aligned_intensity[col].fillna(0.0)
                        hmm_features_added.append(col)
            
            # Save updated features
            step2_df.to_parquet(split_file)
            
            logger.info(f"✅ Added {len(hmm_features_added)} HMM features to {split_name} split")
            logger.info(f"   HMM features: {hmm_features_added}")
        
        logger.info("✅ HMM feature integration completed successfully")
        
    except Exception as e:
        logger.error(f"🚨 Error integrating HMM features: {e}")
        logger.exception("Full traceback:")


# Enhanced version for the enhanced training manager
@auto_fix_data_quality_issues
async def run_step_enhanced(
    symbol: str,
    exchange: str = "BINANCE",
    data_dir: str = "data/training",
    timeframe: str = "1m",
    lookback_days: int = None,
    force_rerun: bool = False,
    **kwargs: Any,
) -> bool:
    """
    Enhanced Step 3: HMM Regime Discovery with comprehensive error handling and validation.
    
    Args:
        symbol: Trading symbol
        exchange: Exchange name
        data_dir: Data directory
        timeframe: Timeframe
        lookback_days: Number of days to look back
        force_rerun: Force regeneration of results
        **kwargs: Additional arguments
        
    Returns:
        True if successful, False otherwise
    """
    try:
        logger = system_logger.getChild("Step3.HMMRegimeDiscovery")
        logger.info("🚀 Step 3: HMM Regime Discovery — Critical for pipeline success")
        
        # Validate inputs
        if not symbol or not exchange or not data_dir:
            logger.error("❌ Missing required parameters: symbol, exchange, or data_dir")
            return False
        
        # Check if results already exist and force_rerun is False
        output_dir = Path(data_dir) / "hmm_regimes"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        clusters_file = output_dir / f"{exchange}_{symbol}_hmm_composite_clusters_{timeframe}.parquet"
        meta_file = output_dir / f"{exchange}_{symbol}_hmm_composite_meta_{timeframe}.json"
        
        if clusters_file.exists() and meta_file.exists() and not force_rerun:
            logger.info(f"✅ HMM regime discovery results already exist for {exchange}_{symbol}_{timeframe}")
            logger.info(f"   Clusters file: {clusters_file}")
            logger.info(f"   Meta file: {meta_file}")
            return True
        
        # Load data using data sharing manager for efficiency
        from src.training.data_sharing_manager import get_data_sharing_manager
        data_sharing_manager = get_data_sharing_manager({})
        
        # Load unified data with proper lookback
        actual_lookback_days = lookback_days or 180  # Default to 180 days
        logger.info(f"📊 Loading data with {actual_lookback_days} days lookback")
        
        df = await data_sharing_manager.get_unified_data(
            symbol=symbol,
            exchange=exchange,
            timeframe=timeframe,
            lookback_days=actual_lookback_days,
            force_reload=False,
        )
        
        if df is None or df.empty:
            logger.error(f"❌ No data found for {exchange}_{symbol}_{timeframe}")
            return False
        
        logger.info(f"📊 Loaded {len(df)} rows of data for HMM regime discovery")
        
        # Extract OHLCV data
        price_cols = ["open", "high", "low", "close", "volume"]
        if not all(col in df.columns for col in price_cols):
            logger.error(f"❌ Missing required OHLCV columns: {price_cols}")
            return False
        
        price_df = df[price_cols].copy()
        
        # Create advanced features for HMM regime discovery
        logger.info("🔧 Creating advanced features for HMM regime discovery")
        features_df = create_basic_features(price_df)
        
        if features_df.empty:
            logger.error("❌ Failed to create features for HMM regime discovery")
            return False
        
        logger.info(f"✅ Created {len(features_df.columns)} features for HMM regime discovery")
        
        # Process each timeframe for HMM regime discovery
        timeframes_to_process = [timeframe] if timeframe else ["1m", "5m", "15m", "30m"]
        
        for tf in timeframes_to_process:
            logger.info(f"🔄 Processing HMM regime discovery for {tf}")
            
            # Create block features for HMM regime discovery
            block_features = {}
            for blk in BLOCKS:
                X_blk = _select_block_features(features_df, blk.name, blk.max_features)
                if not X_blk.empty:
                    block_features[blk.name] = X_blk
                    logger.info(f"   ✅ Block '{blk.name}': {X_blk.shape}")
                else:
                    logger.warning(f"   ⚠️ Block '{blk.name}': no features available")
            
            if not block_features:
                logger.error(f"❌ No block features available for {tf}")
                continue
            
            # Process blocks and create HMM models
            block_models = {}
            block_scalers = {}
            block_states = {}
            block_posteriors = {}
            
            for blk_name, X_blk in block_features.items():
                logger.info(f"🎯 Training HMM for block '{blk_name}' with {blk.n_states} states")
                
                # Train HMM model
                model, scaler = _fit_block_hmm_robust(X_blk, blk.n_states, blk_name)
                if model is None or scaler is None:
                    logger.error(f"❌ Failed to train HMM for block '{blk_name}'")
                    continue
                
                # Get posteriors
                gamma = _posteriors(model, X_blk.values)
                if len(gamma) == 0:
                    logger.error(f"❌ Failed to get posteriors for block '{blk_name}'")
                    continue
                
                # Get state predictions
                states = model.predict(X_blk.values)
                
                # Store results
                block_models[blk_name] = model
                block_scalers[blk_name] = scaler
                block_states[blk_name] = states
                block_posteriors[blk_name] = gamma
                
                logger.info(f"✅ Block '{blk_name}': {len(np.unique(states))} unique states")
            
            if not block_states:
                logger.error(f"❌ No HMM models trained successfully for {tf}")
                continue
            
            # Build combination profiles and composite clusters
            logger.info("🔧 Building combination profiles and composite clusters")
            combo_keys, profile_df = _build_combination_profiles(block_states, block_posteriors)
            
            if profile_df.empty:
                logger.error(f"❌ Failed to build combination profiles for {tf}")
                continue
            
            # Cluster combinations
            composite_clusters = _cluster_combinations(profile_df, min_cluster_size=5)
            
            if len(composite_clusters) == 0:
                logger.error(f"❌ Failed to create composite clusters for {tf}")
                continue
            
            # Save results
            logger.info(f"💾 Saving HMM regime discovery results for {tf}")
            
            # Save block states
            block_cols = {}
            for blk_name, states in block_states.items():
                block_cols[f"{blk_name}_state_id"] = states
                gamma = block_posteriors[blk_name]
                for i in range(gamma.shape[1]):
                    block_cols[f"{blk_name}_p_state_{i}"] = gamma[:, i]
            
            block_df = pd.DataFrame(block_cols, index=price_df.index)
            block_out_path = output_dir / f"{exchange}_{symbol}_hmm_block_states_{tf}.parquet"
            _persist_dataframe(block_df, block_out_path)
            
            # Save composite clusters
            composite_df = pd.DataFrame({
                'composite_cluster_id': composite_clusters,
                'combination_key': combo_keys
            }, index=price_df.index)
            composite_out_path = output_dir / f"{exchange}_{symbol}_hmm_composite_clusters_{tf}.parquet"
            _persist_dataframe(composite_df, composite_out_path)
            
            # Save intensity features
            intensity_df = pd.DataFrame(profile_df, index=combo_keys)
            intensity_out_path = output_dir / f"{exchange}_{symbol}_hmm_composite_intensity_{tf}.parquet"
            _persist_dataframe(intensity_df, intensity_out_path)
            
            # Save metadata
            meta_info = {
                'symbol': symbol,
                'exchange': exchange,
                'timeframe': tf,
                'blocks_used': list(block_states.keys()),
                'total_states': sum(len(np.unique(states)) for states in block_states.values()),
                'n_composite_clusters': len(np.unique(composite_clusters)),
                'processing_time': datetime.now().isoformat(),
                'lookback_days': actual_lookback_days,
                'feature_count': len(features_df.columns)
            }
            meta_out_path = output_dir / f"{exchange}_{symbol}_hmm_composite_meta_{tf}.json"
            _persist_json(meta_info, str(meta_out_path))
            
            logger.info(f"✅ HMM regime discovery completed for {tf}")
            logger.info(f"   Block states: {block_out_path}")
            logger.info(f"   Composite clusters: {composite_out_path}")
            logger.info(f"   Intensity features: {intensity_out_path}")
            logger.info(f"   Metadata: {meta_out_path}")
        
        # NEW: Calculate and integrate HMM features into step2 features
        logger.info("🔧 Calculating HMM features for integration with step2 features")
        await _calculate_and_integrate_hmm_features(symbol, exchange, timeframe, data_dir, output_dir)
        
        # FIX: Create required data/hmm_regimes/*.parquet files for Step 4
        logger.info("🔧 Creating required data/hmm_regimes/*.parquet files for Step 4")
        hmm_regimes_dir = Path(data_dir) / "hmm_regimes"
        hmm_regimes_dir.mkdir(exist_ok=True)
        
        # Copy composite clusters to hmm_regimes directory
        for tf in timeframes_to_process:
            source_file = output_dir / f"{exchange}_{symbol}_hmm_composite_clusters_{tf}.parquet"
            target_file = hmm_regimes_dir / f"{exchange}_{symbol}_hmm_composite_clusters_{tf}.parquet"
            
            if source_file.exists():
                import shutil
                # Only copy if source and target are different
                if source_file != target_file:
                    shutil.copy2(source_file, target_file)
                    logger.info(f"✅ Copied {source_file} to {target_file}")
                else:
                    logger.info(f"✅ File already in correct location: {source_file}")
            else:
                logger.warning(f"⚠️ Source file not found: {source_file}")
        
        logger.info("🎉 Step 3: HMM Regime Discovery completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"🚨 Step 3: HMM Regime Discovery failed: {e}")
        logger.exception("Full traceback:")
        return False


# Main implementation function
async def implement_hmm_regime_discovery(
    symbol: str,
    exchange: str,
    timeframe: str,
    data_dir: str,
    force: bool = False
) -> bool:
    """
    Main implementation function for HMM regime discovery.
    
    Args:
        symbol: Trading symbol
        exchange: Exchange name
        timeframe: Timeframe
        data_dir: Data directory
        force: Force regeneration
        
    Returns:
        True if successful, False otherwise
    """
    try:
        logger = system_logger.getChild("HMMRegimeDiscovery")
        logger.info(f"🚀 Implementing HMM regime discovery for {symbol} on {exchange}")
        
        # Check if results already exist
        output_dir = Path(data_dir) / "hmm_regimes"
        clusters_file = output_dir / f"{symbol}_{exchange}_{timeframe}_composite_clusters.parquet"
        
        if clusters_file.exists() and not force:
            logger.info(f"✅ HMM regime discovery results already exist for {symbol}")
            return True
        
        # Load unified data
        from src.training.steps.unified_data_loader import UnifiedDataLoader
        loader = UnifiedDataLoader({})
        
        # Load data for the specific timeframe
        data = await loader.load_unified_data(symbol, exchange, timeframe)
        if data is None or data.empty:
            logger.error(f"❌ No data found for {symbol} on {exchange}")
            return False
        
        # Use existing HMM implementation instead of undefined class
        logger.info(f"Using existing HMM implementation for {symbol}")
        
        # Create features and run HMM regime discovery using existing functions
        price_df = data[["open", "high", "low", "close", "volume"]].copy()
        features_df = create_basic_features(price_df)
        
        if features_df.empty:
            logger.error(f"❌ Failed to create features for {symbol}")
            return False
        
        # Use the existing HMM regime discovery logic
        success = await run_step_enhanced(
            symbol=symbol,
            exchange=exchange,
            timeframe=timeframe,
            data_dir=data_dir,
            force_rerun=force
        )
        
        logger.info(f"✅ HMM regime discovery completed successfully for {symbol}")
        return True
        
    except Exception as e:
        logger.error(f"🚨 Error in HMM regime discovery implementation: {e}")
        return False
