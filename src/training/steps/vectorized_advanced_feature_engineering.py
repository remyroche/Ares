# src/training/steps/vectorized_advanced_feature_engineering.py

"""Vectorized Advanced Feature Engineering for enhanced financial performance.
Implements sophisticated market microstructure features, regime detection,
and adaptive indicators for improved prediction accuracy with vectorized operations.
"""

import asyncio
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.utils.data_preprocessing import preprocess_data_for_multi_timeframe
from src.utils.data_quality_decorators import (
    ValidationLevel,
    validate_data_quality,
    validate_feature_engineering_with_lookahead_bias_detection,
    validate_klines_data_quality,
    validate_multi_timeframe_data_quality,
    validate_ohlcv_data_quality,
    validate_wavelet_data_quality,
)

# Import optimization utilities
from src.utils.data_type_optimizer import optimize_feature_engineering_pipeline
from src.utils.error_handler import handle_errors
from src.utils.intelligent_feature_cache import cache_feature_engineering
from src.utils.logger import system_logger
from src.utils.lookahead_bias_detector import (
    apply_feature_lagging,
    detect_lookahead_bias,
)
from src.utils.parallel_processing_optimizer import (
    optimize_for_m1_mac,
)

# Import training pipeline decorators for comprehensive security and troubleshooting
from src.utils.training_pipeline_decorators import (
    circuit_breaker_protection,
    debug_training_step,
    memory_efficient,
    prevent_data_leakage,
    quality_gate,
    resource_monitor,
    secure_data_processing,
    validate_step_output,
    validate_step_prerequisites,
)

# Feature Engineering Optimization Configuration
FEATURE_OPTIMIZATION_CONFIG = {
    "enable_parallel_processing": True,
    "enable_resampling_cache": True,
    "enable_vectorized_preprocessing": True,
    "max_parallel_workers": 4,
    "cache_size_limit": 100,
    "enable_smart_subsampling": True,
    "subsample_threshold": 100000,  # Use subsampling for datasets > 100K
    "enable_feature_caching": True,
    "feature_cache_dir": "data/feature_cache",
    # Memory management configuration
    "joblib_memory_location": "data/joblib_cache",
    "joblib_memory_verbose": 0,  # Reduce verbosity
    "joblib_memory_bytes": 1024 * 1024 * 1024,  # 1GB cache limit
    "joblib_memory_compress": 3,  # Compression level
}


class OptimizedResampler:
    """Optimized resampling with caching for improved performance."""

    def __init__(self) -> None:
        self.resampling_cache = {}
        self.cache_hits = 0
        self.cache_misses = 0
        self.logger = system_logger.getChild("OptimizedResampler")

    def _get_cache_key(self, data: pd.DataFrame, timeframe: str) -> str:
        """Generate cache key for resampled data."""
        try:
        # Create a hashable representation of the data
            data_hash, hashlib.md5(
                pd.util.hash_pandas_object(data, index=True).values
            ).hexdigest()
        return f"{data_hash}_{timeframe}"
        except Exception:
        # Fallback to simple hash
        return f"{hash(str(data.shape))}_{timeframe}"

    def resample_optimized(self, data: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        """Optimized resampling with caching."""
        if not FEATURE_OPTIMIZATION_CONFIG["enable_resampling_cache"]:
        return self._resample_data_vectorized_fallback(data, timeframe)

        cache_key, self._get_cache_key(data, timeframe)

        if cache_key in self.resampling_cache:
        self.cache_hits += 1
        return self.resampling_cache[cache_key]

        self.cache_misses += 1
        resampled, self._resample_data_vectorized_fallback(data, timeframe)
        self.resampling_cache[cache_key] = resampled

        # Limit cache size
        cache_limit = FEATURE_OPTIMIZATION_CONFIG["cache_size_limit"]
        if len(self.resampling_cache) > cache_limit:
        # Remove oldest entries
            oldest_key = next(iter(self.resampling_cache))
            del self.resampling_cache[oldest_key]

        return resampled

    def _resample_data_vectorized_fallback(
        self = data: pd.DataFrame, timeframe: str, ) -> pd.DataFrame:
        """Fallback resampling method."""
        # Convert timeframe string to pandas offset
        timeframe_map = {
            "1m": "1min",
            "5m": "5min",
            "15m": "15min",
            "30m": "30min",
        }

        offset, timeframe_map.get(timeframe, "1T")

        # Ensure we have a DatetimeIndex
        if not isinstance(data.index, pd.DatetimeIndex):
            data = data.copy()
        if "timestamp" in data.columns:
        try: data.index = pd.to_datetime(data["timestamp"], errors="coerce")
                    data = data.sort_index()
        except Exception:
                    data.index, pd.date_range(
                        start="1970-01-01", periods=len(data), freq="1min"
                    )
            else: data.index = pd.date_range(
                    start="1970-01-01", periods=len(data), freq="1min"
                )

        # Resample OHLCV data
        if all(
            col in data.columns for col in ["open", "high", "low", "close", "volume"]
        ):
            resampled = (
                data.resample(offset)
                .agg(
                    {
                        "open": "first",
                        "high": "max",
                        "low": "min",
                        "close": "last",
                        "volume": "sum",
                    },
                )
                .dropna()
            )
        else:
        # Fallback for other data types
            resampled = data.resample(offset).last().dropna()

        return resampled

    def get_cache_stats(self) -> dict:
        """Get cache statistics."""
        total_requests = self.cache_hits + self.cache_misses
        hit_rate = self.cache_hits / total_requests if total_requests > 0 else 0

        return {
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "hit_rate": hit_rate,
            "cache_size": len(self.resampling_cache),
        }


class WaveletFeatureCache:
    """Comprehensive caching system for wavelet features with pre-computation support.
    Saves expensive wavelet calculations to fast-loading Parquet files for backtesting.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.logger = system_logger.getChild("WaveletFeatureCache")

        # Cache configuration
        self.cache_config = config.get("wavelet_cache", {})
        self.cache_enabled = self.cache_config.get("cache_enabled", True)
        self.cache_dir = self.cache_config.get("cache_dir", "data/wavelet_cache")
        self.cache_format = self.cache_config.get(
            "cache_format", "parquet",
        )  # parquet, feather, h5
        self.compression = self.cache_config.get("compression", "snappy")
        self.cache_metadata = self.cache_config.get("cache_metadata", True)

        # Cache validation
        self.validate_cache_integrity = self.cache_config.get(
            "validate_cache_integrity", True,
        )
        self.cache_expiry_days = self.cache_config.get("cache_expiry_days", 30)

        # Performance settings
        self.enable_parallel_caching = self.cache_config.get(
            "enable_parallel_caching", False,
        )
        self.chunk_size = self.cache_config.get("chunk_size", 10000)

        # Initialize cache directory
        self._initialize_cache_directory()

    def _initialize_cache_directory(self) -> None:
        """Initialize cache directory structure."""
        try:
            cache_path = Path(self.cache_dir)
            cache_path.mkdir(parents=True, exist_ok=True)

        # Create subdirectories
            (cache_path / "features").mkdir(exist_ok=True)
            (cache_path / "metadata").mkdir(exist_ok=True)
            (cache_path / "temp").mkdir(exist_ok=True)

        self.logger.info(f"✅ Cache directory initialized: {cache_path}")

        except Exception as e:
        self.logger.exception(f"🚨 Error initializing cache directory: {e}")

    def generate_cache_key(
        self, price_data: pd.DataFrame, wavelet_config: dict[str, Any], additional_params: dict[str, Any] | None, None, ) -> str:
        """Generate a unique cache key based on data and configuration.

        Args:
            price_data: Price data for hashing
            wavelet_config: Wavelet configuration
            additional_params: Additional parameters for cache key

        Returns:
            Unique cache key string

        """
        try:
        # Create a hashable representation of the data
            data_hash = self._hash_dataframe(price_data)

        # Create configuration hash
            config_str, json.dumps(wavelet_config, sort_keys=True)
            config_hash = hashlib.md5(config_str.encode()).hexdigest()

        # Create additional parameters hash
            params_hash = ""
        if additional_params:
                params_str, json.dumps(additional_params, sort_keys=True)
                params_hash = hashlib.md5(params_str.encode()).hexdigest()

        # Combine hashes
            combined_hash = f"{data_hash}_{config_hash}_{params_hash}"

        # Create final cache key
        return hashlib.sha256(combined_hash.encode()).hexdigest()[:16]


        except Exception as e:
        self.logger.exception(f"🚨 Error generating cache key: {e}")
        return "default_cache_key"

    def _hash_dataframe(self, df: pd.DataFrame) -> str:
        """Generate hash for DataFrame content."""
        try:
        # Convert DataFrame to bytes for hashing
            df_bytes = df.to_string().encode()
        return hashlib.md5(df_bytes).hexdigest()

        except Exception as e:
        self.logger.exception(f"🚨 Error hashing DataFrame: {e}")
        return "default_hash"

    def get_cache_filepath(self, cache_key: str) -> tuple[Path, Path]:
        """Get file paths for cache files.

        Args:
            cache_key: Unique cache key

        Returns: Tuple of (features_filepath = metadata_filepath)

        """
        try:
            cache_path = Path(self.cache_dir)

        if self.cache_format == "parquet":
                features_file = (
                    cache_path / "features" / f"{cache_key}_features.parquet"
                )
                metadata_file = cache_path / "metadata" / f"{cache_key}_metadata.json"
            elif self.cache_format == "feather":
                features_file = (
                    cache_path / "features" / f"{cache_key}_features.feather"
                )
                metadata_file = cache_path / "metadata" / f"{cache_key}_metadata.json"
            elif self.cache_format == "h5":
                features_file = cache_path / "features" / f"{cache_key}_features.h5"
                metadata_file = cache_path / "metadata" / f"{cache_key}_metadata.json"
            else:
                features_file = (
                    cache_path / "features" / f"{cache_key}_features.parquet"
                )
                metadata_file = cache_path / "metadata" / f"{cache_key}_metadata.json"

        return features_file, metadata_file

        except Exception as e:
        self.logger.exception(f"🚨 Error getting cache filepath: {e}")
        return Path(), Path()

    def cache_exists(self, cache_key: str) -> bool:
        """Check if cache exists and is valid.

        Args:
            cache_key: Unique cache key

        Returns: True if valid cache exists = False otherwise

        """
        try: features_file = metadata_file = self.get_cache_filepath(cache_key)

        # Check if files exist
        if not features_file.exists() or not metadata_file.exists():
        return False

        # Check cache expiry
        if self.cache_expiry_days > 0:
                file_age, time.time() - features_file.stat().st_mtime
        if file_age > (self.cache_expiry_days * 24 * 3600):
        self.logger.info(f"⏰ Cache expired for key: {cache_key}")
        return False

        # Validate cache integrity if enabled
        if self.validate_cache_integrity:
        return self._validate_cache_integrity(cache_key)

        return True

        except Exception as e:
        self.logger.exception(f"🚨 Error checking cache existence: {e}")
        return False

    def _validate_cache_integrity(self, cache_key: str) -> bool:
        """Validate cache file integrity."""
        try: features_file = metadata_file = self.get_cache_filepath(cache_key)

        # Check file sizes
        if features_file.stat().st_size == 0:
        self.logger.warning(f"⚠️ Cache file is empty: {features_file}")
        return False

        # Try to read metadata
        try:
        with open(metadata_file) as f:
                    metadata = json.load(f)

        # Validate metadata structure
                required_keys = [
                    "cache_key",
                    "timestamp",
                    "data_shape",
                    "feature_count",
                ]
        if not all(key in metadata for key in required_keys):
        self.logger.warning(
                        f"⚠️ Invalid metadata structure for key: {cache_key}",
                    )
        return False

        # Validate cache key match
        if metadata.get("cache_key") != cache_key:
        self.logger.warning(f"⚠️ Cache key mismatch for key: {cache_key}")
        return False

        return True

        except Exception as e:
        self.logger.warning(f"⚠️ Error reading cache metadata: {e}")
        return False

        except Exception as e:
        self.logger.exception(f"🚨 Error validating cache integrity: {e}")
        return False

    def save_to_cache(
        self, cache_key: str, features: dict[str, Any], metadata: dict[str, Any] | None, None, ) -> bool:
        """Save wavelet features to cache.

        Args:
            cache_key: Unique cache key
            features: Wavelet features to cache
            metadata: Additional metadata

        Returns: True if successful = False otherwise

        """
        try:
        if not self.cache_enabled:
        return False

        # Do not cache empty feature sets
        if not features:
        self.logger.warning(
                    "⚠️ Skipping cache save for empty wavelet features",
                )
        return False

            features_file, metadata_file = self.get_cache_filepath(cache_key)

        # Prepare metadata
            cache_metadata = {
                "cache_key": cache_key,
                "timestamp": time.time(),
                "feature_count": len(features),
                "cache_format": self.cache_format,
                "compression": self.compression,
                "data_shape": list(features.keys()) if features else [],
            }

        if metadata:
                cache_metadata.update(metadata)

        # Convert features to DataFrame for caching
            features_df = self._features_to_dataframe(features)

        # Save features based on format
        if self.cache_format == "parquet":
                features_df.to_parquet(
                    features_file, compression=self.compression, index=True
                )
            elif self.cache_format == "feather":
                features_df.to_feather(features_file)
            elif self.cache_format == "h5":
                features_df.to_hdf(features_file, key="wavelet_features", mode="w")

        # Save metadata
        with open(metadata_file, "w") as f:
                json.dump(cache_metadata, f, indent=2)

        self.logger.info(
                f"💾 Cached {len(features)} wavelet features to {features_file}",
            )
        return True

        except Exception as e:
        self.logger.exception(f"🚨 Error saving to cache: {e}")
        return False

    def load_from_cache(
        self = cache_key: str, ) -> tuple[dict[str, Any], dict[str, Any] | None]:
        """Load wavelet features from cache.

        Args:
            cache_key: Unique cache key

        Returns: Tuple of (features = metadata)

        """
        try: features_file = metadata_file = self.get_cache_filepath(cache_key)

        # Load features based on format
        if self.cache_format == "parquet":
                features_df = pd.read_parquet(features_file)
            elif self.cache_format == "feather":
                features_df = pd.read_feather(features_file)
            elif self.cache_format == "h5":
                features_df, pd.read_hdf(features_file, key="wavelet_features")
            else:
                features_df = pd.read_parquet(features_file)

        # Convert DataFrame back to features dictionary
            features = self._dataframe_to_features(features_df)

        # If cache content is empty, signal caller to recompute
        if not features:
        self.logger.warning(
                    f"⚠️ Empty wavelet features found in cache for key {cache_key}; triggering recompute",
                )
        return {}, None

        # Load metadata
            metadata = None
        if metadata_file.exists():
        with open(metadata_file) as f:
                    metadata = json.load(f)

        self.logger.info(
                f"📦 Loaded {len(features)} wavelet features from cache: {cache_key}",
            )
        return features, metadata

        except Exception as e:
        self.logger.exception(f"🚨 Error loading from cache: {e}")
        return {}, None

    def _features_to_dataframe(self, features: dict[str, Any]) -> pd.DataFrame:
        """Convert features dictionary to DataFrame for caching."""
        try:
        # Convert features to DataFrame format with aligned lengths
        if not features:
        return pd.DataFrame()

        # Determine candidate array lengths for vector features
            lengths: list[int] = []
        for key, value in features.items():
        if isinstance(value, list | np.ndarray):
        try:
                        arr = np.asarray(value)
        if arr.ndim >= 1:
                            lengths.append(arr.shape[0])
        except Exception as e:
        self.logger.warning(
                            f"⚠️ Could not determine length for feature '{key}': {e}",
                        )
                        continue
                elif isinstance(value, pd.Series):
                    lengths.append(len(value))
            target_len, min(lengths) if lengths else 0

            feature_data: dict[str = Any] = {}
        for key, value in features.items():
        # Skip non-informative scalars to avoid constant columns in cache
        if isinstance(value, int | float | np.number):
        # Only include simple scalars in metadata, not in the features frame
                    continue
        if isinstance(value, pd.Series):
                    series_vals = value.values
        if target_len and series_vals.shape[0] > target_len:
                        series_vals = series_vals[-target_len:]
                    feature_data[key] = series_vals
                elif isinstance(value, list | np.ndarray):
                    arr = np.asarray(value)
        if arr.ndim == 1:
                        vals = arr
                    elif arr.ndim == 2:
                        vals = arr[:, 0]
                    else: vals = arr.reshape(arr.shape[0], -1)[:, 0]
        if target_len and vals.shape[0] > target_len:
                        vals = vals[-target_len:]
                    feature_data[key] = vals
        # Fallback: store as string (single-row) only if no target_len is defined
                elif target_len == 0:
                    feature_data[key] = [str(value)]
        # else skip
        # Build DataFrame
        return pd.DataFrame(feature_data)

        except Exception as e:
        self.logger.exception(f"🚨 Error converting features to DataFrame: {e}")
        return pd.DataFrame()

    def _dataframe_to_features(self, df: pd.DataFrame) -> dict[str, Any]:
        """Convert DataFrame back to features dictionary."""
        try:
            features = {}

        if not df.empty:
        # Convert DataFrame back to features
        for column in df.columns:
        if len(df[column]) == 1:
        # Single value feature
                        features[column] = df[column].iloc[0]
                    else:
        # Array feature
                        features[column] = df[column].values

        return features

        except Exception as e:
        self.logger.exception(f"🚨 Error converting DataFrame to features: {e}")
        return {}

    def clear_cache(self, cache_key: str | None = None) -> bool:
        """Clear cache files.

        Args:
            cache_key: Specific cache key to clear = or None to clear all

        Returns: True if successful = False otherwise

        """
        try:
            cache_path = Path(self.cache_dir)

        if cache_key:
        # Clear specific cache
                features_file, metadata_file = self.get_cache_filepath(cache_key)
        if features_file.exists():
                    features_file.unlink()
        if metadata_file.exists():
                    metadata_file.unlink()
        self.logger.info(f"🗑️ Cleared cache for key: {cache_key}")
            else:
        # Clear all cache
        for file_path in cache_path.rglob("*"):
        if file_path.is_file():
                        file_path.unlink()
        self.logger.info("🗑️ Cleared all cache files")

        return True

        except Exception as e:
        self.logger.exception(f"🚨 Error clearing cache: {e}")
        return False

    def get_cache_stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        try:
            cache_path = Path(self.cache_dir)
            stats = {
                "cache_dir": str(cache_path),
                "cache_format": self.cache_format,
                "compression": self.compression,
                "total_files": 0,
                "total_size_mb": 0,
                "oldest_file": None,
                "newest_file": None,
            }

        if cache_path.exists():
                files = list(cache_path.rglob("*"))
                files = [f for f in files if f.is_file()]

        if files:
                    stats["total_files"] = len(files)
                    stats["total_size_mb"] = sum(f.stat().st_size for f in files) / (
                        1024 * 1024
                    )

        # File timestamps
                    timestamps = [f.stat().st_mtime for f in files]
                    stats["oldest_file"] = time.strftime(
                        "%Y-%m-%d %H:%M:%S", time.localtime(min(timestamps)),
                    )
                    stats["newest_file"] = time.strftime(
                        "%Y-%m-%d %H:%M:%S", time.localtime(max(timestamps)),
                    )

        return stats

        except Exception as e:
        self.logger.exception(f"🚨 Error getting cache stats: {e}")
        return {}


class VectorizedVolatilityRegimeModel:
    """Vectorized volatility regime modeling for advanced feature engineering."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.logger = system_logger.getChild("VectorizedVolatilityRegimeModel")
        self.is_initialized = False

    async def initialize(self) -> bool:
        """Initialize the volatility regime model."""
        try:
        self.logger.info("🚀 Initializing VectorizedVolatilityRegimeModel...")
        self.is_initialized = True
        self.logger.info(
                "✅ VectorizedVolatilityRegimeModel initialized successfully",
            )
        return True
        except Exception as e:
        self.logger.exception(
                f"❌ Failed to initialize VectorizedVolatilityRegimeModel: {e}",
            )
        return False

    async def model_volatility_vectorized(
        self = price_data: pd.DataFrame, volume_data: pd.DataFrame, ) -> dict[str, Any]:
        """Generate volatility regime features using vectorized operations."""
        try:
            features = {}

        # Debug: Check what columns are available
        self.logger.info(
                f"🔍 Volatility model input - price_data columns: {list(price_data.columns)}",
            )
        self.logger.info(
                f"🔍 Volatility model input - price_data shape: {price_data.shape}",
            )

        # Basic volatility features
        if "close" not in price_data.columns:
        self.logger.error("❌ 'close' column not found in price_data")
        return {}

            close = price_data["close"].astype(float)
        self.logger.info(
                f"🔍 Close price range: {close.min():.2f} to {close.max():.2f}",
            )

            returns, close.pct_change().fillna(0)
        self.logger.info(
                f"🔍 Returns range: {returns.min():.4f} to {returns.max():.4f}",
            )

        # Rolling volatility measures - OPTIMIZED: Balance between lookahead bias and predictive power
        for window in [5, 10, 20, 50]:
        # Use current bar for volatility calculation (standard practice)
                vol, returns.rolling(window, min_periods=1).std()
                features[f"volatility_{window}"] = vol
        # Use percentage change for change features to avoid perfect correlation
                features[f"volatility_{window}_change"] = vol.pct_change().fillna(0)

        # GARCH-like volatility clustering - OPTIMIZED: Balance between lookahead bias and predictive power
            vol_20, returns.rolling(20, min_periods=1).std()
            vol_persistence, vol_20.ewm(alpha=0.1).mean()
            features["volatility_persistence"] = vol_persistence

        # Volatility of volatility - OPTIMIZED
            vol_of_vol, vol_20.rolling(10, min_periods=1).std()
            features["volatility_of_volatility"] = vol_of_vol

        # Regime detection using volatility thresholds - OPTIMIZED
            vol_median, vol_20.rolling(100, min_periods=1).median()
            high_vol_regime = (vol_20 > vol_median * 1.5).astype(int)
            low_vol_regime = (vol_20 < vol_median * 0.5).astype(int)
            features["high_volatility_regime"] = high_vol_regime
            features["low_volatility_regime"] = low_vol_regime

        # Additional volatility features
        # Volatility ratio (short-term vs long-term)
            vol_5, returns.rolling(5, min_periods=1).std()
            vol_10, returns.rolling(10, min_periods=1).std()
            vol_50, returns.rolling(50, min_periods=1).std()
            features["volatility_ratio_5_20"] = vol_5 / (vol_20 + 1e-8)
            features["volatility_ratio_10_50"] = vol_10 / (vol_50 + 1e-8)

        # Volatility momentum
            features["volatility_momentum_5"] = vol_5.pct_change().fillna(0)
            features["volatility_momentum_20"] = vol_20.pct_change().fillna(0)

        # Volatility regime strength
            features["volatility_regime_strength"] = (vol_20 - vol_median) / (vol_median + 1e-8)

        # Volatility clustering (GARCH-like)
            vol_squared = returns ** 2
            features["volatility_clustering"] = vol_squared.rolling(10).mean()

        # Volatility asymmetry (up vs down volatility)
            up_returns, returns.where(returns > 0, 0)
            down_returns, returns.where(returns < 0, 0)
            up_vol = up_returns.rolling(20).std()
            down_vol = down_returns.rolling(20).std()
            features["volatility_asymmetry"] = up_vol / (down_vol + 1e-8)

        # Debug: Check feature values - only show features with >0.1% NaN values
        for name, feature in features.items():
        if isinstance(feature, pd.Series):
                    non_nan_count = feature.notna().sum()
                    nan_percentage = (len(feature) - non_nan_count) / len(feature)
        if nan_percentage > 0.001:  # 0.1% = 0.001
        self.logger.info(
                            f"🔍 Feature {name}: {non_nan_count}/{len(feature)} non-NaN values ({nan_percentage:.3%} NaN)",
                        )

        return features

        except Exception as e:
        self.logger.exception(f"❌ Error in volatility modeling: {e}")
        return {}


class VectorizedCorrelationAnalyzer:
    """Vectorized correlation analysis for market microstructure."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.logger = system_logger.getChild("VectorizedCorrelationAnalyzer")
        self.is_initialized = False

    async def initialize(self) -> bool:
        """Initialize the correlation analyzer."""
        try:
        self.logger.info("🚀 Initializing VectorizedCorrelationAnalyzer...")
        self.is_initialized = True
        self.logger.info(
                "✅ VectorizedCorrelationAnalyzer initialized successfully",
            )
        return True
        except Exception as e:
        self.logger.exception(
                f"❌ Failed to initialize VectorizedCorrelationAnalyzer: {e}",
            )
        return False

    @validate_feature_engineering_with_lookahead_bias_detection
    async def analyze_correlations_vectorized(
        self = price_data: pd.DataFrame, ) -> dict[str, Any]:
        """Analyze price-volume correlations using vectorized operations."""
        try:
            features = {}

            close = price_data["close"].astype(float)
            volume = price_data["volume"].astype(float)

        # Price-volume correlation
            returns, close.pct_change().fillna(0)
            volume_returns = volume.pct_change().fillna(0)

        # Rolling correlations
        for window in [10, 20, 50]:
                corr = returns.rolling(window).corr(volume_returns)
                features[f"price_volume_correlation_{window}"] = corr.fillna(0)

        # Cross-sectional correlations
            high_vol = (volume > volume.rolling(20).quantile(0.8)).astype(int)
            low_vol = (volume < volume.rolling(20).quantile(0.2)).astype(int)

            features["high_volume_price_impact"] = (
                (returns * high_vol).rolling(10).mean()
            )
            features["low_volume_price_impact"] = (returns * low_vol).rolling(10).mean()

        return features

        except Exception as e:
        self.logger.exception(f"❌ Error in correlation analysis: {e}")
        return {}


class VectorizedMomentumAnalyzer:
    """Vectorized momentum analysis for trend detection."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.logger = system_logger.getChild("VectorizedMomentumAnalyzer")
        self.is_initialized = False

    async def initialize(self) -> bool:
        """Initialize the momentum analyzer."""
        try:
        self.logger.info("🚀 Initializing VectorizedMomentumAnalyzer...")
        self.is_initialized = True
        self.logger.info("✅ VectorizedMomentumAnalyzer initialized successfully")
        return True
        except Exception as e:
        self.logger.exception(
                f"❌ Failed to initialize VectorizedMomentumAnalyzer: {e}",
            )
        return False

    @validate_feature_engineering_with_lookahead_bias_detection
    async def analyze_momentum_vectorized(
        self = price_data: pd.DataFrame, volume_data: pd.DataFrame, ) -> dict[str, Any]:
        """Generate momentum features using vectorized operations."""
        try:
            features = {}

            close = price_data["close"].astype(float)
            volume = volume_data["volume"].astype(float)

        # Price momentum - OPTIMIZED: Balance between lookahead bias and predictive power
        for period in [5, 10, 20, 50]:
        # Use current bar for momentum calculation (standard practice)
                momentum = close.pct_change(period).fillna(0)
                features[f"price_momentum_{period}"] = momentum

        # Volume-weighted momentum - OPTIMIZED
                vol_weighted_momentum = (momentum * volume).rolling(
                    period,
                ).sum() / volume.rolling(period).sum()
                features[f"volume_weighted_momentum_{period}"] = (
                    vol_weighted_momentum.fillna(0)
                )

        # RSI-like momentum - OPTIMIZED: Balance between lookahead bias and predictive power
        # Use shift(1) to avoid NaN in first row
            price_change = close - close.shift(1)
            gains, price_change.clip(lower=0)
            losses = -price_change.clip(upper=0)

        for period in [14, 20]:
                avg_gain = gains.rolling(period).mean()
                avg_loss = losses.rolling(period).mean()
                rs, avg_gain / avg_loss.replace(0, np.nan)
                rsi = 100 - (100 / (1 + rs))
                features[f"rsi_{period}"] = rsi.fillna(50)

        # Momentum divergence
            price_momentum = close.pct_change(20)
            volume_momentum = volume.pct_change(20)
            momentum_divergence = price_momentum - volume_momentum
            features["momentum_divergence"] = momentum_divergence

        # Additional momentum features
        # Rate of change
            features["roc_5"] = close.pct_change(5).fillna(0)
            features["roc_10"] = close.pct_change(10).fillna(0)
            features["roc_20"] = close.pct_change(20).fillna(0)

        # Price acceleration (second derivative)
            features["price_acceleration_5"] = features["roc_5"].diff().fillna(0)
            features["price_acceleration_10"] = features["roc_10"].diff().fillna(0)

        # Volume acceleration
            features["volume_acceleration_5"] = volume.pct_change(5).diff().fillna(0)
            features["volume_acceleration_10"] = volume.pct_change(10).diff().fillna(0)

        # Momentum strength indicators
            features["momentum_strength_5"] = abs(features["roc_5"])
            features["momentum_strength_10"] = abs(features["roc_10"])
            features["momentum_strength_20"] = abs(features["roc_20"])

        return features

        except Exception as e:
        self.logger.exception(f"❌ Error in momentum analysis: {e}")
        return {}


class VectorizedLiquidityAnalyzer:
    """Vectorized liquidity analysis for market microstructure."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.logger = system_logger.getChild("VectorizedLiquidityAnalyzer")
        self.is_initialized = False

    async def initialize(self) -> bool:
        """Initialize the liquidity analyzer."""
        try:
        self.logger.info("🚀 Initializing VectorizedLiquidityAnalyzer...")
        self.is_initialized = True
        self.logger.info("✅ VectorizedLiquidityAnalyzer initialized successfully")
        return True
        except Exception as e:
        self.logger.exception(
                f"❌ Failed to initialize VectorizedLiquidityAnalyzer: {e}",
            )
        return False

    async def analyze_liquidity_vectorized(
        self = price_data: pd.DataFrame, volume_data: pd.DataFrame, ) -> dict[str, Any]:
        """Generate liquidity features using vectorized operations."""
        try:
            features = {}

            close = price_data["close"].astype(float)
            volume = volume_data["volume"].astype(float)
            high = price_data["high"].astype(float)
            low = price_data["low"].astype(float)

        # Amihud illiquidity measure - IMPROVED: Better handling of edge cases
            returns, close.pct_change().abs()
        # Use a minimum volume threshold to prevent division by very small numbers
            min_volume_threshold, volume.quantile(0.01) * 0.1  # 10% of 1st percentile
            volume_safe, volume.replace(0, min_volume_threshold)
            amihud = returns / volume_safe
            features["amihud_illiquidity"] = amihud.fillna(0)

        # Roll's effective spread proxy - OPTIMIZED
            price_range = (high - low) / close
            features["roll_spread_proxy"] = price_range

        # Additional liquidity features
        # VWAP deviation
            vwap = (close * volume).rolling(20).sum() / volume.rolling(20).sum()
            features["vwap_deviation"] = (close - vwap) / vwap

        # Liquidity ratio (volume / price volatility)
            price_volatility = close.rolling(20).std()
            features["liquidity_ratio"] = volume / (price_volatility + 1e-8)

        # Volume Z-score
            volume_mean = volume.rolling(20).mean()
            volume_std = volume.rolling(20).std()
            features["volume_zscore"] = (volume - volume_mean) / (volume_std + 1e-8)

        # Bid-ask spread approximation using high-low range
            features["spread_approximation"] = (high - low) / ((high + low) / 2)

        # Liquidity pressure (volume * price change)
            features["liquidity_pressure"] = volume * close.pct_change().abs()

        # Volume-weighted average price (VWAP) - OPTIMIZED
            typical_price = (high + low + close) / 3
            vwap = (typical_price * volume).rolling(20).sum() / volume.rolling(20).sum()
            features["vwap_deviation"] = (close - vwap) / vwap

        # Liquidity ratio - IMPROVED: Better handling of edge cases
        # Use a minimum price range to prevent division by zero
            min_price_range, price_range.quantile(0.01) * 0.1  # 10% of 1st percentile
            price_range_safe, price_range.replace(0, min_price_range)
            liquidity_ratio = volume / price_range_safe
            features["liquidity_ratio"] = liquidity_ratio.fillna(0)

        # Volume profile - IMPROVED: Better handling of zero standard deviation
            volume_ma = volume.rolling(20).mean()
            volume_std = volume.rolling(20).std()
        # Use a minimum standard deviation to prevent division by zero
            min_std_threshold, volume_std.quantile(0.01) * 0.1  # 10% of 1st percentile
            volume_std_safe, volume_std.replace(0, min_std_threshold)
            features["volume_zscore"] = (volume - volume_ma) / volume_std_safe

        return features

        except Exception as e:
        self.logger.exception(f"❌ Error in liquidity analysis: {e}")
        return {}


class VectorizedCandlestickPatternAnalyzer:
    """Vectorized candlestick pattern analysis."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.logger = system_logger.getChild("VectorizedCandlestickPatternAnalyzer")
        self.is_initialized = False

    async def initialize(self) -> bool:
        """Initialize the candlestick pattern analyzer."""
        try:
        self.logger.info("🚀 Initializing VectorizedCandlestickPatternAnalyzer...")
        self.logger.info(f"🔍 Config keys: {list(self.config.keys()) if self.config else 'None'}")
        self.is_initialized = True
        self.logger.info(
                "✅ VectorizedCandlestickPatternAnalyzer initialized successfully",
            )
        return True
        except Exception as e:
        self.logger.exception(
                f"❌ Failed to initialize VectorizedCandlestickPatternAnalyzer: {e}",
            )
        self.logger.exception(f"❌ Exception type: {type(e)}")
        self.logger.exception(f"❌ Exception traceback: {e.__traceback__}")
        return False

    @validate_feature_engineering_with_lookahead_bias_detection
    async def analyze_patterns(self, price_data: pd.DataFrame) -> dict[str, Any]:
        """Generate candlestick pattern features using vectorized operations."""
        try:
            features = {}

            open_price = price_data["open"].astype(float)
            high = price_data["high"].astype(float)
            low = price_data["low"].astype(float)
            close = price_data["close"].astype(float)

        # Basic candlestick properties
            body_size = (close - open_price).abs()
            upper_shadow, high - np.maximum(open_price, close)
            lower_shadow, np.minimum(open_price, close) - low
            total_range = high - low

        # Doji pattern
            doji_threshold = total_range * 0.1
            is_doji = (body_size <= doji_threshold).astype(int)
            features["doji_pattern"] = is_doji

        # Hammer pattern
            is_hammer = (
                (lower_shadow > body_size * 2) & (upper_shadow < body_size * 0.5)
            ).astype(int)
            features["hammer_pattern"] = is_hammer

        # Shooting star pattern
            is_shooting_star = (
                (upper_shadow > body_size * 2) & (lower_shadow < body_size * 0.5)
            ).astype(int)
            features["shooting_star_pattern"] = is_shooting_star

        # Engulfing patterns
            prev_open = open_price.shift(1)
            prev_close = close.shift(1)

            bullish_engulfing = (
                (close > prev_open)
                & (open_price < prev_close)
                & (body_size > (prev_close - prev_open).abs())
            ).astype(int)
            features["bullish_engulfing"] = bullish_engulfing

            bearish_engulfing = (
                (close < prev_open)
                & (open_price > prev_close)
                & (body_size > (prev_close - prev_open).abs())
            ).astype(int)
            features["bearish_engulfing"] = bearish_engulfing

        # Body to range ratio
            body_range_ratio, body_size / total_range.replace(0, np.nan)
            features["body_range_ratio"] = body_range_ratio.fillna(0)

        return features

        except Exception as e:
        self.logger.exception(f"❌ Error in candlestick pattern analysis: {e}")
        return {}


class VectorizedSRDistanceCalculator:
    """Vectorized support/resistance distance calculator."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.logger = system_logger.getChild("VectorizedSRDistanceCalculator")
        self.is_initialized = False

    async def initialize(self) -> bool:
        """Initialize the S/R distance calculator."""
        try:
        self.logger.info("🚀 Initializing VectorizedSRDistanceCalculator...")
        self.is_initialized = True
        self.logger.info(
                "✅ VectorizedSRDistanceCalculator initialized successfully",
            )
        return True
        except Exception as e:
        self.logger.exception(
                f"❌ Failed to initialize VectorizedSRDistanceCalculator: {e}",
            )
        return False

    @validate_klines_data_quality
    async def calculate_sr_distances(
        self = price_data: pd.DataFrame, sr_levels: dict[str, Any] | None, ) -> dict[str, Any]:
        """Calculate distances to support/resistance levels."""
        try:
            features = {}

            close = price_data["close"].astype(float)

        if sr_levels is None or not isinstance(sr_levels, dict):
        return features

        # Calculate distances to nearest levels
        for level_type in ["support", "resistance"]:
        if level_type in sr_levels:
                    level_prices = sr_levels[level_type]

        # Convert to numeric if it's a list or array
        if isinstance(level_prices, list | np.ndarray):
                        level_prices = np.array(level_prices).astype(float)
                    else:
                        level_prices = np.array([float(level_prices)])

        # Find nearest level for each price
                    distances = []
        for price in close:
        if not pd.isna(price):
                            level_distances = abs(level_prices - price)
                            min_distance = level_distances.min()
                            distances.append(min_distance)
                        else:
                            distances.append(np.nan)

                    distance_series, pd.Series(distances, index=close.index)
                    features[f"distance_to_{level_type}"] = distance_series.fillna(0)

        # Normalized distance
                    price_range = close.rolling(20).max() - close.rolling(20).min()
                    normalized_distance, distance_series / price_range.replace(
                        0, np.nan,
                    )
                    features[f"normalized_distance_to_{level_type}"] = (
                        normalized_distance.fillna(0)
                    )

        return features

        except Exception as e:
        self.logger.exception(f"❌ Error in S/R distance calculation: {e}")
        return {}


class VectorizedWaveletTransformAnalyzer:
    """Vectorized wavelet transform analyzer."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.logger = system_logger.getChild("VectorizedWaveletTransformAnalyzer")
        self.is_initialized = False
        self.wavelet_config = config.get("wavelet", {})

    async def initialize(self) -> bool:
        """Initialize the wavelet transform analyzer."""
        try:
        self.logger.info("🚀 Initializing VectorizedWaveletTransformAnalyzer...")
        self.is_initialized = True
        self.logger.info(
                "✅ VectorizedWaveletTransformAnalyzer initialized successfully",
            )
        return True
        except Exception as e:
        self.logger.exception(
                f"❌ Failed to initialize VectorizedWaveletTransformAnalyzer: {e}",
            )
        return False

    async def analyze_wavelet_transforms(
        self, price_data: pd.DataFrame, wavelet_type: str = "db4"
    ) -> dict[str, Any]:
        """Generate wavelet transform features with improved safety and performance."""
        try:
            features = {}

        # Validate input data
        if price_data.empty or "close" not in price_data.columns:
        self.logger.warning(
                    "⚠️ Invalid price data for wavelet analysis, returning empty features",
                )
        return {}

            close = price_data["close"].astype(float)

        # Check for valid data
        if close.isna().all() or len(close) < 32:
        self.logger.warning(
                    "⚠️ Insufficient data for wavelet analysis, returning empty features",
                )
        return {}

        # Calculate returns safely
            returns, close.pct_change().fillna(0)

        # Simple, safe wavelet-like features using vectorized operations
        # Avoid complex rolling operations that can cause segmentation faults

        # 1. Simple rolling statistics (safe)
        for window in [8, 16, 32]:
        if len(returns) >= window:
        # Rolling mean (safe)
                    rolling_mean, returns.rolling(window=window, min_periods=1).mean()
                    features[f"wavelet_mean_{window}"] = rolling_mean.fillna(0)

        # Rolling std (safe)
                    rolling_std, returns.rolling(window=window, min_periods=1).std()
                    features[f"wavelet_std_{window}"] = rolling_std.fillna(0)

        # Rolling sum of squares (energy approximation) - IMPROVED: Better normalization
        # Use normalized returns to prevent constant energy values
                    returns_normalized, returns / (returns.rolling(window=window, min_periods=1).std() + 1e-8)
                    rolling_energy = (
                        (returns_normalized**2).rolling(window=window, min_periods=1).sum()
                    )
                    features[f"wavelet_energy_{window}"] = rolling_energy.fillna(0)

        # 2. Simple frequency domain features (safe)
        if len(returns) >= 16:
        # High-frequency component (short-term)
                high_freq, returns.rolling(window=4, min_periods=1).std()
                features["wavelet_high_freq"] = high_freq.fillna(0)

        # Low-frequency component (long-term)
                low_freq, returns.rolling(window=16, min_periods=1).mean()
                features["wavelet_low_freq"] = low_freq.fillna(0)

        # Frequency ratio
                freq_ratio, high_freq / (
                    low_freq.abs() + 1e-8
                )  # Avoid division by zero
                features["wavelet_freq_ratio"] = freq_ratio.fillna(0)

        # 3. Simple volatility features (safe)
        if len(returns) >= 8:
        # Wavelet-like volatility using exponential weighting
                exp_weights, np.exp(-np.arange(8) / 4)  # Exponential decay
                exp_weights, exp_weights / exp_weights.sum()  # Normalize

        # Apply exponential weighting safely
                wavelet_vol, returns.rolling(window=8, min_periods=1).apply(
                    lambda x: np.sqrt(np.sum((x * exp_weights[: len(x)]) ** 2)),
                    raw=True,  # Use raw=True for better performance
                )
                features["wavelet_volatility"] = wavelet_vol.fillna(0)

        # 4. Simple trend features (safe)
        if len(returns) >= 16:
        # Trend strength using linear regression approximation
                trend_strength, returns.rolling(window=16, min_periods=1).apply(
                    lambda x: np.corrcoef(x, np.arange(len(x)))[0, 1]
        if len(x) > 1
                    else 0,
                    raw=True
                )
                features["wavelet_trend_strength"] = trend_strength.fillna(0)

        # 5. Simple momentum features (safe)
        if len(returns) >= 8:
        # Momentum using simple differences
                momentum_8, returns.rolling(window=8, min_periods=1).sum()
                features["wavelet_momentum_8"] = momentum_8.fillna(0)

                momentum_16, returns.rolling(window=16, min_periods=1).sum()
                features["wavelet_momentum_16"] = momentum_16.fillna(0)

        # Clean up any remaining NaN or infinite values
        for key, feature in features.items():
        if isinstance(feature, pd.Series):
                    features[key] = feature.replace([np.inf, -np.inf], 0).fillna(0)

        # Remove truly constant features (zero variance)
            features = self._remove_constant_features(features)

        self.logger.info(f"✅ Generated {len(features)} safe wavelet features")
        return features

        except Exception as e:
        self.logger.exception(f"❌ Error in wavelet transform analysis: {e}")
        # Return empty features instead of crashing
        return {}

    def _remove_constant_features(self, features: dict[str, Any]) -> dict[str, Any]:
        """Remove features with zero or near-zero variance."""
        try:
            non_constant_features = {}
            constant_features = []
            variance_threshold = 1e-12  # Very small threshold for true constants

        for key, value in features.items():
        if isinstance(value, pd.Series):
        # Check if feature has meaningful variance
                    feature_variance = value.var()
        if feature_variance > variance_threshold:
                        non_constant_features[key] = value
                    else:
                        constant_features.append(key)
                else:
        # Keep non-series features
                    non_constant_features[key] = value

        if constant_features:
        self.logger.info(f"🗑️ Removed {len(constant_features)} constant features: {constant_features[:5]}... (showing first 5)")

        return non_constant_features

        except Exception as e:
        self.logger.exception(f"❌ Error removing constant features: {e}")
        return features


class VectorizedAdvancedFeatureEngineering:
    """Comprehensive vectorized advanced feature engineering system.
    Integrates all feature engineering components including wavelet transforms.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.logger = system_logger.getChild("VectorizedAdvancedFeatureEngineering")

        # Configuration
        self.feature_config = config.get("vectorized_advanced_features", {})
        self.enable_volatility_modeling = self.feature_config.get(
            "enable_volatility_modeling", True,
        )
        self.enable_correlation_analysis = self.feature_config.get(
            "enable_correlation_analysis", True,
        )
        self.enable_momentum_analysis = self.feature_config.get(
            "enable_momentum_analysis", True,
        )
        self.enable_liquidity_analysis = self.feature_config.get(
            "enable_liquidity_analysis", True,
        )
        self.enable_candlestick_patterns = self.feature_config.get(
            "enable_candlestick_patterns", True,
        )
        self.enable_sr_distance = self.feature_config.get("enable_sr_distance", True)
        self.enable_wavelet_transforms = self.feature_config.get(
            "enable_wavelet_transforms", True,
        )  # Re-enabled
        self.enable_multi_timeframe = self.feature_config.get(
            "enable_multi_timeframe", True,
        )
        # Meta-labeling deprecated: force disabled
        self.enable_meta_labeling = False
        # Explicit analyst meta-labels deprecated: force disabled
        self.enable_explicit_meta_labels = False

        # Difference and acceleration features (enabled by default)
        self.enable_difference_acceleration_features = self.feature_config.get(
            "enable_difference_acceleration_features", True,
        )

        # Multi-timeframe configuration
        self.timeframes = ["1m", "5m", "15m", "30m"]

        # CWT configuration
        self.cwt_method_preference = self.feature_config.get(
            "cwt_method_preference", "auto",
        )
        self.cwt_fft_threshold = self.feature_config.get("cwt_fft_threshold", 1000)

        # Initialize components
        self.volatility_model = None
        self.correlation_analyzer = None
        self.momentum_analyzer = None
        self.liquidity_analyzer = None
        self.candlestick_analyzer = None
        self.sr_distance_calculator = None
        self.wavelet_analyzer = None
        self.wavelet_cache = None

        # Initialize optimized resampler
        self.optimized_resampler = OptimizedResampler()

        # Configure joblib memory to prevent cache flushing warnings
        try:
            import joblib
            memory_location, FEATURE_OPTIMIZATION_CONFIG.get("joblib_memory_location", "data/joblib_cache")
            memory_verbose, FEATURE_OPTIMIZATION_CONFIG.get("joblib_memory_verbose", 0)
            memory_bytes, FEATURE_OPTIMIZATION_CONFIG.get("joblib_memory_bytes", 1024 * 1024 * 1024)
            memory_compress, FEATURE_OPTIMIZATION_CONFIG.get("joblib_memory_compress", 3)

        # Create memory directory if it doesn't exist
            os.makedirs(memory_location, exist_ok=True)

        # Configure joblib memory
            joblib.memory.Memory.location, memory_location
            joblib.memory.Memory.verbose, memory_verbose
            joblib.memory.Memory.bytes, memory_bytes
            joblib.memory.Memory.compress, memory_compress

        self.logger.info(f"✅ Configured joblib memory cache: {memory_location}")
        except Exception as e:
        self.logger.warning(f"⚠️ Failed to configure joblib memory: {e}")

        # Configuration for problematic features
        self.disable_problematic_wavelets = self.feature_config.get(
            "disable_problematic_wavelets", True,
        )
        self.wavelet_features_to_skip = {
            "volume_wavelet_approx_ts",
            "volume_wavelet_detail_ts",
            "wavelet_packet_approx_ts",
            "wavelet_packet_detail_ts",
            "wavelet_denoised_signal_ts",
            "wavelet_denoised_residual_ts",
            "multi_wavelet_db1_approx_ts",
            "multi_wavelet_db2_approx_ts",
            "multi_wavelet_db4_approx_ts",
            "multi_wavelet_haar_approx_ts",
            "multi_wavelet_sym4_approx_ts",
            "gaus1_detrended_energy_ts",
            "mexh_detrended_energy_ts",
            "gaus1_price_diff_energy_ts",
            "gaus1_close_energy_ts",
            "mexh_price_diff_energy_ts",
            "mexh_close_energy_ts",
            "gaus1_price_diff_2_energy_ts",
            "mexh_price_diff_2_energy_ts",
        }

        self.is_initialized = False

    @handle_errors(
        exceptions=(Exception,)
        default_return=False
        context="vectorized advanced feature engineering initialization"
    )
    async def initialize(self) -> bool:
        """Initialize vectorized advanced feature engineering components."""
        try:
        self.logger.info(
                "🚀 Initializing vectorized advanced feature engineering...",
            )

        # Initialize wavelet cache
        if self.enable_wavelet_transforms:
        self.wavelet_cache = WaveletFeatureCache(self.config)

        # Initialize volatility modeling
        if self.enable_volatility_modeling:
        self.volatility_model = VectorizedVolatilityRegimeModel(self.config)
        await self.volatility_model.initialize()

        # Initialize correlation analysis
        if self.enable_correlation_analysis:
        self.correlation_analyzer = VectorizedCorrelationAnalyzer(self.config)
        await self.correlation_analyzer.initialize()

        # Initialize momentum analysis
        if self.enable_momentum_analysis:
        self.momentum_analyzer = VectorizedMomentumAnalyzer(self.config)
        await self.momentum_analyzer.initialize()

        # Initialize liquidity analysis
        if self.enable_liquidity_analysis:
        try:
        self.logger.info("🔍 Creating VectorizedLiquidityAnalyzer...")
        self.liquidity_analyzer = VectorizedLiquidityAnalyzer(self.config)
        self.logger.info("🔍 VectorizedLiquidityAnalyzer created, initializing...")
                    init_success = await self.liquidity_analyzer.initialize()
        if not init_success:
        self.logger.warning("⚠️ Liquidity analyzer initialization failed, setting to None")
        self.liquidity_analyzer = None
                    else:
        self.logger.info("✅ Liquidity analyzer initialized successfully")
        except Exception as e:
        self.logger.exception(f"🚨 Error creating liquidity analyzer: {e}")
        self.logger.exception(f"🚨 Exception type: {type(e)}")
        self.liquidity_analyzer = None
            else:
        self.logger.info("ℹ️ Liquidity analysis disabled in config")

        # Initialize candlestick pattern analyzer
        if self.enable_candlestick_patterns:
        try:
        self.logger.info("🔍 Creating VectorizedCandlestickPatternAnalyzer...")
        self.candlestick_analyzer = VectorizedCandlestickPatternAnalyzer(
        self.config,
                    )
        self.logger.info("🔍 VectorizedCandlestickPatternAnalyzer created, initializing...")
                    init_success = await self.candlestick_analyzer.initialize()
        if not init_success:
        self.logger.warning("⚠️ Candlestick analyzer initialization failed, setting to None")
        self.candlestick_analyzer = None
                    else:
        self.logger.info("✅ Candlestick analyzer initialized successfully")
        except Exception as e:
        self.logger.exception(f"🚨 Error creating candlestick analyzer: {e}")
        self.logger.exception(f"🚨 Exception type: {type(e)}")
        self.candlestick_analyzer = None
            else:
        self.logger.info("ℹ️ Candlestick patterns disabled in config")

        # Initialize S/R distance calculator
        if self.enable_sr_distance:
        try:
        self.logger.info("🔍 Creating VectorizedSRDistanceCalculator...")
        self.sr_distance_calculator = VectorizedSRDistanceCalculator(
        self.config,
                    )
        self.logger.info("🔍 VectorizedSRDistanceCalculator created, initializing...")
                    init_success = await self.sr_distance_calculator.initialize()
        if not init_success:
        self.logger.warning("⚠️ S/R distance calculator initialization failed, setting to None")
        self.sr_distance_calculator = None
                    else:
        self.logger.info("✅ S/R distance calculator initialized successfully")
        except Exception as e:
        self.logger.exception(f"🚨 Error creating S/R distance calculator: {e}")
        self.logger.exception(f"🚨 Exception type: {type(e)}")
        self.sr_distance_calculator = None
            else:
        self.logger.info("ℹ️ S/R distance disabled in config")

        # Initialize wavelet transform analyzer
        if self.enable_wavelet_transforms:
        try:
        self.logger.info("🔍 Creating VectorizedWaveletTransformAnalyzer...")
        self.wavelet_analyzer = VectorizedWaveletTransformAnalyzer(self.config)
        self.logger.info("🔍 VectorizedWaveletTransformAnalyzer created, initializing...")
                    init_success = await self.wavelet_analyzer.initialize()
        if not init_success:
        self.logger.warning("⚠️ Wavelet analyzer initialization failed, setting to None")
        self.wavelet_analyzer = None
                    else:
        self.logger.info("✅ Wavelet analyzer initialized successfully")
        except Exception as e:
        self.logger.exception(f"🚨 Error creating wavelet analyzer: {e}")
        self.logger.exception(f"🚨 Exception type: {type(e)}")
        self.wavelet_analyzer = None
            else:
        self.logger.info("ℹ️ Wavelet transforms disabled in config")

        # Meta-labeling system removed - using only HMM market regimes
        self.logger.info(
                "ℹ️ Meta-labeling system removed - using only HMM market regimes for labeling",
            )

        self.is_initialized = True
        self.logger.info(
                "✅ Vectorized advanced feature engineering initialized successfully",
            )
        return True

        except Exception as e:
        self.logger.exception(
                f"🚨 Error initializing vectorized advanced feature engineering: {e}",
            )
        return False

    def _calculate_price_impact_vectorized(
        self = price_data: pd.DataFrame, volume_data: pd.DataFrame, ) -> pd.Series:
        """Calculate price impact using vectorized operations with improved NaN handling."""
        try:
        if "close" not in price_data.columns or "volume" not in volume_data.columns:
        return pd.Series(0, index=price_data.index)

            close = price_data["close"].astype(float)
            volume = volume_data["volume"].astype(float)

        # Handle NaN values in input data
            close, close.fillna(method="ffill").fillna(method="bfill")
            volume, volume.fillna(method="ffill").fillna(method="bfill")

        # Ensure we have valid data
        if close.isna().all() or volume.isna().all():
        return pd.Series(0, index=price_data.index)

        # Calculate price impact as the ratio of price change to volume
        # Use shift(1) to avoid NaN in first row, then calculate difference
            price_change = (close - close.shift(1)).abs()

        # Calculate volume normalization with better handling
            volume_ma, volume.rolling(20, min_periods=5).mean()
            volume_normalized = volume / volume_ma

        # Avoid division by zero and handle edge cases
            volume_normalized, volume_normalized.replace([np.inf, -np.inf], np.nan)
            volume_normalized, volume_normalized.fillna(1)  # Use 1 as default for missing values
            volume_normalized, volume_normalized.replace(0, 1)  # Avoid division by zero

        # Price impact, price change / normalized volume
            price_impact = price_change / volume_normalized

        # Clean up infinite and NaN values with better strategy
            price_impact, price_impact.replace([np.inf, -np.inf], np.nan)

        # For price impact, use 0 for the first row (no previous price) and forward fill for other NaN values
        return price_impact.fillna(0)


        except Exception as e:
        self.logger.exception(f"🚨 Error calculating price impact: {e}")
        return pd.Series(0, index=price_data.index)

    def _calculate_volume_price_impact_vectorized(
        self = price_data: pd.DataFrame, volume_data: pd.DataFrame, ) -> pd.Series:
        """Calculate volume-price impact using vectorized operations with improved NaN handling."""
        try:
        if "close" not in price_data.columns or "volume" not in volume_data.columns:
        return pd.Series(0, index=price_data.index)

            close = price_data["close"].astype(float)
            volume = volume_data["volume"].astype(float)

        # Handle NaN values in input data
            close, close.fillna(method="ffill").fillna(method="bfill")
            volume, volume.fillna(method="ffill").fillna(method="bfill")

        # Ensure we have valid data
        if close.isna().all() or volume.isna().all():
        return pd.Series(0, index=price_data.index)

        # Calculate volume-price impact as volume-weighted price change
        # Use shift(1) to avoid NaN in first row, then calculate difference
            price_change = close - close.shift(1)

        # Calculate volume normalization with better handling
            volume_ma, volume.rolling(20, min_periods=5).mean()
            volume_normalized = volume / volume_ma

        # Avoid division by zero and handle edge cases
            volume_normalized, volume_normalized.replace([np.inf, -np.inf], np.nan)
            volume_normalized, volume_normalized.fillna(1)  # Use 1 as default for missing values
            volume_normalized, volume_normalized.replace(0, 1)  # Avoid division by zero

        # Volume-price impact, price change * normalized volume
            volume_price_impact = price_change * volume_normalized

        # Clean up infinite and NaN values with better strategy
            volume_price_impact, volume_price_impact.replace([np.inf, -np.inf], np.nan)

        # For volume-price impact, use 0 for the first row (no previous price) and forward fill for other NaN values
        return volume_price_impact.fillna(0)


        except Exception as e:
        self.logger.exception(f"🚨 Error calculating volume-price impact: {e}")
        return pd.Series(0, index=price_data.index)

    def _calculate_order_flow_imbalance_vectorized(
        self, price_data: pd.DataFrame, volume_data: pd.DataFrame, order_flow_data: pd.DataFrame | None = None
    ) -> pd.Series:
        """Calculate order flow imbalance using vectorized operations with improved NaN handling."""
        try:
        if "close" not in price_data.columns or "volume" not in volume_data.columns:
        return pd.Series(0, index=price_data.index)

            close = price_data["close"].astype(float)
            volume = volume_data["volume"].astype(float)

        # Handle NaN values in input data
            close, close.fillna(method="ffill").fillna(method="bfill")
            volume, volume.fillna(method="ffill").fillna(method="bfill")

        # Ensure we have valid data
        if close.isna().all() or volume.isna().all():
        return pd.Series(0, index=price_data.index)

        # Calculate order flow imbalance as volume-weighted price direction
        # Use shift(1) to avoid NaN in first row, then calculate difference
            price_diff = close - close.shift(1)
        # Handle zero price changes by using a small threshold
            price_direction, np.where(price_diff > 0, 1, np.where(price_diff < 0, -1, 0))

        # Calculate volume normalization with better handling
            volume_ma, volume.rolling(20, min_periods=5).mean()
            volume_normalized = volume / volume_ma

        # Avoid division by zero and handle edge cases
            volume_normalized, volume_normalized.replace([np.inf, -np.inf], np.nan)
            volume_normalized, volume_normalized.fillna(1)  # Use 1 as default for missing values
            volume_normalized, volume_normalized.replace(0, 1)  # Avoid division by zero

        # Order flow imbalance, price direction * normalized volume
            order_flow_imbalance = price_direction * volume_normalized

        # Clean up infinite and NaN values with better strategy
            order_flow_imbalance, order_flow_imbalance.replace([np.inf, -np.inf], np.nan)

        # For order flow imbalance, use 0 for the first row (no previous price) and forward fill for other NaN values
        return order_flow_imbalance.fillna(0)


        except Exception as e:
        self.logger.exception(f"🚨 Error calculating order flow imbalance: {e}")
        return pd.Series(0, index=price_data.index)

    def _validate_and_transform_data(
        self = price_data: pd.DataFrame, volume_data: pd.DataFrame, ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Validate and transform input data to ensure proper structure."""
        try:
        # Debug: Log input data structure
        self.logger.info(f"🔍 Input price_data columns: {list(price_data.columns)}")
        self.logger.info(f"🔍 Input price_data shape: {price_data.shape}")
        self.logger.info(
                f"🔍 Input volume_data columns: {list(volume_data.columns)}",
            )
        self.logger.info(f"🔍 Input volume_data shape: {volume_data.shape}")

        # Ensure we have a DatetimeIndex
        if not isinstance(price_data.index, pd.DatetimeIndex):
        if "timestamp" in price_data.columns:
                    price_data = price_data.copy()
                    price_data["timestamp"] = pd.to_datetime(
                        price_data["timestamp"], errors="coerce"
                    )
                    price_data, price_data.dropna(subset=["timestamp"]).set_index(
                        "timestamp",
                    )
                else:
                    price_data = price_data.copy()
                    price_data.index, pd.to_datetime(price_data.index, errors="coerce")

        # Ensure volume_data has same index
        if not isinstance(volume_data.index, pd.DatetimeIndex):
                volume_data = volume_data.copy()
                volume_data.index, pd.to_datetime(volume_data.index, errors="coerce")

        # Align indices
            common_index = price_data.index.intersection(volume_data.index)
        if len(common_index) == 0:
        self.logger.error("❌ No common index found between price_data and volume_data")
        return price_data, volume_data

            price_data = price_data.loc[common_index]
            volume_data = volume_data.loc[common_index]

        # Debug: Log output data structure
        self.logger.info(
                f"🔍 Output price_data columns: {list(price_data.columns)}",
            )
        self.logger.info(f"🔍 Output price_data shape: {price_data.shape}")
        self.logger.info(
                f"🔍 Output volume_data columns: {list(volume_data.columns)}",
            )
        self.logger.info(f"🔍 Output volume_data shape: {volume_data.shape}")

        return price_data, volume_data

        except Exception as e:
        self.logger.exception(f"🚨 Error validating and transforming data: {e}")
        return price_data, volume_data

    def _handle_nan_values_basic(self, features: dict[str, Any]) -> dict[str, Any]:
        """Basic NaN handling for features when comprehensive method is not available."""
        try:
            cleaned_features = {}
        for feature_name, feature_value in features.items():
        try: if isinstance(feature_value = int | float | np.integer | np.floating):
        # Scalar values - handle safely
        if np.isnan(feature_value) or np.isinf(feature_value):
                            cleaned_features[feature_name] = 0.0
                        else:
                            cleaned_features[feature_name] = feature_value

                    elif isinstance(feature_value, pd.Series):
        # Pandas Series
                        cleaned_series = feature_value.copy()
                        cleaned_series, cleaned_series.fillna(0).replace(
                            [np.inf, -np.inf], 0,
                        )
                        cleaned_features[feature_name] = cleaned_series

                    elif isinstance(feature_value, np.ndarray | list):
        # Numpy arrays and lists
                        arr, np.asarray(feature_value, dtype=np.float64)
                        arr, np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
                        cleaned_features[feature_name] = arr

                    else:
        # Unsupported type - convert to 0
                        cleaned_features[feature_name] = 0.0

        except Exception as e:
        self.logger.warning(f"Error cleaning feature {feature_name}: {e}")
                    cleaned_features[feature_name] = 0.0

        return cleaned_features

        except Exception as e:
        self.logger.exception(f"🚨 Error in basic NaN handling: {e}")
        return features

    def _get_minimum_data_requirement(self, timeframe: str) -> int:
        """Get minimum data requirement for a given timeframe."""
        # Define minimum data requirements based on timeframe
        # Higher timeframes need more data to generate meaningful features
        requirements = {
            "1m": 50,    # 1-minute needs at least 50 data points
            "5m": 30,    # 5-minute needs at least 30 data points (2.5 hours)
            "15m": 20,   # 15-minute needs at least 20 data points (5 hours)
            "30m": 15,   # 30-minute needs at least 15 data points (7.5 hours)
            "1h": 10,    # 1-hour needs at least 10 data points (10 hours)
            "4h": 5,     # 4-hour needs at least 5 data points (20 hours)
            "1d": 3,     # 1-day needs at least 3 data points (3 days)
        }

        return requirements.get(timeframe, 50)  # Default to 50 if timeframe not found

    def _log_multi_timeframe_summary(self, features: dict[str, Any], timeframes: list[str]) -> None:
        """Log a summary of multi-timeframe feature generation."""
        try:
        # Count features by timeframe
            timeframe_counts = {}
        for tf in timeframes:
                tf_features = [f for f in features if f.endswith(f"_{tf}")]
                timeframe_counts[tf] = len(tf_features)

        # Log summary
        self.logger.info("📊 Multi-timeframe feature generation summary:")
        for tf in timeframes:
                count, timeframe_counts.get(tf, 0)
        if count > 0:
        self.logger.info(f"  ✅ {tf}: {count} features generated")
                else:
        self.logger.info(f"  ⏭️ {tf}: skipped (insufficient data)")

            total_features = len(features)
        self.logger.info(f"📈 Total multi-timeframe features: {total_features}")

        except Exception as e:
        self.logger.warning(f"⚠️ Error logging multi-timeframe summary: {e}")

    def _generate_simple_timeframe_features(
        self = price_data: pd.DataFrame, volume_data: pd.DataFrame, timeframe: str, ) -> dict[str, Any]:
        """Generate simple features for timeframes with limited data."""
        try:
            features = {}

        if price_data.empty or len(price_data) < 3:  # Very low minimum requirement
        self.logger.warning(f"⚠️ Insufficient data for simple {timeframe} features: {len(price_data)} rows")
        return features

        # Basic price features
        if "close" in price_data.columns:
                close = price_data["close"].astype(float)
                close, close.fillna(method="ffill").fillna(method="bfill").fillna(0)

        # Very simple features that work with minimal data
                features[f"simple_close_returns_{timeframe}"] = close.pct_change().fillna(0)
                features[f"simple_close_momentum_{timeframe}"] = close / close.shift(1) - 1

        # Simple moving average with very low min_periods
        if len(close) >= 2:
                    features[f"simple_close_ma_{timeframe}"] = close.rolling(2, min_periods=1).mean().fillna(0)

        # Simple volatility
                returns, close.pct_change().fillna(0)
        if len(returns) >= 2:
                    features[f"simple_volatility_{timeframe}"] = returns.rolling(2, min_periods=1).std().fillna(0)

        # Basic volume features
        if volume_data is not None and not volume_data.empty and "volume" in volume_data.columns:
                volume = volume_data["volume"].astype(float)
                volume, volume.fillna(method="ffill").fillna(method="bfill").fillna(0)

        if len(volume) >= 2:
                    features[f"simple_volume_ma_{timeframe}"] = volume.rolling(2, min_periods=1).mean().fillna(0)
                    features[f"simple_volume_ratio_{timeframe}"] = volume / (volume.rolling(2, min_periods=1).mean() + 1e-8)

        # OHLCV features if available
        if all(col in price_data.columns for col in ["open", "high", "low", "close"]):
                high, price_data["high"].astype(float).fillna(method="ffill").fillna(method="bfill").fillna(0)
                low, price_data["low"].astype(float).fillna(method="ffill").fillna(method="bfill").fillna(0)
                open_price, price_data["open"].astype(float).fillna(method="ffill").fillna(method="bfill").fillna(0)
                close, price_data["close"].astype(float).fillna(method="ffill").fillna(method="bfill").fillna(0)

                features[f"simple_high_low_ratio_{timeframe}"] = high / (low + 1e-8)
                features[f"simple_close_open_ratio_{timeframe}"] = close / (open_price + 1e-8)
                features[f"simple_body_size_{timeframe}"] = abs(close - open_price) / ((high - low) + 1e-8)

        # Fill any remaining NaN values
        for key in features:
        if isinstance(features[key], pd.Series):
                    features[key] = features[key].fillna(method="ffill").fillna(method="bfill").fillna(0)

        self.logger.debug(f"✅ Generated {len(features)} simple features for {timeframe} timeframe")
        return features

        except Exception as e:
        self.logger.exception(f"Error generating simple timeframe features for {timeframe}: {e}")
        return {}

    def _handle_nan_values_comprehensive(self, features: dict[str, Any]) -> dict[str, Any]:
        """Comprehensive NaN handling for all feature types."""
        try:
            cleaned_features = {}
            nan_count = 0
            inf_count = 0

        for feature_name, feature_value in features.items():
        try:
        # Handle different data types
        if isinstance(feature_value, int | float | np.integer | np.floating):
        # Scalar values
        if np.isnan(feature_value) or np.isinf(feature_value):
                            cleaned_features[feature_name] = 0.0
                            nan_count += 1
                        else:
                            cleaned_features[feature_name] = feature_value

                    elif isinstance(feature_value, pd.Series):
        # Pandas Series with safe boolean operations
                        cleaned_series = feature_value.copy()

        # Handle NaN values safely
                        nan_mask = cleaned_series.isna()
        if nan_mask.sum() > 0:  # Use sum() instead of any() for safety
                            cleaned_series = cleaned_series.fillna(0)
                            nan_count += int(nan_mask.sum())

        # Handle infinite values safely
        try:
                            inf_mask = np.isinf(cleaned_series.values)
        if inf_mask.sum() > 0:  # Use sum() instead of any() for safety
                                cleaned_series, cleaned_series.replace([np.inf, -np.inf], 0)
                                inf_count += int(inf_mask.sum())
        except Exception:
        # Fallback: use pandas method
                            cleaned_series, cleaned_series.replace([np.inf, -np.inf], 0)

                        cleaned_features[feature_name] = cleaned_series

                    elif isinstance(feature_value, np.ndarray | list):
        # Numpy arrays and lists with safe boolean operations
                        arr, np.asarray(feature_value, dtype=np.float64)

        # Handle NaN values safely
                        nan_mask = np.isnan(arr)
        if nan_mask.sum() > 0:  # Use sum() instead of any() for safety
                            arr, np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
                            nan_count += int(nan_mask.sum())

        # Handle infinite values safely
                        inf_mask = np.isinf(arr)
        if inf_mask.sum() > 0:  # Use sum() instead of any() for safety
                            arr, np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
                            inf_count += int(inf_mask.sum())

                        cleaned_features[feature_name] = arr

                    else:
        # Unsupported type - skip or convert to 0
                        cleaned_features[feature_name] = 0.0

        except Exception as e:
        self.logger.warning(f"Error cleaning feature {feature_name}: {e}")
                    cleaned_features[feature_name] = 0.0

        # Log summary
        if nan_count > 0 or inf_count > 0:
        self.logger.info(
                    f"🔧 Comprehensive NaN handling: {nan_count} NaN values, {inf_count} inf values cleaned",
                )

        return cleaned_features

        except Exception as e:
        self.logger.exception(f"🚨 Error in comprehensive NaN handling: {e}")
        # Return original features if comprehensive handling fails
        return features

    def _handle_nan_values_robust(self, features: dict[str, Any]) -> dict[str, Any]:
        """Robust NaN handling that always works regardless of method availability."""
        try:
        # Filter out coroutine objects before processing
            valid_features = {}
        for key, value in features.items():
        if hasattr(value, "__await__"):
        self.logger.warning(f"⚠️ Skipping coroutine feature in NaN handling: {key}")
                    continue
                valid_features[key] = value

        # Try comprehensive method first
        try:
        return self._handle_nan_values_comprehensive(valid_features)
        except Exception as e1:
        self.logger.debug(f"Comprehensive method failed: {e1}")

        # Fallback to basic method
        try:
        return self._handle_nan_values_basic(valid_features)
        except Exception as e2:
        self.logger.debug(f"Basic method failed: {e2}")

        # Final fallback to inline method
        try:
        return self._handle_nan_values_inline(valid_features)
        except Exception as e3:
        self.logger.debug(f"Inline method failed: {e3}")

        # If all methods fail, return original features
        self.logger.error(f"🚨 All NaN handling methods failed: {e1}, {e2}, {e3}")
        return valid_features

        except Exception as e:
        self.logger.exception(f"🚨 All NaN handling methods failed: {e}")
        return features

    def _handle_nan_values_inline(self, features: dict[str, Any]) -> dict[str, Any]:
        """Inline NaN handling as final fallback."""
        try:
            cleaned_features = {}
            nan_count = 0
            inf_count = 0

        for feature_name, feature_value in features.items():
        try:
        # Handle different data types
        if isinstance(feature_value, int | float | np.integer | np.floating):
        # Scalar values - handle safely
        if np.isnan(feature_value):
                            cleaned_features[feature_name] = 0.0
                            nan_count += 1
                        elif np.isinf(feature_value):
                            cleaned_features[feature_name] = 0.0
                            inf_count += 1
                        else:
                            cleaned_features[feature_name] = feature_value

                    elif isinstance(feature_value, pd.Series):
        # Pandas Series with robust NaN handling
        try:
                            cleaned_series = feature_value.copy()

        # Handle NaN values with detailed logging
                            nan_mask = cleaned_series.isna()
                            nan_count_series = nan_mask.sum()
        if nan_count_series > 0:
        self.logger.debug(
                                    f"🔍 Feature {feature_name}: Found {nan_count_series} NaN values in Series",
                                )
                                cleaned_series = cleaned_series.fillna(0)
                                nan_count += int(nan_count_series)

        # Handle infinite values with detailed logging - Safe boolean operations
        try:
        # Convert to numpy array safely and handle infinite values
                                series_values = cleaned_series.values
        if hasattr(series_values, "dtype") and np.issubdtype(series_values.dtype, np.number):
                                    inf_mask = np.isinf(series_values)
                                    inf_count_series = int(inf_mask.sum())
        if inf_count_series > 0:
        self.logger.debug(
                                            f"🔍 Feature {feature_name}: Found {inf_count_series} inf values in Series",
                                        )
                                        cleaned_series, cleaned_series.replace(
                                            [np.inf, -np.inf], 0,
                                        )
                                        inf_count += inf_count_series
                                else:
        # Fallback for non-numeric data
                                    cleaned_series, cleaned_series.replace(
                                        [np.inf, -np.inf], 0,
                                    )
        except Exception as inf_error:
        # Fallback: use pandas method instead of numpy
        self.logger.debug(
                                    f"🔍 Feature {feature_name}: Using pandas method for inf handling due to: {inf_error}",
                                )
                                cleaned_series, cleaned_series.replace(
                                    [np.inf, -np.inf], 0,
                                )

                            cleaned_features[feature_name] = cleaned_series

        except Exception as series_error:
        self.logger.warning(
                                f"🚨 Error handling Series for {feature_name}: {series_error}",
                            )
        # Fallback: convert to numpy array and handle
        try: arr = np.asarray(feature_value, dtype=np.float64)
                                arr, np.nan_to_num(
                                    arr, nan=0.0, posinf=0.0, neginf=0.0
                                )
                                cleaned_features[feature_name] = arr
        self.logger.info(
                                    f"🔧 Converted Series {feature_name} to numpy array as fallback",
                                )
        except Exception as fallback_error:
        self.logger.exception(
                                    f"🚨 Fallback failed for {feature_name}: {fallback_error}",
                                )
                                cleaned_features[feature_name] = 0.0

                    elif isinstance(feature_value, np.ndarray | list):
        # Numpy arrays and lists
                        arr, np.asarray(feature_value, dtype=np.float64)

        # Handle NaN values safely
                        nan_mask = np.isnan(arr)
        if nan_mask.sum() > 0:  # Use sum() instead of any() for safety
                            arr, np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
                            nan_count += int(nan_mask.sum())

        # Handle infinite values safely
                        inf_mask = np.isinf(arr)
        if inf_mask.sum() > 0:  # Use sum() instead of any() for safety
                            arr, np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
                            inf_count += int(inf_mask.sum())

                        cleaned_features[feature_name] = arr

                    else:
        # Unsupported type - skip or convert to 0
        self.logger.warning(
                            f"Unsupported feature type for {feature_name}: {type(feature_value)}",
                        )
                        cleaned_features[feature_name] = 0.0

        except Exception as e:
        self.logger.warning(f"Error cleaning feature {feature_name}: {e}")
                    cleaned_features[feature_name] = 0.0

        # Log summary
        if nan_count > 0 or inf_count > 0:
        self.logger.info(
                    f"🔧 Inline NaN handling: {nan_count} NaN values, {inf_count} inf values cleaned",
                )

        return cleaned_features

        except Exception as e:
        self.logger.exception(f"🚨 Error in inline NaN handling: {e}")
        return features

    @validate_step_prerequisites(
        required_directories=["data_cache", "data/feature_cache"]
        min_memory_gb=16.0
        min_disk_gb=10.0
        required_packages=["pandas", "numpy", "pywt", "scipy"]
        data_quality_checks={
            "min_rows": 1000,
            "required_columns": ["timestamp", "open", "high", "low", "close", "volume"],
        },
        context="Vectorized Advanced Feature Engineering"
    )
    @secure_data_processing(
        backup_before=True
        integrity_checks=True
        memory_cleanup=True
        data_validation=True
    )
    @prevent_data_leakage(
        temporal_validation=True
        feature_leakage_detection=True
        cross_validation_isolation=True
        lookahead_bias_prevention=True
    )
    @resource_monitor(
        memory_threshold_gb=32.0
        cpu_threshold_percent=90.0
        disk_threshold_gb=20.0
        monitor_interval=60.0
        auto_cleanup=True
    )
    @memory_efficient(
        chunk_size=5000
        streaming_processing=True
        memory_pool=True
        cleanup_frequency=20
    )
    @debug_training_step(
        log_intermediate_results=True
        save_debug_artifacts=True
        performance_profiling=True
        error_context_preservation=True
    )
    @circuit_breaker_protection(
        failure_threshold=3
        recovery_timeout=600.0
        expected_exception=Exception
        monitor_interval=60.0
    )
    @validate_step_output(
        required_files=["data/feature_cache/*.parquet"]
        data_quality_checks={
            "min_rows": 100,
            "required_columns": ["features", "metadata"],
        },
        performance_thresholds={
            "feature_engineering_time_minutes": 120.0,
            "memory_usage_gb": 16.0,
        },
        format_validation=True
    )
    @quality_gate(
        model_performance_thresholds={
            "feature_quality": 0.8,
            "feature_completeness": 0.9,
        },
        data_quality_metrics={"completeness": 0.9, "consistency": 0.8}
        convergence_checks=True
        overfitting_detection=True
        validation_score_requirements={"feature_engineering_score": 0.8}
    )
    @handle_errors(
        exceptions=(ValueError, AttributeError)
        default_return=None
        context="vectorized advanced feature engineering"
    )
    @validate_step_prerequisites(
        required_directories=["data_cache", "data/feature_cache"]
        min_memory_gb=16.0
        min_disk_gb=10.0
        required_packages=["pandas", "numpy", "pywt", "scipy"]
        data_quality_checks={
            "min_rows": 1000,
            "required_columns": ["timestamp", "open", "high", "low", "close", "volume"],
        },
        context="Vectorized Advanced Feature Engineering"
    )
    @secure_data_processing(
        backup_before=True
        integrity_checks=True
        memory_cleanup=True
        data_validation=True
    )
    @prevent_data_leakage(
        temporal_validation=True
        feature_leakage_detection=True
        cross_validation_isolation=True
        lookahead_bias_prevention=True
    )
    @resource_monitor(
        memory_threshold_gb=32.0
        cpu_threshold_percent=90.0
        disk_threshold_gb=20.0
        monitor_interval=60.0
        auto_cleanup=True
    )
    @memory_efficient(
        chunk_size=5000
        streaming_processing=True
        memory_pool=True
        cleanup_frequency=20
    )
    @debug_training_step(
        log_intermediate_results=True
        save_debug_artifacts=True
        performance_profiling=True
        error_context_preservation=True
    )
    @circuit_breaker_protection(
        failure_threshold=3
        recovery_timeout=600.0
        expected_exception=Exception
        monitor_interval=60.0
    )
    # Temporarily disabled decorators for debugging
    # @validate_step_output(
    #     required_files=["data/feature_cache/*.parquet"],
    #     data_quality_checks={
    #         "min_rows": 100,
    #         "required_columns": ["features", "metadata"],
    #     },
    #     performance_thresholds={
    #         "feature_engineering_time_minutes": 120.0,
    #         "memory_usage_gb": 16.0,
    #     },
    #     format_validation=True,
    # )
    # @quality_gate(
    #     model_performance_thresholds={
    #         "feature_quality": 0.8,
    #         "feature_completeness": 0.9,
    #     },
    #     data_quality_metrics={"completeness": 0.9, "consistency": 0.8},
    #     convergence_checks=True,
    #     overfitting_detection=True,
    #     validation_score_requirements={"feature_engineering_score": 0.8},
    # )
    # @handle_errors(
    #     exceptions=(ValueError, AttributeError),
    #     default_return=None,
    #     context="vectorized advanced feature engineering",
    # )
    async def engineer_features(
        self, price_data: pd.DataFrame, volume_data: pd.DataFrame, order_flow_data: pd.DataFrame | None = None, sr_levels: dict[str, Any] | None, None, ) -> dict[str, Any]:
        """Engineer advanced features for improved prediction accuracy using vectorized operations.

        Args:
            price_data: OHLCV price data
            volume_data: Volume and trade flow data
            order_flow_data: Order book and flow data (optional)
            sr_levels: Support/resistance levels (optional)

        Returns:
            Dictionary containing engineered features

        """
        try:
        if not self.is_initialized:
        self.logger.error(
                    "🚨 Vectorized advanced feature engineering not initialized",
                )
        return {}

        # Data quality validation is now handled by decorators
        self.logger.info(
                "🔍 Data quality validation will be performed automatically by decorators",
            )

        # Debug: Log input data
        self.logger.info(f"🔍 Input price_data shape: {price_data.shape}")
        self.logger.info(f"🔍 Input price_data columns: {list(price_data.columns)}")
        self.logger.info(f"🔍 Input volume_data shape: {volume_data.shape}")
        self.logger.info(f"🔍 Input volume_data columns: {list(volume_data.columns)}")

        # Log data quality metrics
        if not price_data.empty:
        self.logger.info(f"🔍 Price data range: {price_data.min().min():.6f} to {price_data.max().max():.6f}")
        self.logger.info(f"🔍 Price data NaN count: {price_data.isna().sum().sum()}")
        if not volume_data.empty:
        self.logger.info(f"🔍 Volume data range: {volume_data.min().min():.6f} to {volume_data.max().max():.6f}")
        self.logger.info(f"🔍 Volume data NaN count: {volume_data.isna().sum().sum()}")

        # Log order flow data if available
        if order_flow_data is not None:
        self.logger.info(f"🔍 Order flow data shape: {order_flow_data.shape}")
        self.logger.info(f"🔍 Order flow data columns: {list(order_flow_data.columns)}")
            else:
        self.logger.info("🔍 No order flow data provided")

        # Preprocess irregular intervals before feature engineering
            from src.training.steps.raw_data_quality_checker import (
                RawDataQualityChecker,
            )

        # Initialize data quality checker
            quality_checker = RawDataQualityChecker()

        # Preprocess price data to handle irregular intervals
        self.logger.info("🔧 Preprocessing price data for irregular intervals...")
        # Use enhanced preprocessing with intelligent gap handling
            symbol, getattr(self, "symbol", "ETHUSDT")
            exchange, getattr(self, "exchange", "BINANCE")

        # Ensure price_data has a proper DatetimeIndex
        if not isinstance(price_data.index, pd.DatetimeIndex):
        self.logger.warning("⚠️ Price data doesn't have DatetimeIndex, attempting to fix...")
        if "timestamp" in price_data.columns:
        # Convert timestamp column to DatetimeIndex
                    price_data = price_data.set_index("timestamp")
        self.logger.info("✅ Set timestamp column as DatetimeIndex")
                elif price_data.index.name == "timestamp":
        # Convert index to DatetimeIndex
                    price_data.index, pd.to_datetime(price_data.index)
        self.logger.info("✅ Converted index to DatetimeIndex")
                else:
        # Try to convert the existing index to datetime if it looks like timestamps
        try:
        if price_data.index.dtype == "object" or str(price_data.index.dtype).startswith("datetime"):
        # Try to parse the index as datetime
                            price_data.index, pd.to_datetime(price_data.index)
        self.logger.info("✅ Converted existing index to DatetimeIndex")
                        else:
        # Create a synthetic datetime index based on the data length
        self.logger.warning("⚠️ Creating synthetic datetime index - verify data alignment")
                            start_time = pd.Timestamp("2024-01-01 00:00:00")
                            interval, pd.Timedelta(minutes=1)  # Default to 1 minute intervals
                            timestamps = [start_time + i * interval for i in range(len(price_data))]
                            price_data.index, timestamps
        self.logger.info("✅ Created synthetic datetime index")
        except Exception as e:
        self.logger.exception(f"❌ Failed to create DatetimeIndex: {e}")
        return {}

            enhanced_price_data, quality_checker.enhanced_preprocess_market_data(
                price_data,
                symbol=symbol
                exchange=exchange
                expected_interval_seconds=60,  # 1-minute intervals
                max_forward_fill_seconds=10,  # Forward-fill gaps ≤10 seconds
                download_missing_data=True,    # Download data for gaps >10 seconds
            )

            preprocessed_price_data = enhanced_price_data
            price_validation = {
                "preprocessing_applied": {
                    "method": "enhanced",
                    "original_shape": price_data.shape,
                    "preprocessed_shape": enhanced_price_data.shape,
                    "improvement": 0.0,
                },
            }

        # Log preprocessing results
        if price_validation.get("preprocessing_applied"):
                preprocessing_info = price_validation["preprocessing_applied"]
        self.logger.info("✅ Price data preprocessing completed:")
        self.logger.info(f"   Method: {preprocessing_info['method']}")
        self.logger.info(f"   Original shape: {preprocessing_info['original_shape']}")
        self.logger.info(f"   Preprocessed shape: {preprocessing_info['preprocessed_shape']}")
        self.logger.info(f"   Quality improvement: {preprocessing_info['improvement']:.3f}")
                price_data = preprocessed_price_data
            else:
        self.logger.info("✅ No price data preprocessing needed")

        # Enhanced preprocessing for volume data if it has timestamps
        if hasattr(volume_data, "index") and isinstance(volume_data.index, pd.DatetimeIndex):
        self.logger.info("🔧 Enhanced preprocessing for volume data...")

                enhanced_volume_data, quality_checker.enhanced_preprocess_market_data(
                    volume_data,
                    symbol=symbol
                    exchange=exchange
                    expected_interval_seconds=60,  # 1-minute intervals
                    max_forward_fill_seconds=10,  # Forward-fill gaps ≤10 seconds
                    download_missing_data=True,    # Download data for gaps >10 seconds
                )

        # Update volume_data with enhanced preprocessed data
                volume_data = enhanced_volume_data
        self.logger.info("✅ Volume data enhanced preprocessing completed")
            else: self.logger.info("🔧 Volume data doesn't have DatetimeIndex = attempting to fix...")
        # Ensure volume_data has a proper DatetimeIndex
        if not isinstance(volume_data.index, pd.DatetimeIndex):
        if "timestamp" in volume_data.columns:
        # Convert timestamp column to DatetimeIndex
                        volume_data = volume_data.set_index("timestamp")
        self.logger.info("✅ Set timestamp column as DatetimeIndex for volume data")
                    elif volume_data.index.name == "timestamp":
        # Convert index to DatetimeIndex
                        volume_data.index, pd.to_datetime(volume_data.index)
        self.logger.info("✅ Converted volume data index to DatetimeIndex")
                    else:
        # Try to convert the existing index to datetime if it looks like timestamps
        try:
        if volume_data.index.dtype == "object" or str(volume_data.index.dtype).startswith("datetime"):
        # Try to parse the index as datetime
                                volume_data.index, pd.to_datetime(volume_data.index)
        self.logger.info("✅ Converted existing volume data index to DatetimeIndex")
        # Try to align volume data with price data index
                            elif hasattr(price_data, "index") and isinstance(price_data.index, pd.DatetimeIndex):
                                volume_data, volume_data.reindex(price_data.index, method="ffill")
        self.logger.info("✅ Aligned volume data with price data index")
                            else: self.logger.warning("⚠️ Cannot determine timestamp column for volume data = skipping preprocessing")
        except Exception as e:
        self.logger.warning(f"⚠️ Failed to fix volume data DatetimeIndex: {e}, skipping preprocessing")

        # Validate and transform data to ensure OHLCV structure
            price_data, volume_data, self._validate_and_transform_data(
                price_data = volume_data,
            )

        # Track NaN origins in input data
        self._track_nan_origins(
                "feature_engineering_input",
                {
                    "price_data": price_data,
                    "volume_data": volume_data,
                    "order_flow_data": order_flow_data,
                },
            )

            features = {}

        # Debug: Log input data information before feature generation
        self.logger.info("🔍 Input data validation before feature generation:")
        self.logger.info(f"   Price data shape: {price_data.shape if price_data is not None else 'None'}")
        self.logger.info(f"   Volume data shape: {volume_data.shape if volume_data is not None else 'None'}")
        self.logger.info(f"   Order flow data shape: {order_flow_data.shape if order_flow_data is not None else 'None'}")

        if price_data is not None and not price_data.empty:
        self.logger.info(f"   Price data index: {price_data.index.min()} to {price_data.index.max()}")
        self.logger.info(f"   Price data columns: {list(price_data.columns)}")
            else:
        self.logger.error("❌ Price data is empty or None")

        if volume_data is not None and not volume_data.empty:
        self.logger.info(f"   Volume data index: {volume_data.index.min()} to {volume_data.index.max()}")
        self.logger.info(f"   Volume data columns: {list(volume_data.columns)}")
            else:
        self.logger.error("❌ Volume data is empty or None")

        # Validate input data before proceeding
        if price_data is None or price_data.empty:
        self.logger.error("❌ Price data is required and cannot be empty")
        return {}

        if volume_data is None or volume_data.empty:
        self.logger.error("❌ Volume data is required and cannot be empty")
        return {}

        # Ensure data has datetime index
        if not isinstance(price_data.index, pd.DatetimeIndex):
        self.logger.error("❌ Price data must have datetime index")
        return {}

        if not isinstance(volume_data.index, pd.DatetimeIndex):
        self.logger.error("❌ Volume data must have datetime index")
        return {}

        # Check for minimum data requirements
        if len(price_data) < 10:
        self.logger.error(f"❌ Insufficient price data: {len(price_data)} records (minimum: 10)")
        return {}

        if len(volume_data) < 10:
        self.logger.error(f"❌ Insufficient volume data: {len(volume_data)} records (minimum: 10)")
        return {}

        self.logger.info("✅ Input data validation passed")
        self.logger.info("🔍 Starting feature generation pipeline...")

        # Add comprehensive coroutine detection and filtering
            def filter_coroutines(feature_dict: dict, source_name: str) -> dict:
                """Filter out any coroutine features from a feature dictionary."""
        if not isinstance(feature_dict, dict):
        self.logger.warning(f"⚠️ {source_name} is not a dict: {type(feature_dict)}")
        return {}

                filtered_features = {}
                coroutine_count = 0
        for key, value in feature_dict.items():
        if hasattr(value, "__await__"):
        self.logger.warning(f"⚠️ Skipping coroutine feature from {source_name}: {key}")
                        coroutine_count += 1
                        continue
                    filtered_features[key] = value

        if coroutine_count > 0:
        self.logger.info(f"⚠️ Filtered out {coroutine_count} coroutine features from {source_name}")

        return filtered_features

        # Market microstructure features
        self.logger.info("🔍 Generating microstructure features...")
            microstructure_features = (
        await self._engineer_microstructure_features_vectorized(
                    price_data,
                    volume_data,
                    order_flow_data,
                )
            )
        self.logger.info(f"🔍 Generated {len(microstructure_features)} microstructure features")
        if microstructure_features:
        self.logger.info(f"🔍 Microstructure feature names: {list(microstructure_features.keys())}")

        # Filter out any coroutine features before updating
            filtered_microstructure_features, filter_coroutines(microstructure_features, "microstructure")
            features.update(filtered_microstructure_features)
        self.logger.info(f"🔍 Total features after microstructure: {len(features)}")

        # Context dynamics for raw contextual signals (avoid using raw magnitudes as features)
        self.logger.info("🔍 Generating context dynamics features...")
            context_features_count = 0
        try:
                idx = price_data.index
        # funding_rate
        if "funding_rate" in price_data.columns:
                    fr, pd.Series(price_data["funding_rate"].values, index=idx)
        # Use multi-period difference to reduce correlation with base feature
                    features["funding_rate_change"] = fr.diff(3).fillna(0)
        with np.errstate(divide="ignore", invalid="ignore"):
                        features["funding_rate_returns"] = (
                            (fr.pct_change())
                            .replace([np.inf, -np.inf], np.nan)
                            .fillna(0)
                        )
        # z-score for stationarity
                    fr_roll, fr.rolling(50, min_periods=5)
                    fr_z = (fr - fr_roll.mean()) / fr_roll.std().replace(0, np.nan)
                    features["funding_rate_zscore"] = fr_z.replace(
                        [np.inf, -np.inf], np.nan,
                    ).fillna(0)
                    context_features_count += 3
        # volume_ratio
        if "volume_ratio" in price_data.columns:
                    vr, pd.Series(price_data["volume_ratio"].values, index=idx)
        # Use multi-period difference to reduce correlation with base feature
                    features["volume_ratio_change"] = vr.diff(3).fillna(0)
        with np.errstate(divide="ignore", invalid="ignore"):
                        features["volume_ratio_returns"] = (
                            (vr.pct_change())
                            .replace([np.inf, -np.inf], np.nan)
                            .fillna(0)
                        )
                        context_features_count += 2
        # trade_count
        if "trade_count" in price_data.columns:
                    tc, pd.Series(price_data["trade_count"].values, index=idx)
        # Use multi-period difference to reduce correlation with base feature
                    features["trade_count_change"] = tc.diff(3).fillna(0)
        with np.errstate(divide="ignore", invalid="ignore"):
                        features["trade_count_returns"] = (
                            (tc.pct_change())
                            .replace([np.inf, -np.inf], np.nan)
                            .fillna(0)
                        )
                        context_features_count += 2
        # trade_volume
        if "trade_volume" in price_data.columns:
                    tv, pd.Series(price_data["trade_volume"].values, index=idx)
        # Use multi-period difference to reduce correlation with base feature
                    features["trade_volume_change"] = tv.diff(3).fillna(0)
        with np.errstate(divide="ignore", invalid="ignore"):
                        features["trade_volume_returns"] = (
                            (tv.pct_change())
                            .replace([np.inf, -np.inf], np.nan)
                            .fillna(0)
                        )
                        context_features_count += 2
        except Exception as _e:
        self.logger.warning(f"⚠️ Context dynamics generation failed: {_e}")

        self.logger.info(f"🔍 Generated {context_features_count} context dynamics features")
        self.logger.info(f"🔍 Total features after context dynamics: {len(features)}")

        # Volatility regime features
        self.logger.info("🔍 Generating volatility regime features...")
        if self.volatility_model:
                volatility_features = (
        await self.volatility_model.model_volatility_vectorized(
                        price_data,
                        volume_data,
                    )
                )
        self.logger.info(f"🔍 Generated {len(volatility_features)} volatility features")
        if volatility_features:
        self.logger.info(f"🔍 Volatility feature names: {list(volatility_features.keys())}")
        # Ensure consistent numeric typing for downstream validation
        if "volatility_regime" in volatility_features:
                    vr = volatility_features["volatility_regime"]
        if isinstance(vr, str):
                        mapping = {"low": 0, "medium": 1, "high": 2}
                        volatility_features["volatility_regime"] = mapping.get(vr, 1)
        # Filter out any coroutine features before updating
                filtered_volatility_features, filter_coroutines(volatility_features, "volatility")
                features.update(filtered_volatility_features)
        self.logger.info(f"🔍 Total features after volatility: {len(features)}")
            else:
        self.logger.warning("⚠️ Volatility model not available")

        # Correlation analysis features
        self.logger.info("🔍 Generating correlation analysis features...")
        if self.correlation_analyzer:
                correlation_features = (
        await self.correlation_analyzer.analyze_correlations_vectorized(
                        price_data,
                    )
                )
        self.logger.info(f"🔍 Generated {len(correlation_features)} correlation features")
        if correlation_features:
        self.logger.info(f"🔍 Correlation feature names: {list(correlation_features.keys())}")
        # Filter out any coroutine features before updating
                filtered_correlation_features, filter_coroutines(correlation_features, "correlation")
                features.update(filtered_correlation_features)
        self.logger.info(f"🔍 Total features after correlation: {len(features)}")
            else:
        self.logger.warning("⚠️ Correlation analyzer not available")

        # Momentum analysis features
        self.logger.info("🔍 Generating momentum analysis features...")
        if self.momentum_analyzer:
                momentum_features = (
        await self.momentum_analyzer.analyze_momentum_vectorized(
                        price_data,
                        volume_data,
                    )
                )
        self.logger.info(f"🔍 Generated {len(momentum_features)} momentum features")
        if momentum_features:
        self.logger.info(f"🔍 Momentum feature names: {list(momentum_features.keys())}")
        # Filter out any coroutine features before updating
                filtered_momentum_features, filter_coroutines(momentum_features, "momentum")
                features.update(filtered_momentum_features)
        self.logger.info(f"🔍 Total features after momentum: {len(features)}")
            else:
        self.logger.warning("⚠️ Momentum analyzer not available")

        # Liquidity analysis features
        self.logger.info("🔍 Generating liquidity analysis features...")
        if self.liquidity_analyzer:
                liquidity_features = (
        await self.liquidity_analyzer.analyze_liquidity_vectorized(
                        price_data,
                        volume_data,
                    )
                )
        self.logger.info(f"🔍 Generated {len(liquidity_features)} liquidity features")
        if liquidity_features:
        self.logger.info(f"🔍 Liquidity feature names: {list(liquidity_features.keys())}")
        # Filter out any coroutine features before updating
                filtered_liquidity_features, filter_coroutines(liquidity_features, "liquidity")
                features.update(filtered_liquidity_features)
        self.logger.info(f"🔍 Total features after liquidity: {len(features)}")
            else:
        self.logger.warning("⚠️ Liquidity analyzer not available")

        # Candlestick pattern features
        self.logger.info("🔍 Generating candlestick pattern features...")
        if self.candlestick_analyzer:
                candlestick_features, await self.candlestick_analyzer.analyze_patterns(
                    price_data,
                )
        self.logger.info(f"🔍 Generated {len(candlestick_features)} candlestick features")
        if candlestick_features:
        self.logger.info(f"🔍 Candlestick feature names: {list(candlestick_features.keys())}")
        # Filter out any coroutine features before updating
                filtered_candlestick_features, filter_coroutines(candlestick_features, "candlestick")
                features.update(filtered_candlestick_features)
        self.logger.info(f"🔍 Total features after candlestick: {len(features)}")
            else:
        self.logger.warning("⚠️ Candlestick analyzer not available")

        # Immediately alongside candlestick/pattern features (requires OHLCV):
        # Compute classic OHLCV-based indicators using actual prices
        self.logger.info("🔍 Generating OHLCV price features...")
            ohlcv_price_features, self._engineer_ohlcv_price_features_vectorized(
                price_data,
            )
        self.logger.info(f"🔍 Generated {len(ohlcv_price_features)} OHLCV price features")
        if ohlcv_price_features:
        self.logger.info(f"🔍 OHLCV price feature names: {list(ohlcv_price_features.keys())}")
        # Filter out any coroutine features before updating
            filtered_ohlcv_price_features, filter_coroutines(ohlcv_price_features, "ohlcv_price")
            features.update(filtered_ohlcv_price_features)
        self.logger.info(f"🔍 Total features after OHLCV price: {len(features)}")

        # S/R distance features — generate sr_levels if not provided
        self.logger.info("🔍 Generating S/R distance features...")
        if self.sr_distance_calculator:
        # Generate S/R levels if not provided
        if not sr_levels:
        self.logger.info("🔍 Generating S/R levels from price data...")
                    sr_levels = self._generate_sr_levels(price_data)
                else:
        # Normalize incoming sr_levels to the format expected by the calculator
        try:
        if "support" not in sr_levels and "support_levels" in sr_levels:
                            sr_levels = {
                                "support": [
                                    lvl["price"] if isinstance(lvl, dict) and "price" in lvl else float(lvl)
        for lvl in sr_levels.get("support_levels", [])
                                ],
                                "resistance": [
                                    lvl["price"] if isinstance(lvl, dict) and "price" in lvl else float(lvl)
        for lvl in sr_levels.get("resistance_levels", [])
                                ],
                            }
                        elif "support" in sr_levels:
        # Ensure numeric arrays
        for k in ("support", "resistance"):
        if k in sr_levels:
                                    vals = sr_levels[k]
        if isinstance(vals, list):
                                        sr_levels[k] = [
                                            v["price"] if isinstance(v, dict) and "price" in v else float(v)
        for v in vals
                                        ]
        except Exception as _e:
        self.logger.warning(
                            f"⚠️ Failed to normalize provided SR levels, will attempt auto-generation instead: {_e}",
                        )
                        sr_levels = self._generate_sr_levels(price_data)

        if sr_levels:
                    sr_distance_features = (
        await self.sr_distance_calculator.calculate_sr_distances(
                            price_data,
                            sr_levels,
                        )
                    )
        self.logger.info(f"🔍 Generated {len(sr_distance_features)} S/R distance features")
        if sr_distance_features:
        self.logger.info(f"🔍 S/R distance feature names: {list(sr_distance_features.keys())}")
                    features.update(sr_distance_features)
        self.logger.info(f"🔍 Total features after S/R distance: {len(features)}")
                else: self.logger.warning("⚠️ Failed to generate S/R levels = skipping S/R distance features")
            else:
        self.logger.warning("⚠️ S/R distance calculator not available")

        # Wavelet transform features with caching
        self.logger.info("🔍 Generating wavelet transform features...")
        if self.wavelet_analyzer:
                wavelet_features, await self._get_wavelet_features_with_caching(
                    price_data,
                    volume_data,
                )
        self.logger.info(f"🔍 Generated {len(wavelet_features)} wavelet features")
        if wavelet_features:
        self.logger.info(f"🔍 Wavelet feature names: {list(wavelet_features.keys())}")
                features.update(wavelet_features)
        self.logger.info(f"🔍 Total features after wavelet: {len(features)}")
            else:
        self.logger.warning("⚠️ Wavelet analyzer not available")

        # Adaptive indicators
        self.logger.info("🔍 Generating adaptive indicators...")
            adaptive_features, self._engineer_adaptive_indicators_vectorized(
                price_data,
            )
        self.logger.info(f"🔍 Generated {len(adaptive_features)} adaptive features")
        if adaptive_features:
        self.logger.info(f"🔍 Adaptive feature names: {list(adaptive_features.keys())}")
            features.update(adaptive_features)
        self.logger.info(f"🔍 Total features after adaptive indicators: {len(features)}")

        # Debug: Log feature generation before selection
        self.logger.info(f"🔍 Generated {len(features)} features before selection")
        if len(features) < 10:
        self.logger.warning(f"⚠️ Very few features generated before selection: {list(features.keys())}")

        # Add basic features as fallback to ensure we have features
        self.logger.info("⚠️ No fallback")

        # Feature selection and dimensionality reduction
        # Re-enable feature selection for comprehensive feature engineering
            selected_features = self._select_optimal_features_vectorized(features)
        self.logger.info("🔍 Feature selection re-enabled for comprehensive feature engineering")

        # Debug: Log feature selection results
        self.logger.info(f"🔍 Selected {len(selected_features)} features after selection")
        if len(selected_features) < 10:
        self.logger.warning(f"⚠️ Very few features selected: {list(selected_features.keys())}")

        # Add multi-timeframe features if enabled
        if self.enable_multi_timeframe:
        self.logger.info("🔍 Generating multi-timeframe features...")
                multi_timeframe_features = (
        await self._engineer_multi_timeframe_features_vectorized(
                        price_data,
                        volume_data,
                        order_flow_data,
                        sr_levels,
                    )
                )
        self.logger.info(f"🔍 Generated {len(multi_timeframe_features)} multi-timeframe features")
        if multi_timeframe_features:
        self.logger.info(f"🔍 Multi-timeframe feature names: {list(multi_timeframe_features.keys())}")
        # Filter out any coroutine features from multi_timeframe_features before updating
                filtered_multi_timeframe_features, filter_coroutines(multi_timeframe_features, "multi_timeframe")
                selected_features.update(filtered_multi_timeframe_features)
        self.logger.info(f"🔍 Total features after multi-timeframe: {len(selected_features)}")
            else:
        self.logger.info("🔍 Multi-timeframe features disabled")

        # Meta-labeling deprecated
        self.logger.info("ℹ️ Meta-labeling is deprecated and disabled")

        # Explicit meta-labels deprecated
        self.logger.info("ℹ️ Explicit meta-labels are deprecated and disabled")

        # Enforce generator contract: ensure all values are 1D arrays of length n
            n = len(price_data)
            sanitized: dict[str = Any] = {}
            offenders: list[str] = []
        for k, v in selected_features.items():
        try: if isinstance(v = pd.Series):
                        arr = v.values.reshape(-1)
                    elif isinstance(v, np.ndarray):
                        arr, v.reshape(-1) if v.ndim >= 1 else None
                    elif isinstance(v, list):
                        arr = np.asarray(v).reshape(-1)
                    else:
        # scalar or unsupported type; mark offender and skip
                        offenders.append(k)
                        continue
        # Align to n rows (pad left with NaN or trim head)
        if len(arr) > n:
                        arr = arr[-n:]
                    elif len(arr) < n:
                        pad = n - len(arr)
                        arr, np.concatenate([np.full(pad, np.nan), arr])
                    sanitized[k] = arr
        except Exception:
                    offenders.append(k)
                    continue

        if offenders:
        self.logger.warning(
                    f"⚠️ Feature generator contract: skipped scalar/invalid outputs for features: {offenders[:20]}"
                    + (" ..." if len(offenders) > 20 else ""),
                )

        # Apply comprehensive NaN handling
            sanitized = self._handle_nan_values_robust(sanitized)

        # Track NaN origins in final output
        self._track_nan_origins("feature_engineering_output", sanitized)

        # Engineer difference and acceleration features
        if self.enable_difference_acceleration_features:
        self.logger.info("🔍 Generating difference and acceleration features...")

        # Validate that sanitized doesn't contain coroutines before processing
                valid_sanitized = {}
        for key, value in sanitized.items():
        if hasattr(value, "__await__"):
        self.logger.warning(f"⚠️ Skipping coroutine feature in sanitized: {key}")
                        continue
                    valid_sanitized[key] = value

        # Validate input data before difference engineering
        self._validate_difference_engineering_inputs(valid_sanitized, price_data)

                enhanced_features, await self._engineer_difference_and_acceleration_features(valid_sanitized, price_data)

        # Validate enhanced features before merging
        self._validate_enhanced_features(enhanced_features)

        self.logger.info(f"🔍 Generated {len(enhanced_features)} difference and acceleration features")
        if enhanced_features:
        self.logger.info(f"🔍 Difference/acceleration feature names: {list(enhanced_features.keys())}")

        # Validate enhanced features before merging to ensure no coroutines
                valid_enhanced_features = {}
                coroutine_count = 0
        for key, value in enhanced_features.items():
        if hasattr(value, "__await__"):
        self.logger.warning(f"⚠️ Skipping coroutine feature before merging: {key}")
                        coroutine_count += 1
                        continue
                    valid_enhanced_features[key] = value

        if coroutine_count > 0:
        self.logger.info(f"⚠️ Filtered out {coroutine_count} coroutine features before merging")

                sanitized.update(valid_enhanced_features)
        self.logger.info(f"🔍 Total features after difference/acceleration: {len(sanitized)}")

        # Apply final NaN handling to enhanced features
                sanitized = self._handle_nan_values_robust(sanitized)

        # Log feature engineering summary
        self._log_feature_engineering_summary(sanitized, enhanced_features)
            else:
        self.logger.info("ℹ️ Difference and acceleration features disabled by configuration")

        # 🔍 LOOKAHEAD BIAS DETECTION - Check for temporal alignment issues
        try:
        # Convert to DataFrame for detection
                features_df = pd.DataFrame(sanitized)

        # Run lookahead bias detection
                bias_results, detect_lookahead_bias(
                    features_df=features_df
                    target_series=pd.Series(
                        [0] * len(features_df),
                    ),  # Dummy target for feature-only check
                    timestamp_col=None
                )

        if bias_results.get("lookahead_bias_detected", False):
        self.logger.critical(
                        "🚨 LOOKAHEAD BIAS DETECTED IN FEATURE ENGINEERING!",
                    )
        for issue in bias_results.get("critical_issues", []):
        self.logger.critical(f"   ❌ {issue}")

        # Apply automatic lagging fix
        self.logger.info("🔧 Applying automatic lagging fix...")
                    lagged_features, apply_feature_lagging(features_df, lag_periods=1)
                    sanitized = lagged_features.to_dict("series")

                elif bias_results.get("warnings", []):
        self.logger.warning("⚠️ LOOKAHEAD BIAS WARNINGS DETECTED:")
        for warning in bias_results.get("warnings", []):
        self.logger.warning(f"   ⚠️ {warning}")

        except Exception as e:
        self.logger.warning(f"⚠️ Lookahead bias detection failed: {e}")

        # Final summary logging
        self.logger.info(
                f"✅ Engineered {len(sanitized)} vectorized advanced features including wavelet transforms",
            )

        # Log feature categories summary
            feature_categories = {}
        for feature_name in sanitized:
        if "wavelet" in feature_name.lower():
                    feature_categories["wavelet"] = feature_categories.get("wavelet", 0) + 1
                elif "momentum" in feature_name.lower() or "rsi" in feature_name.lower() or "macd" in feature_name.lower():
                    feature_categories["momentum"] = feature_categories.get("momentum", 0) + 1
                elif "volatility" in feature_name.lower():
                    feature_categories["volatility"] = feature_categories.get("volatility", 0) + 1
                elif "correlation" in feature_name.lower():
                    feature_categories["correlation"] = feature_categories.get("correlation", 0) + 1
                elif "volume" in feature_name.lower() or "liquidity" in feature_name.lower():
                    feature_categories["liquidity"] = feature_categories.get("liquidity", 0) + 1
                elif "candlestick" in feature_name.lower() or "pattern" in feature_name.lower():
                    feature_categories["candlestick"] = feature_categories.get("candlestick", 0) + 1
                elif "microstructure" in feature_name.lower() or "impact" in feature_name.lower():
                    feature_categories["microstructure"] = feature_categories.get("microstructure", 0) + 1
                elif "sr" in feature_name.lower() or "support" in feature_name.lower() or "resistance" in feature_name.lower():
                    feature_categories["sr_distance"] = feature_categories.get("sr_distance", 0) + 1
                elif "meta" in feature_name.lower():
                    feature_categories["meta_labeling"] = feature_categories.get("meta_labeling", 0) + 1
                elif "timeframe" in feature_name.lower():
                    feature_categories["multi_timeframe"] = feature_categories.get("multi_timeframe", 0) + 1
                else:
                    feature_categories["other"] = feature_categories.get("other", 0) + 1

        self.logger.info(f"📊 Feature categories: {feature_categories}")

        try:
        self.logger.info(
                    f"🧾 Vectorized feature list ({len(sanitized)}): {sorted(sanitized.keys())}",
                )
        except Exception as e:
        self.logger.warning(f"⚠️ Failed to log vectorized feature list: {e}")

        return sanitized

        except Exception as e:
        self.logger.exception(f"🚨 Error engineering vectorized advanced features: {e}")
        return {}

    @validate_wavelet_data_quality
    async def _get_wavelet_features_with_caching(
        self, price_data: pd.DataFrame, volume_data: pd.DataFrame, ) -> dict[str, Any]:
        """Get wavelet features with caching support.

        Args:
            price_data: OHLCV price data
            volume_data: Volume data (not used in current implementation)

        Returns:
            Dictionary containing wavelet features

        """
        try:
        if not self.wavelet_cache:
        # Fallback to direct computation if cache is not available
        return await self.wavelet_analyzer.analyze_wavelet_transforms(
                    price_data = # Only pass price_data
                )

        # Generate cache key
            wavelet_config = self.wavelet_analyzer.wavelet_config
            cache_key, self.wavelet_cache.generate_cache_key(
                price_data,
                wavelet_config,
                {
                    "volume_data_shape": volume_data.shape
        if volume_data is not None
                    else None,
                },
            )

        # Check if cache exists
        if self.wavelet_cache.cache_exists(cache_key):
        self.logger.info(f"📦 Loading wavelet features from cache: {cache_key}")
                cached_features, metadata, self.wavelet_cache.load_from_cache(
                    cache_key,
                )
        if cached_features:
        return cached_features
        # Fallthrough to recompute if cache was empty or invalid

        # Compute wavelet features
        self.logger.info(f"🔧 Computing wavelet features (not cached): {cache_key}")
            wavelet_features, await self.wavelet_analyzer.analyze_wavelet_transforms(
                price_data = # Only pass price_data
            )

        # Save to cache (only if non-empty)
            metadata = {
                "data_shape": price_data.shape,
                "volume_data_shape": volume_data.shape
        if volume_data is not None
                else None,
                "computation_time": time.time(),
            }

            cache_success, self.wavelet_cache.save_to_cache(
                cache_key = wavelet_features, metadata,
            )
        if cache_success:
        self.logger.info(f"💾 Cached wavelet features: {cache_key}")
            else:
        self.logger.warning(f"⚠️ Failed to cache wavelet features: {cache_key}")

        return wavelet_features

        except Exception as e:
        self.logger.exception(f"🚨 Error getting wavelet features with caching: {e}")
        return {}

    def get_cache_stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        if self.wavelet_cache:
        return self.wavelet_cache.get_cache_stats()
        return {"error": "Wavelet cache not initialized"}

    def clear_wavelet_cache(self, cache_key: str | None = None) -> bool:
        """Clear wavelet cache."""
        if self.wavelet_cache:
        return self.wavelet_cache.clear_cache(cache_key)
        return False

    async def _engineer_microstructure_features_vectorized(
        self, price_data: pd.DataFrame, volume_data: pd.DataFrame, order_flow_data: pd.DataFrame | None = None
    ) -> dict[str, Any]:
        """Engineer market microstructure features using vectorized operations."""
        try:
        self.logger.info("🔍 Starting microstructure feature engineering...")
            features = {}

        # Enhanced NaN tracking for microstructure features
        self._track_nan_origins(
                "microstructure_input",
                {
                    "price_data": price_data,
                    "volume_data": volume_data,
                    "order_flow_data": order_flow_data,
                },
            )

        # Price impact features (vectorized per-row)
            price_impact, self._calculate_price_impact_vectorized(
                price_data = volume_data,
            )
        self._track_nan_origins("price_impact", {"price_impact": price_impact})
            features["price_impact"] = price_impact
        self.logger.info(f"🔍 Added price_impact feature, total features: {len(features)}")

            volume_price_impact, self._calculate_volume_price_impact_vectorized(
                price_data = volume_data,
            )
        self._track_nan_origins(
                "volume_price_impact", {"volume_price_impact": volume_price_impact},
            )
            features["volume_price_impact"] = volume_price_impact
        self.logger.info(f"🔍 Added volume_price_impact feature, total features: {len(features)}")

        # Order-flow related features (proxies if book data not available)
            order_flow_imbalance, self._calculate_order_flow_imbalance_vectorized(
                price_data = volume_data, order_flow_data,
            )
        self._track_nan_origins(
                "order_flow_imbalance", {"order_flow_imbalance": order_flow_imbalance},
            )
            features["order_flow_imbalance"] = order_flow_imbalance

        # Generate spread dynamics instead of raw spread
            bas = self._calculate_bid_ask_spread_vectorized(price_data)
        self._track_nan_origins("bid_ask_spread", {"bid_ask_spread": bas})

        # Relative change (returns) and level as separate engineered metrics
            bas_returns, bas.pct_change().replace([np.inf, -np.inf], np.nan).fillna(0)
        self._track_nan_origins(
                "bid_ask_spread_returns", {"bid_ask_spread_returns": bas_returns},
            )
            features["bid_ask_spread_returns"] = bas_returns
            features["bid_ask_spread_level"] = bas  # bounded 0..0.05 already

        # Order book wall features (stationary): use returns/diffs
        try:
        if order_flow_data is not None:
        # Expect optional columns: bid_wall_price/size, ask_wall_price/size, mid
                    df = order_flow_data
        if "mid" in df.columns:
                        mid, pd.Series(df["mid"].values, index=df.index).reindex(
                            price_data.index, method="ffill"
                        )
                    else:
                        mid = price_data["close"].astype(float)
        # Distances to nearest walls in pct
        for side in ["bid", "ask"]:
                        pcol = f"{side}_wall_price"
                        scol = f"{side}_wall_size"
        if pcol in df.columns:
                            wall_p, pd.Series(df[pcol].values, index=df.index).reindex(
                                price_data.index, method="ffill"
                            )
        with np.errstate(divide="ignore", invalid="ignore"):
                                dist = (
                                    ((mid - wall_p).abs() / mid)
                                    .replace([np.inf, -np.inf], np.nan)
                                    .fillna(method="ffill")
                                    .fillna(1.0)
                                )
                            features[f"nearest_{side}_wall_dist_pct"] = dist
        if scol in df.columns:
                            wall_s = (
                                pd.Series(df[scol].values, index=df.index)
                                .reindex(price_data.index, method="ffill")
                                .fillna(0)
                            )
        # Use diff/returns for stationarity
        # Use shift(1) to avoid NaN in first row
                            features[f"nearest_{side}_wall_size_change"] = (
                                (wall_s - wall_s.shift(1)).fillna(0)
                            )
        with np.errstate(divide="ignore", invalid="ignore"):
                                features[f"nearest_{side}_wall_size_returns"] = (
                                    (wall_s.pct_change())
                                    .replace([np.inf, -np.inf], np.nan)
                                    .fillna(0)
                                )
        # Imbalance if total sizes available
        if (
                        "total_bid_size" in df.columns
                        and "total_ask_size" in df.columns
                    ):
                        tb = (
                            pd.Series(df["total_bid_size"].values, index=df.index)
                            .reindex(price_data.index, method="ffill")
                            .fillna(0)
                        )
                        ta = (
                            pd.Series(df["total_ask_size"].values, index=df.index)
                            .reindex(price_data.index, method="ffill")
                            .fillna(0)
                        )
                        denom = (tb + ta).replace(0, np.nan)
                        imb = (
                            ((tb - ta) / denom)
                            .replace([np.inf, -np.inf], np.nan)
                            .fillna(0)
                        )
                        features["orderbook_wall_imbalance"] = imb
        # Depth profile slope proxy: difference between near/far depth (if available)
        if "depth_near" in df.columns and "depth_far" in df.columns:
                        near = (
                            pd.Series(df["depth_near"].values, index=df.index)
                            .reindex(price_data.index, method="ffill")
                            .fillna(0)
                        )
                        far = (
                            pd.Series(df["depth_far"].values, index=df.index)
                            .reindex(price_data.index, method="ffill")
                            .fillna(0)
                        )
                        slope = near - far
        # Use shift(1) to avoid NaN in first row
                        features["depth_profile_slope_proxy"] = (slope - slope.shift(1)).fillna(0)
        # Weighted mid-price (if bid/ask price/size available)
        if all(
                        c in df.columns
        for c in [
                            "best_bid",
                            "best_ask",
                            "best_bid_size",
                            "best_ask_size",
                        ]
                    ):
                        bb = (
                            pd.Series(df["best_bid"].values, index=df.index)
                            .reindex(price_data.index, method="ffill")
                            .astype(float)
                        )
                        ba = (
                            pd.Series(df["best_ask"].values, index=df.index)
                            .reindex(price_data.index, method="ffill")
                            .astype(float)
                        )
                        bbs = (
                            pd.Series(df["best_bid_size"].values, index=df.index)
                            .reindex(price_data.index, method="ffill")
                            .astype(float)
                            .replace(0, np.nan)
                        )
                        bas = (
                            pd.Series(df["best_ask_size"].values, index=df.index)
                            .reindex(price_data.index, method="ffill")
                            .astype(float)
                            .replace(0, np.nan)
                        )
                        wmp = (bb * bbs + ba * bas) / (bbs + bas)
        # Use multi-period difference to reduce correlation with base feature
                        features["weighted_mid_price_change"] = wmp.diff(3).fillna(0)
        with np.errstate(divide="ignore", invalid="ignore"):
                            features["weighted_mid_price_returns"] = (
                                (wmp.pct_change())
                                .replace([np.inf, -np.inf], np.nan)
                                .fillna(0)
                            )
        # Aggregated orderbook pressure (if granular ladders available)
        if all(
                        c in df.columns for c in ["sum_bid_size_5", "sum_ask_size_5"]
                    ):
                        sb = (
                            pd.Series(df["sum_bid_size_5"].values, index=df.index)
                            .reindex(price_data.index, method="ffill")
                            .fillna(0)
                        )
                        sa = (
                            pd.Series(df["sum_ask_size_5"].values, index=df.index)
                            .reindex(price_data.index, method="ffill")
                            .fillna(0)
                        )
                        denom2 = (sb + sa).replace(0, np.nan)
                        press = (
                            ((sb - sa) / denom2)
                            .replace([np.inf, -np.inf], np.nan)
                            .fillna(0)
                        )
                        features["orderbook_pressure"] = press
        # Trade-to-order ratio (if trades and orders counts provided)
        if all(c in df.columns for c in ["trade_count", "order_count"]):
                        tr = (
                            pd.Series(df["trade_count"].values, index=df.index)
                            .reindex(price_data.index, method="ffill")
                            .fillna(0)
                        )
                        oc = (
                            pd.Series(df["order_count"].values, index=df.index)
                            .reindex(price_data.index, method="ffill")
                            .fillna(0)
                        )
        with np.errstate(divide="ignore", invalid="ignore"):
                            tor = (
                                (tr / oc.replace(0, np.nan))
                                .replace([np.inf, -np.inf], np.nan)
                                .fillna(0)
                            )
        # Use multi-period difference to reduce correlation with base feature
                        features["trade_to_order_ratio"] = tor.diff(3).fillna(0)
        except Exception as _e:
        self.logger.warning(
                    f"⚠️ Order book wall feature engineering failed: {_e}",
                )

        # Market depth features (vectorized per-row)
            md, self._calculate_market_depth_vectorized(price_data, volume_data)
        # Depth dynamics - use multi-period difference to reduce correlation
            features["market_depth_change"] = md.diff(3).fillna(0)
        with np.errstate(divide="ignore", invalid="ignore"):
                features["market_depth_returns"] = (
                    (md.pct_change()).replace([np.inf, -np.inf], np.nan).fillna(0)
                )
        # Depth imbalance proxy: short vs long window
            short, volume_data["volume"].rolling(10, min_periods=1).mean()
            long = (
                volume_data["volume"]
                .rolling(50, min_periods=1)
                .mean()
                .replace(0, np.nan)
            )
            features["market_depth_imbalance"] = (
                ((short - long) / long).replace([np.inf, -np.inf], np.nan).fillna(0)
            )

        # Additional kline/aggTrades-based proxies
        try:
        # BB z-score
                close = price_data["close"].astype(float)
                sma20, close.rolling(20, min_periods=5).mean()
                std20, close.rolling(20, min_periods=5).std().replace(0, np.nan)
        with np.errstate(divide="ignore", invalid="ignore"):
                    features["bb_zscore_20"] = (
                        ((close - sma20) / std20)
                        .replace([np.inf, -np.inf], np.nan)
                        .fillna(0)
                    )
        # MA slopes (first difference per bar)
                ema20, close.ewm(span=20, adjust=False).mean()
                sma50, close.rolling(50, min_periods=5).mean()
        # Use multi-period difference to reduce correlation with base features
                features["ema20_slope"] = ema20.diff(3).fillna(0)
                features["sma50_slope"] = sma50.diff(3).fillna(0)
        except Exception as e:
        self.logger.debug(f"⚠️ Error calculating MA slopes: {e}")
        # Use fallback values
                features["ema20_slope"] = pd.Series(0, index=close.index)
                features["sma50_slope"] = pd.Series(0, index=close.index)

        return features

        except Exception as e:
        self.logger.exception(f"🚨 Error engineering vectorized advanced features: {e}")
        return {}

    def _calculate_market_depth_vectorized(
        self = price_data: pd.DataFrame, volume_data: pd.DataFrame, ) -> pd.Series:
        """Calculate market depth using vectorized operations."""
        try:
        # Use volume as a proxy for market depth
        if "volume" in volume_data.columns:
        return volume_data["volume"].fillna(0)
        # Fallback to price-based depth estimation
            close = price_data["close"].astype(float)
        return close.rolling(10, min_periods=1).std().fillna(0)
        except Exception as e:
        self.logger.exception(f"Error calculating market depth: {e}")
        return pd.Series(0, index=price_data.index)

    def _calculate_bid_ask_spread_vectorized(
        self = price_data: pd.DataFrame, ) -> pd.Series:
        """Calculate bid-ask spread using aggtrades data for accurate spread estimation."""
        try:
        if "close" not in price_data.columns:
        return pd.Series(0.001, index=price_data.index)  # Default 0.1% spread

            close = price_data["close"].astype(float)

        # Track NaN origins in input data
        self._track_nan_origins(
                "bid_ask_spread_input",
                {
                    "close": close,
                    "avg_price": price_data.get("avg_price", pd.Series()),
                    "min_price": price_data.get("min_price", pd.Series()),
                    "max_price": price_data.get("max_price", pd.Series()),
                    "trade_count": price_data.get("trade_count", pd.Series()),
                    "volume_ratio": price_data.get("volume_ratio", pd.Series()),
                },
            )

        # Use aggtrades data for accurate spread calculation when available
        if all(
                col in price_data.columns
        for col in ["avg_price", "min_price", "max_price", "trade_count"]
            ):
                avg_price = price_data["avg_price"].astype(float)
                min_price = price_data["min_price"].astype(float)
                max_price = price_data["max_price"].astype(float)
                trade_count = price_data["trade_count"].astype(float)

        # Track NaN origins in aggtrades data
        self._track_nan_origins(
                    "bid_ask_spread_aggtrades",
                    {
                        "avg_price": avg_price,
                        "min_price": min_price,
                        "max_price": max_price,
                        "trade_count": trade_count,
                    },
                )

        # Calculate spread based on price range within the kline
        # Higher min-max range indicates wider spreads
                price_range = max_price - min_price
                mid_price = (max_price + min_price) / 2

        # Spread as percentage of mid price
                spread_pct = (price_range / mid_price).replace(0, np.nan)

        # Adjust for trade count - fewer trades often mean wider spreads
                trade_count_ma, trade_count.rolling(20, min_periods=1).mean()
                trade_count_ratio, trade_count / trade_count_ma.replace(0, 1)

        # Lower trade count ratio increases spread
                trade_adjustment = (
                    1 - trade_count_ratio.clip(0, 1)
                ) * 0.01  # Max 1% adjustment
                spread_pct += trade_adjustment

        # Use volume ratio if available for additional adjustment
        if "volume_ratio" in price_data.columns:
                    volume_ratio = price_data["volume_ratio"].astype(float)
        # Lower volume ratio (less trade volume relative to kline volume) indicates wider spreads
                    volume_adjustment = (
                        1 - volume_ratio.clip(0, 1)
                    ) * 0.005  # Max 0.5% adjustment
                    spread_pct += volume_adjustment

        # Ensure spread is within reasonable bounds (0-5%)
                spread_pct, spread_pct.clip(0, 0.05)

        # Clean up infinite and NaN values
        return spread_pct.replace([np.inf, -np.inf], np.nan).fillna(0.001)


        # Fallback to volatility-based proxy if aggtrades data not available
        self.logger.info(
                "📊 Using volatility-based spread proxy (aggtrades data not available)",
            )

        # Calculate spread proxy based on price volatility
            volatility = (
                close.rolling(20, min_periods=1).std()
                / close.rolling(20, min_periods=1).mean()
            )

        # Track volatility calculation
        self._track_nan_origins(
                "bid_ask_spread_volatility", {"volatility": volatility},
            )

        # Normalize volatility to a reasonable spread range (0-5%)
            spread_proxy = volatility * 0.05  # Scale to 5% max

        # Add volume-based adjustment if available
        if "volume" in price_data.columns:
                volume = price_data["volume"].astype(float)
                volume_ma, volume.rolling(20, min_periods=1).mean()
                volume_ratio, volume / volume_ma.replace(0, 1)
        # Lower volume ratio increases spread
                volume_adjustment = (
                    1 - volume_ratio.clip(0, 1)
                ) * 0.02  # Max 2% adjustment
                spread_proxy += volume_adjustment

        # Ensure spread is within reasonable bounds (0-5%)
            spread_proxy, spread_proxy.clip(0, 0.05)

        # Clean up infinite and NaN values
        return spread_proxy.replace([np.inf, -np.inf], np.nan).fillna(
                0.001,
            )


        except Exception as e:
        self.logger.exception(f"🚨 Error calculating bid-ask spread: {e}")
        return pd.Series(0.001, index=price_data.index)  # Default 0.1% spread

    def _track_nan_origins(self, stage: str, data_dict: dict[str, Any]) -> None:
        """Track NaN values throughout the feature engineering pipeline to identify origins."""
        try:
            nan_report = {}
            total_nans = 0

        for name, data in data_dict.items():
        # Skip coroutine objects
        if hasattr(data, "__await__"):
        self.logger.warning(f"⚠️ Skipping coroutine in NaN tracking for {stage}.{name}")
                    continue

        if isinstance(data, pd.Series):
                    nan_count = data.isna().sum()
        if nan_count > 0:
                        nan_report[name] = {
                            "nan_count": nan_count,
                            "total_count": len(data),
                            "nan_percentage": (nan_count / len(data)) * 100,
                            "first_nan_index": data.index[data.isna()].tolist()[:5]
        if nan_count > 0
                            else [],
                            "sample_values": data.dropna().head(3).tolist()
        if nan_count < len(data)
                            else [],
                        }
                        total_nans += nan_count
                elif isinstance(data, pd.DataFrame):
                    nan_counts = data.isna().sum()
        if nan_counts.any():
                        nan_report[name] = {
                            "nan_counts": nan_counts[nan_counts > 0].to_dict(),
                            "total_count": len(data),
                            "columns_with_nans": nan_counts[
                                nan_counts > 0
                            ].index.tolist(),
                        }
                        total_nans += nan_counts.sum()
                elif isinstance(data, np.ndarray):
                    nan_count = np.isnan(data).sum()
        if nan_count > 0:
                        nan_report[name] = {
                            "nan_count": nan_count,
                            "total_count": data.size,
                            "nan_percentage": (nan_count / data.size) * 100,
                        }
                        total_nans += nan_count

        if nan_report:
        # Only log if there are significant NaN values (more than 100)
        if total_nans > 100:
        self.logger.warning(
                        f"🚨 NaN detected in {stage}: {total_nans} total NaN values",
                    )
        self.logger.warning(f"🚨 NaN details for {stage}: {nan_report}")

        # Log specific problematic features
        for name, details in nan_report.items():
        if isinstance(details, dict) and "nan_percentage" in details:
        if details["nan_percentage"] > 50:
        self.logger.error(
                                    f"🚨 CRITICAL: {stage}.{name} has {details['nan_percentage']:.1f}% NaN values!",
                                )
                            elif details["nan_percentage"] > 10:
        self.logger.warning(
                                    f"⚠️ HIGH: {stage}.{name} has {details['nan_percentage']:.1f}% NaN values",
                                )
        # Remove the LOW logging for 0% or low NaN values

        except Exception as e:
        self.logger.exception(f"🚨 Error in NaN tracking for {stage}: {e}")

    def _choose_cwt_method(self, signal_length: int) -> str:
        """Choose the appropriate CWT method based on signal length."""
        try:
        if self.cwt_method_preference == "conv":
        return "conv"
        # Auto selection: use FFT for longer signals, direct conv for small
        if signal_length >= self.cwt_fft_threshold:
        return "fft"
        return "conv"
        except Exception as e:
        self.logger.exception(f"Error choosing CWT method: {e}")
        return "conv"

    @validate_ohlcv_data_quality
    def _engineer_ohlcv_price_features_vectorized(
        self = price_data: pd.DataFrame, ) -> dict[str, Any]:
        """Engineer basic OHLCV-based technical indicators using vectorized operations."""
        try:
            features = {}

        # Ensure we have the required OHLCV columns
            required_cols = ["open", "high", "low", "close"]
        if not all(col in price_data.columns for col in required_cols):
        self.logger.warning(
                    "⚠️ Missing required OHLCV columns for technical indicators",
                )
        return features

        # Convert to float to ensure numeric operations
            close = price_data["close"].astype(float)
            high = price_data["high"].astype(float)
            low = price_data["low"].astype(float)
            open_price = price_data["open"].astype(float)
            volume, price_data.get(
                "volume", pd.Series(1.0, index=price_data.index)
            ).astype(float)

        # Basic price-based features
        # features["close_returns"] = close.pct_change().fillna(0)  # REMOVED: Duplicate feature
            features["high_low_ratio"] = (high / low).fillna(1.0)
            features["close_open_ratio"] = (close / open_price).fillna(1.0)
            features["body_size"] = (close - open_price).abs() / close
            features["upper_shadow"] = (high - np.maximum(close, open_price)) / close
            features["lower_shadow"] = (np.minimum(close, open_price) - low) / close

        # Moving averages
            features["sma_5"] = close.rolling(5, min_periods=1).mean()
            features["sma_20"] = close.rolling(20, min_periods=1).mean()
            features["ema_12"] = close.ewm(span=12, adjust=False).mean()
            features["ema_26"] = close.ewm(span=26, adjust=False).mean()

        # Momentum indicators
            features["price_momentum_20"] = close / close.shift(20) - 1
            features["volatility_5"] = (
                close.rolling(5, min_periods=1).std()
                / close.rolling(5, min_periods=1).mean()
            )

        # Volume-based features
        if "volume" in price_data.columns:
        # features["volume_ma_5"] = volume.rolling(5, min_periods=1).mean()  # REMOVED: Duplicate feature
        # features["volume_ma_20"] = volume.rolling(20, min_periods=1).mean()  # REMOVED: Duplicate feature
                features["volume_momentum"] = volume / volume.shift(5) - 1
                features["volume_ratio"] = (
                    volume / volume.rolling(20, min_periods=1).mean()
                )

        # RSI
        # Use shift(1) to avoid NaN in first row
            delta = close - close.shift(1)
            gain = (delta.where(delta > 0, 0)).rolling(window=14, min_periods=1).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=1).mean()
            rs, gain / loss.replace(0, np.nan)
            features["rsi"] = 100 - (100 / (1 + rs)).fillna(50)

        # MACD
            ema12, close.ewm(span=12, adjust=False).mean()
            ema26, close.ewm(span=26, adjust=False).mean()
            features["macd"] = ema12 - ema26
            features["macd_signal"] = features["macd"].ewm(span=9, adjust=False).mean()
            features["macd_histogram"] = features["macd"] - features["macd_signal"]

        # Bollinger Bands
            sma20, close.rolling(20, min_periods=1).mean()
            std20, close.rolling(20, min_periods=1).std()
            features["bb_upper"] = sma20 + (std20 * 2)
            features["bb_lower"] = sma20 - (std20 * 2)
            features["bb_position"] = (close - features["bb_lower"]) / (
                features["bb_upper"] - features["bb_lower"]
            )

        # ATR (Average True Range)
            tr1 = high - low
            tr2 = abs(high - close.shift(1))
            tr3 = abs(low - close.shift(1))
            tr, pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            features["atr"] = tr.rolling(14, min_periods=1).mean()

        # Stochastic Oscillator
            lowest_low, low.rolling(14, min_periods=1).min()
            highest_high, high.rolling(14, min_periods=1).max()
            features["stoch_k"] = (
                100 * (close - lowest_low) / (highest_high - lowest_low)
            )
            features["stoch_d"] = features["stoch_k"].rolling(3, min_periods=1).mean()

        # Williams %R
            features["williams_r"] = (
                -100 * (highest_high - close) / (highest_high - lowest_low)
            )

        # Commodity Channel Index
            typical_price = (high + low + close) / 3
            sma_tp, typical_price.rolling(20, min_periods=1).mean()
            mad, typical_price.rolling(20, min_periods=1).apply(
                lambda x: np.mean(np.abs(x - x.mean())),
            )
            features["cci"] = (typical_price - sma_tp) / (0.015 * mad)

        # Rate of Change
            features["roc"] = (close / close.shift(10) - 1) * 100

        # Money Flow Index
            typical_price = (high + low + close) / 3
            money_flow = typical_price * volume
            positive_flow = (
                money_flow.where(typical_price > typical_price.shift(1), 0)
                .rolling(14, min_periods=1)
                .sum()
            )
            negative_flow = (
                money_flow.where(typical_price < typical_price.shift(1), 0)
                .rolling(14, min_periods=1)
                .sum()
            )
            mfi_ratio, positive_flow / negative_flow.replace(0, np.nan)
            features["mfi"] = 100 - (100 / (1 + mfi_ratio)).fillna(50)

        # Additional OHLCV features
        # Price efficiency ratio
            features["price_efficiency_20"] = abs(close - close.shift(20)) / (close.rolling(20).apply(lambda x: np.sum(np.abs(np.diff(x)))))

        # Volume price trend
            features["volume_price_trend"] = (volume * close.pct_change()).cumsum()

        # On-balance volume
            obv = (volume * np.sign(close.diff())).cumsum()
            features["on_balance_volume"] = obv

        # Accumulation/distribution line
            clv = ((close - low) - (high - close)) / (high - low)
            features["accumulation_distribution"] = (clv * volume).cumsum()

        # Chaikin money flow
            features["chaikin_money_flow"] = (clv * volume).rolling(20).sum() / volume.rolling(20).sum()

        # Price momentum oscillator
            features["price_momentum_oscillator"] = close - close.rolling(10).mean()

        # Volume momentum oscillator
            features["volume_momentum_oscillator"] = volume - volume.rolling(10).mean()

        # Price velocity (rate of change of price)
            features["price_velocity_5"] = close.pct_change(5).fillna(0)
            features["price_velocity_10"] = close.pct_change(10).fillna(0)

        # Volume velocity
            features["volume_velocity_5"] = volume.pct_change(5).fillna(0)
            features["volume_velocity_10"] = volume.pct_change(10).fillna(0)

        # Clean up any infinite or NaN values
        for key in list(features.keys():
        if isinstance(features[key], pd.Series):
                    features[key] = (
                        features[key].replace([np.inf, -np.inf], np.nan).fillna(0)
                    )

        self.logger.info(f"✅ Generated {len(features)} OHLCV technical indicators")
        return features

        except Exception as e:
        self.logger.exception(f"🚨 Error engineering OHLCV features: {e}")
        return {}

    @validate_data_quality(validation_level=ValidationLevel.WARNING)
    def _engineer_adaptive_indicators_vectorized(
        self = price_data: pd.DataFrame, ) -> dict[str, Any]:
        """Engineer adaptive indicators that adjust to market conditions."""
        try:
            features = {}

        if "close" not in price_data.columns:
        self.logger.warning(
                    "⚠️ No 'close' column found in price_data for adaptive indicators",
                )
        return features

            close = price_data["close"].astype(float)
        self.logger.debug(
                f"🔍 Adaptive indicators: close price range {close.min():.2f} to {close.max():.2f}",
            )

        # Check for NaN/inf values in close price
            nan_count = close.isna().sum()
            inf_count = np.isinf(close).sum()
        if nan_count > 0 or inf_count > 0:
        self.logger.warning(
                    f"⚠️ Found {nan_count} NaN and {inf_count} inf values in close price",
                )
                close = (
                    close.replace([np.inf, -np.inf], np.nan)
                    .fillna(method="ffill")
                    .fillna(method="bfill")
                )

        # Adaptive moving averages
            volatility = (
                close.rolling(20, min_periods=1).std()
                / close.rolling(20, min_periods=1).mean()
            )

        # Fix: Handle NaN and inf values before converting to int
            volatility, volatility.replace([np.inf, -np.inf], np.nan).fillna(
                0.1,
            )  # Default volatility
        self.logger.debug(
                f"🔍 Volatility range: {volatility.min():.4f} to {volatility.max():.4f}",
            )

            adaptive_period = (20 * volatility).clip(5, 50)
        self.logger.debug(
                f"🔍 Adaptive period range: {adaptive_period.min():.1f} to {adaptive_period.max():.1f}",
            )

        # Convert to int safely after handling non-finite values
        try:
                adaptive_period = adaptive_period.astype(int)
        self.logger.debug("🔍 Successfully converted adaptive_period to int")
        except Exception as e:
        self.logger.warning(
                    f"⚠️ Error converting adaptive_period to int: {e}, using default values",
                )
        self.logger.debug(
                    f"🔍 Adaptive_period sample values: {adaptive_period.head().tolist()}",
                )
                adaptive_period, pd.Series(20, index=close.index, dtype=int)

        # Create adaptive SMA
            adaptive_sma, pd.Series(index=close.index, dtype=float)
        for i in range(len(close):
                period, max(1, int(adaptive_period.iloc[i]))
                start_idx, max(0, i - period + 1)
                adaptive_sma.iloc[i] = close.iloc[start_idx : i + 1].mean()

            features["adaptive_sma"] = adaptive_sma
        # Use shift(1) to avoid NaN in first row
            features["adaptive_sma_slope"] = (adaptive_sma - adaptive_sma.shift(1)).fillna(0)

        # Adaptive RSI
            adaptive_rsi_period = (14 * volatility).clip(5, 30)
        self.logger.debug(
                f"🔍 Adaptive RSI period range: {adaptive_rsi_period.min():.1f} to {adaptive_rsi_period.max():.1f}",
            )

        # Convert to int safely after handling non-finite values
        try:
                adaptive_rsi_period = adaptive_rsi_period.astype(int)
        self.logger.debug(
                    "🔍 Successfully converted adaptive_rsi_period to int",
                )
        except Exception as e:
        self.logger.warning(
                    f"⚠️ Error converting adaptive_rsi_period to int: {e}, using default values",
                )
        self.logger.debug(
                    f"🔍 Adaptive_rsi_period sample values: {adaptive_rsi_period.head().tolist()}",
                )
                adaptive_rsi_period, pd.Series(14, index=close.index, dtype=int)

            adaptive_rsi, pd.Series(index=close.index, dtype=float)

        for i in range(len(close):
                period, max(1, int(adaptive_rsi_period.iloc[i]))
        if i >= period:
        # Use shift(1) to avoid NaN in first row
                    price_slice = close.iloc[i - period + 1 : i + 1]
                    delta = price_slice - price_slice.shift(1)
                    gain, delta.where(delta > 0, 0).mean()
                    loss = (-delta.where(delta < 0, 0)).mean()
        if loss != 0:
                        rs = gain / loss
                        adaptive_rsi.iloc[i] = 100 - (100 / (1 + rs))
                    else:
                        adaptive_rsi.iloc[i] = 50
                else:
                    adaptive_rsi.iloc[i] = 50

            features["adaptive_rsi"] = adaptive_rsi

        # Clean up any infinite or NaN values
        for key in list(features.keys():
        if isinstance(features[key], pd.Series):
                    features[key] = (
                        features[key].replace([np.inf, -np.inf], np.nan).fillna(0)
                    )

        self.logger.info(
                f"✅ Successfully engineered {len(features)} adaptive indicators",
            )
        return features

        except Exception as e:
        self.logger.exception(f"🚨 Error engineering adaptive indicators: {e}")
        self.logger.debug(f"🔍 Exception details: {type(e).__name__}: {e!s}")
        return {}



    def _select_optimal_features_vectorized(
        self = features: dict[str, Any], ) -> dict[str, Any]:
        """Select optimal features based on variance and correlation."""
        try:
        if not features:
        return features

        # Convert to DataFrame for analysis
            df = pd.DataFrame(features)

        # Remove constant features with a more appropriate threshold for financial data
        # Use a more reasonable threshold for financial time series data
        # Increased threshold to be less aggressive - financial data often has small but meaningful variations
            variance = df.var()

        # More lenient threshold for financial data - many features have small but meaningful variations
        # Use 1e-8 instead of 1e-12 to be less aggressive
            non_constant = variance[
                variance > 1e-8
            ].index.tolist()

        if len(non_constant) < len(features):
                constant_features = [
                    col for col in features if col not in non_constant
                ]

        # Only log if we're removing a significant number of features
        if len(constant_features) > 5:
        self.logger.info(
                        f"🗑️ Removed {len(constant_features)} constant features: {constant_features[:5]}... (showing first 5)",
                    )
                elif len(constant_features) > 0:
        self.logger.info(
                        f"🗑️ Removed {len(constant_features)} constant features: {constant_features}",
                    )

        self.logger.info(
                    f"📊 Remaining features: {len(non_constant)} out of {len(features)} total",
                )

        # Log details about the removed features for debugging
        for feature in constant_features:
        if feature in variance.index:
                        feature_variance = variance[feature]
        self.logger.debug(
                            f"🔍 Removed feature '{feature}' with variance: {feature_variance:.2e}",
                        )
            else:
        self.logger.info(
                    f"✅ No constant features found - all {len(features)} features have sufficient variance",
                )

        # Select non-constant features
        return {
                col: features[col] for col in non_constant if col in features
            }


        except Exception as e:
        self.logger.exception(f"🚨 Error selecting optimal features: {e}")
        return features

    @validate_multi_timeframe_data_quality
    @cache_feature_engineering(max_memory_mb=2048)
    async def _engineer_multi_timeframe_features_vectorized(
        self, price_data: pd.DataFrame, volume_data: pd.DataFrame, order_flow_data: pd.DataFrame | None = None, sr_levels: dict[str, Any] | None, None, ) -> dict[str, Any]:
        """Engineer multi-timeframe features with timestamp regularization and optimizations."""
        try:
        # Initialize Mac M1 optimizations
            optimize_for_m1_mac()

            features = {}

        # Preprocess data to ensure regular timestamps
            processed_price, processed_volume, processed_order_flow, preprocess_data_for_multi_timeframe(
                price_data = volume_data, order_flow_data,
            )

        # Apply data type optimization to input data
            processed_price, optimize_feature_engineering_pipeline(processed_price, stage="input")
            processed_volume, optimize_feature_engineering_pipeline(processed_volume, stage="input")
        if processed_order_flow is not None:
                processed_order_flow, optimize_feature_engineering_pipeline(processed_order_flow, stage="input")

        # Generate multi-timeframe features using resampling
            timeframes = ["1m", "5m", "15m", "30m", "1h"]

        for tf in timeframes:
        try:
        # Resample price data to different timeframes
                    tf_price, self._resample_price_data(processed_price, tf)
                    tf_volume, self._resample_volume_data(processed_volume, tf) if processed_volume is not None else None

        if tf_price is not None and not tf_price.empty:
        # Log data quality for debugging
        self.logger.debug(f"🔍 {tf} timeframe - Price data shape: {tf_price.shape}")
        self.logger.debug(f"🔍 {tf} timeframe - Price data NaN count: {tf_price.isna().sum().sum()}")
        if tf_volume is not None and not tf_volume.empty:
        self.logger.debug(f"🔍 {tf} timeframe - Volume data shape: {tf_volume.shape}")
        self.logger.debug(f"🔍 {tf} timeframe - Volume data NaN count: {tf_volume.isna().sum().sum()}")

        # Check if we have sufficient data for this timeframe
                        min_required_data = self._get_minimum_data_requirement(tf)
        if len(tf_price) < min_required_data:
        self.logger.info(f"ℹ️ Skipping {tf} timeframe - insufficient data: {len(tf_price)} rows (minimum required: {min_required_data})")
                            continue

        # Generate features for this timeframe using existing comprehensive generator
                        tf_features, self._generate_timeframe_features(tf_price, tf_volume, tf)

        # For higher timeframes, we need to align the features back to the original 1-minute data
        if tf != "1m" and tf_features:
                            aligned_features = {}
        for feature_name, feature_series in tf_features.items():
        if isinstance(feature_series, pd.Series):
        # Align the feature series to the original 1-minute data index
                                    aligned_feature, feature_series.reindex(processed_price.index, method="ffill")
                                    aligned_feature, aligned_feature.fillna(method="bfill").fillna(0)
                                    aligned_features[feature_name] = aligned_feature
                                else:
                                    aligned_features[feature_name] = feature_series

                            tf_features = aligned_features
        self.logger.debug(f"🔍 Aligned {tf} features to 1-minute data index")

        # Log feature generation results
        if tf_features:
                            features.update(tf_features)
        self.logger.info(f"✅ Generated {len(tf_features)} features for {tf} timeframe")
        self.logger.debug(f"🔍 {tf} features: {list(tf_features.keys())}")
                        else:
        self.logger.info(f"ℹ️ No features generated for {tf} timeframe - insufficient data quality")
                    else:
        self.logger.info(f"ℹ️ Skipping {tf} timeframe - no data available after resampling")

        except Exception as e:
        self.logger.warning(f"⚠️ Failed to generate features for {tf} timeframe: {e}")
                    continue

        # Generate additional cross-timeframe features
            cross_timeframe_features, await self._generate_cross_timeframe_features(features, processed_price)
            features.update(cross_timeframe_features)

        # Log summary of multi-timeframe feature generation
        self._log_multi_timeframe_summary(features, timeframes)

        # Generate regime-aware features if HMM data is available
        try: regime_features = await self._generate_regime_aware_features(processed_price, processed_volume)
        if isinstance(regime_features, dict):
                    features.update(regime_features)
                else:
        self.logger.warning(f"⚠️ Regime features not a dict: {type(regime_features)}")
        except Exception as e:
        self.logger.warning(f"⚠️ Error generating regime features: {e}")
                regime_features = {}

        # Apply data type optimization to output
        if features:
        # Convert features dict to DataFrame for optimization, then back to dict
                features_df = pd.DataFrame(features)
                optimized_features_df, optimize_feature_engineering_pipeline(features_df, stage="output")
                features = optimized_features_df.to_dict("series")
        self.logger.info(f"✅ Generated {len(features)} multi-timeframe features total")

        # Validate features before returning
                features = self._validate_and_clean_features(features)

        # Ensure pickle safety (remove any async objects)
                features = self._ensure_pickle_safe_features(features)

        # Final validation - ensure we have meaningful features
        if len(features) == 0:
        self.logger.warning("⚠️ No valid features generated after validation")

            else:
        self.logger.warning("⚠️ No features generated")

        return features

        except Exception as e:
        self.logger.exception(f"🚨 Error engineering multi-timeframe features: {e}")
        # Don't fall back to basic features - let the error propagate
            raise

    def _remove_constant_features(self, features: dict[str, Any]) -> dict[str, Any]:
        """Remove features with zero or near-zero variance."""
        try:
            non_constant_features = {}
            constant_features = []
            variance_threshold = 1e-12  # Very small threshold for true constants

        for key, value in features.items():
        if isinstance(value, pd.Series):
        # Check if feature has meaningful variance
                    feature_variance = value.var()
        if feature_variance > variance_threshold:
                        non_constant_features[key] = value
                    else:
                        constant_features.append(key)
                else:
        # Keep non-series features
                    non_constant_features[key] = value

        if constant_features:
        self.logger.info(f"🗑️ Removed {len(constant_features)} constant features: {constant_features[:5]}... (showing first 5)")

        return non_constant_features

        except Exception as e:
        self.logger.exception(f"❌ Error removing constant features: {e}")
        return features

    def _ensure_pickle_safe_features(self, features: dict[str, Any]) -> dict[str, Any]:
        """Ensure all features are pickle-safe by removing any coroutines or async objects."""
        try:
            pickle_safe_features = {}
            removed_features = []

        for key, value in features.items():
        # Check if value is a coroutine or async object
        if hasattr(value, "__await__") or asyncio.iscoroutine(value):
                    removed_features.append(key)
        self.logger.warning(f"⚠️ Skipping coroutine feature: {key}")
                    continue
        if hasattr(value, "__aiter__") or hasattr(value, "__anext__"):
                    removed_features.append(key)
        self.logger.warning(f"⚠️ Skipping async iterator feature: {key}")
                    continue
        if callable(value) and asyncio.iscoroutinefunction(value):
                    removed_features.append(key)
        self.logger.warning(f"⚠️ Skipping async function feature: {key}")
                    continue
                pickle_safe_features[key] = value

        if removed_features:
        self.logger.info(f"✅ Removed {len(removed_features)} non-pickle-safe features: {removed_features}")

        return pickle_safe_features

        except Exception as e:
        self.logger.exception(f"❌ Error ensuring pickle safety: {e}")
        return features

    def _resample_price_data(self, price_data: pd.DataFrame, timeframe: str) -> pd.DataFrame | None:
        """Resample price data to target timeframe with irregular interval handling.

        Args:
            price_data: Price data DataFrame
            timeframe: Target timeframe

        Returns:
            Resampled price data or None if failed

        """
        try:
        if price_data.empty:
        return None

        # Handle irregular time intervals first
            regularized_data, self._handle_irregular_time_intervals(price_data, timeframe)

        # Use optimized resampler
            resampled_data, self.optimized_resampler.resample_optimized(regularized_data, timeframe)

        if resampled_data.empty:
        self.logger.warning(f"⚠️ Resampling to {timeframe} resulted in empty data")
        return None

        return resampled_data

        except Exception as e:
        self.logger.exception(f"🚨 Error resampling price data to {timeframe}: {e}")
        return None

    def _resample_volume_data(self, volume_data: pd.DataFrame, timeframe: str) -> pd.DataFrame | None:
        """Resample volume data to target timeframe with irregular interval handling.

        Args:
            volume_data: Volume data DataFrame
            timeframe: Target timeframe

        Returns:
            Resampled volume data or None if failed

        """
        try:
        if volume_data is None or volume_data.empty:
        return None

        # Handle irregular time intervals first
            regularized_data, self._handle_irregular_time_intervals(volume_data, timeframe)

        # Use optimized resampler
            resampled_data, self.optimized_resampler.resample_optimized(regularized_data, timeframe)

        if resampled_data.empty:
        self.logger.warning(f"⚠️ Resampling volume to {timeframe} resulted in empty data")
        return None

        return resampled_data

        except Exception as e:
        self.logger.exception(f"🚨 Error resampling volume data to {timeframe}: {e}")
        return None

    def _generate_timeframe_features(self, price_data: pd.DataFrame, volume_data: pd.DataFrame, timeframe: str) -> dict[str, Any]:
        """Generate features for a specific timeframe with improved NaN handling."""
        try:
            features = {}

        # Get minimum data requirement for this timeframe
            min_required_data = self._get_minimum_data_requirement(timeframe)
        if price_data.empty or len(price_data) < min_required_data:
        self.logger.info(f"ℹ️ Insufficient data for {timeframe} timeframe: {len(price_data)} rows (minimum required: {min_required_data})")
        return features

        # Basic price features for this timeframe
            close = price_data["close"].astype(float)
            high = price_data["high"].astype(float)
            low = price_data["low"].astype(float)
            open_price = price_data["open"].astype(float)

        # Handle NaN values in input data
            close, close.fillna(method="ffill").fillna(method="bfill").fillna(0)
            high, high.fillna(method="ffill").fillna(method="bfill").fillna(0)
            low, low.fillna(method="ffill").fillna(method="bfill").fillna(0)
            open_price, open_price.fillna(method="ffill").fillna(method="bfill").fillna(0)

        # Validate input data
        if close.isna().all() or close.std() == 0:
        self.logger.warning(f"⚠️ Invalid close data for {timeframe} timeframe")
        return features

        # For higher timeframes, we need to be more conservative with rolling windows
        # since we have fewer data points
        if timeframe == "1m":
        # Use standard rolling windows for 1m data
                sma_window = 20
                rsi_window = 14
                vol_window = 20
                min_periods = 5
            elif timeframe in ["5m", "15m"]:
        # Use smaller windows for 5m and 15m data
                sma_window = 10
                rsi_window = 7
                vol_window = 10
                min_periods = 3
            else:
        # Use very small windows for higher timeframes
                sma_window = 5
                rsi_window = 5
                vol_window = 5
                min_periods = 2

        # Moving averages (only if we have enough data)
        if len(close) >= min_periods:
                sma, close.rolling(sma_window, min_periods=min_periods).mean()
                sma, sma.fillna(method="ffill").fillna(method="bfill").fillna(0)
        if sma.var() > 1e-12:  # Check for meaningful variance
                    features[f"sma_{sma_window}_{timeframe}"] = sma

                ema, close.ewm(span=min(12, len(close)//2), adjust=False).mean()
                ema, ema.fillna(method="ffill").fillna(method="bfill").fillna(0)
        if ema.var() > 1e-12:
                    features[f"ema_{min(12, len(close)//2)}_{timeframe}"] = ema

        # Momentum indicators (only if we have enough data)
        if len(close) >= min_periods:
                rsi, self._calculate_rsi(close, rsi_window)
                rsi, rsi.fillna(method="ffill").fillna(method="bfill").fillna(50)
        if rsi.var() > 1e-12:
                    features[f"rsi_{timeframe}"] = rsi

                macd = self._calculate_macd(close)
                macd, macd.fillna(method="ffill").fillna(method="bfill").fillna(0)
        if macd.var() > 1e-12:
                    features[f"macd_{timeframe}"] = macd

        # Volatility (only if we have enough data)
        if len(close) >= min_periods:
                returns, close.pct_change()
        # Handle NaN values properly
                returns, returns.fillna(method="ffill").fillna(method="bfill").fillna(0)
                volatility, returns.rolling(vol_window, min_periods=min_periods).std()
                volatility, volatility.fillna(method="ffill").fillna(method="bfill").fillna(0)
        if volatility.var() > 1e-12:
                    features[f"volatility_{timeframe}"] = volatility

        # Volume features if available
        if volume_data is not None and not volume_data.empty and "volume" in volume_data.columns:
                volume = volume_data["volume"].astype(float)
                volume, volume.fillna(method="ffill").fillna(method="bfill").fillna(0)

        if len(volume) >= min_periods and volume.var() > 1e-12:
                    volume_ma, volume.rolling(vol_window, min_periods=min_periods).mean()
                    volume_ma, volume_ma.fillna(method="ffill").fillna(method="bfill").fillna(0)
        if volume_ma.var() > 1e-12:
                        features[f"volume_ma_{timeframe}"] = volume_ma

        # Volume ratio with safety check
                    volume_ratio, volume / (volume_ma + 1e-8)  # Avoid division by zero
                    volume_ratio, volume_ratio.fillna(method="ffill").fillna(method="bfill").fillna(1)
        if volume_ratio.var() > 1e-12:
                        features[f"volume_ratio_{timeframe}"] = volume_ratio

        # Price position (only if we have enough data)
        if len(close) >= min_periods:
                high_roll, high.rolling(vol_window, min_periods=min_periods).max()
                low_roll, low.rolling(vol_window, min_periods=min_periods).min()
                high_roll, high_roll.fillna(method="ffill").fillna(method="bfill").fillna(close)
                low_roll, low_roll.fillna(method="ffill").fillna(method="bfill").fillna(close)

        # Safety check for division by zero
                price_range = high_roll - low_roll
                price_position = (close - low_roll) / (price_range + 1e-8)  # Avoid division by zero
                price_position, price_position.fillna(method="ffill").fillna(method="bfill").fillna(0.5)
        if price_position.var() > 1e-12:
                    features[f"price_position_{timeframe}"] = price_position

        # Validate generated features with improved NaN handling
            valid_features = {}
        for name, feature in features.items():
        if isinstance(feature, pd.Series):
        # Handle any remaining NaN values
                    feature, feature.fillna(method="ffill").fillna(method="bfill").fillna(0)

        # Check for meaningful variance
        if feature.var() > 1e-12 and not feature.isna().all():
                        valid_features[name] = feature
                    else:
        self.logger.debug(f"⚠️ Skipping constant feature: {name}")
                else:
                    valid_features[name] = feature

        self.logger.debug(f"✅ Generated {len(valid_features)} valid features for {timeframe} timeframe")
        return valid_features

        except Exception as e:
        self.logger.exception(f"🚨 Error generating features for {timeframe}: {e}")
        return {}

    def _generate_cross_timeframe_features(self, price_data: pd.DataFrame, volume_data: pd.DataFrame | None = None) -> dict[str, Any]:
        """Generate comprehensive cross-timeframe features."""
        try:
            features = {}

        if price_data.empty or len(price_data) < 100:  # Need sufficient data
        self.logger.warning(f"⚠️ Insufficient data for cross-timeframe features: {len(price_data)} rows")
        return features

            close = price_data["close"].astype(float)
            high = price_data["high"].astype(float)
            low = price_data["low"].astype(float)
            price_data["open"].astype(float)

        # Validate input data
        if close.isna().all() or close.std() == 0:
        self.logger.warning("⚠️ Invalid close data for cross-timeframe features")
        return features

        # Define multiple timeframes for cross-timeframe analysis (reduced set for safety)
            timeframes = [1, 3, 5, 10, 15, 20]  # Reduced from 10 to 6 timeframes

        # 1. Cross-timeframe momentum features (with validation)
        for i, tf1 in enumerate(timeframes[:4]):  # Use first 4 timeframes
        for tf2 in timeframes[i+1:5]:  # Compare with next timeframes
        if tf1 < len(close) and tf2 < len(close):
        # Price momentum differences
                        momentum_diff = close.pct_change(tf1) - close.pct_change(tf2)
        if momentum_diff.var() > 1e-12:
                            features[f"momentum_{tf1}m_{tf2}m"] = momentum_diff

        # Momentum ratio with safety check
                        momentum_ratio = close.pct_change(tf1) / (close.pct_change(tf2) + 1e-8)
        if momentum_ratio.var() > 1e-12:
                            features[f"momentum_ratio_{tf1}m_{tf2}m"] = momentum_ratio

        # High-Low momentum differences (only if we have enough data)
        if len(close) >= max(tf1, tf2) * 2:
                            hl_momentum_1 = (high.rolling(tf1, min_periods=tf1//2).max() - low.rolling(tf1, min_periods=tf1//2).min()) / (close.rolling(tf1, min_periods=tf1//2).mean() + 1e-8)
                            hl_momentum_2 = (high.rolling(tf2, min_periods=tf2//2).max() - low.rolling(tf2, min_periods=tf2//2).min()) / (close.rolling(tf2, min_periods=tf2//2).mean() + 1e-8)
                            hl_diff = hl_momentum_1 - hl_momentum_2
        if hl_diff.var() > 1e-12:
                                features[f"hl_momentum_{tf1}m_{tf2}m"] = hl_diff

        # 2. Cross-timeframe volatility features (with validation)
        for i, tf1 in enumerate(timeframes[:3]):
        for tf2 in timeframes[i+1:4]:
        if tf1 < len(close) and tf2 < len(close):
                        returns, close.pct_change().fillna(method="ffill").fillna(method="bfill").fillna(0)
                        returns_1, returns.rolling(tf1, min_periods=tf1//2).std()
                        returns_2, returns.rolling(tf2, min_periods=tf2//2).std()

        # Volatility ratio with safety check
                        vol_ratio = returns_1 / (returns_2 + 1e-8)
        if vol_ratio.var() > 1e-12:
                            features[f"volatility_ratio_{tf1}m_{tf2}m"] = vol_ratio

        # Volatility difference
                        vol_diff = returns_1 - returns_2
        if vol_diff.var() > 1e-12:
                            features[f"volatility_diff_{tf1}m_{tf2}m"] = vol_diff

        # Volatility std (only if we have enough data)
        if len(returns_1) >= 20:
                            vol_std = (returns_1 - returns_2).rolling(20, min_periods=10).std()
        if vol_std.var() > 1e-12:
                                features[f"volatility_std_{tf1}m_{tf2}m"] = vol_std

        # 3. Cross-timeframe volume features (with validation)
        if volume_data is not None and isinstance(volume_data, pd.DataFrame) and not volume_data.empty and "volume" in volume_data.columns:
                volume = volume_data["volume"].astype(float)
        if volume.var() > 1e-12:  # Only if volume has meaningful variance
        for i, tf1 in enumerate(timeframes[:3]):
        for tf2 in timeframes[i+1:4]:
        if tf1 < len(volume) and tf2 < len(volume):
                                vol_1, volume.rolling(tf1, min_periods=tf1//2).mean()
                                vol_2, volume.rolling(tf2, min_periods=tf2//2).mean()

        # Volume ratio with safety check
                                vol_ratio = vol_1 / (vol_2 + 1e-8)
        if vol_ratio.var() > 1e-12:
                                    features[f"volume_ratio_{tf1}m_{tf2}m"] = vol_ratio

        # Volume difference
                                vol_diff = vol_1 - vol_2
        if vol_diff.var() > 1e-12:
                                    features[f"volume_diff_{tf1}m_{tf2}m"] = vol_diff

        # Volume momentum
                                vol_momentum = volume.pct_change(tf1) - volume.pct_change(tf2)
        if vol_momentum.var() > 1e-12:
                                    features[f"volume_momentum_{tf1}m_{tf2}m"] = vol_momentum

        # 4. Cross-timeframe price range features (with validation)
        for i, tf1 in enumerate(timeframes[:3]):
        for tf2 in timeframes[i+1:4]:
        if tf1 < len(close) and tf2 < len(close):
                        range_1 = (high.rolling(tf1, min_periods=tf1//2).max() - low.rolling(tf1, min_periods=tf1//2).min()) / (close.rolling(tf1, min_periods=tf1//2).mean() + 1e-8)
                        range_2 = (high.rolling(tf2, min_periods=tf2//2).max() - low.rolling(tf2, min_periods=tf2//2).min()) / (close.rolling(tf2, min_periods=tf2//2).mean() + 1e-8)

        # Price range ratio with safety check
                        range_ratio = range_1 / (range_2 + 1e-8)
        if range_ratio.var() > 1e-12:
                            features[f"price_range_ratio_{tf1}m_{tf2}m"] = range_ratio

        # Price range difference
                        range_diff = range_1 - range_2
        if range_diff.var() > 1e-12:
                            features[f"price_range_diff_{tf1}m_{tf2}m"] = range_diff

        # 5. Cross-timeframe RSI features (with validation)
        for tf1 in [3, 5, 10, 14]:
        for tf2 in [5, 10, 14, 20]:
        if tf1 < tf2 and tf1 < len(close) and tf2 < len(close):
                        rsi_1, self._calculate_rsi(close, tf1)
                        rsi_2, self._calculate_rsi(close, tf2)

        # RSI difference
                        rsi_diff = rsi_1 - rsi_2
        if rsi_diff.var() > 1e-12:
                            features[f"rsi_diff_{tf1}m_{tf2}m"] = rsi_diff

        # RSI ratio with safety check
                        rsi_ratio = rsi_1 / (rsi_2 + 1e-8)
        if rsi_ratio.var() > 1e-12:
                            features[f"rsi_ratio_{tf1}m_{tf2}m"] = rsi_ratio

        # 6. Cross-timeframe MACD features (with validation)
        for fast in [3, 5, 8]:
        for slow in [10, 15, 20]:
        if fast < slow and fast < len(close) and slow < len(close):
                        macd_1, self._calculate_macd(close, fast, slow)
                        macd_2, self._calculate_macd(close, fast*2, slow*2)

        # MACD difference
                        macd_diff = macd_1 - macd_2
        if macd_diff.var() > 1e-12:
                            features[f"macd_diff_{fast}_{slow}"] = macd_diff

        # MACD ratio with safety check
                        macd_ratio = macd_1 / (macd_2 + 1e-8)
        if macd_ratio.var() > 1e-12:
                            features[f"macd_ratio_{fast}_{slow}"] = macd_ratio

        # 7. Cross-timeframe Bollinger Bands features (with validation)
        for window in [10, 15, 20]:
        for std in [1, 1.5, 2]:
        if window < len(close):
                        bb_1, self._calculate_bollinger_bands(close, window, std)
                        bb_2, self._calculate_bollinger_bands(close, window*2, std)

        if bb_1 is not None and bb_2 is not None:
                            bb_diff = bb_1 - bb_2
        if bb_diff.var() > 1e-12:
                                features[f"bb_position_diff_{window}_{std}"] = bb_diff

        # 8. Cross-timeframe stochastic features (with validation)
        for k_period in [5, 10, 14]:
        for d_period in [3, 5, 7]:
        if k_period < len(close) and d_period < len(close):
                        stoch_1, self._calculate_stochastic(high, low, close, k_period, d_period)
                        stoch_2, self._calculate_stochastic(high, low, close, k_period*2, d_period*2)

        if stoch_1 is not None and stoch_2 is not None:
                            stoch_diff = stoch_1 - stoch_2
        if stoch_diff.var() > 1e-12:
                                features[f"stoch_diff_{k_period}_{d_period}"] = stoch_diff

        # Final validation of all generated features
            valid_features = {}
        for name, feature in features.items():
        if isinstance(feature, pd.Series):
        # Check for meaningful variance
        if feature.var() > 1e-12 and not feature.isna().all():
                        valid_features[name] = feature
                    else:
        self.logger.debug(f"⚠️ Skipping constant cross-timeframe feature: {name}")
                else:
                    valid_features[name] = feature

        self.logger.info(f"✅ Generated {len(valid_features)} valid cross-timeframe features")
        return valid_features

        except Exception as e:
        self.logger.exception(f"🚨 Error generating cross-timeframe features: {e}")
        return {}

    def _calculate_rsi(self, close: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI for a given period."""
        try:
            delta = close.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period, min_periods=1).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period, min_periods=1).mean()
            rs = gain / (loss + 1e-8)
            rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(50)
        except Exception as e:
        self.logger.exception(f"🚨 Error calculating RSI: {e}")
        return pd.Series(50, index=close.index)

    def _calculate_macd(self, close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.Series:
        """Calculate MACD for given fast and slow periods."""
        try: ema_fast = close.ewm(span=fast, adjust=False).mean()
            ema_slow, close.ewm(span=slow, adjust=False).mean()
        return ema_fast - ema_slow
        except Exception as e:
        self.logger.exception(f"🚨 Error calculating MACD: {e}")
        return pd.Series(0, index=close.index)

    def _calculate_bollinger_bands(self, close: pd.Series, window: int, std: float) -> pd.Series | None:
        """Calculate Bollinger Bands position."""
        try: sma = close.rolling(window=window).mean()
            std_dev, close.rolling(window=window).std()
            upper_band = sma + (std_dev * std)
            lower_band = sma - (std_dev * std)
        return (close - lower_band) / (upper_band - lower_band + 1e-8)
        except Exception:
        return None

    def _calculate_stochastic(self, high: pd.Series, low: pd.Series, close: pd.Series, k_period: int, d_period: int) -> pd.Series | None:
        """Calculate Stochastic oscillator."""
        try: lowest_low = low.rolling(window=k_period).min()
            highest_high, high.rolling(window=k_period).max()
            k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low + 1e-8))
        return k_percent.rolling(window=d_period).mean()
        except Exception:
        return None

    async def _generate_regime_aware_features(self, price_data: pd.DataFrame, volume_data: pd.DataFrame | None, None) -> dict[str, Any]:
        """Generate regime-aware features if HMM data is available."""
        try:
            features = {}

        # Try to load HMM regime data
        try:
                import glob

        # Look for HMM regime files
                hmm_files = glob.glob("data/hmm_regimes/*_composite_clusters.parquet")
        if hmm_files:
        # Load the most recent HMM data
                    hmm_data = pd.read_parquet(hmm_files[-1])
        if "composite_cluster_id" in hmm_data.columns:
        # Align HMM data with price data
        if len(hmm_data) == len(price_data):
                            cluster_ids = hmm_data["composite_cluster_id"].values

        # Generate regime-aware features
                            features["regime_cluster_id"] = cluster_ids
                            features["regime_stability"] = self._calculate_regime_stability(cluster_ids)
                            features["regime_transition"] = self._calculate_regime_transitions(cluster_ids)

        self.logger.debug(f"✅ Generated {len(features)} regime-aware features")
                        else:
        self.logger.warning("⚠️ HMM data length doesn't match price data length")
                else: self.logger.debug("ℹ️ No HMM regime data found = skipping regime-aware features")

        except Exception as e:
        self.logger.debug(f"ℹ️ Could not load HMM regime data: {e}")

        return features

        except Exception as e:
        self.logger.exception(f"🚨 Error generating regime-aware features: {e}")
        return {}

    def _validate_and_clean_features(self, features: dict[str, Any]) -> dict[str, Any]:
        """Validate and clean generated features."""
        try:
            cleaned_features = {}
            duplicate_count = 0
            constant_count = 0

        for feature_name, feature_value in features.items():
        try: if isinstance(feature_value = pd.Series):
        # Check for excessive NaN values (more lenient threshold)
                        nan_ratio = feature_value.isna().sum() / len(feature_value)
        if nan_ratio > 0.8:  # More than 80% NaN (increased from 50%)
        self.logger.warning(f"⚠️ Skipping feature {feature_name} with {nan_ratio:.2%} NaN values")
                            continue

        # Fill remaining NaN values
                        feature_value, feature_value.fillna(method="ffill").fillna(method="bfill").fillna(0)

        # Check for infinite values
        if np.isinf(feature_value).any():
                            feature_value, feature_value.replace([np.inf, -np.inf], 0)

        # Check for zero variance (constant features) - BUG DETECTION
        if feature_value.var() == 0:
                            constant_count += 1
        self.logger.warning(f"🚨 BUG: Constant feature detected: {feature_name}")
        self.logger.warning(f"🚨 BUG: All values: {feature_value.iloc[:5].tolist()}... (first 5)")
        self.logger.warning(f"🚨 BUG: Unique values: {feature_value.nunique()}")
        self.logger.warning(f"🚨 BUG: Min/Max: {feature_value.min()}/{feature_value.max()}")

        # Skip constant features (they indicate calculation bugs)
                            continue

        # Check for near-constant features (very low variance) - more lenient
        if feature_value.var() < 1e-10:  # Increased threshold from 1e-12
                            constant_count += 1
        self.logger.warning(f"🚨 BUG: Near-constant feature detected: {feature_name} (variance: {feature_value.var()})")

        # Skip near-constant features
                            continue

        # Check for duplicate features (same name already exists)
        if feature_name in cleaned_features:
                            duplicate_count += 1
        self.logger.debug(f"⚠️ Skipping duplicate feature {feature_name}")
                            continue

                        cleaned_features[feature_name] = feature_value
                    else:
        # Check for duplicate features (same name already exists)
        if feature_name in cleaned_features:
                            duplicate_count += 1
        self.logger.debug(f"⚠️ Skipping duplicate feature {feature_name}")
                            continue

                        cleaned_features[feature_name] = feature_value

        except Exception as e:
        self.logger.warning(f"⚠️ Error cleaning feature {feature_name}: {e}")
                    continue

        if duplicate_count > 0:
        self.logger.info(f"🔍 Removed {duplicate_count} duplicate features during cleaning")

        if constant_count > 0:
        self.logger.warning(f"🚨 BUG SUMMARY: Removed {constant_count} constant features due to calculation bugs")

        self.logger.debug(f"✅ Cleaned {len(cleaned_features)} features")
        return cleaned_features

        except Exception as e:
        self.logger.exception(f"🚨 Error validating and cleaning features: {e}")
        return features

    def _ensure_pickle_safe_features(self, features: dict[str, Any]) -> dict[str, Any]:
        """Ensure features are pickle-safe by removing any async objects or coroutines."""
        try:
            safe_features = {}
            removed_count = 0

        for feature_name, feature_value in features.items():
        try:
        # Check if the value is a coroutine or async object
        if hasattr(feature_value, "__await__") or hasattr(feature_value, "send"):
        self.logger.warning(f"⚠️ Removing async object from feature {feature_name}")
                        removed_count += 1
                        continue

        # Check if the value contains any async objects
        if isinstance(feature_value, list | tuple | dict):
        # For now, just skip complex objects that might contain async objects
        if any(hasattr(item, "__await__") for item in feature_value if hasattr(feature_value, "__iter__"):
        self.logger.warning(f"⚠️ Removing feature {feature_name} containing async objects")
                            removed_count += 1
                            continue

                    safe_features[feature_name] = feature_value

        except Exception as e:
        self.logger.warning(f"⚠️ Error checking pickle safety for feature {feature_name}: {e}")
                    continue

        if removed_count > 0:
        self.logger.info(f"🔍 Removed {removed_count} non-pickle-safe features")

        return safe_features

        except Exception as e:
        self.logger.exception(f"🚨 Error ensuring pickle safety: {e}")
        return features

    def _generate_fallback_features(self, price_data: pd.DataFrame, volume_data: pd.DataFrame) -> dict[str, Any]:
        """Generate fallback features when main feature engineering fails."""
        try:
            features = {}

        if price_data.empty:
        return features

            close = price_data["close"].astype(float)

        # Basic price features
            features["close_returns"] = close.pct_change().fillna(0)
            features["close_ma_5"] = close.rolling(5, min_periods=1).mean()
            features["close_ma_20"] = close.rolling(20, min_periods=1).mean()
            features["price_momentum"] = close.pct_change(5).fillna(0)
            features["volatility"] = close.pct_change().rolling(20).std().fillna(0)

        # Basic volume features
        if volume_data is not None and not volume_data.empty and "volume" in volume_data.columns:
                volume = volume_data["volume"].astype(float)
                features["volume_ma_5"] = volume.rolling(5, min_periods=1).mean()
                features["volume_ma_20"] = volume.rolling(20, min_periods=1).mean()
                features["volume_ratio"] = volume / (volume.rolling(20, min_periods=1).mean() + 1e-8)

        self.logger.info(f"✅ Generated {len(features)} fallback features")
        return features

        except Exception as e:
        self.logger.exception(f"🚨 Error generating fallback features: {e}")
        return {}

    def _calculate_regime_stability(self, cluster_ids: np.ndarray) -> np.ndarray:
        """Calculate regime stability based on cluster transitions."""
        try:
            stability = np.zeros(len(cluster_ids))

        for i in range(1, len(cluster_ids):
        # Count how many times the regime changes in a window
                window_size, min(20, i)
                recent_clusters, cluster_ids[max(0, i-window_size):i+1]
                unique_clusters = len(np.unique(recent_clusters))
                stability[i] = 1.0 / (unique_clusters + 1)  # Higher stability, fewer unique clusters

        return stability

        except Exception as e:
        self.logger.exception(f"🚨 Error calculating regime stability: {e}")
        return np.zeros(len(cluster_ids))

    def _calculate_regime_transitions(self, cluster_ids: np.ndarray) -> np.ndarray:
        """Calculate regime transition indicators."""
        try:
            transitions = np.zeros(len(cluster_ids))

        for i in range(1, len(cluster_ids):
        if cluster_ids[i] != cluster_ids[i-1]:
                    transitions[i] = 1.0

        return transitions

        except Exception as e:
        self.logger.exception(f"🚨 Error calculating regime transitions: {e}")
        return np.zeros(len(cluster_ids))





    async def _generate_meta_labels_vectorized(
        self, price_data: pd.DataFrame, volume_data: pd.DataFrame, order_flow_data: pd.DataFrame | None = None
    ) -> dict[str, Any]:
        """Generate meta-labels for ensemble learning."""
        try:
            features = {}

        # Simple meta-label based on price momentum
        if "close" in price_data.columns:
                close = price_data["close"].astype(float)
                returns, close.pct_change().fillna(0)

        # Volatility regime
                vol_20, returns.rolling(20, min_periods=1).std()
                volatility_categories, pd.cut(vol_20, bins=3, labels=[0, 1, 2])
                features["volatility_regime"] = pd.Series(volatility_categories.astype(float), index=price_data.index)

        # Trend regime
                sma_20, close.rolling(20, min_periods=1).mean()
                sma_50, close.rolling(50, min_periods=1).mean()
                features["trend_regime"] = (sma_20 > sma_50).astype(int)

        return features

        except Exception as e:
        self.logger.exception(f"🚨 Error generating meta-labels: {e}")
        return {}

    async def _generate_explicit_meta_labels_vectorized(
        self, price_data: pd.DataFrame, volume_data: pd.DataFrame, timeframe: str = "1m"
    ) -> dict[str, Any]:
        """Generate explicit meta-labels for specific timeframes."""
        try:
            features = {}

        # Simple explicit meta-labels
        if "close" in price_data.columns:
                close = price_data["close"].astype(float)

        # Price position relative to recent range
                high_20, close.rolling(20, min_periods=1).max()
                low_20, close.rolling(20, min_periods=1).min()
                features[f"price_position_{timeframe}"] = (close - low_20) / (
                    high_20 - low_20
                )

        # Momentum strength
                returns, close.pct_change().fillna(0)
                features[f"momentum_strength_{timeframe}"] = returns.rolling(
                    10, min_periods=1
                ).mean()

        return features

        except Exception as e:
        self.logger.exception(f"🚨 Error generating explicit meta-labels: {e}")
        return {}

    @validate_step_prerequisites(
        required_directories=["data_cache", "data/feature_cache"]
        min_memory_gb=8.0
        min_disk_gb=5.0
        required_packages=["pandas", "numpy"]
        data_quality_checks={
            "min_rows": 100,
            "required_columns": ["timestamp", "open", "high", "low", "close", "volume"],
        },
        context="Difference and Acceleration Feature Engineering"
    )
    @secure_data_processing(
        backup_before=True
        integrity_checks=True
        memory_cleanup=True
        data_validation=True
    )
    @prevent_data_leakage(
        temporal_validation=True
        feature_leakage_detection=True
        cross_validation_isolation=True
        lookahead_bias_prevention=True
    )
    @resource_monitor(
        memory_threshold_gb=16.0
        cpu_threshold_percent=90.0
        disk_threshold_gb=10.0
        monitor_interval=30.0
        auto_cleanup=True
    )
    @memory_efficient(
        chunk_size=5000
        streaming_processing=True
        memory_pool=True
        cleanup_frequency=20
    )
    @debug_training_step(
        log_intermediate_results=True
        save_debug_artifacts=True
        performance_profiling=True
        error_context_preservation=True
    )
    @circuit_breaker_protection(
        failure_threshold=3
        recovery_timeout=300.0
        expected_exception=Exception
        monitor_interval=60.0
    )
    @validate_step_output(
        required_files=["data/feature_cache/*.parquet"]
        data_quality_checks={
            "min_rows": 50,
            "required_columns": ["features"],
        },
        performance_thresholds={
            "feature_engineering_time_minutes": 30.0,
            "memory_usage_gb": 8.0,
        },
        format_validation=True
    )
    @quality_gate(
        model_performance_thresholds={
            "feature_quality": 0.8,
            "feature_completeness": 0.9,
        },
        data_quality_metrics={"completeness": 0.9, "consistency": 0.8}
        convergence_checks=True
        overfitting_detection=True
        validation_score_requirements={"feature_engineering_score": 0.8}
    )
    @handle_errors(
        exceptions=(ValueError, AttributeError, MemoryError)
        default_return={}
        context="difference and acceleration feature engineering"
    )
    async def _engineer_difference_and_acceleration_features(
        self = features: dict[str, Any], price_data: pd.DataFrame, ) -> dict[str, Any]:
        """Engineer difference and acceleration features with proper normalization and interaction features.

        Args:
            features: Dictionary of existing features
            price_data: OHLCV price data for reference

        Returns:
            Dictionary containing enhanced features with differences and accelerations

        """
        try:
            enhanced_features = {}

        # Define lookback periods for different timeframes

        # Features that benefit from difference calculations
            difference_candidates = {
        # RSI features
                "rsi": {"priority": "high", "timeframes": ["1m", "5m", "15m", "30m"]},
                "rsi_20": {"priority": "high", "timeframes": ["1m", "5m", "15m", "30m"]},
                "adaptive_rsi": {"priority": "high", "timeframes": ["1m", "5m", "15m", "30m"]},

        # MACD features
                "macd": {"priority": "high", "timeframes": ["5m", "15m", "30m"]},
                "macd_signal": {"priority": "high", "timeframes": ["5m", "15m", "30m"]},
                "macd_histogram": {"priority": "high", "timeframes": ["5m", "15m", "30m"]},

        # Bollinger Bands
                "bb_position": {"priority": "medium", "timeframes": ["1m", "5m", "15m", "30m"]},
                "bb_zscore_20": {"priority": "medium", "timeframes": ["1m", "5m", "15m", "30m"]},

        # Price momentum
                "price_momentum_5": {"priority": "high", "timeframes": ["1m", "5m", "15m"]},
                "price_momentum_20": {"priority": "high", "timeframes": ["15m", "30m"]},
                "volume_weighted_momentum_5": {"priority": "high", "timeframes": ["1m", "5m", "15m"]},
                "volume_weighted_momentum_10": {"priority": "high", "timeframes": ["5m", "15m", "30m"]},

        # Volume features
                "volume_momentum": {"priority": "high", "timeframes": ["1m", "5m", "15m"]},
                "volume_ma_5": {"priority": "medium", "timeframes": ["1m", "5m", "15m"]},
                "volume_ma_20": {"priority": "medium", "timeframes": ["5m", "15m", "30m"]},

        # Volatility features
                "volatility_5": {"priority": "high", "timeframes": ["1m", "5m", "15m"]},
                "volatility_20": {"priority": "high", "timeframes": ["15m", "30m"]},
                "volatility_persistence": {"priority": "medium", "timeframes": ["15m", "30m"]},
                "volatility_of_volatility": {"priority": "medium", "timeframes": ["15m", "30m"]},

        # Technical indicators
                "cci": {"priority": "medium", "timeframes": ["5m", "15m", "30m"]},
                "roc": {"priority": "high", "timeframes": ["5m", "15m", "30m"]},
                "mfi": {"priority": "medium", "timeframes": ["5m", "15m", "30m"]},

        # Microstructure features
                "order_flow_imbalance": {"priority": "high", "timeframes": ["1m", "5m", "15m"]},

        # Adaptive features
                "adaptive_sma": {"priority": "medium", "timeframes": ["15m", "30m"]},
                "adaptive_sma_slope": {"priority": "medium", "timeframes": ["15m", "30m"]},

        # Moving averages
                "sma_5": {"priority": "low", "timeframes": ["1m", "5m", "15m"]},
                "sma_20": {"priority": "low", "timeframes": ["5m", "15m", "30m"]},
                "ema_12": {"priority": "low", "timeframes": ["1m", "5m", "15m"]},
                "ema20_slope": {"priority": "medium", "timeframes": ["5m", "15m", "30m"]},
            }

        # Features that benefit from acceleration (second difference)
            acceleration_candidates = {
                "rsi": {"priority": "high", "timeframes": ["5m", "15m", "30m"]},
                "rsi_14": {"priority": "high", "timeframes": ["5m", "15m", "30m"]},
                "macd": {"priority": "high", "timeframes": ["15m", "30m"]},
                "macd_signal": {"priority": "high", "timeframes": ["15m", "30m"]},
                "price_momentum_5": {"priority": "high", "timeframes": ["1m", "5m", "15m"]},
                "price_momentum_20": {"priority": "high", "timeframes": ["15m", "30m"]},
                "volume_momentum": {"priority": "high", "timeframes": ["1m", "5m", "15m"]},
                "volatility_20": {"priority": "high", "timeframes": ["15m", "30m"]},
                "bb_position": {"priority": "medium", "timeframes": ["5m", "15m", "30m"]},
                "stoch_k": {"priority": "medium", "timeframes": ["1m", "5m", "15m"]},
            }

        # Exclude features that are already difference-based or should be treated as data
            exclude_features = {
                "close_returns", "price_impact", "bid_ask_spread_returns",
                "market_depth_change", "market_depth_returns", "volume_ratio_change",
                "funding_rate_change", "trade_count_change", "trade_volume_change",
                "nearest_bid_wall_size_change", "nearest_ask_wall_size_change",
                "weighted_mid_price_change", "trade_to_order_ratio",
            }

        # Process difference features
        for feature_name, feature_value in features.items():
        if feature_name in exclude_features:
                    continue

        if feature_name not in difference_candidates:
                    continue

        if not isinstance(feature_value, pd.Series | np.ndarray | list):
                    continue

        # Convert to pandas Series for processing
        if isinstance(feature_value, np.ndarray | list):
                    feature_series, pd.Series(feature_value, index=price_data.index)
                else:
                    feature_series = feature_value

        # Get candidate info
                candidate_info = difference_candidates[feature_name]
                priority = candidate_info["priority"]
                candidate_info["timeframes"]

        # Select lookback periods based on priority (tightened to reduce feature count)
        if priority == "high":
                    periods = [1, 3, 5]
                elif priority == "medium":
                    periods = [1, 3]
                else:  # low priority
                    periods = [1]

        # Generate difference features
        for period in periods:
        if len(feature_series) > period:
        # Calculate difference
                        diff_series = feature_series.diff(period)

        # Handle NaN values - fill with 0 for "no change"
                        diff_series = diff_series.fillna(0)

        # Normalize with rolling Z-score for consistent scale
                        diff_normalized, self._normalize_with_rolling_zscore(diff_series, window=20)

        # Store both raw and normalized differences
                        enhanced_features[f"{feature_name}_diff_{period}"] = diff_series
                        enhanced_features[f"{feature_name}_diff_{period}_norm"] = diff_normalized

        # Generate acceleration features for high-priority candidates
        if feature_name in acceleration_candidates and priority in ["high", "medium"]:
        if period in (1, 3) and len(diff_series) > 1:
        # Calculate acceleration (second difference) for limited periods only
                                accel_series = diff_series.diff(1).fillna(0)
                                accel_normalized, self._normalize_with_rolling_zscore(accel_series, window=20)
                                enhanced_features[f"{feature_name}_accel_{period}"] = accel_series
                                enhanced_features[f"{feature_name}_accel_{period}_norm"] = accel_normalized

        # Validate that enhanced_features doesn't contain coroutines before generating interactions
            valid_enhanced_features = {}
        for key, value in enhanced_features.items():
        if hasattr(value, "__await__"):
        self.logger.warning(f"⚠️ Skipping coroutine feature in enhanced_features: {key}")
                    continue
                valid_enhanced_features[key] = value

        # Generate interaction features for high-priority combinations
        try: interaction_features = await self._generate_interaction_features(valid_enhanced_features, features, price_data)
        if isinstance(interaction_features, dict):
        # Filter out any coroutine features from interaction_features before updating
                    valid_interaction_features = {}
                    coroutine_count = 0
        for key, value in interaction_features.items():
        if hasattr(value, "__await__"):
        self.logger.warning(f"⚠️ Skipping coroutine feature from interaction generation: {key}")
                            coroutine_count += 1
                            continue
                        valid_interaction_features[key] = value

        if coroutine_count > 0:
        self.logger.info(f"⚠️ Filtered out {coroutine_count} coroutine features from interaction generation")

                    enhanced_features.update(valid_interaction_features)
                else:
        self.logger.warning(f"⚠️ Interaction features not a dict: {type(interaction_features)}")
        except Exception as e:
        self.logger.warning(f"⚠️ Failed to generate interaction features: {e}")
                interaction_features = {}

        # Validate that features doesn't contain coroutines before generating cross-timeframe features
            valid_features = {}
        for key, value in features.items():
        if hasattr(value, "__await__"):
        self.logger.warning(f"⚠️ Skipping coroutine feature in features: {key}")
                    continue
                valid_features[key] = value

        # Generate cross-timeframe difference features
            cross_timeframe_features, await self._generate_cross_timeframe_features(valid_features, price_data)
        if isinstance(cross_timeframe_features, dict):
        # Filter out any coroutine features from cross_timeframe_features before updating
                valid_cross_timeframe_features = {}
                coroutine_count = 0
        for key, value in cross_timeframe_features.items():
        if hasattr(value, "__await__"):
        self.logger.warning(f"⚠️ Skipping coroutine feature from cross-timeframe generation: {key}")
                        coroutine_count += 1
                        continue
                    valid_cross_timeframe_features[key] = value

        if coroutine_count > 0:
        self.logger.info(f"⚠️ Filtered out {coroutine_count} coroutine features from cross-timeframe generation")

                enhanced_features.update(valid_cross_timeframe_features)
            else:
        self.logger.warning(f"⚠️ Cross-timeframe features not a dict: {type(cross_timeframe_features)}")

        # Final validation: ensure no coroutine features in the final output
            final_enhanced_features = {}
            coroutine_count = 0
        for key, value in enhanced_features.items():
        if hasattr(value, "__await__"):
        self.logger.warning(f"⚠️ Skipping coroutine feature in final output: {key}")
                    coroutine_count += 1
                    continue
                final_enhanced_features[key] = value

        if coroutine_count > 0:
        self.logger.info(f"⚠️ Filtered out {coroutine_count} coroutine features from final output")

        # Apply caps to control feature explosion
        try:
                pre_total = len(final_enhanced_features)
        # Identify RAW-only keys in each category (normalized variants handled separately)
                accel_raw = [k for k in final_enhanced_features if "_accel_" in k and not k.endswith("_norm")]
        # Cross-timeframe raw diff features (e.g., rsi_diff_5m_1m)
                cross_time_raw = [
                    k for k in final_enhanced_features
        if "_diff_" in k and not k.endswith("_norm") and ("m_" in k or "h_" in k)
                ]
        # Non-cross-time raw difference features (exclude acceleration)
                diff_raw = [
                    k for k in final_enhanced_features
        if "_diff_" in k and not k.endswith("_norm") and "_accel_" not in k and not ("m_" in k or "h_" in k)
                ]

        # Priority patterns (keep strongest first)
                accel_priority = [
                    "rsi_accel", "macd_histogram_accel", "macd_accel", "price_momentum_", "volatility_20_accel",
                ]
                diff_priority = [
                    "rsi_diff_", "macd_histogram_diff_", "macd_diff_", "price_momentum_", "volume_momentum",
                    "volatility_20_diff_", "roc_diff_", "cci_diff_", "bb_position_diff_", "order_flow_imbalance_diff_",
                ]
                cross_priority = [
                    "rsi_diff_", "volatility_diff_", "price_range_diff_", "momentum_", "volume_diff_",
                ]

                def rank_keys(keys, patterns):
                    def score(k: str) -> int:
        for idx, p in enumerate(patterns):
        if p in k:
        return idx
        return len(patterns) + 1
        return sorted(keys, key=score)

                accel_ranked, rank_keys(accel_raw, accel_priority)
                diff_ranked, rank_keys(diff_raw, diff_priority)
                cross_ranked, rank_keys(cross_time_raw, cross_priority)

        # Caps (tightened further to meet target totals)
                max_accel = 10   # ~20 with norms
                max_diff = 25    # ~50 with norms
                max_cross_time = 50  # ~100 with norms

                kept_accel_raw = set(accel_ranked[:max_accel])
                kept_cross_raw = set(cross_ranked[:max_cross_time])
                kept_diff_raw = set(diff_ranked[:max_diff])

        # Include normalized counterparts for kept raw keys (do not count against caps)
                kept_keys = set()
        for raw_key in list(kept_accel_raw) + list(kept_cross_raw) + list(kept_diff_raw):
                    kept_keys.add(raw_key)
                    norm_key = f"{raw_key}_norm"
        if norm_key in final_enhanced_features:
                        kept_keys.add(norm_key)

        # Rebuild final features with caps applied
                capped_features: dict[str = Any] = {}
        for k, v in final_enhanced_features.items():
        # Keep capped categories (raw+their norms)
        if k in kept_keys:
                        capped_features[k] = v
                        continue
        # Pass-through for non-targeted categories (e.g., interactions) untouched
                    is_accel = "_accel_" in k
                    is_diff = "_diff_" in k
                    is_cross = is_diff and ("m_" in k or "h_" in k)
        # If not accel/diff/cross-timeframe, keep
        if not is_accel and not is_diff and not is_cross:
                        capped_features[k] = v

                post_total = len(capped_features)
        self.logger.info(
                    f"🔧 Applied feature caps (pre={pre_total}, post={post_total}): "
                    f"accel<=%d, diff<=%d, cross-time<=%d (priority-aware)" % (max_accel, max_diff, max_cross_time)
                )
        except Exception as cap_e:
        self.logger.warning(f"⚠️ Failed to apply feature caps: {cap_e}")
                capped_features = final_enhanced_features

        self.logger.info(f"✅ Generated {len(capped_features)} difference and acceleration features (after caps)")
        return capped_features

        except Exception as e:
        self.logger.exception(f"🚨 Error engineering difference and acceleration features: {e}")
        return {}

    @handle_errors(
        exceptions=(ValueError, AttributeError)
        default_return=pd.Series()
        context="rolling z-score normalization"
    )
    @memory_efficient(
        chunk_size=1000
        streaming_processing=False
        memory_pool=True
        cleanup_frequency=10
    )
    def _normalize_with_rolling_zscore(self, series: pd.Series, window: int = 20) -> pd.Series:
        """Normalize series using rolling Z-score to ensure consistent scale.

        Args:
            series: Input series to normalize
            window: Rolling window size for Z-score calculation

        Returns:
            Normalized series

        """
        try:
        if len(series) < window:
        return series

        # Calculate rolling mean and std
            rolling_mean, series.rolling(window=window, min_periods=1).mean()
            rolling_std, series.rolling(window=window, min_periods=1).std()

        # Avoid division by zero
            rolling_std, rolling_std.replace(0, 1)

        # Calculate Z-score
            z_score = (series - rolling_mean) / rolling_std

        # Clip extreme values to prevent outliers from dominating
            z_score, z_score.clip(-3, 3)

        # Fill NaN values
        return z_score.fillna(0)


        except Exception as e:
        self.logger.exception(f"🚨 Error in rolling Z-score normalization: {e}")
        return series.fillna(0)

    @handle_errors(
        exceptions=(ValueError, AttributeError, MemoryError)
        default_return={}
        context="interaction feature generation"
    )
    @memory_efficient(
        chunk_size=2000
        streaming_processing=True
        memory_pool=True
        cleanup_frequency=15
    )
    @debug_training_step(
        log_intermediate_results=True
        save_debug_artifacts=False
        performance_profiling=True
        error_context_preservation=True
    )
    async def _generate_interaction_features(
        self = enhanced_features: dict[str, Any], original_features: dict[str, Any], price_data: pd.DataFrame, ) -> dict[str, Any]:
        """Generate comprehensive interaction features between difference/acceleration features.

        Args:
            enhanced_features: Dictionary of difference/acceleration features
            original_features: Dictionary of original features

        Returns:
            Dictionary containing interaction features

        """
        try:
            interaction_features = {}

        # Validate that enhanced_features contains actual data, not coroutines
        if not isinstance(enhanced_features, dict):
        self.logger.error(f"🚨 Enhanced features is not a dict: {type(enhanced_features)}")
        return {}

        # Filter out any coroutine objects from enhanced_features
            valid_features = {}
        for key, value in enhanced_features.items():
        if hasattr(value, "__await__"):
        self.logger.warning(f"⚠️ Skipping coroutine feature: {key}")
                    continue
                valid_features[key] = value

        if not valid_features:
        self.logger.warning("⚠️ No valid features found for interaction generation")
        return {}

        # Get all available feature names from valid features only
            all_feature_names = list(valid_features.keys())

        # Define feature categories for intelligent interaction generation
            momentum_features = [f for f in all_feature_names if "momentum" in f.lower()]
            volume_features = [f for f in all_feature_names if "volume" in f.lower()]
            volatility_features = [f for f in all_feature_names if "volatility" in f.lower()]
            rsi_features = [f for f in all_feature_names if "rsi" in f.lower()]
            macd_features = [f for f in all_feature_names if "macd" in f.lower()]
            bb_features = [f for f in all_feature_names if "bb" in f.lower() or "bollinger" in f.lower()]
            stoch_features = [f for f in all_feature_names if "stoch" in f.lower()]
            [f for f in all_feature_names if "price" in f.lower()]
            diff_features = [f for f in all_feature_names if "diff" in f.lower()]
            accel_features = [f for f in all_feature_names if "accel" in f.lower()]

        # 1. High-value interaction combinations (50+ features)
            high_value_combinations = [
        # RSI + Volume interactions (10+ features)
                *[(f1, f2) for f1 in rsi_features[:3] for f2 in volume_features[:3]],

        # Price momentum + Volume interactions (15+ features)
                *[(f1, f2) for f1 in momentum_features[:5] for f2 in volume_features[:3]],

        # MACD + Volume interactions (8+ features)
                *[(f1, f2) for f1 in macd_features[:2] for f2 in volume_features[:4]],

        # Volatility + Volume interactions (10+ features)
                *[(f1, f2) for f1 in volatility_features[:5] for f2 in volume_features[:2]],

        # RSI + Price momentum interactions (12+ features)
                *[(f1, f2) for f1 in rsi_features[:3] for f2 in momentum_features[:4]],

        # MACD + Price momentum interactions (8+ features)
                *[(f1, f2) for f1 in macd_features[:2] for f2 in momentum_features[:4]],

        # Bollinger Bands + Volume interactions (6+ features)
                *[(f1, f2) for f1 in bb_features[:3] for f2 in volume_features[:2]],

        # Stochastic + Volume interactions (6+ features)
                *[(f1, f2) for f1 in stoch_features[:3] for f2 in volume_features[:2]],

        # Difference + Acceleration interactions (20+ features)
                *[(f1, f2) for f1 in diff_features[:5] for f2 in accel_features[:4]],
            ]

        # 2. Cross-timeframe interaction combinations (30+ features)
            cross_timeframe_combinations = []
        for tf1 in ["5m", "15m", "30m"]:
        for tf2 in ["15m", "30m"]:
        if tf1 != tf2:
        # Find features for each timeframe
                        tf1_features = [f for f in all_feature_names if tf1 in f]
                        tf2_features = [f for f in all_feature_names if tf2 in f]

        # Create cross-timeframe interactions
        for f1 in tf1_features[:2]:  # Limit to avoid too many combinations
        for f2 in tf2_features[:2]:
                                cross_timeframe_combinations.append((f1, f2))

        # 3. Polynomial interaction features (20+ features)
            polynomial_combinations = []
        for feature_name in all_feature_names[:10]:  # Use first 10 features for polynomial
        if feature_name in valid_features:
                    polynomial_combinations.append((feature_name, feature_name))  # Self-interaction

        # 4. Volatility regime interactions (15+ features)
            volatility_regime_combinations = []
        for vol_feat in volatility_features[:5]:
        for other_feat in momentum_features[:3]:
                    volatility_regime_combinations.append((vol_feat, other_feat))

        # Combine all interaction combinations with stricter caps
            all_combinations = (
                high_value_combinations +
                cross_timeframe_combinations[:8] +  # further limited
                polynomial_combinations[:6] +
                volatility_regime_combinations[:6]
            )

        # Strict cap on total interaction pairs to control explosion
            MAX_INTERACTION_PAIRS = 5
            selected_combinations = all_combinations[:MAX_INTERACTION_PAIRS]

        # Generate interactions
        for feat1_name, feat2_name in selected_combinations:
        if feat1_name in valid_features and feat2_name in valid_features:
                    feat1 = valid_features[feat1_name]
                    feat2 = valid_features[feat2_name]

        # Additional validation to ensure features are not coroutines
        if hasattr(feat1, "__await__") or hasattr(feat2, "__await__"):
        self.logger.warning(f"⚠️ Skipping interaction for coroutine features: {feat1_name}, {feat2_name}")
                        continue

        # Convert to pandas Series if needed
        if isinstance(feat1, np.ndarray | list):
                        feat1_series, pd.Series(feat1, index=price_data.index)
                    else:
                        feat1_series = feat1

        if isinstance(feat2, np.ndarray | list):
                        feat2_series, pd.Series(feat2, index=price_data.index)
                    else:
                        feat2_series = feat2

        # Ensure same length
                    min_len, min(len(feat1_series), len(feat2_series))
        if min_len > 0:
                        feat1_series = feat1_series.iloc[-min_len:]
                        feat2_series = feat2_series.iloc[-min_len:]

        # Create multiple types of interactions

        # 1. Multiplication interaction
                        interaction_mult = feat1_series * feat2_series
                        interaction_name = f"{feat1_name}_x_{feat2_name}"
                        interaction_features[interaction_name] = interaction_mult
                        interaction_features[f"{interaction_name}_norm"] = self._normalize_with_rolling_zscore(interaction_mult, window=20)

        # 2. Division interaction (with safety check)
        if (feat2_series != 0).any():
                            interaction_div = feat1_series / (feat2_series + 1e-8)
                            interaction_features[f"{interaction_name}_div"] = interaction_div
                            interaction_features[f"{interaction_name}_div_norm"] = self._normalize_with_rolling_zscore(interaction_div, window=20)

        self.logger.info(f"✅ Generated {len(interaction_features)} comprehensive interaction features (capped to {MAX_INTERACTION_PAIRS} pairs)")
        return interaction_features

        except Exception as e:
        self.logger.exception(f"🚨 Error generating interaction features: {e}")
        return {}

    @handle_errors(
        exceptions=(ValueError, AttributeError, MemoryError)
        default_return={}
        context="cross-timeframe feature generation"
    )
    @memory_efficient(
        chunk_size=2000
        streaming_processing=True
        memory_pool=True
        cleanup_frequency=15
    )
    @debug_training_step(
        log_intermediate_results=True
        save_debug_artifacts=False
        performance_profiling=True
        error_context_preservation=True
    )
    async def _generate_cross_timeframe_features(
        self = features: dict[str, Any], price_data: pd.DataFrame, ) -> dict[str, Any]:
        """Generate cross-timeframe difference features.

        Args:
            features: Dictionary of original features
            price_data: OHLCV price data

        Returns:
            Dictionary containing cross-timeframe features

        """
        try:
            cross_timeframe_features = {}

        # Validate that features contains actual data, not coroutines
        if not isinstance(features, dict):
        self.logger.error(f"🚨 Features is not a dict: {type(features)}")
        return {}

        # Filter out any coroutine objects from features
            valid_features = {}
            coroutine_count = 0
        for key, value in features.items():
        if hasattr(value, "__await__"):
        self.logger.warning(f"⚠️ Skipping coroutine feature in cross-timeframe: {key}")
                    coroutine_count += 1
                    continue
                valid_features[key] = value

        if coroutine_count > 0:
        self.logger.info(f"⚠️ Filtered out {coroutine_count} coroutine features from cross-timeframe generation")

        if not valid_features:
        self.logger.warning("⚠️ No valid features found for cross-timeframe generation")
        return {}

        # For now, we'll create cross-timeframe features based on different lookback periods
        # simulating different timeframes with the same data

        # Define cross-timeframe combinations
            cross_combinations = [
        # RSI cross-timeframe differences
                ("rsi", 3, 1, "rsi_diff_3m_1m"),
                ("rsi", 5, 1, "rsi_diff_5m_1m"),
                ("rsi", 10, 3, "rsi_diff_10m_3m"),

        # Price momentum cross-timeframe differences
                ("price_momentum_5", 3, 1, "momentum_5_diff_3m_1m"),
                ("price_momentum_10", 5, 1, "momentum_10_diff_5m_1m"),
                ("price_momentum_20", 10, 3, "momentum_20_diff_10m_3m"),

        # Volume momentum cross-timeframe differences
                ("volume_momentum", 3, 1, "volume_momentum_diff_3m_1m"),
                ("volume_momentum", 5, 1, "volume_momentum_diff_5m_1m"),

        # Volatility cross-timeframe differences
                ("volatility_20", 10, 3, "volatility_20_diff_10m_3m"),
                ("volatility_10", 5, 1, "volatility_10_diff_5m_1m"),

        # MACD cross-timeframe differences
                ("macd", 5, 1, "macd_diff_5m_1m"),
                ("macd_signal", 5, 1, "macd_signal_diff_5m_1m"),
            ]

        for feature_name, long_period, short_period, output_name in cross_combinations:
        if feature_name in valid_features:
                    feature_value = valid_features[feature_name]

        # Additional validation to ensure feature is not a coroutine
        if hasattr(feature_value, "__await__"):
        self.logger.warning(f"⚠️ Skipping coroutine feature for cross-timeframe: {feature_name}")
                        continue

        if not isinstance(feature_value, pd.Series | np.ndarray | list):
                        continue

        # Convert to pandas Series
        if isinstance(feature_value, np.ndarray | list):
                        feature_series, pd.Series(feature_value, index=price_data.index)
                    else:
                        feature_series = feature_value

        if len(feature_series) > max(long_period, short_period):
        # Calculate differences at different periods
                        long_diff = feature_series.diff(long_period).fillna(0)
                        short_diff = feature_series.diff(short_period).fillna(0)

        # Cross-timeframe difference
                        cross_diff = long_diff - short_diff

        # Normalize
                        cross_diff_norm, self._normalize_with_rolling_zscore(cross_diff, window=20)

        # Additional validation to ensure normalized feature is not a coroutine
        if hasattr(cross_diff_norm, "__await__"):
        self.logger.warning(f"⚠️ Skipping coroutine normalized feature for cross-timeframe: {output_name}")
                            continue

        # Store features
                        cross_timeframe_features[output_name] = cross_diff
                        cross_timeframe_features[f"{output_name}_norm"] = cross_diff_norm

        # Final validation: ensure no coroutine features in the output
            final_cross_timeframe_features = {}
            coroutine_count = 0
        for key, value in cross_timeframe_features.items():
        if hasattr(value, "__await__"):
        self.logger.warning(f"⚠️ Skipping coroutine feature in final cross-timeframe output: {key}")
                    coroutine_count += 1
                    continue
                final_cross_timeframe_features[key] = value

        if coroutine_count > 0:
        self.logger.info(f"⚠️ Filtered out {coroutine_count} coroutine features from final cross-timeframe output")

        self.logger.info(f"✅ Generated {len(final_cross_timeframe_features)} cross-timeframe features")
        return final_cross_timeframe_features

        except Exception as e:
        self.logger.exception(f"🚨 Error generating cross-timeframe features: {e}")
        return {}

    def _validate_difference_engineering_inputs(
        self = features: dict[str, Any], price_data: pd.DataFrame, ) -> None:
        """Validate inputs before difference and acceleration feature engineering.

        Args:
            features: Dictionary of existing features
            price_data: OHLCV price data

        """
        try:
        # Validate price data
        if price_data.empty:
                msg = "Price data is empty"
                raise ValueError(msg)

            required_cols = ["open", "high", "low", "close", "volume"]
            missing_cols = [col for col in required_cols if col not in price_data.columns]
        if missing_cols:
                msg = f"Missing required columns in price data: {missing_cols}"
                raise ValueError(msg)

        # Validate features
        if not features:
                msg = "No features provided for enhancement"
                raise ValueError(msg)

        # Check for minimum data length
        if len(price_data) < 100:
                msg, f"Insufficient data length: {len(price_data)} < 100"
                raise ValueError(msg)

        # Validate feature types
        for feature_name, feature_value in features.items():
        if not isinstance(feature_value, pd.Series | np.ndarray | list):
        self.logger.warning(f"Feature {feature_name} is not a supported type: {type(feature_value)}")

        self.logger.info(f"✅ Validated {len(features)} features for difference engineering")

        except Exception as e:
        self.logger.exception(f"❌ Validation failed for difference engineering inputs: {e}")
            raise

    def _validate_enhanced_features(self, enhanced_features: dict[str, Any]) -> None:
        """Validate enhanced features before merging.

        Args:
            enhanced_features: Dictionary of enhanced features

        """
        try:
        if not enhanced_features:
        self.logger.warning("⚠️ No enhanced features generated")
                return

        # Filter out coroutine objects before validation
            valid_features = {}
        for key, value in enhanced_features.items():
        if hasattr(value, "__await__"):
        self.logger.warning(f"⚠️ Skipping coroutine feature in validation: {key}")
                    continue
                valid_features[key] = value

        # Count feature types
            diff_features = [f for f in valid_features if "_diff_" in f]
            accel_features = [f for f in valid_features if "_accel_" in f]
            norm_features = [f for f in valid_features if "_norm" in f]
            interaction_features = [f for f in valid_features if "_x_" in f]
            cross_timeframe_features = [f for f in valid_features if "diff_" in f and ("m_" in f or "h_" in f)]

        # Validate feature counts
        if len(diff_features) == 0:
        self.logger.warning("⚠️ No difference features generated")

        if len(accel_features) == 0:
        self.logger.warning("⚠️ No acceleration features generated")

        # Validate feature quality
        for feature_name, feature_value in valid_features.items():
        if isinstance(feature_value, pd.Series):
        # Check for excessive NaN values
                    nan_ratio = feature_value.isna().sum() / len(feature_value)
        if nan_ratio > 0.1:  # More than 10% NaN
        self.logger.warning(f"⚠️ Feature {feature_name} has {nan_ratio:.2%} NaN values")

        # Check for infinite values
                    inf_count = np.isinf(feature_value).sum()
        if inf_count > 0:
        self.logger.warning(f"⚠️ Feature {feature_name} has {inf_count} infinite values")

        self.logger.info(f"✅ Validated {len(valid_features)} enhanced features")
        self.logger.info(f"  - Difference features: {len(diff_features)}")
        self.logger.info(f"  - Acceleration features: {len(accel_features)}")
        self.logger.info(f"  - Normalized features: {len(norm_features)}")
        self.logger.info(f"  - Interaction features: {len(interaction_features)}")
        self.logger.info(f"  - Cross-timeframe features: {len(cross_timeframe_features)}")

        except Exception as e:
        self.logger.exception(f"❌ Validation failed for enhanced features: {e}")
            raise

    def _log_feature_engineering_summary(
        self = all_features: dict[str, Any], enhanced_features: dict[str, Any], ) -> None:
        """Log a summary of the feature engineering process.

        Args:
            all_features: All features after enhancement
            enhanced_features: Newly added enhanced features

        """
        try:
        # Filter out coroutine objects before logging
            valid_all_features = {}
        for key, value in all_features.items():
        if hasattr(value, "__await__"):
        self.logger.warning(f"⚠️ Skipping coroutine feature in all_features: {key}")
                    continue
                valid_all_features[key] = value

            valid_enhanced_features = {}
        for key, value in enhanced_features.items():
        if hasattr(value, "__await__"):
        self.logger.warning(f"⚠️ Skipping coroutine feature in enhanced_features: {key}")
                    continue
                valid_enhanced_features[key] = value

            total_features = len(valid_all_features)
            enhanced_count = len(valid_enhanced_features)
            original_count = total_features - enhanced_count

        # Categorize enhanced features
            diff_features = [f for f in valid_enhanced_features if "_diff_" in f]
            accel_features = [f for f in valid_enhanced_features if "_accel_" in f]
            norm_features = [f for f in valid_enhanced_features if "_norm" in f]
            interaction_features = [f for f in valid_enhanced_features if "_x_" in f]
            cross_timeframe_features = [f for f in valid_enhanced_features if "diff_" in f and ("m_" in f or "h_" in f)]

        self.logger.info("📊 Feature Engineering Summary:")
        self.logger.info(f"  - Original features: {original_count}")
        self.logger.info(f"  - Enhanced features: {enhanced_count}")
        self.logger.info(f"  - Total features: {total_features}")
        self.logger.info(f"  - Difference features: {len(diff_features)}")
        self.logger.info(f"  - Acceleration features: {len(accel_features)}")
        self.logger.info(f"  - Normalized features: {len(norm_features)}")
        self.logger.info(f"  - Interaction features: {len(interaction_features)}")
        self.logger.info(f"  - Cross-timeframe features: {len(cross_timeframe_features)}")

        # Also log post-cap counts if caps were applied earlier
        try: # If caps were applied = we likely have logging from the cap step; here just echo thresholds
        self.logger.info("📏 Caps: acceleration<=10, difference<=25, cross-timeframe<=50 (priority-aware)")
        except Exception:
                pass

        # Log memory usage
        try:
                import psutil
                memory_usage, psutil.Process().memory_info().rss / 1024 / 1024  # MB
        self.logger.info(f"  - Memory usage: {memory_usage:.1f} MB")
        except ImportError:
        self.logger.debug("ℹ️ psutil not available, skipping memory usage logging")
        except Exception as e:
        self.logger.debug(f"⚠️ Error logging memory usage: {e}")

        except Exception as e:
        self.logger.warning(f"⚠️ Failed to log feature engineering summary: {e}")

    def _generate_sr_levels(self, price_data: pd.DataFrame) -> dict[str, Any]:
        """Generate support/resistance levels from price data."""
        try:
        if price_data.empty or "close" not in price_data.columns:
        self.logger.warning("⚠️ Invalid price data for S/R level generation")
        return {}

        # Calculate daily data for SR levels
            daily_data, price_data.resample("D").agg({
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
                "volume": "sum",
            }).dropna()

        if daily_data.empty:
        self.logger.warning("⚠️ No daily data available for S/R level generation")
        return {}

        # Calculate support and resistance levels
            support_levels = []
            resistance_levels = []

        # Simple approach: use recent highs and lows
            recent_high = daily_data["high"].tail(20).max()
            recent_low = daily_data["low"].tail(20).min()
            current_price = price_data["close"].iloc[-1]

        # Add multiple support levels
            support_levels.extend([
                recent_low * 0.95,  # 5% below recent low
                recent_low * 0.98,  # 2% below recent low
                recent_low = # Recent low
                current_price * 0.95,  # 5% below current price
            ])

        # Add multiple resistance levels
            resistance_levels.extend([
                current_price * 1.02,  # 2% above current price
                current_price * 1.05,  # 5% above current price
                recent_high = # Recent high
                recent_high * 1.02,    # 2% above recent high
                recent_high * 1.05,    # 5% above recent high
            ])

        # Remove duplicates and sort
            support_levels = sorted({level for level in support_levels if level > 0})
            resistance_levels = sorted({level for level in resistance_levels if level > 0})

            sr_levels = {
                "support": support_levels,
                "resistance": resistance_levels,
            }

        self.logger.info(f"✅ Generated {len(support_levels)} support levels and {len(resistance_levels)} resistance levels")
        return sr_levels

        except Exception as e:
        self.logger.exception(f"❌ Error generating S/R levels: {e}")
        return {}

    def _handle_irregular_time_intervals(self, data: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        """Handle irregular time intervals gracefully for multi-timeframe feature generation.

        Args:
            data: Input DataFrame with potentially irregular timestamps
            timeframe: Target timeframe for resampling

        Returns:
            DataFrame with regularized timestamps

        """
        try:
        if data.empty:
        return data

        # Check if we have a DatetimeIndex
        if not isinstance(data.index, pd.DatetimeIndex):
        self.logger.warning(f"⚠️ Data does not have DatetimeIndex for {timeframe} timeframe")
        return data

        # Calculate time differences
            time_diffs = data.index.to_series().diff().dropna()

        if len(time_diffs) == 0:
        return data

        # Calculate expected interval for the timeframe
            timeframe_map = {
                "1m": pd.Timedelta(minutes=1)
                "5m": pd.Timedelta(minutes=5)
                "15m": pd.Timedelta(minutes=15)
                "30m": pd.Timedelta(minutes=30)
                "1h": pd.Timedelta(hours=1)
            }

            expected_interval, timeframe_map.get(timeframe, pd.Timedelta(minutes=1))

        # Calculate irregularity metrics
            irregular_intervals, time_diffs[abs(time_diffs - expected_interval) > pd.Timedelta(seconds=30)]
            irregular_ratio = len(irregular_intervals) / len(time_diffs)

        # If irregularity is low, no action needed
        if irregular_ratio < 0.05:  # Less than 5% irregular
        return data

        self.logger.info(f"🔧 Handling irregular time intervals for {timeframe} timeframe (irregularity: {irregular_ratio:.3f})")

        # Strategy 1: Forward fill small gaps
        if irregular_ratio < 0.15:  # Less than 15% irregular
        # Use forward fill for small gaps
                regularized_data = data.copy()
        # Forward fill any missing values that might have been created by irregular intervals
                regularized_data, regularized_data.fillna(method="ffill").fillna(method="bfill")
        self.logger.debug(f"🔧 Applied forward fill for {timeframe} timeframe")
        return regularized_data

        # Strategy 2: Resample to regular intervals for higher irregularity
        # Resample to regular intervals using only available columns
            available_columns = set(data.columns)
            aggregation_map = {}
        if "open" in available_columns:
                aggregation_map["open"] = "first"
        if "high" in available_columns:
                aggregation_map["high"] = "max"
        if "low" in available_columns:
                aggregation_map["low"] = "min"
        if "close" in available_columns:
                aggregation_map["close"] = "last"
        if "volume" in available_columns:
                aggregation_map["volume"] = "sum"

        if aggregation_map:
                resampled_data = data.resample(timeframe).agg(aggregation_map).dropna()
            else: # Fallback: if no recognized columns = use last observation
                resampled_data = data.resample(timeframe).last().dropna()

        # Forward fill any remaining gaps
            resampled_data, resampled_data.fillna(method="ffill").fillna(method="bfill")

        self.logger.debug(f"🔧 Applied resampling for {timeframe} timeframe (shape: {resampled_data.shape})")
        return resampled_data

        except Exception as e:
        self.logger.warning(f"⚠️ Error handling irregular time intervals for {timeframe}: {e}")
        return data


        self.logger.exception(f"❌ Insufficient price data: {len(price_data)} records (minimum: 10)")
        return {}

        if len(volume_data) < 10:
        self.logger.error(f"❌ Insufficient volume data: {len(volume_data)} records (minimum: 10)")
        return {}

        self.logger.info("✅ Input data validation passed")
        return None

        # Initialize features dictionary