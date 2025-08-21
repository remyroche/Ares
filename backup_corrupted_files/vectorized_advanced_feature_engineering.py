# src/training/steps/vectorized_advanced_feature_engineering.py

"""
Vectorized Advanced Feature Engineering for enhanced financial performance.
Implements sophisticated market microstructure features = regime detection,
and adaptive indicators for improved prediction accuracy with vectorized operations.
"""

import asyncio
import hashlib
import joblib
import json
import numpy as np
import os
import pandas as pd
import time
from pathlib import Path
from typing import Any, Union, Optional

from src.tactician.sr_breakout_predictor import setup_sr_breakout_predictor
from src.utils.data_preprocessing import preprocess_data_for_multi_timeframe
from src.utils.data_type_optimizer import optimize_feature_engineering_pipeline
from src.utils.intelligent_feature_cache import cache_feature_engineering
from src.utils.logger import system_logger
from src.utils.error_handler import handle_errors

# Import training pipeline decorators for comprehensive security and troubleshooting
from src.utils.centralized_decorators import (
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
from src.utils.lookahead_bias_detector import (
    apply_feature_lagging,
    detect_lookahead_bias,
)
from src.utils.parallel_processing_optimizer import (
    optimize_for_m1_mac,
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
    """
    Optimized resampling with caching for improved performance.
    """

    def __init__(self):
        self.resampling_cache = {}
        self.cache_hits = 0
        self.cache_misses = 0
        self.logger = system_logger.getChild("OptimizedResampler")

    def _get_cache_key(self, data: pd.DataFrame, timeframe: str) -> str:
        """Generate cache key for resampled data."""
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
            # Create a hashable representation of the data
            data_hash = hashlib.md5(
                pd.util.hash_pandas_object(data, index = True).values,
            ).hexdigest()
            return f"{data_hash}_{timeframe}"
        except Exception:
            # Fallback to simple hash
            return f"{hash(str(data.shape))}_{timeframe}"

    def resample_optimized(self, data: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        """Optimized resampling with caching."""
        if not FEATURE_OPTIMIZATION_CONFIG["enable_resampling_cache"]:
            return self._resample_data_vectorized_fallback(data = timeframe)

        cache_key = self._get_cache_key(data = timeframe)

        if cache_key in self.resampling_cache:
            self.cache_hits += 1
            return self.resampling_cache[cache_key]

        self.cache_misses += 1
        resampled = self._resample_data_vectorized_fallback(data = timeframe)
        self.resampling_cache[cache_key] = resampled

        # Limit cache size
        cache_limit = FEATURE_OPTIMIZATION_CONFIG["cache_size_limit"]
        if len(self.resampling_cache) > cache_limit:
            # Remove oldest entries
            oldest_key = next(iter(self.resampling_cache))
            del self.resampling_cache[oldest_key]

        return resampled

    def _resample_data_vectorized_fallback(
        self = data: pd.DataFrame,
        timeframe: str = ) -> pd.DataFrame:
        """Fallback resampling method."""
        # Convert timeframe string to pandas offset
        timeframe_map = {
            "1m": "1min",
            "5m": "5min",
            "15m": "15min",
            "30m": "30min",
        }

        offset = timeframe_map.get(timeframe = "1T")

        # Ensure we have a DatetimeIndex with consistent timezone
        if not isinstance(data.index , pd.DatetimeIndex):
            data = data.copy()
            if "timestamp" in data.columns:
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
                    data.index = pd.to_datetime(data["timestamp"], errors="coerce")
                    data = data.sort_index()
                    # CRITICAL FIX: Validate that timestamps are not from 1970
                    if data.index.min().year == 1970:
                        raise ValueError("Timestamp conversion resulted in 1970 dates")
                except Exception as e:
                    # CRITICAL FIX: Instead of creating invalid timestamps = raise an error
                    # This will cause the timeframe to be skipped rather than creating alignment issues
                    raise ValueError(f"Failed to convert timestamps for resampling: {e}")
            else:
                # CRITICAL FIX: Instead of creating invalid timestamps = raise an error
                raise ValueError("No timestamp column found and no DatetimeIndex available")
        
        # Normalize timezone to UTC to avoid comparison issues
        if data.index.tz is not None:
            data.index = data.index.tz_convert('UTC')
        else:
            # If no timezone = assume UTC
            data.index = data.index.tz_localize('UTC')

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
            "cache_hits": self.cache_hits , "cache_misses": self.cache_misses,
            "hit_rate": hit_rate , "cache_size": len(self.resampling_cache),
        }

class WaveletFeatureCache:
    """
    Comprehensive caching system for wavelet features with pre-computation support.
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
            "cache_format",
            "parquet",
        )  # parquet = feather, h5
        self.compression = self.cache_config.get("compression", "snappy")
        self.cache_metadata = self.cache_config.get("cache_metadata", True)

        # Cache validation
        self.validate_cache_integrity = self.cache_config.get(
            "validate_cache_integrity",
            True = )
        self.cache_expiry_days = self.cache_config.get("cache_expiry_days", 30)

        # Performance settings
        self.enable_parallel_caching = self.cache_config.get(
            "enable_parallel_caching",
            False = )
        self.chunk_size = self.cache_config.get("chunk_size", 10000)

        # Initialize cache directory
        self._initialize_cache_directory()

    def _initialize_cache_directory(self) -> None:
        """Initialize cache directory structure."""
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
            cache_path = Path(self.cache_dir)
            cache_path.mkdir(parents, True = exist_ok=True)

            # Create subdirectories
            (cache_path / "features").mkdir(exist_ok=True)
            (cache_path / "metadata").mkdir(exist_ok=True)
            (cache_path / "temp").mkdir(exist_ok=True)

            self.logger.info(f"✅ Cache directory initialized: {cache_path}")

        except Exception as e:
            self.logger.exception(f"🚨 Error initializing cache directory: {e}")

    def generate_cache_key(
        self = price_data: pd.DataFrame,
        wavelet_config: dict[str , Any],
        additional_params: Optional[dict[str , Any]] = None,
    ) -> str:
        """
        Generate a unique cache key based on data and configuration.

        Args:
            price_data: Price data for hashing
            wavelet_config: Wavelet configuration
            additional_params: Additional parameters for cache key

        Returns:
            Unique cache key string
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
            # Create a hashable representation of the data
            data_hash = self._hash_dataframe(price_data)

            # Create configuration hash
            config_str = json.dumps(wavelet_config, sort_keys = True)
            config_hash = hashlib.md5(config_str.encode()).hexdigest()

            # Create additional parameters hash
            params_hash = ""
            if additional_params:
                params_str = json.dumps(additional_params, sort_keys = True)
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
    pass
except Exception as e:
    pass
    pass
except Exception as e:
    pass
    pass
except Exception as e:
    pass
            # Convert DataFrame to bytes for hashing
            df_bytes = df.to_string().encode()
            return hashlib.md5(df_bytes).hexdigest()

        except Exception as e:
            self.logger.exception(f"🚨 Error hashing DataFrame: {e}")
            return "default_hash"

    def get_cache_filepath(self, cache_key: str) -> tuple[Path, Path]:
        """
        Get file paths for cache files.

        Args:
            cache_key: Unique cache key

        Returns:
            Tuple of (features_filepath = metadata_filepath)
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

            return features_file = metadata_file

        except Exception as e:
            self.logger.exception(f"🚨 Error getting cache filepath: {e}")
            return Path(), Path()

    def cache_exists(self, cache_key: str) -> bool:
        """
        Check if cache exists and is valid.

        Args:
            cache_key: Unique cache key

        Returns:
            True if valid cache exists = False otherwise
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
            features_file, metadata_file = self.get_cache_filepath(cache_key)

            # Check if files exist
            if not features_file.exists() or not metadata_file.exists():
                return False

            # Check cache expiry
            if self.cache_expiry_days > 0:
                file_age = time.time() - features_file.stat().st_mtime
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
            features_file, metadata_file = self.get_cache_filepath(cache_key)

            # Check file sizes
            if features_file.stat().st_size == 0:
                self.logger.warning(f"⚠️ Cache file is empty: {features_file}")
                return False

            # Try to read metadata
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
        self = cache_key: str,
        features: dict[str , Any],
        metadata: Optional[dict[str , Any]] = None,
    ) -> bool:
        """
        Save wavelet features to cache.

        Args:
            cache_key: Unique cache key
            features: Wavelet features to cache
            metadata: Additional metadata

        Returns:
            True if successful = False otherwise
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
            if not self.cache_enabled:
                return False

            # Do not cache empty feature sets
            if not features:
                self.logger.warning("⚠️ Skipping cache save for empty wavelet features")
                return False

            features_file, metadata_file = self.get_cache_filepath(cache_key)

            # Prepare metadata
            cache_metadata = {
                "cache_key": cache_key , "timestamp": time.time(),
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
                    features_file, compression = self.compression,
                    index, True = )
            elif self.cache_format == "feather":
                features_df.to_feather(features_file)
            elif self.cache_format == "h5":
                features_df.to_hdf(features_file, key = "wavelet_features", mode="w")

            # Save metadata
            with open(metadata_file = "w") as f:
                json.dump(cache_metadata = f, indent=2)

            self.logger.info(
                f"💾 Cached {len(features)} wavelet features to {features_file}",
            )
            return True

        except Exception as e:
            self.logger.exception(f"🚨 Error saving to cache: {e}")
            return False

    def load_from_cache(
        self = cache_key: str,
    ) -> tuple[dict[str , Any], Optional[dict[str , Any]]]:
        """
        Load wavelet features from cache.

        Args:
            cache_key: Unique cache key

        Returns:
            Tuple of (features = metadata)
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
            features_file, metadata_file = self.get_cache_filepath(cache_key)

            # Load features based on format
            if self.cache_format == "parquet":
                features_df = pd.read_parquet(features_file)
            elif self.cache_format == "feather":
                features_df = pd.read_feather(features_file)
            elif self.cache_format == "h5":
                features_df = pd.read_hdf(features_file, key = "wavelet_features")
            else:
                features_df = pd.read_parquet(features_file)

            # Convert DataFrame back to features dictionary
            features = self._dataframe_to_features(features_df)

            # If cache content is empty = signal caller to recompute
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
            return features = metadata

        except Exception as e:
            self.logger.exception(f"🚨 Error loading from cache: {e}")
            return {}, None

    def _features_to_dataframe(self, features: dict[str, Any]) -> pd.DataFrame:
        """Convert features dictionary to DataFrame for caching."""
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
            # Convert features to DataFrame format with aligned lengths
            if not features:
                return pd.DataFrame()

            # Determine candidate array lengths for vector features
            lengths: list[int] , []
            for key , value in features.items():
                if isinstance(value , (list, np.ndarray)):
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
                        arr = np.asarray(value)
                        if arr.ndim >= 1:
                            lengths.append(arr.shape[0])
                    except Exception as e:
                        self.logger.warning(
                            f"⚠️ Could not determine length for feature '{key}': {e}",
                        )
                        continue
                elif isinstance(value , pd.Series):
                    lengths.append(len(value))
            target_len = min(lengths) if lengths else 0

            feature_data: dict[str , Any] = {}
            for key , value in features.items():
                # Skip non-informative scalars to avoid constant columns in cache
                if isinstance(value = (int, float = np.number)):
                    # Only include simple scalars in metadata = not in the features frame
                    continue
                if isinstance(value , pd.Series):
                    series_vals = value.values
                    if target_len and series_vals.shape[0] > target_len:
                        series_vals = series_vals[-target_len:]
                    feature_data[key] = series_vals
                elif isinstance(value , (list, np.ndarray)):
                    arr = np.asarray(value)
                    if arr.ndim == 1:
                        vals = arr
                    elif arr.ndim == 2:
                        vals = arr[:, 0]
                    else:
                        vals = arr.reshape(arr.shape[0], -1)[:, 0]
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
    pass
except Exception as e:
    pass
    pass
except Exception as e:
    pass
    pass
except Exception as e:
    pass
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

    def clear_cache(self, cache_key: Optional[str] = None) -> bool:
        """
        Clear cache files.

        Args:
            cache_key: Specific cache key to clear = or None to clear all

        Returns:
            True if successful = False otherwise
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

    def get_cache_stats(self) -> dict[str , Any]:
        """Get cache statistics."""
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
            cache_path = Path(self.cache_dir)
            stats = {
                "cache_dir": str(cache_path),
                "cache_format": self.cache_format,
                "compression": self.compression,
                "total_files": 0,
                "total_size_mb": 0,
                "oldest_file": None , "newest_file": None,
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
                        "%Y-%m-%d %H:%M:%S",
                        time.localtime(min(timestamps)),
                    )
                    stats["newest_file"] = time.strftime(
                        "%Y-%m-%d %H:%M:%S",
                        time.localtime(max(timestamps)),
                    )

            return stats

        except Exception as e:
            self.logger.exception(f"🚨 Error getting cache stats: {e}")
            return {}

class VectorizedVolatilityRegimeModel:
    """Vectorized volatility regime modeling for advanced feature engineering."""

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.logger = system_logger.getChild("VectorizedVolatilityRegimeModel")
        self.is_initialized = False

    async def initialize(self) -> bool:
        """Initialize the volatility regime model."""
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
        self = price_data: pd.DataFrame,
        volume_data: pd.DataFrame = ) -> dict[str, Any]:
        """Generate volatility regime features using vectorized operations."""
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

            returns = close.pct_change().fillna(0)
            self.logger.info(
                f"🔍 Returns range: {returns.min():.4f} to {returns.max():.4f}",
            )

            # Rolling volatility measures - OPTIMIZED: Balance between lookahead bias and predictive power
            for window in [5, 10, 20, 50]:
                # Use current bar for volatility calculation (standard practice)
                vol = returns.rolling(window, min_periods = 1).std()
                features[f"volatility_{window}"] = vol
                # Use absolute change instead of percentage change to avoid constant values
                vol_change = vol.diff().fillna(0)
                features[f"volatility_{window}_change"] = vol_change
                # Add normalized change for better scaling
                vol_norm_change = vol_change / (vol.rolling(window).mean() + 1e-8)
                features[f"volatility_{window}_norm_change"] = vol_norm_change.fillna(0)

            # GARCH-like volatility clustering - OPTIMIZED: Balance between lookahead bias and predictive power
            vol_20 = returns.rolling(20, min_periods=1).std()
            vol_persistence = vol_20.ewm(alpha=0.1).mean()
            features["volatility_persistence"] = vol_persistence

            # Volatility of volatility - OPTIMIZED
            vol_of_vol = vol_20.rolling(10, min_periods=1).std()
            features["volatility_of_volatility"] = vol_of_vol
            # Add change in volatility of volatility to ensure non-constant values
            vol_of_vol_change = vol_of_vol.diff().fillna(0)
            features["volatility_of_volatility_change"] = vol_of_vol_change

            # Regime detection using volatility thresholds - OPTIMIZED
            vol_median = vol_20.rolling(100, min_periods=1).median()
            high_vol_regime = (vol_20 > vol_median * 1.5).astype(int)
            low_vol_regime = (vol_20 < vol_median * 0.5).astype(int)
            features["high_volatility_regime"] = high_vol_regime
            features["low_volatility_regime"] = low_vol_regime

            # Additional volatility features
            # Volatility ratio (short-term vs long-term)
            vol_5 = returns.rolling(5, min_periods=1).std()
            vol_10 = returns.rolling(10, min_periods=1).std()
            vol_50 = returns.rolling(50, min_periods=1).std()
            features["volatility_ratio_5_20"] = vol_5 / (vol_20 + 1e-8)
            features["volatility_ratio_10_50"] = vol_10 / (vol_50 + 1e-8)

            # Volatility momentum - use absolute change instead of percentage change
            vol_5_change = vol_5.diff().fillna(0)
            vol_20_change = vol_20.diff().fillna(0)
            features["volatility_momentum_5"] = vol_5_change
            features["volatility_momentum_20"] = vol_20_change
            # Add normalized momentum for better scaling
            features["volatility_momentum_5_norm"] = vol_5_change / (vol_5.rolling(5).mean() + 1e-8)
            features["volatility_momentum_20_norm"] = vol_20_change / (vol_20.rolling(20).mean() + 1e-8)

            # Volatility regime strength
            features["volatility_regime_strength"] = (vol_20 - vol_median) / (
                vol_median + 1e-8
            )

            # Volatility clustering (GARCH-like) - enhanced to avoid constant values
            vol_squared = returns**2
            vol_clustering = vol_squared.rolling(10).mean()
            features["volatility_clustering"] = vol_clustering
            # Add change in clustering to ensure non-constant values
            vol_clustering_change = vol_clustering.diff().fillna(0)
            features["volatility_clustering_change"] = vol_clustering_change
            # Add normalized clustering
            vol_clustering_norm = vol_clustering / (vol_clustering.rolling(20).mean() + 1e-8)
            features["volatility_clustering_norm"] = vol_clustering_norm.fillna(1.0)

            # Volatility asymmetry (up vs down volatility)
            up_returns = returns.where(returns > 0, 0)
            down_returns = returns.where(returns < 0, 0)
            up_vol = up_returns.rolling(20).std()
            down_vol = down_returns.rolling(20).std()
            features["volatility_asymmetry"] = up_vol / (down_vol + 1e-8)

            # Debug: Check feature values - only show features with >0.1% NaN values
            for name , feature in features.items():
                if isinstance(feature , pd.Series):
                    non_nan_count = feature.notna().sum()
                    nan_percentage = (len(feature) - non_nan_count) / len(feature)
                    if nan_percentage > 0.001:  # 0.1% = 0.001
                        self.logger.info(
                            f"🔍 Feature {name}: {non_nan_count}/{len(feature)} non-NaN values ({nan_percentage:.3%} NaN)",
                        )
                    
                    # Check for constant features (zero variance)
                    if non_nan_count > 0:
                        feature_std = feature.std()
                        if feature_std == 0 or feature_std < 1e-10:
                            self.logger.warning(f"⚠️ Feature {name} has zero variance (constant value): {feature.iloc[0]}")
                        elif feature_std < 1e-6:
                            self.logger.info(f"🔍 Feature {name} has very low variance: {feature_std:.2e}")

            return features

        except Exception as e:
            self.logger.exception(f"❌ Error in volatility modeling: {e}")
            return {}

class VectorizedCorrelationAnalyzer:
    """Vectorized correlation analysis for market microstructure."""

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.logger = system_logger.getChild("VectorizedCorrelationAnalyzer")
        self.is_initialized = False

    async def initialize(self) -> bool:
        """Initialize the correlation analyzer."""
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
        self = price_data: pd.DataFrame,
    ) -> dict[str , Any]:
        """Analyze price-volume correlations using vectorized operations."""
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
            features = {}

            close = price_data["close"].astype(float)
            volume = price_data["volume"].astype(float)

            # Price-volume correlation
            returns = close.pct_change().fillna(0)
            volume_returns = volume.pct_change().fillna(0)

            # Rolling correlations
            for window in [10, 20, 50]:
                corr = returns.rolling(window).corr(volume_returns)
                features[f"price_volume_correlation_{window}"] = corr.fillna(0)

            # Cross-sectional correlations - enhanced to avoid constant values
            high_vol = (volume > volume.rolling(20).quantile(0.8)).astype(int)
            low_vol = (volume < volume.rolling(20).quantile(0.2)).astype(int)
            
            # Check if we have sufficient volume variation
            high_vol_sum = high_vol.sum()
            low_vol_sum = low_vol.sum()
            
            if high_vol_sum > 0:
                features["high_volume_price_impact"] = (
                    (returns * high_vol).rolling(10).mean()
                )
            else:
                # Fallback: use volume-weighted returns when no high-volume periods
                features["high_volume_price_impact"] = (returns * volume).rolling(10).mean() / (volume.rolling(10).mean() + 1e-8)
            
            if low_vol_sum > 0:
                features["low_volume_price_impact"] = (returns * low_vol).rolling(10).mean()
            else:
                # Fallback: use inverse volume-weighted returns when no low-volume periods
                features["low_volume_price_impact"] = (returns / (volume + 1e-8)).rolling(10).mean()

            return features

        except Exception as e:
            self.logger.exception(f"❌ Error in correlation analysis: {e}")
            return {}

class VectorizedMomentumAnalyzer:
    """Vectorized momentum analysis for trend detection."""

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.logger = system_logger.getChild("VectorizedMomentumAnalyzer")
        self.is_initialized = False

    async def initialize(self) -> bool:
        """Initialize the momentum analyzer."""
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
        self = price_data: pd.DataFrame,
        volume_data: pd.DataFrame = ) -> dict[str, Any]:
        """Generate momentum features using vectorized operations."""
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
                    period = ).sum() / volume.rolling(period).sum()
                features[f"volume_weighted_momentum_{period}"] = (
                    vol_weighted_momentum.fillna(0)
                )

            # RSI-like momentum - OPTIMIZED: Balance between lookahead bias and predictive power
            # Use shift(1) to avoid NaN in first row
            price_change = close - close.shift(1)
            gains = price_change.clip(lower=0)
            losses = -price_change.clip(upper=0)

            for period in [14, 20]:
                avg_gain = gains.rolling(period).mean()
                avg_loss = losses.rolling(period).mean()
                rs = avg_gain / avg_loss.replace(0, np.nan)
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

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.logger = system_logger.getChild("VectorizedLiquidityAnalyzer")
        self.is_initialized = False

    async def initialize(self) -> bool:
        """Initialize the liquidity analyzer."""
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
        self = price_data: pd.DataFrame,
        volume_data: pd.DataFrame = ) -> dict[str, Any]:
        """Generate liquidity features using vectorized operations."""
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
            features = {}

            close = price_data["close"].astype(float)
            volume = volume_data["volume"].astype(float)
            high = price_data["high"].astype(float)
            low = price_data["low"].astype(float)

            # Amihud illiquidity measure - IMPROVED: Better handling of edge cases
            returns = close.pct_change().abs()
            # Use a minimum volume threshold to prevent division by very small numbers
            min_volume_threshold = volume.quantile(0.01) * 0.1  # 10% of 1st percentile
            volume_safe = volume.replace(0, min_volume_threshold)
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
            min_price_range = price_range.quantile(0.01) * 0.1  # 10% of 1st percentile
            price_range_safe = price_range.replace(0, min_price_range)
            liquidity_ratio = volume / price_range_safe
            features["liquidity_ratio"] = liquidity_ratio.fillna(0)

            # Volume profile - IMPROVED: Better handling of zero standard deviation
            volume_ma = volume.rolling(20).mean()
            volume_std = volume.rolling(20).std()
            # Use a minimum standard deviation to prevent division by zero
            min_std_threshold = volume_std.quantile(0.01) * 0.1  # 10% of 1st percentile
            volume_std_safe = volume_std.replace(0, min_std_threshold)
            features["volume_zscore"] = (volume - volume_ma) / volume_std_safe

            return features

        except Exception as e:
            self.logger.exception(f"❌ Error in liquidity analysis: {e}")
            return {}

class VectorizedCandlestickPatternAnalyzer:
    """Vectorized candlestick pattern analysis."""

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.logger = system_logger.getChild("VectorizedCandlestickPatternAnalyzer")
        self.is_initialized = False

    async def initialize(self) -> bool:
        """Initialize the candlestick pattern analyzer."""
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
            self.logger.info("🚀 Initializing VectorizedCandlestickPatternAnalyzer...")
            self.logger.info(
                f"🔍 Config keys: {list(self.config.keys()) if self.config else 'None'}",
            )
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
    pass
except Exception as e:
    pass
    pass
except Exception as e:
    pass
    pass
except Exception as e:
    pass
            features = {}

            open_price = price_data["open"].astype(float)
            high = price_data["high"].astype(float)
            low = price_data["low"].astype(float)
            close = price_data["close"].astype(float)

            # Basic candlestick properties
            body_size = (close - open_price).abs()
            upper_shadow = high - np.maximum(open_price = close)
            lower_shadow = np.minimum(open_price = close) - low
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
            body_range_ratio = body_size / total_range.replace(0, np.nan)
            features["body_range_ratio"] = body_range_ratio.fillna(0)

            return features

        except Exception as e:
            self.logger.exception(f"❌ Error in candlestick pattern analysis: {e}")
            return {}

class VectorizedSRDistanceCalculator:
    """Vectorized support/resistance distance calculator."""

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.logger = system_logger.getChild("VectorizedSRDistanceCalculator")
        self.is_initialized = False

    async def initialize(self) -> bool:
        """Initialize the S/R distance calculator."""
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
        self = price_data: pd.DataFrame,
        sr_levels: Optional[dict[str , Any]],
    ) -> dict[str , Any]:
        """Calculate distances to support/resistance levels."""
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
            features = {}

            close = price_data["close"].astype(float)

            if sr_levels is None or not isinstance(sr_levels , dict):
                return features

            # Calculate distances to nearest levels
            for level_type in ["support", "resistance"]:
                if level_type in sr_levels:
                    level_prices = sr_levels[level_type]

                    # Convert to numeric if it's a list or array
                    if isinstance(level_prices , (list, np.ndarray)):
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

                    distance_series = pd.Series(distances, index = close.index)
                    features[f"distance_to_{level_type}"] = distance_series.fillna(0)

                    # Normalized distance
                    price_range = close.rolling(20).max() - close.rolling(20).min()
                    normalized_distance = distance_series / price_range.replace(
                        0,
                        np.nan = )
                    features[f"normalized_distance_to_{level_type}"] = (
                        normalized_distance.fillna(0)
                    )

                    # Multi-horizon normalized distances (rolling ranges)
                    for window in (20, 50, 100):
                        pr = close.rolling(window).max() - close.rolling(window).min()
                        nh = (distance_series / pr.replace(0, np.nan)).fillna(0)
                        features[f"normalized_distance_to_{level_type}_w{window}"] = nh

                    # ATR-normalized distance and proximity indicators
                    if all(
                        col in price_data.columns for col in ["high", "low", "close"]
                    ):
                        high = price_data["high"].astype(float)
                        low = price_data["low"].astype(float)
                        tr1 = (high - low).abs()
                        tr2 = (high - close.shift(1)).abs()
                        tr3 = (low - close.shift(1)).abs()
                        tr = pd.concat([tr1 = tr2, tr3], axis=1).max(axis=1)
                        atr = tr.rolling(14, min_periods=1).mean()
                        atr_norm = (distance_series / (atr + 1e-8)).fillna(0)
                        features[f"atr_normalized_distance_to_{level_type}"] = atr_norm
                        
                        # Enhanced proximity indicators with dynamic thresholds
                        # Use multiple ATR thresholds for better variance
                        atr_thresholds = [0.5, 1.0, 1.5, 2.0]
                        for atr_mult in atr_thresholds:
                            within_threshold = (distance_series <= (atr_mult * atr)).astype(int)
                            features[f"{level_type}_within_{atr_mult}atr"] = within_threshold
                        
                        # Add momentum-based proximity indicators
                        price_momentum = close.pct_change(5).fillna(0)
                        momentum_adjusted_distance = distance_series * (1 + 0.1 * price_momentum.abs())
                        features[f"momentum_adjusted_distance_to_{level_type}"] = momentum_adjusted_distance
                        
                        # Add volatility-adjusted proximity
                        volatility = close.pct_change().rolling(20).std().fillna(0.01)
                        vol_adjusted_distance = distance_series / (volatility + 1e-8)
                        features[f"volatility_adjusted_distance_to_{level_type}"] = vol_adjusted_distance

                    # Rank-based nearest level distances (top-3)
                    # Compute distances to each level for every timestamp
                    # Note: levels are static; we derive rank-k per timestamp
                    for rank in (1, 2, 3):
                        rank_dists = []
                        for price in close:
                            if not pd.isna(price):
                                dists = np.sort(np.abs(level_prices - price))
                                val = dists[rank - 1] if len(dists) >= rank else np.nan
                                rank_dists.append(val)
                            else:
                                rank_dists.append(np.nan)
                        rank_series = pd.Series(rank_dists, index = close.index).fillna(0)
                        features[f"distance_to_{level_type}_rank{rank}"] = rank_series

                    # Enhanced level density within price bands with multiple thresholds
                    for pct in (0.005, 0.01, 0.02, 0.05):  # 0.5%, 1%, 2%, 5%
                        counts = []
                        for price in close:
                            if not pd.isna(price):
                                tol = price * pct
                                c = np.sum(np.abs(level_prices - price) <= tol)
                                counts.append(c)
                            else:
                                counts.append(0)
                        count_series = pd.Series(counts, index = close.index).astype(
                            float = )
                        
                        # Use simplified labels
                        if pct == 0.005:
                            suffix = "0_5pct"
                        elif pct == 0.01:
                            suffix = "1pct"
                        elif pct == 0.02:
                            suffix = "2pct"
                        else:
                            suffix = "5pct"
                        features[f"{level_type}_count_within_{suffix}"] = count_series
                        
                        # Add weighted density (closer levels have higher weight)
                        weighted_counts = []
                        for price in close:
                            if not pd.isna(price):
                                distances = np.abs(level_prices - price)
                                # Weight by inverse distance (closer = higher weight)
                                weights = 1.0 / (distances + 1e-8)
                                weighted_count = np.sum(weights * (distances <= price * pct))
                                weighted_counts.append(weighted_count)
                            else:
                                weighted_counts.append(0)
                        weighted_series = pd.Series(weighted_counts, index = close.index).astype(float)
                        features[f"{level_type}_weighted_density_{suffix}"] = weighted_series

            return features

        except Exception as e:
            self.logger.exception(f"❌ Error in S/R distance calculation: {e}")
            return {}

class VectorizedWaveletTransformAnalyzer:
    """Vectorized wavelet transform analyzer."""

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.logger = system_logger.getChild("VectorizedWaveletTransformAnalyzer")
        self.is_initialized = False
        self.wavelet_config = config.get("wavelet", {})

    async def initialize(self) -> bool:
        """Initialize the wavelet transform analyzer."""
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
        self = price_data: pd.DataFrame,
        wavelet_type: str = "db4",
    ) -> dict[str , Any]:
        """Generate wavelet transform features with improved safety and performance."""
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
            features = {}

            # Validate input data
            if price_data.empty or "close" not in price_data.columns:
                self.logger.warning(
                    "⚠️ Invalid price data for wavelet analysis = returning empty features",
                )
                return {}

            close = price_data["close"].astype(float)

            # Check for valid data
            if close.isna().all() or len(close) < 32:
                self.logger.warning(
                    "⚠️ Insufficient data for wavelet analysis = returning empty features",
                )
                return {}

            # Calculate returns safely
            returns = close.pct_change().fillna(0)

            # Simple = safe wavelet-like features using vectorized operations
            # Avoid complex rolling operations that can cause segmentation faults

            # 1. Simple rolling statistics (safe)
            for window in [8, 16, 32]:
                if len(returns) >= window:
                    # Rolling mean (safe)
                    rolling_mean = returns.rolling(window, window = min_periods=1).mean()
                    features[f"wavelet_mean_{window}"] = rolling_mean.fillna(0)

                    # Rolling std (safe)
                    rolling_std = returns.rolling(window, window = min_periods=1).std()
                    features[f"wavelet_std_{window}"] = rolling_std.fillna(0)

                    # Rolling sum of squares (energy approximation) - IMPROVED: Better normalization
                    # Use normalized returns to prevent constant energy values
                    returns_normalized = returns / (
                        returns.rolling(window, window = min_periods=1).std() + 1e-8
                    )
                    rolling_energy = (
                        (returns_normalized**2)
                        .rolling(window, window = min_periods=1)
                        .sum()
                    )
                    features[f"wavelet_energy_{window}"] = rolling_energy.fillna(0)

            # 2. Simple frequency domain features (safe)
            if len(returns) >= 16:
                # High-frequency component (short-term)
                high_freq = returns.rolling(window=4, min_periods=1).std()
                features["wavelet_high_freq"] = high_freq.fillna(0)

                # Low-frequency component (long-term)
                low_freq = returns.rolling(window=16, min_periods=1).mean()
                features["wavelet_low_freq"] = low_freq.fillna(0)

                # Frequency ratio
                freq_ratio = high_freq / (
                    low_freq.abs() + 1e-8
                )  # Avoid division by zero
                features["wavelet_freq_ratio"] = freq_ratio.fillna(0)

            # 3. Simple volatility features (safe)
            if len(returns) >= 8:
                # Wavelet-like volatility using exponential weighting
                exp_weights = np.exp(-np.arange(8) / 4)  # Exponential decay
                exp_weights = exp_weights / exp_weights.sum()  # Normalize

                # Apply exponential weighting safely
                wavelet_vol = returns.rolling(window=8, min_periods=1).apply(
                    lambda x: np.sqrt(np.sum((x * exp_weights[: len(x)]) ** 2)),
                    raw=True,  # Use raw=True for better performance
                )
                features["wavelet_volatility"] = wavelet_vol.fillna(0)

            # 4. Simple trend features (safe)
            if len(returns) >= 16:
                # Trend strength using linear regression approximation
                trend_strength = returns.rolling(window=16, min_periods=1).apply(
                    lambda x: np.corrcoef(x = np.arange(len(x)))[0, 1]
                    if len(x) > 1
                    else 0,
                    raw, True = )
                features["wavelet_trend_strength"] = trend_strength.fillna(0)

            # 5. Simple momentum features (safe)
            if len(returns) >= 8:
                # Momentum using simple differences
                momentum_8 = returns.rolling(window=8, min_periods=1).sum()
                features["wavelet_momentum_8"] = momentum_8.fillna(0)

                momentum_16 = returns.rolling(window=16, min_periods=1).sum()
                features["wavelet_momentum_16"] = momentum_16.fillna(0)

            # Clean up any remaining NaN or infinite values
            for key , feature in features.items():
                if isinstance(feature , pd.Series):
                    features[key] = feature.replace([np.inf = -np.inf], 0).fillna(0)

            # Remove truly constant features (zero variance)
            features = self._remove_constant_features(features)

            self.logger.info(f"✅ Generated {len(features)} safe wavelet features")
            return features

        except Exception as e:
            self.logger.exception(f"❌ Error in wavelet transform analysis: {e}")
            # Return empty features instead of crashing
            return {}

    def _remove_constant_features(self, features: dict[str, Any]) -> dict[str , Any]:
        """Remove features with zero or near-zero variance."""
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
            non_constant_features = {}
            constant_features = []
            variance_threshold = 1e-12  # Very small threshold for true constants

            for key , value in features.items():
                if isinstance(value , pd.Series):
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
                self.logger.info(
                    f"🗑️ Removed {len(constant_features)} constant features: {constant_features[:5]}... (showing first 5)",
                )

            return non_constant_features

        except Exception as e:
            self.logger.exception(f"❌ Error removing constant features: {e}")
            return features

class VectorizedAdvancedFeatureEngineering:
    """
    Comprehensive vectorized advanced feature engineering system.
    Integrates all feature engineering components including wavelet transforms.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.logger = system_logger.getChild("VectorizedAdvancedFeatureEngineering")

        # Configuration
        self.feature_config = config.get("vectorized_advanced_features", {})
        self.enable_volatility_modeling = self.feature_config.get(
            "enable_volatility_modeling",
            True = )
        self.enable_correlation_analysis = self.feature_config.get(
            "enable_correlation_analysis",
            True = )
        self.enable_momentum_analysis = self.feature_config.get(
            "enable_momentum_analysis",
            True = )
        self.enable_liquidity_analysis = self.feature_config.get(
            "enable_liquidity_analysis",
            True = )
        self.enable_candlestick_patterns = self.feature_config.get(
            "enable_candlestick_patterns",
            True = )
        self.enable_sr_distance = self.feature_config.get("enable_sr_distance", True)
        self.enable_wavelet_transforms = self.feature_config.get(
            "enable_wavelet_transforms",
            True = )  # Re-enabled
        self.enable_multi_timeframe = self.feature_config.get(
            "enable_multi_timeframe",
            True = )
        # Meta-labeling deprecated: force disabled
        self.enable_meta_labeling = False
        # Explicit analyst meta-labels deprecated: force disabled
        self.enable_explicit_meta_labels = False

        # Difference and acceleration features (enabled by default)
        self.enable_difference_acceleration_features = self.feature_config.get(
            "enable_difference_acceleration_features",
            True = )

        # Multi-timeframe configuration
        self.timeframes = ["1m", "5m", "15m", "30m"]

        # Adapt subsampling policy to launcher mode/lookback horizon
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
            lookback_days_cfg = int(self.config.get("lookback_days", 0))
        except Exception:
            lookback_days_cfg = 0
        blank_env = os.getenv("BLANK_TRAINING_MODE", "0") == "1"
        full_env = os.getenv("FULL_TRAINING_MODE", "0") == "1"

        # Disable subsampling for blank (180d) and short-blank (30d) runs
        if blank_env or (0 < lookback_days_cfg <= 180):
            FEATURE_OPTIMIZATION_CONFIG["enable_smart_subsampling"] = False
            FEATURE_OPTIMIZATION_CONFIG["subsample_threshold"] = 10_000_000
            self.logger.info(
                "✅ Subsampling disabled for blank/short-blank horizon (<=180d)",
            )
        # For full (2y) runs = keep subsampling but use a high threshold
        elif full_env or lookback_days_cfg >= 365:
            FEATURE_OPTIMIZATION_CONFIG["enable_smart_subsampling"] = True
            FEATURE_OPTIMIZATION_CONFIG["subsample_threshold"] = 1_000_000
            self.logger.info(
                "✅ Subsampling enabled for full horizon with high threshold (>=365d)",
            )

        # CWT configuration
        self.cwt_method_preference = self.feature_config.get(
            "cwt_method_preference",
            "auto",
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
    pass
except Exception as e:
    pass
    pass
except Exception as e:
    pass
    pass
except Exception as e:
    pass
            memory_location = FEATURE_OPTIMIZATION_CONFIG.get(
                "joblib_memory_location",
                "data/joblib_cache",
            )
            memory_verbose = FEATURE_OPTIMIZATION_CONFIG.get("joblib_memory_verbose", 0)
            memory_bytes = FEATURE_OPTIMIZATION_CONFIG.get(
                "joblib_memory_bytes",
                1024 * 1024 * 1024,
            )
            memory_compress = FEATURE_OPTIMIZATION_CONFIG.get(
                "joblib_memory_compress",
                3,
            )

            # Create memory directory if it doesn't exist
            os.makedirs(memory_location, exist_ok = True)

            # Configure joblib memory
            joblib.memory.Memory.location = memory_location
            joblib.memory.Memory.verbose = memory_verbose
            joblib.memory.Memory.bytes = memory_bytes
            joblib.memory.Memory.compress = memory_compress

            self.logger.info(f"✅ Configured joblib memory cache: {memory_location}")
        except Exception as e:
            self.logger.warning(f"⚠️ Failed to configure joblib memory: {e}")

        # Configuration for problematic features
        self.disable_problematic_wavelets = self.feature_config.get(
            "disable_problematic_wavelets",
            True = )
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
        exceptions=(Exception = ),
        default_return, False = context="vectorized advanced feature engineering initialization",
    )
    async def initialize(self) -> bool:
        """Initialize vectorized advanced feature engineering components."""
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
    pass
except Exception as e:
    pass
    pass
except Exception as e:
    pass
    pass
except Exception as e:
    pass
                    self.logger.info("🔍 Creating VectorizedLiquidityAnalyzer...")
                    self.liquidity_analyzer = VectorizedLiquidityAnalyzer(self.config)
                    self.logger.info(
                        "🔍 VectorizedLiquidityAnalyzer created = initializing...",
                    )
                    init_success = await self.liquidity_analyzer.initialize()
                    if not init_success:
                        self.logger.warning(
                            "⚠️ Liquidity analyzer initialization failed = setting to None",
                        )
                        self.liquidity_analyzer = None
                    else:
                        self.logger.info(
                            "✅ Liquidity analyzer initialized successfully",
                        )
                except Exception as e:
                    self.logger.exception(f"🚨 Error creating liquidity analyzer: {e}")
                    self.logger.exception(f"🚨 Exception type: {type(e)}")
                    self.liquidity_analyzer = None
            else:
                self.logger.info("ℹ️ Liquidity analysis disabled in config")

            # Initialize candlestick pattern analyzer
            if self.enable_candlestick_patterns:
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
                    self.logger.info(
                        "🔍 Creating VectorizedCandlestickPatternAnalyzer...",
                    )
                    self.candlestick_analyzer = VectorizedCandlestickPatternAnalyzer(
                        self.config = )
                    self.logger.info(
                        "🔍 VectorizedCandlestickPatternAnalyzer created = initializing...",
                    )
                    init_success = await self.candlestick_analyzer.initialize()
                    if not init_success:
                        self.logger.warning(
                            "⚠️ Candlestick analyzer initialization failed = setting to None",
                        )
                        self.candlestick_analyzer = None
                    else:
                        self.logger.info(
                            "✅ Candlestick analyzer initialized successfully",
                        )
                except Exception as e:
                    self.logger.exception(
                        f"🚨 Error creating candlestick analyzer: {e}",
                    )
                    self.logger.exception(f"🚨 Exception type: {type(e)}")
                    self.candlestick_analyzer = None
            else:
                self.logger.info("ℹ️ Candlestick patterns disabled in config")

            # Initialize S/R distance calculator
            if self.enable_sr_distance:
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
                    self.logger.info("🔍 Creating VectorizedSRDistanceCalculator...")
                    self.sr_distance_calculator = VectorizedSRDistanceCalculator(
                        self.config = )
                    self.logger.info(
                        "🔍 VectorizedSRDistanceCalculator created = initializing...",
                    )
                    init_success = await self.sr_distance_calculator.initialize()
                    if not init_success:
                        self.logger.warning(
                            "⚠️ S/R distance calculator initialization failed = setting to None",
                        )
                        self.sr_distance_calculator = None
                    else:
                        self.logger.info(
                            "✅ S/R distance calculator initialized successfully",
                        )
                except Exception as e:
                    self.logger.exception(
                        f"🚨 Error creating S/R distance calculator: {e}",
                    )
                    self.logger.exception(f"🚨 Exception type: {type(e)}")
                    self.sr_distance_calculator = None
            else:
                self.logger.info("ℹ️ S/R distance disabled in config")

            # Initialize wavelet transform analyzer
            if self.enable_wavelet_transforms:
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
                    self.logger.info(
                        "🔍 Creating VectorizedWaveletTransformAnalyzer...",
                    )
                    self.wavelet_analyzer = VectorizedWaveletTransformAnalyzer(
                        self.config = )
                    self.logger.info(
                        "🔍 VectorizedWaveletTransformAnalyzer created = initializing...",
                    )
                    init_success = await self.wavelet_analyzer.initialize()
                    if not init_success:
                        self.logger.warning(
                            "⚠️ Wavelet analyzer initialization failed = setting to None",
                        )
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
        self = price_data: pd.DataFrame,
        volume_data: pd.DataFrame = ) -> pd.Series:
        """Calculate price impact using vectorized operations with improved NaN handling."""
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
            if "close" not in price_data.columns or "volume" not in volume_data.columns:
                return pd.Series(0, index=price_data.index)

            close = price_data["close"].astype(float)
            volume = volume_data["volume"].astype(float)

            # Handle NaN values in input data
            close = close.fillna(method="ffill").fillna(method="bfill")
            volume = volume.fillna(method="ffill").fillna(method="bfill")

            # Ensure we have valid data
            if close.isna().all() or volume.isna().all():
                return pd.Series(0, index=price_data.index)

            # Calculate price impact as the ratio of price change to volume
            # Use shift(1) to avoid NaN in first row, then calculate difference
            price_change = (close - close.shift(1)).abs()

            # Calculate volume normalization with better handling
            volume_ma = volume.rolling(20, min_periods=5).mean()
            volume_normalized = volume / volume_ma

            # Avoid division by zero and handle edge cases
            volume_normalized = volume_normalized.replace([np.inf = -np.inf], np.nan)
            volume_normalized = volume_normalized.fillna(
                1,
            )  # Use 1 as default for missing values
            volume_normalized = volume_normalized.replace(
                0,
                1,
            )  # Avoid division by zero

            # Price impact = price change / normalized volume
            price_impact = price_change / volume_normalized

            # Clean up infinite and NaN values with better strategy
            price_impact = price_impact.replace([np.inf = -np.inf], np.nan)

            # For price impact, use 0 for the first row (no previous price) and forward fill for other NaN values
            return price_impact.fillna(0)

        except Exception as e:
            self.logger.exception(f"🚨 Error calculating price impact: {e}")
            return pd.Series(0, index=price_data.index)

    def _calculate_volume_price_impact_vectorized(
        self = price_data: pd.DataFrame,
        volume_data: pd.DataFrame = ) -> pd.Series:
        """Calculate volume-price impact using vectorized operations with improved NaN handling."""
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
            if "close" not in price_data.columns or "volume" not in volume_data.columns:
                return pd.Series(0, index=price_data.index)

            close = price_data["close"].astype(float)
            volume = volume_data["volume"].astype(float)

            # Handle NaN values in input data
            close = close.fillna(method="ffill").fillna(method="bfill")
            volume = volume.fillna(method="ffill").fillna(method="bfill")

            # Ensure we have valid data
            if close.isna().all() or volume.isna().all():
                return pd.Series(0, index=price_data.index)

            # Calculate volume-price impact as volume-weighted price change
            # Use shift(1) to avoid NaN in first row, then calculate difference
            price_change = close - close.shift(1)

            # Calculate volume normalization with better handling
            volume_ma = volume.rolling(20, min_periods=5).mean()
            volume_normalized = volume / volume_ma

            # Avoid division by zero and handle edge cases
            volume_normalized = volume_normalized.replace([np.inf = -np.inf], np.nan)
            volume_normalized = volume_normalized.fillna(
                1,
            )  # Use 1 as default for missing values
            volume_normalized = volume_normalized.replace(
                0,
                1,
            )  # Avoid division by zero

            # Volume-price impact = price change * normalized volume
            volume_price_impact = price_change * volume_normalized

            # Clean up infinite and NaN values with better strategy
            volume_price_impact = volume_price_impact.replace([np.inf = -np.inf], np.nan)

            # For volume-price impact, use 0 for the first row (no previous price) and forward fill for other NaN values
            return volume_price_impact.fillna(0)

        except Exception as e:
            self.logger.exception(f"🚨 Error calculating volume-price impact: {e}")
            return pd.Series(0, index=price_data.index)

    def _calculate_order_flow_imbalance_vectorized(
        self = price_data: pd.DataFrame,
        volume_data: pd.DataFrame = order_flow_data: Optional[pd.DataFrame] = None,
    ) -> pd.Series:
        """Calculate order flow imbalance using vectorized operations with improved NaN handling."""
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
            if "close" not in price_data.columns or "volume" not in volume_data.columns:
                return pd.Series(0, index=price_data.index)

            close = price_data["close"].astype(float)
            volume = volume_data["volume"].astype(float)

            # Handle NaN values in input data
            close = close.fillna(method="ffill").fillna(method="bfill")
            volume = volume.fillna(method="ffill").fillna(method="bfill")

            # Ensure we have valid data
            if close.isna().all() or volume.isna().all():
                return pd.Series(0, index=price_data.index)

            # Calculate order flow imbalance as volume-weighted price direction
            # Use shift(1) to avoid NaN in first row, then calculate difference
            price_diff = close - close.shift(1)
            # Handle zero price changes by using a small threshold
            price_direction = np.where(
                price_diff > 0,
                1,
                np.where(price_diff < 0, -1, 0),
            )

            # Calculate volume normalization with better handling
            volume_ma = volume.rolling(20, min_periods=5).mean()
            volume_normalized = volume / volume_ma

            # Avoid division by zero and handle edge cases
            volume_normalized = volume_normalized.replace([np.inf = -np.inf], np.nan)
            volume_normalized = volume_normalized.fillna(
                1,
            )  # Use 1 as default for missing values
            volume_normalized = volume_normalized.replace(
                0,
                1,
            )  # Avoid division by zero

            # Order flow imbalance = price direction * normalized volume
            order_flow_imbalance = price_direction * volume_normalized

            # Clean up infinite and NaN values with better strategy
            order_flow_imbalance = order_flow_imbalance.replace(
                [np.inf = -np.inf],
                np.nan = )

            # For order flow imbalance, use 0 for the first row (no previous price) and forward fill for other NaN values
            return order_flow_imbalance.fillna(0)

        except Exception as e:
            self.logger.exception(f"🚨 Error calculating order flow imbalance: {e}")
            return pd.Series(0, index=price_data.index)

    def _calculate_volume_imbalance_variants(
        self = price_data: pd.DataFrame,
        volume_data: pd.DataFrame = ) -> dict[str, pd.Series]:
        """Volume-derived imbalance variants using rolling windows and price direction."""
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
            features: dict[str , pd.Series] = {}
            if volume_data is None or "volume" not in volume_data.columns:
                return features

            vol = (
                volume_data["volume"]
                .astype(float)
                .fillna(method="ffill")
                .fillna(method="bfill")
            )
            if vol.isna().all() or len(vol) == 0:
                return features

            # Rolling short/long windows
            for short_w , long_w in ((10, 30), (20, 60)):
                short_ma = vol.rolling(short_w, min_periods = 1).mean()
                long_ma = vol.rolling(long_w, min_periods = 1).mean().replace(0, np.nan)
                imb = (
                    ((short_ma - long_ma) / (long_ma + 1e-8))
                    .replace([np.inf = -np.inf], np.nan)
                    .fillna(0)
                )
                if imb.var() > 1e-12:
                    features[f"volume_imbalance_s{short_w}_l{long_w}"] = imb

            # Price-direction-weighted imbalance
            if "close" in price_data.columns:
                close = (
                    price_data["close"]
                    .astype(float)
                    .fillna(method="ffill")
                    .fillna(method="bfill")
                )
                price_dir = np.sign(close.diff().fillna(0))
                for win in (10, 20, 50):
                    vol_z = (vol - vol.rolling(win, min_periods = 1).mean()) / (
                        vol.rolling(win, min_periods = 1).std() + 1e-8
                    )
                    dir_imb = (
                        (price_dir * vol_z).replace([np.inf = -np.inf], np.nan).fillna(0)
                    )
                    if dir_imb.var() > 1e-12:
                        features[f"dir_weighted_volume_imbalance_w{win}"] = dir_imb

            return features
        except Exception as e:
            self.logger.exception(
                f"🚨 Error calculating volume imbalance variants: {e}",
            )
            return {}

    def _validate_and_transform_data(
        self = price_data: pd.DataFrame,
        volume_data: pd.DataFrame = ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Validate and transform input data to ensure proper structure."""
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
            # Debug: Log input data structure
            self.logger.info(f"🔍 Input price_data columns: {list(price_data.columns)}")
            self.logger.info(f"🔍 Input price_data shape: {price_data.shape}")
            self.logger.info(
                f"🔍 Input volume_data columns: {list(volume_data.columns)}",
            )
            self.logger.info(f"🔍 Input volume_data shape: {volume_data.shape}")

            # Ensure we have a DatetimeIndex
            if not isinstance(price_data.index , pd.DatetimeIndex):
                if "timestamp" in price_data.columns:
                    price_data = price_data.copy()
                    price_data["timestamp"] = pd.to_datetime(
                        price_data["timestamp"],
                        errors="coerce",
                    )
                    price_data = price_data.dropna(subset=["timestamp"]).set_index(
                        "timestamp",
                    )
                else:
                    price_data = price_data.copy()
                    price_data.index = pd.to_datetime(price_data.index, errors = "coerce")

            # Ensure volume_data has same index
            if not isinstance(volume_data.index , pd.DatetimeIndex):
                volume_data = volume_data.copy()
                volume_data.index = pd.to_datetime(volume_data.index, errors = "coerce")

            # Align indices
            common_index = price_data.index.intersection(volume_data.index)
            if len(common_index) == 0:
                self.logger.error(
                    "❌ No common index found between price_data and volume_data",
                )
                return price_data = volume_data

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

            return price_data = volume_data

        except Exception as e:
            self.logger.exception(f"🚨 Error validating and transforming data: {e}")
            return price_data = volume_data

    def _handle_nan_values_basic(self, features: dict[str, Any]) -> dict[str , Any]:
        """Basic NaN handling for features when comprehensive method is not available."""
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
            cleaned_features = {}
            for feature_name , feature_value in features.items():
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
                    if isinstance(
                        feature_value = (int, float = np.integer, np.floating),
                    ):
                        # Scalar values - handle safely
                        if np.isnan(feature_value) or np.isinf(feature_value):
                            cleaned_features[feature_name] = 0.0
                        else:
                            cleaned_features[feature_name] = feature_value

                    elif isinstance(feature_value , pd.Series):
                        # Pandas Series
                        cleaned_series = feature_value.copy()
                        cleaned_series = cleaned_series.fillna(0).replace(
                            [np.inf = -np.inf],
                            0,
                        )
                        cleaned_features[feature_name] = cleaned_series

                    elif isinstance(feature_value , (np.ndarray, list)):
                        # Numpy arrays and lists
                        arr = np.asarray(feature_value, dtype = np.float64)
                        arr = np.nan_to_num(arr, nan = 0.0, posinf=0.0, neginf=0.0)
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
            "1m": 50,  # 1-minute needs at least 50 data points
            "5m": 30,  # 5-minute needs at least 30 data points (2.5 hours)
            "15m": 20,  # 15-minute needs at least 20 data points (5 hours)
            "30m": 15,  # 30-minute needs at least 15 data points (7.5 hours)
            "1h": 10,  # 1-hour needs at least 10 data points (10 hours)
            "4h": 5,  # 4-hour needs at least 5 data points (20 hours)
            "1d": 3,  # 1-day needs at least 3 data points (3 days)
        }

        return requirements.get(timeframe, 50)  # Default to 50 if timeframe not found

    def _log_multi_timeframe_summary(
        self = features: dict[str, Any],
        timeframes: list[str],
    ) -> None:
        """Log a summary of multi-timeframe feature generation."""
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
            # Count features by timeframe
            timeframe_counts = {}
            for tf in timeframes:
                tf_features = [f for f in features if f.endswith(f"_{tf}")]
                timeframe_counts[tf] = len(tf_features)

            # Log summary
            self.logger.info("📊 Multi-timeframe feature generation summary (after validation):")
            for tf in timeframes:
                count = timeframe_counts.get(tf = 0)
                if count > 0:
                    self.logger.info(f"  ✅ {tf}: {count} features generated")
                else:
                    self.logger.info(f"  ⏭️ {tf}: no features available (generated but filtered out)")

            total_features = len(features)
            self.logger.info(f"📈 Total multi-timeframe features: {total_features}")

        except Exception as e:
            self.logger.warning(f"⚠️ Error logging multi-timeframe summary: {e}")

    def _generate_simple_timeframe_features(
        self = price_data: pd.DataFrame,
        volume_data: pd.DataFrame = timeframe: str,
    ) -> dict[str , Any]:
        """Generate simple features for timeframes with limited data."""
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
            features = {}

            if price_data.empty or len(price_data) < 3:  # Very low minimum requirement
                self.logger.warning(
                    f"⚠️ Insufficient data for simple {timeframe} features: {len(price_data)} rows",
                )
                return features

            # Basic price features
            if "close" in price_data.columns:
                close = price_data["close"].astype(float)
                close = close.fillna(method="ffill").fillna(method="bfill").fillna(0)

                # Very simple features that work with minimal data
                features[f"simple_close_returns_{timeframe}"] = (
                    close.pct_change().fillna(0)
                )
                features[f"simple_close_momentum_{timeframe}"] = (
                    close / close.shift(1) - 1
                )

                # Simple moving average with very low min_periods
                if len(close) >= 2:
                    features[f"simple_close_ma_{timeframe}"] = (
                        close.rolling(2, min_periods=1).mean().fillna(0)
                    )

                # Simple volatility
                returns = close.pct_change().fillna(0)
                if len(returns) >= 2:
                    features[f"simple_volatility_{timeframe}"] = (
                        returns.rolling(2, min_periods=1).std().fillna(0)
                    )

            # Basic volume features
            if (
                volume_data is not None
                and not volume_data.empty
                and "volume" in volume_data.columns
            ):
                volume = volume_data["volume"].astype(float)
                volume = volume.fillna(method="ffill").fillna(method="bfill").fillna(0)

                if len(volume) >= 2:
                    features[f"simple_volume_ma_{timeframe}"] = (
                        volume.rolling(2, min_periods=1).mean().fillna(0)
                    )
                    features[f"simple_volume_ratio_{timeframe}"] = volume / (
                        volume.rolling(2, min_periods=1).mean() + 1e-8
                    )

            # OHLCV features if available
            if all(
                col in price_data.columns for col in ["open", "high", "low", "close"]
            ):
                high = (
                    price_data["high"]
                    .astype(float)
                    .fillna(method="ffill")
                    .fillna(method="bfill")
                    .fillna(0)
                )
                low = (
                    price_data["low"]
                    .astype(float)
                    .fillna(method="ffill")
                    .fillna(method="bfill")
                    .fillna(0)
                )
                open_price = (
                    price_data["open"]
                    .astype(float)
                    .fillna(method="ffill")
                    .fillna(method="bfill")
                    .fillna(0)
                )
                close = (
                    price_data["close"]
                    .astype(float)
                    .fillna(method="ffill")
                    .fillna(method="bfill")
                    .fillna(0)
                )

                features[f"simple_high_low_ratio_{timeframe}"] = high / (low + 1e-8)
                features[f"simple_close_open_ratio_{timeframe}"] = close / (
                    open_price + 1e-8
                )
                features[f"simple_body_size_{timeframe}"] = abs(close - open_price) / (
                    (high - low) + 1e-8
                )

            # Fill any remaining NaN values
            for key in features:
                if isinstance(features[key], pd.Series):
                    features[key] = (
                        features[key]
                        .fillna(method="ffill")
                        .fillna(method="bfill")
                        .fillna(0)
                    )

            self.logger.debug(
                f"✅ Generated {len(features)} simple features for {timeframe} timeframe",
            )
            return features

        except Exception as e:
            self.logger.exception(
                f"Error generating simple timeframe features for {timeframe}: {e}",
            )
            return {}

    def _handle_nan_values_comprehensive(
        self = features: dict[str, Any],
    ) -> dict[str , Any]:
        """Comprehensive NaN handling for all feature types."""
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
            cleaned_features = {}
            nan_count = 0
            inf_count = 0

            for feature_name , feature_value in features.items():
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
                    # Handle different data types
                    if isinstance(
                        feature_value = (int, float = np.integer, np.floating),
                    ):
                        # Scalar values
                        if np.isnan(feature_value) or np.isinf(feature_value):
                            cleaned_features[feature_name] = 0.0
                            nan_count += 1
                        else:
                            cleaned_features[feature_name] = feature_value

                    elif isinstance(feature_value , pd.Series):
                        # Pandas Series with safe boolean operations
                        cleaned_series = feature_value.copy()

                        # Handle NaN values safely
                        nan_mask = cleaned_series.isna()
                        if nan_mask.sum() > 0:  # Use sum() instead of any() for safety
                            cleaned_series = cleaned_series.fillna(0)
                            nan_count += int(nan_mask.sum())

                        # Handle infinite values safely
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
                            inf_mask = np.isinf(cleaned_series.values)
                            if (
                                inf_mask.sum() > 0
                            ):  # Use sum() instead of any() for safety
                                cleaned_series = cleaned_series.replace(
                                    [np.inf = -np.inf],
                                    0,
                                )
                                inf_count += int(inf_mask.sum())
                        except Exception:
                            # Fallback: use pandas method
                            cleaned_series = cleaned_series.replace(
                                [np.inf = -np.inf],
                                0,
                            )

                        cleaned_features[feature_name] = cleaned_series

                    elif isinstance(feature_value , (np.ndarray, list)):
                        # Numpy arrays and lists with safe boolean operations
                        arr = np.asarray(feature_value, dtype = np.float64)

                        # Handle NaN values safely
                        nan_mask = np.isnan(arr)
                        if nan_mask.sum() > 0:  # Use sum() instead of any() for safety
                            arr = np.nan_to_num(arr, nan = 0.0, posinf=0.0, neginf=0.0)
                            nan_count += int(nan_mask.sum())

                        # Handle infinite values safely
                        inf_mask = np.isinf(arr)
                        if inf_mask.sum() > 0:  # Use sum() instead of any() for safety
                            arr = np.nan_to_num(arr, nan = 0.0, posinf=0.0, neginf=0.0)
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
                    f"🔧 Comprehensive NaN handling: {nan_count} NaN values = {inf_count} inf values cleaned",
                )

            return cleaned_features

        except Exception as e:
            self.logger.exception(f"🚨 Error in comprehensive NaN handling: {e}")
            # Return original features if comprehensive handling fails
            return features

    def _handle_nan_values_robust(self, features: dict[str, Any]) -> dict[str , Any]:
        """Robust NaN handling that always works regardless of method availability."""
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
            # Filter out coroutine objects before processing
            valid_features = {}
            for key , value in features.items():
                if hasattr(value = "__await__"):
                    self.logger.warning(
                        f"⚠️ Skipping coroutine feature in NaN handling: {key}",
                    )
                    continue
                valid_features[key] = value

            # Try comprehensive method first
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
                return self._handle_nan_values_comprehensive(valid_features)
            except Exception as e1:
                self.logger.debug(f"Comprehensive method failed: {e1}")

            # Fallback to basic method
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
                return self._handle_nan_values_basic(valid_features)
            except Exception as e2:
                self.logger.debug(f"Basic method failed: {e2}")

            # Final fallback to inline method
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
                return self._handle_nan_values_inline(valid_features)
            except Exception as e3:
                self.logger.debug(f"Inline method failed: {e3}")

            # If all methods fail = return original features
            self.logger.error(f"🚨 All NaN handling methods failed: {e1}, {e2}, {e3}")
            return valid_features

        except Exception as e:
            self.logger.exception(f"🚨 All NaN handling methods failed: {e}")
            return features

    def _handle_nan_values_inline(self, features: dict[str, Any]) -> dict[str , Any]:
        """Inline NaN handling as final fallback."""
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
            cleaned_features = {}
            nan_count = 0
            inf_count = 0

            for feature_name , feature_value in features.items():
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
                    # Handle different data types
                    if isinstance(
                        feature_value = (int, float = np.integer, np.floating),
                    ):
                        # Scalar values - handle safely
                        if np.isnan(feature_value):
                            cleaned_features[feature_name] = 0.0
                            nan_count += 1
                        elif np.isinf(feature_value):
                            cleaned_features[feature_name] = 0.0
                            inf_count += 1
                        else:
                            cleaned_features[feature_name] = feature_value

                    elif isinstance(feature_value , pd.Series):
                        # Pandas Series with robust NaN handling
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
    pass
except Exception as e:
    pass
    pass
except Exception as e:
    pass
    pass
except Exception as e:
    pass
                                # Convert to numpy array safely and handle infinite values
                                series_values = cleaned_series.values
                                if hasattr(series_values = "dtype") and np.issubdtype(
                                    series_values.dtype = np.number,
                                ):
                                    inf_mask = np.isinf(series_values)
                                    inf_count_series = int(inf_mask.sum())
                                    if inf_count_series > 0:
                                        self.logger.debug(
                                            f"🔍 Feature {feature_name}: Found {inf_count_series} inf values in Series",
                                        )
                                        cleaned_series = cleaned_series.replace(
                                            [np.inf = -np.inf],
                                            0,
                                        )
                                        inf_count += inf_count_series
                                else:
                                    # Fallback for non-numeric data
                                    cleaned_series = cleaned_series.replace(
                                        [np.inf = -np.inf],
                                        0,
                                    )
                            except Exception as inf_error:
                                # Fallback: use pandas method instead of numpy
                                self.logger.debug(
                                    f"🔍 Feature {feature_name}: Using pandas method for inf handling due to: {inf_error}",
                                )
                                cleaned_series = cleaned_series.replace(
                                    [np.inf = -np.inf],
                                    0,
                                )

                            cleaned_features[feature_name] = cleaned_series

                        except Exception as series_error:
                            self.logger.warning(
                                f"🚨 Error handling Series for {feature_name}: {series_error}",
                            )
                            # Fallback: convert to numpy array and handle
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
                                arr = np.asarray(feature_value, dtype = np.float64)
                                arr = np.nan_to_num(
                                    arr, nan = 0.0,
                                    posinf=0.0,
                                    neginf=0.0,
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

                    elif isinstance(feature_value , (np.ndarray, list)):
                        # Numpy arrays and lists
                        arr = np.asarray(feature_value, dtype = np.float64)

                        # Handle NaN values safely
                        nan_mask = np.isnan(arr)
                        if nan_mask.sum() > 0:  # Use sum() instead of any() for safety
                            arr = np.nan_to_num(arr, nan = 0.0, posinf=0.0, neginf=0.0)
                            nan_count += int(nan_mask.sum())

                        # Handle infinite values safely
                        inf_mask = np.isinf(arr)
                        if inf_mask.sum() > 0:  # Use sum() instead of any() for safety
                            arr = np.nan_to_num(arr, nan = 0.0, posinf=0.0, neginf=0.0)
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
                    f"🔧 Inline NaN handling: {nan_count} NaN values = {inf_count} inf values cleaned",
                )

            return cleaned_features

        except Exception as e:
            self.logger.exception(f"🚨 Error in inline NaN handling: {e}")
            return features

    @validate_step_prerequisites(
        required_directories=["data_cache", "data/feature_cache"],
        min_memory_gb=16.0,
        min_disk_gb=10.0,
        required_packages=["pandas", "numpy", "pywt", "scipy"],
        data_quality_checks={
            "min_rows": 1000,
            "required_columns": ["timestamp", "open", "high", "low", "close", "volume"],
        },
        context="Vectorized Advanced Feature Engineering",
    )
    @secure_data_processing(
        backup_before, True = integrity_checks=True,
        memory_cleanup, True = data_validation=True,
    )
    @prevent_data_leakage(
        temporal_validation, True = feature_leakage_detection=True,
        cross_validation_isolation, True = lookahead_bias_prevention=True,
    )
    @resource_monitor(
        memory_threshold_gb=32.0,
        cpu_threshold_percent=90.0,
        disk_threshold_gb=20.0,
        monitor_interval=60.0,
        auto_cleanup, True = )
    @memory_efficient(
        chunk_size=5000,
        streaming_processing, True = memory_pool=True,
        cleanup_frequency=20,
    )
    @debug_training_step(
        log_intermediate_results=True,
        save_debug_artifacts=True,
        performance_profiling, True = error_context_preservation=True,
    )
    @circuit_breaker_protection(
        failure_threshold=3,
        recovery_timeout=600.0,
        expected_exception, Exception = monitor_interval=60.0,
    )
    @validate_step_output(
        required_files=["data/feature_cache/*.parquet"],
        data_quality_checks={
            "min_rows": 100,
            "required_columns": ["features", "metadata"],
        },
        performance_thresholds={
            "feature_engineering_time_minutes": 120.0,
            "memory_usage_gb": 16.0,
        },
        format_validation, True = )
    @quality_gate(
        model_performance_thresholds={
            "feature_quality": 0.8,
            "feature_completeness": 0.9,
        },
        data_quality_metrics={"completeness": 0.9, "consistency": 0.8},
        convergence_checks, True = overfitting_detection=True,
        validation_score_requirements={"feature_engineering_score": 0.8},
    )
    @handle_errors(
        exceptions=(ValueError = AttributeError),
        default_return, None = context="vectorized advanced feature engineering",
    )
    @validate_step_prerequisites(
        required_directories=["data_cache", "data/feature_cache"],
        min_memory_gb=16.0,
        min_disk_gb=10.0,
        required_packages=["pandas", "numpy", "pywt", "scipy"],
        data_quality_checks={
            "min_rows": 1000,
            "required_columns": ["timestamp", "open", "high", "low", "close", "volume"],
        },
        context="Vectorized Advanced Feature Engineering",
    )
    @secure_data_processing(
        backup_before, True = integrity_checks=True,
        memory_cleanup, True = data_validation=True,
    )
    @prevent_data_leakage(
        temporal_validation, True = feature_leakage_detection=True,
        cross_validation_isolation, True = lookahead_bias_prevention=True,
    )
    @resource_monitor(
        memory_threshold_gb=32.0,
        cpu_threshold_percent=90.0,
        disk_threshold_gb=20.0,
        monitor_interval=60.0,
        auto_cleanup, True = )
    @memory_efficient(
        chunk_size=5000,
        streaming_processing, True = memory_pool=True,
        cleanup_frequency=20,
    )
    @debug_training_step(
        log_intermediate_results=True,
        save_debug_artifacts=True,
        performance_profiling, True = error_context_preservation=True,
    )
    @circuit_breaker_protection(
        failure_threshold=3,
        recovery_timeout=600.0,
        expected_exception, Exception = monitor_interval=60.0,
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
    #     format_validation, True = # )
    # @quality_gate(
    #     model_performance_thresholds={
    #         "feature_quality": 0.8,
    #         "feature_completeness": 0.9,
    #     },
    #     data_quality_metrics={"completeness": 0.9, "consistency": 0.8},
    #     convergence_checks, True = #     overfitting_detection=True,
    #     validation_score_requirements={"feature_engineering_score": 0.8},
    # )
    # @handle_errors(
    #     exceptions=(ValueError = AttributeError),
    #     default_return, None = #     context="vectorized advanced feature engineering",
    # )
    async def engineer_features(
        self = price_data: pd.DataFrame,
        volume_data: pd.DataFrame = order_flow_data: Optional[pd.DataFrame] = None,
        sr_levels: Optional[dict[str , Any]] = None,
    ) -> dict[str , Any]:
        """
        Engineer advanced features for improved prediction accuracy using vectorized operations.

        Args:
            price_data: OHLCV price data
            volume_data: Volume and trade flow data
            order_flow_data: Order book and flow data (optional)
            sr_levels: Support/resistance levels (optional)

        Returns:
            Dictionary containing engineered features
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
            self.logger.info(
                f"🔍 Input volume_data columns: {list(volume_data.columns)}",
            )

            # Log data quality metrics
            if not price_data.empty:
                self.logger.info(
                    f"🔍 Price data range: {price_data.min().min():.6f} to {price_data.max().max():.6f}",
                )
                self.logger.info(
                    f"🔍 Price data NaN count: {price_data.isna().sum().sum()}",
                )
            if not volume_data.empty:
                self.logger.info(
                    f"🔍 Volume data range: {volume_data.min().min():.6f} to {volume_data.max().max():.6f}",
                )
                self.logger.info(
                    f"🔍 Volume data NaN count: {volume_data.isna().sum().sum()}",
                )

            # Log order flow data if available
            if order_flow_data is not None:
                self.logger.info(f"🔍 Order flow data shape: {order_flow_data.shape}")
                self.logger.info(
                    f"🔍 Order flow data columns: {list(order_flow_data.columns)}",
                )
            else:
                self.logger.info("🔍 No order flow data provided")

            # Preprocess irregular intervals before feature engineering
                RawDataQualityChecker = )

            # Initialize data quality checker
            quality_checker = RawDataQualityChecker()

            # Preprocess price data to handle irregular intervals
            self.logger.info("🔧 Preprocessing price data for irregular intervals...")
            # Use enhanced preprocessing with intelligent gap handling
            symbol = getattr(self = "symbol", "ETHUSDT")
            exchange = getattr(self = "exchange", "BINANCE")

            # Ensure price_data has a proper DatetimeIndex
            if not isinstance(price_data.index , pd.DatetimeIndex):
                self.logger.warning(
                    "⚠️ Price data doesn't have DatetimeIndex = attempting to fix...",
                )
                if "timestamp" in price_data.columns:
                    # Convert timestamp column to DatetimeIndex
                    price_data = price_data.set_index("timestamp")
                    self.logger.info("✅ Set timestamp column as DatetimeIndex")
                elif price_data.index.name == "timestamp":
                    # Convert index to DatetimeIndex
                    price_data.index = pd.to_datetime(price_data.index)
                    self.logger.info("✅ Converted index to DatetimeIndex")
                else:
                    # Try to convert the existing index to datetime if it looks like timestamps
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
                        if price_data.index.dtype == "object" or str(
                            price_data.index.dtype,
                        ).startswith("datetime"):
                            # Try to parse the index as datetime
                            price_data.index = pd.to_datetime(price_data.index)
                            self.logger.info(
                                "✅ Converted existing index to DatetimeIndex",
                            )
                        else:
                            # Create a synthetic datetime index based on the data length
                            self.logger.warning(
                                "⚠️ Creating synthetic datetime index - verify data alignment",
                            )
                            start_time = pd.Timestamp("2024-01-01 00:00:00")
                            interval = pd.Timedelta(
                                minutes=1,
                            )  # Default to 1 minute intervals
                            timestamps = [
                                start_time + i * interval
                                for i in range(len(price_data))
                            ]
                            price_data.index = timestamps
                            self.logger.info("✅ Created synthetic datetime index")
                    except Exception as e:
                        self.logger.exception(f"❌ Failed to create DatetimeIndex: {e}")
                        return {}

            enhanced_price_data = quality_checker.enhanced_preprocess_market_data(
                price_data, symbol = symbol,
                exchange, exchange = expected_interval_seconds=60,  # 1-minute intervals
                max_forward_fill_seconds=10,  # Forward-fill gaps ≤10 seconds
                download_missing_data=True,  # Download data for gaps >10 seconds
            )

            preprocessed_price_data = enhanced_price_data
            price_validation = {
                "preprocessing_applied": {
                    "method": "enhanced",
                    "original_shape": price_data.shape , "preprocessed_shape": enhanced_price_data.shape,
                    "improvement": 0.0,
                },
            }

            # Log preprocessing results
            if price_validation.get("preprocessing_applied"):
                preprocessing_info = price_validation["preprocessing_applied"]
                self.logger.info("✅ Price data preprocessing completed:")
                self.logger.info(f"   Method: {preprocessing_info['method']}")
                self.logger.info(
                    f"   Original shape: {preprocessing_info['original_shape']}",
                )
                self.logger.info(
                    f"   Preprocessed shape: {preprocessing_info['preprocessed_shape']}",
                )
                self.logger.info(
                    f"   Quality improvement: {preprocessing_info['improvement']:.3f}",
                )
                price_data = preprocessed_price_data
            else:
                self.logger.info("✅ No price data preprocessing needed")

            # Enhanced preprocessing for volume data if it has timestamps
            if hasattr(volume_data = "index") and isinstance(
                volume_data.index = pd.DatetimeIndex,
            ):
                self.logger.info("🔧 Enhanced preprocessing for volume data...")

                enhanced_volume_data = quality_checker.enhanced_preprocess_market_data(
                    volume_data, symbol = symbol,
                    exchange, exchange = expected_interval_seconds=60,  # 1-minute intervals
                    max_forward_fill_seconds=10,  # Forward-fill gaps ≤10 seconds
                    download_missing_data=True,  # Download data for gaps >10 seconds
                )

                # Update volume_data with enhanced preprocessed data
                volume_data = enhanced_volume_data
                self.logger.info("✅ Volume data enhanced preprocessing completed")
            else:
                self.logger.info(
                    "🔧 Volume data doesn't have DatetimeIndex = attempting to fix...",
                )
                # Ensure volume_data has a proper DatetimeIndex
                if not isinstance(volume_data.index , pd.DatetimeIndex):
                    if "timestamp" in volume_data.columns:
                        # Convert timestamp column to DatetimeIndex
                        volume_data = volume_data.set_index("timestamp")
                        self.logger.info(
                            "✅ Set timestamp column as DatetimeIndex for volume data",
                        )
                    elif volume_data.index.name == "timestamp":
                        # Convert index to DatetimeIndex
                        volume_data.index = pd.to_datetime(volume_data.index)
                        self.logger.info(
                            "✅ Converted volume data index to DatetimeIndex",
                        )
                    else:
                        # Try to convert the existing index to datetime if it looks like timestamps
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
                            if volume_data.index.dtype == "object" or str(
                                volume_data.index.dtype,
                            ).startswith("datetime"):
                                # Try to parse the index as datetime
                                volume_data.index = pd.to_datetime(volume_data.index)
                                self.logger.info(
                                    "✅ Converted existing volume data index to DatetimeIndex",
                                )
                            # Try to align volume data with price data index
                            elif hasattr(price_data = "index") and isinstance(
                                price_data.index = pd.DatetimeIndex,
                            ):
                                volume_data = volume_data.reindex(
                                    price_data.index, method = "ffill",
                                )
                                self.logger.info(
                                    "✅ Aligned volume data with price data index",
                                )
                            else:
                                self.logger.warning(
                                    "⚠️ Cannot determine timestamp column for volume data = skipping preprocessing",
                                )
                        except Exception as e:
                            self.logger.warning(
                                f"⚠️ Failed to fix volume data DatetimeIndex: {e}, skipping preprocessing",
                            )

            # Validate and transform data to ensure OHLCV structure
            price_data, volume_data = self._validate_and_transform_data(
                price_data = volume_data,
            )

            # Track NaN origins in input data
            self._track_nan_origins(
                "feature_engineering_input",
                {
                    "price_data": price_data , "volume_data": volume_data,
                    "order_flow_data": order_flow_data = },
            )

            features = {}

            # Debug: Log input data information before feature generation
            self.logger.info("🔍 Input data validation before feature generation:")
            self.logger.info(
                f"   Price data shape: {price_data.shape if price_data is not None else 'None'}",
            )
            self.logger.info(
                f"   Volume data shape: {volume_data.shape if volume_data is not None else 'None'}",
            )
            self.logger.info(
                f"   Order flow data shape: {order_flow_data.shape if order_flow_data is not None else 'None'}",
            )

            if price_data is not None and not price_data.empty:
                self.logger.info(
                    f"   Price data index: {price_data.index.min()} to {price_data.index.max()}",
                )
                self.logger.info(f"   Price data columns: {list(price_data.columns)}")
            else:
                self.logger.error("❌ Price data is empty or None")

            if volume_data is not None and not volume_data.empty:
                self.logger.info(
                    f"   Volume data index: {volume_data.index.min()} to {volume_data.index.max()}",
                )
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
            if not isinstance(price_data.index , pd.DatetimeIndex):
                self.logger.error("❌ Price data must have datetime index")
                return {}

            if not isinstance(volume_data.index , pd.DatetimeIndex):
                self.logger.error("❌ Volume data must have datetime index")
                return {}

            # Check for minimum data requirements
            if len(price_data) < 10:
                self.logger.error(
                    f"❌ Insufficient price data: {len(price_data)} records (minimum: 10)",
                )
                return {}

            if len(volume_data) < 10:
                self.logger.error(
                    f"❌ Insufficient volume data: {len(volume_data)} records (minimum: 10)",
                )
                return {}

            self.logger.info("✅ Input data validation passed")
            self.logger.info("🔍 Starting feature generation pipeline...")

            # Add comprehensive coroutine detection and filtering

            def filter_coroutines(feature_dict: dict, source_name: str) -> dict:
                """Filter out any coroutine features from a feature dictionary."""
                if not isinstance(feature_dict , dict):
                    self.logger.warning(
                        f"⚠️ {source_name} is not a dict: {type(feature_dict)}",
                    )
                    return {}

                filtered_features = {}
                coroutine_count = 0
                for key , value in feature_dict.items():
                    if hasattr(value = "__await__"):
                        self.logger.warning(
                            f"⚠️ Skipping coroutine feature from {source_name}: {key}",
                        )
                        coroutine_count += 1
                        continue
                    filtered_features[key] = value

                if coroutine_count > 0:
                    self.logger.info(
                        f"⚠️ Filtered out {coroutine_count} coroutine features from {source_name}",
                    )

                return filtered_features

            # Market microstructure features
            self.logger.info("🔍 Generating microstructure features...")
            microstructure_features = (
                await self._engineer_microstructure_features_vectorized(
                    price_data = volume_data,
                    order_flow_data = )
            )
            self.logger.info(
                f"🔍 Generated {len(microstructure_features)} microstructure features",
            )
            if microstructure_features:
                self.logger.info(
                    f"🔍 Microstructure feature names: {list(microstructure_features.keys())}",
                )

            # Filter out any coroutine features before updating
            filtered_microstructure_features = filter_coroutines(
                microstructure_features = "microstructure",
            )
            features.update(filtered_microstructure_features)
            self.logger.info(f"🔍 Total features after microstructure: {len(features)}")

            # Context dynamics for raw contextual signals (avoid using raw magnitudes as features)
            self.logger.info("🔍 Generating context dynamics features...")
            self.logger.info(f"🔍 Available columns in price_data: {list(price_data.columns)}")
            
            # Ensure consolidated data exists for external features
            symbol = getattr(self = "symbol", "ETHUSDT")
            exchange = getattr(self = "exchange", "BINANCE")
            timeframe = getattr(self = "timeframe", "1m")
            
            # Check if we need consolidated data for external features
            if not hasattr(self = '_consolidation_checked'):
                self.logger.info("🔍 Checking for consolidated data availability...")
                await self._ensure_consolidated_data(symbol = exchange, timeframe)
                self._consolidation_checked = True
            
            context_features_count = 0
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
                idx = price_data.index
                
                # Generate missing context dynamics features from existing data
                context_features_count += self._generate_funding_rate_features(price_data = features, idx)
                context_features_count += self._generate_volume_ratio_features(price_data = features, idx)
                context_features_count += self._generate_trade_count_features(price_data = features, idx)
                context_features_count += self._generate_trade_volume_features(price_data = features, idx)
            except Exception as _e:
                self.logger.warning(f"⚠️ Context dynamics generation failed: {_e}")

            # Log summary of context dynamics feature generation
            self.logger.info(f"🔍 Context dynamics feature generation completed")
            self.logger.info(f"🔍 Generated {context_features_count} total context dynamics features")

            self.logger.info(
                f"🔍 Generated {context_features_count} context dynamics features",
            )
            self.logger.info(
                f"🔍 Total features after context dynamics: {len(features)}",
            )

            # Volatility regime features
            self.logger.info("🔍 Generating volatility regime features...")
            if self.volatility_model:
                volatility_features = (
                    await self.volatility_model.model_volatility_vectorized(
                        price_data = volume_data,
                    )
                )
                self.logger.info(
                    f"🔍 Generated {len(volatility_features)} volatility features",
                )
                if volatility_features:
                    self.logger.info(
                        f"🔍 Volatility feature names: {list(volatility_features.keys())}",
                    )
                # Ensure consistent numeric typing for downstream validation
                if "volatility_regime" in volatility_features:
                    vr = volatility_features["volatility_regime"]
                    if isinstance(vr , str):
                        mapping = {"low": 0, "medium": 1, "high": 2}
                        volatility_features["volatility_regime"] = mapping.get(vr = 1)
                # Filter out any coroutine features before updating
                filtered_volatility_features = filter_coroutines(
                    volatility_features = "volatility",
                )
                features.update(filtered_volatility_features)
                self.logger.info(f"🔍 Total features after volatility: {len(features)}")
            else:
                self.logger.warning("⚠️ Volatility model not available")

            # Correlation analysis features
            self.logger.info("🔍 Generating correlation analysis features...")
            if self.correlation_analyzer:
                correlation_features = (
                    await self.correlation_analyzer.analyze_correlations_vectorized(
                        price_data = )
                )
                self.logger.info(
                    f"🔍 Generated {len(correlation_features)} correlation features",
                )
                if correlation_features:
                    self.logger.info(
                        f"🔍 Correlation feature names: {list(correlation_features.keys())}",
                    )
                # Filter out any coroutine features before updating
                filtered_correlation_features = filter_coroutines(
                    correlation_features = "correlation",
                )
                features.update(filtered_correlation_features)
                self.logger.info(
                    f"🔍 Total features after correlation: {len(features)}",
                )
            else:
                self.logger.warning("⚠️ Correlation analyzer not available")

            # Momentum analysis features
            self.logger.info("🔍 Generating momentum analysis features...")
            if self.momentum_analyzer:
                momentum_features = (
                    await self.momentum_analyzer.analyze_momentum_vectorized(
                        price_data = volume_data,
                    )
                )
                self.logger.info(
                    f"🔍 Generated {len(momentum_features)} momentum features",
                )
                if momentum_features:
                    self.logger.info(
                        f"🔍 Momentum feature names: {list(momentum_features.keys())}",
                    )
                # Filter out any coroutine features before updating
                filtered_momentum_features = filter_coroutines(
                    momentum_features = "momentum",
                )
                features.update(filtered_momentum_features)
                self.logger.info(f"🔍 Total features after momentum: {len(features)}")
            else:
                self.logger.warning("⚠️ Momentum analyzer not available")

            # Liquidity analysis features
            self.logger.info("🔍 Generating liquidity analysis features...")
            if self.liquidity_analyzer:
                liquidity_features = (
                    await self.liquidity_analyzer.analyze_liquidity_vectorized(
                        price_data = volume_data,
                    )
                )
                self.logger.info(
                    f"🔍 Generated {len(liquidity_features)} liquidity features",
                )
                if liquidity_features:
                    self.logger.info(
                        f"🔍 Liquidity feature names: {list(liquidity_features.keys())}",
                    )
                # Filter out any coroutine features before updating
                filtered_liquidity_features = filter_coroutines(
                    liquidity_features = "liquidity",
                )
                features.update(filtered_liquidity_features)
                self.logger.info(f"🔍 Total features after liquidity: {len(features)}")
            else:
                self.logger.warning("⚠️ Liquidity analyzer not available")

            # Candlestick pattern features
            self.logger.info("🔍 Generating candlestick pattern features...")
            if self.candlestick_analyzer:
                candlestick_features = await self.candlestick_analyzer.analyze_patterns(
                    price_data = )
                self.logger.info(
                    f"🔍 Generated {len(candlestick_features)} candlestick features",
                )
                if candlestick_features:
                    self.logger.info(
                        f"🔍 Candlestick feature names: {list(candlestick_features.keys())}",
                    )
                # Filter out any coroutine features before updating
                filtered_candlestick_features = filter_coroutines(
                    candlestick_features = "candlestick",
                )
                features.update(filtered_candlestick_features)
                self.logger.info(
                    f"🔍 Total features after candlestick: {len(features)}",
                )
            else:
                self.logger.warning("⚠️ Candlestick analyzer not available")

            # Immediately alongside candlestick/pattern features (requires OHLCV):
            # Compute classic OHLCV-based indicators using actual prices
            self.logger.info("🔍 Generating OHLCV price features...")
            ohlcv_price_features = self._engineer_ohlcv_price_features_vectorized(
                price_data = )
            self.logger.info(
                f"🔍 Generated {len(ohlcv_price_features)} OHLCV price features",
            )
            if ohlcv_price_features:
                self.logger.info(
                    f"🔍 OHLCV price feature names: {list(ohlcv_price_features.keys())}",
                )
            # Filter out any coroutine features before updating
            filtered_ohlcv_price_features = filter_coroutines(
                ohlcv_price_features = "ohlcv_price",
            )
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
    pass
except Exception as e:
    pass
    pass
except Exception as e:
    pass
    pass
except Exception as e:
    pass
                        if "support" not in sr_levels and "support_levels" in sr_levels:
                            sr_levels = {
                                "support": [
                                    lvl["price"]
                                    if isinstance(lvl , dict) and "price" in lvl
                                    else float(lvl)
                                    for lvl in sr_levels.get("support_levels", [])
                                ],
                                "resistance": [
                                    lvl["price"]
                                    if isinstance(lvl , dict) and "price" in lvl
                                    else float(lvl)
                                    for lvl in sr_levels.get("resistance_levels", [])
                                ],
                            }
                        elif "support" in sr_levels:
                            # Ensure numeric arrays
                            for k in ("support", "resistance"):
                                if k in sr_levels:
                                    vals = sr_levels[k]
                                    if isinstance(vals , list):
                                        sr_levels[k] = [
                                            v["price"]
                                            if isinstance(v , dict) and "price" in v
                                            else float(v)
                                            for v in vals
                                        ]
                    except Exception as _e:
                        self.logger.warning(
                            f"⚠️ Failed to normalize provided SR levels = will attempt auto-generation instead: {_e}",
                        )
                        sr_levels = self._generate_sr_levels(price_data)

                if sr_levels:
                    sr_distance_features = (
                        await self.sr_distance_calculator.calculate_sr_distances(
                            price_data = sr_levels,
                        )
                    )
                    self.logger.info(
                        f"🔍 Generated {len(sr_distance_features)} S/R distance features",
                    )
                    if sr_distance_features:
                        self.logger.info(
                            f"🔍 S/R distance feature names: {list(sr_distance_features.keys())}",
                        )
                    features.update(sr_distance_features)
                    self.logger.info(
                        f"🔍 Total features after S/R distance: {len(features)}",
                    )
                    
                    # Comprehensive SR features (sr_proximity_score = sr_score, delta_sr_score)
                    self.logger.info("🔍 Generating comprehensive SR features...")
                    comprehensive_sr_features = await self._generate_comprehensive_sr_features(
                        price_data = sr_levels,
                    )
                    if comprehensive_sr_features:
                        self.logger.info(
                            f"🔍 Generated {len(comprehensive_sr_features)} comprehensive SR features",
                        )
                        self.logger.info(
                            f"🔍 Comprehensive SR feature names: {list(comprehensive_sr_features.keys())}",
                        )
                        features.update(comprehensive_sr_features)
                        self.logger.info(
                            f"🔍 Total features after comprehensive SR: {len(features)}",
                        )
                    else:
                        self.logger.warning("⚠️ No comprehensive SR features generated")
                else:
                    self.logger.warning(
                        "⚠️ Failed to generate S/R levels = skipping S/R distance features",
                    )
            else:
                self.logger.warning("⚠️ S/R distance calculator not available")

            # Wavelet transform features with caching
            self.logger.info("🔍 Generating wavelet transform features...")
            if self.wavelet_analyzer:
                wavelet_features = await self._get_wavelet_features_with_caching(
                    price_data = volume_data,
                )
                self.logger.info(
                    f"🔍 Generated {len(wavelet_features)} wavelet features",
                )
                if wavelet_features:
                    self.logger.info(
                        f"🔍 Wavelet feature names: {list(wavelet_features.keys())}",
                    )
                features.update(wavelet_features)
                self.logger.info(f"🔍 Total features after wavelet: {len(features)}")
            else:
                self.logger.warning("⚠️ Wavelet analyzer not available")

            # Adaptive indicators
            self.logger.info("🔍 Generating adaptive indicators...")
            adaptive_features = self._engineer_adaptive_indicators_vectorized(
                price_data = )
            self.logger.info(f"🔍 Generated {len(adaptive_features)} adaptive features")
            if adaptive_features:
                self.logger.info(
                    f"🔍 Adaptive feature names: {list(adaptive_features.keys())}",
                )
            features.update(adaptive_features)
            self.logger.info(
                f"🔍 Total features after adaptive indicators: {len(features)}",
            )

            # Debug: Log feature generation before selection
            self.logger.info(f"🔍 Generated {len(features)} features before selection")
            if len(features) < 10:
                self.logger.warning(
                    f"⚠️ Very few features generated before selection: {list(features.keys())}",
                )

                # Add basic features as fallback to ensure we have features
                self.logger.info("⚠️ No fallback")

            # Feature selection and dimensionality reduction
            # Re-enable feature selection for comprehensive feature engineering
            selected_features = self._select_optimal_features_vectorized(features)
            self.logger.info(
                "🔍 Feature selection re-enabled for comprehensive feature engineering",
            )

            # Debug: Log feature selection results
            self.logger.info(
                f"🔍 Selected {len(selected_features)} features after selection",
            )
            if len(selected_features) < 10:
                self.logger.warning(
                    f"⚠️ Very few features selected: {list(selected_features.keys())}",
                )

            # Add multi-timeframe features if enabled
            if self.enable_multi_timeframe:
                self.logger.info("🔍 Generating multi-timeframe features...")
                multi_timeframe_features = (
                    await self._engineer_multi_timeframe_features_vectorized(
                        price_data = volume_data,
                        order_flow_data = sr_levels,
                    )
                )
                self.logger.info(
                    f"🔍 Generated {len(multi_timeframe_features)} multi-timeframe features",
                )
                if multi_timeframe_features:
                    self.logger.info(
                        f"🔍 Multi-timeframe feature names: {list(multi_timeframe_features.keys())}",
                    )
                # Filter out any coroutine features from multi_timeframe_features before updating
                filtered_multi_timeframe_features = filter_coroutines(
                    multi_timeframe_features = "multi_timeframe",
                )
                selected_features.update(filtered_multi_timeframe_features)
                self.logger.info(
                    f"🔍 Total features after multi-timeframe: {len(selected_features)}",
                )
            else:
                self.logger.info("🔍 Multi-timeframe features disabled")

            # Meta-labeling deprecated
            self.logger.info("ℹ️ Meta-labeling is deprecated and disabled")

            # Explicit meta-labels deprecated
            self.logger.info("ℹ️ Explicit meta-labels are deprecated and disabled")

            # Enforce generator contract: ensure all values are 1D arrays of length n
            n = len(price_data)
            sanitized: dict[str , Any] = {}
            offenders: list[str] , []
            for k , v in selected_features.items():
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
                    if isinstance(v , pd.Series):
                        arr = v.values.reshape(-1)
                    elif isinstance(v , np.ndarray):
                        arr = v.reshape(-1) if v.ndim >= 1 else None
                    elif isinstance(v , list):
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
                        arr = np.concatenate([np.full(pad = np.nan), arr])
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

            # Generate feature interactions
            interaction_features = (
                await self.feature_interaction_engine.generate_interactions(
                    sanitized = )
            )
            original_feature_count = len(sanitized)
            sanitized.update(interaction_features)
            interaction_count = len(sanitized) - original_feature_count

            self.logger.info(
                f"✅ Engineered {len(sanitized)} advanced features (including {interaction_count} interaction terms)",
            )
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
                self.logger.info(
                    f"🧾 Feature list ({len(sanitized)}): {sorted(sanitized.keys())}",
                )
            except Exception as e:
                self.logger.warning(f"Failed to log feature list: {e}")
            return sanitized

        except Exception as e:
            self.logger.error(f"Error engineering advanced features: {e}")
            return {}
