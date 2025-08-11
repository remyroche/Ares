# src/training/steps/vectorized_advanced_feature_engineering.py

"""
Vectorized Advanced Feature Engineering for enhanced financial performance.
Implements sophisticated market microstructure features, regime detection,
and adaptive indicators for improved prediction accuracy with vectorized operations.
"""

import numpy as np
import pandas as pd
import pywt
from typing import Any, Dict, List, Optional, Tuple
from datetime import timedelta
import time
import random
import os
import hashlib
import json
from pathlib import Path

from src.utils.error_handler import handle_errors
from src.utils.logger import system_logger


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
        self.cache_format = self.cache_config.get("cache_format", "parquet")  # parquet, feather, h5
        self.compression = self.cache_config.get("compression", "snappy")
        self.cache_metadata = self.cache_config.get("cache_metadata", True)
        
        # Cache validation
        self.validate_cache_integrity = self.cache_config.get("validate_cache_integrity", True)
        self.cache_expiry_days = self.cache_config.get("cache_expiry_days", 30)
        
        # Performance settings
        self.enable_parallel_caching = self.cache_config.get("enable_parallel_caching", False)
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
            self.logger.error(f"❌ Error initializing cache directory: {e}")

    def generate_cache_key(
        self,
        price_data: pd.DataFrame,
        wavelet_config: dict[str, Any],
        additional_params: dict[str, Any] | None = None,
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
            # Create a hashable representation of the data
            data_hash = self._hash_dataframe(price_data)
            
            # Create configuration hash
            config_str = json.dumps(wavelet_config, sort_keys=True)
            config_hash = hashlib.md5(config_str.encode()).hexdigest()
            
            # Create additional parameters hash
            params_hash = ""
            if additional_params:
                params_str = json.dumps(additional_params, sort_keys=True)
                params_hash = hashlib.md5(params_str.encode()).hexdigest()
            
            # Combine hashes
            combined_hash = f"{data_hash}_{config_hash}_{params_hash}"
            
            # Create final cache key
            cache_key = hashlib.sha256(combined_hash.encode()).hexdigest()[:16]
            
            return cache_key
            
        except Exception as e:
            self.logger.error(f"Error generating cache key: {e}")
            return "default_cache_key"

    def _hash_dataframe(self, df: pd.DataFrame) -> str:
        """Generate hash for DataFrame content."""
        try:
            # Convert DataFrame to bytes for hashing
            df_bytes = df.to_string().encode()
            return hashlib.md5(df_bytes).hexdigest()
            
        except Exception as e:
            self.logger.error(f"Error hashing DataFrame: {e}")
            return "default_hash"

    def get_cache_filepath(self, cache_key: str) -> Tuple[Path, Path]:
        """
        Get file paths for cache files.
        
        Args:
            cache_key: Unique cache key
            
        Returns:
            Tuple of (features_filepath, metadata_filepath)
        """
        try:
            cache_path = Path(self.cache_dir)
            
            if self.cache_format == "parquet":
                features_file = cache_path / "features" / f"{cache_key}_features.parquet"
                metadata_file = cache_path / "metadata" / f"{cache_key}_metadata.json"
            elif self.cache_format == "feather":
                features_file = cache_path / "features" / f"{cache_key}_features.feather"
                metadata_file = cache_path / "metadata" / f"{cache_key}_metadata.json"
            elif self.cache_format == "h5":
                features_file = cache_path / "features" / f"{cache_key}_features.h5"
                metadata_file = cache_path / "metadata" / f"{cache_key}_metadata.json"
            else:
                features_file = cache_path / "features" / f"{cache_key}_features.parquet"
                metadata_file = cache_path / "metadata" / f"{cache_key}_metadata.json"
            
            return features_file, metadata_file
            
        except Exception as e:
            self.logger.error(f"Error getting cache filepath: {e}")
            return Path(""), Path("")

    def cache_exists(self, cache_key: str) -> bool:
        """
        Check if cache exists and is valid.
        
        Args:
            cache_key: Unique cache key
            
        Returns:
            True if valid cache exists, False otherwise
        """
        try:
            features_file, metadata_file = self.get_cache_filepath(cache_key)
            
            # Check if files exist
            if not features_file.exists() or not metadata_file.exists():
                return False
            
            # Check cache expiry
            if self.cache_expiry_days > 0:
                file_age = time.time() - features_file.stat().st_mtime
                if file_age > (self.cache_expiry_days * 24 * 3600):
                    self.logger.info(f"Cache expired for key: {cache_key}")
                    return False
            
            # Validate cache integrity if enabled
            if self.validate_cache_integrity:
                return self._validate_cache_integrity(cache_key)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error checking cache existence: {e}")
            return False

    def _validate_cache_integrity(self, cache_key: str) -> bool:
        """Validate cache file integrity."""
        try:
            features_file, metadata_file = self.get_cache_filepath(cache_key)
            
            # Check file sizes
            if features_file.stat().st_size == 0:
                self.logger.warning(f"Cache file is empty: {features_file}")
                return False
            
            # Try to read metadata
            try:
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                
                # Validate metadata structure
                required_keys = ["cache_key", "timestamp", "data_shape", "feature_count"]
                if not all(key in metadata for key in required_keys):
                    self.logger.warning(f"Invalid metadata structure for key: {cache_key}")
                    return False
                
                # Validate cache key match
                if metadata.get("cache_key") != cache_key:
                    self.logger.warning(f"Cache key mismatch for key: {cache_key}")
                    return False
                
                return True
                
            except Exception as e:
                self.logger.warning(f"Error reading cache metadata: {e}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error validating cache integrity: {e}")
            return False

    def save_to_cache(
        self,
        cache_key: str,
        features: dict[str, Any],
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        """
        Save wavelet features to cache.
        
        Args:
            cache_key: Unique cache key
            features: Wavelet features to cache
            metadata: Additional metadata
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if not self.cache_enabled:
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
                    features_file,
                    compression=self.compression,
                    index=True
                )
            elif self.cache_format == "feather":
                features_df.to_feather(features_file)
            elif self.cache_format == "h5":
                features_df.to_hdf(features_file, key="wavelet_features", mode="w")
            
            # Save metadata
            with open(metadata_file, 'w') as f:
                json.dump(cache_metadata, f, indent=2)
            
            self.logger.info(f"✅ Cached {len(features)} wavelet features to {features_file}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving to cache: {e}")
            return False

    def load_from_cache(self, cache_key: str) -> Tuple[dict[str, Any], dict[str, Any] | None]:
        """
        Load wavelet features from cache.
        
        Args:
            cache_key: Unique cache key
            
        Returns:
            Tuple of (features, metadata)
        """
        try:
            features_file, metadata_file = self.get_cache_filepath(cache_key)
            
            # Load features based on format
            if self.cache_format == "parquet":
                features_df = pd.read_parquet(features_file)
            elif self.cache_format == "feather":
                features_df = pd.read_feather(features_file)
            elif self.cache_format == "h5":
                features_df = pd.read_hdf(features_file, key="wavelet_features")
            else:
                features_df = pd.read_parquet(features_file)
            
            # Convert DataFrame back to features dictionary
            features = self._dataframe_to_features(features_df)
            
            # Load metadata
            metadata = None
            if metadata_file.exists():
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
            
            self.logger.info(f"✅ Loaded {len(features)} wavelet features from cache: {cache_key}")
            return features, metadata
            
        except Exception as e:
            self.logger.error(f"Error loading from cache: {e}")
            return {}, None

    def _features_to_dataframe(self, features: dict[str, Any]) -> pd.DataFrame:
        """Convert features dictionary to DataFrame for caching."""
        try:
            # Convert features to DataFrame format
            if features:
                # Handle different feature types
                feature_data = {}
                for key, value in features.items():
                    if isinstance(value, (int, float, np.number)):
                        feature_data[key] = [value]
                    elif isinstance(value, (list, np.ndarray)):
                        feature_data[key] = value
                    else:
                        feature_data[key] = [str(value)]
                
                df = pd.DataFrame(feature_data)
            else:
                df = pd.DataFrame()
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error converting features to DataFrame: {e}")
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
            self.logger.error(f"Error converting DataFrame to features: {e}")
            return {}

    def clear_cache(self, cache_key: str | None = None) -> bool:
        """
        Clear cache files.
        
        Args:
            cache_key: Specific cache key to clear, or None to clear all
            
        Returns:
            True if successful, False otherwise
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
                self.logger.info(f"Cleared cache for key: {cache_key}")
            else:
                # Clear all cache
                for file_path in cache_path.rglob("*"):
                    if file_path.is_file():
                        file_path.unlink()
                self.logger.info("Cleared all cache files")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error clearing cache: {e}")
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
                    stats["total_size_mb"] = sum(f.stat().st_size for f in files) / (1024 * 1024)
                    
                    # File timestamps
                    timestamps = [f.stat().st_mtime for f in files]
                    stats["oldest_file"] = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(min(timestamps)))
                    stats["newest_file"] = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(max(timestamps)))
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Error getting cache stats: {e}")
            return {}


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
        self.enable_volatility_modeling = self.feature_config.get("enable_volatility_modeling", True)
        self.enable_correlation_analysis = self.feature_config.get("enable_correlation_analysis", True)
        self.enable_momentum_analysis = self.feature_config.get("enable_momentum_analysis", True)
        self.enable_liquidity_analysis = self.feature_config.get("enable_liquidity_analysis", True)
        self.enable_candlestick_patterns = self.feature_config.get("enable_candlestick_patterns", True)
        self.enable_sr_distance = self.feature_config.get("enable_sr_distance", True)
        self.enable_wavelet_transforms = self.feature_config.get("enable_wavelet_transforms", True)
        self.enable_multi_timeframe = self.feature_config.get("enable_multi_timeframe", True)
        self.enable_meta_labeling = self.feature_config.get("enable_meta_labeling", True)

        # Multi-timeframe configuration
        self.timeframes = ["1m", "5m", "15m", "30m"]

        # Initialize components
        self.volatility_model = None
        self.correlation_analyzer = None
        self.momentum_analyzer = None
        self.liquidity_analyzer = None
        self.candlestick_analyzer = None
        self.sr_distance_calculator = None
        self.wavelet_analyzer = None
        self.wavelet_cache = None

        self.is_initialized = False

    @handle_errors(
        exceptions=(Exception,),
        default_return=False,
        context="vectorized advanced feature engineering initialization",
    )
    async def initialize(self) -> bool:
        """Initialize vectorized advanced feature engineering components."""
        try:
            self.logger.info("🚀 Initializing vectorized advanced feature engineering...")

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
                self.liquidity_analyzer = VectorizedLiquidityAnalyzer(self.config)
                await self.liquidity_analyzer.initialize()

            # Initialize candlestick pattern analyzer
            if self.enable_candlestick_patterns:
                self.candlestick_analyzer = VectorizedCandlestickPatternAnalyzer(self.config)
                await self.candlestick_analyzer.initialize()

            # Initialize S/R distance calculator
            if self.enable_sr_distance:
                self.sr_distance_calculator = VectorizedSRDistanceCalculator(self.config)
                await self.sr_distance_calculator.initialize()

            # Initialize wavelet transform analyzer
            if self.enable_wavelet_transforms:
                self.wavelet_analyzer = VectorizedWaveletTransformAnalyzer(self.config)
                await self.wavelet_analyzer.initialize()

            # Initialize meta-labeling system
            if self.enable_meta_labeling:
                from src.analyst.meta_labeling_system import MetaLabelingSystem

                self.meta_labeling_system = MetaLabelingSystem(self.config)
                await self.meta_labeling_system.initialize()

            self.is_initialized = True
            self.logger.info("✅ Vectorized advanced feature engineering initialized successfully")
            return True

        except Exception as e:
            self.logger.error(
                f"❌ Error initializing vectorized advanced feature engineering: {e}",
            )
            return False

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="vectorized advanced feature engineering",
    )
    async def engineer_features(
        self,
        price_data: pd.DataFrame,
        volume_data: pd.DataFrame,
        order_flow_data: pd.DataFrame | None = None,
        sr_levels: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
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
            if not self.is_initialized:
                self.logger.error("Vectorized advanced feature engineering not initialized")
                return {}

            # Validate and transform data to ensure OHLCV structure
            price_data, volume_data = self._validate_and_transform_data(price_data, volume_data)

            features = {}

            # Market microstructure features
            microstructure_features = await self._engineer_microstructure_features_vectorized(
                price_data,
                volume_data,
                order_flow_data,
            )
            features.update(microstructure_features)

            # Volatility regime features
            if self.volatility_model:
                volatility_features = await self.volatility_model.model_volatility_vectorized(
                    price_data,
                )
                # Ensure consistent numeric typing for downstream validation
                if "volatility_regime" in volatility_features:
                    vr = volatility_features["volatility_regime"]
                    if isinstance(vr, str):
                        mapping = {"low": 0, "medium": 1, "high": 2}
                        volatility_features["volatility_regime"] = mapping.get(vr, 1)
                features.update(volatility_features)

            # Correlation analysis features
            if self.correlation_analyzer:
                correlation_features = (
                    await self.correlation_analyzer.analyze_correlations_vectorized(price_data)
                )
                features.update(correlation_features)

            # Momentum analysis features
            if self.momentum_analyzer:
                momentum_features = await self.momentum_analyzer.analyze_momentum_vectorized(
                    price_data,
                )
                features.update(momentum_features)

            # Liquidity analysis features
            if self.liquidity_analyzer:
                liquidity_features = await self.liquidity_analyzer.analyze_liquidity_vectorized(
                    price_data,
                    volume_data,
                    order_flow_data,
                )
                features.update(liquidity_features)

            # Candlestick pattern features
            if self.candlestick_analyzer:
                candlestick_features = await self.candlestick_analyzer.analyze_patterns(
                    price_data,
                )
                features.update(candlestick_features)

            # S/R distance features
            if self.sr_distance_calculator and sr_levels:
                sr_distance_features = await self.sr_distance_calculator.calculate_sr_distances(
                    price_data,
                    sr_levels,
                )
                features.update(sr_distance_features)

            # Wavelet transform features with caching
            if self.wavelet_analyzer:
                wavelet_features = await self._get_wavelet_features_with_caching(
                    price_data,
                    volume_data,
                )
                features.update(wavelet_features)

            # Adaptive indicators
            adaptive_features = self._engineer_adaptive_indicators_vectorized(price_data)
            features.update(adaptive_features)

            # Feature selection and dimensionality reduction
            selected_features = self._select_optimal_features_vectorized(features)

            # Add multi-timeframe features if enabled
            if self.enable_multi_timeframe:
                multi_timeframe_features = (
                    await self._engineer_multi_timeframe_features_vectorized(
                        price_data,
                        volume_data,
                        order_flow_data,
                        sr_levels,
                    )
                )
                selected_features.update(multi_timeframe_features)

            # Add meta-labeling if enabled
            if self.enable_meta_labeling:
                meta_labels = await self._generate_meta_labels_vectorized(
                    price_data,
                    volume_data,
                    order_flow_data,
                )
                selected_features.update(meta_labels)

            # Enforce generator contract: ensure all values are 1D arrays of length n
            n = len(price_data)
            sanitized: dict[str, Any] = {}
            offenders: list[str] = []
            for k, v in selected_features.items():
                try:
                    if isinstance(v, pd.Series):
                        arr = v.values.reshape(-1)
                    elif isinstance(v, np.ndarray):
                        arr = v.reshape(-1) if v.ndim >= 1 else None
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
                        arr = np.concatenate([np.full(pad, np.nan), arr])
                    sanitized[k] = arr
                except Exception:
                    offenders.append(k)
                    continue

            if offenders:
                self.logger.warning(
                    f"⚠️ Feature generator contract: skipped scalar/invalid outputs for features: {offenders[:20]}" +
                    (" ..." if len(offenders) > 20 else "")
                )

            self.logger.info(
                f"✅ Engineered {len(sanitized)} vectorized advanced features including wavelet transforms",
            )
            return sanitized

        except Exception as e:
            self.logger.error(f"Error engineering vectorized advanced features: {e}")
            return {}

    async def _get_wavelet_features_with_caching(
        self,
        price_data: pd.DataFrame,
        volume_data: pd.DataFrame,
    ) -> dict[str, Any]:
        """
        Get wavelet features with caching support.
        
        Args:
            price_data: OHLCV price data
            volume_data: Volume data
            
        Returns:
            Dictionary containing wavelet features
        """
        try:
            if not self.wavelet_cache:
                # Fallback to direct computation if cache is not available
                return await self.wavelet_analyzer.analyze_wavelet_transforms(
                    price_data,
                    volume_data,
                )

            # Generate cache key
            wavelet_config = self.wavelet_analyzer.wavelet_config
            cache_key = self.wavelet_cache.generate_cache_key(
                price_data,
                wavelet_config,
                {"volume_data_shape": volume_data.shape if volume_data is not None else None}
            )

            # Check if cache exists
            if self.wavelet_cache.cache_exists(cache_key):
                self.logger.info(f"📦 Loading wavelet features from cache: {cache_key}")
                cached_features, metadata = self.wavelet_cache.load_from_cache(cache_key)
                return cached_features

            # Compute wavelet features
            self.logger.info(f"🔧 Computing wavelet features (not cached): {cache_key}")
            wavelet_features = await self.wavelet_analyzer.analyze_wavelet_transforms(
                price_data,
                volume_data,
            )

            # Save to cache
            metadata = {
                "data_shape": price_data.shape,
                "volume_data_shape": volume_data.shape if volume_data is not None else None,
                "computation_time": time.time(),
            }
            
            cache_success = self.wavelet_cache.save_to_cache(cache_key, wavelet_features, metadata)
            if cache_success:
                self.logger.info(f"💾 Cached wavelet features: {cache_key}")
            else:
                self.logger.warning(f"⚠️ Failed to cache wavelet features: {cache_key}")

            return wavelet_features

        except Exception as e:
            self.logger.error(f"Error getting wavelet features with caching: {e}")
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
        self,
        price_data: pd.DataFrame,
        volume_data: pd.DataFrame,
        order_flow_data: pd.DataFrame | None = None,
    ) -> dict[str, Any]:
        """Engineer market microstructure features using vectorized operations."""
        try:
            features = {}

            # Price impact features (vectorized per-row)
            features["price_impact"] = self._calculate_price_impact_vectorized(price_data, volume_data)
            features["volume_price_impact"] = self._calculate_volume_price_impact_vectorized(price_data, volume_data)

            # Order-flow related features (proxies if book data not available)
            features["order_flow_imbalance"] = self._calculate_order_flow_imbalance_vectorized(
                price_data, volume_data, order_flow_data
            )
            features["bid_ask_spread"] = self._calculate_bid_ask_spread_vectorized(price_data)

            # Market depth features (vectorized per-row)
            features["market_depth"] = self._calculate_market_depth_vectorized(price_data, volume_data)

            return features

        except Exception as e:
            self.logger.error(f"Error engineering microstructure features: {e}")
            return {}

    def _calculate_price_impact_vectorized(self, price_data: pd.DataFrame, volume_data: pd.DataFrame) -> pd.Series:
        """Calculate per-row price impact using abs(close diff) normalized by rolling average volume."""
        try:
            price_diff = price_data["close"].diff().abs()
            avg_volume = (
                volume_data["volume"].rolling(window=20, min_periods=1).mean().replace(0, np.nan)
            )
            impact = (price_diff / avg_volume).fillna(0)
            return impact
        except Exception as e:
            self.logger.error(f"Error calculating price impact: {e}")
            return pd.Series(np.zeros(len(price_data)), index=price_data.index)

    def _calculate_volume_price_impact_vectorized(self, price_data: pd.DataFrame, volume_data: pd.DataFrame) -> pd.Series:
        """Calculate per-row volume-price impact: volume-weighted price diff normalized by rolling volume sum."""
        try:
            price_diff = price_data["close"].diff()
            rolling_vol_sum = volume_data["volume"].rolling(window=20, min_periods=1).sum().replace(0, np.nan)
            weights = (volume_data["volume"] / rolling_vol_sum).replace([np.inf, -np.inf], np.nan)
            vpi = (price_diff * weights).fillna(0)
            return vpi
        except Exception as e:
            self.logger.error(f"Error calculating volume-price impact: {e}")
            return pd.Series(np.zeros(len(price_data)), index=price_data.index)

    def _calculate_order_flow_imbalance_vectorized(
        self,
        price_data: pd.DataFrame,
        volume_data: pd.DataFrame,
        order_flow_data: pd.DataFrame | None = None,
    ) -> pd.Series:
        """Vectorized order flow imbalance using tick rule; uses book volumes if provided.

        OFI ≈ sum(sign(Δp) * volume) / sum(volume) over a rolling window.
        If order_flow_data has explicit buy/sell volumes, use (buy - sell) / (buy + sell).
        """
        try:
            idx = price_data.index
            # If explicit buy/sell volumes exist, use them
            if order_flow_data is not None:
                cols = {c.lower(): c for c in order_flow_data.columns}
                buy_col = cols.get("buy_volume") or cols.get("bid_volume") or cols.get("buy")
                sell_col = cols.get("sell_volume") or cols.get("ask_volume") or cols.get("sell")
                if buy_col and sell_col:
                    buy = pd.Series(order_flow_data[buy_col].values, index=order_flow_data.index).reindex(idx).fillna(0)
                    sell = pd.Series(order_flow_data[sell_col].values, index=order_flow_data.index).reindex(idx).fillna(0)
                    denom = (buy + sell).replace(0, np.nan)
                    ofi = ((buy - sell) / denom).fillna(0)
                    return ofi
            # Tick rule proxy from close and volume
            close = price_data["close"].astype(float)
            vol = volume_data["volume"].astype(float)
            price_delta = close.diff().fillna(0)
            trade_sign = np.sign(price_delta)
            signed_vol = trade_sign * vol
            win = 20
            num = signed_vol.rolling(win, min_periods=1).sum()
            den = vol.rolling(win, min_periods=1).sum().replace(0, np.nan)
            ofi = (num / den).replace([np.inf, -np.inf], np.nan).fillna(0)
            return ofi
        except Exception as e:
            self.logger.error(f"Error calculating order flow imbalance: {e}")
            return pd.Series(np.zeros(len(price_data)), index=price_data.index)

    def _calculate_bid_ask_spread_vectorized(self, price_data: pd.DataFrame) -> pd.Series:
        """Estimate relative bid-ask spread using Corwin–Schultz with Roll as fallback (per-row)."""
        try:
            close = price_data["close"].astype(float)
            high = price_data["high"].astype(float)
            low = price_data["low"].astype(float)
            # Corwin–Schultz estimator components
            with np.errstate(divide='ignore', invalid='ignore'):
                hl = np.log((high / low).replace(0, np.nan)) ** 2
            beta = hl + hl.shift(1)
            gamma = (np.log(np.maximum(high, high.shift(1)) / np.minimum(low, low.shift(1)).replace(0, np.nan))) ** 2
            alpha = (np.sqrt(2 * beta) - np.sqrt(beta)) / (3 - 2 * np.sqrt(2))
            cs_spread = 2 * (np.exp(alpha) - 1) / (1 + np.exp(alpha))
            cs_spread = pd.Series(cs_spread, index=price_data.index)
            # Roll estimator fallback (relative)
            dp = close.diff()
            roll_cov = dp.rolling(20, min_periods=2).cov(dp.shift(1))
            roll_spread = 2 * np.sqrt(np.maximum(0, -roll_cov))
            with np.errstate(divide='ignore', invalid='ignore'):
                roll_spread_rel = (roll_spread / close).replace([np.inf, -np.inf], np.nan)
            # Prefer CS, fill with Roll
            spread = cs_spread.where(np.isfinite(cs_spread), roll_spread_rel)
            spread = spread.fillna(method="ffill").fillna(method="bfill").fillna(0)
            # Clip to reasonable bounds
            spread = spread.clip(lower=0, upper=0.05)
            return spread
        except Exception as e:
            self.logger.error(f"Error calculating bid-ask spread: {e}")
            return pd.Series(np.zeros(len(price_data)), index=price_data.index)

    def _calculate_market_depth_vectorized(self, price_data: pd.DataFrame, volume_data: pd.DataFrame) -> pd.Series:
        """Calculate market depth as rolling average volume (per-row series)."""
        try:
            market_depth = volume_data["volume"].rolling(window=20, min_periods=1).mean().fillna(0)
            return market_depth
        except Exception as e:
            self.logger.error(f"Error calculating market depth: {e}")
            return pd.Series(np.zeros(len(price_data)), index=price_data.index)

    def _engineer_adaptive_indicators_vectorized(self, price_data: pd.DataFrame) -> dict[str, Any]:
        """Engineer adaptive indicators using vectorized operations."""
        try:
            features = {}

            # Adaptive moving averages (per-row series)
            features["adaptive_sma"] = self._calculate_adaptive_sma_vectorized(price_data)
            features["adaptive_ema"] = self._calculate_adaptive_ema_vectorized(price_data)

            # Adaptive volatility indicators
            features["adaptive_atr"] = self._calculate_adaptive_atr_vectorized(price_data)
            features["adaptive_bollinger"] = self._calculate_adaptive_bollinger_vectorized(price_data)

            return features

        except Exception as e:
            self.logger.error(f"Error engineering adaptive indicators: {e}")
            return {}

    def _calculate_adaptive_sma_vectorized(self, price_data: pd.DataFrame) -> pd.Series:
        """Adaptive SMA as a weighted blend of short and long SMAs per row."""
        try:
            close = price_data["close"]
            vol = close.pct_change().rolling(20, min_periods=1).std().fillna(0)
            weight = (1 / (1 + vol * 100)).clip(0, 1)
            sma_short = close.rolling(5, min_periods=1).mean()
            sma_long = close.rolling(50, min_periods=1).mean()
            adaptive = weight * sma_short + (1 - weight) * sma_long
            return adaptive.fillna(method="ffill").fillna(method="bfill").fillna(0)
        except Exception as e:
            self.logger.error(f"Error calculating adaptive SMA: {e}")
            return pd.Series(np.zeros(len(price_data)), index=price_data.index)

    def _calculate_adaptive_ema_vectorized(self, price_data: pd.DataFrame) -> pd.Series:
        """Adaptive EMA as a weighted blend of short and long EMAs per row."""
        try:
            close = price_data["close"]
            vol = close.pct_change().rolling(20, min_periods=1).std().fillna(0)
            weight = (1 / (1 + vol * 100)).clip(0, 1)
            ema_short = close.ewm(span=5, adjust=False).mean()
            ema_long = close.ewm(span=50, adjust=False).mean()
            adaptive = weight * ema_short + (1 - weight) * ema_long
            return adaptive.fillna(method="ffill").fillna(method="bfill").fillna(0)
        except Exception as e:
            self.logger.error(f"Error calculating adaptive EMA: {e}")
            return pd.Series(np.zeros(len(price_data)), index=price_data.index)

    def _calculate_adaptive_atr_vectorized(self, price_data: pd.DataFrame) -> pd.Series:
        """Adaptive ATR as a weighted blend of short and long ATR per row."""
        try:
            high_low = price_data["high"] - price_data["low"]
            high_close = (price_data["high"] - price_data["close"].shift()).abs()
            low_close = (price_data["low"] - price_data["close"].shift()).abs()
            tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            vol = tr.rolling(20, min_periods=1).std().fillna(0)
            weight = (1 / (1 + vol * 10)).clip(0, 1)
            atr_short = tr.rolling(5, min_periods=1).mean()
            atr_long = tr.rolling(30, min_periods=1).mean()
            adaptive = weight * atr_short + (1 - weight) * atr_long
            return adaptive.fillna(method="ffill").fillna(method="bfill").fillna(0)
        except Exception as e:
            self.logger.error(f"Error calculating adaptive ATR: {e}")
            return pd.Series(np.zeros(len(price_data)), index=price_data.index)

    def _calculate_adaptive_bollinger_vectorized(self, price_data: pd.DataFrame) -> pd.Series:
        """Adaptive Bollinger position as a weighted blend of short/long band positions per row."""
        try:
            close = price_data["close"]
            vol = close.pct_change().rolling(20, min_periods=1).std().fillna(0)
            weight = (1 / (1 + vol * 100)).clip(0, 1)
            sma_s = close.rolling(10, min_periods=1).mean(); std_s = close.rolling(10, min_periods=1).std()
            sma_l = close.rolling(50, min_periods=1).mean(); std_l = close.rolling(50, min_periods=1).std()
            upper_s = sma_s + 2 * std_s; lower_s = sma_s - 2 * std_s
            upper_l = sma_l + 2 * std_l; lower_l = sma_l - 2 * std_l
            with np.errstate(divide='ignore', invalid='ignore'):
                pos_s = ((close - lower_s) / (upper_s - lower_s)).replace([np.inf, -np.inf], np.nan)
                pos_l = ((close - lower_l) / (upper_l - lower_l)).replace([np.inf, -np.inf], np.nan)
            position = weight * pos_s + (1 - weight) * pos_l
            return position.fillna(method="ffill").fillna(method="bfill").fillna(0)
        except Exception as e:
            self.logger.error(f"Error calculating adaptive Bollinger Bands: {e}")
            return pd.Series(np.zeros(len(price_data)), index=price_data.index)

    def _select_optimal_features_vectorized(self, features: dict[str, Any]) -> dict[str, Any]:
        """Select optimal features using vectorized operations."""
        try:
            # Enhanced feature selection - be less restrictive to allow more engineered features
            selected_features: dict[str, Any] = {}
            for feature_name, feature_value in features.items():
                # Accept all array-like features (numpy arrays, pandas series, lists)
                if isinstance(feature_value, (np.ndarray, pd.Series, list)):
                    selected_features[feature_name] = feature_value
                # Accept numeric scalars that are finite
                elif isinstance(feature_value, (int, float)):
                    if not (np.isnan(feature_value) or np.isinf(feature_value)):
                        selected_features[feature_name] = float(feature_value)
                # Accept string features (e.g., regime labels)
                elif isinstance(feature_value, str):
                    selected_features[feature_name] = feature_value
                # Accept boolean features
                elif isinstance(feature_value, bool):
                    selected_features[feature_name] = feature_value
                # For other types, try to convert to a usable format
                else:
                    try:
                        # Try to convert to numpy array
                        arr = np.asarray(feature_value)
                        if arr.size > 0:  # Only include if it has content
                            selected_features[feature_name] = arr
                    except Exception:
                        # If conversion fails, skip this feature
                        self.logger.debug(f"Skipping feature '{feature_name}' with type {type(feature_value)}")
                        continue
            
            self.logger.info(f"✅ Selected {len(selected_features)} features from {len(features)} total features")
            return selected_features

        except Exception as e:
            self.logger.error(f"Error selecting optimal features: {e}")
            return features

    def _validate_and_transform_data(self, price_data: pd.DataFrame, volume_data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Validate and transform data to ensure it has the expected OHLCV structure.
        
        Args:
            price_data: Input price data
            volume_data: Input volume data
            
        Returns:
            Tuple of (transformed_price_data, transformed_volume_data)
        """
        try:
            # Check if price_data has OHLCV structure
            required_ohlcv_columns = ['open', 'high', 'low', 'close']
            available_price_columns = price_data.columns.tolist()
            
            # If price_data doesn't have OHLCV structure, try to create it
            if not all(col in available_price_columns for col in required_ohlcv_columns):
                # Check if we have trade data (price, quantity columns)
                if 'price' in available_price_columns and 'quantity' in available_price_columns:
                    # Create OHLCV from trade data by resampling
                    price_data = self._create_ohlcv_from_trades(price_data)
                    self.logger.info("✅ Transformed trade data to OHLCV structure")
                else:
                    # If we can't create OHLCV, create a minimal structure
                    self.logger.warning(f"Cannot create OHLCV structure. Available columns: {available_price_columns}")
                    price_data = self._create_minimal_ohlcv(price_data)
            
            # Ensure volume_data has 'volume' column
            if 'volume' not in volume_data.columns:
                if 'quantity' in volume_data.columns:
                    volume_data = volume_data.rename(columns={'quantity': 'volume'})
                else:
                    # Create a default volume column if none exists
                    volume_data['volume'] = 1.0
            
            return price_data, volume_data
            
        except Exception as e:
            self.logger.error(f"Error validating and transforming data: {e}")
            return price_data, volume_data

    def _create_ohlcv_from_trades(self, trade_data: pd.DataFrame) -> pd.DataFrame:
        """
        Create OHLCV data from trade data by resampling.
        
        Args:
            trade_data: DataFrame with 'price' and 'quantity' columns
            
        Returns:
            DataFrame with OHLCV structure
        """
        try:
            # Set timestamp as index if it's not already
            if 'timestamp' in trade_data.columns:
                trade_data = trade_data.set_index('timestamp')
            
            # Resample to 1-minute intervals and create OHLCV
            resampled = trade_data.resample('1min').agg({
                'price': ['first', 'max', 'min', 'last'],
                'quantity': 'sum'
            })
            
            # Flatten column names
            resampled.columns = ['open', 'high', 'low', 'close', 'volume']
            
            # Fill any missing values
            resampled = resampled.fillna(method='ffill')
            
            return resampled
            
        except Exception as e:
            self.logger.error(f"Error creating OHLCV from trades: {e}")
            return trade_data

    def _create_minimal_ohlcv(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Create minimal OHLCV structure when proper data is not available.
        
        Args:
            data: Input data
            
        Returns:
            DataFrame with minimal OHLCV structure
        """
        try:
            # If we have a 'price' column, use it for all OHLCV values
            if 'price' in data.columns:
                data['open'] = data['price']
                data['high'] = data['price']
                data['low'] = data['price']
                data['close'] = data['price']
                data['volume'] = data.get('quantity', 1.0)
            else:
                # Create dummy OHLCV structure
                data['open'] = 1.0
                data['high'] = 1.0
                data['low'] = 1.0
                data['close'] = 1.0
                data['volume'] = 1.0
            
            return data
            
        except Exception as e:
            self.logger.error(f"Error creating minimal OHLCV: {e}")
            return data

    async def _engineer_multi_timeframe_features_vectorized(
        self,
        price_data: pd.DataFrame,
        volume_data: pd.DataFrame,
        order_flow_data: pd.DataFrame | None = None,
        sr_levels: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Engineer multi-timeframe features using vectorized operations."""
        try:
            features = {}
            base_index = price_data.index
            if not isinstance(base_index, pd.DatetimeIndex):
                if "timestamp" in price_data.columns:
                    try:
                        base_index = pd.to_datetime(price_data["timestamp"], errors="coerce")
                    except Exception:
                        base_index = pd.date_range(start="1970-01-01", periods=len(price_data), freq="1min")
                else:
                    base_index = pd.date_range(start="1970-01-01", periods=len(price_data), freq="1min")

            # Multi-timeframe features for different timeframes
            for timeframe in self.timeframes:
                # Resample data to timeframe
                resampled_price = self._resample_data_vectorized(price_data, timeframe)
                resampled_volume = self._resample_data_vectorized(volume_data, timeframe)

                # Calculate features for this timeframe
                timeframe_features = await self._calculate_timeframe_features_vectorized(
                    resampled_price,
                    resampled_volume,
                    timeframe,
                )

                # Add timeframe prefix to features and align back to base index
                for feature_name, feature_value in timeframe_features.items():
                    # Ensure we have a Series to reindex
                    if isinstance(feature_value, np.ndarray):
                        series = pd.Series(feature_value, index=resampled_price.index)
                    elif isinstance(feature_value, pd.Series):
                        series = feature_value
                    else:
                        try:
                            series = pd.Series(feature_value, index=resampled_price.index)
                        except Exception:
                            continue
                    aligned = series.reindex(base_index, method="ffill").fillna(method="bfill").fillna(0)
                    features[f"{timeframe}_{feature_name}"] = aligned

            return features

        except Exception as e:
            self.logger.error(f"Error engineering multi-timeframe features: {e}")
            return {}

    def _resample_data_vectorized(self, data: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        """Resample data to specified timeframe using vectorized operations."""
        try:
            # Convert timeframe string to pandas offset
            timeframe_map = {
                "1m": "1min",
                "5m": "5min",
                "15m": "15min",
                "30m": "30min",
            }
            
            offset = timeframe_map.get(timeframe, "1T")
            
            # Check if data has datetime index, if not create one
            if not isinstance(data.index, pd.DatetimeIndex):
                data = data.copy()
                # Create a proper datetime index
                data.index = pd.date_range(
                    start='2023-01-01',
                    periods=len(data),
                    freq='1min'
                )
            
            # Check if OHLCV columns exist, if not, create them from available data
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            available_columns = data.columns.tolist()
            
            # If we don't have OHLCV columns, try to create them from available data
            if not all(col in available_columns for col in required_columns):
                # Check if we have price and quantity columns (like in trade data)
                if 'price' in available_columns and 'quantity' in available_columns:
                    # Create OHLCV from trade data
                    try:
                        resampled = data.resample(offset).agg({
                            'price': ['first', 'max', 'min', 'last'],
                            'quantity': 'sum'
                        })
                        # Flatten column names
                        resampled.columns = ['open', 'high', 'low', 'close', 'volume']
                        return resampled.dropna()
                    except Exception as resample_error:
                        self.logger.error(f"Error resampling trade data: {resample_error}")
                        return data
                # Check if we have volume-only data (like volume features)
                elif any(col.startswith('volume') for col in available_columns):
                    # Handle volume-only data by resampling each volume column appropriately
                    try:
                        # Create aggregation dictionary for volume columns
                        agg_dict = {}
                        for col in available_columns:
                            if col == 'volume':
                                agg_dict[col] = 'sum'
                            elif col.startswith('volume_'):
                                # For derived volume features, use mean aggregation
                                agg_dict[col] = 'mean'
                            else:
                                # For other columns, use mean
                                agg_dict[col] = 'mean'
                        
                        resampled = data.resample(offset).agg(agg_dict)
                        return resampled.dropna()
                    except Exception as resample_error:
                        self.logger.error(f"Error resampling volume data: {resample_error}")
                        return data
                else:
                    # If we can't create OHLCV, return original data without resampling
                    self.logger.warning(f"Cannot resample data: missing required columns. Available: {available_columns}")
                    # Return data with proper index but no resampling
                    return data
            else:
                # Standard OHLCV resampling
                try:
                    resampled = data.resample(offset).agg({
                        'open': 'first',
                        'high': 'max',
                        'low': 'min',
                        'close': 'last',
                        'volume': 'sum',
                    })
                    return resampled.dropna()
                except Exception as resample_error:
                    self.logger.error(f"Error during resampling: {resample_error}")
                    return data

        except Exception as e:
            self.logger.error(f"Error resampling data: {e}")
            return data

    async def _calculate_timeframe_features_vectorized(
        self,
        price_data: pd.DataFrame,
        volume_data: pd.DataFrame,
        timeframe: str,
    ) -> dict[str, Any]:
        """Calculate features for specific timeframe using vectorized operations."""
        try:
            features = {}

            # Basic price features as full series aligned to index
            price_change_series = price_data["close"].pct_change()
            price_vol_series = price_data["close"].pct_change().rolling(window=20).std()
            features["price_change"] = price_change_series
            features["price_volatility"] = price_vol_series

            # Volume features as full series
            volume_change_series = volume_data["volume"].pct_change()
            volume_ma_ratio_series = volume_data["volume"] / volume_data["volume"].rolling(window=20, min_periods=1).mean()
            features["volume_change"] = volume_change_series
            features["volume_ma_ratio"] = volume_ma_ratio_series
            # Additional multi-horizon volume MA ratios
            try:
                vol_ma_5 = volume_data["volume"].rolling(window=5, min_periods=1).mean()
                vol_ma_15 = volume_data["volume"].rolling(window=15, min_periods=1).mean()
                features["5m_volume_ma_ratio"] = (volume_data["volume"] / (vol_ma_5.replace(0, np.nan))).replace([np.inf, -np.inf], np.nan).fillna(1.0)
                features["15m_volume_ma_ratio"] = (volume_data["volume"] / (vol_ma_15.replace(0, np.nan))).replace([np.inf, -np.inf], np.nan).fillna(1.0)
            except Exception:
                pass

            return features

        except Exception as e:
            self.logger.error(f"Error calculating timeframe features: {e}")
            return {}

    async def _generate_meta_labels_vectorized(
        self,
        price_data: pd.DataFrame,
        volume_data: pd.DataFrame,
        order_flow_data: pd.DataFrame | None = None,
    ) -> dict[str, Any]:
        """Generate meta labels using vectorized operations."""
        try:
            n = len(price_data)
            if n == 0:
                return {}
 
            # Volatility regime per row
            vol = price_data["close"].pct_change().rolling(window=20).std()
            # Create 5-bin categorical regimes using quantiles. Handle duplicates.
            try:
                vol_bins = pd.qcut(vol.fillna(0), q=5, labels=False, duplicates="drop")
            except Exception:
                vol_bins = pd.Series(np.zeros(n, dtype=int), index=vol.index)
            vol_regime_series = vol_bins.fillna(0).astype(int)
 
            # Volume regime per row
            vol_ma = volume_data["volume"].rolling(window=20).mean()
            volume_ratio = (volume_data["volume"] / (vol_ma.replace(0, np.nan))).fillna(0)
            # Create 5-bin categorical regimes using quantiles. Handle duplicates.
            try:
                volreg_bins = pd.qcut(volume_ratio, q=5, labels=False, duplicates="drop")
            except Exception:
                volreg_bins = pd.Series(np.zeros(n, dtype=int), index=volume_ratio.index)
            volume_regime_series = volreg_bins.fillna(0).astype(int)
 
            # Trend regime per row
            sma_short = price_data["close"].rolling(window=10).mean()
            sma_long = price_data["close"].rolling(window=30).mean()
            trend_strength = (sma_short - sma_long)
            # Create 5-bin categorical regimes using quantiles. Handle duplicates.
            try:
                trend_bins = pd.qcut(trend_strength.fillna(0), q=5, labels=False, duplicates="drop")
            except Exception:
                trend_bins = pd.Series(np.zeros(n, dtype=int), index=trend_strength.index)
            trend_regime_series = trend_bins.fillna(0).astype(int)
 
            # Diagnostics: log variability and distributions
            try:
                for name, series in {
                    "volatility_regime": vol_regime_series,
                    "volume_regime": volume_regime_series,
                    "trend_regime": trend_regime_series,
                }.items():
                    unique_vals = pd.Series(series).nunique(dropna=True)
                    if unique_vals < 5:
                        self.logger.warning(
                            f"Meta label '{name}' has low variability: {unique_vals} unique bins (<5). "
                            f"value_counts={pd.Series(series).value_counts().to_dict()}"
                        )
            except Exception:
                pass
 
            return {
                "volatility_regime": vol_regime_series.values,
                "volume_regime": volume_regime_series.values,
                "trend_regime": trend_regime_series.values,
            }
 
        except Exception as e:
            self.logger.error(f"Error generating meta labels: {e}")
            return {}


class VectorizedVolatilityRegimeModel:
    """Vectorized volatility regime model using GARCH and other methods with price differences."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.logger = system_logger.getChild("VectorizedVolatilityRegimeModel")

    async def initialize(self) -> bool:
        """Initialize volatility model."""
        try:
            self.logger.info("🚀 Initializing vectorized volatility regime model...")
            self.logger.info("✅ Vectorized volatility regime model initialized successfully")
            return True
        except Exception as e:
            self.logger.error(f"❌ Error initializing volatility model: {e}")
            return False

    async def model_volatility_vectorized(self, price_data: pd.DataFrame) -> dict[str, Any]:
        """Model volatility regimes using price differences."""
        try:
            # Use price differences instead of returns
            price_diff = price_data["close"].diff()

            # Calculate various volatility measures using price differences (series)
            realized_vol = price_diff.rolling(20, min_periods=1).std()

            # Parkinson and Garman-Klass volatility as series
            parkinson_vol = self._calculate_parkinson_volatility_vectorized(price_data)
            garman_klass_vol = self._calculate_garman_klass_volatility_vectorized(price_data)

            # Volatility regime classification as numeric series
            vol_percentile_series = realized_vol.rank(pct=True).fillna(0.5)
            # Map to 0,1,2
            vol_regime_numeric = pd.cut(
                vol_percentile_series,
                bins=[-np.inf, 0.2, 0.8, np.inf],
                labels=[0, 1, 2],
            ).astype("int8")

            return {
                "realized_volatility": realized_vol.fillna(0).values,
                "parkinson_volatility": parkinson_vol.fillna(0).values if isinstance(parkinson_vol, pd.Series) else parkinson_vol,
                "garman_klass_volatility": garman_klass_vol.fillna(0).values if isinstance(garman_klass_vol, pd.Series) else garman_klass_vol,
                "volatility_regime": vol_regime_numeric.values,
                "volatility_percentile": vol_percentile_series.values,
            }

        except Exception as e:
            self.logger.error(f"Error modeling volatility: {e}")
            return {}

    def _calculate_parkinson_volatility_vectorized(self, price_data: pd.DataFrame) -> pd.Series:
        """Calculate Parkinson volatility estimator using price differences."""
        try:
            # Use high-low differences
            high_low_diff = price_data["high"] - price_data["low"]
            high_low_ratio = np.log(high_low_diff / price_data["close"].shift(1) + 1) ** 2
            parkinson_vol = high_low_ratio.rolling(20, min_periods=1).mean()
            return parkinson_vol
        except Exception as e:
            self.logger.error(f"Error calculating Parkinson volatility: {e}")
            return pd.Series(np.zeros(len(price_data)), index=price_data.index)

    def _calculate_garman_klass_volatility_vectorized(self, price_data: pd.DataFrame) -> pd.Series:
        try:
            log_hl = np.log(price_data["high"] / price_data["low"]).pow(2)
            log_co = np.log(price_data["close"] / price_data["open"]).pow(2)
            gk = 0.5 * log_hl - (2 * np.log(2) - 1) * log_co
            return gk.rolling(20, min_periods=1).mean()
        except Exception as e:
            self.logger.error(f"Error calculating Garman-Klass volatility: {e}")
            return pd.Series(np.zeros(len(price_data)), index=price_data.index)


class VectorizedCorrelationAnalyzer:
    """Vectorized correlation analyzer using price differences."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.logger = system_logger.getChild("VectorizedCorrelationAnalyzer")

    async def initialize(self) -> bool:
        """Initialize correlation analyzer."""
        try:
            self.logger.info("🚀 Initializing vectorized correlation analyzer...")
            self.logger.info("✅ Vectorized correlation analyzer initialized successfully")
            return True
        except Exception as e:
            self.logger.error(f"❌ Error initializing correlation analyzer: {e}")
            return False

    async def analyze_correlations_vectorized(self, price_data: pd.DataFrame) -> dict[str, Any]:
        """Analyze correlations using price differences."""
        try:
            # Use price differences for correlation analysis
            price_diff = price_data["close"].diff()
            
            # Rolling correlations as series
            rolling_corr_5_series = price_diff.rolling(5).corr(price_diff.shift(1))
            rolling_corr_10_series = price_diff.rolling(10).corr(price_diff.shift(1))
            
            # Autocorrelations approximated as rolling mean of lagged product correlation
            autocorr_1_series = price_diff.rolling(50).apply(lambda s: s.autocorr(lag=1) if len(s.dropna()) > 1 else 0.0, raw=False)
            autocorr_5_series = price_diff.rolling(50).apply(lambda s: s.autocorr(lag=5) if len(s.dropna()) > 5 else 0.0, raw=False)
            autocorr_10_series = price_diff.rolling(50).apply(lambda s: s.autocorr(lag=10) if len(s.dropna()) > 10 else 0.0, raw=False)
            
            return {
                "autocorr_1": autocorr_1_series.values,
                "autocorr_5": autocorr_5_series.values,
                "autocorr_10": autocorr_10_series.values,
                "rolling_corr_5": rolling_corr_5_series.values,
                "rolling_corr_10": rolling_corr_10_series.values,
                "correlation_strength": np.abs(autocorr_1_series.fillna(0)).values,
            }

        except Exception as e:
            self.logger.error(f"Error analyzing correlations: {e}")
            return {"correlation_strength": np.zeros(len(price_data))}


class VectorizedMomentumAnalyzer:
    """Vectorized momentum analyzer using price differences."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.logger = system_logger.getChild("VectorizedMomentumAnalyzer")

    async def initialize(self) -> bool:
        """Initialize momentum analyzer."""
        try:
            self.logger.info("🚀 Initializing vectorized momentum analyzer...")
            self.logger.info("✅ Vectorized momentum analyzer initialized successfully")
            return True
        except Exception as e:
            self.logger.error(f"❌ Error initializing momentum analyzer: {e}")
            return False

    async def analyze_momentum_vectorized(self, price_data: pd.DataFrame) -> dict[str, Any]:
        """Analyze momentum using price differences."""
        try:
            # Use price differences for momentum analysis
            price_diff = price_data["close"].diff()
            
            # Calculate momentum indicators using price differences (as series)
            momentum_5_series = price_diff.rolling(5).sum()
            momentum_10_series = price_diff.rolling(10).sum()
            momentum_20_series = price_diff.rolling(20).sum()
            
            # Rate of change as series
            roc_5_series = price_data["close"].pct_change(5)
            roc_10_series = price_data["close"].pct_change(10)
            
            # Momentum strength as series
            with np.errstate(divide='ignore', invalid='ignore'):
                momentum_strength_series = np.abs(momentum_10_series) / price_data["close"]
            momentum_strength_series = momentum_strength_series.replace([np.inf, -np.inf], np.nan)
            
            return {
                "momentum_5": momentum_5_series.values,
                "momentum_10": momentum_10_series.values,
                "momentum_20": momentum_20_series.values,
                "roc_5": roc_5_series.values,
                "roc_10": roc_10_series.values,
                "momentum_strength": momentum_strength_series.fillna(0).values,
            }

        except Exception as e:
            self.logger.error(f"Error analyzing momentum: {e}")
            return {"momentum_strength": np.zeros(len(price_data))}


class VectorizedLiquidityAnalyzer:
    """Vectorized liquidity analyzer using price differences."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.logger = system_logger.getChild("VectorizedLiquidityAnalyzer")

    async def initialize(self) -> bool:
        """Initialize liquidity analyzer."""
        try:
            self.logger.info("🚀 Initializing vectorized liquidity analyzer...")
            self.logger.info("✅ Vectorized liquidity analyzer initialized successfully")
            return True
        except Exception as e:
            self.logger.error(f"❌ Error initializing liquidity analyzer: {e}")
            return False

    async def analyze_liquidity_vectorized(
        self,
        price_data: pd.DataFrame,
        volume_data: pd.DataFrame,
        order_flow_data: pd.DataFrame | None = None,
    ) -> dict[str, Any]:
        """Analyze liquidity using price differences."""
        try:
            # Use price differences for liquidity analysis
            price_diff = price_data["close"].diff()
            
            # Liquidity metrics as series
            avg_volume_series = volume_data["volume"].rolling(20).mean()
            volume_volatility_series = volume_data["volume"].rolling(20).std()
            with np.errstate(divide='ignore', invalid='ignore'):
                price_impact_series = np.abs(price_diff) / avg_volume_series
            price_impact_series = price_impact_series.replace([np.inf, -np.inf], np.nan)
            liquidity_score_series = 1.0 / (1.0 + price_impact_series)
            
            return {
                "avg_volume": avg_volume_series.fillna(method="ffill").fillna(0).values,
                "volume_volatility": volume_volatility_series.fillna(0).values,
                "price_impact": price_impact_series.fillna(0).values,
                "liquidity_score": liquidity_score_series.fillna(1.0).values,
            }

        except Exception as e:
            self.logger.error(f"Error analyzing liquidity: {e}")
            return {"liquidity_score": np.ones(len(price_data))}


class VectorizedSRDistanceCalculator:
    """Vectorized support/resistance distance calculator using price differences."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.logger = system_logger.getChild("VectorizedSRDistanceCalculator")

    async def initialize(self) -> bool:
        """Initialize S/R distance calculator."""
        try:
            self.logger.info("🚀 Initializing vectorized S/R distance calculator...")
            self.logger.info("✅ Vectorized S/R distance calculator initialized successfully")
            return True
        except Exception as e:
            self.logger.error(f"❌ Error initializing S/R distance calculator: {e}")
            return False

    async def calculate_sr_distances(
        self,
        price_data: pd.DataFrame,
        sr_levels: dict[str, Any],
    ) -> dict[str, Any]:
        """Calculate per-row distances to nearest support/resistance levels."""
        try:
            close = price_data["close"].astype(float)
            support_levels = sr_levels.get("support_levels", []) or []
            resistance_levels = sr_levels.get("resistance_levels", []) or []
            if len(support_levels) == 0:
                nsd = pd.Series(np.full(len(close), np.nan), index=close.index)
            else:
                dists = np.vstack([np.abs(close.values - lvl) / np.where(close.values!=0, close.values, np.nan) for lvl in support_levels])
                nsd = pd.Series(np.nanmin(dists, axis=0), index=close.index)
            if len(resistance_levels) == 0:
                nrd = pd.Series(np.full(len(close), np.nan), index=close.index)
            else:
                dists = np.vstack([np.abs(close.values - lvl) / np.where(close.values!=0, close.values, np.nan) for lvl in resistance_levels])
                nrd = pd.Series(np.nanmin(dists, axis=0), index=close.index)
            return {
                "nearest_support_distance": nsd.fillna(method="ffill").fillna(method="bfill").fillna(0).values,
                "nearest_resistance_distance": nrd.fillna(method="ffill").fillna(method="bfill").fillna(0).values,
                "support_levels_count": pd.Series(np.full(len(close), len(support_levels)), index=close.index).values,
                "resistance_levels_count": pd.Series(np.full(len(close), len(resistance_levels)), index=close.index).values,
            }
        except Exception as e:
            self.logger.error(f"Error calculating S/R distances: {e}")
            return {
                "nearest_support_distance": pd.Series(np.zeros(len(price_data)), index=price_data.index).values,
                "nearest_resistance_distance": pd.Series(np.zeros(len(price_data)), index=price_data.index).values,
                "support_levels_count": pd.Series(np.zeros(len(price_data)), index=price_data.index).values,
                "resistance_levels_count": pd.Series(np.zeros(len(price_data)), index=price_data.index).values,
            }


class VectorizedCandlestickPatternAnalyzer:
    """
    Comprehensive candlestick pattern analyzer implementing all major patterns
    for enhanced feature engineering and ML model training with vectorized operations.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.logger = system_logger.getChild("VectorizedCandlestickPatternAnalyzer")

        # Pattern detection parameters
        self.pattern_config = config.get("candlestick_patterns", {})
        self.doji_threshold = self.pattern_config.get("doji_threshold", 0.1)
        self.hammer_ratio = self.pattern_config.get("hammer_ratio", 0.3)
        self.shadow_ratio = self.pattern_config.get("shadow_ratio", 2.0)
        self.engulfing_ratio = self.pattern_config.get("engulfing_ratio", 1.1)
        self.tweezer_threshold = self.pattern_config.get("tweezer_threshold", 0.02)
        self.marubozu_threshold = self.pattern_config.get("marubozu_threshold", 0.1)

        self.is_initialized = False

    @handle_errors(
        exceptions=(Exception,),
        default_return=False,
        context="candlestick pattern analyzer initialization",
    )
    async def initialize(self) -> bool:
        """Initialize candlestick pattern analyzer."""
        try:
            self.logger.info("🚀 Initializing vectorized candlestick pattern analyzer...")
            self.is_initialized = True
            self.logger.info("✅ Vectorized candlestick pattern analyzer initialized successfully")
            return True
        except Exception as e:
            self.logger.error(
                f"❌ Error initializing vectorized candlestick pattern analyzer: {e}",
            )
            return False

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return={},
        context="candlestick pattern analysis",
    )
    async def analyze_patterns(self, price_data: pd.DataFrame) -> dict[str, Any]:
        """
        Analyze candlestick patterns and return features for ML training using vectorized operations.

        Args:
            price_data: OHLCV price data

        Returns:
            Dictionary containing candlestick pattern features
        """
        try:
            if not self.is_initialized:
                self.logger.error("Candlestick pattern analyzer not initialized")
                return {}

            if price_data.empty or len(price_data) < 3:
                self.logger.warning("Insufficient data for pattern analysis")
                return {}

            # Prepare data with calculated metrics using vectorized operations
            df = self._prepare_candlestick_data_vectorized(price_data)

            # Analyze all patterns using vectorized operations
            patterns = {
                "engulfing_patterns": self._detect_engulfing_patterns_vectorized(df),
                "hammer_hanging_man": self._detect_hammer_hanging_man_vectorized(df),
                "shooting_star_inverted_hammer": self._detect_shooting_star_inverted_hammer_vectorized(df),
                "tweezer_patterns": self._detect_tweezer_patterns_vectorized(df),
                "marubozu_patterns": self._detect_marubozu_patterns_vectorized(df),
                "three_methods_patterns": self._detect_three_methods_patterns_vectorized(df),
                "doji_patterns": self._detect_doji_patterns_vectorized(df),
                "spinning_top_patterns": self._detect_spinning_top_patterns_vectorized(df),
            }

            # Convert patterns to ML features using vectorized operations
            features = self._convert_patterns_to_features_vectorized(patterns, df)

            self.logger.info(f"✅ Analyzed {len(patterns)} pattern categories using vectorized operations")
            return features

        except Exception as e:
            self.logger.error(f"Error analyzing candlestick patterns: {e}")
            return {}

    def _prepare_candlestick_data_vectorized(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare data with candlestick metrics using vectorized operations."""
        try:
            df = df.copy()

            # Calculate basic candlestick metrics using vectorized operations
            df["body_size"] = np.abs(df["close"] - df["open"])
            df["upper_shadow"] = df["high"] - np.maximum(df["open"], df["close"])
            df["lower_shadow"] = np.minimum(df["open"], df["close"]) - df["low"]
            df["total_range"] = df["high"] - df["low"]
            df["body_ratio"] = df["body_size"] / df["total_range"].replace(0, 1)
            df["is_bullish"] = df["close"] > df["open"]

            # Calculate moving averages for context using vectorized operations
            df["avg_body_size"] = df["body_size"].rolling(window=20).mean()
            df["avg_range"] = df["total_range"].rolling(window=20).mean()

            return df.dropna()

        except Exception as e:
            self.logger.error(f"Error preparing candlestick data: {e}")
            return pd.DataFrame()

    def _detect_engulfing_patterns_vectorized(self, df: pd.DataFrame) -> dict[str, np.ndarray]:
        """Detect bullish and bearish engulfing patterns using vectorized operations."""
        try:
            # Vectorized calculations
            current_body_size = df["body_size"].values
            previous_body_size = df["body_size"].shift(1).values
            current_is_bullish = df["is_bullish"].values
            previous_is_bullish = df["is_bullish"].shift(1).values
            current_open = df["open"].values
            current_close = df["close"].values
            previous_open = df["open"].shift(1).values
            previous_close = df["close"].shift(1).values

            # Bullish engulfing conditions
            bullish_engulfing = (
                current_is_bullish.astype(bool) &
                (~previous_is_bullish.astype(bool)) &
                (current_open < previous_close) &
                (current_close > previous_open) &
                (current_body_size > previous_body_size * self.engulfing_ratio)
            )

            # Bearish engulfing conditions
            bearish_engulfing = (
                (~current_is_bullish.astype(bool)) &
                previous_is_bullish.astype(bool) &
                (current_open > previous_close) &
                (current_close < previous_open) &
                (current_body_size > previous_body_size * self.engulfing_ratio)
            )

            # Calculate confidence scores
            bullish_confidence = np.where(
                bullish_engulfing,
                np.minimum(current_body_size / (previous_body_size + 1e-8), 2.0),
                0.0
            )
            bearish_confidence = np.where(
                bearish_engulfing,
                np.minimum(current_body_size / (previous_body_size + 1e-8), 2.0),
                0.0
            )

            return {
                "bullish_engulfing": bullish_engulfing,
                "bearish_engulfing": bearish_engulfing,
                "bullish_confidence": bullish_confidence,
                "bearish_confidence": bearish_confidence,
            }

        except Exception as e:
            self.logger.error(f"Error detecting engulfing patterns: {e}")
            return {}

    def _detect_hammer_hanging_man_vectorized(self, df: pd.DataFrame) -> dict[str, np.ndarray]:
        """Detect hammer and hanging man patterns using vectorized operations."""
        try:
            # Vectorized calculations
            body_ratio = df["body_ratio"].values
            lower_shadow = df["lower_shadow"].values
            upper_shadow = df["upper_shadow"].values
            body_size = df["body_size"].values
            is_bullish = df["is_bullish"].values
            close_prices = df["close"].values

            # Hammer pattern conditions
            hammer_conditions = (
                (body_ratio <= self.hammer_ratio) &
                (lower_shadow > body_size * self.shadow_ratio) &
                (upper_shadow < body_size * 0.5)
            )

            # Hanging man pattern conditions (need previous close for context)
            previous_close = df["close"].shift(1).values
            hanging_man_conditions = (
                (body_ratio <= self.hammer_ratio) &
                (lower_shadow > body_size * self.shadow_ratio) &
                (upper_shadow < body_size * 0.5) &
                (previous_close > close_prices)
            )

            # Calculate confidence scores
            hammer_confidence = np.where(
                hammer_conditions,
                np.minimum(lower_shadow / (body_size + 1e-8), 3.0),
                0.0
            )
            hanging_man_confidence = np.where(
                hanging_man_conditions,
                np.minimum(lower_shadow / (body_size + 1e-8), 3.0),
                0.0
            )

            return {
                "hammer": hammer_conditions,
                "hanging_man": hanging_man_conditions,
                "hammer_confidence": hammer_confidence,
                "hanging_man_confidence": hanging_man_confidence,
            }

        except Exception as e:
            self.logger.error(f"Error detecting hammer/hanging man patterns: {e}")
            return {}

    def _detect_shooting_star_inverted_hammer_vectorized(self, df: pd.DataFrame) -> dict[str, np.ndarray]:
        """Detect shooting star and inverted hammer patterns using vectorized operations."""
        try:
            # Vectorized calculations
            body_ratio = df["body_ratio"].values
            upper_shadow = df["upper_shadow"].values
            lower_shadow = df["lower_shadow"].values
            body_size = df["body_size"].values
            close_prices = df["close"].values
            previous_close = df["close"].shift(1).values

            # Shooting star pattern conditions
            shooting_star_conditions = (
                (body_ratio <= self.hammer_ratio) &
                (upper_shadow > body_size * self.shadow_ratio) &
                (lower_shadow < body_size * 0.5)
            )

            # Inverted hammer pattern conditions
            inverted_hammer_conditions = (
                (body_ratio <= self.hammer_ratio) &
                (upper_shadow > body_size * self.shadow_ratio) &
                (lower_shadow < body_size * 0.5) &
                (previous_close < close_prices)
            )

            # Calculate confidence scores
            shooting_star_confidence = np.where(
                shooting_star_conditions,
                np.minimum(upper_shadow / (body_size + 1e-8), 3.0),
                0.0
            )
            inverted_hammer_confidence = np.where(
                inverted_hammer_conditions,
                np.minimum(upper_shadow / (body_size + 1e-8), 3.0),
                0.0
            )

            return {
                "shooting_star": shooting_star_conditions,
                "inverted_hammer": inverted_hammer_conditions,
                "shooting_star_confidence": shooting_star_confidence,
                "inverted_hammer_confidence": inverted_hammer_confidence,
            }

        except Exception as e:
            self.logger.error(f"Error detecting shooting star/inverted hammer patterns: {e}")
            return {}

    def _detect_tweezer_patterns_vectorized(self, df: pd.DataFrame) -> dict[str, np.ndarray]:
        """Detect tweezer tops and bottoms patterns using vectorized operations."""
        try:
            # Vectorized calculations
            current_high = df["high"].values
            current_low = df["low"].values
            current_close = df["close"].values
            current_open = df["open"].values
            previous_high = df["high"].shift(1).values
            previous_low = df["low"].shift(1).values
            previous_close = df["close"].shift(1).values
            previous_open = df["open"].shift(1).values

            # Tweezer tops conditions
            tweezer_top_conditions = (
                (np.abs(current_high - previous_high) <= self.tweezer_threshold * current_high) &
                (current_high > current_close) &
                (previous_high > previous_close)
            )

            # Tweezer bottoms conditions
            tweezer_bottom_conditions = (
                (np.abs(current_low - previous_low) <= self.tweezer_threshold * current_low) &
                (current_low < current_open) &
                (previous_low < previous_open)
            )

            # Calculate confidence scores
            tweezer_top_confidence = np.where(
                tweezer_top_conditions,
                1.0 - np.abs(current_high - previous_high) / (current_high + 1e-8),
                0.0
            )
            tweezer_bottom_confidence = np.where(
                tweezer_bottom_conditions,
                1.0 - np.abs(current_low - previous_low) / (current_low + 1e-8),
                0.0
            )

            return {
                "tweezer_top": tweezer_top_conditions,
                "tweezer_bottom": tweezer_bottom_conditions,
                "tweezer_top_confidence": tweezer_top_confidence,
                "tweezer_bottom_confidence": tweezer_bottom_confidence,
            }

        except Exception as e:
            self.logger.error(f"Error detecting tweezer patterns: {e}")
            return {}

    def _detect_marubozu_patterns_vectorized(self, df: pd.DataFrame) -> dict[str, np.ndarray]:
        """Detect bullish and bearish marubozu patterns using vectorized operations."""
        try:
            # Vectorized calculations
            upper_shadow = df["upper_shadow"].values
            lower_shadow = df["lower_shadow"].values
            total_range = df["total_range"].values
            is_bullish = df["is_bullish"].values

            # Marubozu conditions (no shadows or very small shadows)
            marubozu_conditions = (
                (upper_shadow < total_range * self.marubozu_threshold) &
                (lower_shadow < total_range * self.marubozu_threshold)
            )

            # Separate bullish and bearish marubozu
            bullish_marubozu = marubozu_conditions & is_bullish
            bearish_marubozu = marubozu_conditions & ~is_bullish

            # Calculate confidence scores
            marubozu_confidence = np.where(
                marubozu_conditions,
                1.0 - (upper_shadow + lower_shadow) / (total_range + 1e-8),
                0.0
            )

            return {
                "bullish_marubozu": bullish_marubozu,
                "bearish_marubozu": bearish_marubozu,
                "marubozu_confidence": marubozu_confidence,
            }

        except Exception as e:
            self.logger.error(f"Error detecting marubozu patterns: {e}")
            return {}

    def _detect_three_methods_patterns_vectorized(self, df: pd.DataFrame) -> dict[str, np.ndarray]:
        """Detect rising and falling three methods patterns using vectorized operations."""
        try:
            # This is a complex pattern that requires looking at 5 candles
            # For vectorized implementation, we'll use rolling windows
            window_size = 5
            patterns = {
                "rising_three_methods": np.zeros(len(df), dtype=bool),
                "falling_three_methods": np.zeros(len(df), dtype=bool),
                "rising_three_methods_confidence": np.zeros(len(df), dtype=float),
                "falling_three_methods_confidence": np.zeros(len(df), dtype=float),
            }

            # Process each window
            for i in range(window_size - 1, len(df)):
                window = df.iloc[i - window_size + 1:i + 1]
                
                if self._is_rising_three_methods_vectorized(window):
                    patterns["rising_three_methods"][i] = True
                    patterns["rising_three_methods_confidence"][i] = 0.8
                
                if self._is_falling_three_methods_vectorized(window):
                    patterns["falling_three_methods"][i] = True
                    patterns["falling_three_methods_confidence"][i] = 0.8

            return patterns

        except Exception as e:
            self.logger.error(f"Error detecting three methods patterns: {e}")
            return {}

    def _is_rising_three_methods_vectorized(self, window: pd.DataFrame) -> bool:
        """Check if the 5-candle pattern is a rising three methods using vectorized operations."""
        try:
            if len(window) != 5:
                return False

            candles = window.values
            body_sizes = np.abs(candles[:, 3] - candles[:, 0])  # close - open
            highs = candles[:, 1]
            lows = candles[:, 2]
            closes = candles[:, 3]
            opens = candles[:, 0]
            is_bullish = closes > opens

            # First candle should be a long bullish candle
            if not (is_bullish[0] and body_sizes[0] > np.mean(body_sizes)):
                return False

            # Next three candles should be small bearish candles within the range of the first
            for i in range(1, 4):
                if (is_bullish[i] or highs[i] > highs[0] or lows[i] < lows[0]):
                    return False

            # Last candle should be a long bullish candle closing above the first
            if not (is_bullish[4] and closes[4] > closes[0] and body_sizes[4] > np.mean(body_sizes)):
                return False

            return True

        except Exception:
            return False

    def _is_falling_three_methods_vectorized(self, window: pd.DataFrame) -> bool:
        """Check if the 5-candle pattern is a falling three methods using vectorized operations."""
        try:
            if len(window) != 5:
                return False

            candles = window.values
            body_sizes = np.abs(candles[:, 3] - candles[:, 0])  # close - open
            highs = candles[:, 1]
            lows = candles[:, 2]
            closes = candles[:, 3]
            opens = candles[:, 0]
            is_bullish = closes > opens

            # First candle should be a long bearish candle
            if not (~is_bullish[0] and body_sizes[0] > np.mean(body_sizes)):
                return False

            # Next three candles should be small bullish candles within the range of the first
            for i in range(1, 4):
                if (~is_bullish[i] or highs[i] > highs[0] or lows[i] < lows[0]):
                    return False

            # Last candle should be a long bearish candle closing below the first
            if not (~is_bullish[4] and closes[4] < closes[0] and body_sizes[4] > np.mean(body_sizes)):
                return False

            return True

        except Exception:
            return False

    def _detect_doji_patterns_vectorized(self, df: pd.DataFrame) -> dict[str, np.ndarray]:
        """Detect doji patterns using vectorized operations."""
        try:
            # Vectorized calculations
            body_ratio = df["body_ratio"].values

            # Doji pattern conditions (very small body)
            doji_conditions = body_ratio <= self.doji_threshold

            # Calculate confidence scores
            doji_confidence = np.where(doji_conditions, 1.0 - body_ratio, 0.0)

            return {
                "doji": doji_conditions,
                "doji_confidence": doji_confidence,
            }

        except Exception as e:
            self.logger.error(f"Error detecting doji patterns: {e}")
            return {}

    def _detect_spinning_top_patterns_vectorized(self, df: pd.DataFrame) -> dict[str, np.ndarray]:
        """Detect spinning top patterns using vectorized operations."""
        try:
            # Vectorized calculations
            body_ratio = df["body_ratio"].values
            upper_shadow = df["upper_shadow"].values
            lower_shadow = df["lower_shadow"].values
            total_range = df["total_range"].values

            # Spinning top conditions (small body, equal shadows)
            spinning_top_conditions = (
                (body_ratio <= 0.3) &
                (np.abs(upper_shadow - lower_shadow) < 0.2 * total_range) &
                (upper_shadow > 0.1 * total_range) &
                (lower_shadow > 0.1 * total_range)
            )

            # Calculate confidence scores
            spinning_top_confidence = np.where(spinning_top_conditions, 0.7, 0.0)

            return {
                "spinning_top": spinning_top_conditions,
                "spinning_top_confidence": spinning_top_confidence,
            }

        except Exception as e:
            self.logger.error(f"Error detecting spinning top patterns: {e}")
            return {}

    def _convert_patterns_to_features_vectorized(
        self,
        patterns: dict[str, dict[str, np.ndarray]],
        df: pd.DataFrame,
    ) -> dict[str, float]:
        """Convert pattern analysis to ML features using vectorized operations."""
        try:
            features = {}

            window = 20  # rolling window for counts/densities

            # Calculate pattern type features (rolling count and presence per row)
            pattern_types = [
                "engulfing_patterns",
                "hammer_hanging_man",
                "shooting_star_inverted_hammer",
                "tweezer_patterns",
                "marubozu_patterns",
                "three_methods_patterns",
                "doji_patterns",
                "spinning_top_patterns",
            ]

            for pattern_type in pattern_types:
                if pattern_type in patterns:
                    pattern_data = patterns[pattern_type]
                    # Sum boolean arrays within type per row and then apply rolling sum
                    per_row_sum = np.zeros(len(df), dtype=float)
                    for key, arr in pattern_data.items():
                        if isinstance(arr, np.ndarray) and arr.dtype == bool:
                            per_row_sum += arr.astype(float)
                    per_row_series = pd.Series(per_row_sum, index=df.index)
                    count_series = per_row_series.rolling(window, min_periods=1).sum()
                    features[f"{pattern_type}_count"] = count_series.values
                    features[f"{pattern_type}_present"] = (count_series > 0).astype(float).values

            # Specific pattern rolling counts/presence
            specific_patterns = [
                "bullish_engulfing",
                "bearish_engulfing",
                "hammer",
                "hanging_man",
                "shooting_star",
                "inverted_hammer",
                "tweezer_top",
                "tweezer_bottom",
                "bullish_marubozu",
                "bearish_marubozu",
                "rising_three_methods",
                "falling_three_methods",
                "doji",
                "spinning_top",
            ]

            for pattern in specific_patterns:
                per_row = np.zeros(len(df), dtype=float)
                for pattern_data in patterns.values():
                    if pattern in pattern_data and isinstance(pattern_data[pattern], np.ndarray):
                        per_row += pattern_data[pattern].astype(float)
                per_row_series = pd.Series(per_row, index=df.index)
                count_series = per_row_series.rolling(window, min_periods=1).sum()
                features[f"{pattern}_count"] = count_series.values
                features[f"{pattern}_present"] = (count_series > 0).astype(float).values

            # Aggregate densities per row
            total_patterns_series = sum(
                pd.Series(features.get(f"{pt}_count"), index=df.index) for pt in pattern_types if f"{pt}_count" in features
            )
            features["total_patterns"] = total_patterns_series.values
            features["pattern_density"] = (total_patterns_series / window).values

            # Bullish vs bearish per row
            bullish_series = sum(
                pd.Series(features.get(f"{p}_count"), index=df.index)
                for p in [
                    "bullish_engulfing",
                    "hammer",
                    "inverted_hammer",
                    "tweezer_bottom",
                    "bullish_marubozu",
                    "rising_three_methods",
                ]
                if f"{p}_count" in features
            )
            bearish_series = sum(
                pd.Series(features.get(f"{p}_count"), index=df.index)
                for p in [
                    "bearish_engulfing",
                    "hanging_man",
                    "shooting_star",
                    "tweezer_top",
                    "bearish_marubozu",
                    "falling_three_methods",
                ]
                if f"{p}_count" in features
            )
            features["bullish_patterns"] = bullish_series.values
            features["bearish_patterns"] = bearish_series.values
            with np.errstate(divide='ignore', invalid='ignore'):
                ratio = (bullish_series / (bearish_series + 1e-8)).replace([np.inf, -np.inf], np.nan).fillna(0)
            features["bullish_bearish_ratio"] = ratio.values

            return features

        except Exception as e:
            self.logger.error(f"Error converting candlestick patterns to features: {e}")
            return {}


class VectorizedWaveletTransformAnalyzer:
    """
    Comprehensive wavelet transform analyzer for signal processing and feature extraction.
    Implements various wavelet transforms for financial time series analysis with proper
    boundary handling, scale selection, and dimensionality management.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.logger = system_logger.getChild("VectorizedWaveletTransformAnalyzer")

        # Enhanced wavelet configuration
        self.wavelet_config = config.get("wavelet_transforms", {})
        self.wavelet_type = self.wavelet_config.get("wavelet_type", "db4")
        self.decomposition_level = self.wavelet_config.get("decomposition_level", 4)
        
        # Boundary handling configuration
        self.padding_mode = self.wavelet_config.get("padding_mode", "symmetric")
        self.boundary_handling = self.wavelet_config.get("boundary_handling", "truncate")
        self.edge_effect_threshold = self.wavelet_config.get("edge_effect_threshold", 0.1)
        
        # CWT scale selection configuration
        self.cwt_scale_method = self.wavelet_config.get("cwt_scale_method", "logarithmic")
        self.min_scale = self.wavelet_config.get("min_scale", 1)
        self.max_scale = self.wavelet_config.get("max_scale", 32)
        self.num_scales = self.wavelet_config.get("num_scales", 16)
        self.scale_resolution = self.wavelet_config.get("scale_resolution", "octave")
        
        # Feature dimensionality management
        self.max_features_per_wavelet = self.wavelet_config.get("max_features_per_wavelet", 80)
        self.feature_selection_method = self.wavelet_config.get("feature_selection_method", "variance")
        self.min_feature_variance = self.wavelet_config.get("min_feature_variance", 1e-6)
        
        # Stationarity handling
        self.enable_stationary_series = self.wavelet_config.get("enable_stationary_series", True)
        self.stationary_transforms = self.wavelet_config.get("stationary_transforms", ["returns", "log_returns"])
        
        # Computational cost management
        self.max_wavelet_types = self.wavelet_config.get("max_wavelet_types", 3)
        self.enable_parallel_processing = self.wavelet_config.get("enable_parallel_processing", False)
        self.computation_timeout = self.wavelet_config.get("computation_timeout", 30)  # seconds
        
        # Enable/disable specific transforms
        self.enable_continuous_wavelet = self.wavelet_config.get("enable_continuous_wavelet", True)
        self.enable_discrete_wavelet = self.wavelet_config.get("enable_discrete_wavelet", True)
        self.enable_wavelet_packet = self.wavelet_config.get("enable_wavelet_packet", True)
        self.enable_denoising = self.wavelet_config.get("enable_denoising", True)

        # Wavelet types for different analyses (limited for efficiency)
        self.wavelet_types = ["db1", "db2", "db4", "db8", "haar", "sym2", "sym4", "coif1", "coif2"]
        
        # Performance tracking
        self.computation_times = {}
        self.feature_counts = {}
        
        self.is_initialized = False

    @handle_errors(
        exceptions=(Exception,),
        default_return=False,
        context="wavelet transform analyzer initialization",
    )
    async def initialize(self) -> bool:
        """Initialize wavelet transform analyzer with enhanced configuration."""
        try:
            self.logger.info("🚀 Initializing enhanced vectorized wavelet transform analyzer...")
            
            # Validate configuration
            self._validate_wavelet_config()
            
            # Initialize performance tracking
            self.computation_times = {}
            self.feature_counts = {}
            
            self.is_initialized = True
            self.logger.info("✅ Enhanced vectorized wavelet transform analyzer initialized successfully")
            self.logger.info(f"📊 Configuration: padding_mode={self.padding_mode}, "
                           f"cwt_scale_method={self.cwt_scale_method}, "
                           f"max_features_per_wavelet={self.max_features_per_wavelet}")
            return True
        except Exception as e:
            self.logger.error(
                f"❌ Error initializing enhanced vectorized wavelet transform analyzer: {e}",
            )
            return False

    def _validate_wavelet_config(self) -> None:
        """Validate wavelet configuration parameters."""
        try:
            # Validate padding mode
            valid_padding_modes = ["symmetric", "periodic", "zero", "constant", "edge", "reflect"]
            if self.padding_mode not in valid_padding_modes:
                self.logger.warning(f"Invalid padding_mode '{self.padding_mode}', using 'symmetric'")
                self.padding_mode = "symmetric"
            
            # Validate CWT scale method
            valid_scale_methods = ["linear", "logarithmic", "octave", "adaptive"]
            if self.cwt_scale_method not in valid_scale_methods:
                self.logger.warning(f"Invalid cwt_scale_method '{self.cwt_scale_method}', using 'logarithmic'")
                self.cwt_scale_method = "logarithmic"
            
            # Validate feature selection method
            valid_selection_methods = ["variance", "energy", "entropy", "random"]
            if self.feature_selection_method not in valid_selection_methods:
                self.logger.warning(f"Invalid feature_selection_method '{self.feature_selection_method}', using 'variance'")
                self.feature_selection_method = "variance"
            
            # Validate scale parameters
            if self.min_scale <= 0 or self.max_scale <= self.min_scale:
                self.logger.warning("Invalid scale parameters, using defaults")
                self.min_scale = 1
                self.max_scale = 32
            
            if self.num_scales <= 0 or self.num_scales > 100:
                self.logger.warning("Invalid num_scales, using default")
                self.num_scales = 16
                
        except Exception as e:
            self.logger.error(f"Error validating wavelet configuration: {e}")

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return={},
        context="enhanced wavelet transform analysis",
    )
    async def analyze_wavelet_transforms(
        self,
        price_data: pd.DataFrame,
        volume_data: pd.DataFrame | None = None,
    ) -> dict[str, Any]:
        """
        Analyze wavelet transforms for signal processing and feature extraction with enhanced handling.
        
        OPTIMIZED VERSION: Reduced computational complexity for better performance
        """
        try:
            if not self.is_initialized:
                self.logger.error("Wavelet transform analyzer not initialized")
                return {}

            if price_data.empty:
                self.logger.warning("Empty price data provided for wavelet analysis")
                return {}

            self.logger.info("🔍 Performing enhanced wavelet transform analysis...")
            start_time = time.time()

            # OPTIMIZATION: Limit data size for wavelet analysis
            max_records = self.config.get("wavelet_max_records", 50000)
            if len(price_data) > max_records:
                self.logger.info(f"📊 Limiting wavelet analysis to {max_records} records (from {len(price_data)})")
                # Take the most recent records for analysis
                price_data = price_data.tail(max_records).copy()
                if volume_data is not None:
                    volume_data = volume_data.tail(max_records).copy()

            features = {}
            total_features = 0

            # 1. Prepare stationary series for analysis
            stationary_series = self._prepare_stationary_series(price_data)
            
            # 2. Discrete Wavelet Transform (DWT) analysis - OPTIMIZED
            if self.enable_discrete_wavelet:
                dwt_start = time.time()
                self.logger.info("🔄 Computing DWT features...")
                dwt_features = await self._analyze_discrete_wavelet_transforms_enhanced(
                    price_data, stationary_series
                )
                features.update(dwt_features)
                self.computation_times["dwt"] = time.time() - dwt_start
                self.feature_counts["dwt"] = len(dwt_features)
                total_features += len(dwt_features)
                self.logger.info(f"✅ DWT completed: {len(dwt_features)} features in {self.computation_times['dwt']:.2f}s")

            # 3. Continuous Wavelet Transform (CWT) analysis
            if self.enable_continuous_wavelet:
                cwt_start = time.time()
                self.logger.info("🔄 Computing CWT features...")
                cwt_features = await self._analyze_continuous_wavelet_transforms_enhanced(
                    price_data, stationary_series
                )
                features.update(cwt_features)
                self.computation_times["cwt"] = time.time() - cwt_start
                self.feature_counts["cwt"] = len(cwt_features)
                total_features += len(cwt_features)
                self.logger.info(f"✅ CWT completed: {len(cwt_features)} features in {self.computation_times['cwt']:.2f}s")

            # 4. Wavelet Packet analysis
            if self.enable_wavelet_packet:
                packet_start = time.time()
                self.logger.info("🔄 Computing wavelet packet features...")
                packet_features = await self._analyze_wavelet_packets_enhanced(
                    price_data, stationary_series
                )
                features.update(packet_features)
                self.computation_times["packet"] = time.time() - packet_start
                self.feature_counts["packet"] = len(packet_features)
                total_features += len(packet_features)
                self.logger.info(f"✅ Wavelet packets completed: {len(packet_features)} features in {self.computation_times['packet']:.2f}s")

            # OPTIMIZATION: Skip expensive denoising analysis
            if self.enable_denoising and len(price_data) <= 5000:
                denoising_start = time.time()
                self.logger.info("🔄 Computing wavelet denoising features...")
                denoising_features = await self._analyze_wavelet_denoising_enhanced(
                    price_data, stationary_series
                )
                features.update(denoising_features)
                self.computation_times["denoising"] = time.time() - denoising_start
                self.feature_counts["denoising"] = len(denoising_features)
                total_features += len(denoising_features)
                self.logger.info(f"✅ Wavelet denoising completed: {len(denoising_features)} features in {self.computation_times['denoising']:.2f}s")
            else:
                self.logger.info("⏭️ Skipping wavelet denoising analysis (dataset too large or disabled)")

            # OPTIMIZATION: Skip expensive multi-wavelet analysis
            multi_start = time.time()
            self.logger.info("🔄 Computing multi-wavelet features...")
            multi_wavelet_features = await self._analyze_multi_wavelet_transforms_enhanced(
                price_data, stationary_series
            )
            features.update(multi_wavelet_features)
            self.computation_times["multi_wavelet"] = time.time() - multi_start
            self.feature_counts["multi_wavelet"] = len(multi_wavelet_features)
            total_features += len(multi_wavelet_features)
            self.logger.info(f"✅ Multi-wavelet completed: {len(multi_wavelet_features)} features in {self.computation_times['multi_wavelet']:.2f}s")

            # OPTIMIZATION: Skip volume wavelet analysis for large datasets
            if volume_data is not None and not volume_data.empty and len(volume_data) <= 10000:
                volume_start = time.time()
                self.logger.info("🔄 Computing volume wavelet features...")
                volume_wavelet_features = await self._analyze_volume_wavelet_transforms_enhanced(
                    volume_data
                )
                features.update(volume_wavelet_features)
                self.computation_times["volume_wavelet"] = time.time() - volume_start
                self.feature_counts["volume_wavelet"] = len(volume_wavelet_features)
                total_features += len(volume_wavelet_features)
                self.logger.info(f"✅ Volume wavelet completed: {len(volume_wavelet_features)} features in {self.computation_times['volume_wavelet']:.2f}s")
            else:
                self.logger.info("⏭️ Skipping volume wavelet analysis (dataset too large or no volume data)")

            # 8. Feature selection and dimensionality reduction
            self.logger.info("🔄 Selecting optimal wavelet features...")
            selected_features = self._select_optimal_wavelet_features(features)
            
            total_time = time.time() - start_time
            self.computation_times["total"] = total_time
            
            self.logger.info(
                f"✅ Wavelet analysis completed: {len(selected_features)} selected features "
                f"from {total_features} total features in {total_time:.2f}s"
            )
            
            return selected_features

        except Exception as e:
            self.logger.error(f"Error in enhanced wavelet transform analysis: {e}")
            return {}

    def _prepare_stationary_series(self, price_data: pd.DataFrame) -> dict[str, np.ndarray]:
        """Prepare stationary series for wavelet analysis using price differences."""
        try:
            stationary_series = {}
            
            if self.enable_stationary_series:
                # Price differences (first difference) - primary focus
                price_diff = price_data["close"].diff().dropna().values
                if len(price_diff) > 0:
                    stationary_series["price_diff"] = price_diff
                
                # Returns (percentage change)
                returns = price_data["close"].pct_change().dropna().values
                if len(returns) > 0:
                    stationary_series["returns"] = returns
                
                # Log returns
                log_returns = np.log(price_data["close"] / price_data["close"].shift(1)).dropna().values
                if len(log_returns) > 0:
                    stationary_series["log_returns"] = log_returns
                
                # Second differences (acceleration)
                price_diff_2 = price_data["close"].diff().diff().dropna().values
                if len(price_diff_2) > 0:
                    stationary_series["price_diff_2"] = price_diff_2
                
                # Detrended series
                close_prices = price_data["close"].values
                trend = np.polyfit(range(len(close_prices)), close_prices, 1)
                detrended = close_prices - (trend[0] * np.arange(len(close_prices)) + trend[1])
                stationary_series["detrended"] = detrended
            
            # Price differences as primary series (not raw prices)
            price_diff = price_data["close"].diff().dropna().values
            if len(price_diff) > 0:
                stationary_series["close"] = price_diff  # Use price differences instead of raw prices
            
            return stationary_series

        except Exception as e:
            self.logger.error(f"Error preparing stationary series: {e}")
            # Fallback to price differences
            price_diff = price_data["close"].diff().dropna().values
            return {"close": price_diff if len(price_diff) > 0 else price_data["close"].values}

    async def _analyze_discrete_wavelet_transforms_enhanced(
        self, 
        price_data: pd.DataFrame, 
        stationary_series: dict[str, np.ndarray],
        wavelet_types: list[str] | None = None
    ) -> dict[str, float]:
        """Analyze discrete wavelet transforms with enhanced boundary handling."""
        try:
            features = {}
            
            # Use provided wavelet types or fall back to configured types
            if wavelet_types is None:
                wavelet_types = self.wavelet_types
            
            for wavelet_type in wavelet_types:
                try:
                    # Analyze each stationary series
                    for series_name, series_data in stationary_series.items():
                        if len(series_data) < 2 ** self.decomposition_level:
                            self.logger.warning(f"Insufficient data for {series_name} with {wavelet_type}")
                            continue
                        
                        # Perform wavelet decomposition with boundary handling
                        coeffs = pywt.wavedec(
                            series_data, 
                            wavelet_type, 
                            level=self.decomposition_level,
                            mode=self.padding_mode
                        )
                        
                        # Extract features with boundary effect management
                        dwt_features = self._extract_dwt_features_enhanced(
                            coeffs, wavelet_type, series_name
                        )
                        features.update(dwt_features)
                        
                except Exception as e:
                    self.logger.warning(f"Error with wavelet type {wavelet_type}: {e}")
                    continue

            return features

        except Exception as e:
            self.logger.error(f"Error in enhanced discrete wavelet transform analysis: {e}")
            return {}

    def _extract_dwt_features_enhanced(
        self, 
        coeffs: list, 
        wavelet_type: str, 
        series_name: str
    ) -> dict[str, float]:
        """Extract features from DWT coefficients with boundary effect management."""
        try:
            features = {}
            
            # Energy features for each level with boundary effect consideration
            for i, coeff in enumerate(coeffs):
                if len(coeff) > 0:
                    # Remove boundary effects if significant
                    if self.boundary_handling == "truncate" and len(coeff) > 10:
                        # Remove first and last 10% of coefficients to avoid boundary effects
                        truncate_size = max(1, int(len(coeff) * 0.1))
                        coeff_clean = coeff[truncate_size:-truncate_size]
                    else:
                        coeff_clean = coeff
                    
                    if len(coeff_clean) > 0:
                        energy = np.sum(coeff_clean ** 2)
                        features[f"{wavelet_type}_{series_name}_level_{i}_energy"] = energy
                        features[f"{wavelet_type}_{series_name}_level_{i}_energy_normalized"] = energy / len(coeff_clean)
                        
                        # Statistical features
                        features[f"{wavelet_type}_{series_name}_level_{i}_mean"] = np.mean(coeff_clean)
                        features[f"{wavelet_type}_{series_name}_level_{i}_std"] = np.std(coeff_clean)
                        features[f"{wavelet_type}_{series_name}_level_{i}_max"] = np.max(coeff_clean)
                        features[f"{wavelet_type}_{series_name}_level_{i}_min"] = np.min(coeff_clean)
                        
                        # Entropy features
                        if energy > 0:
                            entropy = -np.sum((coeff_clean ** 2) / energy * 
                                            np.log((coeff_clean ** 2) / energy + 1e-10))
                            features[f"{wavelet_type}_{series_name}_level_{i}_entropy"] = entropy
                        
                        # Boundary effect features
                        if len(coeff) > len(coeff_clean):
                            boundary_ratio = len(coeff_clean) / len(coeff)
                            features[f"{wavelet_type}_{series_name}_level_{i}_boundary_ratio"] = boundary_ratio

            # Cross-level features
            if len(coeffs) > 1:
                for i in range(len(coeffs) - 1):
                    energy_i = np.sum(coeffs[i] ** 2)
                    energy_j = np.sum(coeffs[i + 1] ** 2)
                    if energy_i > 0:
                        energy_ratio = energy_j / energy_i
                        features[f"{wavelet_type}_{series_name}_energy_ratio_{i}_{i+1}"] = energy_ratio

            return features

        except Exception as e:
            self.logger.error(f"Error extracting enhanced DWT features: {e}")
            return {}

    async def _analyze_continuous_wavelet_transforms_enhanced(
        self, 
        price_data: pd.DataFrame, 
        stationary_series: dict[str, np.ndarray]
    ) -> dict[str, float]:
        """Analyze continuous wavelet transforms with dynamic scale selection."""
        try:
            features = {}
            
            # Generate scales based on configuration
            scales = self._generate_cwt_scales(len(price_data))
            
            for wavelet_type in ["morl", "cmor1.5-1.0"]:
                try:
                    # Analyze each stationary series
                    for series_name, series_data in stationary_series.items():
                        if len(series_data) < 10:  # Minimum length for CWT
                            continue
                        
                        # Perform continuous wavelet transform with boundary handling
                        coeffs, freqs = pywt.cwt(
                            series_data, 
                            scales, 
                            wavelet_type,
                            method='conv'
                        )
                        
                        # Extract CWT features with scale information
                        cwt_features = self._extract_cwt_features_enhanced(
                            coeffs, freqs, wavelet_type, series_name, scales
                        )
                        features.update(cwt_features)
                        
                except Exception as e:
                    self.logger.warning(f"Error with CWT wavelet type {wavelet_type}: {e}")
                    continue

            return features

        except Exception as e:
            self.logger.error(f"Error in enhanced continuous wavelet transform analysis: {e}")
            return {}

    def _generate_cwt_scales(self, signal_length: int) -> np.ndarray:
        """Generate CWT scales based on configuration and signal length."""
        try:
            if self.cwt_scale_method == "logarithmic":
                # Logarithmic scale distribution
                scales = np.logspace(
                    np.log10(self.min_scale), 
                    np.log10(self.max_scale), 
                    self.num_scales
                )
            elif self.cwt_scale_method == "linear":
                # Linear scale distribution
                scales = np.linspace(self.min_scale, self.max_scale, self.num_scales)
            elif self.cwt_scale_method == "octave":
                # Octave-based scale distribution
                scales = 2 ** np.linspace(
                    np.log2(self.min_scale), 
                    np.log2(self.max_scale), 
                    self.num_scales
                )
            elif self.cwt_scale_method == "adaptive":
                # Adaptive scale selection based on signal length
                min_scale = max(1, signal_length // 100)
                max_scale = min(signal_length // 4, self.max_scale)
                scales = np.logspace(
                    np.log10(min_scale), 
                    np.log10(max_scale), 
                    self.num_scales
                )
            else:
                # Default to logarithmic
                scales = np.logspace(
                    np.log10(self.min_scale), 
                    np.log10(self.max_scale), 
                    self.num_scales
                )
            
            return scales.astype(int)

        except Exception as e:
            self.logger.error(f"Error generating CWT scales: {e}")
            return np.arange(self.min_scale, self.max_scale, 2)

    def _extract_cwt_features_enhanced(
        self, 
        coeffs: np.ndarray, 
        freqs: np.ndarray, 
        wavelet_type: str, 
        series_name: str,
        scales: np.ndarray
    ) -> dict[str, float]:
        """Extract features from CWT coefficients with enhanced analysis."""
        try:
            features = {}
            
            # Energy features with boundary effect consideration
            energy = np.sum(np.abs(coeffs) ** 2, axis=1)
            
            # Remove boundary effects from energy calculation
            if self.boundary_handling == "truncate" and coeffs.shape[1] > 20:
                truncate_size = max(1, coeffs.shape[1] // 10)
                energy_clean = np.sum(np.abs(coeffs[:, truncate_size:-truncate_size]) ** 2, axis=1)
            else:
                energy_clean = energy
            
            features[f"{wavelet_type}_{series_name}_total_energy"] = np.sum(energy_clean)
            features[f"{wavelet_type}_{series_name}_max_energy"] = np.max(energy_clean)
            features[f"{wavelet_type}_{series_name}_min_energy"] = np.min(energy_clean)
            features[f"{wavelet_type}_{series_name}_energy_std"] = np.std(energy_clean)
            
            # Frequency features
            if len(energy_clean) > 0:
                features[f"{wavelet_type}_{series_name}_dominant_freq"] = freqs[np.argmax(energy_clean)]
                features[f"{wavelet_type}_{series_name}_freq_range"] = np.max(freqs) - np.min(freqs)
                features[f"{wavelet_type}_{series_name}_energy_bandwidth"] = np.std(freqs[energy_clean > np.mean(energy_clean)])
            
            # Scale-specific features
            features[f"{wavelet_type}_{series_name}_min_scale"] = np.min(scales)
            features[f"{wavelet_type}_{series_name}_max_scale"] = np.max(scales)
            features[f"{wavelet_type}_{series_name}_scale_range"] = np.max(scales) - np.min(scales)
            
            # Statistical features
            features[f"{wavelet_type}_{series_name}_coeff_mean"] = np.mean(np.abs(coeffs))
            features[f"{wavelet_type}_{series_name}_coeff_std"] = np.std(np.abs(coeffs))
            features[f"{wavelet_type}_{series_name}_coeff_max"] = np.max(np.abs(coeffs))
            features[f"{wavelet_type}_{series_name}_coeff_min"] = np.min(np.abs(coeffs))
            
            # Entropy features
            total_energy = np.sum(np.abs(coeffs) ** 2)
            if total_energy > 0:
                entropy = -np.sum((np.abs(coeffs) ** 2) / total_energy * 
                                np.log((np.abs(coeffs) ** 2) / total_energy + 1e-10))
                features[f"{wavelet_type}_{series_name}_entropy"] = entropy

            return features

        except Exception as e:
            self.logger.error(f"Error extracting enhanced CWT features: {e}")
            return {}

    def _select_optimal_wavelet_features(self, features: dict[str, Any]) -> dict[str, Any]:
        """Select optimal wavelet features based on configured method."""
        try:
            if len(features) <= self.max_features_per_wavelet:
                return features
            
            if self.feature_selection_method == "variance":
                # Select features with highest variance
                feature_vars = {}
                for feature_name, feature_value in features.items():
                    if isinstance(feature_value, (int, float)) and not np.isnan(feature_value):
                        feature_vars[feature_name] = abs(feature_value)
                
                # Sort by variance and select top features
                sorted_features = sorted(feature_vars.items(), key=lambda x: x[1], reverse=True)
                selected_features = dict(sorted_features[:self.max_features_per_wavelet])
                
            elif self.feature_selection_method == "energy":
                # Select features with highest energy content
                energy_features = {}
                for feature_name, feature_value in features.items():
                    if "energy" in feature_name.lower() and isinstance(feature_value, (int, float)):
                        energy_features[feature_name] = abs(feature_value)
                
                sorted_features = sorted(energy_features.items(), key=lambda x: x[1], reverse=True)
                selected_features = dict(sorted_features[:self.max_features_per_wavelet])
                
            elif self.feature_selection_method == "entropy":
                # Select features with highest entropy
                entropy_features = {}
                for feature_name, feature_value in features.items():
                    if "entropy" in feature_name.lower() and isinstance(feature_value, (int, float)):
                        entropy_features[feature_name] = abs(feature_value)
                
                sorted_features = sorted(entropy_features.items(), key=lambda x: x[1], reverse=True)
                selected_features = dict(sorted_features[:self.max_features_per_wavelet])
                
            else:  # random
                # Random selection
                feature_items = list(features.items())
                random.shuffle(feature_items)
                selected_features = dict(feature_items[:self.max_features_per_wavelet])
            
            self.logger.info(f"Selected {len(selected_features)} features from {len(features)} candidates "
                           f"using {self.feature_selection_method} method")
            
            return selected_features

        except Exception as e:
            self.logger.error(f"Error selecting optimal wavelet features: {e}")
            return features

    def _log_performance_metrics(self) -> None:
        """Log performance metrics for wavelet analysis."""
        try:
            self.logger.info("📊 Wavelet Analysis Performance Metrics:")
            for component, time_taken in self.computation_times.items():
                if component != "total":
                    self.logger.info(f"  {component}: {time_taken:.3f}s ({self.feature_counts.get(component, 0)} features)")
            
            total_time = self.computation_times.get("total", 0)
            total_features = self.feature_counts.get("total", 0)
            self.logger.info(f"  Total: {total_time:.3f}s ({total_features} features)")
            
            if total_time > 0:
                features_per_second = total_features / total_time
                self.logger.info(f"  Performance: {features_per_second:.1f} features/second")
                
        except Exception as e:
            self.logger.error(f"Error logging performance metrics: {e}")

    # Placeholder methods for other enhanced wavelet analyses
    async def _analyze_wavelet_packets_enhanced(
        self, 
        price_data: pd.DataFrame, 
        stationary_series: dict[str, np.ndarray]
    ) -> dict[str, float]:
        """Enhanced wavelet packet analysis - OPTIMIZED for large datasets."""
        try:
            features = {}
            
            # For large datasets, use a simplified approach
            if len(price_data) > 5000:
                # Use only close price for efficiency
                close_series = price_data['close'].values
                
                # Simple wavelet packet features using only db4
                try:
                    # Single level decomposition for efficiency
                    coeffs = pywt.wavedec(close_series, 'db4', level=1, mode='symmetric')
                    
                    if len(coeffs) >= 2:
                        # Extract basic features from approximation and detail coefficients
                        approx_coeffs = coeffs[0]
                        detail_coeffs = coeffs[1]
                        
                        # Energy features
                        features['wavelet_packet_approx_energy'] = np.sum(approx_coeffs ** 2)
                        features['wavelet_packet_detail_energy'] = np.sum(detail_coeffs ** 2)
                        
                        # Statistical features
                        features['wavelet_packet_approx_mean'] = np.mean(approx_coeffs)
                        features['wavelet_packet_approx_std'] = np.std(approx_coeffs)
                        features['wavelet_packet_detail_mean'] = np.mean(detail_coeffs)
                        features['wavelet_packet_detail_std'] = np.std(detail_coeffs)
                        
                        # Energy ratio
                        total_energy = features['wavelet_packet_approx_energy'] + features['wavelet_packet_detail_energy']
                        if total_energy > 0:
                            features['wavelet_packet_energy_ratio'] = features['wavelet_packet_detail_energy'] / total_energy
                        else:
                            features['wavelet_packet_energy_ratio'] = 0.0
                            
                except Exception as e:
                    self.logger.warning(f"Error in simplified wavelet packet analysis: {e}")
                    
            return features
            
        except Exception as e:
            self.logger.error(f"Error in enhanced wavelet packet analysis: {e}")
            return {}

    async def _analyze_wavelet_denoising_enhanced(
        self, 
        price_data: pd.DataFrame, 
        stationary_series: dict[str, np.ndarray]
    ) -> dict[str, float]:
        """Enhanced wavelet denoising analysis - OPTIMIZED for large datasets."""
        try:
            features = {}
            
            # For large datasets, use a simplified approach
            if len(price_data) > 5000:
                # Use only close price for efficiency
                close_series = price_data['close'].values
                
                try:
                    # Simple denoising using soft thresholding
                    coeffs = pywt.wavedec(close_series, 'db4', level=2, mode='symmetric')
                    
                    if len(coeffs) >= 3:
                        # Apply soft thresholding to detail coefficients
                        threshold = np.std(coeffs[1]) * np.sqrt(2 * np.log(len(coeffs[1])))
                        
                        # Denoise detail coefficients
                        denoised_coeffs = []
                        for i, coeff in enumerate(coeffs):
                            if i == 0:  # Approximation coefficients (keep as is)
                                denoised_coeffs.append(coeff)
                            else:  # Detail coefficients (apply thresholding)
                                denoised_coeff = np.sign(coeff) * np.maximum(np.abs(coeff) - threshold, 0)
                                denoised_coeffs.append(denoised_coeff)
                        
                        # Reconstruct denoised signal
                        denoised_signal = pywt.waverec(denoised_coeffs, 'db4', mode='symmetric')
                        
                        # Calculate denoising features
                        if len(denoised_signal) == len(close_series):
                            # Noise reduction ratio
                            original_energy = np.sum(close_series ** 2)
                            denoised_energy = np.sum(denoised_signal ** 2)
                            
                            if original_energy > 0:
                                features['wavelet_denoising_noise_reduction'] = 1 - (denoised_energy / original_energy)
                            else:
                                features['wavelet_denoising_noise_reduction'] = 0.0
                            
                            # Signal quality improvement
                            original_std = np.std(close_series)
                            denoised_std = np.std(denoised_signal)
                            
                            if original_std > 0:
                                features['wavelet_denoising_signal_quality'] = denoised_std / original_std
                            else:
                                features['wavelet_denoising_signal_quality'] = 1.0
                                
                except Exception as e:
                    self.logger.warning(f"Error in simplified wavelet denoising: {e}")
                    
            return features
            
        except Exception as e:
            self.logger.error(f"Error in enhanced wavelet denoising analysis: {e}")
            return {}

    async def _analyze_multi_wavelet_transforms_enhanced(
        self, 
        price_data: pd.DataFrame, 
        stationary_series: dict[str, np.ndarray]
    ) -> dict[str, float]:
        """Enhanced multi-wavelet transform analysis - OPTIMIZED for large datasets."""
        try:
            features = {}
            
            # For large datasets, use a simplified approach with multiple wavelet types
            if len(price_data) > 5000:
                # Use only close price for efficiency
                close_series = price_data['close'].values
                
                # Use configured wavelet types for comparison
                wavelet_types = self.wavelet_types[:3]  # Use first 3 types for efficiency
                
                for wavelet_type in wavelet_types:
                    try:
                        # Single level decomposition for efficiency
                        coeffs = pywt.wavedec(close_series, wavelet_type, level=1, mode='symmetric')
                        
                        if len(coeffs) >= 2:
                            approx_coeffs = coeffs[0]
                            detail_coeffs = coeffs[1]
                            
                            # Energy features for each wavelet type
                            features[f'multi_wavelet_{wavelet_type}_approx_energy'] = np.sum(approx_coeffs ** 2)
                            features[f'multi_wavelet_{wavelet_type}_detail_energy'] = np.sum(detail_coeffs ** 2)
                            
                            # Statistical features for each wavelet type
                            features[f'multi_wavelet_{wavelet_type}_approx_std'] = np.std(approx_coeffs)
                            features[f'multi_wavelet_{wavelet_type}_detail_std'] = np.std(detail_coeffs)
                            
                    except Exception as e:
                        self.logger.warning(f"Error with wavelet type {wavelet_type}: {e}")
                        continue
                
                # Cross-wavelet comparison features
                if 'multi_wavelet_db4_approx_energy' in features and 'multi_wavelet_haar_approx_energy' in features:
                    db4_energy = features['multi_wavelet_db4_approx_energy']
                    haar_energy = features['multi_wavelet_haar_approx_energy']
                    
                    if haar_energy > 0:
                        features['multi_wavelet_db4_haar_energy_ratio'] = db4_energy / haar_energy
                    else:
                        features['multi_wavelet_db4_haar_energy_ratio'] = 1.0
                        
            return features
            
        except Exception as e:
            self.logger.error(f"Error in enhanced multi-wavelet transform analysis: {e}")
            return {}

    async def _analyze_volume_wavelet_transforms_enhanced(
        self, 
        volume_data: pd.DataFrame
    ) -> dict[str, float]:
        """Enhanced volume wavelet transform analysis - OPTIMIZED for large datasets."""
        try:
            features = {}
            
            # For large datasets, use a simplified approach
            if len(volume_data) > 10000:
                # Use only volume for efficiency
                volume_series = volume_data['volume'].values
                
                try:
                    # Simple wavelet analysis on volume
                    coeffs = pywt.wavedec(volume_series, 'db4', level=1, mode='symmetric')
                    
                    if len(coeffs) >= 2:
                        approx_coeffs = coeffs[0]
                        detail_coeffs = coeffs[1]
                        
                        # Volume-specific wavelet features
                        features['volume_wavelet_approx_energy'] = np.sum(approx_coeffs ** 2)
                        features['volume_wavelet_detail_energy'] = np.sum(detail_coeffs ** 2)
                        
                        # Volume volatility features
                        features['volume_wavelet_approx_volatility'] = np.std(approx_coeffs)
                        features['volume_wavelet_detail_volatility'] = np.std(detail_coeffs)
                        
                        # Volume trend features
                        features['volume_wavelet_approx_trend'] = np.mean(approx_coeffs)
                        features['volume_wavelet_detail_trend'] = np.mean(detail_coeffs)
                        
                        # Energy ratio for volume
                        total_volume_energy = features['volume_wavelet_approx_energy'] + features['volume_wavelet_detail_energy']
                        if total_volume_energy > 0:
                            features['volume_wavelet_energy_ratio'] = features['volume_wavelet_detail_energy'] / total_volume_energy
                        else:
                            features['volume_wavelet_energy_ratio'] = 0.0
                            
                except Exception as e:
                    self.logger.warning(f"Error in simplified volume wavelet analysis: {e}")
                    
            return features
            
        except Exception as e:
            self.logger.error(f"Error in enhanced volume wavelet transform analysis: {e}")
            return {}
