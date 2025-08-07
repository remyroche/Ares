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

            self.logger.info(
                f"✅ Engineered {len(selected_features)} vectorized advanced features including wavelet transforms",
            )
            return selected_features

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

            # Price impact features
            features["price_impact"] = self._calculate_price_impact_vectorized(price_data, volume_data)
            features["volume_price_impact"] = self._calculate_volume_price_impact_vectorized(price_data, volume_data)

            # Order flow imbalance features
            if order_flow_data is not None:
                features["order_flow_imbalance"] = self._calculate_order_flow_imbalance_vectorized(order_flow_data)
                features["bid_ask_spread"] = self._calculate_bid_ask_spread_vectorized(order_flow_data)

            # Market depth features
            features["market_depth"] = self._calculate_market_depth_vectorized(price_data, volume_data)

            return features

        except Exception as e:
            self.logger.error(f"Error engineering microstructure features: {e}")
            return {}

    def _calculate_price_impact_vectorized(self, price_data: pd.DataFrame, volume_data: pd.DataFrame) -> float:
        """Calculate price impact using vectorized operations."""
        try:
            price_changes = price_data["close"].pct_change().abs()
            volume_changes = volume_data["volume"].pct_change().abs()
            
            # Calculate price impact as correlation between price and volume changes
            correlation = np.corrcoef(price_changes.dropna(), volume_changes.dropna())[0, 1]
            return correlation if not np.isnan(correlation) else 0.0

        except Exception as e:
            self.logger.error(f"Error calculating price impact: {e}")
            return 0.0

    def _calculate_volume_price_impact_vectorized(self, price_data: pd.DataFrame, volume_data: pd.DataFrame) -> float:
        """Calculate volume-price impact using vectorized operations."""
        try:
            # Calculate volume-weighted price changes
            price_changes = price_data["close"].pct_change()
            volume_weights = volume_data["volume"] / volume_data["volume"].sum()
            
            # Volume-weighted average price change
            vwap_change = np.sum(price_changes.dropna() * volume_weights.dropna())
            return vwap_change

        except Exception as e:
            self.logger.error(f"Error calculating volume-price impact: {e}")
            return 0.0

    def _calculate_order_flow_imbalance_vectorized(self, order_flow_data: pd.DataFrame) -> float:
        """Calculate order flow imbalance using vectorized operations."""
        try:
            # Simplified order flow imbalance calculation
            # In practice, this would use actual order book data
            return 0.0  # Placeholder

        except Exception as e:
            self.logger.error(f"Error calculating order flow imbalance: {e}")
            return 0.0

    def _calculate_bid_ask_spread_vectorized(self, order_flow_data: pd.DataFrame) -> float:
        """Calculate bid-ask spread using vectorized operations."""
        try:
            # Simplified bid-ask spread calculation
            # In practice, this would use actual order book data
            return 0.0  # Placeholder

        except Exception as e:
            self.logger.error(f"Error calculating bid-ask spread: {e}")
            return 0.0

    def _calculate_market_depth_vectorized(self, price_data: pd.DataFrame, volume_data: pd.DataFrame) -> float:
        """Calculate market depth using vectorized operations."""
        try:
            # Market depth as average volume over time
            market_depth = volume_data["volume"].rolling(window=20).mean().iloc[-1]
            return market_depth

        except Exception as e:
            self.logger.error(f"Error calculating market depth: {e}")
            return 0.0

    def _engineer_adaptive_indicators_vectorized(self, price_data: pd.DataFrame) -> dict[str, Any]:
        """Engineer adaptive indicators using vectorized operations."""
        try:
            features = {}

            # Adaptive moving averages
            features["adaptive_sma"] = self._calculate_adaptive_sma_vectorized(price_data)
            features["adaptive_ema"] = self._calculate_adaptive_ema_vectorized(price_data)

            # Adaptive volatility indicators
            features["adaptive_atr"] = self._calculate_adaptive_atr_vectorized(price_data)
            features["adaptive_bollinger"] = self._calculate_adaptive_bollinger_vectorized(price_data)

            return features

        except Exception as e:
            self.logger.error(f"Error engineering adaptive indicators: {e}")
            return {}

    def _calculate_adaptive_sma_vectorized(self, price_data: pd.DataFrame) -> float:
        """Calculate adaptive SMA using vectorized operations."""
        try:
            # Adaptive SMA based on volatility
            volatility = price_data["close"].pct_change().rolling(window=20).std()
            adaptive_window = np.clip(20 / (1 + volatility * 100), 5, 50).astype(int)
            
            # Calculate adaptive SMA
            adaptive_sma = price_data["close"].rolling(window=adaptive_window.iloc[-1]).mean().iloc[-1]
            return adaptive_sma

        except Exception as e:
            self.logger.error(f"Error calculating adaptive SMA: {e}")
            return 0.0

    def _calculate_adaptive_ema_vectorized(self, price_data: pd.DataFrame) -> float:
        """Calculate adaptive EMA using vectorized operations."""
        try:
            # Adaptive EMA based on volatility
            volatility = price_data["close"].pct_change().rolling(window=20).std()
            adaptive_span = np.clip(12 / (1 + volatility * 100), 2, 50)
            
            # Calculate adaptive EMA
            adaptive_ema = price_data["close"].ewm(span=adaptive_span.iloc[-1]).mean().iloc[-1]
            return adaptive_ema

        except Exception as e:
            self.logger.error(f"Error calculating adaptive EMA: {e}")
            return 0.0

    def _calculate_adaptive_atr_vectorized(self, price_data: pd.DataFrame) -> float:
        """Calculate adaptive ATR using vectorized operations."""
        try:
            # Adaptive ATR based on volatility regime
            high_low = price_data["high"] - price_data["low"]
            high_close = np.abs(price_data["high"] - price_data["close"].shift())
            low_close = np.abs(price_data["low"] - price_data["close"].shift())
            
            true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            
            # Adaptive window based on volatility
            volatility = true_range.rolling(window=20).std()
            adaptive_window = np.clip(14 / (1 + volatility * 10), 5, 30).astype(int)
            
            adaptive_atr = true_range.rolling(window=adaptive_window.iloc[-1]).mean().iloc[-1]
            return adaptive_atr

        except Exception as e:
            self.logger.error(f"Error calculating adaptive ATR: {e}")
            return 0.0

    def _calculate_adaptive_bollinger_vectorized(self, price_data: pd.DataFrame) -> float:
        """Calculate adaptive Bollinger Bands using vectorized operations."""
        try:
            # Adaptive Bollinger Bands based on volatility
            volatility = price_data["close"].pct_change().rolling(window=20).std()
            adaptive_window = np.clip(20 / (1 + volatility * 100), 10, 50).astype(int)
            
            # Calculate adaptive Bollinger Bands
            sma = price_data["close"].rolling(window=adaptive_window.iloc[-1]).mean()
            std = price_data["close"].rolling(window=adaptive_window.iloc[-1]).std()
            
            upper_band = sma + (std * 2)
            lower_band = sma - (std * 2)
            
            # Return position within bands
            current_price = price_data["close"].iloc[-1]
            position = (current_price - lower_band.iloc[-1]) / (upper_band.iloc[-1] - lower_band.iloc[-1])
            return position

        except Exception as e:
            self.logger.error(f"Error calculating adaptive Bollinger Bands: {e}")
            return 0.0

    def _select_optimal_features_vectorized(self, features: dict[str, Any]) -> dict[str, Any]:
        """Select optimal features using vectorized operations."""
        try:
            # Simple feature selection based on variance
            selected_features = {}
            
            for feature_name, feature_value in features.items():
                if isinstance(feature_value, (int, float)) and not np.isnan(feature_value):
                    selected_features[feature_name] = feature_value
            
            return selected_features

        except Exception as e:
            self.logger.error(f"Error selecting optimal features: {e}")
            return features

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
                
                # Add timeframe prefix to features
                for feature_name, feature_value in timeframe_features.items():
                    features[f"{timeframe}_{feature_name}"] = feature_value

            return features

        except Exception as e:
            self.logger.error(f"Error engineering multi-timeframe features: {e}")
            return {}

    def _resample_data_vectorized(self, data: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        """Resample data to specified timeframe using vectorized operations."""
        try:
            # Convert timeframe string to pandas offset
            timeframe_map = {
                "1m": "1T",
                "5m": "5T",
                "15m": "15T",
                "30m": "30T",
            }
            
            offset = timeframe_map.get(timeframe, "1T")
            resampled = data.resample(offset).agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum',
            })
            
            return resampled.dropna()

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

            # Basic price features
            features["price_change"] = price_data["close"].pct_change().iloc[-1]
            features["price_volatility"] = price_data["close"].pct_change().rolling(window=20).std().iloc[-1]
            
            # Volume features
            features["volume_change"] = volume_data["volume"].pct_change().iloc[-1]
            features["volume_ma_ratio"] = volume_data["volume"].iloc[-1] / volume_data["volume"].rolling(window=20).mean().iloc[-1]

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
            features = {}

            # Meta-labeling based on volatility regime
            volatility = price_data["close"].pct_change().rolling(window=20).std()
            features["volatility_regime"] = 1 if volatility.iloc[-1] > volatility.quantile(0.75) else 0

            # Meta-labeling based on volume regime
            volume_ma = volume_data["volume"].rolling(window=20).mean()
            features["volume_regime"] = 1 if volume_data["volume"].iloc[-1] > volume_ma.iloc[-1] else 0

            # Meta-labeling based on trend regime
            sma_short = price_data["close"].rolling(window=10).mean()
            sma_long = price_data["close"].rolling(window=30).mean()
            features["trend_regime"] = 1 if sma_short.iloc[-1] > sma_long.iloc[-1] else 0

            return features

        except Exception as e:
            self.logger.error(f"Error generating meta labels: {e}")
            return {}


# Placeholder classes for other analyzers
class VectorizedVolatilityRegimeModel:
    """Placeholder for volatility regime modeling."""
    
    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.logger = system_logger.getChild("VectorizedVolatilityRegimeModel")
    
    async def initialize(self) -> bool:
        return True
    
    async def model_volatility_vectorized(self, price_data: pd.DataFrame) -> dict[str, Any]:
        return {"volatility_regime": 0.5}


class VectorizedCorrelationAnalyzer:
    """Placeholder for correlation analysis."""
    
    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.logger = system_logger.getChild("VectorizedCorrelationAnalyzer")
    
    async def initialize(self) -> bool:
        return True
    
    async def analyze_correlations_vectorized(self, price_data: pd.DataFrame) -> dict[str, Any]:
        return {"correlation_strength": 0.3}


class VectorizedMomentumAnalyzer:
    """Placeholder for momentum analysis."""
    
    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.logger = system_logger.getChild("VectorizedMomentumAnalyzer")
    
    async def initialize(self) -> bool:
        return True
    
    async def analyze_momentum_vectorized(self, price_data: pd.DataFrame) -> dict[str, Any]:
        return {"momentum_strength": 0.4}


class VectorizedLiquidityAnalyzer:
    """Placeholder for liquidity analysis."""
    
    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.logger = system_logger.getChild("VectorizedLiquidityAnalyzer")
    
    async def initialize(self) -> bool:
        return True
    
    async def analyze_liquidity_vectorized(
        self,
        price_data: pd.DataFrame,
        volume_data: pd.DataFrame,
        order_flow_data: pd.DataFrame | None = None,
    ) -> dict[str, Any]:
        return {"liquidity_score": 0.6}


class VectorizedSRDistanceCalculator:
    """Placeholder for S/R distance calculation."""
    
    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.logger = system_logger.getChild("VectorizedSRDistanceCalculator")
    
    async def initialize(self) -> bool:
        return True
    
    async def calculate_sr_distances(
        self,
        price_data: pd.DataFrame,
        sr_levels: dict[str, Any],
    ) -> dict[str, Any]:
        return {"nearest_support_distance": 0.02, "nearest_resistance_distance": 0.03}


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
                current_is_bullish &
                ~previous_is_bullish &
                (current_open < previous_close) &
                (current_close > previous_open) &
                (current_body_size > previous_body_size * self.engulfing_ratio)
            )

            # Bearish engulfing conditions
            bearish_engulfing = (
                ~current_is_bullish &
                previous_is_bullish &
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

            # Calculate pattern type features (count and presence)
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
                    # Count patterns
                    pattern_count = sum(
                        np.sum(pattern_data.get(key, np.zeros(len(df), dtype=bool)))
                        for key in pattern_data.keys()
                        if isinstance(pattern_data[key], np.ndarray) and pattern_data[key].dtype == bool
                    )
                    features[f"{pattern_type}_count"] = pattern_count
                    features[f"{pattern_type}_present"] = 1.0 if pattern_count > 0 else 0.0

            # Calculate specific pattern features
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
                pattern_count = 0
                for pattern_data in patterns.values():
                    if pattern in pattern_data:
                        pattern_count += np.sum(pattern_data[pattern])
                
                features[f"{pattern}_count"] = pattern_count
                features[f"{pattern}_present"] = 1.0 if pattern_count > 0 else 0.0

            # Calculate pattern density features
            total_patterns = sum(
                features.get(f"{pt}_count", 0) for pt in pattern_types
            )
            features["total_patterns"] = total_patterns
            features["pattern_density"] = total_patterns / len(df) if len(df) > 0 else 0.0

            # Calculate bullish vs bearish pattern features
            bullish_patterns = sum(
                features.get(f"{pattern}_count", 0)
                for pattern in ["bullish_engulfing", "hammer", "inverted_hammer", "tweezer_bottom", "bullish_marubozu", "rising_three_methods"]
            )
            bearish_patterns = sum(
                features.get(f"{pattern}_count", 0)
                for pattern in ["bearish_engulfing", "hanging_man", "shooting_star", "tweezer_top", "bearish_marubozu", "falling_three_methods"]
            )

            features["bullish_patterns"] = bullish_patterns
            features["bearish_patterns"] = bearish_patterns
            features["bullish_bearish_ratio"] = bullish_patterns / (bearish_patterns + 1e-8)

            # Calculate recent pattern features (last 5 candles)
            recent_patterns = 0
            recent_bullish_patterns = 0
            recent_bearish_patterns = 0

            for pattern_data in patterns.values():
                for key, pattern_array in pattern_data.items():
                    if isinstance(pattern_array, np.ndarray) and pattern_array.dtype == bool:
                        if len(pattern_array) >= 5:
                            recent_count = np.sum(pattern_array[-5:])
                            recent_patterns += recent_count
                            
                            if any(bullish in key for bullish in ["bullish", "hammer", "inverted_hammer", "tweezer_bottom", "rising"]):
                                recent_bullish_patterns += recent_count
                            elif any(bearish in key for bearish in ["bearish", "hanging_man", "shooting_star", "tweezer_top", "falling"]):
                                recent_bearish_patterns += recent_count

            features["recent_patterns_count"] = recent_patterns
            features["recent_bullish_patterns"] = recent_bullish_patterns
            features["recent_bearish_patterns"] = recent_bearish_patterns

            # Calculate pattern confidence features
            all_confidences = []
            for pattern_data in patterns.values():
                for key, confidence_array in pattern_data.items():
                    if "confidence" in key and isinstance(confidence_array, np.ndarray):
                        all_confidences.extend(confidence_array[confidence_array > 0])

            if all_confidences:
                features["avg_pattern_confidence"] = np.mean(all_confidences)
                features["max_pattern_confidence"] = np.max(all_confidences)
                features["pattern_confidence_std"] = np.std(all_confidences)
            else:
                features["avg_pattern_confidence"] = 0.0
                features["max_pattern_confidence"] = 0.0
                features["pattern_confidence_std"] = 0.0

            return features

        except Exception as e:
            self.logger.error(f"Error converting patterns to features: {e}")
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
        self.max_features_per_wavelet = self.wavelet_config.get("max_features_per_wavelet", 20)
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

        Args:
            price_data: OHLCV price data
            volume_data: Volume data (optional)

        Returns:
            Dictionary containing wavelet transform features
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

            features = {}
            total_features = 0

            # 1. Prepare stationary series for analysis
            stationary_series = self._prepare_stationary_series(price_data)
            
            # 2. Discrete Wavelet Transform (DWT) analysis with boundary handling
            if self.enable_discrete_wavelet:
                dwt_start = time.time()
                dwt_features = await self._analyze_discrete_wavelet_transforms_enhanced(
                    price_data, stationary_series
                )
                features.update(dwt_features)
                self.computation_times["dwt"] = time.time() - dwt_start
                self.feature_counts["dwt"] = len(dwt_features)
                total_features += len(dwt_features)

            # 3. Continuous Wavelet Transform (CWT) analysis with dynamic scale selection
            if self.enable_continuous_wavelet:
                cwt_start = time.time()
                cwt_features = await self._analyze_continuous_wavelet_transforms_enhanced(
                    price_data, stationary_series
                )
                features.update(cwt_features)
                self.computation_times["cwt"] = time.time() - cwt_start
                self.feature_counts["cwt"] = len(cwt_features)
                total_features += len(cwt_features)

            # 4. Wavelet Packet analysis with dimensionality control
            if self.enable_wavelet_packet:
                packet_start = time.time()
                packet_features = await self._analyze_wavelet_packets_enhanced(
                    price_data, stationary_series
                )
                features.update(packet_features)
                self.computation_times["packet"] = time.time() - packet_start
                self.feature_counts["packet"] = len(packet_features)
                total_features += len(packet_features)

            # 5. Wavelet denoising with boundary effect management
            if self.enable_denoising:
                denoising_start = time.time()
                denoising_features = await self._analyze_wavelet_denoising_enhanced(
                    price_data, stationary_series
                )
                features.update(denoising_features)
                self.computation_times["denoising"] = time.time() - denoising_start
                self.feature_counts["denoising"] = len(denoising_features)
                total_features += len(denoising_features)

            # 6. Multi-wavelet analysis with feature selection
            multi_start = time.time()
            multi_wavelet_features = await self._analyze_multi_wavelet_transforms_enhanced(
                price_data, stationary_series
            )
            features.update(multi_wavelet_features)
            self.computation_times["multi_wavelet"] = time.time() - multi_start
            self.feature_counts["multi_wavelet"] = len(multi_wavelet_features)
            total_features += len(multi_wavelet_features)

            # 7. Volume wavelet analysis (if available)
            if volume_data is not None and not volume_data.empty:
                volume_start = time.time()
                volume_wavelet_features = await self._analyze_volume_wavelet_transforms_enhanced(
                    volume_data
                )
                features.update(volume_wavelet_features)
                self.computation_times["volume_wavelet"] = time.time() - volume_start
                self.feature_counts["volume_wavelet"] = len(volume_wavelet_features)
                total_features += len(volume_wavelet_features)

            # 8. Feature selection and dimensionality reduction
            selected_features = self._select_optimal_wavelet_features(features)
            
            total_time = time.time() - start_time
            self.computation_times["total"] = total_time
            self.feature_counts["total"] = len(selected_features)

            self.logger.info(
                f"✅ Enhanced wavelet transform analysis completed in {total_time:.2f}s. "
                f"Generated {len(selected_features)} features from {total_features} candidates. "
                f"Feature selection: {self.feature_selection_method}"
            )
            
            # Log performance metrics
            self._log_performance_metrics()
            
            return selected_features

        except Exception as e:
            self.logger.error(f"Error in enhanced wavelet transform analysis: {e}")
            return {}

    def _prepare_stationary_series(self, price_data: pd.DataFrame) -> dict[str, np.ndarray]:
        """Prepare stationary series for wavelet analysis."""
        try:
            stationary_series = {}
            
            if self.enable_stationary_series:
                # Returns (first difference)
                returns = price_data["close"].pct_change().dropna().values
                if len(returns) > 0:
                    stationary_series["returns"] = returns
                
                # Log returns
                log_returns = np.log(price_data["close"] / price_data["close"].shift(1)).dropna().values
                if len(log_returns) > 0:
                    stationary_series["log_returns"] = log_returns
                
                # Detrended series
                close_prices = price_data["close"].values
                trend = np.polyfit(range(len(close_prices)), close_prices, 1)
                detrended = close_prices - (trend[0] * np.arange(len(close_prices)) + trend[1])
                stationary_series["detrended"] = detrended
            
            # Original series
            stationary_series["close"] = price_data["close"].values
            
            return stationary_series

        except Exception as e:
            self.logger.error(f"Error preparing stationary series: {e}")
            return {"close": price_data["close"].values}

    async def _analyze_discrete_wavelet_transforms_enhanced(
        self, 
        price_data: pd.DataFrame, 
        stationary_series: dict[str, np.ndarray]
    ) -> dict[str, float]:
        """Analyze discrete wavelet transforms with enhanced boundary handling."""
        try:
            features = {}
            
            # Use limited wavelet types for efficiency
            wavelet_types = self.wavelet_types[:self.max_wavelet_types]
            
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
        """Enhanced wavelet packet analysis."""
        return {}

    async def _analyze_wavelet_denoising_enhanced(
        self, 
        price_data: pd.DataFrame, 
        stationary_series: dict[str, np.ndarray]
    ) -> dict[str, float]:
        """Enhanced wavelet denoising analysis."""
        return {}

    async def _analyze_multi_wavelet_transforms_enhanced(
        self, 
        price_data: pd.DataFrame, 
        stationary_series: dict[str, np.ndarray]
    ) -> dict[str, float]:
        """Enhanced multi-wavelet transform analysis."""
        return {}

    async def _analyze_volume_wavelet_transforms_enhanced(
        self, 
        volume_data: pd.DataFrame
    ) -> dict[str, float]:
        """Enhanced volume wavelet transform analysis."""
        return {}