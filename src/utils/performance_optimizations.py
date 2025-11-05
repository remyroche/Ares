"""
Performance Optimizations for Trading System

This module provides vectorized computations, memory-efficient processing,
and lazy loading optimizations to address performance bottlenecks.
"""

import logging
import gc
import numpy as np
from src.utils.tprint import tprint_warning
import pandas as pd
from typing import List, Dict
from functools import lru_cache
from contextlib import contextmanager
import psutil

logger = logging.getLogger(__name__)

# Vectorized Feature Generation System
class VectorizedFeatureGenerator:
    """Vectorized feature generation for improved performance."""

    @staticmethod
    def vectorized_rolling_features(data, windows: List[int]) -> pd.DataFrame:
            
        results = {}

        for window in windows:
            rolling_data = data.rolling(window=window, min_periods=1)
            results[f'rolling_mean_{window}'] = rolling_data.mean()
            results[f'rolling_std_{window}'] = rolling_data.std()
            results[f'rolling_skew_{window}'] = rolling_data.skew()
            results[f'rolling_kurt_{window}'] = rolling_data.kurt()

        return pd.concat(results, axis=1)

    @staticmethod
    def vectorized_technical_indicators(prices, volumes) -> Dict[str, np.ndarray]:
        """Compute technical indicators using vectorized operations."""
        # Convert to numpy arrays if needed
        if hasattr(prices, 'values'):
            prices_array = np.asarray(prices.values)
        else:
            prices_array = np.asarray(prices)
            
        if hasattr(volumes, 'values'):
            volumes_array = np.asarray(volumes.values)
        else:
            volumes_array = np.asarray(volumes)
            
        prices_arr = prices_array
        volumes_arr = volumes_array
        
        returns = np.diff(prices_arr) / (prices_arr[:-1] + 1e-8) * 100
        
        # Basic indicators
        features = {
            'returns': returns,
            'price_change': np.diff(prices_arr),
            'volume_change': np.diff(volumes_arr),
            'price_momentum': prices_arr[1:] - prices_arr[:-1],
            'volume_ratio': volumes_arr[1:] / (volumes_arr[:-1] + 1e-8),  # Avoid division by zero
            'volatility': np.std(returns),
            'price_trend': np.gradient(prices_arr),
            'volume_trend': np.gradient(volumes_arr)
        }
        
        # RSI calculation - fully vectorized
        def rsi_vectorized(price_changes, period=14):
            gains = np.where(price_changes > 0, price_changes, 0)
            losses = np.where(price_changes < 0, -price_changes, 0)

            avg_gains = pd.Series(gains).ewm(span=period).mean().values
            avg_losses = pd.Series(losses).ewm(span=period).mean().values

            rs = avg_gains / (avg_losses + 1e-10)
            return 100 - (100 / (1 + rs))

        # MACD - single vectorized computation
        ema12 = pd.Series(prices_arr).ewm(span=12).mean().values
        ema26 = pd.Series(prices_arr).ewm(span=26).mean().values
        macd = ema12 - ema26
        signal = pd.Series(macd).ewm(span=9).mean().values

        # Bollinger Bands
        sma20 = pd.Series(prices_arr).rolling(20).mean().values
        std20 = pd.Series(prices_arr).rolling(20).std().values
        bb_upper = sma20 + (std20 * 2)
        bb_lower = sma20 - (std20 * 2)

        # Volume indicators
        volume_sma = pd.Series(volumes_arr).rolling(20).mean().values
        volume_ratio = volumes_arr / (volume_sma + 1e-10)

        features.update({
            'rsi_14': rsi_vectorized(np.diff(prices_arr, prepend=prices_arr[:1])),
            'macd': macd,
            'macd_signal': signal,
            'macd_histogram': macd - signal,
            'bb_upper': bb_upper,
            'bb_middle': sma20,
            'bb_lower': bb_lower,
            'volume_ratio': volume_ratio,
            'volume_sma': volume_sma
        })

        return features

# Memory-Efficient Processing
@contextmanager
def memory_efficient_processing(max_memory_mb: int = 1024):
    """Context manager for memory-efficient processing."""
    process = psutil.Process()
    initial_memory = process.memory_info().rss / 1024 / 1024

    try:
        yield
    finally:
        # Force cleanup
        gc.collect()

        # Check memory usage
        current_memory = process.memory_info().rss / 1024 / 1024
        memory_used = current_memory - initial_memory

        if memory_used > max_memory_mb:
            # Aggressive cleanup if over limit
            gc.collect(2)  # Full collection
            # Clear any cached data
            if hasattr(pd, '_cache'):
                pd._cache.clear()

def chunked_processing(data: pd.DataFrame, chunk_size: int = 1000):
    """Process data in memory-efficient chunks."""
    results = []

    for i in range(0, len(data), chunk_size):
        chunk = data.iloc[i:i+chunk_size].copy()

        # Process chunk
        with memory_efficient_processing():
            processed_chunk = process_single_chunk(chunk)
            results.append(processed_chunk)

        # Clear chunk from memory
        del chunk
        gc.collect()

    # Efficient concatenation
    return pd.concat(results, ignore_index=True, copy=False)

def streaming_feature_generation(data_stream, batch_size: int = 10000):
    """Generate features using streaming approach to minimize memory."""
    feature_accumulator = []

    for batch in data_stream:
        # Process batch
        batch_features = generate_features_batch(batch)

        # Accumulate results efficiently
        feature_accumulator.append(batch_features)

        # Periodic cleanup
        if len(feature_accumulator) % 10 == 0:
            # Merge accumulated features
            merged = pd.concat(feature_accumulator, axis=0, copy=False)
            feature_accumulator = [merged]
            gc.collect()

    return pd.concat(feature_accumulator, axis=0, copy=False)

# Lazy Loading System
class LazyFeatureRegistry:
    """Lazy loading registry for feature generators."""

    def __init__(self):
        self._generators = {}  # Weak references to avoid memory leaks
        self._loaded_categories = set()
        self._category_dependencies = {
            'basic': [],  # No dependencies
            'technical': ['basic'],
            'volatility': ['basic', 'technical'],
            'momentum': ['basic', 'technical'],
            'volume': ['basic'],
            'regime': ['technical', 'volatility', 'momentum'],
            'advanced': ['regime', 'volatility']
        }

    @lru_cache(maxsize=128)
    def get_generator(self, category: str, generator_name: str | None = None):
        """Lazy load specific generator."""
        if category not in self._loaded_categories:
            self._load_category(category)

        if generator_name:
            return self._generators[category].get(generator_name)
        return self._generators[category]

    def _load_category(self, category: str):
        """Load category and its dependencies on demand."""
        # Load dependencies first
        for dep in self._category_dependencies.get(category, []):
            if dep not in self._loaded_categories:
                self._load_category(dep)

        # Lazy import and instantiation
        if category == 'basic':
            # Basic OHLCV features
            self._generators[category] = {}
        elif category == 'technical':
            try:
                from ..feature_generation.categories.technical_indicators import RSIGenerator, MACDGenerator
                self._generators[category] = {
                    'rsi': RSIGenerator(),
                    'macd': MACDGenerator()
                }
            except ImportError as e:
                tprint_warning(f"⚠️ Could not import technical indicators: {e}")
                self._generators[category] = {}
        elif category == 'volatility':
            try:
                from ..feature_generation.categories.volatility_indicators import BollingerBandsGenerator, ATRGenerator
                self._generators[category] = {
                    'bollinger': BollingerBandsGenerator(),
                    'atr': ATRGenerator()
                }
            except ImportError as e:
                tprint_warning(f"⚠️ Could not import volatility indicators: {e}")
                self._generators[category] = {}
        elif category == 'momentum':
            try:
                from ..feature_generation.categories.momentum import RSIGenerator, MACDGenerator
                self._generators[category] = {
                    'rsi': RSIGenerator(),
                    'macd': MACDGenerator()
                }
            except ImportError as e:
                tprint_warning(f"⚠️ Could not import momentum indicators: {e}")
                self._generators[category] = {}
        elif category == 'volume':
            try:
                # Volume features - using basic implementations for now
                self._generators[category] = {
                    'volume_profile': None,  # Placeholder
                    'vwap': None  # Placeholder
                }
            except ImportError as e:
                tprint_warning(f"⚠️ Could not import volume indicators: {e}")
                self._generators[category] = {}
        elif category == 'regime':
            try:
                from ..feature_generation.categories.regime_features import RegimeFeatureGenerator
                self._generators[category] = {
                    'regime_context': RegimeFeatureGenerator()
                }
            except ImportError as e:
                tprint_warning(f"⚠️ Could not import regime features: {e}")
                self._generators[category] = {}
        elif category == 'advanced':
            # Advanced features combine multiple categories
            self._generators[category] = {}
        else:
            # Unknown category - create empty placeholder
            tprint_warning(f"⚠️ Unknown feature category: {category}")
            self._generators[category] = {}

        self._loaded_categories.add(category)

    def preload_common_categories(self):
        """Preload only commonly used categories."""
        common_categories = ['basic', 'technical', 'volatility']
        for category in common_categories:
            self._load_category(category)

class FeatureGeneratorFactory:
    """Factory with lazy loading and caching."""

    def __init__(self):
        self._registry = LazyFeatureRegistry()
        self._cache = {}  # LRU cache for generated features

    def generate_feature(self, feature_name: str, data: pd.DataFrame, **params):
        """Generate feature with caching."""
        # Use data shape and hash instead of DataFrame itself for cache key
        data_hash = hash(data.values.tobytes()) if hasattr(data, 'values') else hash(str(data))
        cache_key = self._get_cache_key(feature_name, data_hash, params)

        if cache_key in self._cache:
            return self._cache[cache_key]

        # Lazy load generator
        category = self._get_category_for_feature(feature_name)
        generator = self._registry.get_generator(category, feature_name)

        # Generate feature
        result = generator.generate(data, **params)

        # Cache result (with size limit)
        if len(self._cache) < 100:  # Limit cache size
            self._cache[cache_key] = result

        return result

    def _get_cache_key(self, feature_name: str, data_hash: int, params: dict) -> str:
        """Generate cache key."""
        param_str = str(sorted(params.items()))
        return f"{feature_name}_{data_hash}_{param_str}"

    def _get_category_for_feature(self, feature_name: str) -> str:
        """Map feature name to category."""
        if feature_name.startswith('rsi') or feature_name.startswith('macd'):
            return 'technical'
        elif 'volatility' in feature_name or 'atr' in feature_name or 'bb_' in feature_name:
            return 'volatility'
        elif 'momentum' in feature_name or 'roc' in feature_name:
            return 'momentum'
        elif 'volume' in feature_name:
            return 'volume'
        elif 'regime' in feature_name:
            return 'regime'
        elif feature_name == 'basic':
            # Handle 'basic' feature type
            return 'basic'
        else:
            return 'technical'  # Default fallback

class OptimizedFeaturePipeline:
    """Memory-efficient, vectorized feature generation pipeline."""

    def __init__(self, chunk_size: int = 5000, max_memory_mb: int = 2048):
        self.chunk_size = chunk_size
        self.max_memory = max_memory_mb
        self.lazy_factory = FeatureGeneratorFactory()
        self.vectorized_generator = VectorizedFeatureGenerator()

    def generate_all_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate all features with optimizations."""

        # Phase 1: Vectorized basic features
        with memory_efficient_processing(self.max_memory):
            basic_features = self.generate_basic_features_vectorized(data)

        # Phase 2: Lazy-loaded advanced features in chunks
        advanced_features = []
        for i in range(0, len(data), self.chunk_size):
            chunk = data.iloc[i:i+self.chunk_size]

            with memory_efficient_processing(self.max_memory):
                chunk_features = self.generate_advanced_features_lazy(chunk)
                advanced_features.append(chunk_features)

            # Cleanup
            del chunk
            gc.collect()

        # Phase 3: Efficient concatenation
        all_features = pd.concat([basic_features] + advanced_features, axis=1, copy=False)

        # Final cleanup
        gc.collect()
        return self._optimize_dtypes(all_features)

    def generate_basic_features_vectorized(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate basic features using vectorized operations."""
        # All basic calculations in single vectorized operations
        prices = data['close'].values
        volumes = data['volume'].values

        # Vectorized technical indicators
        indicators = self.vectorized_generator.vectorized_technical_indicators(prices, volumes)

        # Vectorized rolling features for multiple windows
        windows = [5, 10, 20, 50]
        rolling_features = self.vectorized_generator.vectorized_rolling_features(
            data[['close', 'volume']], windows
        )

        # Combine efficiently
        return pd.concat([
            pd.DataFrame(indicators, index=data.index),
            rolling_features
        ], axis=1)

    def generate_advanced_features_lazy(self, chunk: pd.DataFrame) -> pd.DataFrame:
        """Generate advanced features using lazy loading."""
        features = {}

        # Only load generators as needed
        regime_features = self.lazy_factory.generate_feature('regime_context', chunk)
        features.update(regime_features)

        # Generate other features on demand
        if self._should_generate_volume_features(chunk):
            volume_features = self.lazy_factory.generate_feature('volume_profile', chunk)
            features.update(volume_features)

        return pd.DataFrame(features, index=chunk.index)

    def _should_generate_volume_features(self, data) -> bool:
        """Check if should generate volume features."""
        return bool('volume' in data.columns if hasattr(data, 'columns') else False) and data['volume'].notna().any()

    def _optimize_dtypes(self, df: pd.DataFrame) -> pd.DataFrame:
        """Optimize DataFrame dtypes for memory efficiency."""
        for col in df.columns:
            if df[col].dtype == 'float64':
                # Check if float32 is sufficient
                if (df[col].min() > np.finfo(np.float32).min and
                    df[col].max() < np.finfo(np.float32).max):
                    df[col] = df[col].astype('float32')
            elif df[col].dtype == 'int64':
                # Check if smaller int type is sufficient
                if (df[col].min() >= np.iinfo(np.int32).min and
                    df[col].max() <= np.iinfo(np.int32).max):
                    df[col] = df[col].astype('int32')

        return df

# Global instances
_optimized_pipeline = None

def get_optimized_feature_pipeline(chunk_size: int = 5000, max_memory_mb: int = 2048):
    """Get optimized feature pipeline instance."""
    global _optimized_pipeline
    if _optimized_pipeline is None:
        _optimized_pipeline = OptimizedFeaturePipeline(chunk_size, max_memory_mb)
    return _optimized_pipeline

# Utility functions
def process_single_chunk(chunk: pd.DataFrame) -> pd.DataFrame:
    """Process a single chunk of data."""
    # Placeholder - implement specific processing logic
    return chunk

def generate_features_batch(batch: pd.DataFrame) -> pd.DataFrame:
    """Generate features for a batch of data."""
    pipeline = get_optimized_feature_pipeline()
    return pipeline._generate_basic_features_vectorized(batch)