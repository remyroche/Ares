"""
VectorBT-Optimized Feature Generation for Multiple Lookback Periods.

This module provides efficient feature generation across multiple lookback periods
using VectorBT's optimized rolling operations and indicators, replacing the
sequential feature generation with parallel batch processing.
"""

import logging
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache

# VectorBT imports
try:
    import vectorbt as vbt
    from vectorbt.indicators.basic import RSI, MA, BBANDS, STOCH
    from vectorbt.indicators.momentum import MACD, ADX, CCI
    from vectorbt.indicators.volatility import ATR, BollingerBands
    from vectorbt.generic import nb
    VECTORBT_AVAILABLE = True
except ImportError:
    VECTORBT_AVAILABLE = False
    vbt = None
    RSI = None
    MA = None
    BBANDS = None
    STOCH = None
    MACD = None
    ADX = None
    CCI = None
    ATR = None
    BollingerBands = None
    nb = None

from src.utils.tprint import tprint, tprint_error, tprint_success, tprint_warning, tprint_debug, tprint_info
from src.utils.logger import get_logger
from .utils.error_handling import safe_operation, get_error_handler

logger = get_logger('VectorBTFeatureGeneration')


class FeatureType(Enum):
    """Available feature types for generation."""
    SMA = "sma"
    EMA = "ema"
    RSI = "rsi"
    MACD = "macd"
    BBANDS = "bbands"
    STOCH = "stoch"
    ADX = "adx"
    CCI = "cci"
    ATR = "atr"
    BOLLINGER = "bollinger"
    CUSTOM = "custom"


@dataclass
class VectorBTFeatureConfig:
    """Configuration for VectorBT feature generation."""
    use_parallel: bool = True
    max_workers: int = 4
    batch_size: int = 50
    memory_efficient: bool = True
    cache_features: bool = True
    cache_size: int = 1000
    min_data_points: int = 20
    fill_method: str = 'forward'  # 'forward', 'backward', 'interpolate', 'drop'
    parallel_processing: bool = True


class VectorBTFeatureGenerator:
    """
    High-performance feature generator using VectorBT indicators.
    
    This class provides efficient generation of technical indicators across
    multiple lookback periods using VectorBT's optimized rolling operations.
    """
    
    def __init__(self, config: Optional[VectorBTFeatureConfig] = None):
        """Initialize VectorBT feature generator."""
        if not VECTORBT_AVAILABLE:
            raise ImportError("VectorBT is required but not available. Install with: pip install vectorbt")
        
        self.config = config or VectorBTFeatureConfig()
        self.logger = get_logger('VectorBTFeatureGenerator')
        self.error_handler = get_error_handler()
        
        # Configure VectorBT settings
        self._configure_vectorbt()
        
        # Initialize feature cache
        self._feature_cache = {}
        self._cache_hits = 0
        self._cache_misses = 0
        
        tprint_success("✅ VectorBT Feature Generator initialized")
    
    def _configure_vectorbt(self):
        """Configure VectorBT for optimal feature generation."""
        try:
            # Configure VectorBT settings
            vbt.settings.set_theme('dark')
            vbt.settings['array_wrapper']['freq_precision'] = 0
            vbt.settings['array_wrapper']['freq_shorten'] = True
            
            if self.config.parallel_processing:
                vbt.settings['array_wrapper']['parallel'] = True
            
            self.logger.debug("VectorBT feature generation configuration applied")
            
        except Exception as e:
            self.logger.warning(f"Could not configure VectorBT settings: {e}")
    
    @safe_operation
    def generate_features_vectorbt(
        self,
        data: pd.DataFrame,
        feature_name: str,
        lookback_periods: List[int],
        feature_type: Optional[FeatureType] = None
    ) -> Dict[int, np.ndarray]:
        """
        Generate features for multiple lookback periods using VectorBT.
        
        Args:
            data: Input data with OHLCV columns
            feature_name: Name of the feature to generate
            lookback_periods: List of lookback periods
            feature_type: Type of feature to generate
            
        Returns:
            Dictionary mapping lookback periods to feature arrays
        """
        tprint_debug(f"🔄 Generating {feature_name} for {len(lookback_periods)} lookback periods")
        start_time = time.time()
        
        try:
            # Determine feature type
            if feature_type is None:
                feature_type = self._infer_feature_type(feature_name)
            
            # Check cache first
            cache_key = self._get_cache_key(data, feature_name, lookback_periods)
            if self.config.cache_features and cache_key in self._feature_cache:
                self._cache_hits += 1
                tprint_debug("📦 Using cached features")
                return self._feature_cache[cache_key]
            
            self._cache_misses += 1
            
            # Generate features
            if self.config.use_parallel and len(lookback_periods) > 1:
                features = self._generate_features_parallel(data, feature_name, lookback_periods, feature_type)
            else:
                features = self._generate_features_sequential(data, feature_name, lookback_periods, feature_type)
            
            # Cache results
            if self.config.cache_features:
                self._feature_cache[cache_key] = features
                self._trim_cache()
            
            computation_time = time.time() - start_time
            tprint_success(f"✅ Generated {len(features)} features in {computation_time:.3f}s")
            
            return features
            
        except Exception as e:
            self.logger.error(f"VectorBT feature generation failed: {e}")
            return {}
    
    def _infer_feature_type(self, feature_name: str) -> FeatureType:
        """Infer feature type from feature name."""
        feature_name_lower = feature_name.lower()
        
        if 'sma' in feature_name_lower or 'simple' in feature_name_lower:
            return FeatureType.SMA
        elif 'ema' in feature_name_lower or 'exponential' in feature_name_lower:
            return FeatureType.EMA
        elif 'rsi' in feature_name_lower:
            return FeatureType.RSI
        elif 'macd' in feature_name_lower:
            return FeatureType.MACD
        elif 'bbands' in feature_name_lower or 'bollinger' in feature_name_lower:
            return FeatureType.BBANDS
        elif 'stoch' in feature_name_lower or 'stochastic' in feature_name_lower:
            return FeatureType.STOCH
        elif 'adx' in feature_name_lower:
            return FeatureType.ADX
        elif 'cci' in feature_name_lower:
            return FeatureType.CCI
        elif 'atr' in feature_name_lower:
            return FeatureType.ATR
        else:
            return FeatureType.CUSTOM
    
    def _generate_features_parallel(
        self,
        data: pd.DataFrame,
        feature_name: str,
        lookback_periods: List[int],
        feature_type: FeatureType
    ) -> Dict[int, np.ndarray]:
        """Generate features in parallel using ThreadPoolExecutor."""
        features = {}
        
        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            # Submit tasks
            future_to_period = {
                executor.submit(
                    self._generate_single_feature,
                    data, feature_name, period, feature_type
                ): period
                for period in lookback_periods
            }
            
            # Collect results
            for future in as_completed(future_to_period):
                period = future_to_period[future]
                try:
                    feature_values = future.result()
                    if feature_values is not None and len(feature_values) > 0:
                        features[period] = feature_values
                except Exception as e:
                    self.logger.warning(f"Feature generation failed for period {period}: {e}")
        
        return features
    
    def _generate_features_sequential(
        self,
        data: pd.DataFrame,
        feature_name: str,
        lookback_periods: List[int],
        feature_type: FeatureType
    ) -> Dict[int, np.ndarray]:
        """Generate features sequentially."""
        features = {}
        
        for period in lookback_periods:
            try:
                feature_values = self._generate_single_feature(data, feature_name, period, feature_type)
                if feature_values is not None and len(feature_values) > 0:
                    features[period] = feature_values
            except Exception as e:
                self.logger.warning(f"Feature generation failed for period {period}: {e}")
        
        return features
    
    def _generate_single_feature(
        self,
        data: pd.DataFrame,
        feature_name: str,
        lookback_period: int,
        feature_type: FeatureType
    ) -> Optional[np.ndarray]:
        """Generate a single feature for a specific lookback period."""
        try:
            # Validate inputs
            if data is None or len(data) == 0:
                return None
            
            if lookback_period < self.config.min_data_points:
                return None
            
            # Get price data
            close_prices = self._get_price_data(data, 'close')
            if close_prices is None or len(close_prices) == 0:
                return None
            
            # Generate feature based on type
            if feature_type == FeatureType.SMA:
                return self._generate_sma(close_prices, lookback_period)
            elif feature_type == FeatureType.EMA:
                return self._generate_ema(close_prices, lookback_period)
            elif feature_type == FeatureType.RSI:
                return self._generate_rsi(close_prices, lookback_period)
            elif feature_type == FeatureType.MACD:
                return self._generate_macd(close_prices, lookback_period)
            elif feature_type == FeatureType.BBANDS:
                return self._generate_bbands(close_prices, lookback_period)
            elif feature_type == FeatureType.STOCH:
                return self._generate_stoch(data, lookback_period)
            elif feature_type == FeatureType.ADX:
                return self._generate_adx(data, lookback_period)
            elif feature_type == FeatureType.CCI:
                return self._generate_cci(data, lookback_period)
            elif feature_type == FeatureType.ATR:
                return self._generate_atr(data, lookback_period)
            elif feature_type == FeatureType.BOLLINGER:
                return self._generate_bollinger(close_prices, lookback_period)
            else:  # CUSTOM
                return self._generate_custom_feature(data, feature_name, lookback_period)
            
        except Exception as e:
            self.logger.warning(f"Single feature generation failed: {e}")
            return None
    
    def _get_price_data(self, data: pd.DataFrame, column: str) -> Optional[np.ndarray]:
        """Get price data from DataFrame."""
        try:
            if column in data.columns:
                return data[column].values
            elif column.upper() in data.columns:
                return data[column.upper()].values
            else:
                return None
        except Exception:
            return None
    
    def _generate_sma(self, prices: np.ndarray, window: int) -> np.ndarray:
        """Generate Simple Moving Average using VectorBT."""
        try:
            ma = MA.run(prices, window=window)
            return ma.ma.values
        except Exception as e:
            self.logger.warning(f"SMA generation failed: {e}")
            return np.array([])
    
    def _generate_ema(self, prices: np.ndarray, window: int) -> np.ndarray:
        """Generate Exponential Moving Average using VectorBT."""
        try:
            # VectorBT doesn't have direct EMA, so we'll use pandas
            ema = pd.Series(prices).ewm(span=window).mean()
            return ema.values
        except Exception as e:
            self.logger.warning(f"EMA generation failed: {e}")
            return np.array([])
    
    def _generate_rsi(self, prices: np.ndarray, window: int) -> np.ndarray:
        """Generate RSI using VectorBT."""
        try:
            rsi = RSI.run(prices, window=window)
            return rsi.rsi.values
        except Exception as e:
            self.logger.warning(f"RSI generation failed: {e}")
            return np.array([])
    
    def _generate_macd(self, prices: np.ndarray, window: int) -> np.ndarray:
        """Generate MACD using VectorBT."""
        try:
            macd = MACD.run(prices, fast_window=window, slow_window=window*2, signal_window=window//2)
            return macd.macd.values
        except Exception as e:
            self.logger.warning(f"MACD generation failed: {e}")
            return np.array([])
    
    def _generate_bbands(self, prices: np.ndarray, window: int) -> np.ndarray:
        """Generate Bollinger Bands using VectorBT."""
        try:
            bb = BBANDS.run(prices, window=window)
            # Return the middle band (SMA)
            return bb.middle.values
        except Exception as e:
            self.logger.warning(f"Bollinger Bands generation failed: {e}")
            return np.array([])
    
    def _generate_stoch(self, data: pd.DataFrame, window: int) -> np.ndarray:
        """Generate Stochastic Oscillator using VectorBT."""
        try:
            high = self._get_price_data(data, 'high')
            low = self._get_price_data(data, 'low')
            close = self._get_price_data(data, 'close')
            
            if high is None or low is None or close is None:
                return np.array([])
            
            stoch = STOCH.run(high, low, close, window=window)
            return stoch.percent_k.values
        except Exception as e:
            self.logger.warning(f"Stochastic generation failed: {e}")
            return np.array([])
    
    def _generate_adx(self, data: pd.DataFrame, window: int) -> np.ndarray:
        """Generate ADX using VectorBT."""
        try:
            high = self._get_price_data(data, 'high')
            low = self._get_price_data(data, 'low')
            close = self._get_price_data(data, 'close')
            
            if high is None or low is None or close is None:
                return np.array([])
            
            adx = ADX.run(high, low, close, window=window)
            return adx.adx.values
        except Exception as e:
            self.logger.warning(f"ADX generation failed: {e}")
            return np.array([])
    
    def _generate_cci(self, data: pd.DataFrame, window: int) -> np.ndarray:
        """Generate CCI using VectorBT."""
        try:
            high = self._get_price_data(data, 'high')
            low = self._get_price_data(data, 'low')
            close = self._get_price_data(data, 'close')
            
            if high is None or low is None or close is None:
                return np.array([])
            
            cci = CCI.run(high, low, close, window=window)
            return cci.cci.values
        except Exception as e:
            self.logger.warning(f"CCI generation failed: {e}")
            return np.array([])
    
    def _generate_atr(self, data: pd.DataFrame, window: int) -> np.ndarray:
        """Generate ATR using VectorBT."""
        try:
            high = self._get_price_data(data, 'high')
            low = self._get_price_data(data, 'low')
            close = self._get_price_data(data, 'close')
            
            if high is None or low is None or close is None:
                return np.array([])
            
            atr = ATR.run(high, low, close, window=window)
            return atr.atr.values
        except Exception as e:
            self.logger.warning(f"ATR generation failed: {e}")
            return np.array([])
    
    def _generate_bollinger(self, prices: np.ndarray, window: int) -> np.ndarray:
        """Generate Bollinger Bands width using VectorBT."""
        try:
            bb = BBANDS.run(prices, window=window)
            # Return the width (upper - lower) / middle
            width = (bb.upper - bb.lower) / bb.middle
            return width.values
        except Exception as e:
            self.logger.warning(f"Bollinger width generation failed: {e}")
            return np.array([])
    
    def _generate_custom_feature(self, data: pd.DataFrame, feature_name: str, window: int) -> np.ndarray:
        """Generate custom feature based on feature name."""
        try:
            # This is a placeholder for custom feature generation
            # In practice, you would implement specific custom features here
            close = self._get_price_data(data, 'close')
            if close is None:
                return np.array([])
            
            # Simple custom feature: price change over window
            if 'price_change' in feature_name.lower():
                return np.diff(close, n=window)
            elif 'volatility' in feature_name.lower():
                returns = np.diff(close) / close[:-1]
                return pd.Series(returns).rolling(window=window).std().values
            else:
                # Default to SMA for unknown features
                return self._generate_sma(close, window)
                
        except Exception as e:
            self.logger.warning(f"Custom feature generation failed: {e}")
            return np.array([])
    
    def _get_cache_key(self, data: pd.DataFrame, feature_name: str, lookback_periods: List[int]) -> str:
        """Generate cache key for feature generation."""
        try:
            # Create a hash of the relevant data
            data_hash = hash(str(data.shape) + str(data.index[0]) + str(data.index[-1]))
            periods_hash = hash(tuple(sorted(lookback_periods)))
            return f"{feature_name}_{data_hash}_{periods_hash}"
        except Exception:
            return f"{feature_name}_{len(data)}_{len(lookback_periods)}"
    
    def _trim_cache(self):
        """Trim cache to prevent memory overflow."""
        if len(self._feature_cache) > self.config.cache_size:
            # Remove oldest entries (simple FIFO)
            keys_to_remove = list(self._feature_cache.keys())[:len(self._feature_cache) - self.config.cache_size]
            for key in keys_to_remove:
                del self._feature_cache[key]
    
    def clear_cache(self):
        """Clear feature cache."""
        self._feature_cache.clear()
        tprint_debug("🧹 VectorBT feature cache cleared")
    
    def get_cache_stats(self) -> Dict[str, int]:
        """Get cache statistics."""
        return {
            'cache_size': len(self._feature_cache),
            'cache_hits': self._cache_hits,
            'cache_misses': self._cache_misses,
            'hit_rate': self._cache_hits / max(1, self._cache_hits + self._cache_misses)
        }


# Convenience functions
def create_vectorbt_feature_generator(
    use_parallel: bool = True,
    max_workers: int = 4,
    cache_features: bool = True
) -> VectorBTFeatureGenerator:
    """Create a VectorBT feature generator with specified configuration."""
    config = VectorBTFeatureConfig(
        use_parallel=use_parallel,
        max_workers=max_workers,
        cache_features=cache_features
    )
    return VectorBTFeatureGenerator(config)


def generate_features_vectorbt(
    data: pd.DataFrame,
    feature_name: str,
    lookback_periods: List[int],
    feature_type: Optional[FeatureType] = None
) -> Dict[int, np.ndarray]:
    """Convenience function to generate features using VectorBT."""
    generator = create_vectorbt_feature_generator()
    return generator.generate_features_vectorbt(data, feature_name, lookback_periods, feature_type)


# Test function
def test_vectorbt_feature_generation():
    """Test VectorBT feature generation."""
    if not VECTORBT_AVAILABLE:
        tprint_error("❌ VectorBT not available for testing")
        return False
    
    tprint("🧪 Testing VectorBT Feature Generation...")
    
    try:
        # Create test data
        np.random.seed(42)
        n_samples = 1000
        
        data = pd.DataFrame({
            'open': np.random.randn(n_samples).cumsum() + 100,
            'high': np.random.randn(n_samples).cumsum() + 105,
            'low': np.random.randn(n_samples).cumsum() + 95,
            'close': np.random.randn(n_samples).cumsum() + 100,
            'volume': np.random.randint(1000, 10000, n_samples)
        })
        
        # Test different feature types
        generator = create_vectorbt_feature_generator()
        
        feature_types = [
            FeatureType.SMA,
            FeatureType.RSI,
            FeatureType.MACD,
            FeatureType.BBANDS
        ]
        
        lookback_periods = [10, 20, 30, 50]
        
        for feature_type in feature_types:
            features = generator.generate_features_vectorbt(
                data, feature_type.value, lookback_periods, feature_type
            )
            
            tprint_success(f"✅ Generated {len(features)} {feature_type.value} features")
            
            # Check feature quality
            for period, values in features.items():
                if len(values) > 0:
                    tprint_info(f"📊 Period {period}: {len(values)} values, range: {values.min():.4f} to {values.max():.4f}")
        
        # Test cache performance
        cache_stats = generator.get_cache_stats()
        tprint_info(f"📦 Cache stats: {cache_stats}")
        
        return True
        
    except Exception as e:
        tprint_error(f"❌ VectorBT feature generation test failed: {e}")
        return False


if __name__ == "__main__":
    test_vectorbt_feature_generation()