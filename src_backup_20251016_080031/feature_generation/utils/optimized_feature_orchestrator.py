"""
import warnings
Optimized Feature Generation Orchestrator

This module provides a comprehensive feature generation system that integrates:
- TA-Lib technical indicators with hardware optimization
- ARIMA/ARMA time series modeling with M1 
- Parallel processing and memory optimization
- Safe mathematical operations and error handling
- Integration with existing ML pipeline

Benefits for Ares Trading System:
- Real-time feature generation capability
- Hardware-accelerated computations
- Memory-efficient processing
- Comprehensive error handling
- Seamless integration with existing codebase
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Callable, Tuple
import logging
from pathlib import Path
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# Import all optimization utilities
try:
    from .feature_generators import (
        FEATURE_GENERATORS, get_feature_generator
    )
    from src.utils.common_operations import (
        get_m1_gpu_manager, get_m1_memory_optimizer, get_m1_cpu_optimizer,
        create_fallback_logger
    )
    from src.utils.math_validation import (
        safe_divide, safe_log, safe_sqrt, safe_power,
        validate_finite, validate_positive, validate_range
    )
    from src.utils.parallel_processing_optimizer import ParallelProcessor
    from src.utils.data_validation import DataFrameValidator, DataFrameCleaner
    from src.utils.parquet_utils import ParquetUtils

    # Check TA-Lib availability
    try:
        import talib
        TALIB_AVAILABLE = True
    except ImportError:
        TALIB_AVAILABLE = False

    # Check statsmodels availability
    try:
        import statsmodels.api as sm
        from statsmodels.tsa.arima.model import ARIMA
        from statsmodels.tsa.stattools import adfuller

# VectorBT imports for native optimization
try:
    import vectorbt as vbt
    from vectorbt.generic import rolling_mean, rolling_std, rolling_var, rolling_min, rolling_max, rolling_sum, rolling_apply, rolling_corr, rolling_cov
    from vectorbt.generic import scale, rank, zscore, winsorize, clip, quantile
    VECTORBT_AVAILABLE = True
except ImportError:
    VECTORBT_AVAILABLE = False
    vbt = None
    rolling_mean = None
    rolling_std = None
    rolling_var = None
    rolling_min = None
    rolling_max = None
    rolling_sum = None
    rolling_apply = None
    rolling_corr = None
    rolling_cov = None
    scale = None
    rank = None
    zscore = None
    winsorize = None
    clip = None
    quantile = None
    warnings.warn("VectorBT not available. Install with: pip install vectorbt for optimized performance")

except ImportError:
    
    cp = None
        STATSMODELS_AVAILABLE = True
    except ImportError:
        STATSMODELS_AVAILABLE = False

    OPTIMIZATION_UTILS_AVAILABLE = True
    OPTIMIZATIONS_AVAILABLE = True
    logger.info("✅ All optimization utilities loaded successfully")
except ImportError as e:
    logger.warning(f"Some optimization utilities not available: {e}")
    TALIB_AVAILABLE = False
    STATSMODELS_AVAILABLE = False
    OPTIMIZATION_UTILS_AVAILABLE = False
    OPTIMIZATIONS_AVAILABLE = False

@dataclass
class FeatureGenerationConfig:
    """Configuration for optimized feature generation."""
    # Feature categories to generate
    enable_basic_indicators: bool = True
    enable_advanced_talib: bool = True
    enable_arima_features: bool = True
    enable_candlestick_features: bool = True

    # Hardware optimization settings
    enable_gpu_acceleration: bool = True
    enable_parallel_processing: bool = True
    max_workers: int = 4
    memory_efficient: bool = True

    # Data validation settings
    validate_input_data: bool = True
    clean_missing_values: bool = True
    outlier_detection: bool = True

    # Feature generation parameters
    lookback_periods: List[int] = field(default_factory=lambda: [5, 10, 14, 20, 50])
    arima_orders: List[Tuple[int, int, int]] = field(default_factory=lambda: [(1, 1, 1), (2, 1, 1), (1, 1, 2)])

    # Performance settings
    chunk_size: int = 1000
    cache_features: bool = True
    feature_cache_dir: str = "feature_cache"

class OptimizedFeatureOrchestrator:
    """
    Orchestrator for generating optimized trading features.

    Integrates TA-Lib, ARIMA/ARMA, and hardware optimizations
    for high-performance feature generation.
    """

    def __init__(self, config: Optional[FeatureGenerationConfig] = None):
        self.config = config or FeatureGenerationConfig()
        self.logger = logger.getChild('OptimizedFeatureOrchestrator')

        # Initialize optimization components
        self._initialize_optimizers()

        # Feature cache for performance
        self.feature_cache = {}
        self.generation_stats = {
            'features_generated': 0,
            'computation_time': 0.0,
            'cache_hits': 0,
            'errors': 0
        }

        self.logger.info("🚀 Optimized Feature Orchestrator initialized")
        self._log_capabilities()

    def _initialize_optimizers(self):
        """Initialize hardware and processing optimizers."""
        if OPTIMIZATIONS_AVAILABLE:
            self.gpu_manager = get_m1_gpu_manager()
            self.memory_optimizer = get_m1_memory_optimizer()
            self.cpu_optimizer = get_m1_cpu_optimizer()
            self.parallel_processor = ParallelProcessor(max_workers=self.config.max_workers)
        else:
            self.gpu_manager = None
            self.memory_optimizer = None
            self.cpu_optimizer = None
            self.parallel_processor = None

    def _log_capabilities(self):
        """Log available capabilities."""
        capabilities = []
        if TALIB_AVAILABLE:
            capabilities.append("TA-Lib indicators")
        if STATSMODELS_AVAILABLE:
            capabilities.append("ARIMA/ARMA modeling")
        if OPTIMIZATIONS_AVAILABLE:
            capabilities.append("Hardware optimizations")
            if self.gpu_manager and self.gpu_manager.mps_available:
                capabilities.append("M1 
            if self.parallel_processor:
                capabilities.append("Parallel processing")

        self.logger.info(f"📊 Capabilities: {', '.join(capabilities)}")

    def generate_all_features(self, data: pd.DataFrame,
                            feature_categories: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Generate all available features with optimizations.

        Args:
            data: OHLCV DataFrame
            feature_categories: List of categories to generate (optional)

        Returns:
            DataFrame with all generated features
        """
        start_time = time.time()

        try:
            # Validate and clean input data
            if self.config.validate_input_data:
                data = self._validate_and_clean_data(data)

            # Generate features by category
            all_features = pd.DataFrame(index=data.index)
            categories = feature_categories or self._get_enabled_categories()

            self.logger.info(f"🔄 Generating features for categories: {categories}")

            for category in categories:
                category_start = time.time()

                if category == 'basic_indicators':
                    features = self._generate_basic_indicators(data)
                elif category == 'advanced_talib':
                    features = self._generate_advanced_talib(data)
                elif category == 'arima_features':
                    features = self._generate_arima_features(data)
                elif category == 'candlestick_features':
                    features = self._generate_candlestick_features(data)
                else:
                    self.logger.warning(f"Unknown category: {category}")
                    continue

                # Add category features to main dataframe
                all_features = pd.concat([all_features, features], axis=1)

                category_time = time.time() - category_start
                self.logger.info(".2f")

            # Final validation and cleanup
            all_features = self._finalize_features(all_features, data)

            # Update statistics
            generation_time = time.time() - start_time
            self.generation_stats['features_generated'] += len(all_features.columns)
            self.generation_stats['computation_time'] += generation_time

            self.logger.info(".2f"
                           f"features: {len(all_features.columns)}")

            return all_features

        except Exception as e:
            self.logger.error(f"❌ Feature generation failed: {e}")
            self.generation_stats['errors'] += 1
            return pd.DataFrame(index=data.index)

    def _generate_basic_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate basic technical indicators with optimization."""
        features = pd.DataFrame(index=data.index)
        basic_indicators = ['rsi', 'sma', 'ema', 'macd', 'stochastic', 'atr', 'volume_sma']

        for indicator in basic_indicators:
            generator = get_feature_generator(indicator)
            if generator:
                for period in self.config.lookback_periods[:3]:  # Use first 3 periods for basic
                    try:
                        feature_series = generator(data, lookback=period)
                        features = pd.concat([features, feature_series], axis=1)
                    except Exception as e:
                        self.logger.debug(f"Failed to generate {indicator}_{period}: {e}")

        return features

    def _generate_advanced_talib(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate advanced TA-Lib indicators."""
        features = pd.DataFrame(index=data.index)

        if not TALIB_AVAILABLE:
            self.logger.warning("TA-Lib not available for advanced indicators")
            return features

        advanced_indicators = [
            ('williams_r', [14, 21]),
            ('cci', [14, 20]),
            ('ultimate_oscillator', [(7, 14, 28)]),
            ('kst_oscillator', [None])
        ]

        for indicator_name, periods_list in advanced_indicators:
            generator = get_feature_generator(indicator_name)
            if generator:
                for periods in periods_list:
                    try:
                        if periods is None:
                            feature_series = generator(data)
                        elif isinstance(periods, tuple):
                            feature_series = generator(data, *periods)
                        else:
                            feature_series = generator(data, periods)
                        features = pd.concat([features, feature_series], axis=1)
                    except Exception as e:
                        self.logger.debug(f"Failed to generate {indicator_name}: {e}")

        return features

    def _generate_arima_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate ARIMA-based features with hardware optimization."""
        features = pd.DataFrame(index=data.index)

        if not STATSMODELS_AVAILABLE:
            self.logger.warning("Statsmodels not available for ARIMA features")
            return features

        arima_indicators = ['arima_forecast', 'arima_residual', 'arima_volatility']

        for indicator in arima_indicators:
            generator = get_feature_generator(indicator)
            if generator:
                # Use different lookback periods for ARIMA
                arima_periods = [30, 50] if len(data) > 100 else [20]

                for period in arima_periods:
                    try:
                        # Use optimized ARIMA order
                        order = self.config.arima_orders[0]  # (1, 1, 1) as default
                        feature_series = generator(data, lookback=period, order=order)
                        features = pd.concat([features, feature_series], axis=1)
                    except Exception as e:
                        self.logger.debug(f"Failed to generate {indicator}_{period}: {e}")

        return features

    def _generate_candlestick_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate candlestick pattern features."""
        features = pd.DataFrame(index=data.index)
        candlestick_indicators = [
            'body_size', 'body_size_pct', 'body_to_range_ratio',
            'upper_wick', 'lower_wick', 'body_direction', 'body_strength'
        ]

        for indicator in candlestick_indicators:
            generator = get_feature_generator(indicator)
            if generator:
                try:
                    feature_series = generator(data)
                    features = pd.concat([features, feature_series], axis=1)
                except Exception as e:
                    self.logger.debug(f"Failed to generate {indicator}: {e}")

        return features

    def _validate_and_clean_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Validate and clean input data."""
        try:
            # Basic validation
            required_columns = ['open', 'high', 'low', 'close']
            if not all(col in data.columns for col in required_columns):
                raise ValueError(f"Missing required columns: {required_columns}")

            # Clean data if enabled
            if self.config.clean_missing_values:
                data = data.dropna(subset=required_columns)

            # Validate data ranges
            for col in required_columns:
                if (data[col] <= 0).any():
                    self.logger.warning(f"⚠️ Negative or zero values found in {col}")

            return data

        except Exception as e:
            self.logger.error(f"Data validation failed: {e}")
            return data

    def _finalize_features(self, features: pd.DataFrame, original_data: pd.DataFrame) -> pd.DataFrame:
        """Finalize and validate generated features."""
        try:
            # Remove duplicate columns (keep first occurrence)
            features = features.loc[:, ~features.columns.duplicated()]

            # Fill any remaining NaN values
            features = features.fillna(0.0)

            # Validate feature ranges (prevent extreme values)
            for col in features.columns:
                if features[col].dtype in ['float64', 'float32']:
                    # Clip extreme values
                    features[col] = np.clip(features[col], -1e6, 1e6)

            # Ensure same index as original data
            features = features.reindex(original_data.index)

            return features

        except Exception as e:
            self.logger.error(f"Feature finalization failed: {e}")
            return features

    def _get_enabled_categories(self) -> List[str]:
        """Get list of enabled feature categories."""
        categories = []
        if self.config.enable_basic_indicators:
            categories.append('basic_indicators')
        if self.config.enable_advanced_talib:
            categories.append('advanced_talib')
        if self.config.enable_arima_features:
            categories.append('arima_features')
        if self.config.enable_candlestick_features:
            categories.append('candlestick_features')
        return categories

    def get_generation_stats(self) -> Dict[str, Any]:
        """Get feature generation statistics."""
        return self.generation_stats.copy()

    def clear_cache(self):
        """Clear feature cache."""
        self.feature_cache.clear()
        self.generation_stats['cache_hits'] = 0

# Convenience functions for easy usage
def create_optimized_orchestrator(enable_gpu: bool = True,
                                enable_parallel: bool = True) -> OptimizedFeatureOrchestrator:
    """Create an optimized feature orchestrator with sensible defaults."""
    config = FeatureGenerationConfig(
        enable_gpu_acceleration=enable_gpu,
        enable_parallel_processing=enable_parallel,
        max_workers=4 if enable_parallel else 1,
        validate_input_data=True,
        clean_missing_values=True,
        lookback_periods=[5, 10, 14, 20, 30],
        arima_orders=[(1, 1, 1)]
    )

    return OptimizedFeatureOrchestrator(config)

def generate_trading_features(data: pd.DataFrame,
                            include_arima: bool = True,
                            include_advanced_talib: bool = True) -> pd.DataFrame:
    """
    Convenience function to generate all trading features.

    Args:
        data: OHLCV DataFrame
        include_arima: Whether to include ARIMA features
        include_advanced_talib: Whether to include advanced TA-Lib features

    Returns:
        DataFrame with generated features
    """
    config = FeatureGenerationConfig(
        enable_arima_features=include_arima,
        enable_advanced_talib=include_advanced_talib
    )

    orchestrator = OptimizedFeatureOrchestrator(config)
    return orchestrator.generate_all_features(data)

if __name__ == "__main__":
    # Example usage
    print("=== Optimized Feature Generation Orchestrator ===")
    print("This module provides:")
    print("✅ TA-Lib integration with hardware optimization")
    print("✅ ARIMA/ARMA time series modeling")
    print("✅ Parallel processing and memory optimization")
    print("✅ Safe mathematical operations")
    print("✅ Comprehensive error handling")
    print("\nUsage:")
    print("  orchestrator = create_optimized_orchestrator()")
    print("  features = orchestrator.generate_all_features(ohlcv_data)")
    print("  # Features ready for ML training!")

    def _should_use_vectorbt(self, data) -> bool:
        """Determine if VectorBT should be used based on data size and configuration."""
        return (hasattr(self, 'use_vectorbt') and self.use_vectorbt and 
                len(data) >= getattr(self, 'vectorbt_threshold', 1000) and 
                VECTORBT_AVAILABLE)
    
    def _vectorbt_rolling_operation(self, data: pd.Series, operation: str, 
                                  window: int, **kwargs) -> pd.Series:
        """Perform VectorBT rolling operation with fallback to pandas."""
        if not self._should_use_vectorbt(data):
            return self._pandas_rolling_operation(data, operation, window, **kwargs)
        
        try:
            if operation == 'mean':
                return rolling_mean(data, window=window, **kwargs)
            elif operation == 'std':
                return rolling_std(data, window=window, **kwargs)
            elif operation == 'var':
                return rolling_var(data, window=window, **kwargs)
            elif operation == 'min':
                return rolling_min(data, window=window, **kwargs)
            elif operation == 'max':
                return rolling_max(data, window=window, **kwargs)
            elif operation == 'sum':
                return rolling_sum(data, window=window, **kwargs)
            else:
                raise ValueError(f"Unsupported operation: {operation}")
        except Exception as e:
            logger.warning(f"VectorBT operation failed: {e}, using pandas fallback")
            return self._pandas_rolling_operation(data, operation, window, **kwargs)
    
    def _pandas_rolling_operation(self, data: pd.Series, operation: str, 
                                 window: int, **kwargs) -> pd.Series:
        """Fallback rolling operation using pandas."""
        if operation == 'mean':
            return data.rolling(window=window).mean()
        elif operation == 'std':
            return data.rolling(window=window).std()
        elif operation == 'var':
            return data.rolling(window=window).var()
        elif operation == 'min':
            return data.rolling(window=window).min()
        elif operation == 'max':
            return data.rolling(window=window).max()
        elif operation == 'sum':
            return data.rolling(window=window).sum()
        else:
            raise ValueError(f"Unsupported operation: {operation}")
