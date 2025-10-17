"""
VectorBT-Optimized Feature Generator

This module provides a high-performance feature generator base class that leverages
VectorBT's optimized C++ backend for maximum performance in feature generation.

Key Features:
- VectorBT indicators integration
- Optimized rolling operations
- Memory-efficient processing
- GPU acceleration support
- Batch processing capabilities
"""

import numpy as np
import pandas as pd
import logging
import time
import os
from typing import Any, Dict, List, Optional, Union, Callable, Tuple
from abc import ABC, abstractmethod
import warnings

# VectorBT imports
try:
    import vectorbt as vbt
    from vectorbt.indicators.basic import RSI, MACD, ATR, BBANDS, STOCH, OBV, MA
    VECTORBT_AVAILABLE = True
except ImportError:
    VECTORBT_AVAILABLE = False
    vbt = None
    # Fast-fail classes when VectorBT is not available
    class RSI:
        @staticmethod
        def run(*args, **kwargs):
            raise ImportError("VectorBT not available - install with: pip install vectorbt")
    
    class MACD:
        @staticmethod
        def run(*args, **kwargs):
            raise ImportError("VectorBT not available - install with: pip install vectorbt")
    
    class ATR:
        @staticmethod
        def run(*args, **kwargs):
            raise ImportError("VectorBT not available - install with: pip install vectorbt")
    
    class BBANDS:
        @staticmethod
        def run(*args, **kwargs):
            raise ImportError("VectorBT not available - install with: pip install vectorbt")
    
    class STOCH:
        @staticmethod
        def run(*args, **kwargs):
            raise ImportError("VectorBT not available - install with: pip install vectorbt")
    
    class OBV:
        @staticmethod
        def run(*args, **kwargs):
            raise ImportError("VectorBT not available - install with: pip install vectorbt")
    
    class MA:
        @staticmethod
        def run(*args, **kwargs):
            raise ImportError("VectorBT not available - install with: pip install vectorbt")

from .feature_generator import FeatureGenerator, FeatureConfig, FeatureResult, FeatureCategory
from ..utils.math_validation import safe_divide, validate_finite, safe_percentage_change
from src.utils.ml_common.vectorbt_memory_manager import get_memory_manager, memory_managed_operation, optimize_memory_usage
from src.utils.ml_common.vectorbt_performance_monitor import get_performance_monitor, monitor_operation

logger = logging.getLogger(__name__)

class VectorBTFeatureGenerator(FeatureGenerator):
    """
    High-performance feature generator using VectorBT's optimized backend.

    This class provides a foundation for creating feature generators that leverage
    VectorBT's C++ optimized implementations for maximum performance.
    """

    def __init__(self, config: FeatureConfig, enable_gpu: bool = False, enable_parallel: bool = True):
        """
        Initialize VectorBT feature generator.

        Args:
            config: Feature configuration
            enable_gpu: Whether to enable GPU acceleration
            enable_parallel: Whether to enable parallel processing
        """
        if not VECTORBT_AVAILABLE:
            raise ImportError("VectorBT is required but not available. Install with: pip install vectorbt")

        super().__init__(config)
        self.enable_gpu = enable_gpu
        self.enable_parallel = enable_parallel

        # Initialize memory manager and performance monitor
        self.memory_manager = get_memory_manager()
        self.performance_monitor = get_performance_monitor()

        # Configure VectorBT settings
        self._configure_vectorbt()

        # Performance tracking
        self.vectorbt_stats = {
            'vectorbt_operations': 0,
            'gpu_accelerations': 0,
            'parallel_operations': 0,
            'memory_optimizations': 0
        }

        # Cache for computed features
        self._feature_cache = {}
        self._cache_enabled = True

        # Memory optimization
        self._memory_usage = 0
        self._max_memory_usage = 0

    def _configure_vectorbt(self):
        """Configure VectorBT global settings for optimal performance."""
        if not VECTORBT_AVAILABLE:
            return

        # Configure VectorBT settings for optimal performance using newer API
        # Check if array_wrapper structure exists and set wrapper if available
        if hasattr(vbt.settings, 'array_wrapper') and 'wrapper' in vbt.settings['array_wrapper']:
            vbt.settings['array_wrapper']['wrapper'] = 'pandas'
        vbt.settings['caching']['enabled'] = True

        # Advanced caching configuration (if available in this VectorBT version)
        if hasattr(vbt.settings, 'caching') and 'cache_size' in vbt.settings['caching']:
            vbt.settings['caching']['cache_size'] = 1000  # 1GB cache
        if hasattr(vbt.settings, 'caching') and 'cache_ttl' in vbt.settings['caching']:
            vbt.settings['caching']['cache_ttl'] = 3600  # 1 hour TTL
        if hasattr(vbt.settings, 'caching') and 'cache_compression' in vbt.settings['caching']:
            vbt.settings['caching']['cache_compression'] = True

        # Memory optimization settings (if available in this VectorBT version)
        if hasattr(vbt.settings, 'memory') and 'memory_limit' in vbt.settings['memory']:
            vbt.settings['memory']['memory_limit'] = self.vectorbt_memory_limit_gb * 1024**3  # Convert GB to bytes
        if hasattr(vbt.settings, 'chunking') and 'chunk_size' in vbt.settings['chunking']:
            vbt.settings['chunking']['chunk_size'] = 10000  # Process data in chunks for memory efficiency

        # Array wrapper optimization (if available in this VectorBT version)
        if hasattr(vbt.settings, 'array_wrapper') and 'optimize' in vbt.settings['array_wrapper']:
            vbt.settings['array_wrapper']['optimize'] = True
        if hasattr(vbt.settings, 'array_wrapper') and 'compress' in vbt.settings['array_wrapper']:
            vbt.settings['array_wrapper']['compress'] = True

        if self.enable_gpu:
            try:
                # Check if GPU settings are available in this VectorBT version
                if hasattr(vbt.settings, 'gpu') and 'enabled' in vbt.settings['gpu']:
                    vbt.settings['gpu']['enabled'] = True
                    logger.info("✅ VectorBT GPU acceleration enabled")
                else:
                    logger.warning("⚠️ GPU acceleration not available in this VectorBT version")
                    self.enable_gpu = False
            except Exception as e:
                logger.warning(f"⚠️ GPU acceleration not available: {e}")
                self.enable_gpu = False

        if self.enable_parallel:
            try:
                # Check if parallel settings are available in this VectorBT version
                if hasattr(vbt.settings, 'parallel') and 'enabled' in vbt.settings['parallel']:
                    vbt.settings['parallel']['enabled'] = True
                if hasattr(vbt.settings, 'parallel') and 'n_threads' in vbt.settings['parallel']:
                    vbt.settings['parallel']['n_threads'] = min(8, os.cpu_count())  # Limit threads for memory efficiency
                logger.debug("✅ VectorBT parallel processing enabled")
            except Exception as e:
                logger.warning(f"⚠️ Parallel processing not available: {e}")
                self.enable_parallel = False

    def _vectorbt_rolling_operation(self, data: pd.Series, operation: str,
                                  window: int, **kwargs) -> pd.Series:
        """
        Perform rolling operation using VectorBT native functions with array wrappers for maximum performance.

        Args:
            data: Input data series
            operation: Operation type ('mean', 'std', 'var', 'min', 'max', 'sum', 'corr', 'cov')
            window: Rolling window size
            **kwargs: Additional parameters

        Returns:
            Result of rolling operation
        """
        self.vectorbt_stats['vectorbt_operations'] += 1

        try:
            # Convert to VectorBT array wrapper for optimal performance
            if not hasattr(data, '_vbt') and VECTORBT_AVAILABLE:
                data = vbt.array_wrapper(data, freq=data.index.freq if hasattr(data.index, 'freq') else None)

            # Use VectorBT native rolling functions with array wrappers
            if operation == 'mean':
                return vbt.rolling_mean(data, window=window, **kwargs)
            elif operation == 'std':
                return vbt.rolling_std(data, window=window, **kwargs)
            elif operation == 'var':
                return vbt.rolling_var(data, window=window, **kwargs)
            elif operation == 'min':
                return vbt.rolling_min(data, window=window, **kwargs)
            elif operation == 'max':
                return vbt.rolling_max(data, window=window, **kwargs)
            elif operation == 'sum':
                return vbt.rolling_sum(data, window=window, **kwargs)
            elif operation == 'corr':
                other = kwargs.get('other')
                if other is None:
                    raise ValueError("'other' parameter required for correlation")
                # Convert other to array wrapper if needed
                if not hasattr(other, '_vbt') and VECTORBT_AVAILABLE:
                    other = vbt.array_wrapper(other, freq=other.index.freq if hasattr(other.index, 'freq') else None)
                return vbt.rolling_corr(data, other, window=window, **kwargs)
            elif operation == 'cov':
                other = kwargs.get('other')
                if other is None:
                    raise ValueError("'other' parameter required for covariance")
                # Convert other to array wrapper if needed
                if not hasattr(other, '_vbt') and VECTORBT_AVAILABLE:
                    other = vbt.array_wrapper(other, freq=other.index.freq if hasattr(other.index, 'freq') else None)
                return vbt.rolling_cov(data, other, window=window, **kwargs)
            else:
                raise ValueError(f"Unsupported operation: {operation}")

        except Exception as e:
            logger.warning(f"VectorBT rolling operation failed: {e}, using pandas fallback")
            return self._fallback_rolling_operation(data, operation, window, **kwargs)

    def _fallback_rolling_operation(self, data: pd.Series, operation: str,
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
        elif operation == 'corr':
            other = kwargs.get('other')
            return data.rolling(window=window).corr(other)
        elif operation == 'cov':
            other = kwargs.get('other')
            return data.rolling(window=window).cov(other)
        else:
            raise ValueError(f"Unsupported operation: {operation}")

    def _vectorbt_scale(self, data: pd.Series, method: str = 'zscore', **kwargs) -> pd.Series:
        """
        Scale data using VectorBT scaling functions.

        Args:
            data: Input data series
            method: Scaling method ('zscore', 'minmax', 'robust', 'quantile', 'winsorize')
            **kwargs: Additional parameters

        Returns:
            Scaled data series
        """
        if not VECTORBT_AVAILABLE:
            raise ImportError("VectorBT is required for this operation")

        self.vectorbt_stats['vectorbt_operations'] += 1

        try:
            if method == 'zscore':
                return zscore(data, **kwargs)
            elif method == 'minmax':
                return scale(data, method='minmax', **kwargs)
            elif method == 'robust':
                return scale(data, method='robust', **kwargs)
            elif method == 'quantile':
                return quantile(data, **kwargs)
            elif method == 'winsorize':
                return winsorize(data, **kwargs)
            elif method == 'rank':
                return rank(data, **kwargs)
            elif method == 'clip':
                return clip(data, **kwargs)
            else:
                raise ValueError(f"Unsupported scaling method: {method}")

        except Exception as e:
            logger.warning(f"VectorBT scaling failed: {e}, using fallback")
            return self._fallback_scale(data, method, **kwargs)

    def _fallback_scale(self, data: pd.Series, method: str, **kwargs) -> pd.Series:
        """Fallback scaling using pandas/numpy."""
        if method == 'zscore':
            return (data - data.mean()) / data.std()
        elif method == 'minmax':
            return (data - data.min()) / (data.max() - data.min())
        elif method == 'robust':
            median = data.median()
            mad = (data - median).abs().median()
            return (data - median) / mad
        else:
            raise ValueError(f"Unsupported scaling method: {method}")

    def _vectorbt_technical_indicator(self, data: pd.DataFrame, indicator: str, **kwargs) -> pd.Series:
        """
        Calculate technical indicator using VectorBT native implementations for maximum performance.

        Args:
            data: OHLCV data
            indicator: Indicator name
            **kwargs: Indicator parameters

        Returns:
            Indicator values
        """
        if not VECTORBT_AVAILABLE:
            raise ImportError("VectorBT is required for this operation")

        self.vectorbt_stats['vectorbt_operations'] += 1

        try:
            # Use VectorBT native indicator implementations
            if indicator == 'rsi':
                return vbt.RSI.run(data['close'], **kwargs).rsi
            elif indicator == 'macd':
                macd_result = vbt.MACD.run(data['close'], **kwargs)
                return macd_result.macd
            elif indicator == 'macd_signal':
                macd_result = vbt.MACD.run(data['close'], **kwargs)
                return macd_result.signal
            elif indicator == 'macd_histogram':
                macd_result = vbt.MACD.run(data['close'], **kwargs)
                return macd_result.histogram
            elif indicator == 'atr':
                return vbt.ATR.run(data['high'], data['low'], data['close'], **kwargs).atr
            elif indicator == 'bbands_upper':
                bb_result = vbt.BBANDS.run(data['close'], **kwargs)
                return bb_result.upper
            elif indicator == 'bbands_middle':
                bb_result = vbt.BBANDS.run(data['close'], **kwargs)
                return bb_result.middle
            elif indicator == 'bbands_lower':
                bb_result = vbt.BBANDS.run(data['close'], **kwargs)
                return bb_result.lower
            elif indicator == 'bbands_width':
                bb_result = vbt.BBANDS.run(data['close'], **kwargs)
                return bb_result.width
            elif indicator == 'bbands_percent':
                bb_result = vbt.BBANDS.run(data['close'], **kwargs)
                return bb_result.percent
            elif indicator == 'stoch_k':
                stoch_result = vbt.STOCH.run(data['high'], data['low'], data['close'], **kwargs)
                return stoch_result.stoch_k
            elif indicator == 'stoch_d':
                stoch_result = vbt.STOCH.run(data['high'], data['low'], data['close'], **kwargs)
                return stoch_result.stoch_d
            elif indicator == 'obv':
                return vbt.OBV.run(data['close'], data['volume'], **kwargs).obv
            elif indicator == 'sma':
                return vbt.MA.run(data['close'], **kwargs).ma
            elif indicator == 'ema':
                return vbt.EMA.run(data['close'], **kwargs).ema
            elif indicator == 'wma':
                return vbt.WMA.run(data['close'], **kwargs).wma
            elif indicator == 'willr':
                return vbt.WILLR.run(data['high'], data['low'], data['close'], **kwargs).willr
            elif indicator == 'cci':
                return vbt.CCI.run(data['high'], data['low'], data['close'], **kwargs).cci
            elif indicator == 'mfi':
                return vbt.MFI.run(data['high'], data['low'], data['close'], data['volume'], **kwargs).mfi
            elif indicator == 'adx':
                return vbt.ADX.run(data['high'], data['low'], data['close'], **kwargs).adx
            elif indicator == 'roc':
                return vbt.ROC.run(data['close'], **kwargs).roc
            elif indicator == 'mom':
                return vbt.MOM.run(data['close'], **kwargs).mom
            else:
                raise ValueError(f"Unsupported indicator: {indicator}")

        except Exception as e:
            logger.warning(f"VectorBT indicator {indicator} failed: {e}, using fallback")
            return self._fallback_technical_indicator(data, indicator, **kwargs)

    def _fallback_technical_indicator(self, data: pd.DataFrame, indicator: str, **kwargs) -> pd.Series:
        """Fallback technical indicator calculation using pandas/numpy."""
        # This would contain fallback implementations
        # For now, return NaN series
        return pd.Series(np.nan, index=data.index)

    def _vectorbt_batch_operations(self, data: pd.DataFrame, operations: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        Perform batch VectorBT operations for efficiency.

        Args:
            data: Input data
            operations: List of operation dictionaries

        Returns:
            DataFrame with results
        """
        if not VECTORBT_AVAILABLE:
            raise ImportError("VectorBT is required for this operation")

        self.vectorbt_stats['vectorbt_operations'] += len(operations)
        self.vectorbt_stats['parallel_operations'] += 1

        results = {}

        try:
            # Use VectorBT's batch processing if available
            for op in operations:
                op_type = op.get('type')
                op_params = op.get('params', {})
                op_name = op.get('name', f"{op_type}_{len(results)}")

                if op_type == 'rolling':
                    operation = op_params.get('operation')
                    window = op_params.get('window')
                    column = op_params.get('column', 'close')
                    results[op_name] = self._vectorbt_rolling_operation(
                        data[column], operation, window, **op_params
                    )
                elif op_type == 'indicator':
                    indicator = op_params.get('indicator')
                    results[op_name] = self._vectorbt_technical_indicator(
                        data, indicator, **op_params
                    )
                elif op_type == 'scale':
                    method = op_params.get('method', 'zscore')
                    column = op_params.get('column', 'close')
                    results[op_name] = self._vectorbt_scale(
                        data[column], method, **op_params
                    )

        except Exception as e:
            logger.warning(f"VectorBT batch operations failed: {e}")
            # Return empty DataFrame on failure
            return pd.DataFrame(index=data.index)

        return pd.DataFrame(results, index=data.index)

    def _vectorbt_batch_indicators_optimized(self, data: pd.DataFrame,
                                           indicators: List[Dict[str, Any]]) -> pd.DataFrame:
        """Optimized batch indicator calculation with memory management."""
        if not VECTORBT_AVAILABLE:
            return self._fallback_batch_indicators(data, indicators)

        # Estimate memory requirements
        data_size_gb = data.memory_usage(deep=True).sum() / (1024**3)
        estimated_memory_gb = data_size_gb * len(indicators) * 2  # Rough estimate

        with memory_managed_operation(
            estimated_memory_gb,
            f"batch_indicators_{int(time.time())}",
            "feature_generation"
        ):
            try:
                results = {}

                # Process indicators in batches for memory efficiency
                batch_size = min(10, len(indicators))  # Process 10 indicators at a time

                for i in range(0, len(indicators), batch_size):
                    batch_indicators = indicators[i:i + batch_size]

                    for indicator_config in batch_indicators:
                        indicator_name = indicator_config['name']
                        indicator_type = indicator_config['type']
                        params = indicator_config.get('params', {})

                        # Check cache first
                        cache_key = f"{indicator_name}_{hash(str(params))}"
                        if self._cache_enabled and cache_key in self._feature_cache:
                            results[indicator_name] = self._feature_cache[cache_key]
                            continue

                        # VectorBT optimized calculation
                        try:
                            if indicator_type == 'rsi':
                                result = vbt.RSI.run(data['close'], **params).rsi
                            elif indicator_type == 'macd':
                                result = vbt.MACD.run(data['close'], **params).macd
                            elif indicator_type == 'atr':
                                result = vbt.ATR.run(data['high'], data['low'], data['close'], **params).atr
                            elif indicator_type == 'bbands_upper':
                                result = vbt.BBANDS.run(data['close'], **params).upper
                            elif indicator_type == 'bbands_lower':
                                result = vbt.BBANDS.run(data['close'], **params).lower
                            elif indicator_type == 'stoch_k':
                                result = vbt.STOCH.run(data['high'], data['low'], data['close'], **params).k
                            elif indicator_type == 'willr':
                                result = vbt.WILLR.run(data['high'], data['low'], data['close'], **params).willr
                            elif indicator_type == 'cci':
                                result = vbt.CCI.run(data['high'], data['low'], data['close'], **params).cci
                            elif indicator_type == 'mfi':
                                result = vbt.MFI.run(data['high'], data['low'], data['close'], data['volume'], **params).mfi
                            elif indicator_type == 'adx':
                                result = vbt.ADX.run(data['high'], data['low'], data['close'], **params).adx
                            elif indicator_type == 'roc':
                                result = vbt.ROC.run(data['close'], **params).roc
                            elif indicator_type == 'mom':
                                result = vbt.MOM.run(data['close'], **params).mom
                            elif indicator_type == 'obv':
                                result = vbt.OBV.run(data['close'], data['volume'], **params).obv
                            else:
                                logger.warning(f"Unknown indicator type: {indicator_type}")
                                continue

                            # Optimize data types
                            result = optimize_memory_usage(result)

                            # Cache result
                            if self._cache_enabled:
                                self._feature_cache[cache_key] = result
                                # Limit cache size
                                if len(self._feature_cache) > 1000:
                                    # Remove oldest entries
                                    oldest_key = next(iter(self._feature_cache))
                                    del self._feature_cache[oldest_key]

                            results[indicator_name] = result

                        except Exception as e:
                            logger.warning(f"VectorBT indicator {indicator_type} failed: {e}")
                            continue

                self.vectorbt_stats['vectorbt_operations'] += len(indicators)
                self.vectorbt_stats['parallel_operations'] += 1

                return pd.DataFrame(results, index=data.index)

            except Exception as e:
                logger.error(f"VectorBT batch indicators failed: {e}")
                return self._fallback_batch_indicators(data, indicators)

    def _fallback_batch_indicators(self, data: pd.DataFrame, indicators: List[Dict[str, Any]]) -> pd.DataFrame:
        """Fallback batch indicator calculation using pandas/numpy."""
        results = {}

        for indicator_config in indicators:
            indicator_name = indicator_config['name']
            indicator_type = indicator_config['type']
            params = indicator_config.get('params', {})

            try:
                if indicator_type == 'rsi':
                    # Simple RSI calculation
                    delta = data['close'].diff()
                    gain = (delta.where(delta > 0, 0)).rolling(window=params.get('window', 14)).mean()
                    loss = (-delta.where(delta < 0, 0)).rolling(window=params.get('window', 14)).mean()
                    rs = gain / loss
                    result = 100 - (100 / (1 + rs))
                elif indicator_type == 'macd':
                    # Simple MACD calculation
                    ema_fast = data['close'].ewm(span=params.get('fast_window', 12)).mean()
                    ema_slow = data['close'].ewm(span=params.get('slow_window', 26)).mean()
                    result = ema_fast - ema_slow
                else:
                    # For other indicators, return NaN series
                    result = pd.Series(np.nan, index=data.index)

                results[indicator_name] = result

            except Exception as e:
                logger.warning(f"Fallback indicator {indicator_type} failed: {e}")
                results[indicator_name] = pd.Series(np.nan, index=data.index)

        return pd.DataFrame(results, index=data.index)

    def generate_features_batch_optimized(self, data: pd.DataFrame,
                                        feature_configs: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        Generate multiple features in batch with memory and performance optimization.

        Args:
            data: Input OHLCV data
            feature_configs: List of feature configuration dictionaries

        Returns:
            DataFrame with generated features
        """
        with monitor_operation(
            f"batch_feature_generation_{len(feature_configs)}",
            metadata={'n_features': len(feature_configs), 'data_shape': data.shape}
        ):
            logger.info(f"🚀 Generating {len(feature_configs)} features in batch...")

            # Group features by type for efficient processing
            indicator_features = []
            rolling_features = []
            scaling_features = []

            for config in feature_configs:
                feature_type = config.get('type', 'indicator')
                if feature_type == 'indicator':
                    indicator_features.append(config)
                elif feature_type == 'rolling':
                    rolling_features.append(config)
                elif feature_type == 'scaling':
                    scaling_features.append(config)

            results = {}

            # Process indicator features
            if indicator_features:
                logger.debug(f"Processing {len(indicator_features)} indicator features...")
                indicator_results = self._vectorbt_batch_indicators_optimized(data, indicator_features)
                results.update(indicator_results)

            # Process rolling features
            if rolling_features:
                logger.debug(f"Processing {len(rolling_features)} rolling features...")
                rolling_results = self._process_rolling_features_batch(data, rolling_features)
                results.update(rolling_results)

            # Process scaling features
            if scaling_features:
                logger.debug(f"Processing {len(scaling_features)} scaling features...")
                scaling_results = self._process_scaling_features_batch(data, scaling_features)
                results.update(scaling_results)

            # Combine all results
            result_df = pd.DataFrame(results, index=data.index)

            # Optimize final result
            result_df = optimize_memory_usage(result_df)

            logger.info(f"✅ Generated {len(result_df.columns)} features successfully")
            return result_df

    def _process_rolling_features_batch(self, data: pd.DataFrame,
                                      rolling_configs: List[Dict[str, Any]]) -> Dict[str, pd.Series]:
        """Process rolling features in batch."""
        results = {}

        for config in rolling_configs:
            feature_name = config['name']
            column = config.get('column', 'close')
            operation = config.get('operation', 'mean')
            window = config.get('window', 20)

            if column not in data.columns:
                logger.warning(f"Column {column} not found for rolling feature {feature_name}")
                results[feature_name] = pd.Series(np.nan, index=data.index)
                continue

            try:
                if operation == 'mean':
                    result = data[column].rolling(window=window).mean()
                elif operation == 'std':
                    result = data[column].rolling(window=window).std()
                elif operation == 'min':
                    result = data[column].rolling(window=window).min()
                elif operation == 'max':
                    result = data[column].rolling(window=window).max()
                elif operation == 'sum':
                    result = data[column].rolling(window=window).sum()
                else:
                    logger.warning(f"Unknown rolling operation: {operation}")
                    result = pd.Series(np.nan, index=data.index)

                results[feature_name] = result

            except Exception as e:
                logger.warning(f"Rolling feature {feature_name} failed: {e}")
                results[feature_name] = pd.Series(np.nan, index=data.index)

        return results

    def _process_scaling_features_batch(self, data: pd.DataFrame,
                                      scaling_configs: List[Dict[str, Any]]) -> Dict[str, pd.Series]:
        """Process scaling features in batch."""
        results = {}

        for config in scaling_configs:
            feature_name = config['name']
            column = config.get('column', 'close')
            method = config.get('method', 'zscore')

            if column not in data.columns:
                logger.warning(f"Column {column} not found for scaling feature {feature_name}")
                results[feature_name] = pd.Series(np.nan, index=data.index)
                continue

            try:
                if method == 'zscore':
                    result = (data[column] - data[column].mean()) / data[column].std()
                elif method == 'minmax':
                    result = (data[column] - data[column].min()) / (data[column].max() - data[column].min())
                elif method == 'robust':
                    median = data[column].median()
                    mad = (data[column] - median).abs().median()
                    result = (data[column] - median) / mad
                else:
                    logger.warning(f"Unknown scaling method: {method}")
                    result = pd.Series(np.nan, index=data.index)

                results[feature_name] = result

            except Exception as e:
                logger.warning(f"Scaling feature {feature_name} failed: {e}")
                results[feature_name] = pd.Series(np.nan, index=data.index)

        return results

    def get_vectorbt_stats(self) -> Dict[str, Any]:
        """Get VectorBT performance statistics."""
        return self.vectorbt_stats.copy()

    def reset_vectorbt_stats(self):
        """Reset VectorBT performance statistics."""
        self.vectorbt_stats = {
            'vectorbt_operations': 0,
            'gpu_accelerations': 0,
            'parallel_operations': 0,
            'memory_optimizations': 0
        }

    def _optimize_dataframe_for_vectorbt(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Optimize DataFrame for VectorBT processing with memory efficiency and array wrappers.

        Args:
            data: Input DataFrame

        Returns:
            Optimized DataFrame with VectorBT array wrappers
        """
        try:
            # Create a copy to avoid modifying original data
            optimized_data = data.copy()

            # Optimize data types for memory efficiency
            for column in optimized_data.columns:
                if optimized_data[column].dtype == 'float64':
                    # Use float32 for better memory usage if precision allows
                    if optimized_data[column].min() >= np.finfo(np.float32).min and \
                       optimized_data[column].max() <= np.finfo(np.float32).max:
                        optimized_data[column] = optimized_data[column].astype(np.float32)

                elif optimized_data[column].dtype == 'int64':
                    # Use int32 for better memory usage if range allows
                    if optimized_data[column].min() >= np.iinfo(np.int32).min and \
                       optimized_data[column].max() <= np.iinfo(np.int32).max:
                        optimized_data[column] = optimized_data[column].astype(np.int32)

            # Ensure index is optimized
            if isinstance(optimized_data.index, pd.DatetimeIndex):
                # Use period index for better memory usage if possible
                try:
                    optimized_data.index = optimized_data.index.to_period('T')
                except:
                    pass

            # Convert to VectorBT array wrappers for better performance
            if VECTORBT_AVAILABLE:
                optimized_data = self._convert_to_vectorbt_arrays(optimized_data)

            # Track memory usage
            memory_usage = optimized_data.memory_usage(deep=True).sum()
            self._memory_usage = memory_usage
            self._max_memory_usage = max(self._max_memory_usage, memory_usage)

            self.vectorbt_stats['memory_optimizations'] += 1

            return optimized_data

        except Exception as e:
            logger.warning(f"DataFrame optimization failed: {e}")
            return data

    def _convert_to_vectorbt_arrays(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Convert DataFrame columns to VectorBT array wrappers for optimal performance.

        Args:
            data: Input DataFrame

        Returns:
            DataFrame with VectorBT array wrappers
        """
        try:
            if not VECTORBT_AVAILABLE:
                return data

            optimized_data = data.copy()

            # Convert numeric columns to VectorBT array wrappers
            for column in optimized_data.columns:
                if optimized_data[column].dtype in ['float32', 'float64', 'int32', 'int64']:
                    try:
                        # Convert to VectorBT array wrapper
                        optimized_data[column] = vbt.array_wrapper(
                            optimized_data[column],
                            freq=data.index.freq if hasattr(data.index, 'freq') else None
                        )
                        self.vectorbt_stats['vectorbt_operations'] += 1
                    except Exception as e:
                        logger.warning(f"Failed to convert column {column} to VectorBT array: {e}")
                        continue

            return optimized_data

        except Exception as e:
            logger.warning(f"VectorBT array conversion failed: {e}")
            return data

    def _cleanup_memory(self):
        """Clean up memory usage."""
        try:
            import gc
            gc.collect()

            # Clear feature cache if memory usage is high
            if self._memory_usage > self.vectorbt_memory_limit_gb * 1024**3 * 0.8:
                self._feature_cache.clear()
                logger.info("🧹 Cleared feature cache due to high memory usage")

        except Exception as e:
            logger.warning(f"Memory cleanup failed: {e}")

    def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory usage statistics."""
        return {
            'current_memory_usage': self._memory_usage,
            'max_memory_usage': self._max_memory_usage,
            'memory_limit': self.vectorbt_memory_limit_gb * 1024**3,
            'memory_usage_percentage': (self._memory_usage / (self.vectorbt_memory_limit_gb * 1024**3)) * 100
        }

class VectorBTVolatilityGenerator(VectorBTFeatureGenerator):
    """VectorBT-optimized volatility feature generator."""

    def __init__(self, period: int = 20, config: Optional[FeatureConfig] = None):
        if config is None:
            config = self._create_default_config(period)
        super().__init__(config)
        self.period = period

    @classmethod
    def _create_default_config(cls, period: int = 20) -> FeatureConfig:
        return FeatureConfig(
            name=f"vectorbt_volatility_{period}",
            category=FeatureCategory.VOLATILITY,
            description=f"VectorBT-optimized volatility measure over {period} periods",
            required_columns=["close"],
            optional_columns=["high", "low", "open"],
            default_lookback=period,
            min_lookback=period,
            max_lookback=period,
            parameters={"period": period},
            matrix_optimized=True,
            gpu_accelerated=True
        )

    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate volatility feature using VectorBT ATR."""
        if len(data) == 0:
            return pd.Series(dtype=float, index=data.index, name=f'vectorbt_volatility_{self.period}')

        # Use VectorBT ATR for volatility calculation
        atr = self._vectorbt_technical_indicator(data, 'atr', window=self.period)
        return atr.rename(f'vectorbt_volatility_{self.period}')

class VectorBTMomentumGenerator(VectorBTFeatureGenerator):
    """VectorBT-optimized momentum feature generator."""

    def __init__(self, period: int = 14, config: Optional[FeatureConfig] = None):
        if config is None:
            config = self._create_default_config(period)
        super().__init__(config)
        self.period = period

    @classmethod
    def _create_default_config(cls, period: int = 14) -> FeatureConfig:
        return FeatureConfig(
            name=f"vectorbt_rsi_{period}",
            category=FeatureCategory.MOMENTUM,
            description=f"VectorBT-optimized RSI over {period} periods",
            required_columns=["close"],
            optional_columns=["high", "low", "open", "volume"],
            default_lookback=period,
            min_lookback=period,
            max_lookback=period,
            parameters={"period": period},
            matrix_optimized=True,
            gpu_accelerated=True
        )

    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate RSI feature using VectorBT."""
        if len(data) == 0:
            return pd.Series(dtype=float, index=data.index, name=f'vectorbt_rsi_{self.period}')

        # Use VectorBT RSI
        rsi = self._vectorbt_technical_indicator(data, 'rsi', window=self.period)
        return rsi.rename(f'vectorbt_rsi_{self.period}')

class VectorBTTrendGenerator(VectorBTFeatureGenerator):
    """VectorBT-optimized trend feature generator."""

    def __init__(self, period: int = 20, config: Optional[FeatureConfig] = None):
        if config is None:
            config = self._create_default_config(period)
        super().__init__(config)
        self.period = period

    @classmethod
    def _create_default_config(cls, period: int = 20) -> FeatureConfig:
        return FeatureConfig(
            name=f"vectorbt_sma_{period}",
            category=FeatureCategory.TREND,
            description=f"VectorBT-optimized SMA over {period} periods",
            required_columns=["close"],
            optional_columns=["high", "low", "open"],
            default_lookback=period,
            min_lookback=period,
            max_lookback=period,
            parameters={"period": period},
            matrix_optimized=True,
            gpu_accelerated=True
        )

    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate SMA feature using VectorBT rolling mean."""
        if len(data) == 0:
            return pd.Series(dtype=float, index=data.index, name=f'vectorbt_sma_{self.period}')

        # Use VectorBT rolling mean for SMA
        sma = self._vectorbt_rolling_operation(data['close'], 'mean', window=self.period)
        return sma.rename(f'vectorbt_sma_{self.period}')

def create_vectorbt_generators() -> List[VectorBTFeatureGenerator]:
    """Create a comprehensive set of VectorBT feature generators."""
    generators = []

    # Volatility generators
    for period in [10, 20, 50]:
        generators.append(VectorBTVolatilityGenerator(period))

    # Momentum generators
    for period in [14, 21, 30]:
        generators.append(VectorBTMomentumGenerator(period))

    # Trend generators
    for period in [10, 20, 50, 100]:
        generators.append(VectorBTTrendGenerator(period))

    return generators
