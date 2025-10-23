"""
Trend Feature Generator

This module provides feature generators for trend-based indicators,
including moving averages, trend lines, and trend strength measures.
Supports different base calculations: price returns, returns-based VWAP, etc.

Enhanced with VectorBT for maximum performance.
"""

import numpy as np
import pandas as pd
import warnings
import logging
from typing import Any, Dict, List, Optional, Union

from ..core.feature_generator import FeatureGenerator, FeatureResult, VectorizedFeatureGenerator, FeatureConfig, FeatureCategory
from ..core.vectorbt_feature_generator import VectorBTFeatureGenerator
from ..core.vectorbt_optimization_mixin import VectorBTOptimizationMixin

# Import hardware optimization decorators
from src.utils.hardware import (
    memory_optimized, gc_optimized, auto_optimize, performance_tracked,
    MemoryOptimizationLevel, WorkloadType
)

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

# VectorBT Rolling Optimizer - NOW USING NEW OPTIMIZED VERSION
try:
    from src.feature_generation.utils.consolidated_rolling_optimizer import (
        ConsolidatedRollingOptimizer as VectorBTRollingOptimizer,
        get_global_rolling_optimizer as get_vectorbt_rolling_optimizer,
        RollingOperationConfig,
        RollingOperationType
    )
    from src.feature_generation.utils.statistical_calculations_optimizer import (
        StatisticalCalculationsOptimizer as VectorizationOptimizer,
        get_global_statistical_optimizer as get_vectorization_optimizer,
        StatisticalOperationConfig,
        StatisticalOperationType
    )
    ROLLING_OPTIMIZER_AVAILABLE = True
    OPTIMIZATION_AVAILABLE = True
except ImportError:
    # Fallback to legacy if new version not available
    try:
        from src.feature_generation.utils.vectorbt_rolling_optimizer import get_vectorbt_rolling_optimizer, VectorBTRollingOptimizer
        ROLLING_OPTIMIZER_AVAILABLE = True
        OPTIMIZATION_AVAILABLE = False
    except ImportError:
        ROLLING_OPTIMIZER_AVAILABLE = False
        OPTIMIZATION_AVAILABLE = False
        get_vectorbt_rolling_optimizer = None
        VectorBTRollingOptimizer = None

# Unified Vectorization Manager
try:
    from ...utils.ml_common.unified_vectorization_manager import get_unified_vectorization_manager, UnifiedVectorizationManager, OperationType, OperationConfig
    UNIFIED_MANAGER_AVAILABLE = True
except ImportError:
    UNIFIED_MANAGER_AVAILABLE = False
    get_unified_vectorization_manager = None
    UnifiedVectorizationManager = None
    OperationType = None
    OperationConfig = None

# Optimization utilities
try:
    from src.feature_generation.utils.vectorization_optimizer import get_vectorization_optimizer
    from src.feature_generation.utils.optimized_feature_pipeline import get_optimized_feature_pipeline
    OPTIMIZATION_AVAILABLE = True
except ImportError:
    OPTIMIZATION_AVAILABLE = False

except ImportError:

    cp = None

from ..base_calculations import (
    BaseCalculator,
    BaseCalculationType,
    BaseCalculationConfig,
    create_base_calculator
)

logger = logging.getLogger(__name__)

class TrendFeatureGenerator(VectorizedFeatureGenerator, VectorBTOptimizationMixin):
    """Feature generator for trend-based features with VectorBT optimization."""

    def __init__(self, config: Optional[FeatureConfig] = None):
        if config is None:
            config = self._create_default_config()
        super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)

        # Initialize VectorBT optimizer
        if ROLLING_OPTIMIZER_AVAILABLE:
            self.vectorbt_rolling_optimizer = get_vectorbt_rolling_optimizer(enable_gpu=False, enable_parallel=True)
        else:
            self.vectorbt_rolling_optimizer = None

        # Initialize Unified Vectorization Manager
        if UNIFIED_MANAGER_AVAILABLE:
            self.unified_manager = get_unified_vectorization_manager()
        else:
            self.unified_manager = None

    @classmethod
    def _create_default_config(cls) -> FeatureConfig:
        return FeatureConfig(
            name="trend_features",
            category=FeatureCategory.TREND,
            description="Comprehensive trend-based features including moving averages and trend indicators",
            required_columns=["close"],
            optional_columns=["high", "low", "open"],
            default_lookback=20,
            min_lookback=2,
            max_lookback=50,
            parameters={
                "sma_periods": [5, 10, 20, 50],
                "ema_periods": [12, 26],
                "trend_windows": [10, 20]
            },
            matrix_optimized=True,
            gpu_accelerated=False
        )

    @classmethod
    def create_default(cls) -> 'TrendFeatureGenerator':
        return cls()

    @memory_optimized(optimization_level=MemoryOptimizationLevel.AGGRESSIVE)
    @performance_tracked
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        # Optimize DataFrame for processing using Unified Vectorization Manager
        if self.unified_manager:
            data = self.unified_manager.optimize_dataframe(data)

        close_prices = data['close']

        # Use Unified Vectorization Manager for optimized rolling operations
        if self.unified_manager and self._should_use_vectorbt(close_prices):
            try:
                # Use the unified manager for rolling operations with automatic optimization
                sma = self.unified_manager.rolling_operation(close_prices, 'mean', 20)
                return sma.rename('sma_20')
            except Exception as e:
                self.logger.warning(f"Unified Vectorization Manager SMA calculation failed: {e}, using VectorBT fallback")
                if self.vectorbt_optimizer:
                    sma = self.vectorbt_rolling_optimizer.rolling_mean(close_prices, 20)
                    return sma.rename('sma_20')
                else:
                    sma = self._calculate_sma(close_prices.values, period=20)
                    return pd.Series(sma, index=data.index, name='sma_20')
        elif self.vectorbt_optimizer and self._should_use_vectorbt(close_prices):
            try:
                sma = self.vectorbt_rolling_optimizer.rolling_mean(close_prices, 20)
                return sma.rename('sma_20')
            except Exception as e:
                self.logger.warning(f"VectorBT SMA calculation failed: {e}, using pandas fallback")
                sma = self._calculate_sma(close_prices.values, period=20)
                return pd.Series(sma, index=data.index, name='sma_20')
        else:
            sma = self._calculate_sma(close_prices.values, period=20)
            return pd.Series(sma, index=data.index, name='sma_20')

    def _calculate_sma(self, prices: np.ndarray, period: int = 20) -> np.ndarray:
        if len(prices) < period:
            return np.full(len(prices), np.nan)

        prices_series = pd.Series(prices)

        # Use VectorBT if available and data is large enough
        if self.vectorbt_optimizer and self._should_use_vectorbt(prices_series):
            try:
                sma = self.vectorbt_rolling_optimizer.rolling_mean(prices_series, period)
                return sma.values
            except Exception as e:
                self.logger.warning(f"VectorBT SMA calculation failed: {e}, using pandas fallback")
                self.performance_stats['pandas_fallbacks'] += 1
                return self._optimized_rolling_operation(prices_series, "mean", period).values
        else:
            return self._calculate_sma_vectorized(prices_series, period).values

    def _calculate_ema(self, prices: np.ndarray, period: int = 20) -> np.ndarray:
        """Calculate Exponential Moving Average."""
        if len(prices) < period:
            return np.full(len(prices), np.nan)

        ema = pd.Series(prices).ewm(span=period).mean().values
        return ema

    def _calculate_adx(self, high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
        """
        Calculate Average Directional Index (ADX) with optimized VectorBT operations.

        Args:
            high: High prices
            low: Low prices
            close: Close prices
            period: ADX period (default 14)

        Returns:
            ADX values
        """
        if len(high) < period or len(low) < period or len(close) < period:
            return np.full(len(close), np.nan)

        # Calculate True Range
        tr = np.maximum.reduce([
            high - low,
            np.abs(high - np.roll(close, 1)),
            np.abs(low - np.roll(close, 1))
        ])
        tr[0] = np.nan  # First value is NaN

        # Calculate Directional Movement
        dm_plus = np.maximum(high - np.roll(high, 1), 0)
        dm_minus = np.maximum(np.roll(low, 1) - low, 0)

        # Convert to pandas Series for rolling operations
        tr_series = pd.Series(tr)
        dm_plus_series = pd.Series(dm_plus)
        dm_minus_series = pd.Series(dm_minus)

        # Use Unified Vectorization Manager for optimized rolling operations
        if self.unified_manager and self._should_use_vectorbt(tr_series):
            try:
                # Use Unified Vectorization Manager for batch rolling operations
                batch_data = pd.DataFrame({
                    'dm_plus': dm_plus_series,
                    'dm_minus': dm_minus_series,
                    'tr': tr_series
                })

                batch_result = self.unified_manager.optimize_operation(
                    OperationType.TECHNICAL_INDICATORS,
                    {
                        'data': batch_data,
                        'operation': 'batch_rolling_mean',
                        'window': period,
                        'columns': ['dm_plus', 'dm_minus', 'tr']
                    },
                    OperationConfig(
                        operation_type=OperationType.TECHNICAL_INDICATORS,
                        data_size=len(batch_data),
                        data_dimensions=batch_data.shape,
                        memory_budget_mb=256.0
                    )
                )

                rolling_results = batch_result.result
                dm_plus_mean = rolling_results['dm_plus']
                dm_minus_mean = rolling_results['dm_minus']
                tr_mean = rolling_results['tr']
            except Exception as e:
                logger.warning(f"Unified Vectorization Manager ADX calculation failed: {e}, using VectorBT fallback")
                if self.vectorbt_optimizer and self._should_use_vectorbt(tr_series):
                    try:
                        dm_plus_mean = self.vectorbt_rolling_optimizer.rolling_mean(dm_plus_series, period)
                        dm_minus_mean = self.vectorbt_rolling_optimizer.rolling_mean(dm_minus_series, period)
                        tr_mean = self.vectorbt_rolling_optimizer.rolling_mean(tr_series, period)
                    except Exception as e2:
                        logger.warning(f"VectorBT ADX calculation failed: {e2}, using pandas fallback")
                        dm_plus_mean = dm_plus_series.rolling(period).mean()
                        dm_minus_mean = dm_minus_series.rolling(period).mean()
                        tr_mean = tr_series.rolling(period).mean()
                else:
                    dm_plus_mean = dm_plus_series.rolling(period).mean()
                    dm_minus_mean = dm_minus_series.rolling(period).mean()
                    tr_mean = tr_series.rolling(period).mean()
        elif self.vectorbt_optimizer and self._should_use_vectorbt(tr_series):
            try:
                dm_plus_mean = self.vectorbt_rolling_optimizer.rolling_mean(dm_plus_series, period)
                dm_minus_mean = self.vectorbt_rolling_optimizer.rolling_mean(dm_minus_series, period)
                tr_mean = self.vectorbt_rolling_optimizer.rolling_mean(tr_series, period)
            except Exception as e:
                logger.warning(f"VectorBT ADX calculation failed: {e}, using pandas fallback")
                dm_plus_mean = dm_plus_series.rolling(period).mean()
                dm_minus_mean = dm_minus_series.rolling(period).mean()
                tr_mean = tr_series.rolling(period).mean()
        else:
            dm_plus_mean = dm_plus_series.rolling(period).mean()
            dm_minus_mean = dm_minus_series.rolling(period).mean()
            tr_mean = tr_series.rolling(period).mean()

        di_plus = 100 * (dm_plus_mean / tr_mean)
        di_minus = 100 * (dm_minus_mean / tr_mean)

        # Calculate ADX
        dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)

        # Use optimized rolling mean for final ADX calculation
        if self.unified_manager and self._should_use_vectorbt(pd.Series(dx)):
            try:
                adx = self.unified_manager.rolling_operation(pd.Series(dx), 'mean', period)
            except Exception as e:
                logger.warning(f"Unified Vectorization Manager ADX rolling mean failed: {e}, using VectorBT fallback")
                if self.vectorbt_optimizer:
                    adx = self.vectorbt_rolling_optimizer.rolling_mean(pd.Series(dx), period)
                else:
                    adx = pd.Series(dx).rolling(period).mean()
        elif self.vectorbt_optimizer and self._should_use_vectorbt(pd.Series(dx)):
            try:
                adx = self.vectorbt_rolling_optimizer.rolling_mean(pd.Series(dx), period)
            except Exception as e:
                logger.warning(f"VectorBT ADX rolling mean failed: {e}, using pandas fallback")
                adx = pd.Series(dx).rolling(period).mean()
        else:
            adx = pd.Series(dx).rolling(period).mean()

        return adx.values

    def _calculate_directional_signal(self, prices: np.ndarray) -> np.ndarray:
        """
        Calculate directional signal as EMA_8 - EMA_20.

        Args:
            prices: Price data

        Returns:
            Directional signal values
        """
        ema_8 = self._calculate_ema(prices, period=8)
        ema_20 = self._calculate_ema(prices, period=20)

        # Calculate directional signal
        directional_signal = ema_8 - ema_20

        return directional_signal

    def _calculate_trend_score(self, prices: np.ndarray, high: np.ndarray, low: np.ndarray, close: np.ndarray) -> np.ndarray:
        """
        Calculate trend score as normalized directional signal multiplied by ADX.

        Args:
            prices: Price data
            high: High prices
            low: Low prices
            close: Close prices

        Returns:
            Trend score values
        """
        # Calculate directional signal
        directional_signal = self._calculate_directional_signal(prices)

        # Calculate ADX
        adx = self._calculate_adx(high, low, close, period=14)

        # Normalize directional signal to [-1, 1] range
        signal_max = np.nanmax(np.abs(directional_signal))
        if signal_max > 0:
            normalized_signal = directional_signal / signal_max
        else:
            normalized_signal = directional_signal

        # Calculate trend score
        trend_score = normalized_signal * adx

        return trend_score

    def _should_use_vectorbt(self, data) -> bool:
        """Determine if VectorBT should be used based on data size and configuration."""
        return (VECTORBT_AVAILABLE and
                len(data) >= getattr(self, 'vectorbt_threshold', 1000))

    @memory_optimized(optimization_level=MemoryOptimizationLevel.AGGRESSIVE)
    @performance_tracked
    def generate_optimized_trend_features(self, data: pd.DataFrame,
                                        feature_configs: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        Generate multiple trend features using optimized batch processing.

        Args:
            data: OHLCV data
            feature_configs: List of feature configuration dictionaries

        Returns:
            DataFrame with generated trend features
        """
        if self.unified_manager:
            try:
                # Use Unified Vectorization Manager for batch processing
                batch_result = self.unified_manager.optimize_operation(
                    OperationType.FEATURE_ENGINEERING,
                    {
                        'data': data,
                        'feature_configs': feature_configs,
                        'operation_type': 'trend_batch'
                    },
                    OperationConfig(
                        operation_type=OperationType.FEATURE_ENGINEERING,
                        data_size=len(data),
                        data_dimensions=data.shape,
                        memory_budget_mb=1024.0
                    )
                )
                return batch_result.result
            except Exception as e:
                logger.warning(f"Unified Vectorization Manager batch processing failed: {e}, using fallback")
                # Fallback to individual processing
                return self._process_trend_features_individually(data, feature_configs)
        elif self.vectorbt_optimizer:
            # Fallback to individual VectorBT operations
            results = {}
            for config in feature_configs:
                feature_name = config['name']
                feature_type = config.get('type', 'rolling')
                params = config.get('params', {})

                try:
                    if feature_type == 'rolling':
                        operation = params.get('operation', 'mean')
                        window = params.get('window', 20)
                        column = params.get('column', 'close')

                        if column in data.columns:
                            if operation == 'mean':
                                results[feature_name] = self.vectorbt_rolling_optimizer.rolling_mean(data[column], window)
                            elif operation == 'std':
                                results[feature_name] = self.vectorbt_rolling_optimizer.rolling_std(data[column], window)
                            elif operation == 'var':
                                results[feature_name] = self.vectorbt_rolling_optimizer.rolling_var(data[column], window)
                            elif operation == 'min':
                                results[feature_name] = self.vectorbt_rolling_optimizer.rolling_min(data[column], window)
                            elif operation == 'max':
                                results[feature_name] = self.vectorbt_rolling_optimizer.rolling_max(data[column], window)
                            elif operation == 'sum':
                                results[feature_name] = self.vectorbt_rolling_optimizer.rolling_sum(data[column], window)

                    elif feature_type == 'scaling':
                        method = params.get('method', 'zscore')
                        column = params.get('column', 'close')

                        if column in data.columns:
                            if method == 'zscore':
                                results[feature_name] = self.vectorbt_rolling_optimizer.rolling_apply(
                                    data[column], lambda x: (x - x.mean()) / x.std(), 20
                                )
                            elif method == 'minmax':
                                results[feature_name] = self.vectorbt_rolling_optimizer.rolling_apply(
                                    data[column], lambda x: (x - x.min()) / (x.max() - x.min()), 20
                                )

                except Exception as e:
                    self.logger.warning(f"Feature {feature_name} failed: {e}")
                    results[feature_name] = pd.Series(np.nan, index=data.index)

            return pd.DataFrame(results, index=data.index)
        else:
            # Fallback to pandas operations
            results = {}
            for config in feature_configs:
                feature_name = config['name']
                feature_type = config.get('type', 'rolling')
                params = config.get('params', {})

                try:
                    if feature_type == 'rolling':
                        operation = params.get('operation', 'mean')
                        window = params.get('window', 20)
                        column = params.get('column', 'close')

                        if column in data.columns:
                            rolling_obj = data[column].rolling(window)
                            if operation == 'mean':
                                results[feature_name] = rolling_obj.mean()
                            elif operation == 'std':
                                results[feature_name] = rolling_obj.std()
                            elif operation == 'var':
                                results[feature_name] = rolling_obj.var()
                            elif operation == 'min':
                                results[feature_name] = rolling_obj.min()
                            elif operation == 'max':
                                results[feature_name] = rolling_obj.max()
                            elif operation == 'sum':
                                results[feature_name] = rolling_obj.sum()

                except Exception as e:
                    self.logger.warning(f"Feature {feature_name} failed: {e}")
                    results[feature_name] = pd.Series(np.nan, index=data.index)

            return pd.DataFrame(results, index=data.index)

    def _process_trend_features_individually(self, data: pd.DataFrame, feature_configs: List[Dict[str, Any]]) -> pd.DataFrame:
        """Process trend features individually as fallback when batch processing fails."""
        results = {}

        # Group features by operation type for better optimization
        rolling_features = []
        scaling_features = []
        custom_features = []

        for config in feature_configs:
            feature_type = config.get('type', 'rolling')
            if feature_type == 'rolling':
                rolling_features.append(config)
            elif feature_type == 'scaling':
                scaling_features.append(config)
            else:
                custom_features.append(config)

        # Process rolling features in batch using VectorBTRollingOptimizer
        if rolling_features and self.vectorbt_optimizer:
            results.update(self._process_rolling_features_batch(data, rolling_features))

        # Process scaling features using UnifiedVectorizationManager
        if scaling_features and self.unified_manager:
            results.update(self._process_scaling_features_batch(data, scaling_features))

        # Process custom features individually
        for config in custom_features:
            feature_name = config['name']
            try:
                results[feature_name] = self._process_custom_feature(data, config)
            except Exception as e:
                self.logger.warning(f"Custom feature {feature_name} failed: {e}")
                results[feature_name] = pd.Series(np.nan, index=data.index)

        return pd.DataFrame(results, index=data.index)

    def _process_rolling_features_batch(self, data: pd.DataFrame, rolling_configs: List[Dict[str, Any]]) -> Dict[str, pd.Series]:
        """Process rolling features in batch using VectorBTRollingOptimizer for better performance."""
        results = {}

        # Group by column and window for batch processing
        column_groups = {}
        for config in rolling_configs:
            column = config['params'].get('column', 'close')
            window = config['params'].get('window', self.period)
            operation = config['params'].get('operation', 'mean')

            if column not in column_groups:
                column_groups[column] = {}
            if window not in column_groups[column]:
                column_groups[column][window] = []

            column_groups[column][window].append((config['name'], operation))

        # Process each column-window combination
        for column, window_groups in column_groups.items():
            if column not in data.columns:
                continue

            series_data = data[column]

            for window, operations in window_groups.items():
                try:
                    # Use VectorBTRollingOptimizer for batch operations
                    for feature_name, operation in operations:
                        if operation == 'mean':
                            results[feature_name] = self.vectorbt_rolling_optimizer.rolling_mean(series_data, window)
                        elif operation == 'std':
                            results[feature_name] = self.vectorbt_rolling_optimizer.rolling_std(series_data, window)
                        elif operation == 'var':
                            results[feature_name] = self.vectorbt_rolling_optimizer.rolling_var(series_data, window)
                        elif operation == 'min':
                            results[feature_name] = self.vectorbt_rolling_optimizer.rolling_min(series_data, window)
                        elif operation == 'max':
                            results[feature_name] = self.vectorbt_rolling_optimizer.rolling_max(series_data, window)
                        elif operation == 'sum':
                            results[feature_name] = self.vectorbt_rolling_optimizer.rolling_sum(series_data, window)
                        elif operation == 'quantile':
                            q = config['params'].get('q', 0.5)
                            results[feature_name] = self.vectorbt_rolling_optimizer.rolling_quantile(series_data, window, q=q)
                        elif operation == 'skew':
                            results[feature_name] = self.vectorbt_rolling_optimizer.rolling_skew(series_data, window)
                        elif operation == 'kurt':
                            results[feature_name] = self.vectorbt_rolling_optimizer.rolling_kurt(series_data, window)
                        elif operation == 'corr':
                            other_column = config['params'].get('other_column')
                            if other_column and other_column in data.columns:
                                results[feature_name] = self.vectorbt_rolling_optimizer.rolling_corr(
                                    series_data, data[other_column], window
                                )
                            else:
                                results[feature_name] = pd.Series(np.nan, index=data.index)
                        elif operation == 'cov':
                            other_column = config['params'].get('other_column')
                            if other_column and other_column in data.columns:
                                results[feature_name] = self.vectorbt_rolling_optimizer.rolling_cov(
                                    series_data, data[other_column], window
                                )
                            else:
                                results[feature_name] = pd.Series(np.nan, index=data.index)
                        else:
                            # Fallback to pandas for unsupported operations
                            rolling_obj = series_data.rolling(window)
                            if hasattr(rolling_obj, operation):
                                results[feature_name] = getattr(rolling_obj, operation)()
                            else:
                                results[feature_name] = pd.Series(np.nan, index=data.index)

                except Exception as e:
                    self.logger.warning(f"Batch rolling operation failed for {column} window {window}: {e}")
                    # Fallback to individual processing
                    for feature_name, operation in operations:
                        try:
                            rolling_obj = series_data.rolling(window)
                            if hasattr(rolling_obj, operation):
                                results[feature_name] = getattr(rolling_obj, operation)()
                            else:
                                results[feature_name] = pd.Series(np.nan, index=data.index)
                        except Exception as e2:
                            self.logger.warning(f"Fallback rolling operation failed for {feature_name}: {e2}")
                            results[feature_name] = pd.Series(np.nan, index=data.index)

        return results

    def _process_scaling_features_batch(self, data: pd.DataFrame, scaling_configs: List[Dict[str, Any]]) -> Dict[str, pd.Series]:
        """Process scaling features in batch using UnifiedVectorizationManager."""
        results = {}

        for config in scaling_configs:
            feature_name = config['name']
            method = config['params'].get('method', 'zscore')
            column = config['params'].get('column', 'close')

            if column not in data.columns:
                results[feature_name] = pd.Series(np.nan, index=data.index)
                continue

            try:
                # Use UnifiedVectorizationManager for scaling
                results[feature_name] = self.unified_manager.scale_data(
                    data[column], method=method, **config['params']
                )
            except Exception as e:
                self.logger.warning(f"Scaling feature {feature_name} failed: {e}")
                # Fallback to manual scaling
                if method == 'zscore':
                    results[feature_name] = (data[column] - data[column].mean()) / data[column].std()
                elif method == 'minmax':
                    results[feature_name] = (data[column] - data[column].min()) / (data[column].max() - data[column].min())
                else:
                    results[feature_name] = pd.Series(np.nan, index=data.index)

        return results

    def _process_custom_feature(self, data: pd.DataFrame, config: Dict[str, Any]) -> pd.Series:
        """Process custom features that don't fit standard patterns."""
        feature_name = config['name']
        feature_type = config.get('type', 'custom')
        params = config.get('params', {})

        if feature_type == 'custom' and 'function' in params:
            # Execute custom function
            func = params['function']
            if callable(func):
                return func(data, **params)
            else:
                raise ValueError(f"Custom function for {feature_name} is not callable")
        else:
            raise ValueError(f"Unsupported custom feature type: {feature_type}")

    def generate_moving_averages_batch(self, data: pd.DataFrame, windows: List[int],
                                     columns: List[str] = None, operation: str = 'mean') -> pd.DataFrame:
        """
        Generate multiple moving averages in batch using VectorBTRollingOptimizer.

        Args:
            data: Input DataFrame with OHLCV data
            windows: List of window sizes for moving averages
            columns: List of columns to calculate moving averages for (default: ['close'])
            operation: Rolling operation type ('mean', 'std', 'var', 'min', 'max', 'sum')

        Returns:
            DataFrame with moving average features
        """
        if columns is None:
            columns = ['close']

        # Validate columns exist in data
        valid_columns = [col for col in columns if col in data.columns]
        if not valid_columns:
            raise ValueError(f"None of the specified columns {columns} found in data")

        # Create feature configurations for batch processing
        feature_configs = []
        for window in windows:
            for column in valid_columns:
                feature_configs.append({
                    'name': f'{operation}_{column}_{window}',
                    'type': 'rolling',
                    'params': {
                        'operation': operation,
                        'window': window,
                        'column': column
                    }
                })

        # Use batch processing if available
        if self.unified_manager:
            try:
                return self.unified_manager.batch_process_features(data, feature_configs)
            except Exception as e:
                self.logger.warning(f"Unified manager batch processing failed: {e}, using individual processing")

        # Fallback to individual processing
        return self._process_trend_features_individually(data, feature_configs)

    def generate_trend_indicators_batch(self, data: pd.DataFrame,
                                      sma_windows: List[int] = None,
                                      ema_windows: List[int] = None,
                                      adx_periods: List[int] = None) -> pd.DataFrame:
        """
        Generate comprehensive trend indicators in batch.

        Args:
            data: Input DataFrame with OHLCV data
            sma_windows: List of SMA window sizes (default: [5, 10, 20, 50])
            ema_windows: List of EMA window sizes (default: [12, 26])
            adx_periods: List of ADX periods (default: [14])

        Returns:
            DataFrame with trend indicator features
        """
        if sma_windows is None:
            sma_windows = [5, 10, 20, 50]
        if ema_windows is None:
            ema_windows = [12, 26]
        if adx_periods is None:
            adx_periods = [14]

        feature_configs = []

        # Add SMA features
        for window in sma_windows:
            feature_configs.append({
                'name': f'sma_{window}',
                'type': 'rolling',
                'params': {'operation': 'mean', 'window': window, 'column': 'close'}
            })

        # Add EMA features (custom implementation)
        for window in ema_windows:
            feature_configs.append({
                'name': f'ema_{window}',
                'type': 'custom',
                'params': {
                    'function': lambda df, w=window: self._calculate_ema(df['close'].values, w),
                    'window': window
                }
            })

        # Add ADX features (custom implementation)
        for period in adx_periods:
            feature_configs.append({
                'name': f'adx_{period}',
                'type': 'custom',
                'params': {
                    'function': lambda df, p=period: self._calculate_adx(
                        df['high'].values, df['low'].values, df['close'].values, p
                    ),
                    'period': period
                }
            })

        # Process all features
        if self.unified_manager:
            try:
                return self.unified_manager.batch_process_features(data, feature_configs)
            except Exception as e:
                self.logger.warning(f"Unified manager batch processing failed: {e}, using individual processing")

        return self._process_trend_features_individually(data, feature_configs)

    def _vectorbt_rolling_operation(self, data: pd.Series, operation: str,
                                  window: int, **kwargs) -> pd.Series:
        """Perform VectorBT rolling operation with fallback to pandas."""
        if not self._should_use_vectorbt(data):
            return self._pandas_rolling_operation(data, operation, window, **kwargs)

        try:
            if operation == 'mean':
                return self.vectorbt_rolling_optimizer.rolling_mean(data, window, **kwargs)
            elif operation == 'std':
                return self.vectorbt_rolling_optimizer.rolling_std(data, window, **kwargs)
            elif operation == 'var':
                return self.vectorbt_rolling_optimizer.rolling_var(data, window, **kwargs)
            elif operation == 'min':
                return self.vectorbt_rolling_optimizer.rolling_min(data, window, **kwargs)
            elif operation == 'max':
                return self.vectorbt_rolling_optimizer.rolling_max(data, window, **kwargs)
            elif operation == 'sum':
                return self.vectorbt_rolling_optimizer.rolling_sum(data, window, **kwargs)
            else:
                raise ValueError(f"Unsupported operation: {operation}")
        except Exception as e:
            self.logger.warning(f"VectorBT operation failed: {e}, using pandas fallback")
            return self._pandas_rolling_operation(data, operation, window, **kwargs)

    def _pandas_rolling_operation(self, data: pd.Series, operation: str,
                                 window: int, **kwargs) -> pd.Series:
        """Fallback rolling operation using pandas."""
        if operation == 'mean':
            return self._calculate_sma_vectorized(data, window)
        elif operation == 'std':
            return self._calculate_rolling_std_vectorized(data, window)
        elif operation == 'var':
            return self._optimized_rolling_operation(data, "var", window)
        elif operation == 'min':
            return self._calculate_rolling_min_vectorized(data, window)
        elif operation == 'max':
            return self._calculate_rolling_max_vectorized(data, window)
        elif operation == 'sum':
            return self._calculate_rolling_sum_vectorized(data, window)
        else:
            raise ValueError(f"Unsupported operation: {operation}")

class ADXGenerator(VectorizedFeatureGenerator, VectorBTOptimizationMixin):
    """Generator for Average Directional Index (ADX) with VectorBT optimization."""

    def __init__(self, period: int = 14):
        """
        Initialize ADX generator.

        Args:
            period: ADX period (default 14)
        """
        config = FeatureConfig(
            name=f"adx_{period}",
            category=FeatureCategory.TREND,
            description=f"Average Directional Index over {period} periods",
            required_columns=["high", "low", "close"],
            default_lookback=period * 2,  # Need more data for ADX calculation
            min_lookback=period * 2,
            max_lookback=period * 2,
            parameters={
                'period': period
            }
        )
        super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)
        self.period = period

        # Initialize VectorBT optimizer
        if ROLLING_OPTIMIZER_AVAILABLE:
            self.vectorbt_rolling_optimizer = get_vectorbt_rolling_optimizer(enable_gpu=False, enable_parallel=True)
        else:
            self.vectorbt_rolling_optimizer = None

    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        # Optimize DataFrame for processing
        if hasattr(self, 'optimize_dataframe_processing'):
            data = self.optimize_dataframe_processing(data)

        high = data['high']
        low = data['low']
        close = data['close']

        # Use VectorBT for ADX calculation
        if self.vectorbt_optimizer and self._should_use_vectorbt(close):
            try:
                # VectorBT doesn't have direct ADX, so we use our custom calculation
                adx = self._calculate_adx(high.values, low.values, close.values, period=self.period)
                return pd.Series(adx, index=data.index, name=f'adx_{self.period}')
            except Exception as e:
                self.logger.warning(f"VectorBT ADX calculation failed: {e}, using pandas fallback")
                adx = self._calculate_adx(high.values, low.values, close.values, period=self.period)
                return pd.Series(adx, index=data.index, name=f'adx_{self.period}')
        else:
            adx = self._calculate_adx(high.values, low.values, close.values, period=self.period)
            return pd.Series(adx, index=data.index, name=f'adx_{self.period}')

    def _calculate_adx(self, high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
        """Calculate Average Directional Index (ADX)."""
        if len(high) < period or len(low) < period or len(close) < period:
            return np.full(len(close), np.nan)

        # Calculate True Range
        tr = np.maximum.reduce([
            high - low,
            np.abs(high - np.roll(close, 1)),
            np.abs(low - np.roll(close, 1))
        ])
        tr[0] = np.nan  # First value is NaN

        # Calculate Directional Movement
        dm_plus = np.maximum(high - np.roll(high, 1), 0)
        dm_minus = np.maximum(np.roll(low, 1) - low, 0)

        # Convert to pandas Series for rolling operations
        dm_plus_series = pd.Series(dm_plus)
        dm_minus_series = pd.Series(dm_minus)
        tr_series = pd.Series(tr)

        # Calculate Directional Indicators using VectorBT optimization
        if self.vectorbt_optimizer and self._should_use_vectorbt(tr_series):
            try:
                dm_plus_mean = self.vectorbt_rolling_optimizer.rolling_mean(dm_plus_series, period)
                dm_minus_mean = self.vectorbt_rolling_optimizer.rolling_mean(dm_minus_series, period)
                tr_mean = self.vectorbt_rolling_optimizer.rolling_mean(tr_series, period)
            except Exception as e:
                logger.warning(f"VectorBT ADX calculation failed: {e}, using pandas fallback")
                dm_plus_mean = dm_plus_series.rolling(period).mean()
                dm_minus_mean = dm_minus_series.rolling(period).mean()
                tr_mean = tr_series.rolling(period).mean()
        else:
            dm_plus_mean = dm_plus_series.rolling(period).mean()
            dm_minus_mean = dm_minus_series.rolling(period).mean()
            tr_mean = tr_series.rolling(period).mean()

        # Calculate Directional Indicators
        di_plus = 100 * (dm_plus_mean / tr_mean)
        di_minus = 100 * (dm_minus_mean / tr_mean)

        # Calculate ADX
        dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)

        if self.vectorbt_optimizer and self._should_use_vectorbt(pd.Series(dx)):
            try:
                adx = self.vectorbt_rolling_optimizer.rolling_mean(pd.Series(dx), period)
            except Exception as e:
                logger.warning(f"VectorBT ADX rolling mean failed: {e}, using pandas fallback")
                adx = pd.Series(dx).rolling(period).mean()
        else:
            adx = pd.Series(dx).rolling(period).mean()

        return adx.values

class DirectionalSignalGenerator(VectorizedFeatureGenerator, VectorBTOptimizationMixin):
    """Generator for Directional Signal (EMA_8 - EMA_20) with VectorBT optimization."""
# ADXGenerator moved to oscillator.py to avoid duplication
# Import from oscillator module instead
from .oscillator import ADXGenerator
# Centralized utility imports
from src.feature_generation.utils.vectorbt_rolling_optimizer import get_vectorbt_rolling_optimizer
# Removed VectorBTScaler import to avoid circular import
# Lazy import to avoid circular dependency
def get_global_feature_bank():
    from ..core.feature_bank import get_global_feature_bank as _get_global_feature_bank
    return _get_global_feature_bank()
class DirectionalSignalGenerator(VectorizedFeatureGenerator):
    """Generator for Directional Signal (EMA_8 - EMA_20)."""

    def __init__(self):
        """Initialize Directional Signal generator."""
        config = FeatureConfig(
            name="directional_signal",
            category=FeatureCategory.TREND,
            description="Directional signal calculated as EMA_8 - EMA_20",
            required_columns=["close"],
            default_lookback=20,  # Need enough data for both EMAs
            min_lookback=20,
            max_lookback=20,
            parameters={}
        )
        super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)

        # Initialize VectorBT optimizer
        if ROLLING_OPTIMIZER_AVAILABLE:
            self.vectorbt_rolling_optimizer = get_vectorbt_rolling_optimizer(enable_gpu=False, enable_parallel=True)
        else:
            self.vectorbt_rolling_optimizer = None

    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        # Optimize DataFrame for processing
        if hasattr(self, 'optimize_dataframe_processing'):
            data = self.optimize_dataframe_processing(data)

        prices = data['close']

        # Use VectorBT for directional signal calculation
        if self.vectorbt_optimizer and self._should_use_vectorbt(prices):
            try:
                # Calculate EMA using VectorBT
                ema_8 = self._calculate_ema_vectorized(prices, 8)
                ema_20 = self._calculate_ema_vectorized(prices, 20)
                directional_signal = ema_8 - ema_20
                return directional_signal.rename('directional_signal')
            except Exception as e:
                self.logger.warning(f"VectorBT directional signal calculation failed: {e}, using pandas fallback")
                directional_signal = self._calculate_directional_signal(prices.values)
                return pd.Series(directional_signal, index=data.index, name='directional_signal')
        else:
            directional_signal = self._calculate_directional_signal(prices.values)
            return pd.Series(directional_signal, index=data.index, name='directional_signal')

    def _calculate_ema(self, prices: np.ndarray, period: int = 20) -> np.ndarray:
        """Calculate Exponential Moving Average."""
        if len(prices) < period:
            return np.full(len(prices), np.nan)

        ema = pd.Series(prices).ewm(span=period).mean().values
        return ema

    def _calculate_directional_signal(self, prices: np.ndarray) -> np.ndarray:
        """Calculate directional signal as EMA_8 - EMA_20."""
        ema_8 = self._calculate_ema(prices, period=8)
        ema_20 = self._calculate_ema(prices, period=20)

        # Calculate directional signal
        directional_signal = ema_8 - ema_20

        return directional_signal

class TrendScoreGenerator(VectorizedFeatureGenerator, VectorBTOptimizationMixin):
    """Generator for Trend Score (normalized directional signal * ADX) with VectorBT optimization."""

    def __init__(self, adx_period: int = 14):
        """
        Initialize Trend Score generator.

        Args:
            adx_period: ADX period (default 14)
        """
        config = FeatureConfig(
            name=f"trend_score_{adx_period}",
            category=FeatureCategory.TREND,
            description=f"Trend score calculated as normalized directional signal multiplied by ADX ({adx_period})",
            required_columns=["close", "high", "low"],
            default_lookback=adx_period * 2,  # Need enough data for both calculations
            min_lookback=adx_period * 2,
            max_lookback=adx_period * 2,
            parameters={
                'adx_period': adx_period
            }
        )
        super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)
        self.adx_period = adx_period

        # Initialize VectorBT optimizer
        if ROLLING_OPTIMIZER_AVAILABLE:
            self.vectorbt_rolling_optimizer = get_vectorbt_rolling_optimizer(enable_gpu=False, enable_parallel=True)
        else:
            self.vectorbt_rolling_optimizer = None

    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        # Optimize DataFrame for processing
        if hasattr(self, 'optimize_dataframe_processing'):
            data = self.optimize_dataframe_processing(data)

        prices = data['close']
        high = data['high']
        low = data['low']
        close = data['close']

        # Use VectorBT for trend score calculation
        if self.vectorbt_optimizer and self._should_use_vectorbt(prices):
            try:
                # Calculate directional signal using VectorBT
                ema_8 = self._calculate_ema_vectorized(prices, 8)
                ema_20 = self._calculate_ema_vectorized(prices, 20)
                directional_signal = ema_8 - ema_20

                # Calculate ADX using our custom method
                adx = self._calculate_adx(high.values, low.values, close.values, period=self.adx_period)
                adx_series = pd.Series(adx, index=data.index)

                # Normalize directional signal to [-1, 1] range
                signal_max = np.nanmax(np.abs(directional_signal))
                if signal_max > 0:
                    normalized_signal = directional_signal / signal_max
                else:
                    normalized_signal = directional_signal

                # Calculate trend score
                trend_score = normalized_signal * adx_series
                return trend_score.rename(f'trend_score_{self.adx_period}')
            except Exception as e:
                self.logger.warning(f"VectorBT trend score calculation failed: {e}, using pandas fallback")
                trend_score = self._calculate_trend_score(prices.values, high.values, low.values, close.values)
                return pd.Series(trend_score, index=data.index, name=f'trend_score_{self.adx_period}')
        else:
            trend_score = self._calculate_trend_score(prices.values, high.values, low.values, close.values)
            return pd.Series(trend_score, index=data.index, name=f'trend_score_{self.adx_period}')

    def _calculate_ema(self, prices: np.ndarray, period: int = 20) -> np.ndarray:
        """Calculate Exponential Moving Average."""
        if len(prices) < period:
            return np.full(len(prices), np.nan)

        ema = pd.Series(prices).ewm(span=period).mean().values
        return ema

    def _calculate_adx(self, high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
        """Calculate ADX (Average Directional Index) using VectorBT optimization."""
        # Simplified ADX calculation - in production, use proper ADX implementation
        high_low_diff = high - low
        high_close_diff = np.abs(high - np.roll(close, 1))
        low_close_diff = np.abs(low - np.roll(close, 1))

        true_range = np.maximum(high_low_diff, np.maximum(high_close_diff, low_close_diff))
        true_range[0] = high_low_diff[0]  # First value

        # Calculate directional movement
        high_diff = high - np.roll(high, 1)
        low_diff = np.roll(low, 1) - low

        plus_dm = np.where((high_diff > low_diff) & (high_diff > 0), high_diff, 0)
        minus_dm = np.where((low_diff > high_diff) & (low_diff > 0), low_diff, 0)

        # Calculate smoothed values
        plus_di = 100 * (np.convolve(plus_dm, np.ones(period)/period, mode='same') /
                         np.convolve(true_range, np.ones(period)/period, mode='same'))
        minus_di = 100 * (np.convolve(minus_dm, np.ones(period)/period, mode='same') /
                          np.convolve(true_range, np.ones(period)/period, mode='same'))

        # Calculate ADX
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-8)
        adx = np.convolve(dx, np.ones(period)/period, mode='same')

        return adx

    def _calculate_directional_signal(self, prices: np.ndarray) -> np.ndarray:
        """Calculate directional signal as EMA_8 - EMA_20."""
        ema_8 = self._calculate_ema(prices, period=8)
        ema_20 = self._calculate_ema(prices, period=20)

        # Calculate directional signal
        directional_signal = ema_8 - ema_20

        return directional_signal

    def _calculate_trend_score(self, prices: np.ndarray, high: np.ndarray, low: np.ndarray, close: np.ndarray) -> np.ndarray:
        """Calculate trend score as normalized directional signal multiplied by ADX."""
        # Calculate directional signal
        directional_signal = self._calculate_directional_signal(prices)

        # Calculate ADX
        adx = self._calculate_adx(high, low, close, period=self.adx_period)

        # Normalize directional signal to [-1, 1] range
        signal_max = np.nanmax(np.abs(directional_signal))
        if signal_max > 0:
            normalized_signal = directional_signal / signal_max
        else:
            normalized_signal = directional_signal

        # Calculate trend score
        trend_score = normalized_signal * adx

        return trend_score

class SMAGenerator(VectorizedFeatureGenerator, VectorBTOptimizationMixin):
    """Generator for Simple Moving Average with different base calculations and VectorBT optimization."""

    def __init__(self,
                 period: int = 20,
                 base_calculation: Union[str, BaseCalculationType] = BaseCalculationType.RETURNS_VWAP,
                 **base_kwargs):
        """
        Initialize SMA generator.

        Args:
            period: SMA period
            base_calculation: Base calculation type (price_levels, returns_vwap, etc.)
            **base_kwargs: Additional parameters for base calculation
        """
        if isinstance(base_calculation, str):
            base_calculation = BaseCalculationType(base_calculation)

        # Create base calculator
        self.base_calculator = create_base_calculator(base_calculation, **base_kwargs)

        # Update required columns based on base calculation
        required_columns = self.base_calculator.get_required_columns()

        config = FeatureConfig(
            name=f"sma_{period}_{base_calculation.value}",
            category=FeatureCategory.TREND,
            description=f"Simple Moving Average over {period} periods based on {base_calculation.value}",
            required_columns=required_columns,
            default_lookback=period,
            min_lookback=period,
            max_lookback=period,
            parameters={
                'period': period,
                'base_calculation': base_calculation.value,
                **base_kwargs
            }
        )
        super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)
        self.period = period
        self.base_calculation = base_calculation

        # Initialize VectorBT optimizer
        if ROLLING_OPTIMIZER_AVAILABLE:
            self.vectorbt_rolling_optimizer = get_vectorbt_rolling_optimizer(enable_gpu=False, enable_parallel=True)
        else:
            self.vectorbt_rolling_optimizer = None

    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        # Optimize DataFrame for processing
        if hasattr(self, 'optimize_dataframe_processing'):
            data = self.optimize_dataframe_processing(data)

        # Calculate base values
        base_values = self.base_calculator.calculate(data)

        # Use VectorBT for SMA calculation
        if self.vectorbt_optimizer and self._should_use_vectorbt(base_values):
            try:
                sma = self.vectorbt_rolling_optimizer.rolling_mean(base_values, self.period)
                return sma
            except Exception as e:
                self.logger.warning(f"VectorBT SMA calculation failed: {e}, using pandas fallback")
                sma = base_values.rolling(self.period).mean()
                return sma
        else:
            sma = base_values.rolling(self.period).mean()
            return sma

class EMAGenerator(VectorizedFeatureGenerator, VectorBTOptimizationMixin):
    """Generator for Exponential Moving Average with different base calculations and VectorBT optimization."""

    def __init__(self,
                 period: int = 20,
                 base_calculation: Union[str, BaseCalculationType] = BaseCalculationType.RETURNS_VWAP,
                 **base_kwargs):
        """
        Initialize EMA generator.

        Args:
            period: EMA period
            base_calculation: Base calculation type (price_levels, returns_vwap, etc.)
            **base_kwargs: Additional parameters for base calculation
        """
        if isinstance(base_calculation, str):
            base_calculation = BaseCalculationType(base_calculation)

        # Create base calculator
        self.base_calculator = create_base_calculator(base_calculation, **base_kwargs)

        # Update required columns based on base calculation
        required_columns = self.base_calculator.get_required_columns()

        config = FeatureConfig(
            name=f"ema_{period}_{base_calculation.value}",
            category=FeatureCategory.TREND,
            description=f"Exponential Moving Average over {period} periods based on {base_calculation.value}",
            required_columns=required_columns,
            default_lookback=period,
            min_lookback=period,
            max_lookback=period,
            parameters={
                'period': period,
                'base_calculation': base_calculation.value,
                **base_kwargs
            }
        )
        super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)
        self.period = period
        self.base_calculation = base_calculation

        # Initialize VectorBT optimizer
        if ROLLING_OPTIMIZER_AVAILABLE:
            self.vectorbt_rolling_optimizer = get_vectorbt_rolling_optimizer(enable_gpu=False, enable_parallel=True)
        else:
            self.vectorbt_rolling_optimizer = None

    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        # Optimize DataFrame for processing
        if hasattr(self, 'optimize_dataframe_processing'):
            data = self.optimize_dataframe_processing(data)

        # Calculate base values
        base_values = self.base_calculator.calculate(data)

        # Use VectorBT for EMA calculation
        if VECTORBT_AVAILABLE and self._should_use_vectorbt(base_values):
            try:
                ema_indicator = vbt.EMA.run(base_values, window=self.period)
                ema_series = ema_indicator.ema
                # Ensure we always return a Series aligned with the original index
                if not isinstance(ema_series, pd.Series):
                    ema_series = pd.Series(ema_series, index=base_values.index)
                ema_series.name = f"ema_{self.period}_{self.base_calculation.value}"
                return ema_series
            except Exception as e:
                self.logger.warning(f"VectorBT EMA calculation failed: {e}, using pandas fallback")

        ema = base_values.ewm(span=self.period).mean()
        ema.name = f"ema_{self.period}_{self.base_calculation.value}"
        return ema

def create_trend_generators(periods: Dict[str, List[int]] = None) -> List[FeatureGenerator]:
    """Create a set of trend feature generators."""
    if periods is None:
        periods = {
            'sma': [5, 10, 20, 50],
            'ema': [12, 26]
        }

    generators = []

    # SMA generators
    for period in periods.get('sma', [20]):
        generators.append(SMAGenerator(period))

    # EMA generators
    for period in periods.get('ema', [12, 26]):
        generators.append(EMAGenerator(period))

    return generators

def create_default_trend_generators() -> List[FeatureGenerator]:
    return create_trend_generators()

class WMAGenerator(VectorizedFeatureGenerator):
    """Generator for WMA (Weighted Moving Average) with different base calculations."""

    def __init__(self,
                 period: int = 20,
                 base_calculation: Union[str, BaseCalculationType] = BaseCalculationType.PRICE_RETURNS,
                 **base_kwargs):
        """
        Initialize WMA generator.

        Args:
            period: WMA period
            base_calculation: Base calculation type (price_returns, returns_vwap, etc.)
            **base_kwargs: Additional parameters for base calculation
        """
        if isinstance(base_calculation, str):
            base_calculation = BaseCalculationType(base_calculation)

        # Create base calculator
        self.base_calculator = create_base_calculator(base_calculation, **base_kwargs)

        # Update required columns based on base calculation
        required_columns = self.base_calculator.get_required_columns()

        config = FeatureConfig(
            name=f"wma_{period}_{base_calculation.value}",
            category=FeatureCategory.TREND,
            description=f"Weighted Moving Average over {period} periods based on {base_calculation.value}",
            required_columns=required_columns,
            default_lookback=period,
            min_lookback=period,
            max_lookback=period,
            parameters={
                'period': period,
                'base_calculation': base_calculation.value,
                **base_kwargs
            }
        )
        super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)
        self.period = period
        self.base_calculation = base_calculation

    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        # Optimize DataFrame for processing
        if hasattr(self, 'optimize_dataframe_processing'):
            data = self.optimize_dataframe_processing(data)

        """Generate WMA based on the specified base calculation."""
        base_values = self.base_calculator.calculate(data)

        # Use VectorBT for WMA calculation
        if self.vectorbt_optimizer and self._should_use_vectorbt(base_values):
            try:
                # VectorBT doesn't have direct WMA, so we use rolling apply
                weights = np.arange(1, self.period + 1)
                wma = base_values.rolling(self.period).apply(
                    lambda x: np.average(x, weights=weights)
                )
                return wma
            except Exception as e:
                self.logger.warning(f"VectorBT WMA calculation failed: {e}, using pandas fallback")
                weights = np.arange(1, self.period + 1)
                wma = base_values.rolling(self.period).apply(
                    lambda x: np.average(x, weights=weights)
                )
                return wma
        else:
            weights = np.arange(1, self.period + 1)
            wma = base_values.rolling(self.period).apply(
                lambda x: np.average(x, weights=weights)
            )
            return wma

class DEMAGenerator(VectorizedFeatureGenerator):
    """Generator for DEMA (Double Exponential Moving Average) with different base calculations."""

    def __init__(self,
                 period: int = 20,
                 base_calculation: Union[str, BaseCalculationType] = BaseCalculationType.PRICE_RETURNS,
                 **base_kwargs):
        """
        Initialize DEMA generator.

        Args:
            period: DEMA period
            base_calculation: Base calculation type (price_returns, returns_vwap, etc.)
            **base_kwargs: Additional parameters for base calculation
        """
        if isinstance(base_calculation, str):
            base_calculation = BaseCalculationType(base_calculation)

        # Create base calculator
        self.base_calculator = create_base_calculator(base_calculation, **base_kwargs)

        # Update required columns based on base calculation
        required_columns = self.base_calculator.get_required_columns()

        config = FeatureConfig(
            name=f"dema_{period}_{base_calculation.value}",
            category=FeatureCategory.TREND,
            description=f"Double Exponential Moving Average over {period} periods based on {base_calculation.value}",
            required_columns=required_columns,
            default_lookback=period,
            min_lookback=period,
            max_lookback=period,
            parameters={
                'period': period,
                'base_calculation': base_calculation.value,
                **base_kwargs
            }
        )
        super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)
        self.period = period
        self.base_calculation = base_calculation

    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        # Optimize DataFrame for processing
        if hasattr(self, 'optimize_dataframe_processing'):
            data = self.optimize_dataframe_processing(data)

        """Generate DEMA based on the specified base calculation."""
        base_values = self.base_calculator.calculate(data)

        # Use VectorBT for DEMA calculation
        if self.vectorbt_optimizer and self._should_use_vectorbt(base_values):
            try:
                # Calculate DEMA using ewm
                ema1 = base_values.ewm(span=self.period).mean()
                ema2 = ema1.ewm(span=self.period).mean()
                dema = 2 * ema1 - ema2
                return dema
            except Exception as e:
                self.logger.warning(f"VectorBT DEMA calculation failed: {e}, using pandas fallback")
                ema1 = base_values.ewm(span=self.period).mean()
                ema2 = ema1.ewm(span=self.period).mean()
                dema = 2 * ema1 - ema2
                return dema
        else:
            ema1 = base_values.ewm(span=self.period).mean()
            ema2 = ema1.ewm(span=self.period).mean()
            dema = 2 * ema1 - ema2
            return dema

class TEMAGenerator(VectorizedFeatureGenerator):
    """Generator for TEMA (Triple Exponential Moving Average) with different base calculations."""

    def __init__(self,
                 period: int = 20,
                 base_calculation: Union[str, BaseCalculationType] = BaseCalculationType.PRICE_RETURNS,
                 **base_kwargs):
        """
        Initialize TEMA generator.

        Args:
            period: TEMA period
            base_calculation: Base calculation type (price_returns, returns_vwap, etc.)
            **base_kwargs: Additional parameters for base calculation
        """
        if isinstance(base_calculation, str):
            base_calculation = BaseCalculationType(base_calculation)

        # Create base calculator
        self.base_calculator = create_base_calculator(base_calculation, **base_kwargs)

        # Update required columns based on base calculation
        required_columns = self.base_calculator.get_required_columns()

        config = FeatureConfig(
            name=f"tema_{period}_{base_calculation.value}",
            category=FeatureCategory.TREND,
            description=f"Triple Exponential Moving Average over {period} periods based on {base_calculation.value}",
            required_columns=required_columns,
            default_lookback=period,
            min_lookback=period,
            max_lookback=period,
            parameters={
                'period': period,
                'base_calculation': base_calculation.value,
                **base_kwargs
            }
        )
        super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)
        self.period = period
        self.base_calculation = base_calculation

    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        # Optimize DataFrame for processing
        if hasattr(self, 'optimize_dataframe_processing'):
            data = self.optimize_dataframe_processing(data)

        """Generate TEMA based on the specified base calculation."""
        base_values = self.base_calculator.calculate(data)

        # Use VectorBT for TEMA calculation
        if self.vectorbt_optimizer and self._should_use_vectorbt(base_values):
            try:
                # Calculate TEMA using ewm
                ema1 = base_values.ewm(span=self.period).mean()
                ema2 = ema1.ewm(span=self.period).mean()
                ema3 = ema2.ewm(span=self.period).mean()
                tema = 3 * ema1 - 3 * ema2 + ema3
                return tema
            except Exception as e:
                self.logger.warning(f"VectorBT TEMA calculation failed: {e}, using pandas fallback")
                ema1 = base_values.ewm(span=self.period).mean()
                ema2 = ema1.ewm(span=self.period).mean()
                ema3 = ema2.ewm(span=self.period).mean()
                tema = 3 * ema1 - 3 * ema2 + ema3
                return tema
        else:
            ema1 = base_values.ewm(span=self.period).mean()
            ema2 = ema1.ewm(span=self.period).mean()
            ema3 = ema2.ewm(span=self.period).mean()
            tema = 3 * ema1 - 3 * ema2 + ema3
            return tema

class TRIMAGenerator(VectorizedFeatureGenerator):
    """Generator for TRIMA (Triangular Moving Average) with different base calculations."""

    def __init__(self,
                 period: int = 20,
                 base_calculation: Union[str, BaseCalculationType] = BaseCalculationType.PRICE_RETURNS,
                 **base_kwargs):
        """
        Initialize TRIMA generator.

        Args:
            period: TRIMA period
            base_calculation: Base calculation type (price_returns, returns_vwap, etc.)
            **base_kwargs: Additional parameters for base calculation
        """
        if isinstance(base_calculation, str):
            base_calculation = BaseCalculationType(base_calculation)

        # Create base calculator
        self.base_calculator = create_base_calculator(base_calculation, **base_kwargs)

        # Update required columns based on base calculation
        required_columns = self.base_calculator.get_required_columns()

        config = FeatureConfig(
            name=f"trima_{period}_{base_calculation.value}",
            category=FeatureCategory.TREND,
            description=f"Triangular Moving Average over {period} periods based on {base_calculation.value}",
            required_columns=required_columns,
            default_lookback=period,
            min_lookback=period,
            max_lookback=period,
            parameters={
                'period': period,
                'base_calculation': base_calculation.value,
                **base_kwargs
            }
        )
        super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)
        self.period = period
        self.base_calculation = base_calculation

    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        # Optimize DataFrame for processing
        if hasattr(self, 'optimize_dataframe_processing'):
            data = self.optimize_dataframe_processing(data)

        """Generate TRIMA based on the specified base calculation."""
        base_values = self.base_calculator.calculate(data)

        # Use VectorBT for TRIMA calculation
        if self.vectorbt_optimizer and self._should_use_vectorbt(base_values):
            try:
                # Calculate TRIMA using VectorBT rolling mean
                half_period = self.period // 2
                trima = self._optimized_rolling_operation(
                    self.vectorbt_rolling_optimizer.rolling_mean(base_values, half_period),
                    "mean",
                    half_period
                )
                return trima
            except Exception as e:
                self.logger.warning(f"VectorBT TRIMA calculation failed: {e}, using pandas fallback")
                half_period = self.period // 2
                trima = self._optimized_rolling_operation(
                    self._calculate_sma_vectorized(base_values, half_period),
                    "mean",
                    half_period
                )
                return trima
        else:
            half_period = self.period // 2
            trima = self._optimized_rolling_operation(
                self._calculate_sma_vectorized(base_values, half_period),
                "mean",
                half_period
            )
            return trima

# MAMA (MESA Adaptive Moving Average)class MAMAGenerator(VectorizedFeatureGenerator):
    """Generator for MAMA (MESA Adaptive Moving Average) with different base calculations."""

    def __init__(self,
                 fast_limit: float = 0.5,
                 slow_limit: float = 0.05,
                 base_calculation: Union[str, BaseCalculationType] = BaseCalculationType.PRICE_RETURNS,
                 **base_kwargs):
        """
        Initialize MAMA generator.

        Args:
            fast_limit: Fast limit
            slow_limit: Slow limit
            base_calculation: Base calculation type (price_returns, returns_vwap, etc.)
            **base_kwargs: Additional parameters for base calculation
        """
        if isinstance(base_calculation, str):
            base_calculation = BaseCalculationType(base_calculation)

        # Create base calculator
        self.base_calculator = create_base_calculator(base_calculation, **base_kwargs)

        # Update required columns based on base calculation
        required_columns = self.base_calculator.get_required_columns()

        config = FeatureConfig(
            name=f"mama_{fast_limit}_{slow_limit}_{base_calculation.value}",
            category=FeatureCategory.TREND,
            description=f"MESA Adaptive Moving Average with fast_limit={fast_limit}, slow_limit={slow_limit} based on {base_calculation.value}",
            required_columns=required_columns,
            default_lookback=20,
            min_lookback=1,
            max_lookback=50,
            parameters={
                'fast_limit': fast_limit,
                'slow_limit': slow_limit,
                'base_calculation': base_calculation.value,
                **base_kwargs
            }
        )
        super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)
        self.fast_limit = fast_limit
        self.slow_limit = slow_limit
        self.base_calculation = base_calculation

    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        # Optimize DataFrame for processing
        if hasattr(self, 'optimize_dataframe_processing'):
            data = self.optimize_dataframe_processing(data)

        """Generate MAMA based on the specified base calculation."""
        base_values = self.base_calculator.calculate(data)

        # Use VectorBT for MAMA calculation
        if self.vectorbt_optimizer and self._should_use_vectorbt(base_values):
            try:
                # Calculate MAMA (simplified version) using ewm
                mama = self._calculate_ema_vectorized(base_values, 20)
                return mama
            except Exception as e:
                self.logger.warning(f"VectorBT MAMA calculation failed: {e}, using pandas fallback")
                mama = self._calculate_ema_vectorized(base_values, 20)
                return mama
        else:
            mama = self._calculate_ema_vectorized(base_values, 20)
            return mama

class VWMAGenerator(VectorizedFeatureGenerator):
    """Generator for VWMA (Volume Weighted Moving Average) with different base calculations."""

    def __init__(self,
                 period: int = 20,
                 base_calculation: Union[str, BaseCalculationType] = BaseCalculationType.PRICE_RETURNS,
                 **base_kwargs):
        """
        Initialize VWMA generator.

        Args:
            period: VWMA period
            base_calculation: Base calculation type (price_returns, returns_vwap, etc.)
            **base_kwargs: Additional parameters for base calculation
        """
        if isinstance(base_calculation, str):
            base_calculation = BaseCalculationType(base_calculation)

        # Create base calculator
        self.base_calculator = create_base_calculator(base_calculation, **base_kwargs)

        # Update required columns based on base calculation
        required_columns = self.base_calculator.get_required_columns()
        if 'volume' not in required_columns:
            required_columns.append('volume')

        config = FeatureConfig(
            name=f"vwma_{period}_{base_calculation.value}",
            category=FeatureCategory.TREND,
            description=f"Volume Weighted Moving Average over {period} periods based on {base_calculation.value}",
            required_columns=required_columns,
            default_lookback=period,
            min_lookback=period,
            max_lookback=period,
            parameters={
                'period': period,
                'base_calculation': base_calculation.value,
                **base_kwargs
            }
        )
        super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)
        self.period = period
        self.base_calculation = base_calculation

    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        # Optimize DataFrame for processing
        if hasattr(self, 'optimize_dataframe_processing'):
            data = self.optimize_dataframe_processing(data)

        """Generate VWMA based on the specified base calculation."""
        base_values = self.base_calculator.calculate(data)
        volume = data['volume']

        # Calculate VWMA using VectorBT optimization
        if self.vectorbt_optimizer and self._should_use_vectorbt(base_values):
            try:
                numerator = self.vectorbt_rolling_optimizer.rolling_sum(base_values * volume, self.period)
                denominator = self.vectorbt_rolling_optimizer.rolling_sum(volume, self.period)
                vwma = numerator / denominator
            except Exception as e:
                logger.warning(f"VectorBT VWMA calculation failed: {e}, using pandas fallback")
                vwma = (base_values * volume).rolling(self.period).sum() / volume.rolling(self.period).sum()
        else:
            vwma = (base_values * volume).rolling(self.period).sum() / volume.rolling(self.period).sum()

        return vwma

class KeltnerChannelsGenerator(VectorizedFeatureGenerator):
    """Generator for Keltner Channels with different base calculations."""

    def __init__(self,
                 period: int = 20,
                 atr_period: int = 14,
                 multiplier: float = 2.0,
                 base_calculation: Union[str, BaseCalculationType] = BaseCalculationType.PRICE_RETURNS,
                 **base_kwargs):
        """
        Initialize Keltner Channels generator.

        Args:
            period: EMA period for middle line
            atr_period: ATR period for channel width
            multiplier: ATR multiplier for channel width
            base_calculation: Base calculation type
            **base_kwargs: Additional parameters for base calculation
        """
        if isinstance(base_calculation, str):
            base_calculation = BaseCalculationType(base_calculation)

        # Create base calculator
        self.base_calculator = create_base_calculator(base_calculation, **base_kwargs)

        # Update required columns based on base calculation
        required_columns = self.base_calculator.get_required_columns()
        if base_calculation == BaseCalculationType.PRICE_LEVELS:
            required_columns.extend(["high", "low"])  # ATR requires high/low

        config = FeatureConfig(
            name=f"keltner_channels_{period}_{atr_period}_{base_calculation.value}",
            category=FeatureCategory.TREND,
            description=f"Keltner Channels with EMA period={period}, ATR period={atr_period}, multiplier={multiplier} based on {base_calculation.value}",
            required_columns=required_columns,
            default_lookback=max(period, atr_period),
            min_lookback=max(period, atr_period),
            max_lookback=max(period, atr_period),
            parameters={
                'period': period,
                'atr_period': atr_period,
                'multiplier': multiplier,
                'base_calculation': base_calculation.value,
                **base_kwargs
            }
        )
        super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)
        self.period = period
        self.atr_period = atr_period
        self.multiplier = multiplier

    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        # Optimize DataFrame for processing
        if hasattr(self, 'optimize_dataframe_processing'):
            data = self.optimize_dataframe_processing(data)

        """Generate Keltner Channels middle line (EMA) based on the specified base calculation."""
        if self.base_calculator.config.calculation_type == BaseCalculationType.PRICE_LEVELS:
            # Traditional Keltner Channels calculation on price levels
            close = data['close']
            high = data['high']
            low = data['low']

            # Calculate EMA of close prices (middle line)
            ema = close.ewm(span=self.period).mean()

            # Calculate ATR
            tr1 = high - low
            tr2 = abs(high - close.shift(1))
            tr3 = abs(low - close.shift(1))
            true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            # Use VectorBT for optimized ATR calculation
            if self.vectorbt_optimizer and self._should_use_vectorbt(true_range):
                try:
                    atr = self.vectorbt_rolling_optimizer.rolling_mean(true_range, self.atr_period)
                except Exception as e:
                    logger.warning(f"VectorBT ATR calculation failed: {e}, using pandas fallback")
                    atr = true_range.rolling(self.atr_period).mean()
            else:
                atr = true_range.rolling(self.atr_period).mean()

            # Return middle line (EMA) as the main feature
            # Upper and lower bands would be: ema ± (multiplier * atr)
            return ema
        else:
            # For other base calculations, use EMA of base values
            base_values = self.base_calculator.calculate(data)
            ema = base_values.ewm(span=self.period).mean()
            return ema

def create_trend_generators(periods: Dict[str, List[int]] = None) -> List[FeatureGenerator]:
    """Create a set of trend feature generators."""
    if periods is None:
        periods = {
            'sma': [5, 10, 20, 50],
            'ema': [12, 26],
            'wma': [20],
            'dema': [21],
            'tema': [21],
            'trima': [21],
            'mama': [0.5, 0.05],
            'vwma': [20],
            'keltner_channels': [20],
            'adx': [14],
            'trend_score': [14]
        }

    generators = []

    # SMA generators
    for period in periods.get('sma', [20]):
        generators.append(SMAGenerator(period))

    # EMA generators
    for period in periods.get('ema', [12, 26]):
        generators.append(EMAGenerator(period))

    # WMA generators
    for period in periods.get('wma', [20]):
        generators.append(WMAGenerator(period))

    # DEMA generators
    for period in periods.get('dema', [21]):
        generators.append(DEMAGenerator(period))

    # TEMA generators
    for period in periods.get('tema', [21]):
        generators.append(TEMAGenerator(period))

    # TRIMA generators
    for period in periods.get('trima', [21]):
        generators.append(TRIMAGenerator(period))

    # VWMA generators
    for period in periods.get('vwma', [20]):
        generators.append(VWMAGenerator(period))

    # Keltner Channels generators
    for period in periods.get('keltner_channels', [20]):
        generators.append(KeltnerChannelsGenerator(period))

    # ADX generators
    for period in periods.get('adx', [14]):
        generators.append(ADXGenerator(period))

    # Directional Signal generators
    generators.append(DirectionalSignalGenerator())

    # Trend Score generators
    for period in periods.get('trend_score', [14]):
        generators.append(TrendScoreGenerator(adx_period=period))

    return generators

class OptimizedTrendFeatureGenerator(VectorizedFeatureGenerator, VectorBTOptimizationMixin):
    """Optimized trend feature generator using VectorBTRollingOptimizer and UnifiedVectorizationManager."""

    def __init__(self, config: Optional[FeatureConfig] = None):
        if config is None:
            config = self._create_default_config()
        super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)

        # Initialize VectorBT optimizer
        if ROLLING_OPTIMIZER_AVAILABLE:
            self.vectorbt_rolling_optimizer = get_vectorbt_rolling_optimizer(enable_gpu=False, enable_parallel=True)
        else:
            self.vectorbt_rolling_optimizer = None

        # Initialize Unified Vectorization Manager
        if UNIFIED_MANAGER_AVAILABLE:
            self.unified_manager = get_unified_vectorization_manager()
        else:
            self.unified_manager = None

    @classmethod
    def _create_default_config(cls) -> FeatureConfig:
        return FeatureConfig(
            name="optimized_trend_features",
            category=FeatureCategory.TREND,
            description="Optimized trend features using VectorBTRollingOptimizer and UnifiedVectorizationManager",
            required_columns=["close"],
            optional_columns=["high", "low", "open", "volume"],
            default_lookback=20,
            min_lookback=5,
            max_lookback=100,
            parameters={
                "periods": [5, 10, 20, 50],
                "use_unified_manager": True,
                "batch_processing": True
            },
            matrix_optimized=True,
            gpu_accelerated=False
        )

    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate optimized trend features using VectorBTRollingOptimizer and UnifiedVectorizationManager."""
        if data.empty:
            return pd.Series(dtype=float, index=data.index, name='optimized_trend')

        # Use UnifiedVectorizationManager for batch processing if available
        if self.unified_manager and self.config.parameters.get("use_unified_manager", True):
            return self._generate_with_unified_manager(data, **kwargs)
        else:
            return self._generate_with_vectorbt_optimizer(data, **kwargs)

    def _generate_with_unified_manager(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate features using UnifiedVectorizationManager for optimal performance."""
        try:
            # Prepare data for unified manager
            operation_data = {
                'data': data,
                'operation_type': 'trend_features',
                'periods': self.config.parameters.get("periods", [20]),
                'use_vectorbt': True
            }

            # Use unified manager for optimization
            result = self.unified_manager.optimize_operation(
                OperationType.FEATURE_ENGINEERING,
                operation_data,
                **kwargs
            )

            # Extract the primary trend feature
            if hasattr(result, 'result') and isinstance(result.result, pd.Series):
                return result.result
            else:
                # Fallback to VectorBT optimizer
                return self._generate_with_vectorbt_optimizer(data, **kwargs)

        except Exception as e:
            logger.warning(f"UnifiedVectorizationManager failed: {e}, falling back to VectorBTRollingOptimizer")
            return self._generate_with_vectorbt_optimizer(data, **kwargs)

    def _generate_with_vectorbt_optimizer(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate features using VectorBTRollingOptimizer for batch processing."""
        if not self.vectorbt_optimizer:
            return self._generate_basic_trend(data, **kwargs)

        try:
            periods = self.config.parameters.get("periods", [20])
            close_prices = data['close']

            # Batch process multiple periods using VectorBTRollingOptimizer
            trend_features = []

            for period in periods:
                if len(close_prices) >= period:
                    # Use VectorBTRollingOptimizer for each period
                    sma = self.vectorbt_rolling_optimizer.rolling_mean(close_prices, period)
                    trend_features.append(sma)

            if trend_features:
                # Combine features (use weighted average of different periods)
                weights = np.array([1.0 / len(trend_features)] * len(trend_features))
                combined_trend = sum(w * feature for w, feature in zip(weights, trend_features))
                return combined_trend.rename('optimized_trend')
            else:
                return self._generate_basic_trend(data, **kwargs)

        except Exception as e:
            logger.warning(f"VectorBTRollingOptimizer failed: {e}, using basic trend")
            return self._generate_basic_trend(data, **kwargs)

    def _generate_basic_trend(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate basic trend feature as fallback."""
        close_prices = data['close']
        period = self.config.default_lookback

        if len(close_prices) >= period:
            return close_prices.rolling(period).mean().rename('optimized_trend')
        else:
            return pd.Series(np.nan, index=data.index, name='optimized_trend')

    def generate_batch_features(self, data: pd.DataFrame, **kwargs) -> Dict[str, pd.Series]:
        """Generate multiple trend features in batch for optimal performance."""
        if data.empty:
            return {}

        features = {}
        periods = self.config.parameters.get("periods", [20])
        close_prices = data['close']

        if self.vectorbt_optimizer:
            try:
                # Batch process all periods at once
                for period in periods:
                    if len(close_prices) >= period:
                        features[f'sma_{period}'] = self.vectorbt_rolling_optimizer.rolling_mean(close_prices, period)
                        features[f'std_{period}'] = self.vectorbt_rolling_optimizer.rolling_std(close_prices, period)

                        # Additional trend indicators
                        if 'high' in data.columns and 'low' in data.columns:
                            high_low_avg = (data['high'] + data['low']) / 2
                            features[f'hl2_sma_{period}'] = self.vectorbt_rolling_optimizer.rolling_mean(high_low_avg, period)

                        if 'volume' in data.columns:
                            # Volume-weighted price
                            vwp = (close_prices * data['volume']).rolling(period).sum() / data['volume'].rolling(period).sum()
                            features[f'vwp_{period}'] = vwp

            except Exception as e:
                logger.warning(f"Batch feature generation failed: {e}, using individual processing")
                return self._generate_individual_features(data, **kwargs)
        else:
            return self._generate_individual_features(data, **kwargs)

        return features

    def _generate_individual_features(self, data: pd.DataFrame, **kwargs) -> Dict[str, pd.Series]:
        """Generate features individually as fallback."""
        features = {}
        periods = self.config.parameters.get("periods", [20])
        close_prices = data['close']

        for period in periods:
            if len(close_prices) >= period:
                features[f'sma_{period}'] = close_prices.rolling(period).mean()
                features[f'std_{period}'] = close_prices.rolling(period).std()

        return features

class VectorBTTrendFeatureGenerator(VectorBTFeatureGenerator):
    """VectorBT-optimized trend feature generator with comprehensive indicators."""

    def __init__(self, period: int = 20, config: Optional[FeatureConfig] = None):
        if config is None:
            config = self._create_default_config(period)
        super().__init__(config, enable_gpu=True, enable_parallel=True)
        self.period = period

        # Initialize VectorBT optimizer for enhanced performance
        if ROLLING_OPTIMIZER_AVAILABLE:
            self.vectorbt_rolling_optimizer = get_vectorbt_rolling_optimizer(enable_gpu=True, enable_parallel=True)
        else:
            self.vectorbt_rolling_optimizer = None

    @classmethod
    def _create_default_config(cls, period: int = 20) -> FeatureConfig:
        return FeatureConfig(
            name=f"vectorbt_trend_comprehensive_{period}",
            category=FeatureCategory.TREND,
            description=f"VectorBT-optimized comprehensive trend features over {period} periods",
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
        """Generate comprehensive trend features using VectorBTRollingOptimizer."""
        if data.empty:
            return pd.Series(dtype=float, index=data.index, name=f'vectorbt_trend_{self.period}')

        # Use VectorBTRollingOptimizer for enhanced performance
        if self.vectorbt_optimizer:
            try:
                close_prices = data['close']

                # Generate SMA using VectorBTRollingOptimizer
                sma = self.vectorbt_rolling_optimizer.rolling_mean(close_prices, self.period)

                # Generate additional trend indicators
                if 'high' in data.columns and 'low' in data.columns:
                    high_low_avg = (data['high'] + data['low']) / 2
                    hl2_sma = self.vectorbt_rolling_optimizer.rolling_mean(high_low_avg, self.period)

                    # Combine SMA and HL2 SMA for comprehensive trend
                    trend = (sma + hl2_sma) / 2
                else:
                    trend = sma

                return trend.rename(f'vectorbt_trend_{self.period}')

            except Exception as e:
                logger.warning(f"VectorBTRollingOptimizer failed: {e}, using fallback")
                return self._generate_fallback_trend(data, **kwargs)
        else:
            return self._generate_fallback_trend(data, **kwargs)

    def _generate_fallback_trend(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate trend using fallback method."""
        if data.empty:
            return pd.Series(dtype=float, index=data.index, name=f'vectorbt_trend_{self.period}')

        # Generate multiple trend indicators using VectorBT
        operations = [
            {'type': 'rolling', 'name': 'sma', 'params': {'operation': 'mean', 'window': self.period, 'column': 'close'}},
            {'type': 'indicator', 'name': 'ema', 'params': {'indicator': 'ema', 'window': self.period}},
            {'type': 'indicator', 'name': 'wma', 'params': {'indicator': 'wma', 'window': self.period}},
            {'type': 'indicator', 'name': 'dema', 'params': {'indicator': 'dema', 'window': self.period}},
            {'type': 'indicator', 'name': 'tema', 'params': {'indicator': 'tema', 'window': self.period}},
            {'type': 'indicator', 'name': 'kama', 'params': {'indicator': 'kama', 'window': self.period}},
            {'type': 'indicator', 'name': 'adx', 'params': {'indicator': 'adx', 'window': self.period}},
            {'type': 'indicator', 'name': 'plus_di', 'params': {'indicator': 'plus_di', 'window': self.period}},
            {'type': 'indicator', 'name': 'minus_di', 'params': {'indicator': 'minus_di', 'window': self.period}},
            {'type': 'indicator', 'name': 'aroon_up', 'params': {'indicator': 'aroon_up', 'window': self.period}},
            {'type': 'indicator', 'name': 'aroon_down', 'params': {'indicator': 'aroon_down', 'window': self.period}}
        ]

        # Use batch operations for efficiency
        results = self._vectorbt_batch_operations(data, operations)

        # Combine results into a single trend measure
        if not results.empty:
            # Weighted combination of different trend measures
            trend = (
                0.20 * results.get('sma', 0) +
                0.15 * results.get('ema', 0) +
                0.15 * results.get('wma', 0) +
                0.10 * results.get('dema', 0) +
                0.10 * results.get('tema', 0) +
                0.10 * results.get('kama', 0) +
                0.10 * results.get('adx', 0) +
                0.05 * results.get('plus_di', 0) +
                0.05 * results.get('minus_di', 0)
            )
        else:
            # Fallback to simple SMA
            trend = self._vectorbt_rolling_operation(data['close'], 'mean', self.period)

        return trend.rename(f'vectorbt_trend_{self.period}')

class VectorBTSMAGenerator(VectorBTFeatureGenerator):
    """VectorBT-optimized Simple Moving Average generator."""

    def __init__(self, period: int = 20, config: Optional[FeatureConfig] = None):
        if config is None:
            config = self._create_default_config(period)
        super().__init__(config, enable_gpu=True, enable_parallel=True)
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
        """Generate SMA using VectorBT rolling mean."""
        if data.empty:
            return pd.Series(dtype=float, index=data.index, name=f'vectorbt_sma_{self.period}')

        # Generate SMA using VectorBT rolling mean
        sma = self._vectorbt_rolling_operation(data['close'], 'mean', self.period)

        return sma.rename(f'vectorbt_sma_{self.period}')

class VectorBTEMAGenerator(VectorBTFeatureGenerator):
    """VectorBT-optimized Exponential Moving Average generator."""

    def __init__(self, period: int = 20, config: Optional[FeatureConfig] = None):
        if config is None:
            config = self._create_default_config(period)
        super().__init__(config, enable_gpu=True, enable_parallel=True)
        self.period = period

    @classmethod
    def _create_default_config(cls, period: int = 20) -> FeatureConfig:
        return FeatureConfig(
            name=f"vectorbt_ema_{period}",
            category=FeatureCategory.TREND,
            description=f"VectorBT-optimized EMA over {period} periods",
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
        """Generate EMA using VectorBT."""
        if data.empty:
            return pd.Series(dtype=float, index=data.index, name=f'vectorbt_ema_{self.period}')

        # Generate EMA using VectorBT
        ema = self._vectorbt_technical_indicator(data, 'ema', self.period)

        return ema.rename(f'vectorbt_ema_{self.period}')

class VectorBTADXGenerator(VectorBTFeatureGenerator):
    """VectorBT-optimized ADX generator."""

    def __init__(self, period: int = 14, config: Optional[FeatureConfig] = None):
        if config is None:
            config = self._create_default_config(period)
        super().__init__(config, enable_gpu=True, enable_parallel=True)
        self.period = period

    @classmethod
    def _create_default_config(cls, period: int = 14) -> FeatureConfig:
        return FeatureConfig(
            name=f"vectorbt_adx_{period}",
            category=FeatureCategory.TREND,
            description=f"VectorBT-optimized ADX over {period} periods",
            required_columns=["high", "low", "close"],
            optional_columns=["open"],
            default_lookback=period,
            min_lookback=period,
            max_lookback=period,
            parameters={"period": period},
            matrix_optimized=True,
            gpu_accelerated=True
        )

    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate ADX using VectorBT."""
        if data.empty:
            return pd.Series(dtype=float, index=data.index, name=f'vectorbt_adx_{self.period}')

        # Generate ADX using VectorBT
        adx = self._vectorbt_technical_indicator(data, 'adx', self.period)

        return adx.rename(f'vectorbt_adx_{self.period}')

def create_default_trend_generators() -> List[FeatureGenerator]:
    """Create default trend generators with VectorBT optimization."""
    generators = []

    if VECTORBT_AVAILABLE:
        # VectorBT-optimized generators
        for period in [5, 10, 20, 50, 100]:
            generators.append(VectorBTTrendFeatureGenerator(period))
            generators.append(VectorBTSMAGenerator(period))
            generators.append(VectorBTEMAGenerator(period))

        # ADX with different periods
        for period in [9, 14, 21]:
            generators.append(VectorBTADXGenerator(period))

        # Advanced trend indicators
        # Ichimoku Cloud with different parameters
        for tenkan in [9, 12]:
            for kijun in [26, 30]:
                generators.append(VectorBTIchimokuCloudGenerator(tenkan, kijun, 52, 26))

        # Parabolic SAR with different parameters
        for acc in [0.02, 0.05, 0.1]:
            for max_acc in [0.2, 0.3]:
                generators.append(VectorBTParabolicSARGenerator(acc, max_acc))

        # ZigZag with different parameters
        for deviation in [3.0, 5.0, 7.0, 10.0]:
            for backstep in [2, 3, 5]:
                generators.append(VectorBTZigZagGenerator(deviation, backstep))
    else:
        # Fallback to original generators
        periods = {
            'sma': [5, 10, 20, 50, 100],
            'ema': [12, 26, 50],
            'wma': [20],
            'dema': [21],
            'tema': [21],
            'trima': [21],
            'vwma': [20],
            'keltner_channels': [20],
            'adx': [14],
            'trend_score': [14]
        }

        # SMA generators
        for period in periods.get('sma', [20]):
            generators.append(SMAGenerator(period))

        # EMA generators
        for period in periods.get('ema', [12, 26]):
            generators.append(EMAGenerator(period))

        # WMA generators
        for period in periods.get('wma', [20]):
            generators.append(WMAGenerator(period))

        # DEMA generators
        for period in periods.get('dema', [21]):
            generators.append(DEMAGenerator(period))

        # TEMA generators
        for period in periods.get('tema', [21]):
            generators.append(TEMAGenerator(period))

        # TRIMA generators
        for period in periods.get('trima', [21]):
            generators.append(TRIMAGenerator(period))

        # VWMA generators
        for period in periods.get('vwma', [20]):
            generators.append(VWMAGenerator(period))

        # Keltner Channels generators
        for period in periods.get('keltner_channels', [20]):
            generators.append(KeltnerChannelsGenerator(period))
    # All generators now use VectorBT optimization through the mixin
    periods = {
        'sma': [5, 10, 20, 50, 100],
        'ema': [12, 26, 50],
        'wma': [20],
        'dema': [21],
        'tema': [21],
        'trima': [21],
        'vwma': [20],
        'keltner_channels': [20],
        'adx': [14],
        'trend_score': [14]
    }

    # SMA generators
    for period in periods.get('sma', [20]):
        generators.append(SMAGenerator(period))

    # EMA generators
    for period in periods.get('ema', [12, 26]):
        generators.append(EMAGenerator(period))

    # WMA generators
    for period in periods.get('wma', [20]):
        generators.append(WMAGenerator(period))

    # DEMA generators
    for period in periods.get('dema', [21]):
        generators.append(DEMAGenerator(period))

    # TEMA generators
    for period in periods.get('tema', [21]):
        generators.append(TEMAGenerator(period))

    # TRIMA generators
    for period in periods.get('trima', [21]):
        generators.append(TRIMAGenerator(period))

    # VWMA generators
    for period in periods.get('vwma', [20]):
        generators.append(VWMAGenerator(period))

    # Keltner Channels generators
    for period in periods.get('keltner_channels', [20]):
        generators.append(KeltnerChannelsGenerator(period))

    # ADX generators
    for period in periods.get('adx', [14]):
        generators.append(ADXGenerator(period))

    # Directional Signal generators
    generators.append(DirectionalSignalGenerator())

    # Trend Score generators
    for period in periods.get('trend_score', [14]):
        generators.append(TrendScoreGenerator(adx_period=period))

    return generators

class VectorBTIchimokuCloudGenerator(VectorBTFeatureGenerator):
    """VectorBT-optimized Ichimoku Cloud generator."""

    def __init__(self, tenkan_period: int = 9, kijun_period: int = 26, senkou_span_b_period: int = 52,
                 displacement: int = 26, config: Optional[FeatureConfig] = None):
        if config is None:
            config = self._create_default_config(tenkan_period, kijun_period, senkou_span_b_period, displacement)
        super().__init__(config, enable_gpu=True, enable_parallel=True)
        self.tenkan_period = tenkan_period
        self.kijun_period = kijun_period
        self.senkou_span_b_period = senkou_span_b_period
        self.displacement = displacement

        # Initialize VectorBT rolling optimizer
        if ROLLING_OPTIMIZER_AVAILABLE:
            self.rolling_optimizer = get_vectorbt_rolling_optimizer(enable_gpu=True, enable_parallel=True)
        else:
            self.rolling_optimizer = None

    @classmethod
    def _create_default_config(cls, tenkan_period: int = 9, kijun_period: int = 26,
                              senkou_span_b_period: int = 52, displacement: int = 26) -> FeatureConfig:
        return FeatureConfig(
            name=f"vectorbt_ichimoku_cloud_{tenkan_period}_{kijun_period}_{senkou_span_b_period}",
            category=FeatureCategory.TREND,
            description=f"VectorBT-optimized Ichimoku Cloud with Tenkan={tenkan_period}, Kijun={kijun_period}, Senkou Span B={senkou_span_b_period}",
            required_columns=["high", "low", "close"],
            optional_columns=["open"],
            default_lookback=max(tenkan_period, kijun_period, senkou_span_b_period) + displacement,
            min_lookback=max(tenkan_period, kijun_period, senkou_span_b_period) + displacement,
            max_lookback=max(tenkan_period, kijun_period, senkou_span_b_period) + displacement,
            parameters={
                "tenkan_period": tenkan_period,
                "kijun_period": kijun_period,
                "senkou_span_b_period": senkou_span_b_period,
                "displacement": displacement
            },
            matrix_optimized=True,
            gpu_accelerated=True
        )

    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate Ichimoku Cloud features using VectorBT."""
        if data.empty or not all(col in data.columns for col in ['high', 'low', 'close']):
            return pd.Series(dtype=float, index=data.index, name=f'vectorbt_ichimoku_cloud_{self.tenkan_period}')

        try:
            high = data['high']
            low = data['low']
            close = data['close']

            # Calculate Tenkan-sen (Conversion Line)
            if self.rolling_optimizer:
                try:
                    tenkan_high = self.rolling_optimizer.rolling_max(high, self.tenkan_period)
                    tenkan_low = self.rolling_optimizer.rolling_min(low, self.tenkan_period)
                except Exception as e:
                    self.logger.warning(f"VectorBT rolling optimizer failed: {e}, using fallback")
                    tenkan_high = high.rolling(self.tenkan_period).max()
                    tenkan_low = low.rolling(self.tenkan_period).min()
            else:
                tenkan_high = high.rolling(self.tenkan_period).max()
                tenkan_low = low.rolling(self.tenkan_period).min()

            tenkan_sen = (tenkan_high + tenkan_low) / 2

            # Calculate Kijun-sen (Base Line)
            if self.rolling_optimizer:
                try:
                    kijun_high = self.rolling_optimizer.rolling_max(high, self.kijun_period)
                    kijun_low = self.rolling_optimizer.rolling_min(low, self.kijun_period)
                except Exception as e:
                    self.logger.warning(f"VectorBT rolling optimizer failed: {e}, using fallback")
                    kijun_high = high.rolling(self.kijun_period).max()
                    kijun_low = low.rolling(self.kijun_period).min()
            else:
                kijun_high = high.rolling(self.kijun_period).max()
                kijun_low = low.rolling(self.kijun_period).min()

            kijun_sen = (kijun_high + kijun_low) / 2

            # Calculate Senkou Span A (Leading Span A)
            senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(self.displacement)

            # Calculate Senkou Span B (Leading Span B)
            if self.rolling_optimizer:
                try:
                    senkou_high = self.rolling_optimizer.rolling_max(high, self.senkou_span_b_period)
                    senkou_low = self.rolling_optimizer.rolling_min(low, self.senkou_span_b_period)
                except Exception as e:
                    self.logger.warning(f"VectorBT rolling optimizer failed: {e}, using fallback")
                    senkou_high = high.rolling(self.senkou_span_b_period).max()
                    senkou_low = low.rolling(self.senkou_span_b_period).min()
            else:
                senkou_high = high.rolling(self.senkou_span_b_period).max()
                senkou_low = low.rolling(self.senkou_span_b_period).min()

            senkou_span_b = ((senkou_high + senkou_low) / 2).shift(self.displacement)

            # Calculate Chikou Span (Lagging Span)
            chikou_span = close.shift(-self.displacement)

            # Return the cloud position indicator (price relative to cloud)
            cloud_position = close - ((senkou_span_a + senkou_span_b) / 2)

            return cloud_position.rename(f'vectorbt_ichimoku_cloud_{self.tenkan_period}')

        except Exception as e:
            self.logger.error(f"Error generating Ichimoku Cloud: {e}")
            return pd.Series(np.nan, index=data.index, name=f'vectorbt_ichimoku_cloud_{self.tenkan_period}')

class VectorBTParabolicSARGenerator(VectorBTFeatureGenerator):
    """VectorBT-optimized Parabolic SAR generator."""

    def __init__(self, acceleration: float = 0.02, maximum: float = 0.2, config: Optional[FeatureConfig] = None):
        if config is None:
            config = self._create_default_config(acceleration, maximum)
        super().__init__(config, enable_gpu=True, enable_parallel=True)
        self.acceleration = acceleration
        self.maximum = maximum

        # Initialize VectorBT rolling optimizer
        if ROLLING_OPTIMIZER_AVAILABLE:
            self.rolling_optimizer = get_vectorbt_rolling_optimizer(enable_gpu=True, enable_parallel=True)
        else:
            self.rolling_optimizer = None

    @classmethod
    def _create_default_config(cls, acceleration: float = 0.02, maximum: float = 0.2) -> FeatureConfig:
        return FeatureConfig(
            name=f"vectorbt_parabolic_sar_{acceleration}_{maximum}",
            category=FeatureCategory.TREND,
            description=f"VectorBT-optimized Parabolic SAR with acceleration={acceleration}, maximum={maximum}",
            required_columns=["high", "low", "close"],
            optional_columns=["open"],
            default_lookback=50,  # Need enough data for SAR calculation
            min_lookback=10,
            max_lookback=100,
            parameters={
                "acceleration": acceleration,
                "maximum": maximum
            },
            matrix_optimized=True,
            gpu_accelerated=True
        )

    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate Parabolic SAR using VectorBT."""
        if data.empty or not all(col in data.columns for col in ['high', 'low', 'close']):
            return pd.Series(dtype=float, index=data.index, name=f'vectorbt_parabolic_sar_{self.acceleration}')

        try:
            high = data['high']
            low = data['low']
            close = data['close']

            # Initialize SAR arrays
            sar = np.zeros(len(close))
            trend = np.zeros(len(close))
            af = np.zeros(len(close))
            ep = np.zeros(len(close))

            # Initialize first values
            sar[0] = low.iloc[0]
            trend[0] = 1  # 1 for uptrend, -1 for downtrend
            af[0] = self.acceleration
            ep[0] = high.iloc[0]

            # Calculate SAR using VectorBT-optimized operations where possible
            for i in range(1, len(close)):
                # Calculate previous SAR
                prev_sar = sar[i-1]
                prev_trend = trend[i-1]
                prev_af = af[i-1]
                prev_ep = ep[i-1]

                # Calculate new SAR
                new_sar = prev_sar + prev_af * (prev_ep - prev_sar)

                # Check for trend reversal
                if prev_trend == 1:  # Uptrend
                    if low.iloc[i] <= new_sar:
                        # Trend reversal to downtrend
                        trend[i] = -1
                        sar[i] = prev_ep
                        ep[i] = low.iloc[i]
                        af[i] = self.acceleration
                    else:
                        # Continue uptrend
                        trend[i] = 1
                        sar[i] = new_sar
                        if high.iloc[i] > prev_ep:
                            ep[i] = high.iloc[i]
                            af[i] = min(prev_af + self.acceleration, self.maximum)
                        else:
                            ep[i] = prev_ep
                            af[i] = prev_af
                else:  # Downtrend
                    if high.iloc[i] >= new_sar:
                        # Trend reversal to uptrend
                        trend[i] = 1
                        sar[i] = prev_ep
                        ep[i] = high.iloc[i]
                        af[i] = self.acceleration
                    else:
                        # Continue downtrend
                        trend[i] = -1
                        sar[i] = new_sar
                        if low.iloc[i] < prev_ep:
                            ep[i] = low.iloc[i]
                            af[i] = min(prev_af + self.acceleration, self.maximum)
                        else:
                            ep[i] = prev_ep
                            af[i] = prev_af

            # Calculate SAR signal (price relative to SAR)
            sar_signal = close - pd.Series(sar, index=close.index)

            return sar_signal.rename(f'vectorbt_parabolic_sar_{self.acceleration}')

        except Exception as e:
            self.logger.error(f"Error generating Parabolic SAR: {e}")
            return pd.Series(np.nan, index=data.index, name=f'vectorbt_parabolic_sar_{self.acceleration}')

class VectorBTZigZagGenerator(VectorBTFeatureGenerator):
    """VectorBT-optimized ZigZag indicator generator."""

    def __init__(self, deviation: float = 5.0, backstep: int = 3, config: Optional[FeatureConfig] = None):
        if config is None:
            config = self._create_default_config(deviation, backstep)
        super().__init__(config, enable_gpu=True, enable_parallel=True)
        self.deviation = deviation
        self.backstep = backstep

        # Initialize VectorBT rolling optimizer
        if ROLLING_OPTIMIZER_AVAILABLE:
            self.rolling_optimizer = get_vectorbt_rolling_optimizer(enable_gpu=True, enable_parallel=True)
        else:
            self.rolling_optimizer = None

    @classmethod
    def _create_default_config(cls, deviation: float = 5.0, backstep: int = 3) -> FeatureConfig:
        return FeatureConfig(
            name=f"vectorbt_zigzag_{deviation}_{backstep}",
            category=FeatureCategory.TREND,
            description=f"VectorBT-optimized ZigZag indicator with deviation={deviation}%, backstep={backstep}",
            required_columns=["high", "low", "close"],
            optional_columns=["open"],
            default_lookback=50,  # Need enough data for ZigZag calculation
            min_lookback=10,
            max_lookback=200,
            parameters={
                "deviation": deviation,
                "backstep": backstep
            },
            matrix_optimized=True,
            gpu_accelerated=True
        )

    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate ZigZag indicator using VectorBT."""
        if data.empty or not all(col in data.columns for col in ['high', 'low', 'close']):
            return pd.Series(dtype=float, index=data.index, name=f'vectorbt_zigzag_{self.deviation}')

        try:
            high = data['high']
            low = data['low']
            close = data['close']

            # Initialize ZigZag arrays
            zigzag = np.zeros(len(close))
            last_high_idx = 0
            last_low_idx = 0
            last_high = high.iloc[0]
            last_low = low.iloc[0]
            direction = 0  # 0 = neutral, 1 = up, -1 = down

            # Calculate ZigZag points
            for i in range(1, len(close)):
                current_high = high.iloc[i]
                current_low = low.iloc[i]

                if direction == 0:  # Neutral - looking for first significant move
                    if current_high > last_high * (1 + self.deviation / 100):
                        direction = 1
                        last_high_idx = i
                        last_high = current_high
                        zigzag[i] = current_high
                    elif current_low < last_low * (1 - self.deviation / 100):
                        direction = -1
                        last_low_idx = i
                        last_low = current_low
                        zigzag[i] = current_low
                elif direction == 1:  # Uptrend - looking for peak
                    if current_high > last_high:
                        # New high found
                        zigzag[last_high_idx] = 0  # Remove previous high
                        last_high_idx = i
                        last_high = current_high
                        zigzag[i] = current_high
                    elif current_low < last_high * (1 - self.deviation / 100):
                        # Significant pullback - start downtrend
                        direction = -1
                        last_low_idx = i
                        last_low = current_low
                        zigzag[i] = current_low
                else:  # Downtrend - looking for trough
                    if current_low < last_low:
                        # New low found
                        zigzag[last_low_idx] = 0  # Remove previous low
                        last_low_idx = i
                        last_low = current_low
                        zigzag[i] = current_low
                    elif current_high > last_low * (1 + self.deviation / 100):
                        # Significant bounce - start uptrend
                        direction = 1
                        last_high_idx = i
                        last_high = current_high
                        zigzag[i] = current_high

            # Calculate ZigZag trend strength
            zigzag_series = pd.Series(zigzag, index=close.index)
            non_zero_points = zigzag_series[zigzag_series != 0]

            if len(non_zero_points) > 1:
                # Calculate trend strength as the slope of the last ZigZag segment
                last_two_points = non_zero_points.tail(2)
                if len(last_two_points) == 2:
                    trend_strength = (last_two_points.iloc[-1] - last_two_points.iloc[0]) / len(last_two_points)
                else:
                    trend_strength = 0
            else:
                trend_strength = 0

            # Create trend strength series
            trend_strength_series = pd.Series(trend_strength, index=close.index)

            return trend_strength_series.rename(f'vectorbt_zigzag_{self.deviation}')

        except Exception as e:
            self.logger.error(f"Error generating ZigZag indicator: {e}")
            return pd.Series(np.nan, index=data.index, name=f'vectorbt_zigzag_{self.deviation}')

def create_optimized_trend_generators(periods: List[int] = None, use_unified_manager: bool = True) -> List[FeatureGenerator]:
    """Create a list of optimized trend feature generators using VectorBTRollingOptimizer and UnifiedVectorizationManager."""
    if periods is None:
        periods = [5, 10, 20, 50]

    generators = []

    # Add the new optimized generator
    optimized_config = FeatureConfig(
        name="optimized_trend_features",
        category=FeatureCategory.TREND,
        description="Optimized trend features using VectorBTRollingOptimizer and UnifiedVectorizationManager",
        required_columns=["close"],
        optional_columns=["high", "low", "open", "volume"],
        default_lookback=20,
        min_lookback=5,
        max_lookback=100,
        parameters={
            "periods": periods,
            "use_unified_manager": use_unified_manager,
            "batch_processing": True
        },
        matrix_optimized=True,
        gpu_accelerated=False
    )
    generators.append(OptimizedTrendFeatureGenerator(config=optimized_config))

    # Add VectorBT generators for each period
    for period in periods:
        generators.append(VectorBTTrendFeatureGenerator(period=period))
        generators.append(VectorBTSMAGenerator(period=period))
        generators.append(VectorBTEMAGenerator(period=period))
        generators.append(VectorBTADXGenerator(period=period))
        generators.append(VectorBTIchimokuCloudGenerator(tenkan_period=period))
        generators.append(VectorBTParabolicSARGenerator(acceleration=0.02))
        generators.append(VectorBTZigZagGenerator(deviation=0.05))

    return generators

    def _optimized_rolling_operation(self, data: pd.Series, operation: str,
                                   window: int, **kwargs) -> pd.Series:
        """Perform rolling operation using centralized VectorBTRollingOptimizer."""
        if not hasattr(self, 'rolling_optimizer'):
            self.rolling_optimizer = get_vectorbt_rolling_optimizer()

        try:
            if operation == 'mean':
                return self.rolling_optimizer.rolling_mean(data, window, **kwargs)
            elif operation == 'std':
                return self.rolling_optimizer.rolling_std(data, window, **kwargs)
            elif operation == 'var':
                return self.rolling_optimizer.rolling_var(data, window, **kwargs)
            elif operation == 'min':
                return self.rolling_optimizer.rolling_min(data, window, **kwargs)
            elif operation == 'max':
                return self.rolling_optimizer.rolling_max(data, window, **kwargs)
            elif operation == 'sum':
                return self.rolling_optimizer.rolling_sum(data, window, **kwargs)
            else:
                raise ValueError(f"Unsupported operation: {operation}")
        except Exception as e:
            logger.warning(f"VectorBT rolling operation failed: {e}, using fallback")
            return self._fallback_rolling_operation(data, operation, window, **kwargs)

    def _fallback_rolling_operation(self, data: pd.Series, operation: str,
                                  window: int, **kwargs) -> pd.Series:
        """Fallback rolling operation using pandas."""
        rolling_obj = data.rolling(window, **kwargs)

        if operation == 'mean':
            return rolling_obj.mean()
        elif operation == 'std':
            return rolling_obj.std()
        elif operation == 'var':
            return rolling_obj.var()
        elif operation == 'min':
            return rolling_obj.min()
        elif operation == 'max':
            return rolling_obj.max()
        elif operation == 'sum':
            return rolling_obj.sum()
        else:
            raise ValueError(f"Unsupported operation: {operation}")

    def _normalize_feature(self, data: pd.Series, method: str = 'zscore') -> pd.Series:
        """Normalize feature using direct scaling to avoid circular imports."""
        try:
            if method == 'zscore':
                return (data - data.mean()) / data.std()
            elif method == 'minmax':
                return (data - data.min()) / (data.max() - data.min())
            elif method == 'robust':
                median = data.median()
                mad = (data - median).abs().median()
                return (data - median) / mad
            else:
                logger.warning(f"Unsupported normalization method: {method}, using zscore")
                return (data - data.mean()) / data.std()
        except Exception as e:
            logger.warning(f"Normalization failed: {e}, using simple zscore")
            return (data - data.mean()) / data.std()

    def _fallback_normalize(self, data: pd.Series, method: str = 'zscore') -> pd.Series:
        """Fallback normalization using pandas/numpy."""
        if method == 'zscore':
            return (data - data.mean()) / data.std()
        elif method == 'minmax':
            return (data - data.min()) / (data.max() - data.min())
        elif method == 'robust':
            median = data.median()
            mad = (data - median).abs().median()
            return (data - median) / mad
        else:
            return data

    def _should_use_vectorbt(self, data) -> bool:
        """Determine if VectorBT should be used based on data size and configuration."""
        return (hasattr(self, 'use_vectorbt') and self.use_vectorbt and
                len(data) >= getattr(self, 'vectorbt_threshold', 1000) and
                VECTORBT_AVAILABLE)

    def _vectorbt_rolling_operation(self, data: pd.Series, operation: str,
                                  window: int, **kwargs) -> pd.Series:
        """Perform VectorBT rolling operation with fallback to pandas using VectorBTRollingOptimizer."""
        # Use VectorBTRollingOptimizer if available
        if self.vectorbt_optimizer and self._should_use_vectorbt(data):
            try:
                if operation == 'mean':
                    return self.vectorbt_rolling_optimizer.rolling_mean(data, window, **kwargs)
                elif operation == 'std':
                    return self.vectorbt_rolling_optimizer.rolling_std(data, window, **kwargs)
                elif operation == 'var':
                    return self.vectorbt_rolling_optimizer.rolling_var(data, window, **kwargs)
                elif operation == 'min':
                    return self.vectorbt_rolling_optimizer.rolling_min(data, window, **kwargs)
                elif operation == 'max':
                    return self.vectorbt_rolling_optimizer.rolling_max(data, window, **kwargs)
                elif operation == 'sum':
                    return self.vectorbt_rolling_optimizer.rolling_sum(data, window, **kwargs)
                elif operation == 'quantile':
                    q = kwargs.get('q', 0.5)
                    return self.vectorbt_rolling_optimizer.rolling_quantile(data, window, q=q, **kwargs)
                elif operation == 'apply':
                    func = kwargs.get('func')
                    return self.vectorbt_rolling_optimizer.rolling_apply(data, window, func=func, **kwargs)
                else:
                    raise ValueError(f"Unsupported operation: {operation}")
            except Exception as e:
                logger.warning(f"VectorBTRollingOptimizer operation failed: {e}, using pandas fallback")
                return self._pandas_rolling_operation(data, operation, window, **kwargs)

        # Fallback to direct VectorBT or pandas
        if not self._should_use_vectorbt(data):
            return self._pandas_rolling_operation(data, operation, window, **kwargs)

        try:
            if operation == 'mean':
                return rolling_mean(data, window, **kwargs)
            elif operation == 'std':
                return rolling_std(data, window, **kwargs)
            elif operation == 'var':
                return rolling_var(data, window, **kwargs)
            elif operation == 'min':
                return rolling_min(data, window, **kwargs)
            elif operation == 'max':
                return rolling_max(data, window, **kwargs)
            elif operation == 'sum':
                return rolling_sum(data, window, **kwargs)
            else:
                raise ValueError(f"Unsupported operation: {operation}")
        except Exception as e:
            logger.warning(f"VectorBT operation failed: {e}, using pandas fallback")
            return self._pandas_rolling_operation(data, operation, window, **kwargs)

    def _pandas_rolling_operation(self, data: pd.Series, operation: str,
                                 window: int, **kwargs) -> pd.Series:
        """Fallback rolling operation using pandas."""
        if operation == 'mean':
            return self._calculate_sma_vectorized(data, window)
        elif operation == 'std':
            return self._calculate_rolling_std_vectorized(data, window)
        elif operation == 'var':
            return self._optimized_rolling_operation(data, "var", window)
        elif operation == 'min':
            return self._calculate_rolling_min_vectorized(data, window)
        elif operation == 'max':
            return self._calculate_rolling_max_vectorized(data, window)
        elif operation == 'sum':
            return self._calculate_rolling_sum_vectorized(data, window)
        else:
            raise ValueError(f"Unsupported operation: {operation}")

class GeneralTrendFeatureGenerator(VectorizedFeatureGenerator, VectorBTOptimizationMixin):
    """
    Generator for a general trend feature that combines ADX (strength) and MACD (direction).

    This feature provides a comprehensive trend measure that captures both:
    - Trend strength (via ADX)
    - Trend direction (via MACD)

    The general trend is calculated as:
    general_trend = ADX_normalized * MACD_normalized

    Where:
    - ADX_normalized: ADX value normalized to [0, 1] range
    - MACD_normalized: MACD value normalized to [-1, 1] range
    """

    def __init__(self,
                 adx_period: int = 14,
                 macd_fast: int = 12,
                 macd_slow: int = 26,
                 macd_signal: int = 9,
                 sma_period: int = 20,
                 use_sma_instead_of_macd: bool = False,
                 config: Optional[FeatureConfig] = None):
        """
        Initialize General Trend Feature Generator.

        Args:
            adx_period: Period for ADX calculation (default 14)
            macd_fast: Fast period for MACD (default 12)
            macd_slow: Slow period for MACD (default 26)
            macd_signal: Signal period for MACD (default 9)
            sma_period: Period for SMA if using SMA instead of MACD (default 20)
            use_sma_instead_of_macd: Whether to use SMA instead of MACD for direction (default False)
        """
        if config is None:
            config = self._create_default_config(
                adx_period, macd_fast, macd_slow, macd_signal, sma_period, use_sma_instead_of_macd
            )
        super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)

        self.adx_period = adx_period
        self.macd_fast = macd_fast
        self.macd_slow = macd_slow
        self.macd_signal = macd_signal
        self.sma_period = sma_period
        self.use_sma_instead_of_macd = use_sma_instead_of_macd

        # Initialize VectorBT optimizer
        if ROLLING_OPTIMIZER_AVAILABLE:
            self.vectorbt_rolling_optimizer = get_vectorbt_rolling_optimizer(enable_gpu=False, enable_parallel=True)
        else:
            self.vectorbt_rolling_optimizer = None

        # Initialize Unified Vectorization Manager
        if UNIFIED_MANAGER_AVAILABLE:
            self.unified_manager = get_unified_vectorization_manager()
        else:
            self.unified_manager = None

    @classmethod
    def _create_default_config(cls, adx_period: int, macd_fast: int, macd_slow: int,
                              macd_signal: int, sma_period: int, use_sma_instead_of_macd: bool) -> FeatureConfig:
        """Create default configuration for the general trend feature."""
        direction_method = "SMA" if use_sma_instead_of_macd else "MACD"
        name = f"general_trend_adx{adx_period}_{direction_method.lower()}"

        if use_sma_instead_of_macd:
            description = f"General trend combining ADX({adx_period}) strength with SMA({sma_period}) direction"
            required_columns = ["close", "high", "low"]
        else:
            description = f"General trend combining ADX({adx_period}) strength with MACD({macd_fast},{macd_slow},{macd_signal}) direction"
            required_columns = ["close", "high", "low"]

        return FeatureConfig(
            name=name,
            category=FeatureCategory.TREND,
            description=description,
            required_columns=required_columns,
            optional_columns=["open", "volume"],
            default_lookback=max(adx_period * 2, macd_slow * 2 if not use_sma_instead_of_macd else sma_period * 2),
            min_lookback=max(adx_period, macd_slow if not use_sma_instead_of_macd else sma_period),
            max_lookback=max(adx_period * 3, macd_slow * 3 if not use_sma_instead_of_macd else sma_period * 3),
            parameters={
                'adx_period': adx_period,
                'macd_fast': macd_fast,
                'macd_slow': macd_slow,
                'macd_signal': macd_signal,
                'sma_period': sma_period,
                'use_sma_instead_of_macd': use_sma_instead_of_macd
            },
            matrix_optimized=True,
            gpu_accelerated=False
        )

    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate the general trend feature combining ADX and MACD/SMA."""
        if data.empty:
            return pd.Series(dtype=float, index=data.index, name=self.config.name)

        # Optimize DataFrame for processing
        if self.unified_manager:
            data = self.unified_manager.optimize_dataframe(data)

        try:
            # Calculate ADX for trend strength
            adx_values = self._calculate_adx(data)

            # Calculate direction indicator (MACD or SMA)
            if self.use_sma_instead_of_macd:
                direction_values = self._calculate_sma_direction(data)
            else:
                direction_values = self._calculate_macd_direction(data)

            # Combine ADX (strength) and direction indicator
            general_trend = self._combine_trend_components(adx_values, direction_values)

            return general_trend.rename(self.config.name)

        except Exception as e:
            self.logger.warning(f"General trend calculation failed: {e}, using fallback")
            return self._generate_fallback_trend(data)

    def _calculate_adx(self, data: pd.DataFrame) -> pd.Series:
        """Calculate ADX for trend strength."""
        high = data['high']
        low = data['low']
        close = data['close']

        if len(high) < self.adx_period or len(low) < self.adx_period or len(close) < self.adx_period:
            return pd.Series(np.nan, index=data.index)

        # Calculate True Range
        tr = np.maximum.reduce([
            high - low,
            np.abs(high - close.shift(1)),
            np.abs(low - close.shift(1))
        ])

        # Calculate Directional Movement
        dm_plus = np.maximum(high - high.shift(1), 0)
        dm_minus = np.maximum(low.shift(1) - low, 0)

        # Use VectorBT optimization if available
        if self.vectorbt_optimizer and self._should_use_vectorbt(close):
            try:
                dm_plus_mean = self.vectorbt_rolling_optimizer.rolling_mean(dm_plus, self.adx_period)
                dm_minus_mean = self.vectorbt_rolling_optimizer.rolling_mean(dm_minus, self.adx_period)
                tr_mean = self.vectorbt_rolling_optimizer.rolling_mean(tr, self.adx_period)
            except Exception as e:
                self.logger.warning(f"VectorBT ADX calculation failed: {e}, using pandas fallback")
                dm_plus_mean = dm_plus.rolling(self.adx_period).mean()
                dm_minus_mean = dm_minus.rolling(self.adx_period).mean()
                tr_mean = tr.rolling(self.adx_period).mean()
        else:
            dm_plus_mean = dm_plus.rolling(self.adx_period).mean()
            dm_minus_mean = dm_minus.rolling(self.adx_period).mean()
            tr_mean = tr.rolling(self.adx_period).mean()

        # Calculate Directional Indicators
        di_plus = 100 * (dm_plus_mean / tr_mean)
        di_minus = 100 * (dm_minus_mean / tr_mean)

        # Calculate ADX
        dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)

        if self.vectorbt_optimizer and self._should_use_vectorbt(pd.Series(dx)):
            try:
                adx = self.vectorbt_rolling_optimizer.rolling_mean(pd.Series(dx), self.adx_period)
            except Exception as e:
                self.logger.warning(f"VectorBT ADX rolling mean failed: {e}, using pandas fallback")
                adx = pd.Series(dx).rolling(self.adx_period).mean()
        else:
            adx = pd.Series(dx).rolling(self.adx_period).mean()

        return adx

    def _calculate_macd_direction(self, data: pd.DataFrame) -> pd.Series:
        """Calculate MACD for trend direction."""
        close = data['close']

        if len(close) < self.macd_slow:
            return pd.Series(np.nan, index=data.index)

        # Calculate EMAs
        ema_fast = close.ewm(span=self.macd_fast).mean()
        ema_slow = close.ewm(span=self.macd_slow).mean()

        # Calculate MACD line
        macd = ema_fast - ema_slow

        # Calculate signal line
        signal = macd.ewm(span=self.macd_signal).mean()

        # Calculate MACD histogram (MACD - Signal)
        macd_histogram = macd - signal

        return macd_histogram

    def _calculate_sma_direction(self, data: pd.DataFrame) -> pd.Series:
        """Calculate SMA-based direction indicator."""
        close = data['close']

        if len(close) < self.sma_period:
            return pd.Series(np.nan, index=data.index)

        # Calculate SMA
        if self.vectorbt_optimizer and self._should_use_vectorbt(close):
            try:
                sma = self.vectorbt_rolling_optimizer.rolling_mean(close, self.sma_period)
            except Exception as e:
                self.logger.warning(f"VectorBT SMA calculation failed: {e}, using pandas fallback")
                sma = close.rolling(self.sma_period).mean()
        else:
            sma = close.rolling(self.sma_period).mean()

        # Calculate price position relative to SMA (normalized)
        price_position = (close - sma) / sma

        return price_position

    def _combine_trend_components(self, adx_values: pd.Series, direction_values: pd.Series) -> pd.Series:
        """Combine ADX (strength) and direction indicator into general trend."""
        # Normalize ADX to [0, 1] range (ADX is typically 0-100)
        adx_normalized = adx_values / 100.0

        # Normalize direction to [-1, 1] range
        if self.use_sma_instead_of_macd:
            # For SMA, use tanh to bound the values
            direction_normalized = np.tanh(direction_values)
        else:
            # For MACD, normalize using rolling statistics
            if len(direction_values.dropna()) > 0:
                rolling_std = direction_values.rolling(min(20, len(direction_values))).std()
                direction_normalized = direction_values / (rolling_std * 2)  # Scale by 2 standard deviations
                direction_normalized = np.clip(direction_normalized, -1, 1)
            else:
                direction_normalized = pd.Series(0, index=direction_values.index)

        # Combine: general_trend = ADX_strength * direction
        general_trend = adx_normalized * direction_normalized

        return general_trend

    def _generate_fallback_trend(self, data: pd.DataFrame) -> pd.Series:
        """Generate fallback trend feature using simple SMA."""
        close = data['close']
        period = min(self.sma_period, len(close))

        if len(close) >= period:
            sma = close.rolling(period).mean()
            return ((close - sma) / sma).rename(self.config.name)
        else:
            return pd.Series(np.nan, index=data.index, name=self.config.name)

    def _should_use_vectorbt(self, data: pd.Series) -> bool:
        """Determine if VectorBT should be used based on data size and availability."""
        return (VECTORBT_AVAILABLE and
                len(data) > 100 and
                self.vectorbt_optimizer is not None and
                not data.isna().all())

def create_general_trend_generators(adx_periods: List[int] = None,
                                  macd_configs: List[Dict[str, int]] = None,
                                  sma_periods: List[int] = None,
                                  use_sma_variants: bool = True) -> List[FeatureGenerator]:
    """
    Create general trend feature generators with various configurations.

    Args:
        adx_periods: List of ADX periods to use (default: [14, 21])
        macd_configs: List of MACD configurations (default: [{"fast": 12, "slow": 26, "signal": 9}])
        sma_periods: List of SMA periods for SMA-based direction (default: [20, 50])
        use_sma_variants: Whether to create SMA-based variants (default: True)

    Returns:
        List of GeneralTrendFeatureGenerator instances
    """
    if adx_periods is None:
        adx_periods = [14, 21]

    if macd_configs is None:
        macd_configs = [
            {"fast": 12, "slow": 26, "signal": 9},
            {"fast": 8, "slow": 21, "signal": 5}
        ]

    if sma_periods is None:
        sma_periods = [20, 50]

    generators = []

    # Create MACD-based general trend generators
    for adx_period in adx_periods:
        for macd_config in macd_configs:
            generator = GeneralTrendFeatureGenerator(
                adx_period=adx_period,
                macd_fast=macd_config["fast"],
                macd_slow=macd_config["slow"],
                macd_signal=macd_config["signal"],
                use_sma_instead_of_macd=False
            )
            generators.append(generator)

    # Create SMA-based general trend generators if requested
    if use_sma_variants:
        for adx_period in adx_periods:
            for sma_period in sma_periods:
                generator = GeneralTrendFeatureGenerator(
                    adx_period=adx_period,
                    sma_period=sma_period,
                    use_sma_instead_of_macd=True
                )
                generators.append(generator)

    return generators
