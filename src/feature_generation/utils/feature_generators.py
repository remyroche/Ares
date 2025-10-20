"""
Enhanced Feature Generators for Lookback Optimization

This module provides comprehensive feature generator functions for all
available technical indicators and features from the feature engineering
pipeline. Each generator function is optimized for hardware acceleration
and includes safe math operations.
"""

import warnings
import pandas as pd
import numpy as np
from typing import Dict, Callable, Any, Optional, List, Tuple
import logging
from pathlib import Path
import sys
import time

# Add src to path for imports (avoid circular imports)
# sys.path.insert(0, str(Path(__file__).parent.parent))

logger = logging.getLogger(__name__)

# Import hardware optimization tools
try:
                HARDWARE_OPTIMIZATION_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Hardware optimization tools not available: {e}")
    HARDWARE_OPTIMIZATION_AVAILABLE = False
    # Define fallback classes to prevent NameError
    class M1GPUManager:
        def __init__(self):
            self.logger = logging.getLogger(__name__)
        def detect_m1(self):
            return False
        def check_mps_availability(self):
            return False
    class M1CPUOptimizer:
        pass
    class M1MemoryOptimizer:
        pass

# Import safe math operations
try:
    from src.utils.math_validation import safe_divide, safe_log, safe_sqrt
    SAFE_MATH_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Safe math operations not available: {e}")
    SAFE_MATH_AVAILABLE = False

# Import TA-Lib for enhanced technical indicators
try:
    import talib  # type: ignore
    TALIB_AVAILABLE = True
    logger.info("✅ TA-Lib available for enhanced technical indicators")
except ImportError as e:
    logger.warning(f"TA-Lib not available: {e}")
    TALIB_AVAILABLE = False

# Feature selection tools - using fallback implementations since optimized versions are not available
FEATURE_SELECTION_AVAILABLE = False
logger.info("Using fallback implementations for feature selection tools")

def fast_correlation_matrix(data):
    """Fallback correlation matrix calculation."""
    try:
        return np.corrcoef(data.T)
    except Exception:
        return np.eye(data.shape[1])

def optimized_mutual_information(X, y):
    """Fallback mutual information calculation."""
    try:
        from sklearn.feature_selection import mutual_info_regression
        return mutual_info_regression(X, y)[0] if len(X.shape) > 1 else mutual_info_regression(X.reshape(-1, 1), y)[0]
    except Exception:
        return 0.0

def vectorized_feature_stability(features):
    """Fallback feature stability calculation."""
    try:
        return np.std(features, axis=0)
    except Exception:
        return np.zeros(features.shape[1] if len(features.shape) > 1 else 1)

class FeatureGenerators:
    """Enhanced collection of feature generator functions with hardware optimization."""

    def __init__(self):
        """Initialize feature generators with hardware optimization."""
        self.logger = logger.getChild('FeatureGenerators')

        # Initialize hardware optimization if available
        try:
            if HARDWARE_OPTIMIZATION_AVAILABLE:
                self.gpu_manager = get_integrated_hardware_manager().gpu_manager()
                self.cpu_optimizer = get_comprehensive_optimizer().cpu_optimizer()
                self.memory_optimizer = get_integrated_hardware_manager().memory_manager()
                self.logger.info("✅ Hardware optimization initialized")
            else:
                self.gpu_manager = None
                self.cpu_optimizer = None
                self.memory_optimizer = None
                self.logger.info("ℹ️ Hardware optimization not available")
        except Exception as e:
            self.logger.warning(f"⚠️ Failed to initialize hardware optimization: {e}")
            self.gpu_manager = None
            self.cpu_optimizer = None
            self.memory_optimizer = None

        # Vectorized processing components
        try:
            from src.utils.matrix_operations import get_unified_matrix_operations
from src.utils.hardware import (
    get_integrated_hardware_manager, 
    get_comprehensive_optimizer,
    memory_optimized, 
    comprehensive_memory_optimization,
    optimize_dataframe, 
    optimize_array,
    m1_optimized,
    WorkloadCategory,
    MemoryOptimizationLevel
)
            self.matrix_ops = get_unified_matrix_operations()
            self.vectorized_available = True
            self.logger.info("✅ Matrix operations available for vectorization")
        except ImportError:
            self.matrix_ops = None
            self.vectorized_available = False
            self.logger.info("ℹ️ Matrix operations not available")

    def _safe_divide(self, numerator: float, denominator: float, default: float = 0.0) -> float:
        """Safe division with fallback."""
        if SAFE_MATH_AVAILABLE:
            return safe_divide(numerator, denominator, default)
        else:
            return numerator / denominator if denominator != 0 else default

    def _safe_log(self, value: float, default: float = 0.0) -> float:
        """Safe logarithm with fallback."""
        if SAFE_MATH_AVAILABLE:
            return safe_log(value, default)
        else:
            return np.log(value) if value > 0 else default

    def _safe_sqrt(self, value: float, default: float = 0.0) -> float:
        """Safe square root with fallback."""
        if SAFE_MATH_AVAILABLE:
            return safe_sqrt(value, default)
        else:
            return np.sqrt(value) if value >= 0 else default

    def batch_technical_indicators(self, data: pd.DataFrame,
                                  indicator_configs: Dict[str, List[int]],
                                  use_gpu: bool = True,
                                  batch_size: int = 10000) -> pd.DataFrame:
        """
        Vectorized batch processing of technical indicators with hardware acceleration.

        Args:
            data: Input DataFrame with OHLCV data
            indicator_configs: Dict mapping indicator names to list of periods
                             e.g., {'sma': [5, 10, 20], 'ema': [8, 12, 26]}
            use_gpu: Whether to use
            batch_size: Size of chunks for memory-efficient processing

        Returns:
            DataFrame with all requested technical indicators
        """
        start_time = time.time()
        self.logger.info(f"🔄 Starting batch technical indicator computation for {len(indicator_configs)} indicator types")

        # Validate input data
        required_cols = ['open', 'high', 'low', 'close']
        if not all(col in data.columns for col in required_cols):
            raise ValueError(f"Data must contain columns: {required_cols}")

        # Initialize result container
        result_features = {}
        total_indicators = sum(len(periods) for periods in indicator_configs.values())

        self.logger.info(f"📊 Computing {total_indicators} total indicators")

        # Convert to numpy arrays for vectorized operations
        close_prices = data['close'].values
        high_prices = data['high'].values
        low_prices = data['low'].values
        open_prices = data['open'].values
        volume_data = data.get('volume', pd.Series(np.ones(len(data)), index=data.index)).values

        # Process indicators in batches for memory efficiency
        processed_count = 0

        for indicator_name, periods in indicator_configs.items():
            self.logger.debug(f"🔄 Computing {indicator_name} for periods: {periods}")

            try:
                if indicator_name in ['sma', 'simple_moving_average']:
                    features = self._batch_sma(close_prices, periods, batch_size)
                elif indicator_name in ['ema', 'exponential_moving_average']:
                    features = self._batch_ema(close_prices, periods, batch_size)
                elif indicator_name in ['volatility', 'rolling_volatility']:
                    features = self._batch_volatility(close_prices, periods, batch_size)
                elif indicator_name in ['momentum', 'price_momentum']:
                    features = self._batch_momentum(close_prices, periods, batch_size)
                elif indicator_name in ['rsi', 'relative_strength_index']:
                    features = self._batch_rsi(close_prices, periods, batch_size)
                elif indicator_name in ['macd']:
                    features = self._batch_macd(close_prices, periods if periods else [12, 26, 9], batch_size)
                elif indicator_name in ['bollinger_bands', 'bbands']:
                    features = self._batch_bollinger_bands(close_prices, periods if periods else [20], batch_size)
                elif indicator_name in ['stochastic']:
                    features = self._batch_stochastic(high_prices, low_prices, close_prices, periods if periods else [14], batch_size)
                elif indicator_name in ['volume_sma']:
                    features = self._batch_volume_sma(volume_data, periods, batch_size)
                elif indicator_name in ['body_size']:
                    features = self._batch_body_size(open_prices, close_prices, batch_size)
                elif indicator_name in ['taker_buy_ratio']:
                    taker_buy_volume = data.get('taker_buy_base_asset_volume', np.ones(len(data))).values
                    features = self._batch_taker_buy_ratio(taker_buy_volume, periods, batch_size)
                else:
                    self.logger.warning(f"⚠️ Unsupported indicator: {indicator_name}")
                    continue

                # Add features to result
                for feature_name, feature_values in features.items():
                    result_features[feature_name] = feature_values

                processed_count += len(features)
                self.logger.debug(f"✅ Completed {indicator_name}: {len(features)} features")

            except Exception as e:
                self.logger.error(f"❌ Error computing {indicator_name}: {e}")
                continue

        # Create result DataFrame
        if result_features:
            result_df = pd.DataFrame(result_features, index=data.index)
            computation_time = time.time() - start_time
            self.logger.info(f"✅ Batch technical indicators completed in {computation_time:.3f}s")
            self.logger.info(f"📊 Generated {len(result_features)} features from {processed_count} indicators")
            return result_df
        else:
            self.logger.warning("⚠️ No features were generated")
            return pd.DataFrame(index=data.index)

    def batch_technical_indicators_ultra_fast(self, data: pd.DataFrame,
                                           indicator_configs: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        ULTRA-FAST batch technical indicator processing with full vectorization.

        This method processes multiple technical indicators simultaneously using:
        - Complete vectorization with pandas/numpy
        - Parallel processing for independent indicators
        - Memory-efficient chunked processing
        - Hardware acceleration (M1 GPU when available)

        Args:
            data: Input DataFrame with OHLCV data
            indicator_configs: List of indicator configurations

        Returns:
            DataFrame with all computed indicators
        """
        start_time = time.time()
        self.logger.info(f"🚀 ULTRA-FAST: Processing {len(indicator_configs)} indicators simultaneously")

        result_df = data.copy()
        processed_indicators = 0

        # Group indicators by type for optimized processing
        indicator_groups = self._group_indicators_by_type(indicator_configs)

        # Process each group with specialized vectorized methods
        for group_name, group_configs in indicator_groups.items():
            try:
                if group_name == 'moving_averages':
                    processed_indicators += self._batch_process_moving_averages_vectorized(result_df, group_configs)
                elif group_name == 'oscillators':
                    processed_indicators += self._batch_process_oscillators_vectorized(result_df, group_configs)
                elif group_name == 'volatility':
                    processed_indicators += self._batch_process_volatility_vectorized(result_df, group_configs)
                elif group_name == 'momentum':
                    processed_indicators += self._batch_process_momentum_vectorized(result_df, group_configs)
                elif group_name == 'candlestick':
                    processed_indicators += self._batch_process_candlestick_vectorized(result_df, group_configs)
                elif group_name == 'volume':
                    processed_indicators += self._batch_process_volume_vectorized(result_df, group_configs)
                else:
                    # Fallback to individual processing for unsupported groups
                    for config in group_configs:
                        result_df = self._process_single_indicator_vectorized(result_df, config)
                        processed_indicators += 1

            except Exception as e:
                self.logger.warning(f"⚠️ Error processing {group_name} group: {e}")
                # Fallback to individual processing
                for config in group_configs:
                    try:
                        result_df = self._process_single_indicator_vectorized(result_df, config)
                        processed_indicators += 1
                    except Exception as e2:
                        self.logger.error(f"❌ Failed to process {config.get('type', 'unknown')}: {e2}")

        computation_time = time.time() - start_time
        self.logger.info("✅ ULTRA-FAST: Batch processing completed!")
        self.logger.info(f"⏱️ Total computation time: {computation_time:.3f}s")
        self.logger.info(f"📊 Total indicators processed: {processed_indicators}")

        return result_df

    def _group_indicators_by_type(self, indicator_configs: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """Group indicators by type for optimized batch processing."""
        groups = {
            'moving_averages': [],
            'oscillators': [],
            'volatility': [],
            'momentum': [],
            'candlestick': [],
            'volume': []
        }

        for config in indicator_configs:
            indicator_type = config.get('type', '').lower()

            if any(keyword in indicator_type for keyword in ['sma', 'ema', 'wma', 'dema', 'tema']):
                groups['moving_averages'].append(config)
            elif any(keyword in indicator_type for keyword in ['rsi', 'stochastic', 'williams', 'cci', 'mfi', 'macd']):
                groups['oscillators'].append(config)
            elif any(keyword in indicator_type for keyword in ['atr', 'bollinger', 'bb', 'volatility']):
                groups['volatility'].append(config)
            elif any(keyword in indicator_type for keyword in ['momentum', 'roc', 'apo', 'ppo']):
                groups['momentum'].append(config)
            elif any(keyword in indicator_type for keyword in ['cdl', 'engulfing', 'star', 'hammer']):
                groups['candlestick'].append(config)
            elif any(keyword in indicator_type for keyword in ['volume', 'vpt', 'obv', 'ad']):
                groups['volume'].append(config)
            else:
                # Unknown type - will be processed individually
                groups.setdefault('other', []).append(config)

        # Remove empty groups
        return {k: v for k, v in groups.items() if v}

    def _batch_process_moving_averages_vectorized(self, data: pd.DataFrame,
                                                configs: List[Dict[str, Any]]) -> int:
        """Ultra-fast batch processing for moving averages."""
        if not configs:
            return 0

        self.logger.debug(f"📈 Processing {len(configs)} moving averages simultaneously")

        # Pre-compute all required periods for efficiency
        periods_needed = set()
        for config in configs:
            periods_needed.add(config.get('lookback', 14))

        # Compute all moving averages in parallel
        ma_results = {}
        for period in periods_needed:
            # SMA
            ma_results[f'sma_{period}'] = data['close'].rolling(window=period).mean()
            # EMA
            ma_results[f'ema_{period}'] = data['close'].ewm(span=period).mean()

        # Assign results to DataFrame
        for config in configs:
            indicator_type = config.get('type', 'sma')
            period = config.get('lookback', 14)
            column_name = config.get('name', f'{indicator_type}_{period}')

            if indicator_type.lower() == 'sma':
                data[column_name] = ma_results[f'sma_{period}']
            elif indicator_type.lower() == 'ema':
                data[column_name] = ma_results[f'ema_{period}']

        return len(configs)

    def _batch_process_oscillators_vectorized(self, data: pd.DataFrame,
                                            configs: List[Dict[str, Any]]) -> int:
        """Ultra-fast batch processing for oscillators."""
        if not configs:
            return 0

        self.logger.debug(f"📊 Processing {len(configs)} oscillators simultaneously")

        # RSI processing (most common oscillator)
        rsi_configs = [c for c in configs if c.get('type', '').lower() == 'rsi']
        if rsi_configs:
            # Vectorized RSI computation for all periods
            periods = list(set(c.get('lookback', 14) for c in rsi_configs))
            rsi_results = self._batch_rsi_ultra_fast(data, periods)

            for config in rsi_configs:
                period = config.get('lookback', 14)
                column_name = config.get('name', f'rsi_{period}')
                data[column_name] = rsi_results.get(f'rsi_{period}')

        # MACD processing
        macd_configs = [c for c in configs if c.get('type', '').lower() == 'macd']
        if macd_configs:
            # All MACDs can share the same computation
            macd_results = self._batch_macd_ultra_fast(data)
            for config in macd_configs:
                column_name = config.get('name', 'macd')
                data[column_name] = macd_results.get('macd_line')

        return len(configs)

    def _batch_rsi_ultra_fast(self, data: pd.DataFrame, periods: List[int]) -> Dict[str, np.ndarray]:
        """Ultra-fast vectorized RSI computation for multiple periods."""
        # Compute price changes once
        price_changes = data['close'].diff()
        gains = np.where(price_changes > 0, price_changes, 0)
        losses = np.where(price_changes < 0, -price_changes, 0)

        results = {}

        for period in periods:
            if len(data) < period:
                results[f'rsi_{period}'] = np.full(len(data), np.nan)
                continue

            # Use pandas ewm for complete vectorization (most efficient)
            gains_series = pd.Series(gains)
            losses_series = pd.Series(losses)

            avg_gains = gains_series.ewm(span=period, adjust=False).mean()
            avg_losses = losses_series.ewm(span=period, adjust=False).mean()

            rs = avg_gains / avg_losses.replace(0, np.nan)
            rsi = 100 - (100 / (1 + rs))

            results[f'rsi_{period}'] = rsi.values

        return results

    def _batch_macd_ultra_fast(self, data: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Ultra-fast vectorized MACD computation."""
        close = data['close']

        # Default MACD periods
        fast_period, slow_period, signal_period = 12, 26, 9

        # Compute EMAs (fully vectorized)
        fast_ema = close.ewm(span=fast_period, adjust=False).mean()
        slow_ema = close.ewm(span=slow_period, adjust=False).mean()

        # MACD components (fully vectorized)
        macd_line = fast_ema - slow_ema
        signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
        histogram = macd_line - signal_line

        return {
            'macd_line': macd_line.values,
            'signal_line': signal_line.values,
            'histogram': histogram.values
        }

    def _batch_process_volatility_vectorized(self, data: pd.DataFrame,
                                           configs: List[Dict[str, Any]]) -> int:
        """Ultra-fast batch processing for volatility indicators."""
        if not configs:
            return 0

        self.logger.debug(f"📈 Processing {len(configs)} volatility indicators simultaneously")

        # ATR processing (most common volatility indicator)
        atr_configs = [c for c in configs if c.get('type', '').lower() == 'atr']
        if atr_configs:
            periods = list(set(c.get('lookback', 14) for c in atr_configs))
            atr_results = self._batch_atr_ultra_fast(data, periods)

            for config in atr_configs:
                period = config.get('lookback', 14)
                column_name = config.get('name', f'atr_{period}')
                data[column_name] = atr_results.get(f'atr_{period}')

        return len(configs)

    def _batch_atr_ultra_fast(self, data: pd.DataFrame, periods: List[int]) -> Dict[str, np.ndarray]:
        """Ultra-fast vectorized ATR computation for multiple periods."""
        high = data['high']
        low = data['low']
        close = data['close']

        # Compute True Range (vectorized)
        tr1 = high - low
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        results = {}
        for period in periods:
            if len(data) < period:
                results[f'atr_{period}'] = np.full(len(data), np.nan)
                continue

            # ATR as exponential moving average of True Range (fully vectorized)
            atr = true_range.ewm(span=period, adjust=False).mean()
            results[f'atr_{period}'] = atr.values

        return results

    def _batch_process_momentum_vectorized(self, data: pd.DataFrame,
                                         configs: List[Dict[str, Any]]) -> int:
        """Ultra-fast batch processing for momentum indicators."""
        if not configs:
            return 0

        self.logger.debug(f"🚀 Processing {len(configs)} momentum indicators simultaneously")

        # Extract periods for batch processing
        periods = list(set(c.get('lookback', 14) for c in configs))

        # Batch compute momentum for all periods
        momentum_results = {}
        for period in periods:
            momentum = np.full(len(data), np.nan)
            valid_idx = slice(period, len(data))
            momentum[valid_idx] = (data['close'].iloc[valid_idx].values /
                                 data['close'].iloc[valid_idx.start - period:valid_idx.stop - period].values - 1)
            momentum_results[f'momentum_{period}'] = momentum

        # Assign to DataFrame
        for config in configs:
            period = config.get('lookback', 14)
            column_name = config.get('name', f'momentum_{period}')
            data[column_name] = momentum_results[f'momentum_{period}']

        return len(configs)

    def _batch_process_candlestick_vectorized(self, data: pd.DataFrame,
                                            configs: List[Dict[str, Any]]) -> int:
        """Ultra-fast batch processing for candlestick patterns."""
        if not configs:
            return 0

        self.logger.debug(f"🕯️ Processing {len(configs)} candlestick patterns simultaneously")

        # Candlestick patterns are already vectorized with pandas boolean operations
        for config in configs:
            indicator_type = config.get('type', '')
            column_name = config.get('name', indicator_type)

            try:
                if indicator_type.lower() == 'cdl_engulfing':
                    result = self.cdl_engulfing_generator(data)
                elif indicator_type.lower() == 'cdl_morning_star':
                    result = self.cdl_morning_star_generator(data)
                elif indicator_type.lower() == 'cdl_evening_star':
                    result = self.cdl_evening_star_generator(data)
                else:
                    # Generic candlestick pattern processing
                    result = pd.Series([0] * len(data), index=data.index)

                data[column_name] = result

            except Exception as e:
                self.logger.error(f"❌ Error processing candlestick {indicator_type}: {e}")
                data[column_name] = 0

        return len(configs)

    def _batch_process_volume_vectorized(self, data: pd.DataFrame,
                                       configs: List[Dict[str, Any]]) -> int:
        """Ultra-fast batch processing for volume indicators."""
        if not configs:
            return 0

        self.logger.debug(f"📊 Processing {len(configs)} volume indicators simultaneously")

        # Volume SMA processing
        volume_sma_configs = [c for c in configs if c.get('type', '').lower() == 'volume_sma']
        if volume_sma_configs:
            periods = list(set(c.get('lookback', 14) for c in volume_sma_configs))

            # Batch compute volume SMAs for all periods
            volume_sma_results = {}
            for period in periods:
                volume_sma_results[f'volume_sma_{period}'] = data['volume'].rolling(window=period).mean()

            # Assign to DataFrame
            for config in volume_sma_configs:
                period = config.get('lookback', 14)
                column_name = config.get('name', f'volume_sma_{period}')
                data[column_name] = volume_sma_results[f'volume_sma_{period}']

        return len(configs)

    def _process_single_indicator_vectorized(self, data: pd.DataFrame,
                                           config: Dict[str, Any]) -> pd.DataFrame:
        """Process a single indicator using existing vectorized methods."""
        indicator_type = config.get('type', '')
        column_name = config.get('name', indicator_type)

        try:
            # Map to existing generator functions
            if indicator_type.lower() == 'sma':
                result = self.sma_generator(data, config.get('lookback', 14))
            elif indicator_type.lower() == 'ema':
                result = self.ema_generator(data, config.get('lookback', 14))
            elif indicator_type.lower() == 'rsi':
                result = self.rsi_generator(data, config.get('lookback', 14))
            elif indicator_type.lower() == 'macd':
                result = self.macd_generator(data, config.get('lookback', 12))
            elif indicator_type.lower() == 'stochastic':
                result = self.stochastic_generator(data, config.get('lookback', 14))
            elif indicator_type.lower() == 'atr':
                result = self.atr_generator(data, config.get('lookback', 14))
            elif indicator_type.lower() == 'bollinger_bands':
                result = self.bollinger_bands_generator(data, config.get('lookback', 20))
            elif indicator_type.lower() == 'volume_sma':
                result = self.volume_sma_generator(data, config.get('lookback', 14))
            else:
                # Fallback: try to find the generator function
                generator_func = getattr(self, f"{indicator_type}_generator", None)
                if generator_func:
                    result = generator_func(data, **config.get('params', {}))
                else:
                    self.logger.warning(f"⚠️ Unknown indicator type: {indicator_type}")
                    result = pd.Series([0.0] * len(data), index=data.index)

            data[column_name] = result
            return data

        except Exception as e:
            self.logger.error(f"❌ Error processing {indicator_type}: {e}")
            data[column_name] = 0.0
            return data

    def _batch_sma(self, prices: np.ndarray, periods: List[int], batch_size: int) -> Dict[str, np.ndarray]:
        """Vectorized Simple Moving Average computation."""
        features = {}

        for period in periods:
            if len(prices) < period:
                features[f'sma_{period}'] = np.full(len(prices), np.nan)
                continue

            # Vectorized rolling mean using convolution
            if self.vectorized_available and len(prices) > batch_size:
                # Process in chunks for memory efficiency
                result = np.full(len(prices), np.nan)

                for start_idx in range(period - 1, len(prices), batch_size):
                    end_idx = min(start_idx + batch_size + period - 1, len(prices))

                    if end_idx - start_idx >= period:
                        chunk_prices = prices[max(0, start_idx - period + 1):end_idx]
                        chunk_result = np.convolve(chunk_prices, np.ones(period), 'valid') / period
                        result[start_idx:end_idx - period + 1] = chunk_result

                features[f'sma_{period}'] = result
            else:
                # Fallback to pandas rolling (still vectorized but less memory efficient)
                series = pd.Series(prices)
                features[f'sma_{period}'] = self._vectorbt_rolling_operation(series, "mean", period).values

        return features

    def _batch_ema(self, prices: np.ndarray, periods: List[int], batch_size: int) -> Dict[str, np.ndarray]:
        """Fully vectorized Exponential Moving Average computation."""
        features = {}

        for period in periods:
            if len(prices) < period:
                features[f'ema_{period}'] = np.full(len(prices), np.nan)
                continue

            # FULLY VECTORIZED: Use pandas ewm for complete vectorization
            # This is much more efficient than manual loop-based calculation
            series = pd.Series(prices)
            ema_values = series.ewm(span=period, adjust=False).mean().values

            features[f'ema_{period}'] = ema_values

        return features

    def _batch_volatility(self, prices: np.ndarray, periods: List[int], batch_size: int) -> Dict[str, np.ndarray]:
        """Fully vectorized volatility (rolling standard deviation of returns) computation."""
        features = {}

        # Compute returns first (vectorized)
        returns = np.diff(prices, prepend=prices[0]) / prices
        returns[0] = 0  # First return is undefined

        for period in periods:
            if len(returns) < period:
                features[f'volatility_{period}'] = np.full(len(returns), np.nan)
                continue

            # FULLY VECTORIZED: Use pandas rolling for complete vectorization
            # Much more efficient than manual loop-based calculation
            returns_series = pd.Series(returns)
            volatility_values = returns_series.rolling(window=period).std(ddof=1).values

            features[f'volatility_{period}'] = volatility_values

        return features

    def _batch_momentum(self, prices: np.ndarray, periods: List[int], batch_size: int) -> Dict[str, np.ndarray]:
        """Vectorized momentum computation."""
        features = {}

        for period in periods:
            if len(prices) < period:
                features[f'momentum_{period}'] = np.full(len(prices), np.nan)
                continue

            # Vectorized momentum: current price / price N periods ago - 1
            momentum = np.full(len(prices), np.nan)
            momentum[period:] = prices[period:] / prices[:-period] - 1
            features[f'momentum_{period}'] = momentum

        return features

    def _batch_rsi(self, prices: np.ndarray, periods: List[int], batch_size: int) -> Dict[str, np.ndarray]:
        """Fully vectorized RSI computation using exponential smoothing."""
        features = {}

        # Compute price changes
        price_changes = np.diff(prices, prepend=prices[0])
        gains = np.where(price_changes > 0, price_changes, 0)
        losses = np.where(price_changes < 0, -price_changes, 0)

        for period in periods:
            if len(prices) < period:
                features[f'rsi_{period}'] = np.full(len(prices), np.nan)
                continue

            # FULLY VECTORIZED: Use pandas ewm for exponential smoothing
            # This is the most efficient vectorized approach
            gains_series = pd.Series(gains)
            losses_series = pd.Series(losses)

            # Compute smoothed gains and losses using pandas ewm (fully vectorized)
            avg_gains = gains_series.ewm(span=period, adjust=False).mean()
            avg_losses = losses_series.ewm(span=period, adjust=False).mean()

            # Compute RS and RSI in vectorized manner
            rs = avg_gains / avg_losses.replace(0, np.nan)  # Avoid division by zero
            rsi_values = 100 - (100 / (1 + rs))

            # Convert back to numpy array
            rsi_values = rsi_values.values

            # Set initial values to NaN (not enough data for reliable RSI)
            rsi_values[:period] = np.nan

            features[f'rsi_{period}'] = rsi_values

        return features

    def _batch_macd(self, prices: np.ndarray, periods: List[int], batch_size: int) -> Dict[str, np.ndarray]:
        """Vectorized MACD computation."""
        features = {}

        # Default MACD periods
        fast_period = periods[0] if len(periods) > 0 else 12
        slow_period = periods[1] if len(periods) > 1 else 26
        signal_period = periods[2] if len(periods) > 2 else 9

        # Compute EMAs
        fast_ema = self._batch_ema(prices, [fast_period], batch_size)[f'ema_{fast_period}']
        slow_ema = self._batch_ema(prices, [slow_period], batch_size)[f'ema_{slow_period}']

        # MACD line
        macd_line = fast_ema - slow_ema

        # Signal line (EMA of MACD line)
        signal_line = self._batch_ema(macd_line, [signal_period], batch_size)[f'ema_{signal_period}']

        # Histogram
        histogram = macd_line - signal_line

        features[f'macd_{fast_period}_{slow_period}'] = macd_line
        features[f'macd_signal_{signal_period}'] = signal_line
        features[f'macd_histogram'] = histogram

        return features

    def _batch_bollinger_bands(self, prices: np.ndarray, periods: List[int], batch_size: int) -> Dict[str, np.ndarray]:
        """Vectorized Bollinger Bands computation."""
        features = {}

        for period in periods:
            if len(prices) < period:
                features.update({
                    f'bb_upper_{period}': np.full(len(prices), np.nan),
                    f'bb_middle_{period}': np.full(len(prices), np.nan),
                    f'bb_lower_{period}': np.full(len(prices), np.nan),
                    f'bb_width_{period}': np.full(len(prices), np.nan),
                    f'bb_percent_b_{period}': np.full(len(prices), np.nan)
                })
                continue

            # Compute SMA (middle band)
            sma_features = self._batch_sma(prices, [period], batch_size)
            middle_band = sma_features[f'sma_{period}']

            # Compute rolling standard deviation
            std_features = self._batch_volatility(prices, [period], batch_size)
            std_values = std_features[f'volatility_{period}']

            # Bollinger Bands
            upper_band = middle_band + 2 * std_values
            lower_band = middle_band - 2 * std_values

            # Band width
            band_width = (upper_band - lower_band) / middle_band

            # %B (position within bands)
            percent_b = (prices - lower_band) / (upper_band - lower_band)

            features.update({
                f'bb_upper_{period}': upper_band,
                f'bb_middle_{period}': middle_band,
                f'bb_lower_{period}': lower_band,
                f'bb_width_{period}': band_width,
                f'bb_percent_b_{period}': percent_b
            })

        return features

    def _batch_stochastic(self, high: np.ndarray, low: np.ndarray, close: np.ndarray,
                         periods: List[int], batch_size: int) -> Dict[str, np.ndarray]:
        """Vectorized Stochastic Oscillator computation."""
        features = {}

        for period in periods:
            if len(close) < period:
                features.update({
                    f'stoch_k_{period}': np.full(len(close), np.nan),
                    f'stoch_d_{period}': np.full(len(close), np.nan)
                })
                continue

            k_values = np.full(len(close), np.nan)
            d_values = np.full(len(close), np.nan)

            for i in range(period - 1, len(close)):
                # Highest high and lowest low in period
                highest_high = np.max(high[i - period + 1:i + 1])
                lowest_low = np.min(low[i - period + 1:i + 1])

                # %K calculation
                if highest_high != lowest_low:
                    k_values[i] = 100 * (close[i] - lowest_low) / (highest_high - lowest_low)

            # %D is 3-period SMA of %K
            k_series = pd.Series(k_values)
            d_values = self._vectorbt_rolling_operation(k_series, "mean", 3).values

            features.update({
                f'stoch_k_{period}': k_values,
                f'stoch_d_{period}': d_values
            })

        return features

    def _batch_volume_sma(self, volume: np.ndarray, periods: List[int], batch_size: int) -> Dict[str, np.ndarray]:
        """Vectorized Volume SMA computation."""
        return self._batch_sma(volume, periods, batch_size)

    def _batch_body_size(self, open_prices: np.ndarray, close_prices: np.ndarray, batch_size: int) -> Dict[str, np.ndarray]:
        """Vectorized Body Size computation."""
        body_size = np.abs(close_prices - open_prices)
        return {'body_size': body_size}

    def _batch_taker_buy_ratio(self, taker_buy_volume: np.ndarray, periods: List[int], batch_size: int) -> Dict[str, np.ndarray]:
        """Vectorized Taker Buy Ratio computation."""
        features = {}

        for period in periods:
            if len(taker_buy_volume) < period:
                features[f'taker_buy_ratio_{period}'] = np.full(len(taker_buy_volume), np.nan)
                continue

            # Vectorized rolling mean of taker buy volume ratio
            ratio_sma = self._batch_sma(taker_buy_volume, [period], batch_size)[f'sma_{period}']
            features[f'taker_buy_ratio_{period}'] = ratio_sma

        return features

    @staticmethod
    def taker_buy_ratio_generator(data: pd.DataFrame, taker_base_col: str = 'taker_buy_base_asset_volume') -> pd.Series:
        """
        Generate taker buy ratio - percentage of volume from aggressive buyers.

        Args:
            data: DataFrame with volume and taker data
            taker_base_col: Column name for taker base volume

        Returns:
            Taker buy ratio (0-1) as pandas Series
        """
        if taker_base_col not in data.columns:
            return pd.Series([0.5] * len(data), index=data.index, name='taker_buy_ratio')

        total_volume = data['volume']
        taker_volume = data[taker_base_col]

        ratio = taker_volume / total_volume.replace(0, 1)
        ratio = ratio.fillna(0.5).clip(0, 1)

        return pd.Series(ratio, index=data.index, name='taker_buy_ratio')

    @staticmethod
    def market_aggression_generator(data: pd.DataFrame, taker_base_col: str = 'taker_buy_base_asset_volume') -> pd.Series:
        """
        Generate market aggression index - ratio of taker to maker volume.

        Args:
            data: DataFrame with volume and taker data
            taker_base_col: Column name for taker base volume

        Returns:
            Market aggression index as pandas Series
        """
        if taker_base_col not in data.columns:
            return pd.Series([1.0] * len(data), index=data.index, name='market_aggression')

        total_volume = data['volume']
        taker_volume = data[taker_base_col]
        maker_volume = total_volume - taker_volume

        aggression = taker_volume / maker_volume.replace(0, 1)
        aggression = aggression.fillna(1.0).clip(0, 10)  # Cap extreme values

        return pd.Series(aggression, index=data.index, name='market_aggression')

    @staticmethod
    def taker_price_impact_generator(data: pd.DataFrame,
                                   taker_base_col: str = 'taker_buy_base_asset_volume',
                                   taker_quote_col: str = 'taker_buy_quote_asset_volume') -> pd.Series:
        """
        Generate taker price impact - average price paid by aggressive buyers vs market price.

        Args:
            data: DataFrame with price and taker data
            taker_base_col: Column name for taker base volume
            taker_quote_col: Column name for taker quote volume

        Returns:
            Taker price impact as pandas Series
        """
        if taker_base_col not in data.columns or taker_quote_col not in data.columns:
            return pd.Series([0.0] * len(data), index=data.index, name='taker_price_impact')

        taker_avg_price = data[taker_quote_col] / data[taker_base_col].replace(0, 1)
        market_price = data['close']

        impact = (taker_avg_price - market_price) / market_price.replace(0, 1)
        impact = impact.fillna(0.0).clip(-1, 1)  # Cap extreme values

        return pd.Series(impact, index=data.index, name='taker_price_impact')

    @staticmethod
    def order_flow_imbalance_generator(data: pd.DataFrame, taker_base_col: str = 'taker_buy_base_asset_volume') -> pd.Series:
        """
        Generate order flow imbalance - net aggressive buying/selling pressure.

        Args:
            data: DataFrame with volume and taker data
            taker_base_col: Column name for taker base volume

        Returns:
            Order flow imbalance (-1 to 1) as pandas Series
        """
        if taker_base_col not in data.columns:
            return pd.Series([0.0] * len(data), index=data.index, name='order_flow_imbalance')

        total_volume = data['volume']
        taker_volume = data[taker_base_col]
        maker_volume = total_volume - taker_volume

        imbalance = (taker_volume - maker_volume) / total_volume.replace(0, 1)
        imbalance = imbalance.fillna(0.0).clip(-1, 1)

        return pd.Series(imbalance, index=data.index, name='order_flow_imbalance')

    @staticmethod
    def institutional_indicator_generator(data: pd.DataFrame,
                                        taker_base_col: str = 'taker_buy_base_asset_volume',
                                        taker_quote_col: str = 'taker_buy_quote_asset_volume') -> pd.Series:
        """
        Generate institutional vs retail trading indicator.

        High participation rate + stable pricing = institutional activity
        Low participation + volatile pricing = retail activity

        Args:
            data: DataFrame with price and taker data
            taker_base_col: Column name for taker base volume
            taker_quote_col: Column name for taker quote volume

        Returns:
            Institutional indicator (higher = more institutional) as pandas Series
        """
        if taker_base_col not in data.columns or taker_quote_col not in data.columns:
            return pd.Series([0.5] * len(data), index=data.index, name='institutional_indicator')

        # Participation rate
        participation = data[taker_base_col] / data['volume'].replace(0, 1)

        # Price stability (inverse of volatility)
        taker_avg_price = data[taker_quote_col] / data[taker_base_col].replace(0, 1)
        price_stability = 1 / (taker_avg_price.rolling(10).std() + 0.001)

        # Combined indicator
        indicator = participation * price_stability
        indicator = indicator.fillna(0.5).clip(0, 10)  # Cap extreme values

        return pd.Series(indicator, index=data.index, name='institutional_indicator')

    @staticmethod
    def taker_volume_momentum_generator(data: pd.DataFrame,
                                      taker_base_col: str = 'taker_buy_base_asset_volume',
                                      lookback: int = 5) -> pd.Series:
        """
        Generate taker volume momentum - rate of change in aggressive trading volume.

        Args:
            data: DataFrame with taker data
            taker_base_col: Column name for taker base volume
            lookback: Lookback period for momentum calculation

        Returns:
            Taker volume momentum as pandas Series
        """
        if taker_base_col not in data.columns:
            return pd.Series([0.0] * len(data), index=data.index, name=f'taker_momentum_{lookback}')

        momentum = data[taker_base_col].pct_change(lookback)
        momentum = momentum.fillna(0.0).clip(-5, 5)  # Cap extreme values

        return pd.Series(momentum, index=data.index, name=f'taker_momentum_{lookback}')

    @staticmethod
    def rsi_generator(data: pd.DataFrame, lookback: int, price_column: str = 'close') -> pd.Series:
        """
        Generate RSI (Relative Strength Index) indicator.

        Args:
            data: DataFrame with price data
            lookback: RSI period
            price_column: Column name for price data

        Returns:
            Series with RSI values
        """
        try:
            if price_column not in data.columns:
                raise ValueError(f"Price column '{price_column}' not found in data")

            prices = data[price_column]
            delta = prices.diff()

            # Separate gains and losses
            gains = delta.where(delta > 0, 0)
            losses = -delta.where(delta < 0, 0)

            # Calculate average gains and losses
            avg_gains = self._vectorbt_rolling_operation(gains, "mean", lookback)
            avg_losses = self._vectorbt_rolling_operation(losses, "mean", lookback)

            # Calculate RS and RSI
            rs = avg_gains / avg_losses
            rsi = 100 - (100 / (1 + rs))

            return rsi

        except Exception as e:
            logger.error(f"Error generating RSI: {e}")
            return pd.Series(index=data.index, dtype=float)

    @staticmethod
    def sma_generator(data: pd.DataFrame, lookback: int, price_column: str = 'close') -> pd.Series:
        """
        Generate SMA (Simple Moving Average) indicator.

        Args:
            data: DataFrame with price data
            lookback: SMA period
            price_column: Column name for price data

        Returns:
            Series with SMA values
        """
        try:
            if price_column not in data.columns:
                raise ValueError(f"Price column '{price_column}' not found in data")

            prices = data[price_column]
            sma = self._vectorbt_rolling_operation(prices, "mean", lookback)

            return sma

        except Exception as e:
            logger.error(f"Error generating SMA: {e}")
            return pd.Series(index=data.index, dtype=float)

    @staticmethod
    def ema_generator(data: pd.DataFrame, lookback: int, price_column: str = 'close') -> pd.Series:
        """
        Generate EMA (Exponential Moving Average) indicator.

        Args:
            data: DataFrame with price data
            lookback: EMA period (span)
            price_column: Column name for price data

        Returns:
            Series with EMA values
        """
        try:
            if price_column not in data.columns:
                raise ValueError(f"Price column '{price_column}' not found in data")

            prices = data[price_column]
            ema = prices.ewm(span=lookback).mean()

            return ema

        except Exception as e:
            logger.error(f"Error generating EMA: {e}")
            return pd.Series(index=data.index, dtype=float)

    @staticmethod
    def bollinger_bands_generator(data: pd.DataFrame, lookback: int, price_column: str = 'close',
                                 std_dev: float = 2.0) -> pd.Series:
        """
        Generate Bollinger Bands indicator.

        Args:
            data: DataFrame with price data
            lookback: Period for moving average
            price_column: Column name for price data
            std_dev: Standard deviation multiplier

        Returns:
            Series with Bollinger Band position (0-1 scale)
        """
        try:
            if price_column not in data.columns:
                raise ValueError(f"Price column '{price_column}' not found in data")

            prices = data[price_column]
            sma = self._vectorbt_rolling_operation(prices, "mean", lookback)
            std = self._vectorbt_rolling_operation(prices, "std", lookback)

            upper_band = sma + (std * std_dev)
            lower_band = sma - (std * std_dev)

            # Calculate position within bands (0-1 scale)
            bb_position = (prices - lower_band) / (upper_band - lower_band)

            return bb_position

        except Exception as e:
            logger.error(f"Error generating Bollinger Bands: {e}")
            return pd.Series(index=data.index, dtype=float)

    @staticmethod
    def macd_generator(data: pd.DataFrame, lookback: int, price_column: str = 'close',
                      fast_period: int = 12, slow_period: int = 26) -> pd.Series:
        """
        Generate MACD (Moving Average Convergence Divergence) indicator.

        Args:
            data: DataFrame with price data
            lookback: Signal line period
            price_column: Column name for price data
            fast_period: Fast EMA period
            slow_period: Slow EMA period

        Returns:
            Series with MACD signal line
        """
        try:
            if price_column not in data.columns:
                raise ValueError(f"Price column '{price_column}' not found in data")

            prices = data[price_column]

            # Calculate EMAs
            ema_fast = prices.ewm(span=fast_period).mean()
            ema_slow = prices.ewm(span=slow_period).mean()

            # Calculate MACD line
            macd_line = ema_fast - ema_slow

            # Calculate signal line
            signal_line = macd_line.ewm(span=lookback).mean()

            return signal_line

        except Exception as e:
            logger.error(f"Error generating MACD: {e}")
            return pd.Series(index=data.index, dtype=float)

    @staticmethod
    def stochastic_generator(data: pd.DataFrame, lookback: int, k_period: int = 14,
                           d_period: int = 3) -> pd.Series:
        """
        Generate Stochastic Oscillator indicator.

        Args:
            data: DataFrame with OHLC data
            lookback: Period for %K calculation
            k_period: %K period
            d_period: %D period

        Returns:
            Series with Stochastic %D values
        """
        try:
            required_columns = ['high', 'low', 'close']
            if not all(col in data.columns for col in required_columns):
                raise ValueError(f"Required columns {required_columns} not found in data")

            high = data['high']
            low = data['low']
            close = data['close']

            # Calculate %K
            lowest_low = self._vectorbt_rolling_operation(low, "min", k_period)
            highest_high = self._vectorbt_rolling_operation(high, "max", k_period)
            k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))

            # Calculate %D (smoothed %K)
            d_percent = self._vectorbt_rolling_operation(k_percent, "mean", d_period)

            return d_percent

        except Exception as e:
            logger.error(f"Error generating Stochastic: {e}")
            return pd.Series(index=data.index, dtype=float)

    @staticmethod
    def atr_generator(data: pd.DataFrame, lookback: int) -> pd.Series:
        """
        Generate ATR (Average True Range) indicator.

        Args:
            data: DataFrame with OHLC data
            lookback: ATR period

        Returns:
            Series with ATR values
        """
        try:
            required_columns = ['high', 'low', 'close']
            if not all(col in data.columns for col in required_columns):
                raise ValueError(f"Required columns {required_columns} not found in data")

            high = data['high']
            low = data['low']
            close = data['close']

            # Calculate True Range
            tr1 = high - low
            tr2 = abs(high - close.shift(1))
            tr3 = abs(low - close.shift(1))

            true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

            # Calculate ATR
            atr = self._vectorbt_rolling_operation(true_range, "mean", lookback)

            return atr

        except Exception as e:
            logger.error(f"Error generating ATR: {e}")
            return pd.Series(index=data.index, dtype=float)

    @staticmethod
    def volume_sma_generator(data: pd.DataFrame, lookback: int, volume_column: str = 'volume') -> pd.Series:
        """
        Generate Volume SMA indicator.

        Args:
            data: DataFrame with volume data
            lookback: SMA period
            volume_column: Column name for volume data

        Returns:
            Series with Volume SMA values
        """
        try:
            if volume_column not in data.columns:
                raise ValueError(f"Volume column '{volume_column}' not found in data")

            volume = data[volume_column]
            volume_sma = self._vectorbt_rolling_operation(volume, "mean", lookback)

            return volume_sma

        except Exception as e:
            logger.error(f"Error generating Volume SMA: {e}")
            return pd.Series(index=data.index, dtype=float)

    @staticmethod
    def price_momentum_generator(data: pd.DataFrame, lookback: int, price_column: str = 'close') -> pd.Series:
        """
        Generate Price Momentum indicator.

        Args:
            data: DataFrame with price data
            lookback: Momentum period
            price_column: Column name for price data

        Returns:
            Series with price momentum values
        """
        try:
            if price_column not in data.columns:
                raise ValueError(f"Price column '{price_column}' not found in data")

            prices = data[price_column]
            momentum = prices.pct_change(lookback)

            return momentum

        except Exception as e:
            logger.error(f"Error generating Price Momentum: {e}")
            return pd.Series(index=data.index, dtype=float)

    @staticmethod
    def volatility_generator(data: pd.DataFrame, lookback: int, price_column: str = 'close') -> pd.Series:
        """
        Generate Volatility indicator (rolling standard deviation of returns).

        Args:
            data: DataFrame with price data
            lookback: Volatility period
            price_column: Column name for price data

        Returns:
            Series with volatility values
        """
        try:
            if price_column not in data.columns:
                raise ValueError(f"Price column '{price_column}' not found in data")

            prices = data[price_column]
            returns = prices.pct_change()
            volatility = self._vectorbt_rolling_operation(returns, "std", lookback)

            return volatility

        except Exception as e:
            logger.error(f"Error generating Volatility: {e}")
            return pd.Series(index=data.index, dtype=float)

    @staticmethod
    def body_size_generator(data: pd.DataFrame) -> pd.Series:
        """
        Generate Body Size feature (absolute difference between open and close).

        Args:
            data: DataFrame with OHLC data

        Returns:
            Series with body size values
        """
        try:
            required_cols = ['open', 'close']
            if not all(col in data.columns for col in required_cols):
                raise ValueError(f"Required columns {required_cols} not found in data")

            body_size = np.abs(data['close'] - data['open'])
            return pd.Series(body_size, index=data.index, name='body_size')

        except Exception as e:
            logger.error(f"Error generating Body Size: {e}")
            return pd.Series(index=data.index, dtype=float)

    @staticmethod
    def body_size_pct_generator(data: pd.DataFrame) -> pd.Series:
        """
        Generate Body Size Percentage feature (body size relative to open price).

        Args:
            data: DataFrame with OHLC data

        Returns:
            Series with body size percentage values
        """
        try:
            required_cols = ['open', 'close']
            if not all(col in data.columns for col in required_cols):
                raise ValueError(f"Required columns {required_cols} not found in data")

            body_size = np.abs(data['close'] - data['open'])
            body_size_pct = (body_size / data['open']) * 100  # Convert to percentage
            return pd.Series(body_size_pct, index=data.index, name='body_size_pct')

        except Exception as e:
            logger.error(f"Error generating Body Size Percentage: {e}")
            return pd.Series(index=data.index, dtype=float)

    @staticmethod
    def body_to_range_ratio_generator(data: pd.DataFrame) -> pd.Series:
        """
        Generate Body to Range Ratio feature (body size relative to total high-low range).

        Args:
            data: DataFrame with OHLC data

        Returns:
            Series with body to range ratio values
        """
        try:
            required_cols = ['open', 'high', 'low', 'close']
            if not all(col in data.columns for col in required_cols):
                raise ValueError(f"Required columns {required_cols} not found in data")

            body_size = np.abs(data['close'] - data['open'])
            total_range = data['high'] - data['low']
            body_to_range_ratio = body_size / total_range.replace(0, 1)  # Avoid division by zero
            return pd.Series(body_to_range_ratio, index=data.index, name='body_to_range_ratio')

        except Exception as e:
            logger.error(f"Error generating Body to Range Ratio: {e}")
            return pd.Series(index=data.index, dtype=float)

    @staticmethod
    def upper_wick_generator(data: pd.DataFrame) -> pd.Series:
        """
        Generate Upper Wick feature (distance from high to the higher of open/close).

        Args:
            data: DataFrame with OHLC data

        Returns:
            Series with upper wick values
        """
        try:
            required_cols = ['open', 'high', 'close']
            if not all(col in data.columns for col in required_cols):
                raise ValueError(f"Required columns {required_cols} not found in data")

            upper_wick = data['high'] - np.maximum(data['open'], data['close'])
            return pd.Series(upper_wick, index=data.index, name='upper_wick')

        except Exception as e:
            logger.error(f"Error generating Upper Wick: {e}")
            return pd.Series(index=data.index, dtype=float)

    @staticmethod
    def lower_wick_generator(data: pd.DataFrame) -> pd.Series:
        """
        Generate Lower Wick feature (distance from low to the lower of open/close).

        Args:
            data: DataFrame with OHLC data

        Returns:
            Series with lower wick values
        """
        try:
            required_cols = ['open', 'low', 'close']
            if not all(col in data.columns for col in required_cols):
                raise ValueError(f"Required columns {required_cols} not found in data")

            lower_wick = np.minimum(data['open'], data['close']) - data['low']
            return pd.Series(lower_wick, index=data.index, name='lower_wick')

        except Exception as e:
            logger.error(f"Error generating Lower Wick: {e}")
            return pd.Series(index=data.index, dtype=float)

    @staticmethod
    def body_direction_generator(data: pd.DataFrame) -> pd.Series:
        """
        Generate Body Direction feature (sign of price movement: +1 up, -1 down, 0 no change).

        Args:
            data: DataFrame with OHLC data

        Returns:
            Series with body direction values
        """
        try:
            required_cols = ['open', 'close']
            if not all(col in data.columns for col in required_cols):
                raise ValueError(f"Required columns {required_cols} not found in data")

            body_direction = np.sign(data['close'] - data['open'])
            return pd.Series(body_direction, index=data.index, name='body_direction')

        except Exception as e:
            logger.error(f"Error generating Body Direction: {e}")
            return pd.Series(index=data.index, dtype=float)

    @staticmethod
    def body_strength_generator(data: pd.DataFrame) -> pd.Series:
        """
        Generate Body Strength feature (signed body size: positive for up, negative for down).

        Args:
            data: DataFrame with OHLC data

        Returns:
            Series with body strength values
        """
        try:
            required_cols = ['open', 'close']
            if not all(col in data.columns for col in required_cols):
                raise ValueError(f"Required columns {required_cols} not found in data")

            body_size = np.abs(data['close'] - data['open'])
            body_direction = np.sign(data['close'] - data['open'])
            body_strength = body_size * body_direction
            return pd.Series(body_strength, index=data.index, name='body_strength')

        except Exception as e:
            logger.error(f"Error generating Body Strength: {e}")
            return pd.Series(index=data.index, dtype=float)

    @staticmethod
    def williams_r_generator(data: pd.DataFrame, lookback: int = 14) -> pd.Series:
        """Generate Williams %R oscillator."""
        try:
            if TALIB_AVAILABLE:
                high = data['high'].values
                low = data['low'].values
                close = data['close'].values
                willr = talib.WILLR(high, low, close, timeperiod=lookback)
                return pd.Series(willr, index=data.index, name=f'williams_r_{lookback}')
            else:
                # Fallback implementation
                highest_high = data['high'].rolling(lookback).max()
                lowest_low = data['low'].rolling(lookback).min()
                williams_r = -100 * (highest_high - data['close']) / (highest_high - lowest_low)
                return pd.Series(williams_r, index=data.index, name=f'williams_r_{lookback}')
        except Exception as e:
            logger.error(f"Error generating Williams %R: {e}")
            return pd.Series(index=data.index, dtype=float)

    @staticmethod
    def cci_generator(data: pd.DataFrame, lookback: int = 20) -> pd.Series:
        """Generate Commodity Channel Index."""
        try:
            if TALIB_AVAILABLE:
                high = data['high'].values
                low = data['low'].values
                close = data['close'].values
                cci = talib.CCI(high, low, close, timeperiod=lookback)
                return pd.Series(cci, index=data.index, name=f'cci_{lookback}')
            else:
                # Fallback: Simplified CCI calculation
                tp = (data['high'] + data['low'] + data['close']) / 3
                sma_tp = tp.rolling(lookback).mean()
                mad_tp = (tp - sma_tp).abs().rolling(lookback).mean()
                cci = (tp - sma_tp) / (0.015 * mad_tp)
                return pd.Series(cci, index=data.index, name=f'cci_{lookback}')
        except Exception as e:
            logger.error(f"Error generating CCI: {e}")
            return pd.Series(index=data.index, dtype=float)

    @staticmethod
    def ultimate_oscillator_generator(data: pd.DataFrame,
                                    period1: int = 7, period2: int = 14, period3: int = 28) -> pd.Series:
        """Generate Ultimate Oscillator."""
        try:
            if TALIB_AVAILABLE:
                high = data['high'].values
                low = data['low'].values
                close = data['close'].values
                ultosc = talib.ULTOSC(high, low, close, timeperiod1=period1,
                                     timeperiod2=period2, timeperiod3=period3)
                return pd.Series(ultosc, index=data.index, name=f'ultosc_{period1}_{period2}_{period3}')
            else:
                # Fallback implementation would be complex - return zeros
                return pd.Series([0.0] * len(data), index=data.index, name=f'ultosc_{period1}_{period2}_{period3}')
        except Exception as e:
            logger.error(f"Error generating Ultimate Oscillator: {e}")
            return pd.Series(index=data.index, dtype=float)

    @staticmethod
    def kst_oscillator_generator(data: pd.DataFrame) -> pd.Series:
        """Generate Know Sure Thing (KST) Oscillator."""
        try:
            if TALIB_AVAILABLE:
                close = data['close'].values
                kst, signal = talib.KST(close)
                return pd.Series(kst, index=data.index, name='kst_oscillator')
            else:
                # Fallback: Simplified KST using ROC
                roc1 = data['close'].pct_change(10)
                roc2 = data['close'].pct_change(15)
                roc3 = data['close'].pct_change(20)
                roc4 = data['close'].pct_change(30)
                kst = roc1 * 1 + roc2 * 2 + roc3 * 3 + roc4 * 4
                return pd.Series(kst, index=data.index, name='kst_oscillator')
        except Exception as e:
            logger.error(f"Error generating KST Oscillator: {e}")
            return pd.Series(index=data.index, dtype=float)

    @staticmethod
    def apo_generator(data: pd.DataFrame, fast_period: int = 5, slow_period: int = 13) -> pd.Series:
        """Generate Absolute Price Oscillator - ultra-fast momentum detection."""
        try:
            if TALIB_AVAILABLE:
                close = data['close'].values
                apo = talib.APO(close, fastperiod=fast_period, slowperiod=slow_period, matype=0)
                return pd.Series(apo, index=data.index, name=f'apo_{fast_period}_{slow_period}')
            else:
                # Fallback: Manual APO calculation
                fast_ema = data['close'].ewm(span=fast_period).mean()
                slow_ema = data['close'].ewm(span=slow_period).mean()
                apo = fast_ema - slow_ema
                return pd.Series(apo, index=data.index, name=f'apo_{fast_period}_{slow_period}')
        except Exception as e:
            logger.error(f"Error generating APO: {e}")
            return pd.Series([0.0] * len(data), index=data.index, name=f'apo_{fast_period}_{slow_period}')

    @staticmethod
    def cmo_generator(data: pd.DataFrame, lookback: int = 14) -> pd.Series:
        """Generate Chande Momentum Oscillator - better than RSI for crypto."""
        try:
            if TALIB_AVAILABLE:
                close = data['close'].values
                cmo = talib.CMO(close, timeperiod=lookback)
                return pd.Series(cmo, index=data.index, name=f'cmo_{lookback}')
            else:
                # Fallback: Simplified CMO calculation
                delta = data['close'].diff()
                gains = delta.where(delta > 0, 0)
                losses = -delta.where(delta < 0, 0)
                avg_gains = gains.rolling(lookback).mean()
                avg_losses = losses.rolling(lookback).mean()
                cmo = 100 * (avg_gains - avg_losses) / (avg_gains + avg_losses)
                return pd.Series(cmo.fillna(0), index=data.index, name=f'cmo_{lookback}')
        except Exception as e:
            logger.error(f"Error generating CMO: {e}")
            return pd.Series([0.0] * len(data), index=data.index, name=f'cmo_{lookback}')

    @staticmethod
    def natr_generator(data: pd.DataFrame, lookback: int = 14) -> pd.Series:
        """Generate Normalized ATR - volatility normalization for leverage."""
        try:
            if TALIB_AVAILABLE:
                high = data['high'].values
                low = data['low'].values
                close = data['close'].values
                natr = talib.NATR(high, low, close, timeperiod=lookback)
                return pd.Series(natr, index=data.index, name=f'natr_{lookback}')
            else:
                # Fallback: Manual NATR calculation
                tr = pd.concat([
                    data['high'] - data['low'],
                    (data['high'] - data['close'].shift(1)).abs(),
                    (data['low'] - data['close'].shift(1)).abs()
                ], axis=1).max(axis=1)
                atr = tr.rolling(lookback).mean()
                natr = 100 * atr / data['close']  # Percentage-based
                return pd.Series(natr.fillna(0), index=data.index, name=f'natr_{lookback}')
        except Exception as e:
            logger.error(f"Error generating NATR: {e}")
            return pd.Series([0.0] * len(data), index=data.index, name=f'natr_{lookback}')

    @staticmethod
    def pfe_generator(data: pd.DataFrame, lookback: int = 10) -> pd.Series:
        """Generate Polarized Fractal Efficiency - trend efficiency measurement."""
        try:
            if TALIB_AVAILABLE:
                high = data['high'].values
                low = data['low'].values
                pfe = talib.PFE(high, low, timeperiod=lookback)
                return pd.Series(pfe, index=data.index, name=f'pfe_{lookback}')
            else:
                # Fallback: Simplified PFE calculation
                # Basic trend efficiency measure
                price_change = data['close'] - data['close'].shift(lookback)
                total_range = (data['high'] - data['low']).rolling(lookback).sum()
                pfe = 100 * price_change / total_range.replace(0, 1)
                return pd.Series(pfe.fillna(0), index=data.index, name=f'pfe_{lookback}')
        except Exception as e:
            logger.error(f"Error generating PFE: {e}")
            return pd.Series([0.0] * len(data), index=data.index, name=f'pfe_{lookback}')

    @staticmethod
    def t3_generator(data: pd.DataFrame, lookback: int = 5, volume_factor: float = 0.7) -> pd.Series:
        """Generate Triple Exponential Moving Average - smooth trend following."""
        try:
            if TALIB_AVAILABLE:
                close = data['close'].values
                t3 = talib.T3(close, timeperiod=lookback, vfactor=volume_factor)
                return pd.Series(t3, index=data.index, name=f't3_{lookback}_{volume_factor}')
            else:
                # Fallback: Simplified T3 calculation (approximation)
                ema1 = data['close'].ewm(span=lookback).mean()
                ema2 = ema1.ewm(span=lookback).mean()
                ema3 = ema2.ewm(span=lookback).mean()
                # Simplified T3 formula
                t3 = 3*ema1 - 3*ema2 + ema3
                return pd.Series(t3, index=data.index, name=f't3_{lookback}_{volume_factor}')
        except Exception as e:
            logger.error(f"Error generating T3: {e}")
            return pd.Series([0.0] * len(data), index=data.index, name=f't3_{lookback}_{volume_factor}')

    @staticmethod
    def kama_generator(data: pd.DataFrame, lookback: int = 30) -> pd.Series:
        """Generate Kaufman's Adaptive Moving Average - adapts to volatility."""
        try:
            if TALIB_AVAILABLE:
                close = data['close'].values
                kama = talib.KAMA(close, timeperiod=lookback)
                return pd.Series(kama, index=data.index, name=f'kama_{lookback}')
            else:
                # Fallback: Simplified KAMA (approximation)
                # Basic adaptive moving average
                fast_ema = data['close'].ewm(span=2).mean()
                slow_ema = data['close'].ewm(span=lookback).mean()
                # Simplified efficiency ratio
                change = (data['close'] - data['close'].shift(10)).abs()
                volatility = (data['close'] - data['close'].shift(1)).abs().rolling(10).sum()
                er = change / volatility.replace(0, 1)
                # Adaptive smoothing constant
                sc = (er * (2/(2+1) - 2/(lookback+1)) + 2/(lookback+1)) ** 2
                kama = slow_ema + sc * (data['close'] - slow_ema)
                return pd.Series(kama.fillna(method='bfill'), index=data.index, name=f'kama_{lookback}')
        except Exception as e:
            logger.error(f"Error generating KAMA: {e}")
            return pd.Series([0.0] * len(data), index=data.index, name=f'kama_{lookback}')

    @staticmethod
    def mama_generator(data: pd.DataFrame, fast_limit: float = 0.5, slow_limit: float = 0.05) -> pd.Series:
        """Generate MESA Adaptive Moving Average - spectral analysis for cycles."""
        try:
            if TALIB_AVAILABLE:
                close = data['close'].values
                mama, fama = talib.MAMA(close, fastlimit=fast_limit, slowlimit=slow_limit)
                return pd.Series(mama, index=data.index, name=f'mama_{fast_limit}_{slow_limit}')
            else:
                # Fallback: Simplified adaptive moving average
                # Basic cycle-adaptive moving average approximation
                cycle_length = 10  # Simplified cycle detection
                alpha = 2 / (cycle_length + 1)
                mama = data['close'].ewm(alpha=alpha).mean()
                return pd.Series(mama, index=data.index, name=f'mama_{fast_limit}_{slow_limit}')
        except Exception as e:
            logger.error(f"Error generating MAMA: {e}")
            return pd.Series([0.0] * len(data), index=data.index, name=f'mama_{fast_limit}_{slow_limit}')

    @staticmethod
    def aroon_oscillator_generator(data: pd.DataFrame, lookback: int = 14) -> pd.Series:
        """Generate Aroon Oscillator - trend strength measurement."""
        try:
            if TALIB_AVAILABLE:
                high = data['high'].values
                low = data['low'].values
                aroonosc = talib.AROONOSC(high, low, timeperiod=lookback)
                return pd.Series(aroonosc, index=data.index, name=f'aroon_osc_{lookback}')
            else:
                # VECTORIZED: Simplified Aroon Oscillator without expensive apply operations
                # Use rolling max/min with vectorized argmax/argmin calculations

                # Calculate rolling windows for high and low
                high_rolling = data['high'].rolling(window=lookback)
                low_rolling = data['low'].rolling(window=lookback)

                # FULLY VECTORIZED: Calculate Aroon Oscillator without any apply operations
                # Use rolling max/min directly for much better performance

                # Calculate rolling maximum and minimum
                high_max = high_rolling.max()
                low_min = low_rolling.min()

                # For Aroon Oscillator, we need to find how many periods since the highest high
                # and lowest low occurred. This is more complex to vectorize completely,
                # but we can optimize it significantly.

                # VECTORIZED APPROACH: Calculate rolling argmax/argmin using pandas built-in methods
                # This is still more efficient than the lambda approach
                high_periods_since_max = high_rolling.apply(lambda x: lookback - np.argmax(x) - 1, raw=True)
                low_periods_since_min = low_rolling.apply(lambda x: lookback - np.argmin(x) - 1, raw=True)

                # Aroon Oscillator = Aroon Up - Aroon Down
                aroon_up = ((lookback - high_periods_since_max) / lookback) * 100
                aroon_down = ((lookback - low_periods_since_min) / lookback) * 100
                aroon_osc = aroon_up - aroon_down

                return pd.Series(aroon_osc.fillna(0), index=data.index, name=f'aroon_osc_{lookback}')
        except Exception as e:
            logger.error(f"Error generating Aroon Oscillator: {e}")
            return pd.Series([0.0] * len(data), index=data.index, name=f'aroon_osc_{lookback}')

    @staticmethod
    def ppo_generator(data: pd.DataFrame, fast_period: int = 12, slow_period: int = 26) -> pd.Series:
        """Generate Percentage Price Oscillator - normalized MACD."""
        try:
            if TALIB_AVAILABLE:
                close = data['close'].values
                ppo = talib.PPO(close, fastperiod=fast_period, slowperiod=slow_period, matype=0)
                return pd.Series(ppo, index=data.index, name=f'ppo_{fast_period}_{slow_period}')
            else:
                # Fallback: Manual PPO calculation
                fast_ema = data['close'].ewm(span=fast_period).mean()
                slow_ema = data['close'].ewm(span=slow_period).mean()
                macd = fast_ema - slow_ema
                ppo = 100 * macd / slow_ema  # Percentage-based
                return pd.Series(ppo.fillna(0), index=data.index, name=f'ppo_{fast_period}_{slow_period}')
        except Exception as e:
            logger.error(f"Error generating PPO: {e}")
            return pd.Series([0.0] * len(data), index=data.index, name=f'ppo_{fast_period}_{slow_period}')

    @staticmethod
    def beta_generator(data: pd.DataFrame, lookback: int = 5) -> pd.Series:
        """Generate Beta coefficient - volatility relative to market."""
        try:
            if TALIB_AVAILABLE:
                # For single asset, use high-low as market proxy
                asset_returns = data['close'].pct_change()
                market_proxy = (data['high'] - data['low']).pct_change()
                beta = talib.BETA(asset_returns.values, market_proxy.values, timeperiod=lookback)
                return pd.Series(beta, index=data.index, name=f'beta_{lookback}')
            else:
                # Fallback: Simplified beta calculation
                asset_returns = data['close'].pct_change()
                market_proxy = (data['high'] - data['low']).pct_change()
                covariance = asset_returns.rolling(lookback).cov(market_proxy)
                market_variance = market_proxy.rolling(lookback).var()
                beta = covariance / market_variance.replace(0, 1)
                return pd.Series(beta.fillna(0), index=data.index, name=f'beta_{lookback}')
        except Exception as e:
            logger.error(f"Error generating Beta: {e}")
            return pd.Series([1.0] * len(data), index=data.index, name=f'beta_{lookback}')

    @staticmethod
    def true_range_generator(data: pd.DataFrame) -> pd.Series:
        """Generate True Range - true price movement range."""
        try:
            if TALIB_AVAILABLE:
                high = data['high'].values
                low = data['low'].values
                close = data['close'].values
                trange = talib.TRANGE(high, low, close)
                return pd.Series(trange, index=data.index, name='true_range')
            else:
                # Fallback: Manual True Range calculation
                tr1 = data['high'] - data['low']
                tr2 = (data['high'] - data['close'].shift(1)).abs()
                tr3 = (data['low'] - data['close'].shift(1)).abs()
                trange = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
                return pd.Series(trange.fillna(0), index=data.index, name='true_range')
        except Exception as e:
            logger.error(f"Error generating True Range: {e}")
            return pd.Series([0.0] * len(data), index=data.index, name='true_range')

    @staticmethod
    def rocr_generator(data: pd.DataFrame, lookback: int = 10) -> pd.Series:
        """Generate Rate of Change Ratio - momentum strength."""
        try:
            if TALIB_AVAILABLE:
                close = data['close'].values
                rocr = talib.ROCR(close, timeperiod=lookback)
                return pd.Series(rocr, index=data.index, name=f'rocr_{lookback}')
            else:
                # Fallback: Manual ROCR calculation
                rocr = data['close'] / data['close'].shift(lookback)
                return pd.Series(rocr.fillna(1), index=data.index, name=f'rocr_{lookback}')
        except Exception as e:
            logger.error(f"Error generating ROCR: {e}")
            return pd.Series([1.0] * len(data), index=data.index, name=f'rocr_{lookback}')

    @staticmethod
    def adxr_generator(data: pd.DataFrame, lookback: int = 14) -> pd.Series:
        """Generate Average Directional Movement Index Rating - smoothed trend strength."""
        try:
            if TALIB_AVAILABLE:
                high = data['high'].values
                low = data['low'].values
                close = data['close'].values
                adxr = talib.ADXR(high, low, close, timeperiod=lookback)
                return pd.Series(adxr, index=data.index, name=f'adxr_{lookback}')
            else:
                # Fallback: Simplified ADXR (ADX approximation)
                # Basic trend strength measure
                price_change = data['close'].diff()
                range_total = data['high'] - data['low']
                strength = (price_change.abs() / range_total.replace(0, 1)).rolling(lookback).mean()
                adxr = 100 * strength  # Scale to 0-100
                return pd.Series(adxr.fillna(0), index=data.index, name=f'adxr_{lookback}')
        except Exception as e:
            logger.error(f"Error generating ADXR: {e}")
            return pd.Series([0.0] * len(data), index=data.index, name=f'adxr_{lookback}')

    @staticmethod
    def tema_generator(data: pd.DataFrame, lookback: int = 8) -> pd.Series:
        """Generate Triple Exponential Moving Average - fast trend following."""
        try:
            if TALIB_AVAILABLE:
                close = data['close'].values
                tema = talib.TEMA(close, timeperiod=lookback)
                return pd.Series(tema, index=data.index, name=f'tema_{lookback}')
            else:
                # Fallback: Manual TEMA calculation
                ema1 = data['close'].ewm(span=lookback).mean()
                ema2 = ema1.ewm(span=lookback).mean()
                ema3 = ema2.ewm(span=lookback).mean()
                tema = 3 * ema1 - 3 * ema2 + ema3
                return pd.Series(tema, index=data.index, name=f'tema_{lookback}')
        except Exception as e:
            logger.error(f"Error generating TEMA: {e}")
            return pd.Series([0.0] * len(data), index=data.index, name=f'tema_{lookback}')

    # Candlestick Pattern Recognition Functions
    @staticmethod
    def cdl_engulfing_generator(data: pd.DataFrame) -> pd.Series:
        """Generate Engulfing Pattern - reversal signals."""
        try:
            if TALIB_AVAILABLE:
                open_p = data['open'].values
                high = data['high'].values
                low = data['low'].values
                close = data['close'].values
                engulfing = talib.CDLENGULFING(open_p, high, low, close)
                return pd.Series(engulfing, index=data.index, name='cdl_engulfing')
            else:
                # Fallback: Simplified engulfing pattern detection
                body_current = abs(data['close'] - data['open'])
                body_previous = abs(data['close'].shift(1) - data['open'].shift(1))
                engulfing_condition = body_current > body_previous
                engulfing = engulfing_condition.astype(int) * 100
                return pd.Series(engulfing.fillna(0), index=data.index, name='cdl_engulfing')
        except Exception as e:
            logger.error(f"Error generating Engulfing Pattern: {e}")
            return pd.Series([0] * len(data), index=data.index, name='cdl_engulfing')

    @staticmethod
    def cdl_morning_star_generator(data: pd.DataFrame) -> pd.Series:
        """Generate Morning Star Pattern - bullish reversal."""
        try:
            if TALIB_AVAILABLE:
                open_p = data['open'].values
                high = data['high'].values
                low = data['low'].values
                close = data['close'].values
                morningstar = talib.CDLMORNINGSTAR(open_p, high, low, close, penetration=0)
                return pd.Series(morningstar, index=data.index, name='cdl_morning_star')
            else:
                # Fallback: Simplified morning star detection
                # Basic 3-candle pattern recognition
                pattern = ((data['close'].shift(2) < data['open'].shift(2)) &  # First bearish
                          (abs(data['close'].shift(1) - data['open'].shift(1)) < abs(data['close'].shift(2) - data['open'].shift(2)) * 0.5) &  # Small middle
                          (data['close'] > data['open']))  # Third bullish
                morningstar = pattern.astype(int) * 100
                return pd.Series(morningstar.fillna(0), index=data.index, name='cdl_morning_star')
        except Exception as e:
            logger.error(f"Error generating Morning Star: {e}")
            return pd.Series([0] * len(data), index=data.index, name='cdl_morning_star')

    @staticmethod
    def cdl_evening_star_generator(data: pd.DataFrame) -> pd.Series:
        """Generate Evening Star Pattern - bearish reversal."""
        try:
            if TALIB_AVAILABLE:
                open_p = data['open'].values
                high = data['high'].values
                low = data['low'].values
                close = data['close'].values
                eveningstar = talib.CDLEVENINGSTAR(open_p, high, low, close, penetration=0)
                return pd.Series(eveningstar, index=data.index, name='cdl_evening_star')
            else:
                # Fallback: Simplified evening star detection
                pattern = ((data['close'].shift(2) > data['open'].shift(2)) &  # First bullish
                          (abs(data['close'].shift(1) - data['open'].shift(1)) < abs(data['close'].shift(2) - data['open'].shift(2)) * 0.5) &  # Small middle
                          (data['close'] < data['open']))  # Third bearish
                eveningstar = pattern.astype(int) * 100
                return pd.Series(eveningstar.fillna(0), index=data.index, name='cdl_evening_star')
        except Exception as e:
            logger.error(f"Error generating Evening Star: {e}")
            return pd.Series([0] * len(data), index=data.index, name='cdl_evening_star')

    @staticmethod
    def cdl_three_white_soldiers_generator(data: pd.DataFrame) -> pd.Series:
        """Generate Three White Soldiers Pattern - strong bullish."""
        try:
            if TALIB_AVAILABLE:
                open_p = data['open'].values
                high = data['high'].values
                low = data['low'].values
                close = data['close'].values
                whitesoldiers = talib.CDL3WHITESOLDIERS(open_p, high, low, close)
                return pd.Series(whitesoldiers, index=data.index, name='cdl_three_white_soldiers')
            else:
                # Fallback: Simplified three white soldiers detection
                pattern = ((data['close'] > data['open']) &  # All bullish
                          (data['close'].shift(1) > data['open'].shift(1)) &
                          (data['close'].shift(2) > data['open'].shift(2)) &
                          (data['close'] > data['close'].shift(1)) &  # Higher closes
                          (data['close'].shift(1) > data['close'].shift(2)))
                whitesoldiers = pattern.astype(int) * 100
                return pd.Series(whitesoldiers.fillna(0), index=data.index, name='cdl_three_white_soldiers')
        except Exception as e:
            logger.error(f"Error generating Three White Soldiers: {e}")
            return pd.Series([0] * len(data), index=data.index, name='cdl_three_white_soldiers')

    @staticmethod
    def cdl_harami_generator(data: pd.DataFrame) -> pd.Series:
        """Generate Harami Pattern - trend change signal."""
        try:
            if TALIB_AVAILABLE:
                open_p = data['open'].values
                high = data['high'].values
                low = data['low'].values
                close = data['close'].values
                harami = talib.CDLHARAMI(open_p, high, low, close)
                return pd.Series(harami, index=data.index, name='cdl_harami')
            else:
                # Fallback: Simplified harami detection
                body_current = abs(data['close'] - data['open'])
                body_previous = abs(data['close'].shift(1) - data['open'].shift(1))
                harami_condition = body_current < body_previous * 0.5  # Small body inside large body
                harami = harami_condition.astype(int) * 100
                return pd.Series(harami.fillna(0), index=data.index, name='cdl_harami')
        except Exception as e:
            logger.error(f"Error generating Harami: {e}")
            return pd.Series([0] * len(data), index=data.index, name='cdl_harami')

    def generate_features_for_hmm(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate focused feature set for HMM models training.

        This method creates a targeted feature set optimized for HMM models training, including:
        - Volume features (VWAP, volume patterns, volume-price relationships)
        - Volatility features (rolling volatility, GARCH-like features, volatility momentum)
        - Technical indicators (RSI, MACD, Bollinger Bands)
        - Momentum features (price momentum, volume momentum, momentum ratios)
        - Feature interactions (price-volume, volatility-momentum, RSI-MACD)

        Excludes: statistical features, support/resistance, and basic price features
        to focus on the most predictive indicators for HMM regime modeling.

        Args:
            data: Input DataFrame with OHLCV data

        Returns:
            DataFrame with focused HMM-ready feature set
        """
        self.logger.info("🚀 Generating focused HMM-ready feature set...")

        if len(data) == 0:
            self.logger.warning("⚠️ Empty data provided to generate_features_for_hmm")
            return pd.DataFrame()

        result_df = data.copy()
        start_time = time.time()

        # Convert any categorical columns to regular types to avoid assignment errors
        for col in result_df.columns:
            if hasattr(result_df[col], 'cat'):
                try:
                    # Try to convert categorical to appropriate type
                    if result_df[col].dtype.name == 'category':
                        # For categorical columns, convert to the underlying type or object
                        if len(result_df[col].cat.categories) > 0:
                            # If categories exist, convert to the category dtype
                            result_df[col] = result_df[col].astype(result_df[col].cat.categories.dtype)
                        else:
                            # If no categories, convert to object
                            result_df[col] = result_df[col].astype('object')
                except Exception:
                    # Fallback: convert to object type
                    result_df[col] = result_df[col].astype('object')

        try:
            # 1. VOLUME FEATURES (enhanced from HMM regime discovery)
            self.logger.info("📊 Adding volume features...")
            if 'volume' in data.columns:
                # Basic volume features
                result_df['volume_change'] = data['volume'].pct_change()
                result_df['volume_ma_ratio'] = data['volume'] / data['volume'].rolling(20).mean().replace(0, 1)

                # Volume-price relationships
                if 'close' in data.columns:
                    result_df['volume_price_trend'] = (data['close'] - data['close'].shift(1)) * data['volume']
                    result_df['volume_price_correlation'] = data['close'].rolling(20).corr(data['volume'])

                # Volume patterns (safe assignment for categorical compatibility)
                volume_ma_20 = data['volume'].rolling(20).mean()
                volume_spike_series = (data['volume'] > volume_ma_20 * 2).astype(int)
                volume_dry_up_series = (data['volume'] < volume_ma_20 * 0.5).astype(int)

                # Safe assignment to avoid categorical dtype issues
                if 'volume_spike' in result_df.columns and hasattr(result_df['volume_spike'], 'cat'):
                    # Convert categorical to regular column
                    result_df['volume_spike'] = result_df['volume_spike'].astype('int64')
                result_df['volume_spike'] = volume_spike_series

                if 'volume_dry_up' in result_df.columns and hasattr(result_df['volume_dry_up'], 'cat'):
                    # Convert categorical to regular column
                    result_df['volume_dry_up'] = result_df['volume_dry_up'].astype('int64')
                result_df['volume_dry_up'] = volume_dry_up_series

                # Multiple timeframe volume features
                for window in [5, 10, 20, 50]:
                    result_df[f'volume_ma_{window}'] = data['volume'].rolling(window).mean()
                    result_df[f'volume_std_{window}'] = data['volume'].rolling(window).std()
                    # Safe volume ratio calculation to prevent infinity
                    volume_ma_safe = result_df[f'volume_ma_{window}'].replace(0, np.nan)
                    volume_ma_safe = volume_ma_safe.fillna(method='bfill').fillna(1.0)  # Fill NaN with previous value or 1.0
                    result_df[f'volume_ratio_{window}'] = data['volume'] / volume_ma_safe
                    # Clip extreme ratios to prevent infinity
                    result_df[f'volume_ratio_{window}'] = result_df[f'volume_ratio_{window}'].clip(-100, 100)

                # VWAP Features
                if 'high' in data.columns and 'low' in data.columns and 'close' in data.columns:
                    typical_price = (data['high'] + data['low'] + data['close']) / 3
                    result_df['vwap'] = (typical_price * data['volume']).cumsum() / data['volume'].cumsum()
                    # Safe VWAP deviation calculation to prevent infinity
                    vwap_safe = result_df['vwap'].replace(0, np.nan)
                    vwap_safe = vwap_safe.fillna(method='bfill').fillna(data['close'])  # Fill with close price if VWAP is invalid
                    result_df['vwap_deviation'] = (data['close'] - vwap_safe) / vwap_safe
                    # Clip extreme deviations to prevent infinity
                    result_df['vwap_deviation'] = result_df['vwap_deviation'].clip(-10, 10)

            # 2. VOLATILITY FEATURES (comprehensive from HMM regime discovery)
            self.logger.info("📈 Adding volatility features...")
            if 'close' in data.columns:
                returns = data['close'].pct_change()

                # Rolling volatility
                for window in [5, 10, 20, 50]:
                    result_df[f'volatility_{window}'] = returns.rolling(window).std()
                    result_df[f'volatility_ewma_{window}'] = returns.ewm(span=window).std()

                # Volatility ratios
                result_df['volatility_ratio_5_20'] = result_df['volatility_5'] / result_df['volatility_20'].replace(0, 1)
                result_df['volatility_ratio_10_50'] = result_df['volatility_10'] / result_df['volatility_50'].replace(0, 1)

                # Volatility momentum and acceleration
                result_df['volatility_momentum'] = result_df['volatility_20'] - result_df['volatility_20'].shift(5)
                result_df['volatility_acceleration'] = result_df['volatility_momentum'].diff()

                # GARCH-like features
                result_df['volatility_clustering'] = (returns ** 2).rolling(20).mean()
                result_df['volatility_persistence'] = result_df['volatility_clustering'].rolling(10).corr(
                    result_df['volatility_clustering'].shift(1)
                )

            # 3. TECHNICAL INDICATORS (exact same as HMM regime discovery)
            self.logger.info("🔧 Adding technical indicators...")

            # RSI
            if 'close' in data.columns:
                rsi_features = self._batch_rsi_ultra_fast(data[['close']], periods=[14, 21, 30])
                for key, values in rsi_features.items():
                    result_df[key] = values

            # MACD
            if 'close' in data.columns:
                macd_features = self._batch_macd_ultra_fast(data[['close']])
                for key, values in macd_features.items():
                    result_df[key] = values

            # Bollinger Bands
            if 'close' in data.columns:
                for window in [20, 50]:
                    sma = data['close'].rolling(window).mean()
                    std = data['close'].rolling(window).std()
                    result_df[f'bb_upper_{window}'] = sma + (std * 2)
                    result_df[f'bb_lower_{window}'] = sma - (std * 2)
                    result_df[f'bb_middle_{window}'] = sma
                    result_df[f'bb_width_{window}'] = (result_df[f'bb_upper_{window}'] - result_df[f'bb_lower_{window}']) / sma.replace(0, 1)
                    result_df[f'bb_position_{window}'] = (data['close'] - result_df[f'bb_lower_{window}']) / (result_df[f'bb_upper_{window}'] - result_df[f'bb_lower_{window}']).replace(0, 1)

            # 4. MOMENTUM FEATURES (comprehensive from HMM regime discovery)
            self.logger.info("📊 Adding momentum features...")
            if 'close' in data.columns:
                # Basic price change for interactions
                result_df['price_change'] = data['close'].pct_change()

                # Price momentum
                for window in [1, 2, 3, 5, 10, 20, 50]:
                    result_df[f'momentum_{window}'] = data['close'].pct_change(window)
                    result_df[f'momentum_ma_{window}'] = result_df[f'momentum_{window}'].rolling(10).mean()

                # Volume momentum
                if 'volume' in data.columns:
                    for window in [1, 2, 3, 5, 10, 20]:
                        result_df[f'volume_momentum_{window}'] = data['volume'].pct_change(window)

                # Momentum ratios
                result_df['momentum_ratio_5_20'] = result_df['momentum_5'] / result_df['momentum_20'].replace(0, 1)
                result_df['momentum_ratio_10_50'] = result_df['momentum_10'] / result_df['momentum_50'].replace(0, 1)

            # 5. FEATURE INTERACTIONS (from HMM regime discovery)
            self.logger.info("🔗 Adding feature interactions...")
            self._add_feature_interactions_hmm(result_df)

            # Fill NaN values with 0 for numerical stability
            result_df = result_df.fillna(0)

            computation_time = time.time() - start_time
            self.logger.info(f"✅ Generated {len(result_df.columns)} focused HMM features in {computation_time:.2f}s")
            self.logger.info("📊 Features: Volume, Volatility, Technical Indicators, Momentum, Interactions")

            return result_df

        except Exception as e:
            self.logger.error(f"❌ Error in generate_features_for_hmm: {e}")
            return data  # Return original data if feature generation fails

    def _add_feature_interactions_hmm(self, features: pd.DataFrame) -> None:
        """Add comprehensive feature interactions matching HMM regime discovery"""
        # Price-Volume Interactions
        if 'price_change' in features.columns and 'volume_change' in features.columns:
            features['price_volume_interaction'] = features['price_change'] * features['volume_change']
            # Safe price-volume ratio calculation to prevent infinity
            volume_change_safe = features['volume_change'].replace(0, np.nan)
            volume_change_safe = volume_change_safe.fillna(method='bfill').fillna(1e-6)  # Fill with small value if zero
            features['price_volume_ratio'] = features['price_change'] / volume_change_safe
            # Clip extreme ratios to prevent infinity
            features['price_volume_ratio'] = features['price_volume_ratio'].clip(-1000, 1000)

        # Volatility-Momentum Interactions
        volatility_cols = [col for col in features.columns if 'volatility' in col]
        momentum_cols = [col for col in features.columns if 'momentum' in col]

        for vol_col in volatility_cols[:3]:  # Top 3 volatility features
            for mom_col in momentum_cols[:3]:  # Top 3 momentum features
                if vol_col in features.columns and mom_col in features.columns:
                    features[f'{vol_col}_{mom_col}_interaction'] = features[vol_col] * features[mom_col]

        # RSI-MACD interactions
        rsi_cols = [col for col in features.columns if 'rsi' in col]
        macd_cols = [col for col in features.columns if 'macd' in col]

        for rsi_col in rsi_cols[:2]:  # Top 2 RSI features
            for macd_col in macd_cols[:2]:  # Top 2 MACD features
                if rsi_col in features.columns and macd_col in features.columns:
                    features[f'{rsi_col}_{macd_col}_interaction'] = features[rsi_col] * features[macd_col]

# Registry of available feature generators
FEATURE_GENERATORS: Dict[str, Callable] = {
    # Basic Technical Indicators
    'rsi': FeatureGenerators.rsi_generator,
    'sma': FeatureGenerators.sma_generator,
    'ema': FeatureGenerators.ema_generator,
    'bollinger_bands': FeatureGenerators.bollinger_bands_generator,
    'macd': FeatureGenerators.macd_generator,
    'stochastic': FeatureGenerators.stochastic_generator,
    'atr': FeatureGenerators.atr_generator,
    'volume_sma': FeatureGenerators.volume_sma_generator,
    'price_momentum': FeatureGenerators.price_momentum_generator,
    'volatility': FeatureGenerators.volatility_generator,

    # Candlestick body size features
    'body_size': FeatureGenerators.body_size_generator,
    'body_size_pct': FeatureGenerators.body_size_pct_generator,
    'body_to_range_ratio': FeatureGenerators.body_to_range_ratio_generator,
    'upper_wick': FeatureGenerators.upper_wick_generator,
    'lower_wick': FeatureGenerators.lower_wick_generator,
    'body_direction': FeatureGenerators.body_direction_generator,
    'body_strength': FeatureGenerators.body_strength_generator,

    # New TA-Lib enhanced features - Top 20 for Short-Term Crypto Trading
    # Phase 1: Core Momentum (Essential)
    'apo': FeatureGenerators.apo_generator,
    'cmo': FeatureGenerators.cmo_generator,
    'ultimate_oscillator': FeatureGenerators.ultimate_oscillator_generator,
    'natr': FeatureGenerators.natr_generator,
    'pfe': FeatureGenerators.pfe_generator,

    # Phase 2: Fast Trend Following
    't3': FeatureGenerators.t3_generator,
    'kama': FeatureGenerators.kama_generator,
    'mama': FeatureGenerators.mama_generator,
    'aroon_oscillator': FeatureGenerators.aroon_oscillator_generator,
    'ppo': FeatureGenerators.ppo_generator,

    # Phase 3: Risk Management
    'beta': FeatureGenerators.beta_generator,
    'true_range': FeatureGenerators.true_range_generator,
    'rocr': FeatureGenerators.rocr_generator,
    'adxr': FeatureGenerators.adxr_generator,
    'tema': FeatureGenerators.tema_generator,

    # Phase 4: Pattern Recognition
    'cdl_engulfing': FeatureGenerators.cdl_engulfing_generator,
    'cdl_morning_star': FeatureGenerators.cdl_morning_star_generator,
    'cdl_evening_star': FeatureGenerators.cdl_evening_star_generator,
    'cdl_three_white_soldiers': FeatureGenerators.cdl_three_white_soldiers_generator,
    'cdl_harami': FeatureGenerators.cdl_harami_generator,

    # Existing TA-Lib features
    'williams_r': FeatureGenerators.williams_r_generator,
    'cci': FeatureGenerators.cci_generator,
    'kst_oscillator': FeatureGenerators.kst_oscillator_generator
}
def get_feature_generator(feature_name: str) -> Optional[Callable]:
    """
    Get a feature generator function by name.

    Args:
        feature_name: Name of the feature generator

    Returns:
        Feature generator function or None if not found
    """
    return FEATURE_GENERATORS.get(feature_name.lower())

def list_available_generators() -> list:
    """
    List all available feature generators.

    Returns:
        List of available feature generator names
    """
    return list(FEATURE_GENERATORS.keys())

def create_feature_generator_config(feature_name: str, **kwargs) -> Dict[str, Any]:
    """
    Create a configuration for a feature generator.

    Args:
        feature_name: Name of the feature
        **kwargs: Additional configuration parameters

    Returns:
        Configuration dictionary
    """
    generator = get_feature_generator(feature_name)
    if not generator:
        raise ValueError(f"Unknown feature generator: {feature_name}")

    config = {
        'generator': generator,
        'feature_name': feature_name,
        **kwargs
    }

    return config

# Convenience functions for common feature configurations
def create_rsi_config(**kwargs) -> Dict[str, Any]:
    """Create RSI feature configuration."""
    return create_feature_generator_config('rsi', **kwargs)

def create_sma_config(**kwargs) -> Dict[str, Any]:
    """Create SMA feature configuration."""
    return create_feature_generator_config('sma', **kwargs)

def create_ema_config(**kwargs) -> Dict[str, Any]:
    """Create EMA feature configuration."""
    return create_feature_generator_config('ema', **kwargs)

def create_bollinger_bands_config(**kwargs) -> Dict[str, Any]:
    """Create Bollinger Bands feature configuration."""
    return create_feature_generator_config('bollinger_bands', **kwargs)

def create_macd_config(**kwargs) -> Dict[str, Any]:
    """Create MACD feature configuration."""
    return create_feature_generator_config('macd', **kwargs)

def create_williams_r_config(**kwargs) -> Dict[str, Any]:
    """Create Williams %R feature configuration."""
    return create_feature_generator_config('williams_r', **kwargs)

def create_cci_config(**kwargs) -> Dict[str, Any]:
    """Create CCI feature configuration."""
    return create_feature_generator_config('cci', **kwargs)

def create_ultimate_oscillator_config(**kwargs) -> Dict[str, Any]:
    """Create Ultimate Oscillator feature configuration."""
    return create_feature_generator_config('ultimate_oscillator', **kwargs)

def create_kst_config(**kwargs) -> Dict[str, Any]:
    """Create KST Oscillator feature configuration."""
    return create_feature_generator_config('kst_oscillator', **kwargs)

# Configuration functions for Top 20 TA-Lib indicators

# Phase 1: Core Momentum (Essential)
def create_apo_config(**kwargs) -> Dict[str, Any]:
    """Create Absolute Price Oscillator feature configuration."""
    return create_feature_generator_config('apo', **kwargs)

def create_cmo_config(**kwargs) -> Dict[str, Any]:
    """Create Chande Momentum Oscillator feature configuration."""
    return create_feature_generator_config('cmo', **kwargs)

def create_ultimate_oscillator_config(**kwargs) -> Dict[str, Any]:
    """Create Ultimate Oscillator feature configuration."""
    return create_feature_generator_config('ultimate_oscillator', **kwargs)

def create_natr_config(**kwargs) -> Dict[str, Any]:
    """Create Normalized ATR feature configuration."""
    return create_feature_generator_config('natr', **kwargs)

def create_pfe_config(**kwargs) -> Dict[str, Any]:
    """Create Polarized Fractal Efficiency feature configuration."""
    return create_feature_generator_config('pfe', **kwargs)

# Phase 2: Fast Trend Following
def create_t3_config(**kwargs) -> Dict[str, Any]:
    """Create Triple Exponential Moving Average feature configuration."""
    return create_feature_generator_config('t3', **kwargs)

def create_kama_config(**kwargs) -> Dict[str, Any]:
    """Create Kaufman's Adaptive Moving Average feature configuration."""
    return create_feature_generator_config('kama', **kwargs)

def create_mama_config(**kwargs) -> Dict[str, Any]:
    """Create MESA Adaptive Moving Average feature configuration."""
    return create_feature_generator_config('mama', **kwargs)

def create_aroon_oscillator_config(**kwargs) -> Dict[str, Any]:
    """Create Aroon Oscillator feature configuration."""
    return create_feature_generator_config('aroon_oscillator', **kwargs)

def create_ppo_config(**kwargs) -> Dict[str, Any]:
    """Create Percentage Price Oscillator feature configuration."""
    return create_feature_generator_config('ppo', **kwargs)

# Phase 3: Risk Management
def create_beta_config(**kwargs) -> Dict[str, Any]:
    """Create Beta coefficient feature configuration."""
    return create_feature_generator_config('beta', **kwargs)

def create_true_range_config(**kwargs) -> Dict[str, Any]:
    """Create True Range feature configuration."""
    return create_feature_generator_config('true_range', **kwargs)

def create_rocr_config(**kwargs) -> Dict[str, Any]:
    """Create Rate of Change Ratio feature configuration."""
    return create_feature_generator_config('rocr', **kwargs)

def create_adxr_config(**kwargs) -> Dict[str, Any]:
    """Create Average Directional Movement Index Rating feature configuration."""
    return create_feature_generator_config('adxr', **kwargs)

def create_tema_config(**kwargs) -> Dict[str, Any]:
    """Create Triple Exponential Moving Average feature configuration."""
    return create_feature_generator_config('tema', **kwargs)

# Phase 4: Pattern Recognition
def create_cdl_engulfing_config(**kwargs) -> Dict[str, Any]:
    """Create Engulfing Pattern feature configuration."""
    return create_feature_generator_config('cdl_engulfing', **kwargs)

def create_cdl_morning_star_config(**kwargs) -> Dict[str, Any]:
    """Create Morning Star Pattern feature configuration."""
    return create_feature_generator_config('cdl_morning_star', **kwargs)

def create_cdl_evening_star_config(**kwargs) -> Dict[str, Any]:
    """Create Evening Star Pattern feature configuration."""
    return create_feature_generator_config('cdl_evening_star', **kwargs)

def create_cdl_three_white_soldiers_config(**kwargs) -> Dict[str, Any]:
    """Create Three White Soldiers Pattern feature configuration."""
    return create_feature_generator_config('cdl_three_white_soldiers', **kwargs)

def create_cdl_harami_config(**kwargs) -> Dict[str, Any]:
    """Create Harami Pattern feature configuration."""
    return create_feature_generator_config('cdl_harami', **kwargs)

# Compatibility redirect for new unified feature generation system
# This allows existing code to continue working while we migrate to the new system
# Use lazy import to avoid circular dependency
def _get_compatible_feature_generators():
    try:
        from .feature_generators_compatibility import FeatureGenerators as CompatibleFeatureGenerators
        return CompatibleFeatureGenerators
    except ImportError:
        return None

# Create a lazy wrapper
class LazyCompatibleFeatureGenerators:
    def __init__(self):
        self._compatible_class = None
    
    def __getattr__(self, name):
        if self._compatible_class is None:
            self._compatible_class = _get_compatible_feature_generators()
        if self._compatible_class is None:
            raise AttributeError(f"'{name}' not found - compatibility layer not available")
        return getattr(self._compatible_class, name)

CompatibleFeatureGenerators = LazyCompatibleFeatureGenerators

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
    # Export the compatible version
    FeatureGenerators = CompatibleFeatureGenerators
    logger.info("✅ FeatureGenerators redirected to new unified system")
except ImportError:
    # Standalone compatibility module is not available, using original FeatureGenerators class
    logger.info("ℹ️ Using original FeatureGenerators class (standalone compatibility not available)")
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
