"""
import warnings
Feature calculation utilities for matrix optimization.

This module contains all feature calculation methods extracted from the main optimizer
to reduce complexity and improve maintainability. Enhanced with advanced matrix operations
for
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List, Union
import logging
import warnings

# VectorBT imports for native optimization
try:
    import vectorbt as vbt
    from src.utils.vectorbt_compat import rolling_mean, rolling_std, rolling_var, rolling_min, rolling_max, rolling_sum, rolling_apply, rolling_corr, rolling_cov
    from src.utils.vectorbt_compat import scale, rank, zscore, winsorize, clip, quantile
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

# Import advanced matrix operations
try:
    from src.utils.matrix_operations import (
        get_enhanced_matrix_operations, get_vectorized_processing_core,
        get_batch_matrix_processor, compute_trading_indicators,
        optimize_matrix_operation_with_hardware, safe_matrix_multiply,
        safe_correlation_matrix, safe_matrix_inverse, gpu_matrix_multiply,
        correlation_matrix_gpu, eigendecomposition_gpu, batch_matrix_multiply,
        batch_feature_transformation, batch_correlation_analysis,
        create_ml_pipeline, execute_ml_pipeline, optimize_pipeline_config
    )
    MATRIX_OPS_AVAILABLE = True
except ImportError as e:
    MATRIX_OPS_AVAILABLE = False
    logging.warning(f"Advanced matrix operations not available: {e}")

# Import common operations for enhanced functionality
try:
    from src.utils.common_operations import (
        safe_divide, safe_log, safe_sqrt, safe_power, safe_mean, safe_std,
        validate_finite, get_memory_usage, timed_operation
    )
    COMMON_OPERATIONS_AVAILABLE = True
except ImportError as e:
    COMMON_OPERATIONS_AVAILABLE = False
    logging.warning(f"Common operations not available: {e}")

# Set up logging
logger = logging.getLogger(__name__)

class FeatureCalculator:
    """
    Enhanced utility class for calculating various technical indicators.

    Features:
    - Traditional technical indicators (RSI, SMA, EMA, Bollinger Bands, ATR)
    - GPU-accelerated calculations for large datasets
    - Batch processing for memory efficiency
    - Advanced matrix operations integration
    - Safe mathematical operations with error handling
    """

    def __init__(self, enable_gpu_acceleration: bool = True, enable_batch_processing: bool = True):
        """Initialize the enhanced feature calculator."""
        self.enable_gpu_acceleration = enable_gpu_acceleration and MATRIX_OPS_AVAILABLE
        self.enable_batch_processing = enable_batch_processing and MATRIX_OPS_AVAILABLE

        # Initialize matrix operations components
        self.enhanced_matrix_ops = None
        self.vectorized_core = None
        self.batch_processor = None

        if MATRIX_OPS_AVAILABLE:
            try:
                self.enhanced_matrix_ops = get_enhanced_matrix_operations()
                self.vectorized_core = get_vectorized_processing_core()
                self.batch_processor = get_batch_matrix_processor()
                logger.info("✅ Advanced matrix operations initialized for feature calculations")
            except Exception as e:
                logger.warning(f"Failed to initialize matrix operations: {e}")

        logger.info(f"🔧 FeatureCalculator initialized - GPU: {self.enable_gpu_acceleration}, Batch: {self.enable_batch_processing}")

    @staticmethod
    def calculate_rsi(prices: pd.Series, period: int) -> pd.Series:
        """Calculate RSI with specific period."""
        delta = prices.diff()
        gain = delta.where(delta > 0, 0).rolling(window = period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window = period).mean()
        rs = gain / loss
        return 100 - 100 / (1 + rs)

    @staticmethod
    def calculate_sma(prices: pd.Series, period: int) -> pd.Series:
        """Calculate SMA with specific period."""
        return prices.rolling(window = period).mean()

    @staticmethod
    def calculate_ema(prices: pd.Series, period: int) -> pd.Series:
        """Calculate EMA with specific period."""
        return prices.ewm(span = period).mean()

    @staticmethod
    def calculate_bollinger_position(data: pd.DataFrame, period: int) -> pd.Series:
        """Calculate Bollinger Bands position with specific period."""
        sma = data['close'].rolling(window = period).mean()
        std = data['close'].rolling(window = period).std()
        upper = sma + 2 * std
        lower = sma - 2 * std
        return (data['close'] - lower) / (upper - lower)

    @staticmethod
    def calculate_atr(data: pd.DataFrame, period: int) -> pd.Series:
        """Calculate ATR with specific period."""
        high_low = data['high'] - data['low']
        high_close = np.abs(data['high'] - data['close'].shift())
        low_close = np.abs(data['low'] - data['close'].shift())
        true_range = pd.concat([high_low, high_close, low_close], axis = 1).max(axis = 1)
        return true_range.rolling(window = period).mean()

    def calculate_enhanced_rsi(self, prices: pd.Series, period: int) -> pd.Series:
        """Calculate RSI using enhanced matrix operations for better performance."""
        try:
            if self.enable_gpu_acceleration and self.enhanced_matrix_ops:
                # Use GPU-accelerated RSI calculation
                prices_array = prices.values.reshape(-1, 1)
                delta = np.diff(prices_array, axis=0)
                gain = np.maximum(delta, 0)
                loss = np.maximum(-delta, 0)

                # Use rolling operations with
                gain_ma = self.enhanced_matrix_ops.rolling_mean(gain, period)
                loss_ma = self.enhanced_matrix_ops.rolling_mean(loss, period)

                rs = safe_divide(gain_ma, loss_ma, default=1.0)
                rsi = 100 - 100 / (1 + rs)

                # Convert back to pandas Series
                result = pd.Series(index=prices.index, dtype=float)
                result.iloc[period:] = rsi.flatten()
                return result
            else:
                # Fallback to traditional calculation
                return self.calculate_rsi(prices, period)
        except Exception as e:
            logger.warning(f"Enhanced RSI calculation failed, using fallback: {e}")
            return self.calculate_rsi(prices, period)

    def calculate_enhanced_bollinger_bands(self, data: pd.DataFrame, period: int) -> Dict[str, pd.Series]:
        """Calculate Bollinger Bands using enhanced matrix operations."""
        try:
            if self.enable_gpu_acceleration and self.enhanced_matrix_ops:
                close_prices = data['close'].values.reshape(-1, 1)

                # Calculate SMA and STD using
                sma = self.enhanced_matrix_ops.rolling_mean(close_prices, period)
                std = self.enhanced_matrix_ops.rolling_std(close_prices, period)

                # Calculate bands
                upper_band = sma + 2 * std
                lower_band = sma - 2 * std

                # Convert back to pandas Series
                result = {
                    'upper': pd.Series(index=data.index, dtype=float),
                    'middle': pd.Series(index=data.index, dtype=float),
                    'lower': pd.Series(index=data.index, dtype=float)
                }

                result['upper'].iloc[period-1:] = upper_band.flatten()
                result['middle'].iloc[period-1:] = sma.flatten()
                result['lower'].iloc[period-1:] = lower_band.flatten()

                return result
            else:
                # Fallback to traditional calculation
                sma = rolling_mean(data["close"], window=period) if VECTORBT_AVAILABLE and len(data) > 1000 else data["close"].rolling(window=period).mean()
                std = rolling_std(data["close"], window=period) if VECTORBT_AVAILABLE and len(data) > 1000 else data["close"].rolling(window=period).std()
                return {
                    'upper': sma + 2 * std,
                    'middle': sma,
                    'lower': sma - 2 * std
                }
        except Exception as e:
            logger.warning(f"Enhanced Bollinger Bands calculation failed, using fallback: {e}")
            sma = rolling_mean(data["close"], window=period) if VECTORBT_AVAILABLE and len(data) > 1000 else data["close"].rolling(window=period).mean()
            std = rolling_std(data["close"], window=period) if VECTORBT_AVAILABLE and len(data) > 1000 else data["close"].rolling(window=period).std()
            return {
                'upper': sma + 2 * std,
                'middle': sma,
                'lower': sma - 2 * std
            }

    def calculate_batch_indicators(self, data: pd.DataFrame, indicators: List[str], periods: List[int]) -> Dict[str, pd.Series]:
        """Calculate multiple indicators in batch for better performance."""
        try:
            if self.enable_batch_processing and self.batch_processor:
                results = {}

                # Process in batches for memory efficiency
                batch_size = min(1000, len(data) // 4)
                batches = [data.iloc[i:i+batch_size] for i in range(0, len(data), batch_size)]

                for indicator in indicators:
                    indicator_results = []

                    for batch in batches:
                        if indicator == 'rsi':
                            for period in periods:
                                batch_result = self.calculate_enhanced_rsi(batch['close'], period)
                                indicator_results.append(batch_result)
                        elif indicator == 'sma':
                            for period in periods:
                                batch_result = self.calculate_sma(batch['close'], period)
                                indicator_results.append(batch_result)
                        elif indicator == 'ema':
                            for period in periods:
                                batch_result = self.calculate_ema(batch['close'], period)
                                indicator_results.append(batch_result)

                    # Combine batch results
                    if indicator_results:
                        combined_result = pd.concat(indicator_results, ignore_index=True)
                        results[f"{indicator}_combined"] = combined_result

                return results
            else:
                # Fallback to individual calculations
                results = {}
                for indicator in indicators:
                    for period in periods:
                        if indicator == 'rsi':
                            results[f"rsi_{period}"] = self.calculate_rsi(data['close'], period)
                        elif indicator == 'sma':
                            results[f"sma_{period}"] = self.calculate_sma(data['close'], period)
                        elif indicator == 'ema':
                            results[f"ema_{period}"] = self.calculate_ema(data['close'], period)
                return results
        except Exception as e:
            logger.warning(f"Batch indicators calculation failed: {e}")
            return {}

    def calculate_correlation_features(self, data: pd.DataFrame, feature_columns: List[str]) -> Dict[str, pd.Series]:
        """Calculate correlation-based features using advanced matrix operations."""
        try:
            if self.enable_gpu_acceleration and self.enhanced_matrix_ops:
                # Extract feature data
                feature_data = data[feature_columns].values

                # Calculate correlation matrix using
                corr_matrix = correlation_matrix_gpu(pd.DataFrame(feature_data, columns=feature_columns))

                # Extract correlation features
                n = corr_matrix.shape[0]
                upper_triangle = corr_matrix[np.triu_indices(n, k=1)]

                results = {}
                for i, corr_value in enumerate(upper_triangle):
                    # Create a feature with the correlation value repeated for all rows
                    results[f"correlation_{i}"] = pd.Series(
                        np.full(len(data), corr_value),
                        index=data.index
                    )

                return results
            else:
                # Fallback to traditional correlation calculation
                corr_matrix = data[feature_columns].corr()
                n = corr_matrix.shape[0]
                upper_triangle = corr_matrix.values[np.triu_indices(n, k=1)]

                results = {}
                for i, corr_value in enumerate(upper_triangle):
                    results[f"correlation_{i}"] = pd.Series(
                        np.full(len(data), corr_value),
                        index=data.index
                    )

                return results
        except Exception as e:
            logger.warning(f"Correlation features calculation failed: {e}")
            return {}

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for the feature calculator."""
        return {
            'matrix_operations_available': MATRIX_OPS_AVAILABLE,
            'common_operations_available': COMMON_OPERATIONS_AVAILABLE,
            'gpu_acceleration_enabled': self.enable_gpu_acceleration,
            'batch_processing_enabled': self.enable_batch_processing,
            'enhanced_matrix_ops_initialized': self.enhanced_matrix_ops is not None,
            'vectorized_core_initialized': self.vectorized_core is not None,
            'batch_processor_initialized': self.batch_processor is not None,
            'memory_usage': get_memory_usage() if COMMON_OPERATIONS_AVAILABLE else 0.0
        }

    @staticmethod
    def calculate_stochastic_k(data: pd.DataFrame, period: int) -> pd.Series:
        """Calculate Stochastic %K with specific period."""
        lowest_low = data['low'].rolling(window = period).min()
        highest_high = data['high'].rolling(window = period).max()
        return 100 * ((data['close'] - lowest_low) / (highest_high - lowest_low))

    @staticmethod
    def calculate_stochastic_d(data: pd.DataFrame, period: int) -> pd.Series:
        """Calculate Stochastic %D with specific period."""
        k = FeatureCalculator.calculate_stochastic_k(data, period)
        return k.rolling(window = 3).mean()

    @staticmethod
    def calculate_adx(data: pd.DataFrame, period: int) -> pd.Series:
        """Calculate ADX with specific period."""
        high_low = data['high'] - data['low']
        high_close = np.abs(data['high'] - data['close'].shift())
        low_close = np.abs(data['low'] - data['close'].shift())
        tr = pd.concat([high_low, high_close, low_close], axis = 1).max(axis = 1)
        atr = tr.rolling(window = period).mean()
        dm_plus = (data['high'] - data['high'].shift()).where(data['high'] - data['high'].shift() > data['low'].shift() - data['low'], 0)
        dm_minus = (data['low'].shift() - data['low']).where(data['low'].shift() - data['low'] > data['high'] - data['high'].shift(), 0)
        di_plus = 100 * (dm_plus.rolling(window = period).mean() / atr)
        di_minus = 100 * (dm_minus.rolling(window = period).mean() / atr)
        dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
        return dx.rolling(window = period).mean()

    @staticmethod
    def calculate_cci(data: pd.DataFrame, period: int) -> pd.Series:
        """Calculate CCI with specific period."""
        typical_price = (data['high'] + data['low'] + data['close']) / 3
        sma = typical_price.rolling(window = period).mean()
        mad = typical_price.rolling(window = period).apply(lambda x: np.mean(np.abs(x - x.mean())))
        return (typical_price - sma) / (0.015 * mad)

    @staticmethod
    def calculate_williams_r(data: pd.DataFrame, period: int) -> pd.Series:
        """Calculate Williams %R with specific period."""
        highest_high = data['high'].rolling(window = period).max()
        lowest_low = data['low'].rolling(window = period).min()
        return -100 * ((highest_high - data['close']) / (highest_high - lowest_low))

    @staticmethod
    def calculate_mfi(data: pd.DataFrame, period: int) -> pd.Series:
        """Calculate Money Flow Index with specific period."""
        typical_price = (data['high'] + data['low'] + data['close']) / 3
        money_flow = typical_price * data['volume']
        positive_flow = money_flow.where(typical_price > typical_price.shift(1), 0).rolling(window = period).sum()
        negative_flow = money_flow.where(typical_price < typical_price.shift(1), 0).rolling(window = period).sum()
        return 100 - 100 / (1 + positive_flow / negative_flow)

    @staticmethod
    def calculate_roc(prices: pd.Series, period: int) -> pd.Series:
        """Calculate Rate of Change with specific period."""
        return (prices - prices.shift(period)) / prices.shift(period) * 100

    @staticmethod
    def calculate_mom(prices: pd.Series, period: int) -> pd.Series:
        """Calculate Momentum with specific period."""
        return prices - prices.shift(period)

    @staticmethod
    def calculate_tsi(prices: pd.Series, period: int) -> pd.Series:
        """Calculate True Strength Index with specific period."""
        price_change = prices.diff()
        abs_price_change = abs(price_change)
        smoothed_change = price_change.ewm(span = period).mean()
        smoothed_abs_change = abs_price_change.ewm(span = period).mean()
        return 100 * (smoothed_change / smoothed_abs_change)

    @staticmethod
    def calculate_uo(data: pd.DataFrame, period: int) -> pd.Series:
        """Calculate Ultimate Oscillator with specific period."""
        tr = pd.concat([data['high'] - data['low'], abs(data['high'] - data['close'].shift(1)), abs(data['low'] - data['close'].shift(1))], axis = 1).max(axis = 1)
        bp = data['close'] - pd.concat([data['low'], data['close'].shift(1)], axis = 1).min(axis = 1)
        avg7 = bp.rolling(window = 7).sum() / tr.rolling(window = 7).sum()
        avg14 = bp.rolling(window = 14).sum() / tr.rolling(window = 14).sum()
        avg28 = bp.rolling(window = 28).sum() / tr.rolling(window = 28).sum()
        return 100 * (4 * avg7 + 2 * avg14 + avg28) / (4 + 2 + 1)

    @staticmethod
    def calculate_ao(data: pd.DataFrame, period: int) -> pd.Series:
        """Calculate Awesome Oscillator with specific period."""
        median_price = (data['high'] + data['low']) / 2
        return median_price.rolling(window = 5).mean() - median_price.rolling(window = 34).mean()

    @staticmethod
    def calculate_cmf(data: pd.DataFrame, period: int) -> pd.Series:
        """Calculate Chaikin Money Flow with specific period."""
        mfm = (data['close'] - data['low'] - (data['high'] - data['close'])) / (data['high'] - data['low'])
        mfm = mfm.replace([np.inf, -np.inf], 0)
        mfv = mfm * data['volume']
        return mfv.rolling(window = period).sum() / data['volume'].rolling(window = period).sum()

    @staticmethod
    def calculate_vwap(data: pd.DataFrame, period: int) -> pd.Series:
        """Calculate Volume Weighted Average Price with specific period."""
        typical_price = (data['high'] + data['low'] + data['close']) / 3
        return (typical_price * data['volume']).rolling(window = period).sum() / data['volume'].rolling(window = period).sum()

    @staticmethod
    def calculate_obv(data: pd.DataFrame) -> pd.Series:
        """Calculate On Balance Volume."""
        obv = pd.Series(index = data.index, dtype = float)
        obv.iloc[0] = data['volume'].iloc[0]
        for i in range(1, len(data)):
            if data['close'].iloc[i] > data['close'].iloc[i - 1]:
                obv.iloc[i] = obv.iloc[i - 1] + data['volume'].iloc[i]
            elif data['close'].iloc[i] < data['close'].iloc[i - 1]:
                obv.iloc[i] = obv.iloc[i - 1] - data['volume'].iloc[i]
            else:
                obv.iloc[i] = obv.iloc[i - 1]
        return obv

    @staticmethod
    def calculate_ad(data: pd.DataFrame) -> pd.Series:
        """Calculate Accumulation/Distribution Line."""
        clv = (data['close'] - data['low'] - (data['high'] - data['close'])) / (data['high'] - data['low'])
        clv = clv.replace([np.inf, -np.inf], 0)
        return (clv * data['volume']).cumsum()

    @staticmethod
    def calculate_volume_price_trend(data: pd.DataFrame) -> pd.Series:
        """Calculate Volume Price Trend."""
        price_change = data['close'].pct_change()
        return (price_change * data['volume']).cumsum()

    @staticmethod
    def calculate_volume_price_oscillator(data: pd.DataFrame, period: int) -> pd.Series:
        """Calculate Volume Price Oscillator with specific period."""
        typical_price = (data['high'] + data['low'] + data['close']) / 3
        vwap = (typical_price * data['volume']).rolling(window = period).sum() / data['volume'].rolling(window = period).sum()
        return (typical_price - vwap) / vwap * 100

    @staticmethod
    def calculate_vwap_momentum(data: pd.DataFrame, period: int) -> pd.Series:
        """Calculate VWAP momentum with specific period."""
        vwap = FeatureCalculator.calculate_vwap(data, period)
        return vwap / vwap.shift(period) - 1

    @staticmethod
    def calculate_vwap_returns(data: pd.DataFrame, period: int) -> pd.Series:
        """Calculate VWAP returns with specific period."""
        vwap = FeatureCalculator.calculate_vwap(data, period)
        return vwap.pct_change()

    @staticmethod
    def calculate_price_vwap_ratio(data: pd.DataFrame, period: int) -> pd.Series:
        """Calculate price to VWAP ratio with specific period."""
        vwap = FeatureCalculator.calculate_vwap(data, period)
        return data['close'] / vwap

    @staticmethod
    def calculate_price_vwap_deviation(data: pd.DataFrame, period: int) -> pd.Series:
        """Calculate price to VWAP deviation with specific period."""
        vwap = FeatureCalculator.calculate_vwap(data, period)
        return (data['close'] - vwap) / vwap

    @staticmethod
    def calculate_price_vwap_spread(data: pd.DataFrame, period: int) -> pd.Series:
        """Calculate price to VWAP spread with specific period."""
        vwap = FeatureCalculator.calculate_vwap(data, period)
        return data['close'] - vwap

class FeatureCalculatorRegistry:
    """Registry for feature calculation methods."""

    _calculators = {
        # Basic features
        'ret_1': lambda data, period: data['close'].pct_change(1),
        'ret_5': lambda data, period: data['close'].pct_change(5),
        'ret_20': lambda data, period: data['close'].pct_change(20),
        'vol_20': lambda data, period: data['close'].pct_change().rolling(20).std(),
        'volume_ratio': lambda data, period: data['volume'] / data['volume'].rolling(20).mean(),

        # RSI variations
        'rsi_7': lambda data, period: FeatureCalculator.calculate_rsi(data, 7),
        'rsi_14': lambda data, period: FeatureCalculator.calculate_rsi(data, 14),
        'rsi_21': lambda data, period: FeatureCalculator.calculate_rsi(data, 21),

        # Moving averages
        'sma_5': lambda data, period: FeatureCalculator.calculate_sma(data, 5),
        'sma_10': lambda data, period: FeatureCalculator.calculate_sma(data, 10),
        'sma_20': lambda data, period: FeatureCalculator.calculate_sma(data, 20),
        'sma_50': lambda data, period: FeatureCalculator.calculate_sma(data, 50),
        'sma_100': lambda data, period: FeatureCalculator.calculate_sma(data, 100),
        'ema_5': lambda data, period: FeatureCalculator.calculate_ema(data, 5),
        'ema_10': lambda data, period: FeatureCalculator.calculate_ema(data, 10),
        'ema_20': lambda data, period: FeatureCalculator.calculate_ema(data, 20),
        'ema_50': lambda data, period: FeatureCalculator.calculate_ema(data, 50),
        'ema_100': lambda data, period: FeatureCalculator.calculate_ema(data, 100),

        # MACD
        'macd_line': lambda data, period: FeatureCalculator.calculate_ema(data, 12) - FeatureCalculator.calculate_ema(data, 26),
        'macd_signal': lambda data, period: (FeatureCalculator.calculate_ema(data, 12) - FeatureCalculator.calculate_ema(data, 26)).ewm(span=9).mean(),

        # Bollinger Bands
        'bb_middle_10': lambda data, period: FeatureCalculator.calculate_sma(data, 10),
        'bb_middle_20': lambda data, period: FeatureCalculator.calculate_sma(data, 20),
        'bb_middle_30': lambda data, period: FeatureCalculator.calculate_sma(data, 30),
        'bb_upper_10': lambda data, period: FeatureCalculator.calculate_sma(data, 10) + 2 * data['close'].rolling(10).std(),
        'bb_upper_20': lambda data, period: FeatureCalculator.calculate_sma(data, 20) + 2 * data['close'].rolling(20).std(),
        'bb_upper_30': lambda data, period: FeatureCalculator.calculate_sma(data, 30) + 2 * data['close'].rolling(30).std(),
        'bb_lower_10': lambda data, period: FeatureCalculator.calculate_sma(data, 10) - 2 * data['close'].rolling(10).std(),
        'bb_lower_20': lambda data, period: FeatureCalculator.calculate_sma(data, 20) - 2 * data['close'].rolling(20).std(),
        'bb_lower_30': lambda data, period: FeatureCalculator.calculate_sma(data, 30) - 2 * data['close'].rolling(30).std(),
        'bb_position_10': lambda data, period: FeatureCalculator.calculate_bollinger_position(data, 10),
        'bb_position_20': lambda data, period: FeatureCalculator.calculate_bollinger_position(data, 20),
        'bb_position_30': lambda data, period: FeatureCalculator.calculate_bollinger_position(data, 30),

        # ATR
        'atr_7': lambda data, period: FeatureCalculator.calculate_atr(data, 7),
        'atr_14': lambda data, period: FeatureCalculator.calculate_atr(data, 14),
        'atr_21': lambda data, period: FeatureCalculator.calculate_atr(data, 21),

        # Stochastic
        'stoch_k_14': lambda data, period: FeatureCalculator.calculate_stochastic_k(data, 14),
        'stoch_k_21': lambda data, period: FeatureCalculator.calculate_stochastic_k(data, 21),
        'stoch_d_14_3': lambda data, period: FeatureCalculator.calculate_stochastic_k(data, 14).rolling(3).mean(),
        'stoch_d_21_5': lambda data, period: FeatureCalculator.calculate_stochastic_k(data, 21).rolling(5).mean(),

        # Williams %R
        'williams_r_14': lambda data, period: FeatureCalculator.calculate_williams_r(data, 14),
        'williams_r_21': lambda data, period: FeatureCalculator.calculate_williams_r(data, 21),

        # Momentum and ROC
        'momentum_15': lambda data, period: data['close'] - data['close'].shift(15),
        'momentum_25': lambda data, period: data['close'] - data['close'].shift(25),
        'momentum_30': lambda data, period: data['close'] - data['close'].shift(30),
        'roc_15': lambda data, period: FeatureCalculator.calculate_roc(data, 15),
        'roc_25': lambda data, period: FeatureCalculator.calculate_roc(data, 25),
        'roc_30': lambda data, period: FeatureCalculator.calculate_roc(data, 30),
        'momentum_ratio_5': lambda data, period: data['close'] / data['close'].shift(5) - 1,
        'momentum_ratio_10': lambda data, period: data['close'] / data['close'].shift(10) - 1,
        'momentum_ratio_20': lambda data, period: data['close'] / data['close'].shift(20) - 1,

        # VWAP
        'vwap': lambda data, period: FeatureCalculator.calculate_vwap(data, period),
        'vwap_deviation': lambda data, period: FeatureCalculator.calculate_price_vwap_deviation(data, period),

        # CCI
        'cci_14': lambda data, period: FeatureCalculator.calculate_cci(data, 14),
        'cci_20': lambda data, period: FeatureCalculator.calculate_cci(data, 20),

        # Volume features
        'volume_sma_5': lambda data, period: data['volume'].rolling(5).mean(),
        'volume_sma_10': lambda data, period: data['volume'].rolling(10).mean(),
        'volume_sma_15': lambda data, period: data['volume'].rolling(15).mean(),
        'volume_sma_30': lambda data, period: data['volume'].rolling(30).mean(),
        'volume_ratio_5': lambda data, period: data['volume'] / data['volume'].rolling(5).mean(),
        'volume_ratio_10': lambda data, period: data['volume'] / data['volume'].rolling(10).mean(),
        'volume_ratio_15': lambda data, period: data['volume'] / data['volume'].rolling(15).mean(),
        'volume_ratio_30': lambda data, period: data['volume'] / data['volume'].rolling(30).mean(),
        'obv': lambda data, period: FeatureCalculator.calculate_obv(data, period),

        # Volatility
        'volatility_5': lambda data, period: data['close'].pct_change().rolling(5).std(),
        'volatility_10': lambda data, period: data['close'].pct_change().rolling(10).std(),
        'volatility_20': lambda data, period: data['close'].pct_change().rolling(20).std(),
        'volatility_30': lambda data, period: data['close'].pct_change().rolling(30).std(),
        'high_low_ratio_5': lambda data, period: (data['high'] / data['low']).rolling(5).mean(),
        'high_low_ratio_10': lambda data, period: (data['high'] / data['low']).rolling(10).mean(),
        'high_low_ratio_20': lambda data, period: (data['high'] / data['low']).rolling(20).mean(),
        'high_low_ratio_30': lambda data, period: (data['high'] / data['low']).rolling(30).mean(),

        # Advanced momentum features
        'momentum_40': lambda data, period: data['close'].pct_change().rolling(40).mean(),
        'momentum_60': lambda data, period: data['close'].pct_change().rolling(60).mean(),
        'momentum_100': lambda data, period: data['close'].pct_change().rolling(100).mean(),
        'momentum_acceleration': lambda data, period: (data['close'].pct_change().rolling(40).mean() - data['close'].pct_change().rolling(60).mean()),
        'momentum_strength': lambda data, period: data['close'].pct_change().rolling(40).mean() / (data['close'].pct_change().rolling(60).std() + 1e-8),
        'momentum_divergence': lambda data, period: (data['close'].pct_change(10) - data['volume'].pct_change(10)),
        'momentum_trend_strength': lambda data, period: (data['close'].pct_change().rolling(20).mean().abs() / (data['close'].pct_change().rolling(20).std() + 1e-8)),
        'momentum_volatility_adjusted': lambda data, period: (data['close'].pct_change().rolling(40).mean() / (data['close'].pct_change().rolling(40).std() + 1e-8)),

        # Correlation features
        'autocorrelation_5': lambda data, period: data['close'].pct_change().rolling(5).corr(data['close'].pct_change().shift(1)),
        'autocorrelation_20': lambda data, period: data['close'].pct_change().rolling(20).corr(data['close'].pct_change().shift(1)),
        'cross_timeframe_correlation': lambda data, period: data['close'].pct_change().rolling(20).corr(data['close'].pct_change().rolling(5).mean()),

        # Liquidity features
        'volume_liquidity': lambda data, period: data['volume'] / (data['volume'].rolling(20).mean() + 1e-8),
        'price_impact': lambda data, period: data['close'].pct_change().abs() / (data['volume'] + 1e-8),
        'price_impact_smooth': lambda data, period: (data['close'].pct_change().abs() / (data['volume'] + 1e-8)).rolling(20).mean(),
        'liquidity_percentile': lambda data, period: (data['volume'] / (data['volume'].rolling(100).mean() + 1e-8)).rolling(100).rank(pct=True),

        # Adaptive features
        'adaptive_period': lambda data, period: ((20 * (data['close'].pct_change().rolling(20).std() / (data['close'].pct_change().rolling(100).mean() + 1e-8))).clip(5, 50)),
        'adaptive_ma': lambda data, period: data['close'].rolling(20).mean(),  # Simplified adaptive MA

        # Legacy support
        'RSI': FeatureCalculator.calculate_rsi,
        'MACD_fast': FeatureCalculator.calculate_ema,
        'MACD_slow': FeatureCalculator.calculate_ema,
        'Bollinger_Bands': FeatureCalculator.calculate_bollinger_position,
        'SMA_short': FeatureCalculator.calculate_sma,
        'SMA_long': FeatureCalculator.calculate_sma,
        'EMA_short': FeatureCalculator.calculate_ema,
        'EMA_long': FeatureCalculator.calculate_ema,
        'ATR': FeatureCalculator.calculate_atr,
        'Stochastic_k': FeatureCalculator.calculate_stochastic_k,
        'Stochastic_d': FeatureCalculator.calculate_stochastic_d,
        'ADX': FeatureCalculator.calculate_adx,
        'CCI': FeatureCalculator.calculate_cci,
        'Williams_R': FeatureCalculator.calculate_williams_r,
        'MFI': FeatureCalculator.calculate_mfi,
        'ROC': FeatureCalculator.calculate_roc,
        'MOM': FeatureCalculator.calculate_mom,
        'TSI': FeatureCalculator.calculate_tsi,
        'UO': FeatureCalculator.calculate_uo,
        'AO': FeatureCalculator.calculate_ao,
        'CMF': FeatureCalculator.calculate_cmf,
        'VWAP': FeatureCalculator.calculate_vwap,
        'OBV': FeatureCalculator.calculate_obv,
        'AD': FeatureCalculator.calculate_ad,
        'Chaikin_Money_Flow': FeatureCalculator.calculate_cmf,
        'Money_Flow_Index': FeatureCalculator.calculate_mfi,
        'Volume_Price_Trend': FeatureCalculator.calculate_volume_price_trend,
        'Accumulation_Distribution': FeatureCalculator.calculate_ad,
        'On_Balance_Volume': FeatureCalculator.calculate_obv,
        'Volume_Weighted_Average_Price': FeatureCalculator.calculate_vwap,
        'Volume_Price_Oscillator': FeatureCalculator.calculate_volume_price_oscillator,
        'VWAP_Momentum': FeatureCalculator.calculate_vwap_momentum,
        'VWAP_Returns': FeatureCalculator.calculate_vwap_returns,
        'Price_VWAP_Ratio': FeatureCalculator.calculate_price_vwap_ratio,
        'Price_VWAP_Deviation': FeatureCalculator.calculate_price_vwap_deviation,
        'Price_VWAP_Spread': FeatureCalculator.calculate_price_vwap_spread,
    }

    @classmethod
    def calculate_feature(cls, data: pd.DataFrame, feature_name: str, period: int) -> Optional[pd.Series]:
        """Calculate feature using the appropriate calculator."""
        calculator = cls._calculators.get(feature_name)
        if calculator is None:
            return None

        try:
            if feature_name in ['RSI', 'ROC', 'MOM', 'TSI']:
                return calculator(data['close'], period)
            elif feature_name in ['MACD_fast', 'MACD_slow', 'SMA_short', 'SMA_long', 'EMA_short', 'EMA_long']:
                return calculator(data['close'], period)
            elif feature_name == 'OBV':
                return calculator(data)
            elif feature_name == 'AD':
                return calculator(data)
            elif feature_name == 'Volume_Price_Trend':
                return calculator(data)
            elif feature_name == 'Accumulation_Distribution':
                return calculator(data)
            elif feature_name == 'On_Balance_Volume':
                return calculator(data)
            else:
                return calculator(data, period)
        except Exception:
            return None

    def _should_use_vectorbt(self, data) -> bool:
        """Determine if VectorBT should be used based on data size and configuration."""
        return (hasattr(self, 'use_vectorbt') and getattr(self, 'use_vectorbt', True) and
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

    def _vectorbt_apply_operation(self, data: pd.Series, func,
                                 window: int, **kwargs) -> pd.Series:
        """Perform VectorBT rolling apply operation with fallback to pandas."""
        if not self._should_use_vectorbt(data):
            return data.rolling(window=window).apply(func, **kwargs)

        try:
            return rolling_apply(data, func, window=window, **kwargs)
        except Exception as e:
            logger.warning(f"VectorBT rolling apply failed: {e}, using pandas fallback")
            return data.rolling(window=window).apply(func, **kwargs)
