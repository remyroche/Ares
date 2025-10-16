from src.utils.tprint import tprint
import warnings

from functools import lru_cache
import numba
import numpy as np
import pandas as pd
import datetime
import typing

class LocationClassifierOptimized:
    """Optimized version of location classifier with caching and vectorization."""

    def __init__(self, cache_size: int = 128) -> None:
        self.cache_size = cache_size
        self._sr_cache = {}
        self._calculation_cache = {}

    @lru_cache(maxsize = 128)
    def _cached_atr(self, prices_hash: int, period: int = 14) -> float:
        """Cached ATR calculation."""
        return self._calculate_atr_vectorized(prices_hash, period)

    @staticmethod
    @numba.jit(nopython = True)
    def _calculate_distances_vectorized(current_price: float, support_prices: np.ndarray, resistance_prices: np.ndarray) -> tuple:
        """Vectorized distance calculation using Numba."""
        support_distances = (current_price - support_prices) / current_price
        resistance_distances = (resistance_prices - current_price) / current_price
        weights = np.exp(-np.arange(len(support_distances)))
        weights = weights / weights.sum()
        weighted_support_dist = np.sum(support_distances * weights[:len(support_distances)])
        weights_r = np.exp(-np.arange(len(resistance_distances)))
        weights_r = weights_r / weights_r.sum()
        weighted_resistance_dist = np.sum(resistance_distances * weights_r[:len(resistance_distances)])
        return (support_distances, resistance_distances, weighted_support_dist, weighted_resistance_dist)

    def process_batch_predictions(self, price_series: pd.Series, sr_levels_dict: Dict[int, Dict[str, List[float]]], window_size: int = 100) -> pd.DataFrame:
        """
        Process multiple price points efficiently in batch.
        Useful for backtesting and real-time streaming.
        """
        results = []
        atr_series = self._calculate_rolling_atr(price_series, 14)
        for i in range(window_size, len(price_series)):
            current_price = price_series.iloc[i]
            current_atr = atr_series.iloc[i]
            if i in sr_levels_dict:
                support_prices = np.array(sr_levels_dict[i]['support'])
                resistance_prices = np.array(sr_levels_dict[i]['resistance'])
            else:
                support_prices, resistance_prices = self._get_cached_levels(i)
            distances = self._calculate_distances_vectorized(current_price, support_prices[:10], resistance_prices[:10])
            results.append({'timestamp': price_series.index[i], 'price': current_price, 'support_distance': distances[0][0] if len(distances[0]) > 0 else 1.0, 'resistance_distance': distances[1][0] if len(distances[1]) > 0 else 1.0, 'weighted_support_distance': distances[2], 'weighted_resistance_distance': distances[3]})
        return pd.DataFrame(results)

    def _calculate_rolling_atr(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Efficient rolling ATR calculation."""
        high = prices.rolling(2).max()
        low = prices.rolling(2).min()
        hl = high - low
        hc = (high - prices.shift(1)).abs()
        lc = (low - prices.shift(1)).abs()
        tr = pd.concat([hl, hc, lc], axis = 1).max(axis = 1)
        atr = tr.ewm(span = period, adjust = False).mean()
        return atr

    def incremental_update(self, classifier_state: Dict[str, Any], new_candle: Dict[str, float]) -> Dict[str, Any]:
        """
        Incrementally update location metrics with new candle.
        Avoids full recalculation.
        """
        classifier_state['price_history'].append(new_candle['close'])
        if len(classifier_state['price_history']) > 1000:
            classifier_state['price_history'].pop(0)
        new_tr = self._calculate_true_range(new_candle, classifier_state['last_close'])
        alpha = 2 / (14 + 1)
        classifier_state['atr'] = alpha * new_tr + (1 - alpha) * classifier_state['atr']
        if self._should_update_sr_levels(classifier_state, new_candle):
            classifier_state['sr_levels'] = self._update_sr_levels_incremental(classifier_state['sr_levels'], new_candle)
        current_price = new_candle['close']
        support_prices = [s['price'] for s in classifier_state['sr_levels']['support']]
        resistance_prices = [r['price'] for r in classifier_state['sr_levels']['resistance']]
        distances = self._calculate_distances_vectorized(current_price, np.array(support_prices[:10]), np.array(resistance_prices[:10]))
        classifier_state['last_close'] = current_price
        classifier_state['distances'] = distances
        classifier_state['last_update'] = pd.Timestamp.now()
        return classifier_state

    def _should_update_sr_levels(self, state: Dict[str, Any], new_candle: Dict[str, float]) -> bool:
        """Determine if S/R levels need recalculation."""
        price_change = abs(new_candle['close'] - state['last_close']) / state['last_close']
        if price_change > 0.01:
            return True
        if new_candle['volume'] > state['avg_volume'] * 2:
            return True
        candles_since_update = state.get('candles_since_sr_update', 0)
        if candles_since_update > 20:
            return True
        return False

    def parallel_timeframe_analysis(self, market_data: Dict[str, pd.DataFrame], timeframes: List[str]) -> Dict[str, Any]:
        """
        Analyze multiple timeframes in parallel using multiprocessing.
        """
        from concurrent.futures import ProcessPoolExecutor

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

        with ProcessPoolExecutor(max_workers = len(timeframes)) as executor:
            futures = {executor.submit(self._analyze_single_timeframe, market_data[tf], tf): tf for tf in timeframes}
            results = {}
            for future in futures:
                tf = futures[future]
                try:
                    results[tf] = future.result(timeout = 5)
                except Exception as e:
                    tprint(f'Error analyzing {tf}: {e}')
                    results[tf] = None
        return self._aggregate_parallel_results(results)

    def _calculate_true_range(self, candle: Dict[str, float], prev_close: float) -> float:
        """Calculate true range for a single candle."""
        hl = candle['high'] - candle['low']
        hc = abs(candle['high'] - prev_close)
        lc = abs(candle['low'] - prev_close)
        return max(hl, hc, lc)

    def get_memory_efficient_features(self, basic_metrics: Dict[str, float]) -> np.ndarray:
        """
        Convert metrics to memory-efficient numpy array.
        Useful for high-frequency trading with memory constraints.
        """
        feature_names = ['support_distance', 'resistance_distance', 'support_strength', 'resistance_strength', 'combined_location_score', 'location_quality']
        features = np.zeros(len(feature_names), dtype = np.float32)
        for i, name in enumerate(feature_names):
            features[i] = basic_metrics.get(name, 0.0)
        return features

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
