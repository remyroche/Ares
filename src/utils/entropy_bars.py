"""
Entropy Bars Implementation for Financial Time Series - Optimized Version

This module provides entropy-based bar generation for financial data processing
with Numba JIT optimization for maximum performance.

Key Features:
- Numba-optimized Shannon entropy calculation (10-100x speedup)
- Entropy bar processing targeting specific time intervals
- Calibration helpers for optimal threshold settings
- Specialized features for Layer 3/4 modeling with JIT optimization
- Vectorized operations for memory efficiency
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional, Dict, Any
from pathlib import Path
import logging

try:
    from numba import njit, jit
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    # Dummy decorators if numba is not installed
    def njit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator

# Import optimized entropy functions
try:
    from src.utils.entropy_optimized import (
        vectorized_entropy_features,
        rolling_entropy_numba,
        shannon_entropy_numba,
        lempel_ziv_complexity_numba
    )
    OPTIMIZED_AVAILABLE = True
except ImportError:
    OPTIMIZED_AVAILABLE = False

from src.utils.tprint import tprint_info, tprint_warning, tprint_error

logger = logging.getLogger(__name__)


@njit(fastmath=True)
def fast_shannon_entropy(symbols: np.ndarray, alphabet_size: int) -> float:
    """
    Step 3: Shannon Entropy (H) Calculation
    O(N) optimized for Numba.
    
    Args:
        symbols: Array of symbol indices
        alphabet_size: Size of the symbol alphabet
        
    Returns:
        Shannon entropy value
    """
    n = len(symbols)
    if n == 0: 
        return 0.0
    
    counts = np.zeros(alphabet_size, dtype=np.int32)
    for i in range(n):
        counts[symbols[i]] += 1
        
    entropy_val = 0.0
    inv_n = 1.0 / n
    inv_log2 = 1.0 / np.log(2.0)
    
    for i in range(alphabet_size):
        if counts[i] > 0:
            p = counts[i] * inv_n
            entropy_val -= p * np.log(p) * inv_log2
            
    return entropy_val


@njit
def process_ohlcv_entropy_bars(
    close_prices: np.ndarray, 
    thresholds: np.ndarray, 
    window_size: int, 
    bar_threshold: float
) -> np.ndarray:
    """
    Processes 1-minute OHLCV 'Close' data into Entropy Bars.
    Targets ~15 min intervals based on the bar_threshold.
    
    Args:
        close_prices: Array of close prices
        thresholds: Array of return thresholds for symbolization
        window_size: Window size for entropy calculation
        bar_threshold: Threshold for triggering new bars
        
    Returns:
        Array of bar indices where entropy bars are triggered
    """
    n_minutes = len(close_prices)
    alphabet_size = len(thresholds) + 1
    
    # Calculate log returns
    log_returns = np.zeros(n_minutes)
    for i in range(1, n_minutes):
        log_returns[i] = np.log(close_prices[i] / close_prices[i-1])
    
    # Symbols array
    symbols = np.zeros(n_minutes, dtype=np.int32)
    for i in range(n_minutes):
        symbols[i] = np.searchsorted(thresholds, log_returns[i])
    
    fuel_tank = 0.0
    bar_indices = []
    
    # Core loop starting from window_size
    for i in range(window_size, n_minutes):
        window = symbols[i - window_size + 1 : i + 1]
        h_t = fast_shannon_entropy(window, alphabet_size)
        
        fuel_tank += h_t
        
        # Trigger approx every 15 minutes of information
        if fuel_tank >= bar_threshold:
            bar_indices.append(i)
            fuel_tank = 0.0
            
    return np.array(bar_indices)


def get_15min_threshold(
    close_prices: np.ndarray, 
    n_bins: int = 10, 
    window_size: int = 100
) -> Tuple[np.ndarray, float]:
    """
    Calculates the threshold required for ~15 min bars.
    
    Args:
        close_prices: Array of close prices
        n_bins: Number of bins for symbolization
        window_size: Window size for entropy calculation
        
    Returns:
        Tuple of (thresholds, target_threshold)
    """
    log_rets = np.diff(np.log(close_prices))
    thresholds = np.percentile(log_rets, np.linspace(0, 100, n_bins+1)[1:-1])
    
    # Estimate average entropy per 1-minute bar
    sample_symbols = np.searchsorted(thresholds, log_rets)
    avg_h = 0
    sample_end = min(len(sample_symbols), 1000)
    
    for i in range(window_size, sample_end):
        avg_h += fast_shannon_entropy(sample_symbols[i-window_size:i], n_bins)
    
    if sample_end > window_size:
        h_per_min = avg_h / (sample_end - window_size)
    else:
        h_per_min = 0.1  # Default fallback
    
    # To get 1 bar every 15 mins:
    target_threshold = h_per_min * 15
    return thresholds, target_threshold


def generate_entropy_bars_from_ohlcv(
    ohlcv_data: pd.DataFrame,
    n_bins: int = 10,
    window_size: int = 100,
    target_minutes: int = 15,
    symbol: str = "UNKNOWN",
    exchange: str = "binance"
) -> pd.DataFrame:
    """
    Generate entropy bars from OHLCV data with optimization.
    
    Args:
        ohlcv_data: DataFrame with OHLCV data (1-minute bars)
        n_bins: Number of bins for return symbolization
        window_size: Window size for entropy calculation
        target_minutes: Target minutes per entropy bar
        symbol: Trading symbol
        exchange: Exchange name
        
    Returns:
        DataFrame with entropy bars data
    """
    if ohlcv_data.empty:
        tprint_warning("⚠️ Empty OHLCV data provided")
        return pd.DataFrame()
    
    required_cols = ['close', 'open', 'high', 'low', 'volume']
    missing_cols = [col for col in required_cols if col not in ohlcv_data.columns]
    if missing_cols:
        tprint_error(f"❌ Missing required columns: {missing_cols}")
        return pd.DataFrame()
    
    close_prices = ohlcv_data['close'].values
    
    # Calibrate thresholds for target interval
    thresholds, bar_threshold = get_15min_threshold(close_prices, n_bins, window_size)
    
    # Adjust threshold for target minutes
    if target_minutes != 15:
        h_per_min = bar_threshold / 15.0
        bar_threshold = h_per_min * target_minutes
    
    tprint_info(f"🔧 Entropy bar calibration: n_bins={n_bins}, threshold={bar_threshold:.4f}")
    
    # Generate entropy bar indices
    bar_indices = process_ohlcv_entropy_bars(close_prices, thresholds, window_size, bar_threshold)
    
    if len(bar_indices) == 0:
        tprint_warning("⚠️ No entropy bars generated")
        return pd.DataFrame()
    
    # Build entropy bars OHLCV (vectorized)
    entropy_bars = []
    
    for i, idx in enumerate(bar_indices):
        # Determine the start of this bar
        start_idx = bar_indices[i-1] + 1 if i > 0 else 0
        
        # Extract bar data
        bar_data = ohlcv_data.iloc[start_idx:idx+1]
        
        if bar_data.empty:
            continue
            
        entropy_bar = {
            'timestamp': bar_data.index[-1],  # Use end timestamp
            'open': bar_data['open'].iloc[0],
            'high': bar_data['high'].max(),
            'low': bar_data['low'].min(),
            'close': bar_data['close'].iloc[-1],
            'volume': bar_data['volume'].sum(),
            'n_minutes': len(bar_data),
            'entropy_contribution': fast_shannon_entropy(
                np.searchsorted(thresholds, np.diff(np.log(bar_data['close'].values))),
                n_bins
            )
        }
        entropy_bars.append(entropy_bar)
    
    if not entropy_bars:
        tprint_warning("⚠️ No valid entropy bars created")
        return pd.DataFrame()
    
    result_df = pd.DataFrame(entropy_bars)
    result_df.set_index('timestamp', inplace=True)
    
    tprint_info(f"✅ Generated {len(result_df)} entropy bars for {symbol}/{exchange}")
    tprint_info(f"📊 Average bars per hour: {len(result_df) / (len(result_df) * target_minutes / 60):.2f}")
    
    return result_df


def calculate_specialized_entropy_features(
    entropy_bars: pd.DataFrame,
    base_model_updates: Optional[pd.DataFrame] = None,
    specialist_prices: Optional[pd.Series] = None,
    volatility_window: int = 20,
    use_optimized: bool = True
) -> pd.DataFrame:
    """
    Calculate specialized features for entropy bars with optimization.
    
    Args:
        entropy_bars: DataFrame with entropy bars data
        base_model_updates: DataFrame with base model update timestamps
        specialist_prices: Series with specialist bar close prices
        volatility_window: Window for volatility calculation
        use_optimized: Whether to use optimized functions
        
    Returns:
        DataFrame with specialized features
    """
    if entropy_bars.empty:
        return pd.DataFrame()
    
    if use_optimized and OPTIMIZED_AVAILABLE:
        # Use optimized entropy feature calculation
        tprint_info("🚀 Using optimized entropy feature calculation")
        
        config = {
            'entropy_bins': 10,
            'entropy_window': 100,
            'volatility_window': volatility_window
        }
        
        return vectorized_entropy_features(
            df=entropy_bars,
            entropy_bars=entropy_bars,
            config=config,
            use_numba=True
        )
    
    # Fall back to original implementation
    features = pd.DataFrame(index=entropy_bars.index)
    
    # 1. Staleness Feature: Time elapsed since base models updated
    if base_model_updates is not None and not base_model_updates.empty:
        last_update_time = base_model_updates.index.max()
        features['staleness_seconds'] = (entropy_bars.index - last_update_time).total_seconds()
        features['staleness_minutes'] = features['staleness_seconds'] / 60.0
    else:
        features['staleness_seconds'] = 0.0
        features['staleness_minutes'] = 0.0
    
    # 2. Drift Proxy: Price change since specialist's last bar closed
    if specialist_prices is not None and len(specialist_prices) > 0:
        # Align with entropy bar timestamps
        aligned_specialist = specialist_prices.reindex(entropy_bars.index, method='ffill')
        if not aligned_specialist.isna().all():
            features['drift_proxy'] = (entropy_bars['close'] - aligned_specialist) / aligned_specialist
        else:
            features['drift_proxy'] = 0.0
    else:
        features['drift_proxy'] = 0.0
    
    # 3. Lempel-Ziv (LZ) Complexity Feature
    if OPTIMIZED_AVAILABLE:
        features['lz_complexity'] = lempel_ziv_complexity_numba(entropy_bars['close'].values)
    else:
        features['lz_complexity'] = 0.5  # Placeholder
    
    # 4. Trend Conviction Index (TCI): Delta Entropy / Delta Time
    if 'entropy_contribution' in entropy_bars.columns:
        entropy_diff = entropy_bars['entropy_contribution'].diff().fillna(0)
        time_diff = pd.Series(entropy_bars.index).diff().dt.total_seconds().fillna(60)  # Default 60 seconds
        features['trend_conviction_index'] = entropy_diff / (time_diff + 1e-9)
    else:
        features['trend_conviction_index'] = 0.0
    
    # 5. Staleness-Adjusted Drift: (CurrentPrice - SpecialistPrice_last_update) / Volatility
    returns = entropy_bars['close'].pct_change().fillna(0)
    volatility = returns.rolling(volatility_window).std().fillna(returns.std())
    
    if specialist_prices is not None and len(specialist_prices) > 0:
        aligned_specialist = specialist_prices.reindex(entropy_bars.index, method='ffill')
        if not aligned_specialist.isna().all():
            price_diff = entropy_bars['close'] - aligned_specialist
            features['staleness_adjusted_drift'] = price_diff / (volatility + 1e-9)
        else:
            features['staleness_adjusted_drift'] = 0.0
    else:
        features['staleness_adjusted_drift'] = 0.0
    
    # Add additional entropy-based features
    if 'entropy_contribution' in entropy_bars.columns:
        features['entropy_ma'] = entropy_bars['entropy_contribution'].rolling(10).mean()
        features['entropy_std'] = entropy_bars['entropy_contribution'].rolling(10).std()
        features['entropy_zscore'] = (entropy_bars['entropy_contribution'] - features['entropy_ma']) / (features['entropy_std'] + 1e-9)
    
    tprint_info(f"✅ Calculated {len(features.columns)} specialized entropy features")
    
    return features


def calculate_lz_complexity(series: np.ndarray, normalize: bool = True) -> np.ndarray:
    """
    Calculate Lempel-Ziv complexity for a time series.
    
    Args:
        series: Input time series
        normalize: Whether to normalize by series length
        
    Returns:
        Array of LZ complexity values (same length as input)
    """
    if OPTIMIZED_AVAILABLE:
        return lempel_ziv_complexity_numba(series, normalize)
    
    # Fallback implementation
    n = len(series)
    if n == 0:
        return np.array([0.0])
    
    # Simple LZ complexity approximation
    median_val = np.median(series)
    binary_seq = (series > median_val).astype(int)
    
    complexity_values = np.zeros(n)
    
    for i in range(1, n):
        subseq = binary_seq[:i+1]
        unique_patterns = set()
        for j in range(len(subseq)):
            for k in range(j+1, len(subseq)+1):
                pattern = tuple(subseq[j:k])
                unique_patterns.add(pattern)
        
        complexity = len(unique_patterns)
        complexity_values[i] = complexity / (i+1) if normalize else complexity
    
    return complexity_values


def fetch_1min_data_for_entropy_bars(
    symbol: str,
    exchange: str = "binance",
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    data_dir: str = "historical_data"
) -> Optional[pd.DataFrame]:
    """
    Fetch 1-minute OHLCV data for entropy bar generation.
    
    Args:
        symbol: Trading symbol
        exchange: Exchange name
        start_date: Start date for data fetch
        end_date: End date for data fetch
        data_dir: Base data directory
        
    Returns:
        DataFrame with 1-minute OHLCV data or None
    """
    try:
        from src.utils.data.klines_parquet import KlinesParquetManager
        
        manager = KlinesParquetManager(data_dir=data_dir, exchange=exchange)
        
        # Read 1-minute raw data
        data = manager.read_data(
            symbol=symbol,
            interval="1m",
            start_date=start_date,
            end_date=end_date,
            data_type="raw"
        )
        
        if data is not None and not data.empty:
            tprint_info(f"✅ Fetched {len(data)} 1-minute bars for {symbol}/{exchange}")
            return data
        else:
            tprint_warning(f"⚠️ No 1-minute data found for {symbol}/{exchange}")
            return None
            
    except Exception as e:
        tprint_error(f"❌ Error fetching 1-minute data: {e}")
        return None


# Export main functions
__all__ = [
    'fast_shannon_entropy',
    'process_ohlcv_entropy_bars', 
    'get_15min_threshold',
    'generate_entropy_bars_from_ohlcv',
    'calculate_specialized_entropy_features',
    'calculate_lz_complexity',
    'fetch_1min_data_for_entropy_bars',
    'vectorized_entropy_features'  # Export optimized function
]
