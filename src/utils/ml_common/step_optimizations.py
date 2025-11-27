"""
Step-Specific Optimizations for ML Pipeline Steps

This module provides optimized implementations for common operations across ML steps:
1. float32 conversion for memory efficiency
2. Vectorized ATR/S/R calculations
3. Caching utilities for computed features
4. Rolling window optimizations
5. HMM configuration helpers
6. Path encoding utilities

Usage:
    from src.utils.ml_common.step_optimizations import (
        convert_to_float32,
        VectorizedATRCalculator,
        FeatureCache,
        OptimizedRollingWindow,
        get_optimized_hmm_config,
        FixedLengthPathEncoder,
    )
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ============================================================================
# Memory Optimization: float32 Conversion
# ============================================================================

def convert_to_float32(
    df: pd.DataFrame,
    exclude_cols: Optional[List[str]] = None,
    inplace: bool = False
) -> pd.DataFrame:
    """
    Convert DataFrame float columns to float32 for 50% memory reduction.
    
    Args:
        df: Input DataFrame
        exclude_cols: Columns to exclude from conversion
        inplace: If True, modify in place
        
    Returns:
        DataFrame with float32 columns
    """
    exclude_cols = exclude_cols or []
    
    if not inplace:
        df = df.copy()
    
    float_cols = df.select_dtypes(include=['float64', 'float']).columns
    cols_to_convert = [c for c in float_cols if c not in exclude_cols]
    
    for col in cols_to_convert:
        df[col] = df[col].astype(np.float32)
    
    return df


def convert_arrays_to_float32(*arrays: np.ndarray) -> Tuple[np.ndarray, ...]:
    """Convert multiple numpy arrays to float32."""
    return tuple(arr.astype(np.float32) if arr.dtype == np.float64 else arr for arr in arrays)


# ============================================================================
# Vectorized ATR and S/R Calculations
# ============================================================================

class VectorizedATRCalculator:
    """
    Vectorized ATR (Average True Range) calculation with caching.
    
    Much faster than per-bar calculations, especially for large datasets.
    """
    
    def __init__(self, cache_ttl_minutes: int = 60):
        self.cache_ttl = timedelta(minutes=cache_ttl_minutes)
        self._cache: Dict[str, Tuple[datetime, np.ndarray]] = {}
    
    def calculate_atr(
        self,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
        period: int = 14,
        cache_key: Optional[str] = None
    ) -> np.ndarray:
        """
        Calculate ATR using vectorized operations.
        
        Args:
            high: High prices
            low: Low prices
            close: Close prices
            period: ATR period
            cache_key: Optional key for caching
            
        Returns:
            ATR values
        """
        # Check cache
        if cache_key and cache_key in self._cache:
            cached_time, cached_atr = self._cache[cache_key]
            if datetime.now() - cached_time < self.cache_ttl:
                return cached_atr
        
        # Vectorized True Range calculation
        high = np.asarray(high, dtype=np.float32)
        low = np.asarray(low, dtype=np.float32)
        close = np.asarray(close, dtype=np.float32)
        
        prev_close = np.roll(close, 1)
        prev_close[0] = close[0]
        
        tr1 = high - low
        tr2 = np.abs(high - prev_close)
        tr3 = np.abs(low - prev_close)
        
        true_range = np.maximum(np.maximum(tr1, tr2), tr3)
        
        # Vectorized EMA for ATR
        atr = self._ema_vectorized(true_range, period)
        
        # Cache result
        if cache_key:
            self._cache[cache_key] = (datetime.now(), atr)
        
        return atr
    
    @staticmethod
    def _ema_vectorized(data: np.ndarray, period: int) -> np.ndarray:
        """Vectorized EMA calculation."""
        alpha = 2.0 / (period + 1)
        
        # Initialize with SMA for first 'period' values
        sma = np.convolve(data, np.ones(period)/period, mode='valid')
        result = np.full_like(data, np.nan, dtype=np.float32)
        
        if len(sma) > 0:
            result[period-1] = sma[0]
            
            # Vectorized EMA calculation using cumsum trick
            for i in range(period, len(data)):
                result[i] = alpha * data[i] + (1 - alpha) * result[i-1]
        
        return result
    
    def calculate_distance_to_level(
        self,
        price: np.ndarray,
        level: float,
        atr: np.ndarray
    ) -> np.ndarray:
        """Calculate distance to price level in ATR units (vectorized)."""
        price = np.asarray(price, dtype=np.float32)
        atr_safe = np.where(atr > 0, atr, np.nan)
        return (price - level) / atr_safe
    
    def filter_by_atr_distance(
        self,
        price: np.ndarray,
        level: float,
        atr: np.ndarray,
        max_atr_distance: float = 2.0
    ) -> np.ndarray:
        """
        Create boolean mask for prices within ATR distance of level (vectorized).
        
        Returns:
            Boolean mask where True = within distance
        """
        distance = np.abs(self.calculate_distance_to_level(price, level, atr))
        return distance <= max_atr_distance
    
    def clear_cache(self):
        """Clear the ATR cache."""
        self._cache.clear()


class SupportResistanceCache:
    """
    Cache for support/resistance levels with time-based expiry.
    
    S/R levels change slowly, so caching provides significant speedup.
    """
    
    def __init__(self, ttl_hours: int = 1):
        self.ttl = timedelta(hours=ttl_hours)
        self._levels_cache: Dict[str, Tuple[datetime, Dict[str, Any]]] = {}
    
    def get_levels(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Get cached S/R levels if not expired."""
        if cache_key in self._levels_cache:
            cached_time, levels = self._levels_cache[cache_key]
            if datetime.now() - cached_time < self.ttl:
                return levels
        return None
    
    def set_levels(self, cache_key: str, levels: Dict[str, Any]) -> None:
        """Cache S/R levels."""
        self._levels_cache[cache_key] = (datetime.now(), levels)
    
    def invalidate(self, cache_key: Optional[str] = None) -> None:
        """Invalidate cache for specific key or all."""
        if cache_key:
            self._levels_cache.pop(cache_key, None)
        else:
            self._levels_cache.clear()


# ============================================================================
# Feature Caching
# ============================================================================

class FeatureCache:
    """
    General-purpose feature cache with TTL and size limits.
    
    Useful for caching:
    - Order flow features
    - Trend features
    - Volatility regime features
    - Path geometry features
    """
    
    def __init__(
        self,
        cache_dir: Optional[Path] = None,
        max_memory_mb: int = 500,
        default_ttl_minutes: int = 30
    ):
        self.cache_dir = cache_dir or Path("cache/features")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_memory_mb = max_memory_mb
        self.default_ttl = timedelta(minutes=default_ttl_minutes)
        
        self._memory_cache: Dict[str, Tuple[datetime, int, Any]] = {}  # key -> (time, size_bytes, data)
        self._total_bytes = 0
    
    def _compute_key(self, base_key: str, params: Dict[str, Any]) -> str:
        """Compute cache key from base key and parameters."""
        params_str = json.dumps(params, sort_keys=True, default=str)
        hash_str = hashlib.md5(params_str.encode()).hexdigest()[:8]
        return f"{base_key}_{hash_str}"
    
    def get(
        self,
        key: str,
        params: Optional[Dict[str, Any]] = None
    ) -> Optional[Any]:
        """Get cached feature data."""
        cache_key = self._compute_key(key, params or {})
        
        if cache_key in self._memory_cache:
            cached_time, size, data = self._memory_cache[cache_key]
            if datetime.now() - cached_time < self.default_ttl:
                return data
            else:
                # Expired - remove
                self._total_bytes -= size
                del self._memory_cache[cache_key]
        
        return None
    
    def set(
        self,
        key: str,
        data: Any,
        params: Optional[Dict[str, Any]] = None,
        ttl_minutes: Optional[int] = None
    ) -> None:
        """Cache feature data."""
        cache_key = self._compute_key(key, params or {})
        
        # Estimate size
        if isinstance(data, (pd.DataFrame, pd.Series)):
            size_bytes = data.memory_usage(deep=True).sum() if isinstance(data, pd.DataFrame) else data.memory_usage(deep=True)
        elif isinstance(data, np.ndarray):
            size_bytes = data.nbytes
        else:
            size_bytes = 1000  # Default estimate
        
        # Evict if over limit
        while self._total_bytes + size_bytes > self.max_memory_mb * 1024 * 1024 and self._memory_cache:
            # Remove oldest entry
            oldest_key = min(self._memory_cache.keys(), key=lambda k: self._memory_cache[k][0])
            self._total_bytes -= self._memory_cache[oldest_key][1]
            del self._memory_cache[oldest_key]
        
        self._memory_cache[cache_key] = (datetime.now(), size_bytes, data)
        self._total_bytes += size_bytes
    
    def clear(self) -> None:
        """Clear all cached data."""
        self._memory_cache.clear()
        self._total_bytes = 0


# ============================================================================
# Optimized Rolling Window Operations
# ============================================================================

class OptimizedRollingWindow:
    """
    Memory-efficient rolling window operations using EWM instead of rolling where appropriate.
    
    EWM is O(1) per update vs O(window) for rolling windows.
    """
    
    @staticmethod
    def ewm_mean(data: np.ndarray, span: int) -> np.ndarray:
        """Exponentially weighted mean (O(1) per point)."""
        alpha = 2.0 / (span + 1)
        result = np.empty_like(data, dtype=np.float32)
        result[0] = data[0]
        
        for i in range(1, len(data)):
            result[i] = alpha * data[i] + (1 - alpha) * result[i-1]
        
        return result
    
    @staticmethod
    def ewm_std(data: np.ndarray, span: int) -> np.ndarray:
        """Exponentially weighted standard deviation."""
        alpha = 2.0 / (span + 1)
        
        mean = np.empty_like(data, dtype=np.float32)
        var = np.empty_like(data, dtype=np.float32)
        
        mean[0] = data[0]
        var[0] = 0
        
        for i in range(1, len(data)):
            mean[i] = alpha * data[i] + (1 - alpha) * mean[i-1]
            diff = data[i] - mean[i-1]
            var[i] = (1 - alpha) * (var[i-1] + alpha * diff * diff)
        
        return np.sqrt(var)
    
    @staticmethod
    def rolling_aggregation_vectorized(
        data: pd.DataFrame,
        window: int,
        aggregations: Dict[str, str]
    ) -> pd.DataFrame:
        """
        Vectorized rolling aggregation for multiple columns.
        
        Args:
            data: Input DataFrame
            window: Rolling window size
            aggregations: Dict of {column: agg_func} where agg_func is 'mean', 'std', 'sum', etc.
            
        Returns:
            DataFrame with aggregated columns
        """
        result = pd.DataFrame(index=data.index)
        
        for col, agg_func in aggregations.items():
            if col not in data.columns:
                continue
                
            if agg_func == 'mean':
                result[f"{col}_roll_mean"] = data[col].rolling(window, min_periods=1).mean()
            elif agg_func == 'std':
                result[f"{col}_roll_std"] = data[col].rolling(window, min_periods=1).std()
            elif agg_func == 'sum':
                result[f"{col}_roll_sum"] = data[col].rolling(window, min_periods=1).sum()
            elif agg_func == 'min':
                result[f"{col}_roll_min"] = data[col].rolling(window, min_periods=1).min()
            elif agg_func == 'max':
                result[f"{col}_roll_max"] = data[col].rolling(window, min_periods=1).max()
            elif agg_func == 'ewm_mean':
                result[f"{col}_ewm_mean"] = data[col].ewm(span=window, adjust=False).mean()
            elif agg_func == 'ewm_std':
                result[f"{col}_ewm_std"] = data[col].ewm(span=window, adjust=False).std()
        
        return result


# ============================================================================
# HMM Configuration Helpers
# ============================================================================

@dataclass
class OptimizedHMMConfig:
    """Optimized HMM configuration."""
    n_components: int = 4
    covariance_type: str = 'diag'  # 'diag' is 5-10x faster than 'full'
    n_iter: int = 50  # Reduced from 100-200
    tol: float = 1e-3  # Less strict tolerance
    min_covar: float = 1e-3
    random_state: int = 42


def get_optimized_hmm_config(
    n_features: int,
    n_samples: int,
    task: str = "regime_detection"
) -> OptimizedHMMConfig:
    """
    Get optimized HMM configuration based on data characteristics.
    
    Args:
        n_features: Number of features
        n_samples: Number of samples
        task: Type of task ('regime_detection', 'macro_trend', 'alpha')
        
    Returns:
        Optimized HMM configuration
    """
    config = OptimizedHMMConfig()
    
    # Use diagonal covariance for speed (5-10x faster)
    # Only use full covariance if features are highly correlated AND sample size is large
    if n_features > 20 or n_samples < 5000:
        config.covariance_type = 'diag'
    
    # Reduce iterations based on task
    if task == "regime_detection":
        config.n_iter = 50
        config.n_components = 4
    elif task == "macro_trend":
        config.n_iter = 75
        config.n_components = 3
    elif task == "alpha":
        config.n_iter = 50
        config.n_components = 4
    
    # Adjust tolerance based on sample size
    if n_samples > 50000:
        config.tol = 1e-2  # More relaxed for large datasets
    elif n_samples > 10000:
        config.tol = 5e-3
    else:
        config.tol = 1e-3
    
    return config


# ============================================================================
# Fixed-Length Path Encoding
# ============================================================================

class FixedLengthPathEncoder:
    """
    Fixed-length path encoding for efficient path regime detection.
    
    Instead of variable-length paths, uses fixed k-bar patterns for consistency
    and vectorization.
    """
    
    def __init__(
        self,
        lookback: int = 48,  # Reduced from 200+ to 48
        encoding_dim: int = 16,
        use_float32: bool = True
    ):
        self.lookback = lookback
        self.encoding_dim = encoding_dim
        self.use_float32 = use_float32
    
    def encode_paths_batch(
        self,
        close: np.ndarray,
        high: np.ndarray,
        low: np.ndarray
    ) -> np.ndarray:
        """
        Batch encode price paths into fixed-length features.
        
        Features computed:
        - Normalized price changes at fixed intervals
        - Rolling volatility at fixed intervals
        - Directional efficiency
        - Shape statistics (skew, kurtosis of returns)
        
        Returns:
            Array of shape (n_samples, encoding_dim)
        """
        n = len(close)
        dtype = np.float32 if self.use_float32 else np.float64
        
        # Pre-allocate output
        encoded = np.zeros((n, self.encoding_dim), dtype=dtype)
        
        # Compute returns
        returns = np.diff(close) / close[:-1]
        returns = np.concatenate([[0], returns])
        
        # Fixed intervals for feature extraction
        intervals = [4, 8, 16, 24, 48][:min(5, self.lookback // 4)]
        
        feature_idx = 0
        
        for interval in intervals:
            if interval > self.lookback:
                continue
            
            # Rolling mean return
            if feature_idx < self.encoding_dim:
                roll_mean = pd.Series(returns).rolling(interval, min_periods=1).mean().values.astype(dtype)
                encoded[:, feature_idx] = roll_mean
                feature_idx += 1
            
            # Rolling volatility
            if feature_idx < self.encoding_dim:
                roll_std = pd.Series(returns).rolling(interval, min_periods=1).std().fillna(0).values.astype(dtype)
                encoded[:, feature_idx] = roll_std
                feature_idx += 1
        
        # Directional efficiency (path directness)
        if feature_idx < self.encoding_dim:
            price_change = np.zeros(n, dtype=dtype)
            price_change[self.lookback:] = (close[self.lookback:] - close[:-self.lookback]) / close[:-self.lookback]
            
            sum_abs_returns = np.zeros(n, dtype=dtype)
            abs_returns = np.abs(returns)
            for i in range(self.lookback, n):
                sum_abs_returns[i] = np.sum(abs_returns[i-self.lookback:i])
            
            efficiency = np.where(sum_abs_returns > 0, np.abs(price_change) / sum_abs_returns, 0)
            encoded[:, feature_idx] = efficiency
            feature_idx += 1
        
        # High-low range ratio
        if feature_idx < self.encoding_dim:
            range_ratio = (high - low) / np.where(close > 0, close, 1)
            roll_range = pd.Series(range_ratio).rolling(self.lookback // 2, min_periods=1).mean().values.astype(dtype)
            encoded[:, feature_idx] = roll_range
            feature_idx += 1
        
        # Return skewness (shape)
        if feature_idx < self.encoding_dim:
            skew = pd.Series(returns).rolling(self.lookback, min_periods=self.lookback//2).skew().fillna(0).values.astype(dtype)
            encoded[:, feature_idx] = np.clip(skew, -3, 3)
            feature_idx += 1
        
        # Return kurtosis (shape)
        if feature_idx < self.encoding_dim:
            kurt = pd.Series(returns).rolling(self.lookback, min_periods=self.lookback//2).kurt().fillna(0).values.astype(dtype)
            encoded[:, feature_idx] = np.clip(kurt, -10, 10)
            feature_idx += 1
        
        return encoded
    
    def get_feature_names(self) -> List[str]:
        """Get names of encoded features."""
        names = []
        intervals = [4, 8, 16, 24, 48][:min(5, self.lookback // 4)]
        
        for interval in intervals:
            names.append(f"path_return_mean_{interval}")
            names.append(f"path_volatility_{interval}")
        
        names.extend([
            "path_directional_efficiency",
            "path_range_ratio",
            "path_return_skewness",
            "path_return_kurtosis"
        ])
        
        return names[:self.encoding_dim]


# ============================================================================
# Volatility Regime Pre-computation
# ============================================================================

def precompute_volatility_regimes(
    returns: np.ndarray,
    windows: List[int] = [12, 24, 48],
    n_regimes: int = 3
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """
    Pre-compute volatility regime features using exponential smoothing.
    
    Uses EWM instead of rolling for O(1) per-point computation.
    
    Args:
        returns: Return series
        windows: List of window sizes for multi-scale volatility
        n_regimes: Number of volatility regimes
        
    Returns:
        Tuple of (regime_labels, feature_dict)
    """
    returns = np.asarray(returns, dtype=np.float32)
    features = {}
    
    # Multi-scale volatility using EWM
    for window in windows:
        vol = pd.Series(returns).ewm(span=window, adjust=False).std().fillna(0).values.astype(np.float32)
        features[f"ewm_vol_{window}"] = vol
    
    # Volatility of volatility
    vol_short = features[f"ewm_vol_{windows[0]}"]
    vol_of_vol = pd.Series(vol_short).ewm(span=windows[-1], adjust=False).std().fillna(0).values.astype(np.float32)
    features["vol_of_vol"] = vol_of_vol
    
    # Volatility trend (short vs long)
    vol_long = features[f"ewm_vol_{windows[-1]}"]
    vol_trend = np.where(vol_long > 0, vol_short / vol_long - 1, 0).astype(np.float32)
    features["vol_trend"] = vol_trend
    
    # Simple regime assignment based on volatility percentiles
    vol_main = vol_short
    percentiles = np.nanpercentile(vol_main, [33, 66])
    
    regimes = np.zeros(len(returns), dtype=np.int32)
    regimes[vol_main > percentiles[1]] = 2  # High volatility
    regimes[(vol_main > percentiles[0]) & (vol_main <= percentiles[1])] = 1  # Medium
    # regimes already 0 for low volatility
    
    return regimes, features


# ============================================================================
# SMC Vectorization Utilities
# ============================================================================

def vectorize_smc_calculations(
    df: pd.DataFrame,
    config: Dict[str, Any]
) -> pd.DataFrame:
    """
    Vectorized Smart Money Concept calculations.
    
    Replaces loop-based implementations with vectorized operations.
    
    Args:
        df: OHLCV DataFrame
        config: Configuration dictionary
        
    Returns:
        DataFrame with SMC features
    """
    result = pd.DataFrame(index=df.index)
    
    # Ensure float32
    close = df['close'].values.astype(np.float32)
    high = df['high'].values.astype(np.float32)
    low = df['low'].values.astype(np.float32)
    volume = df['volume'].values.astype(np.float32) if 'volume' in df.columns else np.ones_like(close)
    
    # Vectorized swing detection
    lookback = config.get('smc_swing_lookback', 5)
    
    # Swing highs: higher than surrounding bars
    roll_max = pd.Series(high).rolling(2*lookback+1, center=True, min_periods=1).max().values
    swing_highs = (high == roll_max).astype(np.float32)
    result['smc_swing_high'] = swing_highs
    
    # Swing lows: lower than surrounding bars
    roll_min = pd.Series(low).rolling(2*lookback+1, center=True, min_periods=1).min().values
    swing_lows = (low == roll_min).astype(np.float32)
    result['smc_swing_low'] = swing_lows
    
    # Order blocks (simplified vectorized version)
    # Bullish OB: down candle followed by strong up move
    body = close - df['open'].values.astype(np.float32) if 'open' in df.columns else np.zeros_like(close)
    is_down = body < 0
    
    # Rolling max close after each bar
    future_max = pd.Series(close).rolling(lookback, min_periods=1).max().shift(-lookback).values
    up_move = (future_max - close) / np.where(close > 0, close, 1)
    
    bullish_ob = (is_down & (up_move > config.get('smc_ob_threshold', 0.01))).astype(np.float32)
    result['smc_bullish_ob'] = bullish_ob
    
    # Bearish OB: up candle followed by strong down move
    future_min = pd.Series(close).rolling(lookback, min_periods=1).min().shift(-lookback).values
    down_move = (close - future_min) / np.where(close > 0, close, 1)
    
    is_up = body > 0
    bearish_ob = (is_up & (down_move > config.get('smc_ob_threshold', 0.01))).astype(np.float32)
    result['smc_bearish_ob'] = bearish_ob
    
    # Fair value gaps (FVG) - vectorized
    if len(df) >= 3:
        # Bullish FVG: gap between candle i-1 high and candle i+1 low
        prev_high = pd.Series(high).shift(1).values
        next_low = pd.Series(low).shift(-1).values
        bullish_fvg = (next_low > prev_high).astype(np.float32)
        result['smc_bullish_fvg'] = bullish_fvg
        
        # Bearish FVG
        prev_low = pd.Series(low).shift(1).values
        next_high = pd.Series(high).shift(-1).values
        bearish_fvg = (next_high < prev_low).astype(np.float32)
        result['smc_bearish_fvg'] = bearish_fvg
    
    # Displacement (strong momentum move) - vectorized
    returns = np.diff(close) / close[:-1]
    returns = np.concatenate([[0], returns])
    vol = pd.Series(returns).ewm(span=20, adjust=False).std().values.astype(np.float32)
    
    displacement = np.where(vol > 0, np.abs(returns) / vol, 0).astype(np.float32)
    result['smc_displacement'] = displacement
    
    # Liquidity sweep (break of recent high/low with reversal)
    recent_high = pd.Series(high).rolling(lookback, min_periods=1).max().shift(1).values
    recent_low = pd.Series(low).rolling(lookback, min_periods=1).min().shift(1).values
    
    sweep_high = ((high > recent_high) & (close < recent_high)).astype(np.float32)
    sweep_low = ((low < recent_low) & (close > recent_low)).astype(np.float32)
    result['smc_sweep_high'] = sweep_high
    result['smc_sweep_low'] = sweep_low
    
    return result


# ============================================================================
# Order Flow Feature Caching
# ============================================================================

class OrderFlowCache(FeatureCache):
    """
    Specialized cache for order flow features.
    
    Order flow features (from OHLCV) change slowly, so caching with
    5-minute TTL provides significant speedup.
    """
    
    def __init__(self):
        super().__init__(default_ttl_minutes=5)
    
    def get_or_compute(
        self,
        df: pd.DataFrame,
        config: Dict[str, Any],
        compute_fn: callable
    ) -> pd.DataFrame:
        """
        Get cached order flow features or compute and cache.
        
        Args:
            df: Input OHLCV DataFrame
            config: Configuration for feature computation
            compute_fn: Function to compute features if not cached
            
        Returns:
            Order flow features DataFrame
        """
        # Create cache key from data hash and config
        data_hash = hashlib.md5(
            f"{df.index[0]}_{df.index[-1]}_{len(df)}".encode()
        ).hexdigest()[:8]
        
        cached = self.get("order_flow", {"hash": data_hash, **config})
        if cached is not None:
            return cached
        
        # Compute features
        features = compute_fn(df, config)
        
        # Cache result
        self.set("order_flow", features, {"hash": data_hash, **config})
        
        return features


# ============================================================================
# Integration Helpers
# ============================================================================

def apply_all_optimizations(
    df: pd.DataFrame,
    config: Dict[str, Any]
) -> pd.DataFrame:
    """
    Apply all memory and computation optimizations to a DataFrame.
    
    Args:
        df: Input DataFrame
        config: Configuration dictionary
        
    Returns:
        Optimized DataFrame
    """
    # Convert to float32
    df = convert_to_float32(df)
    
    # Handle any infinite values
    df = df.replace([np.inf, -np.inf], np.nan)
    
    # Fill NaN with forward fill then zero
    df = df.ffill().fillna(0)
    
    return df


def get_optimal_stride(n_samples: int, target_samples: int = 5000) -> int:
    """
    Calculate optimal stride for HMM training to achieve target sample count.
    
    Args:
        n_samples: Original number of samples
        target_samples: Target number of samples for training
        
    Returns:
        Stride value
    """
    if n_samples <= target_samples:
        return 1
    
    stride = max(1, n_samples // target_samples)
    return min(stride, 20)  # Cap at 20 to maintain temporal resolution
