"""
OHLCV-Only Regime and Context Features Module

This module provides regime and context features derived solely from OHLCV data,
including volatility regimes, microstructure proxies, and lagged residual features.
"""

from typing import Any, Dict, List, Optional, Union, Tuple
import pandas as pd
import numpy as np
from enum import Enum

# Import tprint utilities
try:
    from src.utils.tprint import (
        tprint, tprint_info, tprint_success, tprint_warning, tprint_error, tprint_debug
    )
except ImportError:
    # Fallback implementation if tprint not available
    def tprint(*args, **kwargs): print("TPRINT:", *args, **kwargs)
    def tprint_info(*args, **kwargs): print("INFO:", *args, **kwargs)
    def tprint_success(*args, **kwargs): print("SUCCESS:", *args, **kwargs)
    def tprint_warning(*args, **kwargs): print("WARNING:", *args, **kwargs)
    def tprint_error(*args, **kwargs): print("ERROR:", *args, **kwargs)
    def tprint_debug(*args, **kwargs): print("DEBUG:", *args, **kwargs)


class RegimeType(Enum):
    """Enum for regime types."""
    VOLATILITY = "volatility"
    TREND = "trend"
    MICROSTRUCTURE = "microstructure"
    COMPOSITE = "composite"


class OHLCVRegimeFeatures:
    """
    Generate OHLCV-only regime and context features.
    """
    
    def __init__(
        self,
        volatility_window: int = 20,
        trend_window: int = 50,
        microstructure_window: int = 10,
        lag_periods: List[int] = [1, 2, 3, 5, 10],
        residual_window: int = 20,
        enable_volatility_regimes: bool = True,
        enable_microstructure_proxies: bool = True,
        enable_lagged_residuals: bool = True,
        enable_trend_features: bool = True,
        n_volatility_regimes: int = 3,
        n_trend_regimes: int = 3
    ):
        """
        Initialize OHLCV regime features generator.
        
        Args:
            volatility_window: Window for volatility calculations
            trend_window: Window for trend calculations
            microstructure_window: Window for microstructure features
            lag_periods: List of lag periods for residual features
            residual_window: Window for residual calculations
            enable_volatility_regimes: Whether to generate volatility regime features
            enable_microstructure_proxies: Whether to generate microstructure proxies
            enable_lagged_residuals: Whether to generate lagged residual features
            enable_trend_features: Whether to generate trend features
            n_volatility_regimes: Number of volatility regimes
            n_trend_regimes: Number of trend regimes
        """
        self.volatility_window = volatility_window
        self.trend_window = trend_window
        self.microstructure_window = microstructure_window
        self.lag_periods = lag_periods
        self.residual_window = residual_window
        
        self.enable_volatility_regimes = enable_volatility_regimes
        self.enable_microstructure_proxies = enable_microstructure_proxies
        self.enable_lagged_residuals = enable_lagged_residuals
        self.enable_trend_features = enable_trend_features
        
        self.n_volatility_regimes = n_volatility_regimes
        self.n_trend_regimes = n_trend_regimes
    
    def generate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate all OHLCV-only regime and context features.
        
        Args:
            df: Input DataFrame with OHLCV columns
            
        Returns:
            DataFrame with additional regime features
        """
        # Validate input
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        result_df = df.copy()
        
        tprint_info("Generating OHLCV-only regime features...")
        
        if self.enable_volatility_regimes:
            tprint_info("  Generating volatility regime features...")
            vol_features = self._generate_volatility_regimes(df)
            result_df = pd.concat([result_df, vol_features], axis=1)
        
        if self.enable_trend_features:
            tprint_info("  Generating trend features...")
            trend_features = self._generate_trend_features(df)
            result_df = pd.concat([result_df, trend_features], axis=1)
        
        if self.enable_microstructure_proxies:
            tprint_info("  Generating microstructure proxies...")
            micro_features = self._generate_microstructure_proxies(df)
            result_df = pd.concat([result_df, micro_features], axis=1)
        
        if self.enable_lagged_residuals:
            tprint_info("  Generating lagged residual features...")
            residual_features = self._generate_lagged_residuals(df)
            result_df = pd.concat([result_df, residual_features], axis=1)
        
        tprint_success(f"Generated {len(result_df.columns) - len(df.columns)} regime features")
        
        return result_df
    
    def _generate_volatility_regimes(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate volatility regime features."""
        features = pd.DataFrame(index=df.index)
        
        # Calculate returns
        returns = df['close'].pct_change()
        log_returns = np.log(df['close']).diff()
        
        # Rolling volatility measures
        features['volatility_std'] = returns.rolling(self.volatility_window).std()
        features['volatility_atr'] = self._calculate_atr(df, self.volatility_window)
        features['volatility_range'] = (df['high'] - df['low']) / df['close']
        features['volatility_gap'] = (df['open'] - df['close'].shift(1)) / df['close'].shift(1)
        
        # Volatility of volatility
        features['vol_of_vol'] = features['volatility_std'].rolling(self.volatility_window).std()
        
        # Volatility regime classification
        vol_percentiles = features['volatility_std'].quantile([1/self.n_volatility_regimes, 2/self.n_volatility_regimes])
        
        # Create regime indicators
        features['vol_regime_low'] = (features['volatility_std'] <= vol_percentiles.iloc[0]).astype(int)
        features['vol_regime_med'] = ((features['volatility_std'] > vol_percentiles.iloc[0]) & 
                                     (features['volatility_std'] <= vol_percentiles.iloc[1])).astype(int)
        features['vol_regime_high'] = (features['volatility_std'] > vol_percentiles.iloc[1]).astype(int)
        
        # Volatility relative to recent history
        vol_mean = features['volatility_std'].rolling(self.volatility_window * 2).mean()
        features['vol_relative'] = features['volatility_std'] / (vol_mean + 1e-8)
        
        # Volatility trend
        features['vol_trend'] = features['volatility_std'].rolling(10).apply(
            lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) > 1 else 0
        )
        
        return features
    
    def _generate_trend_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate trend features."""
        features = pd.DataFrame(index=df.index)
        
        # Price-based trend indicators
        features['price_sma'] = df['close'].rolling(self.trend_window).mean()
        features['price_ema'] = df['close'].ewm(span=self.trend_window).mean()
        features['price_trend'] = (df['close'] - features['price_sma']) / features['price_sma']
        
        # Moving average crossovers
        sma_short = df['close'].rolling(self.trend_window // 2).mean()
        features['ma_crossover'] = (sma_short > features['price_sma']).astype(int)
        
        # Trend strength (R-squared of linear trend)
        def trend_strength(prices):
            if len(prices) < 3:
                return 0
            x = np.arange(len(prices))
            slope, intercept = np.polyfit(x, prices, 1)
            y_pred = slope * x + intercept
            ss_res = np.sum((prices - y_pred) ** 2)
            ss_tot = np.sum((prices - np.mean(prices)) ** 2)
            return 1 - (ss_res / (ss_tot + 1e-8))
        
        features['trend_strength'] = df['close'].rolling(self.trend_window).apply(trend_strength)
        
        # Trend regime classification
        trend_percentiles = features['price_trend'].quantile([1/self.n_trend_regimes, 2/self.n_trend_regimes])
        
        features['trend_regime_down'] = (features['price_trend'] <= trend_percentiles.iloc[0]).astype(int)
        features['trend_regime_sideways'] = ((features['price_trend'] > trend_percentiles.iloc[0]) & 
                                            (features['price_trend'] <= trend_percentiles.iloc[1])).astype(int)
        features['trend_regime_up'] = (features['price_trend'] > trend_percentiles.iloc[1]).astype(int)
        
        # Momentum indicators
        features['momentum_5'] = df['close'] / df['close'].shift(5) - 1
        features['momentum_10'] = df['close'] / df['close'].shift(10) - 1
        features['momentum_20'] = df['close'] / df['close'].shift(20) - 1
        
        return features
    
    def _generate_microstructure_proxies(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate microstructure proxy features."""
        features = pd.DataFrame(index=df.index)
        
        # Price efficiency measures
        features['bid_ask_proxy'] = (df['high'] - df['low']) / df['close']
        features['spread_proxy'] = (df['high'] - df['low']) / df['close'].rolling(self.microstructure_window).mean()
        
        # Volume-based microstructure
        volume_change = df['volume'].pct_change()
        features['volume_volatility'] = volume_change.rolling(self.microstructure_window).std()
        features['volume_price_corr'] = df['close'].pct_change().rolling(self.microstructure_window).corr(
            df['volume'].pct_change()
        )
        
        # Order flow imbalance proxies
        features['buy_pressure'] = (df['close'] - df['low']) / (df['high'] - df['low'] + 1e-8)
        features['sell_pressure'] = (df['high'] - df['close']) / (df['high'] - df['low'] + 1e-8)
        features['pressure_imbalance'] = features['buy_pressure'] - features['sell_pressure']
        
        # Intraday patterns
        features['intraday_return'] = (df['close'] - df['open']) / df['open']
        features['overnight_gap'] = (df['open'] - df['close'].shift(1)) / df['close'].shift(1)
        
        # Microstructure noise
        price_noise = df['close'] - df['close'].rolling(self.microstructure_window).mean()
        features['microstructure_noise'] = price_noise.rolling(self.microstructure_window).std()
        
        # Liquidity proxy
        features['liquidity_proxy'] = df['volume'] / (features['bid_ask_proxy'] + 1e-8)
        
        # Efficiency ratio (Kaufman's Efficiency Ratio)
        change = abs(df['close'] - df['close'].shift(self.microstructure_window))
        volatility = abs(df['close'].diff()).rolling(self.microstructure_window).sum()
        features['efficiency_ratio'] = change / (volatility + 1e-8)
        
        return features
    
    def _generate_lagged_residuals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate lagged residual features."""
        features = pd.DataFrame(index=df.index)
        
        # Calculate price residuals from simple trend
        price_trend = df['close'].rolling(self.residual_window).mean()
        residuals = df['close'] - price_trend
        
        # Create lagged residual features
        for lag in self.lag_periods:
            features[f'residual_lag_{lag}'] = residuals.shift(lag)
            features[f'residual_lag_{lag}_abs'] = abs(residuals.shift(lag))
        
        # Residual statistics
        features['residual_mean'] = residuals.rolling(self.residual_window).mean()
        features['residual_std'] = residuals.rolling(self.residual_window).std()
        features['residual_zscore'] = residuals / (features['residual_std'] + 1e-8)
        
        # Residual momentum
        features['residual_momentum'] = residuals - residuals.shift(5)
        
        # Residual regime (positive vs negative)
        features['residual_regime_pos'] = (residuals > 0).astype(int)
        features['residual_regime_neg'] = (residuals <= 0).astype(int)
        
        # Residual autocorrelation
        def autocorr(series, lag=1):
            if len(series) < lag + 1:
                return np.nan
            return series.autocorr(lag=lag)
        
        features['residual_autocorr_1'] = residuals.rolling(self.residual_window).apply(
            lambda x: autocorr(x, 1)
        )
        features['residual_autocorr_5'] = residuals.rolling(self.residual_window).apply(
            lambda x: autocorr(x, 5)
        )
        
        return features
    
    def _calculate_atr(self, df: pd.DataFrame, window: int) -> pd.Series:
        """Calculate Average True Range."""
        high_low = df['high'] - df['low']
        high_close = abs(df['high'] - df['close'].shift(1))
        low_close = abs(df['low'] - df['close'].shift(1))
        
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = true_range.rolling(window).mean()
        
        return atr


def add_ohlcv_regime_features(
    df: pd.DataFrame,
    config: Optional[Dict[str, Any]] = None,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Add OHLCV-only regime and context features to DataFrame.
    
    Args:
        df: Input DataFrame with OHLCV columns
        config: Configuration dictionary for feature generation
        verbose: Whether to print progress information
        
    Returns:
        DataFrame with added regime features
    """
    if config is None:
        config = {}
    
    # Initialize feature generator
    generator = OHLCVRegimeFeatures(
        volatility_window=config.get('volatility_window', 20),
        trend_window=config.get('trend_window', 50),
        microstructure_window=config.get('microstructure_window', 10),
        lag_periods=config.get('lag_periods', [1, 2, 3, 5, 10]),
        residual_window=config.get('residual_window', 20),
        enable_volatility_regimes=config.get('enable_volatility_regimes', True),
        enable_microstructure_proxies=config.get('enable_microstructure_proxies', True),
        enable_lagged_residuals=config.get('enable_lagged_residuals', True),
        enable_trend_features=config.get('enable_trend_features', True),
        n_volatility_regimes=config.get('n_volatility_regimes', 3),
        n_trend_regimes=config.get('n_trend_regimes', 3)
    )
    
    # Generate features
    result_df = generator.generate_features(df)
    
    if verbose:
        n_features = len(result_df.columns) - len(df.columns)
        tprint_success(f"Added {n_features} OHLCV regime features")
    
    return result_df
