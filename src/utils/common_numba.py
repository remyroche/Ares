"""
Common Numba-optimized functions for financial calculations.

This module provides JIT-compiled versions of frequently used operations
in financial time series analysis for maximum performance.
"""

import numpy as np
import pandas as pd
from typing import Optional, Tuple, Union
from numba import jit, njit, prange, types
from numba.typed import List
import warnings

try:
    from src.utils.tprint import tprint_info, tprint_warning
except ImportError:
    def tprint_info(msg): print(f"[INFO] {msg}")
    def tprint_warning(msg): print(f"[WARNING] {msg}")


@njit(parallel=True, fastmath=True)
def rolling_correlation_numba(
    x: np.ndarray, 
    y: np.ndarray, 
    window: int
) -> np.ndarray:
    """
    Calculate rolling correlation between two arrays using Numba.
    
    Args:
        x: First array
        y: Second array
        window: Rolling window size
        
    Returns:
        Array of rolling correlations
    """
    n = len(x)
    if len(y) != n:
        raise ValueError("Arrays must have same length")
    
    result = np.full(n, np.nan)
    
    if window >= n:
        return result
    
    for i in prange(window - 1, n):
        x_window = x[i - window + 1:i + 1]
        y_window = y[i - window + 1:i + 1]
        
        if len(x_window) > 1:
            x_mean = np.mean(x_window)
            y_mean = np.mean(y_window)
            
            x_centered = x_window - x_mean
            y_centered = y_window - y_mean
            
            numerator = np.sum(x_centered * y_centered)
            denominator = np.sqrt(np.sum(x_centered**2) * np.sum(y_centered**2))
            
            if denominator > 0:
                result[i] = numerator / denominator
            else:
                result[i] = 0.0
    
    return result


@njit(parallel=True, fastmath=True)
def rolling_beta_numba(
    returns: np.ndarray, 
    benchmark_returns: np.ndarray, 
    window: int
) -> np.ndarray:
    """
    Calculate rolling beta using Numba.
    
    Args:
        returns: Asset returns
        benchmark_returns: Benchmark returns
        window: Rolling window size
        
    Returns:
        Array of rolling betas
    """
    n = len(returns)
    if len(benchmark_returns) != n:
        raise ValueError("Arrays must have same length")
    
    result = np.full(n, np.nan)
    
    if window >= n:
        return result
    
    for i in prange(window - 1, n):
        ret_window = returns[i - window + 1:i + 1]
        bench_window = benchmark_returns[i - window + 1:i + 1]
        
        if len(ret_window) > 1:
            covariance = np.cov(ret_window, bench_window)[0, 1]
            benchmark_variance = np.var(bench_window)
            
            if benchmark_variance > 0:
                result[i] = covariance / benchmark_variance
            else:
                result[i] = 0.0
    
    return result


@njit(parallel=True, fastmath=True)
def rolling_sharpe_numba(
    returns: np.ndarray, 
    window: int, 
    risk_free_rate: float = 0.0
) -> np.ndarray:
    """
    Calculate rolling Sharpe ratio using Numba.
    
    Args:
        returns: Array of returns
        window: Rolling window size
        risk_free_rate: Risk-free rate (annualized)
        
    Returns:
        Array of rolling Sharpe ratios
    """
    n = len(returns)
    result = np.full(n, np.nan)
    
    if window >= n:
        return result
    
    # Convert annual risk-free rate to per-period
    rf_per_period = risk_free_rate / 252  # Assuming daily data
    
    for i in prange(window - 1, n):
        ret_window = returns[i - window + 1:i + 1]
        excess_returns = ret_window - rf_per_period
        
        if len(ret_window) > 1:
            mean_excess = np.mean(excess_returns)
            std_excess = np.std(excess_returns)
            
            if std_excess > 0:
                # Annualize
                result[i] = mean_excess / std_excess * np.sqrt(252)
            else:
                result[i] = 0.0
    
    return result


@njit(parallel=True, fastmath=True)
def rolling_sortino_numba(
    returns: np.ndarray, 
    window: int, 
    target_return: float = 0.0
) -> np.ndarray:
    """
    Calculate rolling Sortino ratio using Numba.
    
    Args:
        returns: Array of returns
        window: Rolling window size
        target_return: Target return
        
    Returns:
        Array of rolling Sortino ratios
    """
    n = len(returns)
    result = np.full(n, np.nan)
    
    if window >= n:
        return result
    
    for i in prange(window - 1, n):
        ret_window = returns[i - window + 1:i + 1]
        
        if len(ret_window) > 1:
            mean_return = np.mean(ret_window)
            downside_returns = ret_window[ret_window < target_return]
            
            if len(downside_returns) > 0:
                downside_deviation = np.std(downside_returns)
                if downside_deviation > 0:
                    # Annualize
                    result[i] = (mean_return - target_return) / downside_deviation * np.sqrt(252)
                else:
                    result[i] = 0.0 if mean_return <= target_return else np.inf
            else:
                result[i] = np.inf if mean_return > target_return else 0.0
    
    return result


@njit(parallel=True, fastmath=True)
def rolling_max_drawdown_numba(
    prices: np.ndarray, 
    window: int
) -> np.ndarray:
    """
    Calculate rolling maximum drawdown using Numba.
    
    Args:
        prices: Array of prices
        window: Rolling window size
        
    Returns:
        Array of rolling maximum drawdowns
    """
    n = len(prices)
    result = np.full(n, np.nan)
    
    if window >= n:
        return result
    
    for i in prange(window - 1, n):
        price_window = prices[i - window + 1:i + 1]
        
        if len(price_window) > 1:
            peak = price_window[0]
            max_dd = 0.0
            
            for j in range(1, len(price_window)):
                if price_window[j] > peak:
                    peak = price_window[j]
                
                drawdown = (price_window[j] - peak) / peak
                if drawdown < max_dd:
                    max_dd = drawdown
            
            result[i] = max_dd
    
    return result


@njit(parallel=True, fastmath=True)
def calculate_risk_metrics_numba(
    returns: np.ndarray,
    window: int = 252
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate comprehensive risk metrics using Numba.
    
    Args:
        returns: Array of returns
        window: Rolling window size
        
    Returns:
        Tuple of (volatility, skewness, kurtosis, var_95)
    """
    n = len(returns)
    volatility = np.full(n, np.nan)
    skewness = np.full(n, np.nan)
    kurtosis = np.full(n, np.nan)
    var_95 = np.full(n, np.nan)
    
    if window >= n:
        return volatility, skewness, kurtosis, var_95
    
    for i in prange(window - 1, n):
        ret_window = returns[i - window + 1:i + 1]
        
        if len(ret_window) > 1:
            # Volatility
            vol = np.std(ret_window)
            volatility[i] = vol
            
            # Skewness
            mean_ret = np.mean(ret_window)
            std_ret = np.std(ret_window)
            if std_ret > 0:
                skew = np.mean(((ret_window - mean_ret) / std_ret)**3)
                skewness[i] = skew
            else:
                skewness[i] = 0.0
            
            # Kurtosis
            if std_ret > 0:
                kurt = np.mean(((ret_window - mean_ret) / std_ret)**4) - 3
                kurtosis[i] = kurt
            else:
                kurtosis[i] = 0.0
            
            # Value at Risk (95%)
            var_95[i] = np.percentile(ret_window, 5)
    
    return volatility, skewness, kurtosis, var_95


@njit(parallel=True, fastmath=True)
def calculate_portfolio_metrics_numba(
    returns_matrix: np.ndarray,
    weights: np.ndarray
) -> Tuple[float, float, float, float]:
    """
    Calculate portfolio metrics using Numba.
    
    Args:
        returns_matrix: Matrix of asset returns (n_periods x n_assets)
        weights: Portfolio weights
        
    Returns:
        Tuple of (portfolio_return, portfolio_volatility, sharpe_ratio, max_drawdown)
    """
    # Portfolio returns
    portfolio_returns = np.zeros(returns_matrix.shape[0])
    
    for i in prange(len(portfolio_returns)):
        portfolio_returns[i] = np.sum(returns_matrix[i] * weights)
    
    # Basic metrics
    portfolio_return = np.mean(portfolio_returns)
    portfolio_volatility = np.std(portfolio_returns)
    
    # Sharpe ratio
    sharpe_ratio = portfolio_return / portfolio_volatility if portfolio_volatility > 0 else 0.0
    
    # Maximum drawdown
    cumulative_returns = np.cumprod(1 + portfolio_returns)
    peak = cumulative_returns[0]
    max_drawdown = 0.0
    
    for i in range(1, len(cumulative_returns)):
        if cumulative_returns[i] > peak:
            peak = cumulative_returns[i]
        
        drawdown = (cumulative_returns[i] - peak) / peak
        if drawdown < max_drawdown:
            max_drawdown = drawdown
    
    return portfolio_return, portfolio_volatility, sharpe_ratio, max_drawdown


@njit(fastmath=True)
def efficient_frontier_numba(
    returns_matrix: np.ndarray,
    n_portfolios: int = 1000
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate efficient frontier using Numba (simplified Monte Carlo).
    
    Args:
        returns_matrix: Matrix of asset returns
        n_portfolios: Number of random portfolios to generate
        
    Returns:
        Tuple of (returns, volatilities, sharpe_ratios)
    """
    n_assets = returns_matrix.shape[1]
    
    portfolio_returns = np.zeros(n_portfolios)
    portfolio_volatilities = np.zeros(n_portfolios)
    portfolio_sharpes = np.zeros(n_portfolios)
    
    for i in range(n_portfolios):
        # Generate random weights
        weights = np.random.random(n_assets)
        weights = weights / np.sum(weights)
        
        # Calculate portfolio metrics
        port_return, port_vol, port_sharpe, _ = calculate_portfolio_metrics_numba(
            returns_matrix, weights
        )
        
        portfolio_returns[i] = port_return
        portfolio_volatilities[i] = port_vol
        portfolio_sharpes[i] = port_sharpe
    
    return portfolio_returns, portfolio_volatilities, portfolio_sharpes


def vectorized_financial_metrics(
    df: pd.DataFrame,
    price_col: str = 'close',
    benchmark_col: Optional[str] = None,
    windows: Optional[list] = None,
    use_numba: bool = True
) -> pd.DataFrame:
    """
    Calculate comprehensive financial metrics with optional Numba acceleration.
    
    Args:
        df: DataFrame with price data
        price_col: Column name for prices
        benchmark_col: Column name for benchmark prices
        windows: List of windows for calculations
        use_numba: Whether to use Numba acceleration
        
    Returns:
        DataFrame with calculated metrics
    """
    if windows is None:
        windows = [20, 60, 252]
    
    if price_col not in df.columns:
        raise ValueError(f"Column '{price_col}' not found in DataFrame")
    
    # Calculate returns
    prices = df[price_col].values
    returns = np.diff(np.log(prices))
    returns = np.concatenate([np.array([0.0]), returns])  # Pad to match length
    
    metrics = pd.DataFrame(index=df.index)
    metrics['returns'] = returns
    
    if use_numba:
        tprint_info("🚀 Using Numba-optimized financial metrics calculation")
        
        # Risk metrics
        volatility, skewness, kurtosis, var_95 = calculate_risk_metrics_numba(returns, windows[0])
        metrics['volatility'] = volatility
        metrics['skewness'] = skewness
        metrics['kurtosis'] = kurtosis
        metrics['var_95'] = var_95
        
        # Performance metrics for each window
        for window in windows:
            if window < len(returns):
                # Sharpe ratio
                sharpe = rolling_sharpe_numba(returns, window)
                metrics[f'sharpe_{window}'] = sharpe
                
                # Sortino ratio
                sortino = rolling_sortino_numba(returns, window)
                metrics[f'sortino_{window}'] = sortino
                
                # Maximum drawdown
                max_dd = rolling_max_drawdown_numba(prices, window)
                metrics[f'max_drawdown_{window}'] = max_dd
        
        # Beta if benchmark provided
        if benchmark_col and benchmark_col in df.columns:
            benchmark_prices = df[benchmark_col].values
            benchmark_returns = np.diff(np.log(benchmark_prices))
            benchmark_returns = np.concatenate([np.array([0.0]), benchmark_returns])
            
            for window in windows:
                if window < len(returns):
                    beta = rolling_beta_numba(returns, benchmark_returns, window)
                    metrics[f'beta_{window}'] = beta
                    
                    # Correlation
                    correlation = rolling_correlation_numba(returns, benchmark_returns, window)
                    metrics[f'correlation_{window}'] = correlation
    
    else:
        # Fall back to pandas implementation
        tprint_info("📊 Using pandas implementation for financial metrics")
        
        returns_series = pd.Series(returns, index=df.index)
        
        for window in windows:
            if window < len(df):
                # Basic metrics
                metrics[f'volatility_{window}'] = returns_series.rolling(window).std()
                metrics[f'sharpe_{window}'] = returns_series.rolling(window).mean() / returns_series.rolling(window).std()
                metrics[f'max_drawdown_{window}'] = (prices / pd.Series(prices).rolling(window).max() - 1)
                
                # Beta if benchmark provided
                if benchmark_col and benchmark_col in df.columns:
                    benchmark_returns = df[benchmark_col].pct_change()
                    rolling_cov = returns_series.rolling(window).cov(benchmark_returns.rolling(window))
                    benchmark_var = benchmark_returns.rolling(window).var()
                    metrics[f'beta_{window}'] = rolling_cov / benchmark_var
    
    return metrics


# Export main functions
__all__ = [
    'vectorized_financial_metrics',
    'rolling_correlation_numba',
    'rolling_beta_numba',
    'rolling_sharpe_numba',
    'rolling_sortino_numba',
    'rolling_max_drawdown_numba',
    'calculate_risk_metrics_numba',
    'calculate_portfolio_metrics_numba',
    'efficient_frontier_numba'
]
