"""
VectorBT import stub for environments without native VectorBT support.

By default this stub prevents importing the real ``vectorbt`` package, allowing
the rest of the codebase to detect the missing dependency and fall back to
safe pandas/numpy implementations without crashing the interpreter.

To enable the real VectorBT package (if it is installed and compatible with
the current platform) set the environment variable ``ARES_ENABLE_VECTORBT=1``
before launching Python. When enabled, this stub delegates the import to the
next ``vectorbt`` distribution found on ``sys.path`` (excluding the project
workspace) so the genuine package can be used.
"""

from __future__ import annotations

import importlib.machinery
import importlib.util
import os
import sys
import warnings
from typing import Any, Optional, Union, List, Dict, Tuple
import pandas as pd
import numpy as np

_ENV_FLAG = os.environ.get("ARES_ENABLE_VECTORBT", "").strip().lower()
_ALLOW_REAL_VECTORBT = _ENV_FLAG in {"1", "true", "yes", "on"}

# Determine the workspace root (the path entry that contains this stub).
_STUB_DIR = os.path.dirname(__file__)
_WORKSPACE_ROOT = os.path.abspath(os.path.join(_STUB_DIR, os.pardir))

if _ALLOW_REAL_VECTORBT:
    # Search for the real vectorbt package on the rest of sys.path.
    _search_paths = [path for path in sys.path if os.path.abspath(path) != _WORKSPACE_ROOT]

    spec = importlib.machinery.PathFinder.find_spec("vectorbt", _search_paths)
    if spec is None or spec.loader is None:
        raise ImportError(
            "ARES_ENABLE_VECTORBT is set, but the real 'vectorbt' package could not be located "
            "outside the project workspace. Install vectorbt in your environment or unset "
            "ARES_ENABLE_VECTORBT to use the stub fallback."
        )

    # Load the genuine vectorbt module and replace this stub in sys.modules.
    module = importlib.util.module_from_spec(spec)
    sys.modules[__name__] = module
    spec.loader.exec_module(module)
    globals().update(module.__dict__)
else:
    # VectorBT Stub Implementation
    # Provide fallback implementations for VectorBT functionality using pandas/numpy
    
    warnings.warn(
        "VectorBT is disabled. Using pandas/numpy fallback implementations. "
        "Set ARES_ENABLE_VECTORBT=1 to enable the real VectorBT package if installed.",
        UserWarning
    )
    
    # Core VectorBT classes and functions
    class Portfolio:
        """Stub Portfolio class for VectorBT fallback."""
        
        def __init__(self, *args, **kwargs):
            self._data = None
            self._orders = None
            self._trades = None
            self._returns = None
            
        @property
        def data(self):
            return self._data
            
        @property
        def orders(self):
            return self._orders
            
        @property
        def trades(self):
            return self._trades
            
        @property
        def returns(self):
            return self._returns
            
        def total_return(self):
            """Calculate total return."""
            if self._returns is not None and len(self._returns) > 0:
                return (1 + self._returns).prod() - 1
            return 0.0
            
        def sharpe_ratio(self, risk_free_rate=0.0):
            """Calculate Sharpe ratio."""
            if self._returns is not None and len(self._returns) > 0:
                excess_returns = self._returns - risk_free_rate
                if excess_returns.std() > 0:
                    return excess_returns.mean() / excess_returns.std() * np.sqrt(252)
            return 0.0
            
        def max_drawdown(self):
            """Calculate maximum drawdown."""
            if self._returns is not None and len(self._returns) > 0:
                cumulative = (1 + self._returns).cumprod()
                running_max = cumulative.expanding().max()
                drawdown = (cumulative - running_max) / running_max
                return drawdown.min()
            return 0.0
    
    class PortfolioFactory:
        """Stub PortfolioFactory class for VectorBT fallback."""
        
        @staticmethod
        def from_signals(
            close: pd.Series,
            entries: pd.Series,
            exits: pd.Series,
            **kwargs
        ) -> Portfolio:
            """Create portfolio from signals using pandas fallback."""
            portfolio = Portfolio()
            
            # Simple signal-based portfolio simulation
            positions = pd.Series(0.0, index=close.index)
            positions[entries] = 1.0
            positions[exits] = 0.0
            positions = positions.fillna(method='ffill').fillna(0.0)
            
            # Calculate returns
            returns = close.pct_change()
            portfolio_returns = returns * positions.shift(1)
            
            portfolio._data = close
            portfolio._returns = portfolio_returns
            portfolio._orders = pd.DataFrame({
                'timestamp': close.index,
                'side': positions,
                'price': close
            })
            
            return portfolio
    
    # Generic rolling functions using pandas
    def rolling_mean(data: pd.Series, window: int, **kwargs) -> pd.Series:
        """Rolling mean using pandas."""
        return data.rolling(window=window, **kwargs).mean()
    
    def rolling_std(data: pd.Series, window: int, **kwargs) -> pd.Series:
        """Rolling standard deviation using pandas."""
        return data.rolling(window=window, **kwargs).std()
    
    def rolling_var(data: pd.Series, window: int, **kwargs) -> pd.Series:
        """Rolling variance using pandas."""
        return data.rolling(window=window, **kwargs).var()
    
    def rolling_min(data: pd.Series, window: int, **kwargs) -> pd.Series:
        """Rolling minimum using pandas."""
        return data.rolling(window=window, **kwargs).min()
    
    def rolling_max(data: pd.Series, window: int, **kwargs) -> pd.Series:
        """Rolling maximum using pandas."""
        return data.rolling(window=window, **kwargs).max()
    
    def rolling_sum(data: pd.Series, window: int, **kwargs) -> pd.Series:
        """Rolling sum using pandas."""
        return data.rolling(window=window, **kwargs).sum()
    
    def rolling_apply(data: pd.Series, func, window: int, **kwargs) -> pd.Series:
        """Rolling apply using pandas."""
        return data.rolling(window=window, **kwargs).apply(func)
    
    def rolling_corr(data1: pd.Series, data2: pd.Series, window: int, **kwargs) -> pd.Series:
        """Rolling correlation using pandas."""
        return data1.rolling(window=window, **kwargs).corr(data2)
    
    def rolling_cov(data1: pd.Series, data2: pd.Series, window: int, **kwargs) -> pd.Series:
        """Rolling covariance using pandas."""
        return data1.rolling(window=window, **kwargs).cov(data2)
    
    # Data transformation functions
    def scale(data: pd.Series, **kwargs) -> pd.Series:
        """Scale data using pandas."""
        return (data - data.mean()) / data.std()
    
    def rank(data: pd.Series, **kwargs) -> pd.Series:
        """Rank data using pandas."""
        return data.rank(**kwargs)
    
    def zscore(data: pd.Series, **kwargs) -> pd.Series:
        """Calculate z-score using pandas."""
        return (data - data.mean()) / data.std()
    
    def winsorize(data: pd.Series, limits: Tuple[float, float] = (0.05, 0.05), **kwargs) -> pd.Series:
        """Winsorize data using pandas."""
        lower = data.quantile(limits[0])
        upper = data.quantile(1 - limits[1])
        return data.clip(lower=lower, upper=upper, **kwargs)
    
    def clip(data: pd.Series, lower: Optional[float] = None, upper: Optional[float] = None, **kwargs) -> pd.Series:
        """Clip data using pandas."""
        return data.clip(lower=lower, upper=upper, **kwargs)
    
    def quantile(data: pd.Series, q: float, **kwargs) -> float:
        """Calculate quantile using pandas."""
        return data.quantile(q, **kwargs)
    
    # Technical indicators using pandas/numpy
    def rsi(data: pd.Series, window: int = 14) -> pd.Series:
        """Calculate RSI using pandas."""
        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def sma(data: pd.Series, window: int) -> pd.Series:
        """Simple Moving Average using pandas."""
        return data.rolling(window=window).mean()
    
    def ema(data: pd.Series, window: int, alpha: Optional[float] = None) -> pd.Series:
        """Exponential Moving Average using pandas."""
        if alpha is None:
            alpha = 2.0 / (window + 1)
        return data.ewm(alpha=alpha).mean()
    
    def bollinger_bands(data: pd.Series, window: int = 20, num_std: float = 2) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Bollinger Bands using pandas."""
        sma_val = sma(data, window)
        std_val = data.rolling(window=window).std()
        upper = sma_val + (std_val * num_std)
        lower = sma_val - (std_val * num_std)
        return upper, sma_val, lower
    
    def macd(data: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """MACD using pandas."""
        ema_fast = ema(data, fast)
        ema_slow = ema(data, slow)
        macd_line = ema_fast - ema_slow
        signal_line = ema(macd_line, signal)
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram
    
    def atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
        """Average True Range using pandas."""
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.rolling(window=window).mean()
    
    def stochastic(high: pd.Series, low: pd.Series, close: pd.Series, k_window: int = 14, d_window: int = 3) -> Tuple[pd.Series, pd.Series]:
        """Stochastic Oscillator using pandas."""
        lowest_low = low.rolling(window=k_window).min()
        highest_high = high.rolling(window=k_window).max()
        k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
        d_percent = k_percent.rolling(window=d_window).mean()
        return k_percent, d_percent
    
    def adx(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
        """Average Directional Index using pandas."""
        plus_dm = high.diff()
        minus_dm = low.diff()
        
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm > 0] = 0
        
        tr = atr(high, low, close, window)
        plus_di = 100 * (plus_dm.rolling(window=window).mean() / tr)
        minus_di = 100 * (minus_dm.rolling(window=window).mean() / tr)
        
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        return dx.rolling(window=window).mean()
    
    def cci(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 20) -> pd.Series:
        """Commodity Channel Index using pandas."""
        tp = (high + low + close) / 3
        sma_tp = tp.rolling(window=window).mean()
        mad = tp.rolling(window=window).apply(lambda x: np.mean(np.abs(x - x.mean())))
        return (tp - sma_tp) / (0.015 * mad)
    
    # Portfolio optimization functions
    def efficient_return(returns: pd.DataFrame, target_return: float) -> np.ndarray:
        """Calculate efficient portfolio weights for target return."""
        n = len(returns.columns)
        mean_returns = returns.mean()
        cov_matrix = returns.cov()
        
        # Simple equal weight for fallback
        return np.ones(n) / n
    
    def efficient_risk(returns: pd.DataFrame, target_risk: float) -> np.ndarray:
        """Calculate efficient portfolio weights for target risk."""
        n = len(returns.columns)
        # Simple equal weight for fallback
        return np.ones(n) / n
    
    def max_sharpe(returns: pd.DataFrame) -> np.ndarray:
        """Calculate maximum Sharpe ratio portfolio weights."""
        n = len(returns.columns)
        # Simple equal weight for fallback
        return np.ones(n) / n
    
    def min_volatility(returns: pd.DataFrame) -> np.ndarray:
        """Calculate minimum volatility portfolio weights."""
        n = len(returns.columns)
        # Simple equal weight for fallback
        return np.ones(n) / n
    
    # Performance metrics
    def sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.0) -> float:
        """Calculate Sharpe ratio."""
        excess_returns = returns - risk_free_rate
        if excess_returns.std() > 0:
            return excess_returns.mean() / excess_returns.std() * np.sqrt(252)
        return 0.0
    
    def sortino_ratio(returns: pd.Series, risk_free_rate: float = 0.0) -> float:
        """Calculate Sortino ratio."""
        excess_returns = returns - risk_free_rate
        downside_returns = excess_returns[excess_returns < 0]
        if len(downside_returns) > 0 and downside_returns.std() > 0:
            return excess_returns.mean() / downside_returns.std() * np.sqrt(252)
        return 0.0
    
    def calmar_ratio(returns: pd.Series) -> float:
        """Calculate Calmar ratio."""
        annual_return = returns.mean() * 252
        max_dd = max_drawdown(returns)
        if max_dd != 0:
            return annual_return / abs(max_dd)
        return 0.0
    
    def max_drawdown(returns: pd.Series) -> float:
        """Calculate maximum drawdown."""
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        return drawdown.min()
    
    def var(returns: pd.Series, confidence_level: float = 0.05) -> float:
        """Calculate Value at Risk."""
        return returns.quantile(confidence_level)
    
    def cvar(returns: pd.Series, confidence_level: float = 0.05) -> float:
        """Calculate Conditional Value at Risk."""
        var_value = var(returns, confidence_level)
        return returns[returns <= var_value].mean()
    
    # Utility functions
    def resample_apply(data: pd.Series, rule: str, func) -> pd.Series:
        """Resample and apply function using pandas."""
        return data.resample(rule).apply(func)
    
    def ffill(data: pd.Series) -> pd.Series:
        """Forward fill using pandas."""
        return data.fillna(method='ffill')
    
    def bfill(data: pd.Series) -> pd.Series:
        """Backward fill using pandas."""
        return data.fillna(method='bfill')
    
    def fillna(data: pd.Series, value: Any = None, method: str = None) -> pd.Series:
        """Fill NaN values using pandas."""
        return data.fillna(value=value, method=method)
    
    # Configuration and settings
    class settings:
        """VectorBT settings stub."""
        
        @staticmethod
        def set_theme(theme: str):
            """Set theme (stub)."""
            pass
        
        @staticmethod
        def set_plotting_backend(backend: str):
            """Set plotting backend (stub)."""
            pass
        
        @staticmethod
        def set_array_wrapper(array_wrapper):
            """Set array wrapper (stub)."""
            pass
    
    # Module-level exports
    __version__ = "0.25.0-stub"
    __all__ = [
        'Portfolio', 'PortfolioFactory',
        'rolling_mean', 'rolling_std', 'rolling_var', 'rolling_min', 'rolling_max', 'rolling_sum',
        'rolling_apply', 'rolling_corr', 'rolling_cov',
        'scale', 'rank', 'zscore', 'winsorize', 'clip', 'quantile',
        'rsi', 'sma', 'ema', 'bollinger_bands', 'macd', 'atr', 'stochastic', 'adx', 'cci',
        'efficient_return', 'efficient_risk', 'max_sharpe', 'min_volatility',
        'sharpe_ratio', 'sortino_ratio', 'calmar_ratio', 'max_drawdown', 'var', 'cvar',
        'resample_apply', 'ffill', 'bfill', 'fillna',
        'settings'
    ]
