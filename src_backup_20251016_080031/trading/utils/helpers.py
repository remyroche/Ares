"""Trading helper functions."""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from src.utils.tprint import LogLevel, tprint_info, tprint_structured, tprint_success

# ---------------------------------------------------------------------------
# Core statistical helpers
# ---------------------------------------------------------------------------

def calculate_returns(
    prices: Union[pd.Series, np.ndarray, List[float]],
    method: str = 'simple'
) -> np.ndarray:
    """
    Calculate price returns.
    
    Args:
        prices: Price series
        method: 'simple' or 'log' returns
        
    Returns:
        Array of returns
    """
    if isinstance(prices, list):
        prices = np.array(prices)
    elif isinstance(prices, pd.Series):
        prices = prices.values
    
    if len(prices) < 2:
        return np.array([])
    
    if method == 'log':
        returns = np.diff(np.log(prices))
    else:  # simple returns
        returns = np.diff(prices) / prices[:-1]
    
    return returns

def calculate_volatility(
    returns: Union[pd.Series, np.ndarray, List[float]],
    annualize: bool = True,
    periods_per_year: int = 365
) -> float:
    """
    Calculate volatility from returns.
    
    Args:
        returns: Return series
        annualize: Whether to annualize the volatility
        periods_per_year: Number of periods per year for annualization
        
    Returns:
        Volatility value
    """
    if isinstance(returns, list):
        returns = np.array(returns)
    elif isinstance(returns, pd.Series):
        returns = returns.values
    
    if len(returns) < 2:
        return 0.0
    
    vol = np.std(returns)
    
    if annualize:
        vol *= np.sqrt(periods_per_year)
    
    return float(vol)

def calculate_sharpe_ratio(
    returns: Union[pd.Series, np.ndarray, List[float]],
    risk_free_rate: float = 0.0,
    annualize: bool = True,
    periods_per_year: int = 365
) -> float:
    """
    Calculate Sharpe ratio.
    
    Args:
        returns: Return series
        risk_free_rate: Risk-free rate (annual)
        annualize: Whether to annualize the ratio
        periods_per_year: Number of periods per year
        
    Returns:
        Sharpe ratio
    """
    if isinstance(returns, list):
        returns = np.array(returns)
    elif isinstance(returns, pd.Series):
        returns = returns.values
    
    if len(returns) < 2:
        return 0.0
    
    mean_return = np.mean(returns)
    volatility = np.std(returns)
    
    if volatility == 0:
        return 0.0
    
    # Adjust risk-free rate to period frequency
    period_risk_free_rate = risk_free_rate / periods_per_year if annualize else risk_free_rate
    
    sharpe = (mean_return - period_risk_free_rate) / volatility
    
    if annualize:
        sharpe *= np.sqrt(periods_per_year)
    
    return float(sharpe)

def calculate_max_drawdown(
    prices: Union[pd.Series, np.ndarray, List[float]]
) -> Tuple[float, int, int]:
    """
    Calculate maximum drawdown.
    
    Args:
        prices: Price series
        
    Returns:
        Tuple of (max_drawdown, start_index, end_index)
    """
    if isinstance(prices, list):
        prices = np.array(prices)
    elif isinstance(prices, pd.Series):
        prices = prices.values
    
    if len(prices) < 2:
        return 0.0, 0, 0
    
    # Calculate running maximum
    peak = np.maximum.accumulate(prices)
    
    # Calculate drawdown
    drawdown = (prices - peak) / peak
    
    # Find maximum drawdown
    max_dd_idx = np.argmin(drawdown)
    max_drawdown = drawdown[max_dd_idx]
    
    # Find the peak before the maximum drawdown
    peak_idx = np.argmax(peak[:max_dd_idx + 1])
    
    return float(abs(max_drawdown)), int(peak_idx), int(max_dd_idx)

# ---------------------------------------------------------------------------
# Volatility-aware trailing helpers
# ---------------------------------------------------------------------------

def _ensure_datetime_index(data: pd.DataFrame) -> pd.DataFrame:
    """Ensure a DataFrame is indexed by datetime for resampling."""

    if data.empty:
        return data

    df = data.copy()

    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df = df.set_index("timestamp")
    elif not isinstance(df.index, pd.DatetimeIndex):
        # Assume consecutive bars spaced 1 minute apart when no timestamp exists
        end_time = datetime.utcnow()
        index = pd.date_range(end=end_time, periods=len(df), freq="1T")
        df.index = index

    return df.sort_index()

def _resample_ohlcv(data: pd.DataFrame, rule: str) -> pd.DataFrame:
    """Resample OHLCV data to a new timeframe."""

    if data.empty:
        return data

    agg = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }
    resampled = data.resample(rule).agg(agg).dropna()
    return resampled

def compute_atr(data: pd.DataFrame, window: int = 14) -> Tuple[pd.Series, float]:
    """Compute the Average True Range and return the full series and latest value."""

    if data.empty or len(data) < max(window, 2):
        return pd.Series(dtype=float), 0.0

    high = data["high"]
    low = data["low"]
    close = data["close"]
    prev_close = close.shift(1)

    true_range = pd.concat(
        [
            (high - low).abs(),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    atr_series = true_range.rolling(window=window, min_periods=window).mean()
    latest_atr = float(atr_series.iloc[-1]) if not atr_series.dropna().empty else 0.0
    return atr_series, latest_atr

def compute_realized_volatility(data: pd.DataFrame, window: int = 20) -> Tuple[pd.Series, float]:
    """Compute realized volatility using log returns."""

    if data.empty or len(data) < window + 1:
        return pd.Series(dtype=float), 0.0

    close = data["close"]
    log_returns = np.log(close / close.shift(1)).dropna()
    vol_series = (
        log_returns.rolling(window=window, min_periods=window).std() * np.sqrt(window)
    )
    latest_vol = float(vol_series.iloc[-1]) if not vol_series.dropna().empty else 0.0
    return vol_series, latest_vol

def compute_momentum(data: pd.DataFrame, window: int = 3) -> float:
    """Compute simple momentum over a rolling window."""

    if data.empty or len(data) <= window:
        return 0.0

    close = data["close"]
    momentum = float(close.iloc[-1] - close.iloc[-window - 1])
    return momentum

def compute_rsi(data: pd.DataFrame, window: int = 3) -> float:
    """Compute a short-term RSI."""

    if data.empty or len(data) < window + 1:
        return 50.0

    close = data["close"]
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(window=window, min_periods=window).mean()
    avg_loss = loss.rolling(window=window, min_periods=window).mean()

    if avg_loss.iloc[-1] == 0:
        return 100.0

    rs = avg_gain.iloc[-1] / avg_loss.iloc[-1]
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return float(rsi)

def compute_volatility_slope(series: pd.Series, lookback: int = 5) -> float:
    """Compute a simple slope for a volatility series."""

    if series.empty or len(series.dropna()) < lookback:
        return 0.0

    recent = series.dropna().iloc[-lookback:]
    slope = float(recent.iloc[-1] - recent.iloc[0])
    return slope

@dataclass
class TrailingFeatureBundle:
    """Container for trailing management metrics."""

    timestamp: datetime
    current_price: float
    bar_seconds: int
    tactician: Dict[str, float]
    analyst: Dict[str, float]

def prepare_trailing_feature_bundle(
    market_data: pd.DataFrame,
    tactician_rule: str = "15T",
    analyst_rule: str = "1H",
) -> Optional[TrailingFeatureBundle]:
    """Prepare volatility and momentum features for trailing management."""

    if market_data is None or market_data.empty:
        return None

    ohlcv = market_data[[col for col in ["open", "high", "low", "close", "volume"] if col in market_data.columns]].copy()
    if ohlcv.empty:
        return None

    base = _ensure_datetime_index(ohlcv)
    if base.empty:
        return None

    bar_seconds = 60
    if len(base.index) > 1:
        delta = base.index[-1] - base.index[-2]
        bar_seconds = int(max(delta.total_seconds(), 1))

    tactician_df = _resample_ohlcv(base, tactician_rule)
    analyst_df = _resample_ohlcv(base, analyst_rule)

    tact_atr_series, tact_atr = compute_atr(tactician_df)
    tact_vol_series, tact_sigma = compute_realized_volatility(tactician_df)
    tact_momentum = compute_momentum(tactician_df)
    tact_rsi = compute_rsi(tactician_df)
    tact_vol_slope = compute_volatility_slope(tact_vol_series)

    analyst_atr_series, analyst_atr = compute_atr(analyst_df)
    analyst_vol_series, analyst_sigma = compute_realized_volatility(analyst_df)
    analyst_momentum = compute_momentum(analyst_df, window=4)
    analyst_rsi = compute_rsi(analyst_df, window=6)
    analyst_vol_slope = compute_volatility_slope(analyst_vol_series)

    timestamp = base.index[-1].to_pydatetime()
    current_price = float(base["close"].iloc[-1])

    tactician_features = {
        "atr": tact_atr,
        "sigma": tact_sigma,
        "momentum": tact_momentum,
        "rsi": tact_rsi,
        "vol_slope": tact_vol_slope,
    }

    analyst_features = {
        "atr": analyst_atr,
        "sigma": analyst_sigma,
        "momentum": analyst_momentum,
        "rsi": analyst_rsi,
        "vol_slope": analyst_vol_slope,
    }

    return TrailingFeatureBundle(
        timestamp=timestamp,
        current_price=current_price,
        bar_seconds=bar_seconds,
        tactician=tactician_features,
        analyst=analyst_features,
    )

def normalize_price_data(
    data: pd.DataFrame,
    method: str = 'minmax',
    columns: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Normalize price data.
    
    Args:
        data: Price data DataFrame
        method: 'minmax', 'zscore', or 'robust'
        columns: Columns to normalize (default: all numeric columns)
        
    Returns:
        Normalized DataFrame
    """
    result = data.copy()
    
    if columns is None:
        columns = result.select_dtypes(include=[np.number]).columns.tolist()
    
    for col in columns:
        if col in result.columns:
            if method == 'minmax':
                min_val = result[col].min()
                max_val = result[col].max()
                if max_val != min_val:
                    result[col] = (result[col] - min_val) / (max_val - min_val)
            
            elif method == 'zscore':
                mean_val = result[col].mean()
                std_val = result[col].std()
                if std_val != 0:
                    result[col] = (result[col] - mean_val) / std_val
            
            elif method == 'robust':
                median_val = result[col].median()
                mad = np.median(np.abs(result[col] - median_val))
                if mad != 0:
                    result[col] = (result[col] - median_val) / mad
    
    return result

def calculate_technical_indicators(
    data: pd.DataFrame,
    indicators: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Calculate common technical indicators.
    
    Args:
        data: OHLCV data
        indicators: List of indicators to calculate
        
    Returns:
        DataFrame with indicators added
    """
    result = data.copy()
    
    if indicators is None:
        indicators = ['sma_20', 'ema_12', 'rsi', 'macd', 'bollinger']
    
    # Simple Moving Average
    if 'sma_20' in indicators and 'close' in result.columns:
        result['sma_20'] = result['close'].rolling(window=20).mean()
    
    # Exponential Moving Average
    if 'ema_12' in indicators and 'close' in result.columns:
        result['ema_12'] = result['close'].ewm(span=12).mean()
    
    # RSI
    if 'rsi' in indicators and 'close' in result.columns:
        delta = result['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        result['rsi'] = 100 - (100 / (1 + rs))
    
    # MACD
    if 'macd' in indicators and 'close' in result.columns:
        ema_12 = result['close'].ewm(span=12).mean()
        ema_26 = result['close'].ewm(span=26).mean()
        result['macd'] = ema_12 - ema_26
        result['macd_signal'] = result['macd'].ewm(span=9).mean()
        result['macd_histogram'] = result['macd'] - result['macd_signal']
    
    # Bollinger Bands
    if 'bollinger' in indicators and 'close' in result.columns:
        sma_20 = result['close'].rolling(window=20).mean()
        std_20 = result['close'].rolling(window=20).std()
        result['bb_upper'] = sma_20 + (std_20 * 2)
        result['bb_lower'] = sma_20 - (std_20 * 2)
        result['bb_middle'] = sma_20
        result['bb_width'] = (result['bb_upper'] - result['bb_lower']) / result['bb_middle']
        result['bb_position'] = (result['close'] - result['bb_lower']) / (result['bb_upper'] - result['bb_lower'])
    
    return result

def format_trading_metrics(
    metrics: Dict[str, Any],
    precision: int = 4
) -> Dict[str, str]:
    """
    Format trading metrics for display.
    
    Args:
        metrics: Dictionary of metrics
        precision: Number of decimal places
        
    Returns:
        Dictionary of formatted metrics
    """
    formatted = {}
    
    for key, value in metrics.items():
        if isinstance(value, float):
            if key in ['return', 'pnl', 'profit', 'loss']:
                # Format as percentage
                formatted[key] = f"{value * 100:.{precision-2}f}%"
            elif key in ['price', 'balance', 'value', 'amount']:
                # Format as currency
                formatted[key] = f"${value:,.{precision-2}f}"
            elif key in ['ratio', 'factor', 'multiplier']:
                # Format as ratio
                formatted[key] = f"{value:.{precision}f}x"
            elif key in ['probability', 'confidence', 'score']:
                # Format as percentage
                formatted[key] = f"{value * 100:.{precision-2}f}%"
            else:
                # Default float formatting
                formatted[key] = f"{value:.{precision}f}"
        elif isinstance(value, int):
            formatted[key] = f"{value:,}"
        elif isinstance(value, datetime):
            formatted[key] = value.strftime("%Y-%m-%d %H:%M:%S")
        else:
            formatted[key] = str(value)
    
    return formatted

def calculate_position_metrics(
    entry_price: float,
    current_price: float,
    quantity: float,
    side: str = 'long'
) -> Dict[str, float]:
    """
    Calculate position metrics.
    
    Args:
        entry_price: Entry price
        current_price: Current price
        quantity: Position quantity
        side: 'long' or 'short'
        
    Returns:
        Dictionary of position metrics
    """
    if side.lower() == 'long':
        unrealized_pnl = (current_price - entry_price) * quantity
        return_pct = (current_price - entry_price) / entry_price
    else:  # short
        unrealized_pnl = (entry_price - current_price) * quantity
        return_pct = (entry_price - current_price) / entry_price
    
    market_value = current_price * quantity
    cost_basis = entry_price * quantity
    
    return {
        'unrealized_pnl': unrealized_pnl,
        'return_pct': return_pct,
        'market_value': market_value,
        'cost_basis': cost_basis,
        'quantity': quantity,
        'entry_price': entry_price,
        'current_price': current_price
    }

def create_trading_summary(
    trades: List[Dict[str, Any]],
    account_balance: float = 10000.0
) -> Dict[str, Any]:
    """
    Create trading performance summary.
    
    Args:
        trades: List of trade dictionaries
        account_balance: Starting account balance
        
    Returns:
        Trading summary dictionary
    """
    if not trades:
        return {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'win_rate': 0.0,
            'total_pnl': 0.0,
            'total_return': 0.0,
            'avg_win': 0.0,
            'avg_loss': 0.0,
            'profit_factor': 0.0,
            'max_drawdown': 0.0,
            'sharpe_ratio': 0.0
        }
    
    # Calculate basic metrics
    total_trades = len(trades)
    pnls = []
    
    for trade in trades:
        if 'pnl' in trade:
            pnls.append(trade['pnl'])
        elif 'return' in trade:
            pnls.append(trade['return'] * account_balance)
    
    if not pnls:
        return create_trading_summary([], account_balance)
    
    winning_trades = sum(1 for pnl in pnls if pnl > 0)
    losing_trades = sum(1 for pnl in pnls if pnl < 0)
    
    win_rate = winning_trades / total_trades if total_trades > 0 else 0.0
    total_pnl = sum(pnls)
    total_return = total_pnl / account_balance
    
    # Calculate win/loss averages
    wins = [pnl for pnl in pnls if pnl > 0]
    losses = [pnl for pnl in pnls if pnl < 0]
    
    avg_win = np.mean(wins) if wins else 0.0
    avg_loss = abs(np.mean(losses)) if losses else 0.0
    
    profit_factor = (avg_win * winning_trades) / (avg_loss * losing_trades) if avg_loss > 0 and losing_trades > 0 else 0.0
    
    # Calculate equity curve and drawdown
    equity_curve = np.cumsum([account_balance] + pnls)
    max_drawdown_pct, _, _ = calculate_max_drawdown(equity_curve)
    
    # Calculate Sharpe ratio
    if len(pnls) > 1:
        returns = np.array(pnls) / account_balance
        sharpe_ratio = calculate_sharpe_ratio(returns)
    else:
        sharpe_ratio = 0.0
    
    summary = {
        'total_trades': total_trades,
        'winning_trades': winning_trades,
        'losing_trades': losing_trades,
        'win_rate': win_rate,
        'total_pnl': total_pnl,
        'total_return': total_return,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'profit_factor': profit_factor,
        'max_drawdown': max_drawdown_pct,
        'sharpe_ratio': sharpe_ratio,
        'final_balance': account_balance + total_pnl
    }
    
    return summary

def log_trading_summary(summary: Dict[str, Any], title: str = "Trading Summary"):
    """
    Log formatted trading summary.
    
    Args:
        summary: Trading summary dictionary
        title: Title for the summary
    """
    formatted_summary = format_trading_metrics(summary)
    
    tprint_info(f"📊 {title}")
    tprint_structured(formatted_summary, LogLevel.INFO)
    
    # Highlight key metrics
    if summary.get('total_trades', 0) > 0:
        win_rate = summary.get('win_rate', 0) * 100
        total_return = summary.get('total_return', 0) * 100
        
        if win_rate >= 60 and total_return > 0:
            tprint_success(f"✅ Strong performance: {win_rate:.1f}% win rate, {total_return:.2f}% return")
        elif total_return > 0:
            tprint_info(f"📈 Positive return: {total_return:.2f}%")
        else:
            tprint_info(f"📉 Negative return: {total_return:.2f}%")

def save_trading_data(
    data: Dict[str, Any],
    filename: str,
    directory: str = "data_cache/trading"
) -> bool:
    """
    Save trading data to JSON file.
    
    Args:
        data: Data to save
        filename: Filename (without extension)
        directory: Directory to save to
        
    Returns:
        bool: True if successful
    """
    try:
        import os
        return os.path.exists(file_path)
    except Exception as e:
        logger.error(f"Error checking file existence: {e}")
        return False

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

def calculate_atr14(data: pd.DataFrame, period: int = 14) -> pd.Series:
    """Calculate the Average True Range (ATR) for OHLCV input data."""

    required_columns = {"high", "low", "close"}
    if not required_columns.issubset(data.columns):
        missing = ", ".join(sorted(required_columns - set(data.columns)))
        raise ValueError(f"Missing required columns for ATR calculation: {missing}")

    high = pd.to_numeric(data["high"], errors="coerce")
    low = pd.to_numeric(data["low"], errors="coerce")
    close = pd.to_numeric(data["close"], errors="coerce")

    previous_close = close.shift(1)
    tr_components = pd.concat(
        [
            high - low,
            (high - previous_close).abs(),
            (low - previous_close).abs(),
        ],
        axis=1,
    )
    true_range = tr_components.max(axis=1)
    atr = true_range.rolling(window=period, min_periods=period).mean()
    return atr

def calculate_realized_volatility(data: pd.DataFrame, window: int = 20) -> pd.Series:
    """Compute realized volatility using log returns over the supplied window."""

    if "close" not in data.columns:
        raise ValueError("Missing required column 'close' for realized volatility calculation")

    close = pd.to_numeric(data["close"], errors="coerce")
    close = close.replace(0, np.nan)
    log_returns = np.log(close / close.shift(1))
    realized_vol = log_returns.rolling(window=window, min_periods=window).std()
    realized_vol *= np.sqrt(window)
    return realized_vol

def calculate_three_bar_momentum(data: pd.DataFrame) -> pd.Series:
    """Calculate momentum between the latest close and the close three bars prior."""

    if "close" not in data.columns:
        raise ValueError("Missing required column 'close' for momentum calculation")

    close = pd.to_numeric(data["close"], errors="coerce")
    momentum = close - close.shift(3)
    return momentum

def calculate_three_bar_rsi(data: pd.DataFrame, period: int = 3) -> pd.Series:
    """Calculate a fast RSI value over a three-period window."""

    if "close" not in data.columns:
        raise ValueError("Missing required column 'close' for RSI calculation")

    close = pd.to_numeric(data["close"], errors="coerce")
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()

    rs = avg_gain / avg_loss.replace(to_replace=0, value=np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_volatility_slope(
    data: pd.DataFrame,
    volatility_window: int = 20,
    slope_window: int = 5,
) -> pd.Series:
    """Compute the slope of realized volatility to gauge acceleration or deceleration."""

    volatility = calculate_realized_volatility(data, window=volatility_window)

    def _slope(values: np.ndarray) -> float:
        idx = np.arange(len(values))
        mask = ~np.isnan(values)
        if mask.sum() < 2:
            return np.nan
        x = idx[mask]
        y = values[mask]
        slope, _ = np.polyfit(x, y, 1)
        return slope

    slope_series = volatility.rolling(window=slope_window, min_periods=2).apply(_slope, raw=True)
    return slope_series

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
