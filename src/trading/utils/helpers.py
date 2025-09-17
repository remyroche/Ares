"""
Trading Helper Functions

Utility functions for common trading operations including
calculations, data processing, and formatting.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Union, Tuple
from datetime import datetime, timedelta
import json

from src.utils.tprint import tprint_info, tprint_success, tprint_structured, LogLevel

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
        
        # Create directory if it doesn't exist
        os.makedirs(directory, exist_ok=True)
        
        # Add timestamp to filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        full_filename = f"{filename}_{timestamp}.json"
        filepath = os.path.join(directory, full_filename)
        
        # Convert any datetime objects to strings
        def convert_datetime(obj):
            if isinstance(obj, datetime):
                return obj.isoformat()
            elif isinstance(obj, dict):
                return {k: convert_datetime(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_datetime(v) for v in obj]
            else:
                return obj
        
        converted_data = convert_datetime(data)
        
        # Save to file
        with open(filepath, 'w') as f:
            json.dump(converted_data, f, indent=2, default=str)
        
        tprint_success(f"✅ Trading data saved to {filepath}")
        return True
        
    except Exception as e:
        tprint_info(f"❌ Failed to save trading data: {e}")
        return False