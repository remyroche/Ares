"""Ridge-based position sizer combining meta model outputs.

This module sits between 'train_meta' and 'optimise' steps in the pipeline.
It takes OOF predictions from meta_model.py and learns optimal combination weights
using Huber loss with asymmetric sample weighting (losing trades weighted more).

Key Features:
- Per-trade label computation with transaction costs
- L2-regularized constrained linear combiner with Huber loss
- Asymmetric sample weights: losing trades get 2-4x weight
- Hyperparameter selection via composite J z-score
- Output aligned with tpsl_optimiser pipeline
- Policy-aware label computation using TP/SL/trailing simulator

Note: This is NOT standard Ridge regression. It's a constrained linear combiner
solved via SLSQP optimization with Huber loss + L2 penalty.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import rankdata, spearmanr

try:
    from numba import jit, prange
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    # Fallback: create a no-op decorator
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        if len(args) == 1 and callable(args[0]):
            return args[0]
        return decorator
    prange = range

from extreme_price_movements.utils import tprint
from extreme_price_movements.tpsl_optimiser.metrics_utils import compute_comprehensive_metrics


# ═══════════════════════════════════════════════════════════════════════════════
# Exit Reason Enumeration
# ═══════════════════════════════════════════════════════════════════════════════

class ExitReason(Enum):
    """Enumeration of possible trade exit reasons.
    
    Used to track why a trade was closed, which is important for:
    - Understanding strategy behavior
    - Computing accurate labels for training
    - Analyzing exit patterns
    """
    TP_HIT = "tp_hit"           # Take-profit triggered
    SL_HIT = "sl_hit"           # Stop-loss triggered
    TRAILING_EXIT = "trailing"  # Trailing stop captured profit
    TIMEOUT = "timeout"         # Max holding period reached
    # Note: LIQUIDATION removed - was never produced by simulator


# ═══════════════════════════════════════════════════════════════════════════════
# Numba-Optimized Trade Exit Simulator
# ═══════════════════════════════════════════════════════════════════════════════

@jit(nopython=True, cache=True)
def simulate_trade_exit(
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    entry_price: float,
    is_long: bool,
    tp_price: float,
    sl_price: float,
    trailing_pct: float,
    max_bars: int,
) -> Tuple[float, int, int]:
    """Simulate trade exit using TP/SL/trailing rules.
    
    This is a Numba-optimized simulator that processes price bars sequentially
    to determine the actual exit price based on the trading policy.
    
    The order of checking is important:
    1. Update peak/trough for trailing calculations
    2. Check TP hit (take-profit)
    3. Check SL hit (stop-loss)
    4. Check trailing exit (if profit has been made)
    
    TODO: The trailing exit logic uses a simplified `peak * (1 - trailing_pct)` 
    formula. Consider integrating the full tpsl_optimiser policy which has
    more sophisticated parameters (act_n, be_act_n, d_min, d_max) for 
    activation threshold and break-even triggers.
    
    Args:
        highs: Array of future high prices (length >= max_bars)
        lows: Array of future low prices (length >= max_bars)
        closes: Array of future close prices (length >= max_bars)
        entry_price: Entry price of the trade
        is_long: True for long position, False for short
        tp_price: Take-profit price level
        sl_price: Stop-loss price level
        trailing_pct: Trailing percentage (e.g., 0.5 = 50% of peak retracement)
        max_bars: Maximum number of bars to hold before timeout
    
    Returns:
        Tuple of (exit_price, exit_bar, exit_reason):
        - exit_price: The price at which the trade exited
        - exit_bar: The bar index (0-based) when exit occurred
        - exit_reason: 0=TP, 1=SL, 2=Trailing, 3=Timeout
    """
    peak = entry_price
    trough = entry_price
    
    for bar in range(max_bars):
        h = highs[bar]
        l = lows[bar]
        c = closes[bar]
        
        # Check for NaN (synthetic padded data) - force timeout
        if np.isnan(h) or np.isnan(l) or np.isnan(c):
            # Return at the last valid close
            for prev_bar in range(bar - 1, -1, -1):
                if not np.isnan(closes[prev_bar]):
                    return closes[prev_bar], prev_bar, 3
            return entry_price, 0, 3  # No valid data
        
        if is_long:
            # Update peak for trailing calculation
            if h > peak:
                peak = h
            
            # Check if both TP and SL are hit in this bar
            tp_hit = h >= tp_price
            sl_hit = l <= sl_price
            
            if tp_hit and sl_hit:
                # Both hit - choose the one closest to entry (happened first)
                # For a long: if TP is closer to entry than SL, TP happened first
                tp_distance = tp_price - entry_price
                sl_distance = entry_price - sl_price
                if tp_distance <= sl_distance:
                    return tp_price, bar, 0
                else:
                    return sl_price, bar, 1
            elif tp_hit:
                return tp_price, bar, 0
            elif sl_hit:
                return sl_price, bar, 1
            
            # Check trailing exit (only if we have profit)
            if peak > entry_price:
                trailing_price = peak * (1.0 - trailing_pct)
                if l <= trailing_price:
                    return trailing_price, bar, 2
        else:
            # Short position logic
            # Update trough for trailing calculation
            if l < trough:
                trough = l
            
            # Check if both TP and SL are hit in this bar
            tp_hit = l <= tp_price
            sl_hit = h >= sl_price
            
            if tp_hit and sl_hit:
                # Both hit - choose the one closest to entry (happened first)
                # For a short: if TP is closer to entry than SL, TP happened first
                tp_distance = entry_price - tp_price
                sl_distance = sl_price - entry_price
                if tp_distance <= sl_distance:
                    return tp_price, bar, 0
                else:
                    return sl_price, bar, 1
            elif tp_hit:
                return tp_price, bar, 0
            elif sl_hit:
                return sl_price, bar, 1
            
            # Check trailing exit (only if we have profit)
            if trough < entry_price:
                trailing_price = trough * (1.0 + trailing_pct)
                if h >= trailing_price:
                    return trailing_price, bar, 2
    
    # Timeout - exit at close of last bar
    return closes[max_bars - 1], max_bars - 1, 3


@jit(nopython=True, parallel=True, cache=True)
def simulate_trade_exit_batch(
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    entry_prices: np.ndarray,
    is_longs: np.ndarray,
    tp_prices: np.ndarray,
    sl_prices: np.ndarray,
    trailing_pcts: np.ndarray,
    max_bars: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Batch version of simulate_trade_exit for parallel processing.
    
    Processes multiple trades in parallel using Numba's prange.
    
    Args:
        highs: 2D array of shape (n_trades, max_bars) with future high prices
        lows: 2D array of shape (n_trades, max_bars) with future low prices
        closes: 2D array of shape (n_trades, max_bars) with future close prices
        entry_prices: Array of entry prices for each trade
        is_longs: Binary array (1 for long, 0 for short) for each trade
        tp_prices: Array of take-profit prices for each trade
        sl_prices: Array of stop-loss prices for each trade
        trailing_pcts: Array of trailing percentages for each trade
        max_bars: Maximum holding period in bars
    
    Returns:
        Tuple of (exit_prices, exit_bars, exit_reasons) arrays
    """
    n_trades = len(entry_prices)
    exit_prices = np.empty(n_trades, dtype=np.float64)
    exit_bars = np.empty(n_trades, dtype=np.int64)
    exit_reasons = np.empty(n_trades, dtype=np.int64)
    
    for i in prange(n_trades):
        exit_prices[i], exit_bars[i], exit_reasons[i] = simulate_trade_exit(
            highs[i], lows[i], closes[i],
            entry_prices[i],
            bool(is_longs[i]),
            tp_prices[i],
            sl_prices[i],
            trailing_pcts[i],
            max_bars,
        )
    
    return exit_prices, exit_bars, exit_reasons


# ═══════════════════════════════════════════════════════════════════════════════
# Robust Z-Score Functions
# ═══════════════════════════════════════════════════════════════════════════════

def robust_z(x: pd.Series, eps: float = 1e-12) -> pd.Series:
    """Robust z-score using median and MAD.
    
    More resistant to outliers than standard z-score.
    MAD is scaled by 1.4826 to be consistent with standard deviation
    for normally distributed data.
    
    Args:
        x: Input series to compute z-scores for
        eps: Small constant to prevent division by zero
        
    Returns:
        Series of robust z-scores, zeros if scale cannot be determined
    """
    x = x.astype(float)
    med = x.median()
    mad = (x - med).abs().median()
    scale = 1.4826 * mad  # ~ std if normal
    
    if not np.isfinite(scale) or scale < eps:
        std = x.std(ddof=0)
        if not np.isfinite(std) or std < eps:
            return pd.Series(np.zeros(len(x)), index=x.index)
        return (x - x.mean()) / (std + eps)
    
    return (x - med) / (scale + eps)


def composite_J_zscore(
    df: pd.DataFrame,
    pnl_col: str = "PnL_ann",
    sortino_col: str = "Sortino",
    maxdd_col: str = "MaxDD",
    a: float = 1.0,
    b: float = 1.0,
    group_col: str | None = None,
    use_robust: bool = True,
) -> pd.Series:
    """Compute a scale-stable composite objective for hyperparameter selection.
    
    The composite J score balances return, risk-adjusted return, and drawdown:
        J = z(PnL_ann) + a * z(Sortino) - b * z(MaxDD)
    
    Higher J is better. The z-scores make different metrics comparable
    regardless of their native scales.
    
    Args:
        df: DataFrame with metric columns
        pnl_col: Column name for annualized PnL
        sortino_col: Column name for Sortino ratio
        maxdd_col: Column name for maximum drawdown
        a: Weight for Sortino component (default 1.0)
        b: Weight for MaxDD penalty (default 1.0)
        group_col: Optional column to group by before computing z-scores
        use_robust: If True, use robust z-scores (median/MAD), else standard
        
    Returns:
        Series of composite J z-scores
    """
    z = robust_z if use_robust else (lambda x: (x - x.mean()) / (x.std(ddof=0) + 1e-12))

    def _calc(g: pd.DataFrame) -> pd.Series:
        z_pnl = z(g[pnl_col])
        z_sort = z(g[sortino_col])
        z_dd = z(g[maxdd_col])
        return z_pnl + a * z_sort - b * z_dd

    if group_col is None:
        return _calc(df)

    return df.groupby(group_col, group_keys=False).apply(_calc)


# ═══════════════════════════════════════════════════════════════════════════════
# Per-Trade Label Computation
# ═══════════════════════════════════════════════════════════════════════════════

def compute_trade_labels(
    entry_prices: np.ndarray,
    exit_prices: np.ndarray,
    is_long: np.ndarray,
    cost_pct: float = 0.0005,
) -> np.ndarray:
    """Compute per-trade log-return labels net of transaction costs.
    
    Uses exact log-return formula for numerical stability:
        long:  yi = log(exit/entry) - cost_pct
        short: yi = log(entry/exit) - cost_pct
    
    This is symmetric and exact, unlike approximations based on
    simple returns converted to log.
    
    Args:
        entry_prices: Array of entry prices
        exit_prices: Array of exit prices
        is_long: Binary array (1 for long, 0 for short)
        cost_pct: Transaction cost as decimal (default 0.05%)
        
    Returns:
        Array of log-return labels
    """
    entry_prices = np.asarray(entry_prices, dtype=float)
    exit_prices = np.asarray(exit_prices, dtype=float)
    is_long = np.asarray(is_long, dtype=float)
    
    # Compute log returns directly for numerical stability
    # Long: log(exit/entry), Short: log(entry/exit)
    log_returns = np.where(
        is_long == 1,
        np.log(exit_prices / entry_prices),
        np.log(entry_prices / exit_prices)
    )
    
    # Handle edge cases (zero/negative prices)
    log_returns = np.nan_to_num(log_returns, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Subtract transaction costs
    yi = log_returns - cost_pct
    
    return yi.astype(np.float64)


# ═══════════════════════════════════════════════════════════════════════════════
# Policy-Aware Label Computation
# ═══════════════════════════════════════════════════════════════════════════════

def compute_policy_aware_labels(
    candidates_df: pd.DataFrame,
    price_panel: Dict[str, pd.DataFrame],
    policy_params: Dict,
    max_hold_hours: int = 24,
    cost_pct: float = 0.0005,
    bars_per_hour: int = 4,
) -> pd.DataFrame:
    """Compute policy-aware per-trade labels using exact TP/SL/trailing simulator.
    
    For each candidate trade:
    1. Entry at next hour open (t+1)
    2. Run simulator forward with TP/SL/trailing rules
    3. Determine exit price and time
    4. Compute realized return: log(exit/entry) - costs
    
    This function addresses the mismatch between training labels and actual
    trading behavior by simulating the exact exit logic used in production.
    
    Args:
        candidates_df: DataFrame with columns [timestamp, symbol, is_long, entry_price]
                      - timestamp: Decision time (trade would enter at next bar open)
                      - symbol: Trading pair identifier
                      - is_long: Boolean/0-1 for direction
                      - entry_price: Expected entry price (typically next bar open)
        price_panel: dict with 'open', 'high', 'low', 'close' DataFrames
                    - Each DataFrame indexed by time, columns by symbol
                    - Must cover the period from candidates_df.timestamp to
                      timestamp + max_hold_hours
        policy_params: dict with TP/SL params:
            - tp_mult: Take-profit multiplier (e.g., 2.0 = 2x ATR)
            - sl_mult: Stop-loss multiplier
            - trailing_pct: Trailing exit percentage (e.g., 0.5 = 50% of peak)
            - atr: ATR values per symbol (dict or Series)
        max_hold_hours: Maximum holding period before timeout exit
        cost_pct: Round-trip cost percentage
        bars_per_hour: Number of price bars per hour (4 for 15-min bars)
    
    Returns:
        DataFrame with columns:
        - timestamp: Original decision timestamp
        - symbol: Trading pair
        - is_long: Direction
        - entry_price: Entry price used
        - exit_price: Simulated exit price
        - exit_time: Timestamp of exit
        - exit_reason: ExitReason enum value
        - label: Log return net of costs
        - exit_bar: Bar index when exit occurred
    """
    # Extract policy parameters
    tp_mult = policy_params.get('tp_mult', 3.0)
    sl_mult = policy_params.get('sl_mult', 1.0)
    trailing_pct = policy_params.get('trailing_pct', 0.5)
    atr_dict = policy_params.get('atr', {})
    
    # Calculate max bars
    max_bars = max_hold_hours * bars_per_hour
    
    # Get price data from panel
    opens = price_panel.get('open')
    highs = price_panel.get('high')
    lows = price_panel.get('low')
    closes = price_panel.get('close')
    
    if opens is None or highs is None or lows is None or closes is None:
        raise ValueError("price_panel must contain 'open', 'high', 'low', 'close' DataFrames")
    
    # Ensure timestamps are datetime
    candidates_df = candidates_df.copy()
    if not pd.api.types.is_datetime64_any_dtype(candidates_df['timestamp']):
        candidates_df['timestamp'] = pd.to_datetime(candidates_df['timestamp'])
    
    # Results storage
    results = []
    
    # Process each candidate trade
    for idx, row in candidates_df.iterrows():
        ts = row['timestamp']
        symbol = row['symbol']
        is_long = bool(row['is_long'])
        entry_price = row['entry_price']
        
        # Get ATR for this symbol
        if isinstance(atr_dict, dict):
            atr = atr_dict.get(symbol, 0.02)  # Default 2% ATR
        elif isinstance(atr_dict, pd.Series):
            atr = atr_dict.get(symbol, 0.02)
        else:
            atr = 0.02
        
        # Calculate TP/SL prices
        if is_long:
            tp_price = entry_price * (1 + tp_mult * atr)
            sl_price = entry_price * (1 - sl_mult * atr)
        else:
            tp_price = entry_price * (1 - tp_mult * atr)
            sl_price = entry_price * (1 + sl_mult * atr)
        
        # Get future price data for this symbol
        try:
            # Find the index of the entry timestamp
            if ts not in opens.index:
                # Find nearest future timestamp
                future_mask = opens.index >= ts
                if not future_mask.any():
                    # No future data available
                    continue
                entry_idx = future_mask.argmax()
            else:
                entry_idx = opens.index.get_loc(ts)
            
            # Extract future price arrays
            end_idx = min(entry_idx + max_bars, len(opens) - 1)
            
            if entry_idx >= end_idx:
                continue
            
            # Get arrays for this trade
            future_highs = highs[symbol].iloc[entry_idx:end_idx].values.astype(np.float64)
            future_lows = lows[symbol].iloc[entry_idx:end_idx].values.astype(np.float64)
            future_closes = closes[symbol].iloc[entry_idx:end_idx].values.astype(np.float64)
            
            actual_bars = len(future_highs)
            if actual_bars == 0:
                continue
            
            # Run simulator
            exit_price, exit_bar, exit_reason_int = simulate_trade_exit(
                future_highs, future_lows, future_closes,
                float(entry_price),
                is_long,
                float(tp_price),
                float(sl_price),
                float(trailing_pct),
                min(max_bars, actual_bars),
            )
            
            # Determine exit timestamp
            exit_time = opens.index[entry_idx + exit_bar]
            
            # Map exit reason int to enum
            exit_reason_map = {
                0: ExitReason.TP_HIT,
                1: ExitReason.SL_HIT,
                2: ExitReason.TRAILING_EXIT,
                3: ExitReason.TIMEOUT,
            }
            exit_reason = exit_reason_map.get(exit_reason_int, ExitReason.TIMEOUT)
            
            # Compute label (log return net of costs)
            if is_long:
                label = np.log(exit_price / entry_price) - cost_pct
            else:
                label = np.log(entry_price / exit_price) - cost_pct
            
            # Handle edge cases
            if not np.isfinite(label):
                label = 0.0
            
            results.append({
                'timestamp': ts,
                'symbol': symbol,
                'is_long': is_long,
                'entry_price': entry_price,
                'exit_price': exit_price,
                'exit_time': exit_time,
                'exit_reason': exit_reason,
                'label': label,
                'exit_bar': exit_bar,
                'tp_price': tp_price,
                'sl_price': sl_price,
            })
            
        except (KeyError, IndexError) as e:
            # Skip trades with missing price data
            continue
    
    return pd.DataFrame(results)


def compute_policy_aware_labels_batch(
    candidates_df: pd.DataFrame,
    price_panel: Dict[str, pd.DataFrame],
    policy_params: Dict,
    max_hold_hours: int = 24,
    cost_pct: float = 0.0005,
    bars_per_hour: int = 4,
) -> pd.DataFrame:
    """Batch version of compute_policy_aware_labels for better performance.
    
    Uses the parallel Numba simulator for improved throughput on large datasets.
    
    Args:
        Same as compute_policy_aware_labels
    
    Returns:
        Same as compute_policy_aware_labels
    """
    # Extract policy parameters
    tp_mult = policy_params.get('tp_mult', 3.0)
    sl_mult = policy_params.get('sl_mult', 1.0)
    trailing_pct = policy_params.get('trailing_pct', 0.5)
    atr_dict = policy_params.get('atr', {})
    
    # Calculate max bars
    max_bars = max_hold_hours * bars_per_hour
    
    # Get price data from panel
    opens = price_panel.get('open')
    highs = price_panel.get('high')
    lows = price_panel.get('low')
    closes = price_panel.get('close')
    
    if opens is None or highs is None or lows is None or closes is None:
        raise ValueError("price_panel must contain 'open', 'high', 'low', 'close' DataFrames")
    
    # Ensure timestamps are datetime
    candidates_df = candidates_df.copy()
    if not pd.api.types.is_datetime64_any_dtype(candidates_df['timestamp']):
        candidates_df['timestamp'] = pd.to_datetime(candidates_df['timestamp'])
    
    # Prepare batch arrays
    n_candidates = len(candidates_df)
    
    # Storage for valid trades
    valid_indices = []
    entry_prices = []
    is_longs = []
    tp_prices = []
    sl_prices = []
    trailing_pcts = []
    highs_arrays = []
    lows_arrays = []
    closes_arrays = []
    timestamps = []
    symbols = []
    exit_times_list = []
    
    for idx, row in candidates_df.iterrows():
        ts = row['timestamp']
        symbol = row['symbol']
        is_long = bool(row['is_long'])
        entry_price = row['entry_price']
        
        # Get ATR for this symbol
        if isinstance(atr_dict, dict):
            atr = atr_dict.get(symbol, 0.02)
        elif isinstance(atr_dict, pd.Series):
            atr = atr_dict.get(symbol, 0.02)
        else:
            atr = 0.02
        
        # Calculate TP/SL prices
        if is_long:
            tp_price = entry_price * (1 + tp_mult * atr)
            sl_price = entry_price * (1 - sl_mult * atr)
        else:
            tp_price = entry_price * (1 - tp_mult * atr)
            sl_price = entry_price * (1 + sl_mult * atr)
        
        # Get future price data
        try:
            if ts not in opens.index:
                future_mask = opens.index >= ts
                if not future_mask.any():
                    continue
                entry_idx = future_mask.argmax()
            else:
                entry_idx = opens.index.get_loc(ts)
            
            end_idx = min(entry_idx + max_bars, len(opens) - 1)
            
            if entry_idx >= end_idx:
                continue
            
            future_highs = highs[symbol].iloc[entry_idx:end_idx].values.astype(np.float64)
            future_lows = lows[symbol].iloc[entry_idx:end_idx].values.astype(np.float64)
            future_closes = closes[symbol].iloc[entry_idx:end_idx].values.astype(np.float64)
            
            actual_bars = len(future_highs)
            if actual_bars == 0:
                continue
            
            # Pad arrays to max_bars if needed, but use NaN for padding
            # to indicate synthetic data (will force timeout at actual_bars)
            if actual_bars < max_bars:
                pad_size = max_bars - actual_bars
                # Use NaN for padded values - simulator will handle timeout at actual_bars
                future_highs = np.concatenate([future_highs, np.full(pad_size, np.nan)])
                future_lows = np.concatenate([future_lows, np.full(pad_size, np.nan)])
                future_closes = np.concatenate([future_closes, np.full(pad_size, np.nan)])
            
            valid_indices.append(idx)
            entry_prices.append(entry_price)
            is_longs.append(int(is_long))
            tp_prices.append(tp_price)
            sl_prices.append(sl_price)
            trailing_pcts.append(trailing_pct)
            highs_arrays.append(future_highs[:max_bars])
            lows_arrays.append(future_lows[:max_bars])
            closes_arrays.append(future_closes[:max_bars])
            timestamps.append(ts)
            symbols.append(symbol)
            exit_times_list.append((entry_idx, opens.index, actual_bars))  # Track actual bars
            
        except (KeyError, IndexError):
            continue
    
    if len(valid_indices) == 0:
        return pd.DataFrame()
    
    # Convert to arrays
    entry_prices_arr = np.array(entry_prices, dtype=np.float64)
    is_longs_arr = np.array(is_longs, dtype=np.int64)
    tp_prices_arr = np.array(tp_prices, dtype=np.float64)
    sl_prices_arr = np.array(sl_prices, dtype=np.float64)
    trailing_pcts_arr = np.array(trailing_pcts, dtype=np.float64)
    highs_arr = np.array(highs_arrays, dtype=np.float64)
    lows_arr = np.array(lows_arrays, dtype=np.float64)
    closes_arr = np.array(closes_arrays, dtype=np.float64)
    
    # Run batch simulation
    exit_prices, exit_bars, exit_reasons = simulate_trade_exit_batch(
        highs_arr, lows_arr, closes_arr,
        entry_prices_arr, is_longs_arr,
        tp_prices_arr, sl_prices_arr, trailing_pcts_arr,
        max_bars,
    )
    
    # Build results DataFrame
    exit_reason_map = {
        0: ExitReason.TP_HIT,
        1: ExitReason.SL_HIT,
        2: ExitReason.TRAILING_EXIT,
        3: ExitReason.TIMEOUT,
    }
    
    results = []
    for i, orig_idx in enumerate(valid_indices):
        is_long = bool(is_longs_arr[i])
        exit_price = exit_prices[i]
        entry_price = entry_prices_arr[i]
        exit_bar = int(exit_bars[i])
        exit_reason_int = int(exit_reasons[i])
        
        # Get exit time and actual bars
        entry_idx, price_index, actual_bars = exit_times_list[i]
        # Clamp exit_bar to actual_bars for timestamp lookup
        clamped_bar = min(exit_bar, actual_bars - 1)
        exit_time = price_index[min(entry_idx + clamped_bar, len(price_index) - 1)]
        
        # Compute label
        if is_long:
            label = np.log(exit_price / entry_price) - cost_pct
        else:
            label = np.log(entry_price / exit_price) - cost_pct
        
        if not np.isfinite(label):
            label = 0.0
        
        results.append({
            'timestamp': timestamps[i],
            'symbol': symbols[i],
            'is_long': is_long,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'exit_time': exit_time,
            'exit_reason': exit_reason_map.get(exit_reason_int, ExitReason.TIMEOUT),
            'label': label,
            'exit_bar': exit_bar,
            'tp_price': tp_prices_arr[i],
            'sl_price': sl_prices_arr[i],
        })
    
    return pd.DataFrame(results)


# ═══════════════════════════════════════════════════════════════════════════════
# Extended Trade Label Computation with Policy-Aware Mode
# ═══════════════════════════════════════════════════════════════════════════════

def compute_trade_labels_extended(
    trade_outcomes: pd.DataFrame,
    cost_pct: float = 0.0005,
    use_policy_aware: bool = False,
    price_panel: Optional[Dict[str, pd.DataFrame]] = None,
    policy_params: Optional[Dict] = None,
    max_hold_hours: int = 24,
    bars_per_hour: int = 4,
) -> Tuple[np.ndarray, Optional[pd.DataFrame]]:
    """Compute per-trade labels with optional policy-aware simulation.
    
    This function provides a unified interface for label computation:
    - If use_policy_aware=True and required data provided: Use TP/SL simulator
    - Otherwise: Use simple log returns from provided exit_price
    
    Args:
        trade_outcomes: DataFrame with trade information
            - For simple mode: entry_price, exit_price, is_long columns
            - For policy-aware mode: timestamp, symbol, entry_price, is_long columns
        cost_pct: Round-trip transaction cost percentage
        use_policy_aware: If True, use TP/SL simulator for label computation
        price_panel: Required for policy-aware mode - dict with OHLC DataFrames
        policy_params: Required for policy-aware mode - dict with TP/SL params
        max_hold_hours: Maximum holding period for policy-aware simulation
        bars_per_hour: Number of price bars per hour
    
    Returns:
        Tuple of (labels array, optional detailed outcomes DataFrame)
        - labels: Array of log-return labels net of costs
        - outcomes: DataFrame with detailed exit info (policy-aware mode only)
    """
    if use_policy_aware and price_panel is not None and policy_params is not None:
        # Run policy-aware simulation
        outcomes = compute_policy_aware_labels(
            trade_outcomes,
            price_panel,
            policy_params,
            max_hold_hours=max_hold_hours,
            cost_pct=cost_pct,
            bars_per_hour=bars_per_hour,
        )
        
        if outcomes.empty:
            # Fallback to simple mode if no valid simulations
            return compute_trade_labels(
                trade_outcomes['entry_price'].values,
                trade_outcomes.get('exit_price', trade_outcomes['entry_price']).values,
                trade_outcomes['is_long'].values,
                cost_pct,
            ), None
        
        return outcomes['label'].values, outcomes
    
    else:
        # Simple mode: use provided exit prices
        if 'exit_price' not in trade_outcomes.columns:
            raise ValueError("trade_outcomes must have 'exit_price' column for simple mode")
        
        labels = compute_trade_labels(
            trade_outcomes['entry_price'].values,
            trade_outcomes['exit_price'].values,
            trade_outcomes['is_long'].values,
            cost_pct,
        )
        return labels, None


# ═══════════════════════════════════════════════════════════════════════════════
# Pipeline Integration Functions
# ═══════════════════════════════════════════════════════════════════════════════

def run_policy_aware_labeling_step(
    candidates_df: pd.DataFrame,
    price_panel: Dict[str, pd.DataFrame],
    policy_params: Dict,
    output_path: Optional[str] = None,
    max_hold_hours: int = 24,
    cost_pct: float = 0.0005,
    bars_per_hour: int = 4,
    use_batch: bool = True,
) -> pd.DataFrame:
    """Run policy-aware label computation for all candidate trades.
    
    This function should be called before ridge_position_sizer training to
    compute labels that reflect actual trading behavior with TP/SL/trailing.
    
    Args:
        candidates_df: DataFrame with columns [timestamp, symbol, is_long, entry_price]
        price_panel: dict with 'open', 'high', 'low', 'close' DataFrames
        policy_params: dict with TP/SL params (tp_mult, sl_mult, trailing_pct, atr)
        output_path: Optional path to save labeled trades
        max_hold_hours: Maximum holding period before timeout
        cost_pct: Round-trip transaction cost
        bars_per_hour: Number of price bars per hour
        use_batch: If True, use batch processing for better performance
    
    Returns:
        DataFrame with columns:
        - timestamp, symbol, is_long, entry_price
        - exit_price, exit_time, exit_reason, label
        - exit_bar, tp_price, sl_price
    """
    tprint("=" * 80)
    tprint("POLICY-AWARE LABEL COMPUTATION")
    tprint("=" * 80)
    
    tprint(f"  Candidates: {len(candidates_df)}")
    tprint(f"  Max hold: {max_hold_hours} hours ({max_hold_hours * bars_per_hour} bars)")
    tprint(f"  Cost: {cost_pct * 100:.3f}%")
    tprint(f"  Policy params:")
    tprint(f"    TP mult: {policy_params.get('tp_mult', 3.0)}")
    tprint(f"    SL mult: {policy_params.get('sl_mult', 1.0)}")
    tprint(f"    Trailing: {policy_params.get('trailing_pct', 0.5) * 100:.1f}%")
    
    # Run label computation
    if use_batch:
        outcomes = compute_policy_aware_labels_batch(
            candidates_df,
            price_panel,
            policy_params,
            max_hold_hours=max_hold_hours,
            cost_pct=cost_pct,
            bars_per_hour=bars_per_hour,
        )
    else:
        outcomes = compute_policy_aware_labels(
            candidates_df,
            price_panel,
            policy_params,
            max_hold_hours=max_hold_hours,
            cost_pct=cost_pct,
            bars_per_hour=bars_per_hour,
        )
    
    if outcomes.empty:
        tprint("  WARNING: No valid trades computed!")
        return outcomes
    
    # Compute summary statistics
    tprint(f"  Valid trades: {len(outcomes)}")
    
    # Exit reason breakdown
    exit_counts = outcomes['exit_reason'].value_counts()
    tprint("  Exit reasons:")
    for reason, count in exit_counts.items():
        pct = count / len(outcomes) * 100
        tprint(f"    {reason.value}: {count} ({pct:.1f}%)")
    
    # Label statistics
    labels = outcomes['label']
    tprint(f"  Label statistics:")
    tprint(f"    Mean: {labels.mean():.6f}")
    tprint(f"    Std: {labels.std():.6f}")
    tprint(f"    Min: {labels.min():.6f}")
    tprint(f"    Max: {labels.max():.6f}")
    tprint(f"    Win rate: {(labels > 0).mean() * 100:.1f}%")
    
    # Holding time statistics
    if 'exit_bar' in outcomes.columns:
        avg_bars = outcomes['exit_bar'].mean()
        avg_hours = avg_bars / bars_per_hour
        tprint(f"    Avg hold: {avg_hours:.1f} hours ({avg_bars:.1f} bars)")
    
    # Save to disk if path provided
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        outcomes.to_csv(output_path, index=False)
        tprint(f"  Saved to: {output_path}")
    
    tprint("=" * 80)
    
    return outcomes


def prepare_policy_params_from_tpsl_optimiser(
    tpsl_results: Dict,
    atr_values: Optional[Dict[str, float]] = None,
) -> Dict:
    """Prepare policy_params dict from tpsl_optimiser output.
    
    This function converts the output format from tpsl_optimiser steps
    into the format expected by compute_policy_aware_labels.
    
    Args:
        tpsl_results: Dict from tpsl_optimiser containing:
            - tp_mult: Take-profit multiplier
            - sl_mult: Stop-loss multiplier
            - act_n, be_act_n: Trailing activation parameters
            - Other params from profit_exit_opt
        atr_values: Optional dict of ATR values per symbol
    
    Returns:
        Dict with policy_params format:
            - tp_mult: Take-profit multiplier
            - sl_mult: Stop-loss multiplier
            - trailing_pct: Trailing percentage
            - atr: ATR values per symbol
    """
    # Extract TP/SL multipliers
    tp_mult = tpsl_results.get('tp_mult', 3.0)
    sl_mult = tpsl_results.get('sl_mult', 1.0)
    
    # Convert trailing params from profit_exit_opt
    # act_n and be_act_n define the trailing behavior
    act_n = tpsl_results.get('act_n', 1.0)
    be_act_n = tpsl_results.get('be_act_n', 1.0)
    
    # Convert to trailing_pct
    # The penalty formula from profit_exit_opt: (act_n - be_act_n) * tp_pct
    # This represents the give-back from peak
    # For trailing_pct, we use a simplified conversion
    if act_n <= be_act_n:
        # No trailing penalty - trailing is effectively disabled
        trailing_pct = 1.0  # Very loose trailing
    else:
        # Convert to percentage of TP distance
        # This is an approximation - actual trailing is more complex
        trailing_pct = (act_n - be_act_n) * 0.5
    
    # Clamp to reasonable range
    trailing_pct = max(0.01, min(trailing_pct, 1.0))
    
    return {
        'tp_mult': tp_mult,
        'sl_mult': sl_mult,
        'trailing_pct': trailing_pct,
        'atr': atr_values or {},
    }


def validate_price_panel(
    price_panel: Dict[str, pd.DataFrame],
    symbols: List[str],
    start_time: pd.Timestamp,
    end_time: pd.Timestamp,
) -> bool:
    """Validate that price_panel has sufficient data for label computation.
    
    Args:
        price_panel: dict with OHLC DataFrames
        symbols: List of required symbols
        start_time: Start of required time range
        end_time: End of required time range
    
    Returns:
        True if valid, raises ValueError otherwise
    """
    required_keys = ['open', 'high', 'low', 'close']
    
    for key in required_keys:
        if key not in price_panel:
            raise ValueError(f"price_panel missing required key: {key}")
        
        df = price_panel[key]
        
        # Check symbols
        missing_symbols = set(symbols) - set(df.columns)
        if missing_symbols:
            raise ValueError(f"price_panel['{key}'] missing symbols: {missing_symbols}")
        
        # Check time range
        if df.index.min() > start_time:
            raise ValueError(f"price_panel['{key}'] starts after required start_time")
        if df.index.max() < end_time:
            raise ValueError(f"price_panel['{key}'] ends before required end_time")
    
    return True


# ═══════════════════════════════════════════════════════════════════════════════
# Target Representation Builders & Race
# ═══════════════════════════════════════════════════════════════════════════════

EPS = 1e-12


def _ewma_vol_series(returns: pd.Series, span: int = 96) -> pd.Series:
    """Backward-looking EWMA vol proxy (no leakage)."""
    mu = returns.ewm(span=span, adjust=False).mean()
    var = (returns - mu).pow(2).ewm(span=span, adjust=False).mean()
    return np.sqrt(var).clip(lower=EPS)


def _partial_vol_adjust(x: np.ndarray, vol: np.ndarray, lam: float = 0.5) -> np.ndarray:
    """x / vol^lam  (lam=0 → no adjust, lam=1 → full)."""
    return x / np.power(np.maximum(vol, EPS), lam)


def _rolling_rankpct(vals: np.ndarray, window: int = 2000) -> np.ndarray:
    """Percentile rank of current value within trailing window (exclude current)."""
    n = len(vals)
    out = np.full(n, np.nan, dtype=np.float64)
    min_hist = max(20, window // 5)
    for i in range(n):
        j0 = max(0, i - window)
        hist = vals[j0:i]
        if len(hist) < min_hist:
            continue
        out[i] = float(np.mean(hist <= vals[i]))
    return out


def build_target_winsorized(
    returns: np.ndarray,
    clip_L: float = 0.02,
    vol: np.ndarray | None = None,
    vol_mode: str = "none",
    lam: float = 0.5,
) -> np.ndarray:
    """Target 1: Cost-adjusted winsorized log return, optionally vol-adjusted."""
    y = np.clip(returns, -clip_L, clip_L)
    if vol_mode == "full" and vol is not None:
        y = y / np.maximum(vol, EPS)
    elif vol_mode == "partial" and vol is not None:
        y = _partial_vol_adjust(y, vol, lam)
    return y.astype(np.float64)


def build_target_huber_advantage(
    returns: np.ndarray,
    symbols: np.ndarray,
    delta: float = 0.01,
    baseline_window: int = 200,
    vol: np.ndarray | None = None,
    vol_mode: str = "none",
    lam: float = 0.5,
) -> np.ndarray:
    """Target 2: Huberized advantage over per-symbol rolling baseline."""
    df = pd.DataFrame({'r': returns, 'symbol': symbols})
    min_periods = max(20, baseline_window // 5)
    df['baseline'] = (
        df.groupby('symbol')['r']
        .transform(lambda s: s.rolling(baseline_window, min_periods=min_periods).median().shift(1))
    )
    resid = (df['r'] - df['baseline']).fillna(0.0).values
    y = np.clip(resid, -delta, delta)
    if vol_mode == "full" and vol is not None:
        y = y / np.maximum(vol, EPS)
    elif vol_mode == "partial" and vol is not None:
        y = _partial_vol_adjust(y, vol, lam)
    return y.astype(np.float64)


def build_target_rolling_rank(
    returns: np.ndarray,
    symbols: np.ndarray,
    window: int = 2000,
    vol: np.ndarray | None = None,
    vol_mode: str = "none",
    lam: float = 0.5,
) -> np.ndarray:
    """Target 3: Rolling per-symbol rank of return mapped to [-1, 1]."""
    x = returns.copy()
    if vol_mode == "full" and vol is not None:
        x = x / np.maximum(vol, EPS)
    elif vol_mode == "partial" and vol is not None:
        x = _partial_vol_adjust(x, vol, lam)

    df = pd.DataFrame({'x': x, 'symbol': symbols})
    df['u01'] = (
        df.groupby('symbol')['x']
        .transform(lambda s: pd.Series(_rolling_rankpct(s.values, window), index=s.index))
    )
    y = (2.0 * df['u01'].fillna(0.5) - 1.0).values
    return y.astype(np.float64)


def build_target_rolling_rank_residual(
    returns: np.ndarray,
    symbols: np.ndarray,
    baseline_window: int = 500,
    rank_window: int = 2000,
    baseline_kind: str = "median",
    vol: np.ndarray | None = None,
    vol_mode: str = "none",
    lam: float = 0.5,
) -> np.ndarray:
    """Target 4: Rolling rank of excess-over-baseline return mapped to [-1, 1]."""
    df = pd.DataFrame({'r': returns, 'symbol': symbols})
    min_periods = max(20, baseline_window // 5)
    if baseline_kind == "median":
        df['b'] = df.groupby('symbol')['r'].transform(
            lambda s: s.rolling(baseline_window, min_periods=min_periods).median().shift(1))
    elif baseline_kind == "mean":
        df['b'] = df.groupby('symbol')['r'].transform(
            lambda s: s.rolling(baseline_window, min_periods=min_periods).mean().shift(1))
    elif baseline_kind == "ewma":
        df['b'] = df.groupby('symbol')['r'].transform(
            lambda s: s.ewm(span=baseline_window, adjust=False).mean().shift(1))
    else:
        df['b'] = 0.0
    e = (df['r'] - df['b']).fillna(0.0).values

    if vol_mode == "full" and vol is not None:
        e = e / np.maximum(vol, EPS)
    elif vol_mode == "partial" and vol is not None:
        e = _partial_vol_adjust(e, vol, lam)

    df['x'] = e
    df['u01'] = (
        df.groupby('symbol')['x']
        .transform(lambda s: pd.Series(_rolling_rankpct(s.values, rank_window), index=s.index))
    )
    y = (2.0 * df['u01'].fillna(0.5) - 1.0).values
    return y.astype(np.float64)


def build_trade_vol_proxy(
    returns: np.ndarray,
    symbols: np.ndarray,
    timestamps: np.ndarray | None = None,
    span: int = 96,
) -> np.ndarray:
    """Build per-trade EWMA vol proxy from returns grouped by symbol."""
    df = pd.DataFrame({'r': returns, 'symbol': symbols})
    if timestamps is not None:
        df['ts'] = timestamps
        df = df.sort_values(['symbol', 'ts'])
    df['vol'] = (
        df.groupby('symbol')['r']
        .transform(lambda s: _ewma_vol_series(s, span=span))
    )
    if timestamps is not None:
        df = df.sort_index()
    return df['vol'].fillna(df['vol'].median()).values.astype(np.float64)


def run_ridge_target_race(
    X: np.ndarray,
    returns: np.ndarray,
    symbols: np.ndarray | None,
    timestamps: np.ndarray | None,
    alpha: float = 0.5,
    cost_pct: float = 0.0005,
    clip_L: float = 0.02,
) -> tuple:
    """Race target representations for Ridge position sizer.

    Builds candidate targets, fits Ridge (alpha fixed) on each via 3-fold
    purged CV, picks the one with highest OOF Spearman IC.

    Returns:
        (best_name, best_y, race_log): winning target name, array, and log lines
    """
    race_log = []
    n = len(returns)

    # Build vol proxy if symbols available
    vol = None
    if symbols is not None and len(np.unique(symbols)) > 1:
        vol = build_trade_vol_proxy(returns, symbols, timestamps)

    # Ensure symbols is usable (fallback to single symbol)
    if symbols is None:
        symbols = np.full(n, "ALL", dtype=object)

    # Build candidate targets
    candidates = {}

    # 1. Winsorized log return
    candidates["winsorized"] = build_target_winsorized(returns, clip_L=clip_L)
    if vol is not None:
        candidates["winsorized_voladj"] = build_target_winsorized(
            returns, clip_L=clip_L, vol=vol, vol_mode="partial")

    # 2. Huber advantage
    candidates["huber_adv"] = build_target_huber_advantage(
        returns, symbols, delta=clip_L)
    if vol is not None:
        candidates["huber_adv_voladj"] = build_target_huber_advantage(
            returns, symbols, delta=clip_L, vol=vol, vol_mode="partial")

    # 3. Rolling rank
    candidates["rolling_rank"] = build_target_rolling_rank(returns, symbols)
    if vol is not None:
        candidates["rolling_rank_voladj"] = build_target_rolling_rank(
            returns, symbols, vol=vol, vol_mode="partial")

    # 4. Rolling rank residual
    candidates["rank_residual"] = build_target_rolling_rank_residual(returns, symbols)
    if vol is not None:
        candidates["rank_residual_voladj"] = build_target_rolling_rank_residual(
            returns, symbols, vol=vol, vol_mode="partial")

    race_log.append(f"    Ridge target race: {len(candidates)} candidates")

    # Evaluate each candidate via Ridge CV
    from sklearn.linear_model import Ridge as SkRidge

    best_ic = -np.inf
    best_name = "winsorized"
    best_y = candidates["winsorized"]

    for tname, y_cand in candidates.items():
        fin = np.isfinite(y_cand)
        if fin.sum() < 100:
            race_log.append(f"      {tname}: skipped (only {fin.sum()} finite)")
            continue

        # Simple 3-fold walk-forward CV with Ridge
        fold_size = n // 3
        ics = []
        for fold in range(3):
            val_start = fold * fold_size
            val_end = val_start + fold_size if fold < 2 else n
            # Train on everything before val_start (walk-forward)
            train_end = max(0, val_start - 5)  # 5-sample purge
            if train_end < 50:
                continue
            tr_idx = np.arange(0, train_end)
            va_idx = np.arange(val_start, val_end)

            X_tr, X_va = X[tr_idx], X[va_idx]
            y_tr, y_va = y_cand[tr_idx], y_cand[va_idx]

            # Scale per fold
            mu = np.nanmean(X_tr, axis=0)
            sd = np.nanstd(X_tr, axis=0)
            sd = np.where(sd < 1e-9, 1.0, sd)
            X_tr_s = (X_tr - mu) / sd
            X_va_s = (X_va - mu) / sd

            try:
                mdl = SkRidge(alpha=alpha, fit_intercept=True)
                mdl.fit(X_tr_s, y_tr)
                pred = mdl.predict(X_va_s)
                ic_fold = float(spearmanr(pred, y_va).correlation)
                if np.isfinite(ic_fold):
                    ics.append(ic_fold)
            except Exception:
                continue

        mean_ic = float(np.mean(ics)) if ics else -1.0
        race_log.append(f"      {tname}: IC={mean_ic:.4f} ({len(ics)} folds)")

        if mean_ic > best_ic:
            best_ic = mean_ic
            best_name = tname
            best_y = y_cand

    race_log.append(f"    Winner: {best_name} (IC={best_ic:.4f})")
    return best_name, best_y, race_log


# ═══════════════════════════════════════════════════════════════════════════════
# Huber Loss with L2 Regularization
# ═══════════════════════════════════════════════════════════════════════════════

def huber_loss(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    delta: float = 1.0,
    sample_weight: np.ndarray | None = None,
) -> float:
    """Compute Huber loss, robust to outliers.
    
    For |residual| <= delta: 0.5 * residual^2
    For |residual| > delta: delta * (|residual| - 0.5 * delta)
    
    Args:
        y_true: True values
        y_pred: Predicted values
        delta: Threshold for quadratic vs linear loss
        sample_weight: Optional sample weights
        
    Returns:
        Mean Huber loss
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    
    residual = y_true - y_pred
    abs_residual = np.abs(residual)
    
    # Huber loss formula
    quadratic = np.minimum(abs_residual, delta)
    linear = abs_residual - quadratic
    
    loss = 0.5 * quadratic ** 2 + delta * linear
    
    if sample_weight is not None:
        sample_weight = np.asarray(sample_weight, dtype=float)
        loss = loss * sample_weight
        return float(np.sum(loss) / np.sum(sample_weight))
    
    return float(np.mean(loss))


def huber_loss_gradient(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    delta: float = 1.0,
    sample_weight: np.ndarray | None = None,
) -> np.ndarray:
    """Compute gradient of Huber loss with respect to predictions.
    
    The gradient is computed to match the objective function exactly:
    - If sample_weight is provided: objective = sum(w_i * L_i) / sum(w_i)
    - Gradient w.r.t. predictions: w_i * g_i / sum(w_i)
    
    Args:
        y_true: True values
        y_pred: Predicted values
        delta: Threshold for quadratic vs linear loss
        sample_weight: Optional sample weights
        
    Returns:
        Gradient array (same scale as objective derivative)
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    
    residual = y_pred - y_true  # Note: gradient w.r.t. prediction
    abs_residual = np.abs(residual)
    
    # Gradient: residual for quadratic region, delta * sign(residual) for linear
    grad = np.where(
        abs_residual <= delta,
        residual,
        delta * np.sign(residual)
    )
    
    if sample_weight is not None:
        sample_weight = np.asarray(sample_weight, dtype=float)
        # Match the weighted objective: sum(w_i * L_i) / sum(w_i)
        # Gradient is: w_i * g_i / sum(w_i)
        grad = grad * sample_weight / (np.sum(sample_weight) + 1e-12)
    
    return grad


# ═══════════════════════════════════════════════════════════════════════════════
# Standardization Helper
# ═══════════════════════════════════════════════════════════════════════════════

class PredictionScaler:
    """Standardize prediction columns to zero mean and unit variance.
    
    Different base models may output predictions on different scales
    (probabilities, raw returns, log-odds). This scaler normalizes
    them for stable linear combination.
    """
    
    def __init__(self):
        self.means_: Optional[np.ndarray] = None
        self.stds_: Optional[np.ndarray] = None
        
    def fit(self, X: np.ndarray) -> 'PredictionScaler':
        """Fit scaler on training data."""
        X = np.asarray(X, dtype=float)
        self.means_ = np.nanmean(X, axis=0)
        self.stds_ = np.nanstd(X, axis=0, ddof=0)
        # Prevent division by zero
        self.stds_ = np.where(self.stds_ < 1e-9, 1.0, self.stds_)
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Apply standardization."""
        X = np.asarray(X, dtype=float)
        if self.means_ is None or self.stds_ is None:
            raise RuntimeError("Scaler must be fitted before transform")
        return (X - self.means_) / self.stds_
    
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit and transform in one step."""
        return self.fit(X).transform(X)


# ═══════════════════════════════════════════════════════════════════════════════
# Ridge Position Sizer Class
# ═══════════════════════════════════════════════════════════════════════════════

class RidgePositionSizer:
    """L2-regularized constrained linear combiner for meta model outputs.
    
    This class learns optimal combination weights for multiple model predictions,
    with asymmetric loss that penalizes losing trades more heavily.
    
    Note: Despite the name, this is NOT standard Ridge regression. It's a
    constrained linear combination solved via SLSQP optimization with:
    - Huber loss (robust to outliers)
    - L2 regularization
    - Asymmetric sample weights (losing trades weighted more)
    - Optional simplex constraints (non-negative, sum-to-one)
    
    Features:
    - Takes OOF predictions from multiple models/timeframes
    - Learns optimal combination weights with asymmetric loss
    - Uses composite J zscore for hyperparameter selection
    - Outputs position sizing aligned with tpsl_optimiser
    
    Example:
        >>> sizer = RidgePositionSizer(gamma_range=(1.0, 3.0), alpha_range=(1e-4, 1e-1))
        >>> sizer.fit(oof_preds, trade_outcomes, timestamps=timestamps)
        >>> position_signals = sizer.predict(new_model_preds)
        >>> weights = sizer.get_weights()
    """
    
    def __init__(
        self,
        gamma_range: Tuple[float, float] = (1.0, 3.0),
        alpha_range: Tuple[float, float] = (1e-4, 1e-1),
        delta_range: Tuple[float, float] = (0.5, 2.0),
        n_grid_points: int = 10,
        cost_pct: float = 0.0005,
        sum_to_one: bool = True,
        non_negative: bool = True,
        top_k_pct: float = 0.30,
        random_state: int = 42,
    ):
        """Initialize the Ridge Position Sizer.
        
        Args:
            gamma_range: Range for asymmetric weight parameter (losing trades weight)
            alpha_range: Range for L2 regularization strength
            delta_range: Range for Huber loss delta parameter
            n_grid_points: Number of grid points for hyperparameter search
            cost_pct: Transaction cost percentage for label computation
            sum_to_one: If True, constrain weights to sum to 1
            non_negative: If True, constrain weights to be non-negative
            top_k_pct: Percentage of top predictions to select for evaluation
            random_state: Random seed for reproducibility
        """
        self.gamma_range = gamma_range
        self.alpha_range = alpha_range
        self.delta_range = delta_range
        self.n_grid_points = n_grid_points
        self.cost_pct = cost_pct
        self.sum_to_one = sum_to_one
        self.non_negative = non_negative
        self.top_k_pct = top_k_pct
        self.random_state = random_state
        
        # Fitted attributes
        self.weights_: Optional[np.ndarray] = None
        self.model_names_: Optional[List[str]] = None
        self.best_params_: Optional[Dict] = None
        self.cv_results_: Optional[pd.DataFrame] = None
        self.scaler_: Optional[PredictionScaler] = None
        self.is_fitted_: bool = False
        
    def _compute_sample_weights(
        self,
        y: np.ndarray,
        gamma: float,
    ) -> np.ndarray:
        """Compute asymmetric sample weights.
        
        Losing trades (y < 0) get higher weight:
        wi = 1 + gamma * 1[yi < 0]
        
        For gamma in [1, 3], losing trades get 2-4x weight.
        
        Args:
            y: Trade outcome labels
            gamma: Asymmetric weight parameter
            
        Returns:
            Array of sample weights
        """
        y = np.asarray(y, dtype=float)
        is_loser = (y < 0).astype(float)
        weights = 1.0 + gamma * is_loser
        return weights
    
    def _objective(
        self,
        weights: np.ndarray,
        X: np.ndarray,
        y: np.ndarray,
        alpha: float,
        delta: float,
        sample_weight: np.ndarray,
    ) -> float:
        """Objective function: Huber loss + L2 regularization.
        
        Args:
            weights: Model combination weights
            X: Prediction matrix (n_samples, n_models), already standardized
            y: True labels
            alpha: L2 regularization strength
            delta: Huber loss delta parameter
            sample_weight: Sample weights
            
        Returns:
            Total loss value
        """
        # Compute combined prediction
        y_pred = X @ weights
        
        # Huber loss
        loss = huber_loss(y, y_pred, delta, sample_weight)
        
        # L2 regularization
        reg = 0.5 * alpha * np.sum(weights ** 2)
        
        return loss + reg
    
    def _objective_gradient(
        self,
        weights: np.ndarray,
        X: np.ndarray,
        y: np.ndarray,
        alpha: float,
        delta: float,
        sample_weight: np.ndarray,
    ) -> np.ndarray:
        """Gradient of the objective function.
        
        The gradient is computed to match the objective exactly:
        - Objective = Huber loss + L2 regularization
        - Huber loss gradient is already properly scaled by huber_loss_gradient()
        - Chain rule: gradient w.r.t. weights = X.T @ grad_pred
        
        Args:
            weights: Model combination weights
            X: Prediction matrix (n_samples, n_models), already standardized
            y: True labels
            alpha: L2 regularization strength
            delta: Huber loss delta parameter
            sample_weight: Sample weights
            
        Returns:
            Gradient array
        """
        # Compute combined prediction
        y_pred = X @ weights
        
        # Gradient of Huber loss (already properly scaled)
        huber_grad = huber_loss_gradient(y, y_pred, delta, sample_weight)
        
        # Chain rule: gradient w.r.t. weights
        # No extra /len(y) since huber_grad is already scaled correctly
        grad = X.T @ huber_grad
        
        # Add L2 regularization gradient
        grad = grad + alpha * weights
        
        return grad
    
    def _fit_weights(
        self,
        X: np.ndarray,
        y: np.ndarray,
        alpha: float,
        delta: float,
        gamma: float,
    ) -> np.ndarray:
        """Fit combination weights using constrained optimization.
        
        Args:
            X: Prediction matrix (n_samples, n_models), already standardized
            y: True labels
            alpha: L2 regularization strength
            delta: Huber loss delta parameter
            gamma: Asymmetric weight parameter
            
        Returns:
            Optimized weights
        """
        n_models = X.shape[1]
        
        # Compute sample weights
        sample_weight = self._compute_sample_weights(y, gamma)
        
        # Initial weights: uniform
        w0 = np.ones(n_models) / n_models
        
        # Set up constraints
        constraints = []
        if self.sum_to_one:
            constraints.append({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0})
        
        # Set up bounds - use (0, +inf) with sum-to-one, no upper bound needed
        if self.non_negative:
            bounds = [(0.0, None) for _ in range(n_models)]
        else:
            bounds = [(None, None) for _ in range(n_models)]
        
        # Optimize
        result = minimize(
            self._objective,
            w0,
            args=(X, y, alpha, delta, sample_weight),
            method='SLSQP',
            jac=self._objective_gradient,
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 500, 'ftol': 1e-8}
        )
        
        # Check for optimization failure
        if not result.success:
            tprint(f"  WARNING: SLSQP optimization failed: {result.message}")
            tprint(f"  Falling back to uniform weights")
            return w0
        
        return result.x
    
    def _evaluate_params(
        self,
        X: np.ndarray,
        y: np.ndarray,
        timestamps: np.ndarray | None,
        alpha: float,
        delta: float,
        gamma: float,
        groups: np.ndarray | None = None,
    ) -> Dict:
        """Evaluate hyperparameters using purged cross-validation.
        
        Computes metrics using a realistic trading policy:
        1. Select top-k% predictions per time slice
        2. Compute returns with proper time ordering
        3. Aggregate to daily returns for Sortino/MaxDD
        
        Args:
            X: Prediction matrix (unscaled - scaling done per-fold to avoid leakage)
            y: True labels
            timestamps: Array of timestamps for time-based operations
            alpha: L2 regularization strength
            delta: Huber loss delta parameter
            gamma: Asymmetric weight parameter
            groups: Optional group labels for CV splits
            
        Returns:
            Dictionary of evaluation metrics
        """
        from extreme_price_movements.purged_cv import PurgedKFold
        
        # Use purged CV to avoid lookahead bias
        # If timestamps available, use time-based purging (purge=5 means 5 seconds)
        # Otherwise use index-based purging (purge=5 means 5 samples)
        if timestamps is not None:
            # Time-based purging: purge 5 hours (18000 seconds) before test set
            pkf = PurgedKFold(n_splits=3, purge=18000, embargo=7200, times=timestamps)
        else:
            # Index-based purging: purge 5 samples before test set
            pkf = PurgedKFold(n_splits=3, purge=5, embargo=2)
        
        oof_preds = np.full(len(y), np.nan)
        oof_weights = None
        
        # Get indices for CV split - pass groups if available
        split_args = [X]
        if groups is not None:
            split_args.append(groups)
        
        for fold_idx, (train_idx, val_idx) in enumerate(pkf.split(*split_args)):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train = y[train_idx]
            
            # Fit scaler on training fold only
            fold_scaler = PredictionScaler()
            X_train_scaled = fold_scaler.fit_transform(X_train)
            X_val_scaled = fold_scaler.transform(X_val)
            
            weights = self._fit_weights(X_train_scaled, y_train, alpha, delta, gamma)
            oof_preds[val_idx] = X_val_scaled @ weights
            
            if oof_weights is None:
                oof_weights = weights.copy()
        
        # Compute metrics on valid OOF predictions
        mask = np.isfinite(oof_preds)
        if mask.sum() < 10:
            return {
                'alpha': alpha,
                'delta': delta,
                'gamma': gamma,
                'PnL_total': 0.0,
                'Sortino': 0.0,
                'MaxDD': 1.0,
                'IC': 0.0,
                'WinRate': 0.0,
            }
        
        pred = oof_preds[mask]
        true = y[mask]
        ts_masked = timestamps[mask] if timestamps is not None else None
        
        # Apply top-k% selection policy PER TIMESTAMP SLICE
        # This is more realistic: at each timestamp, we select top-k assets
        selected_indices = []
        
        if ts_masked is not None:
            # Group by timestamp and select top-k within each group
            import pandas as pd
            df = pd.DataFrame({
                'pred': pred,
                'true': true,
                'ts': ts_masked,
                'orig_idx': np.arange(len(pred))
            })
            
            for ts_val, group in df.groupby('ts'):
                n_in_group = len(group)
                k = max(1, int(self.top_k_pct * n_in_group))
                # Select top-k by prediction within this timestamp
                top_k_local = group.nlargest(k, 'pred')['orig_idx'].values
                selected_indices.extend(top_k_local)
            
            selected_indices = np.array(selected_indices)
        else:
            # Fallback: global top-k if no timestamps
            k = max(1, int(self.top_k_pct * len(pred)))
            selected_indices = np.argpartition(pred, -k)[-k:]
        
        # Get returns for selected trades
        selected_returns = true[selected_indices]
        
        # Sort by timestamp if available for proper equity curve
        if ts_masked is not None:
            selected_ts = ts_masked[selected_indices]
            sort_order = np.argsort(selected_ts)
            selected_returns = selected_returns[sort_order]
        else:
            sort_order = None
        
        # Compute average PnL per selected trade (normalized by N_selected)
        # This prevents PnL from scaling with the number of time slices
        n_selected = len(selected_returns)
        pnl_total = float(np.mean(selected_returns)) if n_selected > 0 else 0.0
        
        # Aggregate to daily returns if timestamps available
        if ts_masked is not None and len(selected_returns) > 1:
            selected_ts_sorted = ts_masked[selected_indices][sort_order] if sort_order is not None else ts_masked[selected_indices]
            if selected_ts_sorted is not None:
                # Convert to pandas for easy daily aggregation
                try:
                    daily_df = pd.DataFrame({
                        'return': selected_returns,
                        'date': pd.to_datetime(selected_ts_sorted).date
                    })
                    daily_returns = daily_df.groupby('date')['return'].sum().values
                except (ValueError, TypeError, AttributeError):
                    # Fall back to unaggregated returns if date parsing fails
                    daily_returns = selected_returns
            else:
                daily_returns = selected_returns
        else:
            daily_returns = selected_returns
        
        # Sortino ratio on daily returns
        neg_returns = daily_returns[daily_returns < 0]
        if len(neg_returns) > 0 and np.std(neg_returns) > 1e-9:
            sortino = float(np.mean(daily_returns) / np.std(neg_returns))
        else:
            sortino = 0.0
        
        # Max drawdown on cumulative returns (time-ordered)
        # Use percentage-based drawdown for scale-invariance
        equity = np.cumsum(daily_returns)
        if len(equity) > 0:
            peak = np.maximum.accumulate(equity)
            # Percentage drawdown: (peak - equity) / peak
            # For log returns, this is approximately the log drawdown
            # Avoid division by zero by using absolute drawdown when peak is near zero
            safe_peak = np.where(np.abs(peak) > 1e-9, peak, 1.0)
            pct_drawdown = np.where(np.abs(peak) > 1e-9, (peak - equity) / np.abs(peak), peak - equity)
            max_dd = float(np.max(pct_drawdown))
        else:
            max_dd = 0.0
        max_dd = max(max_dd, 1e-6)  # Prevent division by zero
        
        # IC (Spearman correlation) on all predictions
        try:
            ic = float(spearmanr(pred, true).correlation)
            if not np.isfinite(ic):
                ic = 0.0
        except (ValueError, TypeError):
            # Handle constant arrays or invalid inputs
            ic = 0.0
        
        # Win rate on selected trades
        win_rate = float(np.mean(selected_returns > 0))
        
        return {
            'alpha': alpha,
            'delta': delta,
            'gamma': gamma,
            'PnL_total': pnl_total,
            'Sortino': sortino,
            'MaxDD': max_dd,
            'IC': ic,
            'WinRate': win_rate,
            'N_selected': len(selected_returns),
        }
    
    def fit(
        self,
        oof_preds: pd.DataFrame,
        trade_outcomes: pd.DataFrame,
        timestamps: np.ndarray | None = None,
        groups: np.ndarray | None = None,
        labels: np.ndarray | None = None,
        symbols: np.ndarray | None = None,
    ) -> 'RidgePositionSizer':
        """Fit the ridge combiner on OOF predictions.
        
        Args:
            oof_preds: DataFrame with columns [model_name, pred] per model,
                      or wide format with one column per model.
                      IMPORTANT: These must be TRUE out-of-fold predictions
                      from training, NOT in-sample predictions from predict().
            trade_outcomes: DataFrame with entry_price, exit_price, is_long columns,
                           OR DataFrame with 'return' column if labels not provided.
            timestamps: Array of timestamps for each trade (required for proper
                       time-based CV and drawdown computation)
            groups: Optional group labels for CV splits (e.g., day/week)
            labels: Optional pre-computed labels (log returns). If provided,
                   trade_outcomes entry/exit prices are not needed.
            symbols: Optional array of symbol names per trade (for target race
                    vol proxy and per-symbol baselines)
            
        Returns:
            self
        """
        tprint("RidgePositionSizer.fit: Starting...")
        
        # Extract model predictions
        if 'model_name' in oof_preds.columns and 'pred' in oof_preds.columns:
            # Long format: pivot to wide
            pred_wide = oof_preds.pivot(columns='model_name', values='pred')
            self.model_names_ = list(pred_wide.columns)
            X = pred_wide.values
        else:
            # Wide format: one column per model
            self.model_names_ = list(oof_preds.columns)
            X = oof_preds.values
        
        # Compute raw trade returns (used for target race)
        if labels is not None:
            y_raw = np.asarray(labels, dtype=float)
            tprint(f"  Using provided labels: mean={np.mean(y_raw):.6f}, std={np.std(y_raw):.6f}")
        elif 'return' in trade_outcomes.columns:
            y_raw = np.asarray(trade_outcomes['return'].values, dtype=float)
            tprint(f"  Using returns from trade_outcomes: mean={np.mean(y_raw):.6f}, std={np.std(y_raw):.6f}")
        elif all(c in trade_outcomes.columns for c in ['entry_price', 'exit_price', 'is_long']):
            y_raw = compute_trade_labels(
                trade_outcomes['entry_price'].values,
                trade_outcomes['exit_price'].values,
                trade_outcomes['is_long'].values,
                self.cost_pct,
            )
            tprint(f"  Computed labels from prices: mean={np.mean(y_raw):.6f}, std={np.std(y_raw):.6f}")
        else:
            raise ValueError(
                "trade_outcomes must have either 'return' column, "
                "or 'entry_price', 'exit_price', 'is_long' columns, "
                "or labels must be provided directly"
            )
        
        # Extract symbols from trade_outcomes if not provided
        if symbols is None and 'symbol' in trade_outcomes.columns:
            symbols = trade_outcomes['symbol'].values
        
        # Align lengths
        n = min(len(X), len(y_raw))
        X = X[:n]
        y_raw = y_raw[:n]
        if timestamps is not None:
            timestamps = timestamps[:n]
        if groups is not None:
            groups = groups[:n]
        if symbols is not None:
            symbols = symbols[:n]
        
        # Handle NaN/Inf in predictions
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        y_raw = np.nan_to_num(y_raw, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Run target representation race to find best y for this bucket
        tprint("  Running target representation race...")
        tgt_name, y, race_log = run_ridge_target_race(
            X, y_raw, symbols, timestamps,
            alpha=0.5, cost_pct=self.cost_pct,
        )
        for line in race_log:
            tprint(line)
        self.best_target_name_ = tgt_name
        
        # NOTE: Do NOT scale globally here - scaling is done per-fold in _evaluate_params
        # to prevent data leakage. The final scaler is fit after CV on all data.
        
        # Grid search over hyperparameters
        gamma_vals = np.linspace(self.gamma_range[0], self.gamma_range[1], self.n_grid_points)
        alpha_vals = np.logspace(
            np.log10(self.alpha_range[0]),
            np.log10(self.alpha_range[1]),
            self.n_grid_points
        )
        delta_vals = np.linspace(self.delta_range[0], self.delta_range[1], 5)
        
        results = []
        total_combos = len(gamma_vals) * len(alpha_vals) * len(delta_vals)
        tprint(f"  Evaluating {total_combos} hyperparameter combinations...")
        
        for gamma in gamma_vals:
            for alpha in alpha_vals:
                for delta in delta_vals:
                    metrics = self._evaluate_params(
                        X, y, timestamps, alpha, delta, gamma, groups
                    )
                    results.append(metrics)
        
        # Create results DataFrame
        self.cv_results_ = pd.DataFrame(results)
        
        # Compute composite J z-score for selection
        # Use PnL_total instead of annualized since we don't have proper day count
        self.cv_results_['J_zscore'] = composite_J_zscore(
            self.cv_results_,
            pnl_col='PnL_total',
            sortino_col='Sortino',
            maxdd_col='MaxDD',
            a=1.0,
            b=1.0,
            group_col=None,
            use_robust=True,
        )
        
        # Select best hyperparameters
        best_idx = self.cv_results_['J_zscore'].idxmax()
        best_row = self.cv_results_.loc[best_idx]
        
        self.best_params_ = {
            'alpha': float(best_row['alpha']),
            'delta': float(best_row['delta']),
            'gamma': float(best_row['gamma']),
            'J_zscore': float(best_row['J_zscore']),
        }
        
        tprint(f"  Best params: alpha={self.best_params_['alpha']:.6f}, "
               f"delta={self.best_params_['delta']:.3f}, "
               f"gamma={self.best_params_['gamma']:.3f}, "
               f"J_zscore={self.best_params_['J_zscore']:.4f}")
        
        # Fit final scaler on all data (after CV to avoid data leakage)
        self.scaler_ = PredictionScaler()
        X_scaled = self.scaler_.fit_transform(X)
        
        # Fit final weights on all data (scaled)
        self.weights_ = self._fit_weights(
            X_scaled, y,
            self.best_params_['alpha'],
            self.best_params_['delta'],
            self.best_params_['gamma'],
        )
        
        self.is_fitted_ = True
        
        tprint(f"RidgePositionSizer.fit: Done. Weights: {self.weights_}")
        
        return self
    
    def predict(self, model_preds: pd.DataFrame) -> np.ndarray:
        """Return combined position sizing signal.
        
        Args:
            model_preds: DataFrame with same column structure as training data.
                        These should be new predictions (not OOF).
            
        Returns:
            Array of combined position sizing signals
        """
        if not self.is_fitted_:
            raise RuntimeError("RidgePositionSizer must be fitted before predict")
        
        # Extract predictions in correct column order
        if self.model_names_ is None:
            raise RuntimeError("Model names not set during fitting")
        
        X = model_preds[self.model_names_].values
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Apply saved scaler
        if self.scaler_ is not None:
            X = self.scaler_.transform(X)
        
        return X @ self.weights_
    
    def get_weights(self) -> Dict[str, float]:
        """Return learned combination weights per model.
        
        Returns:
            Dictionary mapping model names to weights
        """
        if not self.is_fitted_ or self.weights_ is None or self.model_names_ is None:
            raise RuntimeError("RidgePositionSizer must be fitted before get_weights")
        
        return {name: float(w) for name, w in zip(self.model_names_, self.weights_)}
    
    def get_position_sizes(
        self,
        model_preds: pd.DataFrame,
        timestamps: np.ndarray | None = None,
        k: float = 11.0,
        c0: float = 0.70,
        s_min: float = 0.03,
        s_max: float = 0.15,
    ) -> np.ndarray:
        """Compute position sizes using sigmoid sizing aligned with tpsl_optimiser.
        
        This method produces position sizes compatible with the tpsl_optimiser
        pipeline, using the sigmoid sizing function from 40_position_sizing_opt.py.
        
        Position sizing is done per time-slice (cross-sectionally) to avoid
        lookahead bias and match deployment conditions.
        
        Args:
            model_preds: DataFrame with model predictions
            timestamps: Array of timestamps for each prediction (required for
                       cross-sectional ranking)
            k: Sigmoid steepness parameter (default from tpsl_optimiser)
            c0: Sigmoid center parameter (default from tpsl_optimiser)
            s_min: Minimum position size (default 3%)
            s_max: Maximum position size (default 15%)
            
        Returns:
            Array of position sizes (fraction of capital)
        """
        # Get combined signal
        signal = self.predict(model_preds)
        
        # Compute confidence per time-slice (cross-sectional rank)
        if timestamps is not None and len(np.unique(timestamps)) > 1:
            # Cross-sectional ranking within each timestamp
            confidence = np.zeros(len(signal))
            ts_arr = np.asarray(timestamps)
            for ts in np.unique(ts_arr):
                mask = ts_arr == ts
                if mask.sum() > 1:
                    confidence[mask] = rankdata(signal[mask], method='average') / mask.sum()
                else:
                    confidence[mask] = 0.5  # Single asset = neutral confidence
        else:
            # No timestamps: use global rank (may have lookahead bias)
            confidence = rankdata(signal, method='average') / len(signal)
        
        # Sigmoid sizing (from 40_position_sizing_opt.py)
        sig = 1.0 / (1.0 + np.exp(-k * (confidence - c0)))
        pos_sizes = s_min + (s_max - s_min) * sig
        
        return pos_sizes
    
    def save(self, path: str | Path) -> None:
        """Save fitted model to disk.
        
        Args:
            path: Path to save the model
        """
        if not self.is_fitted_:
            raise RuntimeError("Cannot save unfitted model")
        
        save_dict = {
            'weights': self.weights_.tolist() if self.weights_ is not None else None,
            'model_names': self.model_names_,
            'best_params': self.best_params_,
            'best_target_name': getattr(self, 'best_target_name_', None),
            'gamma_range': self.gamma_range,
            'alpha_range': self.alpha_range,
            'delta_range': self.delta_range,
            'cost_pct': self.cost_pct,
            'sum_to_one': self.sum_to_one,
            'non_negative': self.non_negative,
            'top_k_pct': self.top_k_pct,
            'scaler_means': self.scaler_.means_.tolist() if self.scaler_ else None,
            'scaler_stds': self.scaler_.stds_.tolist() if self.scaler_ else None,
        }
        
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w') as f:
            json.dump(save_dict, f, indent=2)
        
        tprint(f"RidgePositionSizer saved to {path}")
    
    @classmethod
    def load(cls, path: str | Path) -> 'RidgePositionSizer':
        """Load fitted model from disk.
        
        Args:
            path: Path to load the model from
            
        Returns:
            Loaded RidgePositionSizer instance
        """
        with open(path, 'r') as f:
            save_dict = json.load(f)
        
        instance = cls(
            gamma_range=tuple(save_dict['gamma_range']),
            alpha_range=tuple(save_dict['alpha_range']),
            delta_range=tuple(save_dict.get('delta_range', (0.5, 2.0))),
            cost_pct=save_dict['cost_pct'],
            sum_to_one=save_dict['sum_to_one'],
            non_negative=save_dict['non_negative'],
            top_k_pct=save_dict.get('top_k_pct', 0.30),
        )
        
        instance.weights_ = np.array(save_dict['weights']) if save_dict['weights'] else None
        instance.model_names_ = save_dict['model_names']
        instance.best_params_ = save_dict['best_params']
        instance.is_fitted_ = True
        
        # Restore scaler
        if save_dict.get('scaler_means') is not None:
            instance.scaler_ = PredictionScaler()
            instance.scaler_.means_ = np.array(save_dict['scaler_means'])
            instance.scaler_.stds_ = np.array(save_dict['scaler_stds'])
        
        tprint(f"RidgePositionSizer loaded from {path}")
        
        return instance


# ═══════════════════════════════════════════════════════════════════════════════
# Pipeline Integration Function
# ═══════════════════════════════════════════════════════════════════════════════

def run_ridge_position_sizer_step(
    oof_preds: pd.DataFrame,
    trade_outcomes: pd.DataFrame,
    timestamps: np.ndarray | None = None,
    groups: np.ndarray | None = None,
    cfg: dict | None = None,
    save_model: bool = True,
    run_id: str | None = None,
    labels: np.ndarray | None = None,
    symbols: np.ndarray | None = None,
) -> Tuple[RidgePositionSizer, Dict]:
    """Run the ridge position sizer step in the pipeline.
    
    This function is designed to be called between 'train_meta' and 'optimise'
    steps. It takes OOF predictions from meta models and learns optimal
    combination weights for position sizing.
    
    IMPORTANT: oof_preds must contain TRUE out-of-fold predictions from the
    meta model training process, NOT in-sample predictions from predict().
    Using in-sample predictions will lead to overfitting and poor performance.
    
    Args:
        oof_preds: DataFrame with OOF predictions from meta models.
                  Either wide format (one column per model) or long format
                  with 'model_name' and 'pred' columns.
        trade_outcomes: DataFrame with trade outcomes. Can have:
                       - entry_price, exit_price, is_long columns, OR
                       - 'return' column with pre-computed log returns
        timestamps: Array of timestamps for each trade (STRONGLY RECOMMENDED
                   for proper time-based CV and drawdown computation)
        groups: Optional group labels for CV splits (e.g., day/week)
        cfg: Optional configuration dictionary with parameters:
            - gamma_range: Tuple for gamma parameter range
            - alpha_range: Tuple for alpha parameter range
            - delta_range: Tuple for delta parameter range
            - n_grid_points: Number of grid points for search
            - cost_pct: Transaction cost percentage
            - top_k_pct: Percentage of top predictions to select
        save_model: If True, save the fitted model to disk
        run_id: Optional run ID for saving
        labels: Optional pre-computed labels (log returns). If provided,
               trade_outcomes entry/exit prices are not needed.
        
    Returns:
        Tuple of (fitted RidgePositionSizer, metrics dict)
    """
    tprint("=" * 80)
    tprint("RIDGE POSITION SIZER STEP")
    tprint("=" * 80)
    
    # Extract configuration
    cfg = cfg or {}
    gamma_range = cfg.get('gamma_range', (1.0, 3.0))
    alpha_range = cfg.get('alpha_range', (1e-4, 1e-1))
    delta_range = cfg.get('delta_range', (0.5, 2.0))
    n_grid_points = cfg.get('n_grid_points', 10)
    cost_pct = cfg.get('cost_pct', 0.0005)
    top_k_pct = cfg.get('top_k_pct', 0.30)
    
    # Initialize sizer
    sizer = RidgePositionSizer(
        gamma_range=gamma_range,
        alpha_range=alpha_range,
        delta_range=delta_range,
        n_grid_points=n_grid_points,
        cost_pct=cost_pct,
        top_k_pct=top_k_pct,
    )
    
    # Fit
    sizer.fit(oof_preds, trade_outcomes, timestamps=timestamps, groups=groups,
              labels=labels, symbols=symbols)
    
    # Compute metrics
    weights = sizer.get_weights()
    metrics = {
        'weights': weights,
        'best_params': sizer.best_params_,
        'best_target_name': getattr(sizer, 'best_target_name_', None),
        'n_models': len(weights),
        'n_trades': len(trade_outcomes),
    }
    
    # Add CV results summary
    if sizer.cv_results_ is not None:
        best_idx = sizer.cv_results_['J_zscore'].idxmax()
        best_row = sizer.cv_results_.loc[best_idx]
        metrics['cv_best_pnl_total'] = float(best_row['PnL_total'])
        metrics['cv_best_sortino'] = float(best_row['Sortino'])
        metrics['cv_best_maxdd'] = float(best_row['MaxDD'])
        metrics['cv_best_ic'] = float(best_row['IC'])
        metrics['cv_best_winrate'] = float(best_row['WinRate'])
        metrics['cv_best_n_selected'] = int(best_row['N_selected'])
    
    # Save model
    if save_model:
        run_id = run_id or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        model_dir = Path("extreme_price_movements/models")
        model_dir.mkdir(parents=True, exist_ok=True)
        model_path = model_dir / f"ridge_position_sizer_{run_id}.json"
        sizer.save(model_path)
        metrics['model_path'] = str(model_path)
    
    # Save CV results
    if sizer.cv_results_ is not None:
        reports_dir = Path("extreme_price_movements/reports")
        reports_dir.mkdir(parents=True, exist_ok=True)
        cv_path = reports_dir / f"ridge_position_sizer_cv_{run_id or 'latest'}.csv"
        sizer.cv_results_.to_csv(cv_path, index=False)
        metrics['cv_results_path'] = str(cv_path)
    
    tprint("-" * 80)
    tprint("Ridge Position Sizer Results:")
    tprint(f"  Models combined: {len(weights)}")
    for name, w in weights.items():
        tprint(f"    {name}: {w:.4f}")
    tprint(f"  Best hyperparameters:")
    if sizer.best_params_:
        tprint(f"    alpha: {sizer.best_params_['alpha']:.6f}")
        tprint(f"    delta: {sizer.best_params_['delta']:.3f}")
        tprint(f"    gamma: {sizer.best_params_['gamma']:.3f}")
    tprint("=" * 80)
    
    return sizer, metrics


# ═══════════════════════════════════════════════════════════════════════════════
# Convenience Functions for Integration
# ═══════════════════════════════════════════════════════════════════════════════

def load_meta_oof_predictions(data_root: str, run_id: str) -> pd.DataFrame:
    """Load meta model OOF predictions from a training run.
    
    Args:
        data_root: Root directory for data
        run_id: Training run identifier
        
    Returns:
        DataFrame with OOF predictions (one column per model) and trade context
        
    Raises:
        FileNotFoundError: If meta OOF directory doesn't exist
    """
    from pathlib import Path
    
    meta_oof_dir = Path(data_root) / "artifacts" / run_id / "meta_oof"
    
    if not meta_oof_dir.exists():
        raise FileNotFoundError(f"No meta OOF directory at {meta_oof_dir}")
    
    oof_dfs = {}
    for parquet_file in meta_oof_dir.glob("meta_oof_*.parquet"):
        model_name = parquet_file.stem.replace("meta_oof_", "")
        df = pd.read_parquet(parquet_file)
        oof_dfs[model_name] = df
    
    if not oof_dfs:
        raise FileNotFoundError(f"No meta OOF parquet files found in {meta_oof_dir}")
    
    # Combine into wide format DataFrame for predictions
    pred_cols = {name: df["oof_pred"].values for name, df in oof_dfs.items()}
    result = pd.DataFrame(pred_cols)
    
    # Attach metadata from first model (all should have same index/timestamp/symbol)
    first_df = list(oof_dfs.values())[0]
    for col in ["timestamp", "symbol", "return", "is_long", "index"]:
        if col in first_df.columns:
            result[col] = first_df[col].values
    
    tprint(f"Loaded OOF predictions from {len(oof_dfs)} models: {list(oof_dfs.keys())}")
    return result


def load_trade_outcomes_from_oof(data_root: str, run_id: str, oof_df: pd.DataFrame) -> pd.DataFrame:
    """Load or construct trade outcomes from OOF predictions data.
    
    Args:
        data_root: Root directory for data
        run_id: Training run identifier
        oof_df: DataFrame with OOF predictions and trade context
        
    Returns:
        DataFrame with columns [return, is_long] and optionally [timestamp, symbol]
    """
    # Check if we have the return column directly in OOF data
    if "return" in oof_df.columns:
        outcomes = pd.DataFrame({
            "return": oof_df["return"].values,
            "is_long": oof_df["is_long"].values if "is_long" in oof_df.columns else 1,
        })
        if "timestamp" in oof_df.columns:
            outcomes["timestamp"] = oof_df["timestamp"].values
        if "symbol" in oof_df.columns:
            outcomes["symbol"] = oof_df["symbol"].values
        tprint(f"Constructed trade outcomes from OOF context: {len(outcomes)} trades")
        return outcomes
    
    raise FileNotFoundError(
        f"No trade outcomes found. The OOF predictions must include 'return' column."
    )


def load_ridge_sizer_weights(data_root: str, run_id: str) -> Optional[Dict]:
    """Load ridge position sizer weights from a training run.
    
    Args:
        data_root: Root directory for data
        run_id: Training run identifier
        
    Returns:
        Dict with weights and best_params, or None if not found
    """
    from pathlib import Path
    
    weights_path = Path(data_root) / "artifacts" / run_id / "ridge_sizer" / "sizer_weights.json"
    
    if not weights_path.exists():
        return None
    
    with open(weights_path, 'r') as f:
        return json.load(f)


def prepare_trade_outcomes_from_labels(
    df: pd.DataFrame,
    entry_col: str = 'entry_price',
    exit_col: str = 'exit_price',
    is_long_col: str = 'is_long',
) -> pd.DataFrame:
    """Prepare trade outcomes DataFrame from labeled data.
    
    Args:
        df: DataFrame with trade information
        entry_col: Column name for entry prices
        exit_col: Column name for exit prices
        is_long_col: Column name for direction indicator
        
    Returns:
        DataFrame with required columns for RidgePositionSizer
    """
    required_cols = [entry_col, exit_col, is_long_col]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    
    return df[required_cols].rename(columns={
        entry_col: 'entry_price',
        exit_col: 'exit_price',
        is_long_col: 'is_long',
    })
