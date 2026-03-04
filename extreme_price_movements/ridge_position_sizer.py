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
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import rankdata, spearmanr
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

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
from extreme_price_movements.path_utils import resolve_reports_dir, resolve_data_root
from extreme_price_movements.tpsl_optimiser.metrics_utils import compute_comprehensive_metrics
from extreme_price_movements.label_policy_optimizer import optimize_label_policy
from extreme_price_movements.elasticnet_feature_selection import run_fold_safe_feature_pruning_and_elasticnet


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

@jit(nopython=True, nogil=True, cache=True)
def simulate_trade_exit(
    highs: np.ndarray,
    lows: np.ndarray,
    opens: np.ndarray,
    closes: np.ndarray,
    entry_price: float,
    is_long: bool,
    tp_price: float,
    sl_price: float,
    trailing_pct: float,
    max_bars: int,
) -> Tuple[float, int, int]:
    """Simulate one trade exit with TP/SL/trailing barriers and timeout.

    Returns `(exit_price, exit_bar, exit_reason)` where reasons are:
    `0=TP`, `1=SL`, `2=Trailing`, `3=Timeout`.
    """
    peak = entry_price
    trough = entry_price
    
    for bar in range(max_bars):
        h = highs[bar]
        l = lows[bar]
        o = opens[bar]
        c = closes[bar]
        
        # Check for NaN (synthetic padded data) - force timeout
        if np.isnan(h) or np.isnan(l) or np.isnan(o) or np.isnan(c):
            # Return at the last valid close
            for prev_bar in range(bar - 1, -1, -1):
                if not np.isnan(closes[prev_bar]):
                    return closes[prev_bar], prev_bar, 3
            return entry_price, 0, 3  # No valid data
        
        if is_long:
            # Update peak for trailing calculation
            if h > peak:
                peak = h
            
            # Check triggered exits in this bar
            tp_hit = h >= tp_price
            sl_hit = l <= sl_price
            # Check trailing exit (only if we have profit)
            trailing_hit = False
            trailing_price = 0.0
            if peak > entry_price:
                trailing_price = peak * (1.0 - trailing_pct)
                trailing_hit = l <= trailing_price

            if tp_hit or sl_hit or trailing_hit:
                # Resolve collisions by CLOSE proximity to barrier.
                # Deterministic tie-break: worst outcome first (SL > TRAILING > TP).
                best_price = c
                best_reason = 3
                best_dist = 1e100
                best_rank = 10

                if sl_hit:
                    d = abs(o - sl_price)
                    if d < best_dist or (d == best_dist and 0 < best_rank):
                        best_price, best_reason, best_dist, best_rank = sl_price, 1, d, 0

                if trailing_hit:
                    d = abs(o - trailing_price)
                    if d < best_dist or (d == best_dist and 1 < best_rank):
                        best_price, best_reason, best_dist, best_rank = trailing_price, 2, d, 1

                if tp_hit:
                    d = abs(o - tp_price)
                    if d < best_dist or (d == best_dist and 2 < best_rank):
                        best_price, best_reason, best_dist, best_rank = tp_price, 0, d, 2

                return best_price, bar, best_reason
        else:
            # Short position logic
            # Update trough for trailing calculation
            if l < trough:
                trough = l
            
            # Check triggered exits in this bar
            tp_hit = l <= tp_price
            sl_hit = h >= sl_price
            # Check trailing exit (only if we have profit)
            trailing_hit = False
            trailing_price = 0.0
            if trough < entry_price:
                trailing_price = trough * (1.0 + trailing_pct)
                trailing_hit = h >= trailing_price

            if tp_hit or sl_hit or trailing_hit:
                best_price = c
                best_reason = 3
                best_dist = 1e100
                best_rank = 10

                if sl_hit:
                    d = abs(o - sl_price)
                    if d < best_dist or (d == best_dist and 0 < best_rank):
                        best_price, best_reason, best_dist, best_rank = sl_price, 1, d, 0

                if trailing_hit:
                    d = abs(o - trailing_price)
                    if d < best_dist or (d == best_dist and 1 < best_rank):
                        best_price, best_reason, best_dist, best_rank = trailing_price, 2, d, 1

                if tp_hit:
                    d = abs(o - tp_price)
                    if d < best_dist or (d == best_dist and 2 < best_rank):
                        best_price, best_reason, best_dist, best_rank = tp_price, 0, d, 2

                return best_price, bar, best_reason
    
    # Timeout - exit at close of last bar
    return closes[max_bars - 1], max_bars - 1, 3


@jit(nopython=True, parallel=True, cache=True)
def simulate_trade_exit_batch(
    highs: np.ndarray,
    lows: np.ndarray,
    opens: np.ndarray,
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
        opens: 2D array of shape (n_trades, max_bars) with future open prices
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
            highs[i], lows[i], opens[i], closes[i],
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


def _simulate_policy_utility_from_arrays(
    entry_price: float,
    is_long: bool,
    future_opens: np.ndarray,
    future_highs: np.ndarray,
    future_lows: np.ndarray,
    future_closes: np.ndarray,
    tp_pct: float,
    sl_pct: float,
    trailing_pct: float,
    max_bars: int,
    cost_pct: float,
) -> float:
    """Compute policy utility using the shared `simulate_trade_exit` implementation."""
    if is_long:
        tp_price = entry_price * (1.0 + tp_pct)
        sl_price = entry_price * (1.0 - sl_pct)
    else:
        tp_price = entry_price * (1.0 - tp_pct)
        sl_price = entry_price * (1.0 + sl_pct)

    exit_price, _, _ = simulate_trade_exit(
        highs=np.asarray(future_highs, dtype=np.float64),
        lows=np.asarray(future_lows, dtype=np.float64),
        opens=np.asarray(future_opens, dtype=np.float64),
        closes=np.asarray(future_closes, dtype=np.float64),
        entry_price=float(entry_price),
        is_long=bool(is_long),
        tp_price=float(tp_price),
        sl_price=float(sl_price),
        trailing_pct=float(trailing_pct),
        max_bars=int(max_bars),
    )
    if is_long:
        ret = np.log(max(exit_price, 1e-12) / max(entry_price, 1e-12))
    else:
        ret = np.log(max(entry_price, 1e-12) / max(exit_price, 1e-12))
    return float(ret - cost_pct)


def compute_optimal_limit_offset_labels(
    trade_outcomes: pd.DataFrame,
    tick_size: float = 0.1,
    k_max: int = 5,
    entry_fill_horizon_bars: int = 4,
    max_hold_bars: int = 48,
    tp_pct: float = 0.005,
    sl_pct: float = 0.0025,
    trailing_pct: float = 0.0,
    cost_pct: float = 0.002,
    eta: float = 0.0,
    tie_break_smallest_k: bool = True,
) -> Optional[np.ndarray]:
    """Build k* labels via discrete argmax utility over limit offsets (0..k_max).

    Uses shared `simulate_trade_exit` through `_simulate_policy_utility_from_arrays`.
    Returns None when required path/price columns are unavailable.
    """
    req_cols = {"entry_price", "is_long", "future_opens", "future_highs", "future_lows", "future_closes"}
    if not req_cols.issubset(set(trade_outcomes.columns)):
        return None

    k_labels = np.zeros(len(trade_outcomes), dtype=float)
    for i, row in enumerate(trade_outcomes.itertuples(index=False)):
        entry_price = float(getattr(row, "entry_price"))
        is_long = bool(getattr(row, "is_long"))
        opens = np.asarray(getattr(row, "future_opens"), dtype=float)
        highs = np.asarray(getattr(row, "future_highs"), dtype=float)
        lows = np.asarray(getattr(row, "future_lows"), dtype=float)
        closes = np.asarray(getattr(row, "future_closes"), dtype=float)
        if len(opens) == 0 or len(highs) == 0 or len(lows) == 0 or len(closes) == 0:
            k_labels[i] = 0.0
            continue
        h_fill = min(int(entry_fill_horizon_bars), len(highs))
        best_k = 0
        best_u = -1e18
        # If label-policy params are attached by label_policy_optimizer, consume them.
        row_sl_atr_mult = float(getattr(row, "label_policy_sl_atr_mult", np.nan))
        row_tp_sl_ratio = float(getattr(row, "label_policy_tp_sl_ratio", np.nan))
        row_max_hold_bars = int(getattr(row, "label_policy_max_hold_bars", max_hold_bars))
        row_giveback_pct = float(getattr(row, "label_policy_giveback_pct", trailing_pct))

        if np.isfinite(row_sl_atr_mult) and np.isfinite(row_tp_sl_ratio):
            atr_entry = float(getattr(row, "atr_12_15m", 0.0))
            sl_abs = max(row_sl_atr_mult * max(atr_entry, 1e-9), 1e-9)
            tp_abs = row_tp_sl_ratio * sl_abs
            eff_sl_pct = sl_abs / max(entry_price, 1e-9)
            eff_tp_pct = tp_abs / max(entry_price, 1e-9)
        else:
            eff_sl_pct = sl_pct
            eff_tp_pct = tp_pct

        for k in range(int(k_max) + 1):
            limit_price = entry_price - tick_size * k if is_long else entry_price + tick_size * k
            if is_long:
                hit_idx = np.where(lows[:h_fill] <= limit_price)[0]
            else:
                hit_idx = np.where(highs[:h_fill] >= limit_price)[0]
            if hit_idx.size == 0:
                u = 0.0
            else:
                fill_i = int(hit_idx[0])
                o2 = opens[fill_i: fill_i + max_hold_bars]
                h2 = highs[fill_i: fill_i + max_hold_bars]
                l2 = lows[fill_i: fill_i + max_hold_bars]
                c2 = closes[fill_i: fill_i + max_hold_bars]
                u = _simulate_policy_utility_from_arrays(
                    entry_price=limit_price,
                    is_long=is_long,
                    future_opens=o2,
                    future_highs=h2,
                    future_lows=l2,
                    future_closes=c2,
                    tp_pct=eff_tp_pct,
                    sl_pct=eff_sl_pct,
                    trailing_pct=row_giveback_pct,
                    max_bars=max(1, min(row_max_hold_bars, len(h2))),
                    cost_pct=cost_pct,
                ) - eta * float(fill_i + 1)
            if (u > best_u) or (u == best_u and ((k < best_k) if tie_break_smallest_k else (k > best_k))):
                best_u = u
                best_k = k
        k_labels[i] = float(best_k)
    return k_labels


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
            end_idx = min(entry_idx + max_bars, len(opens))
            
            if entry_idx >= end_idx:
                continue
            
            # Get arrays for this trade
            future_opens = opens[symbol].iloc[entry_idx:end_idx].values.astype(np.float64)
            future_highs = highs[symbol].iloc[entry_idx:end_idx].values.astype(np.float64)
            future_lows = lows[symbol].iloc[entry_idx:end_idx].values.astype(np.float64)
            future_closes = closes[symbol].iloc[entry_idx:end_idx].values.astype(np.float64)
            
            actual_bars = len(future_highs)
            if actual_bars == 0:
                continue
            
            # Run simulator
            exit_price, exit_bar, exit_reason_int = simulate_trade_exit(
                future_highs, future_lows, future_opens, future_closes,
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

    Uses the parallel Numba simulator and preallocated arrays to reduce
    Python overhead and transient memory allocations.
    """
    tp_mult = policy_params.get('tp_mult', 3.0)
    sl_mult = policy_params.get('sl_mult', 1.0)
    trailing_pct = policy_params.get('trailing_pct', 0.5)
    atr_dict = policy_params.get('atr', {})

    max_bars = max_hold_hours * bars_per_hour

    opens = price_panel.get('open')
    highs = price_panel.get('high')
    lows = price_panel.get('low')
    closes = price_panel.get('close')

    if opens is None or highs is None or lows is None or closes is None:
        raise ValueError("price_panel must contain 'open', 'high', 'low', 'close' DataFrames")

    candidates_df = candidates_df.copy()
    if not pd.api.types.is_datetime64_any_dtype(candidates_df['timestamp']):
        candidates_df['timestamp'] = pd.to_datetime(candidates_df['timestamp'])

    n_candidates = len(candidates_df)
    if n_candidates == 0:
        return pd.DataFrame()

    ts_values = pd.to_datetime(candidates_df['timestamp'])
    symbol_values = candidates_df['symbol'].to_numpy()
    is_long_values = candidates_df['is_long'].to_numpy(dtype=bool)
    entry_price_values = np.asarray(candidates_df['entry_price'].to_numpy(), dtype=np.float64)

    # Pre-index timestamps once (monotonic index expected for price panels).
    price_index = opens.index
    left_idx = price_index.searchsorted(ts_values, side='left')

    # Pre-allocate dense blocks once to avoid list append + np.array conversion.
    entry_prices_arr = np.empty(n_candidates, dtype=np.float64)
    is_longs_arr = np.empty(n_candidates, dtype=np.int64)
    tp_prices_arr = np.empty(n_candidates, dtype=np.float64)
    sl_prices_arr = np.empty(n_candidates, dtype=np.float64)
    trailing_pcts_arr = np.full(n_candidates, float(trailing_pct), dtype=np.float64)
    opens_arr = np.full((n_candidates, max_bars), np.nan, dtype=np.float64)
    highs_arr = np.full((n_candidates, max_bars), np.nan, dtype=np.float64)
    lows_arr = np.full((n_candidates, max_bars), np.nan, dtype=np.float64)
    closes_arr = np.full((n_candidates, max_bars), np.nan, dtype=np.float64)
    entry_indices = np.empty(n_candidates, dtype=np.int64)
    actual_bars_arr = np.empty(n_candidates, dtype=np.int64)

    timestamps: List[pd.Timestamp] = []
    symbols: List[str] = []
    valid_count = 0

    # Cache symbol vectors to avoid repeated pandas indexing in the hot loop.
    symbol_cache: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}

    for i in range(n_candidates):
        ts = ts_values[i]
        symbol = symbol_values[i]
        is_long = bool(is_long_values[i])
        entry_price = float(entry_price_values[i])

        if isinstance(atr_dict, dict):
            atr = atr_dict.get(symbol, 0.02)
        elif isinstance(atr_dict, pd.Series):
            atr = atr_dict.get(symbol, 0.02)
        else:
            atr = 0.02

        if is_long:
            tp_price = entry_price * (1 + tp_mult * atr)
            sl_price = entry_price * (1 - sl_mult * atr)
        else:
            tp_price = entry_price * (1 - tp_mult * atr)
            sl_price = entry_price * (1 + sl_mult * atr)

        entry_idx = int(left_idx[i])
        if entry_idx >= len(price_index):
            continue

        try:
            if symbol not in symbol_cache:
                symbol_cache[symbol] = (
                    np.asarray(opens[symbol].values, dtype=np.float64),
                    np.asarray(highs[symbol].values, dtype=np.float64),
                    np.asarray(lows[symbol].values, dtype=np.float64),
                    np.asarray(closes[symbol].values, dtype=np.float64),
                )
            open_vec, high_vec, low_vec, close_vec = symbol_cache[symbol]

            end_idx = min(entry_idx + max_bars, len(open_vec))
            if entry_idx >= end_idx:
                continue

            actual_bars = int(end_idx - entry_idx)
            if actual_bars <= 0:
                continue

            slot = valid_count
            entry_prices_arr[slot] = entry_price
            is_longs_arr[slot] = int(is_long)
            tp_prices_arr[slot] = tp_price
            sl_prices_arr[slot] = sl_price
            entry_indices[slot] = entry_idx
            actual_bars_arr[slot] = actual_bars

            opens_arr[slot, :actual_bars] = open_vec[entry_idx:end_idx]
            highs_arr[slot, :actual_bars] = high_vec[entry_idx:end_idx]
            lows_arr[slot, :actual_bars] = low_vec[entry_idx:end_idx]
            closes_arr[slot, :actual_bars] = close_vec[entry_idx:end_idx]

            timestamps.append(ts)
            symbols.append(symbol)
            valid_count += 1

        except (KeyError, IndexError):
            continue

    if valid_count == 0:
        return pd.DataFrame()

    entry_prices_arr = entry_prices_arr[:valid_count]
    is_longs_arr = is_longs_arr[:valid_count]
    tp_prices_arr = tp_prices_arr[:valid_count]
    sl_prices_arr = sl_prices_arr[:valid_count]
    trailing_pcts_arr = trailing_pcts_arr[:valid_count]
    opens_arr = opens_arr[:valid_count]
    highs_arr = highs_arr[:valid_count]
    lows_arr = lows_arr[:valid_count]
    closes_arr = closes_arr[:valid_count]
    entry_indices = entry_indices[:valid_count]
    actual_bars_arr = actual_bars_arr[:valid_count]

    exit_prices, exit_bars, exit_reasons = simulate_trade_exit_batch(
        highs_arr, lows_arr, opens_arr, closes_arr,
        entry_prices_arr, is_longs_arr,
        tp_prices_arr, sl_prices_arr, trailing_pcts_arr,
        max_bars,
    )

    exit_reason_map = {
        0: ExitReason.TP_HIT,
        1: ExitReason.SL_HIT,
        2: ExitReason.TRAILING_EXIT,
        3: ExitReason.TIMEOUT,
    }

    results = []
    for i in range(valid_count):
        is_long = bool(is_longs_arr[i])
        exit_price = float(exit_prices[i])
        entry_price = float(entry_prices_arr[i])
        exit_bar = int(exit_bars[i])
        exit_reason_int = int(exit_reasons[i])

        entry_idx = int(entry_indices[i])
        actual_bars = int(actual_bars_arr[i])
        clamped_bar = min(exit_bar, actual_bars - 1)
        exit_time = price_index[min(entry_idx + clamped_bar, len(price_index) - 1)]

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
            'tp_price': float(tp_prices_arr[i]),
            'sl_price': float(sl_prices_arr[i]),
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


def make_confidence_conditional_regression_objective(
    alpha=3.0,          # boosts error penalty in high-confidence region
    threshold=0.0,      # location of confidence boundary in prediction space
    temperature=0.5,    # softness of transition (larger = smoother)
    lambda_conf=0.005,  # >0 encourages coverage/confidence (prevents "all low confidence")
    hess_floor=1e-6,    # numerical stability
    use_magnitude=False,# if True, gate on |prediction| instead of prediction
    eps=1e-6
):
    """
    Two-term objective aligned with "only trust predictions when confident":

      L_i = 0.5 * w(pred) * (pred - y)^2  -  lambda_conf * g(pred)

    where:
      g(pred) in [0,1] is a smooth "confidence gate"
      w(pred) = 1 + alpha * g(pred)
    """

    def sigmoid(x):
        # numerically stable sigmoid
        x = np.clip(x, -60.0, 60.0)
        return 1.0 / (1.0 + np.exp(-x))

    def objective(y_true, y_pred):
        y_true = np.asarray(y_true, dtype=np.float64)
        y_pred = np.asarray(y_pred, dtype=np.float64)

        e = y_pred - y_true  # error

        # ----- gate g(pred) and its derivatives g', g'' -----
        if use_magnitude:
            # smooth |pred|
            a = np.sqrt(y_pred**2 + eps)
            z = (a - threshold) / temperature
            s = sigmoid(z)
            t = s * (1.0 - s)

            a_prime = y_pred / a
            a_second = eps / (a**3)

            z_prime = a_prime / temperature
            z_second = a_second / temperature

            g = s
            g_prime = t * z_prime
            g_second = t * (1.0 - 2.0 * s) * (z_prime**2) + t * z_second
        else:
            z = (y_pred - threshold) / temperature
            s = sigmoid(z)
            t = s * (1.0 - s)

            g = s
            g_prime = t / temperature
            g_second = t * (1.0 - 2.0 * s) / (temperature**2)

        # ----- confidence-weighted squared loss term -----
        w = 1.0 + alpha * g
        w_prime = alpha * g_prime
        w_second = alpha * g_second

        # L1 = 0.5*w*e^2
        grad = w * e + 0.5 * w_prime * (e**2)
        hess = w + 2.0 * w_prime * e + 0.5 * w_second * (e**2)

        # ----- confidence reward term -----
        if lambda_conf != 0.0:
            grad -= lambda_conf * g_prime
            hess -= lambda_conf * g_second

        # keep Hessian positive for XGBoost
        hess = np.maximum(hess, hess_floor)

        return grad, hess

    return objective


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
    min_hist = min(20, max(5, window // 5))  # Require at least 5 to 20 samples depending on window
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
    min_periods = min(20, max(5, baseline_window // 5))
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
    min_periods = min(20, max(5, baseline_window // 5))
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
    select_metric: str = "topq_u_policy",
    topq: float = 0.30,
    u_policy: np.ndarray | None = None,
    require_positive_topq_u: bool = True,
    topq_min_samples: int = 50,
    trade_mask: np.ndarray | None = None,
    tree_probe: bool = False,
) -> tuple:
    """Race target representations for Ridge position sizer.

    Builds candidate targets, fits Ridge (alpha fixed) on each via 3-fold
    purged CV, picks the one with highest OOF Spearman IC.

    Returns:
        (best_name, best_y, race_log): winning target name, array, and log lines
    """
    race_log = []
    n = len(returns)

    # Build net returns version for candidate generation (hurdle-centered)
    y_net = returns - cost_pct
    
    # Build vol proxy if symbols available
    vol = None
    if symbols is not None and len(np.unique(symbols)) > 1:
        vol = build_trade_vol_proxy(y_net, symbols, timestamps)

    # Ensure symbols is usable (fallback to single symbol)
    if symbols is None:
        symbols = np.full(n, "ALL", dtype=object)

    # Build candidate targets
    candidates = {}

    # 1. Winsorized log return
    candidates["winsorized"] = build_target_winsorized(y_net, clip_L=clip_L)
    if vol is not None:
        candidates["winsorized_voladj"] = build_target_winsorized(
            y_net, clip_L=clip_L, vol=vol, vol_mode="partial")

    # 2. Huber advantage
    candidates["huber_adv"] = build_target_huber_advantage(
        y_net, symbols, delta=clip_L)
    if vol is not None:
        candidates["huber_adv_voladj"] = build_target_huber_advantage(
            y_net, symbols, delta=clip_L, vol=vol, vol_mode="partial")

    # 3. Rolling rank
    candidates["rolling_rank"] = build_target_rolling_rank(y_net, symbols)
    if vol is not None:
        candidates["rolling_rank_voladj"] = build_target_rolling_rank(
            y_net, symbols, vol=vol, vol_mode="partial")

    # 4. Rolling rank residual
    candidates["rank_residual"] = build_target_rolling_rank_residual(y_net, symbols)
    if vol is not None:
        candidates["rank_residual_voladj"] = build_target_rolling_rank_residual(
            y_net, symbols, vol=vol, vol_mode="partial")

    race_log.append(f"    Ridge target race: {len(candidates)} candidates")

    # Evaluate each candidate via Ridge CV
    from sklearn.linear_model import Ridge as SkRidge

    best_ic = -np.inf
    best_name = "winsorized"
    best_y = candidates["winsorized"]
    candidate_rows: List[Dict[str, Any]] = []

    for tname, y_cand in candidates.items():
        fin = np.isfinite(y_cand)
        if fin.sum() < 100:
            race_log.append(f"      {tname}: skipped (only {fin.sum()} finite)")
            continue

        # Simple 3-fold walk-forward CV with Ridge
        fold_size = n // 3
        ics = []
        oof_pred = np.full(n, np.nan, dtype=float)
        fold_topq_u_list = []
        fold_prec_list = []
        fold_lift_list = []

        for fold in range(3):
            val_start = fold * fold_size
            val_end = val_start + fold_size if fold < 2 else n
            # Train on everything before val_start (walk-forward)
            train_end = max(0, val_start - 12)  # 12-bar purge
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
                oof_pred[va_idx] = pred
                # Diagnostic IC (always computed/logged)
                y_va_ret = returns[va_idx]
                if np.std(pred) < 1e-12 or np.std(y_va_ret) < 1e-12:
                    ic_fold = 0.0
                else:
                    ic_fold = float(spearmanr(pred, y_va_ret).correlation)
                if np.isfinite(ic_fold):
                    ics.append(ic_fold)

                # Track per-fold TopQ utility for stability penalty
                if select_metric == "topq_u_policy" and u_policy is not None:
                    up = np.asarray(u_policy, dtype=float)
                    _tm = np.ones(len(oof_pred), dtype=bool) if trade_mask is None else np.asarray(trade_mask[:len(oof_pred)], dtype=bool)

                    # Compute only on valid current fold indices
                    valid_fold = np.isfinite(pred) & np.isfinite(up[va_idx]) & _tm[va_idx]
                    if np.any(valid_fold):
                        pred_fold_valid = pred[valid_fold]
                        up_fold_valid = up[va_idx][valid_fold]
                        k_top = max(1, int(np.ceil(float(topq) * len(pred_fold_valid))))
                        idx_top = np.argsort(pred_fold_valid)[-k_top:]
                        fold_topq_u_list.append(float(np.mean(up_fold_valid[idx_top])))

                        # Compute precision@30 and lift@30
                        ret_fold_valid = returns[va_idx][valid_fold]
                        overall_prec = float(np.mean(ret_fold_valid > 0))
                        top_prec = float(np.mean(ret_fold_valid[idx_top] > 0))
                        fold_prec_list.append(top_prec)
                        fold_lift_list.append(top_prec / max(overall_prec, 1e-9))
            except Exception:
                continue

        mean_ic = float(np.mean(ics)) if ics else -1.0

        # Primary selector: top-q realized policy utility
        topq_u = float("nan")
        topq_u_std = float("nan")
        topq_n = 0
        prec_30 = float("nan")
        lift_30 = float("nan")
        pass_gate = True

        if select_metric == "topq_u_policy":
            if u_policy is None:
                raise ValueError("u_policy is required when select_metric='topq_u_policy'")
            up = np.asarray(u_policy, dtype=float)
            _tm = np.ones(len(oof_pred), dtype=bool) if trade_mask is None else np.asarray(trade_mask[:len(oof_pred)], dtype=bool)
            mask = np.isfinite(oof_pred) & np.isfinite(up[:len(oof_pred)]) & _tm
            if np.any(mask):
                pred_use = oof_pred[mask]
                up_use = up[:len(oof_pred)][mask]
                ret_use = returns[:len(oof_pred)][mask]

                k_top = max(1, int(np.ceil(float(topq) * len(pred_use))))
                idx_top = np.argsort(pred_use)[-k_top:]

                # Global TopQ metrics
                topq_u = float(np.mean(up_use[idx_top]))
                topq_n = int(k_top)

                # Fold stability penalty
                if len(fold_topq_u_list) >= 2:
                    topq_u_std = float(np.std(fold_topq_u_list))
                else:
                    topq_u_std = 0.0

                # Precision & Lift
                overall_prec = float(np.mean(ret_use > 0))
                prec_30 = float(np.mean(ret_use[idx_top] > 0))
                lift_30 = float(prec_30 / max(overall_prec, 1e-9))

                pass_gate = (topq_n >= int(topq_min_samples)) and ((topq_u > 0.0) if bool(require_positive_topq_u) else True)

        # Composite score: mean - lambda * std
        composite_score = float("-inf")
        if pass_gate and select_metric == "topq_u_policy" and np.isfinite(topq_u):
            stability_penalty_lambda = 0.5
            composite_score = topq_u - (stability_penalty_lambda * topq_u_std if np.isfinite(topq_u_std) else 0.0)
        elif not pass_gate:
            composite_score = -1e12

        race_log.append(
            f"      {tname}: IC={mean_ic:.4f} TopQMeanU={topq_u:.6f} (std={topq_u_std:.6f}) "
            f"score={composite_score:.6f} P@30={prec_30:.3f} Lift@30={lift_30:.3f} gate={pass_gate}"
        )
        candidate_rows.append(
            {
                "target_name": tname,
                "ridge_ic": float(mean_ic),
                "ridge_topq_u": float(topq_u) if np.isfinite(topq_u) else float("nan"),
                "ridge_topq_u_std": float(topq_u_std) if np.isfinite(topq_u_std) else float("nan"),
                "ridge_composite_score": float(composite_score),
                "precision_at_30": float(prec_30) if np.isfinite(prec_30) else float("nan"),
                "lift_at_30": float(lift_30) if np.isfinite(lift_30) else float("nan"),
                "ridge_topq_n": int(topq_n),
                "ridge_pass_gate": bool(pass_gate),
            }
        )

        score = composite_score if select_metric == "topq_u_policy" else mean_ic
        if not np.isfinite(score):
            score = -1e12
        if score > best_ic:
            best_ic = score
            best_name = tname
            best_y = y_cand

    winner_model_metrics: Dict[str, Any] = {}
    if False and tree_probe:
        def _eval_model_cv(model_name: str, model_factory):
            fold_size = n // 3
            ics_loc = []
            oof_pred = np.full(n, np.nan, dtype=float)
            for fold in range(3):
                val_start = fold * fold_size
                val_end = val_start + fold_size if fold < 2 else n
                train_end = max(0, val_start - 12)
                if train_end < 50:
                    continue
                tr_idx = np.arange(0, train_end)
                va_idx = np.arange(val_start, val_end)
                X_tr, X_va = X[tr_idx], X[va_idx]
                y_tr = best_y[tr_idx]
                y_va_ret = returns[va_idx]
                mu = np.nanmean(X_tr, axis=0)
                sd = np.nanstd(X_tr, axis=0)
                sd = np.where(sd < 1e-9, 1.0, sd)
                X_tr_s = (X_tr - mu) / sd
                X_va_s = (X_va - mu) / sd
                try:
                    mdl = model_factory()
                    mdl.fit(X_tr_s, y_tr)
                    pred = np.asarray(mdl.predict(X_va_s), dtype=float)
                    oof_pred[va_idx] = pred
                    if np.std(pred) < 1e-12 or np.std(y_va_ret) < 1e-12:
                        ic_fold = 0.0
                    else:
                        ic_fold = float(spearmanr(pred, y_va_ret).correlation)
                    if np.isfinite(ic_fold):
                        ics_loc.append(ic_fold)
                except Exception:
                    continue
            mean_ic_loc = float(np.mean(ics_loc)) if ics_loc else float("nan")
            topq_u_loc = float("nan")
            topq_n_loc = 0
            if u_policy is not None:
                up = np.asarray(u_policy, dtype=float)
                _tm = np.ones(len(oof_pred), dtype=bool) if trade_mask is None else np.asarray(trade_mask[:len(oof_pred)], dtype=bool)
                mask = np.isfinite(oof_pred) & np.isfinite(up[:len(oof_pred)]) & _tm
                if np.any(mask):
                    pred_use = oof_pred[mask]
                    up_use = up[:len(oof_pred)][mask]
                    k_top = max(1, int(np.ceil(float(topq) * len(pred_use))))
                    idx_top = np.argsort(pred_use)[-k_top:]
                    topq_u_loc = float(np.mean(up_use[idx_top]))
                    topq_n_loc = int(k_top)
            return {
                "model_name": model_name,
                "ic": mean_ic_loc,
                "topq_u": topq_u_loc,
                "topq_n": int(topq_n_loc),
            }

        winner_model_metrics["ridge"] = next(
            (r for r in candidate_rows if r.get("target_name") == best_name),
            {"target_name": best_name},
        )
        try:
            from sklearn.ensemble import ExtraTreesRegressor
            winner_model_metrics["extratrees"] = _eval_model_cv(
                "extratrees",
                lambda: ExtraTreesRegressor(
                    n_estimators=200,
                    min_samples_leaf=80,
                    min_samples_split=200,
                    max_features="sqrt",
                    bootstrap=True,
                    max_samples=0.7,
                    random_state=42,
                    n_jobs=1,
                ),
            )
        except Exception:
            pass
        try:
            from sklearn.ensemble import HistGradientBoostingRegressor
            winner_model_metrics["hgbt"] = _eval_model_cv(
                "hgbt",
                lambda: HistGradientBoostingRegressor(
                    loss="squared_error",
                    max_depth=3,
                    max_iter=220,
                    min_samples_leaf=80,
                    l2_regularization=0.2,
                    learning_rate=0.05,
                    random_state=42,
                ),
            )
        except Exception:
            pass

    race_log.append(f"    Winner: {best_name} (score={best_ic:.6f}, metric={select_metric})")
    race_diag = {
        "select_metric": str(select_metric),
        "candidate_metrics": candidate_rows,
        "winner_model_metrics": winner_model_metrics,
    }
    return best_name, best_y, race_log, race_diag


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
        gamma_range: Tuple[float, float] = (0.0, 0.8),
        alpha_range: Tuple[float, float] = (1e-4, 1e-1),
        delta_range: Tuple[float, float] = (0.5, 2.0),
        n_grid_points: int = 10,
        cost_pct: float = 0.005,
        sum_to_one: bool = True,
        non_negative: bool = True,
        top_k_pct: float = 0.30,
        top_k_hard_cap: float | None = 0.30,
        returns_are_net: bool = True,
        random_state: int = 42,
        select_metric: str = "topq_u_policy",
        select_topq: float = 0.30,
        require_positive_topq_u: bool = True,
        topq_min_samples: int = 50,
        winsor_q_low: float = 0.01,
        winsor_q_high: float = 0.99,
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
            top_k_hard_cap: Optional hard cap applied to top_k_pct during evaluation
            returns_are_net: True if `return`/labels already include costs
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
        self.top_k_hard_cap = None if top_k_hard_cap is None else float(top_k_hard_cap)
        self.returns_are_net = bool(returns_are_net)
        self._top_k_cap_warned = False
        self.random_state = random_state
        self.select_metric = str(select_metric)
        self.select_topq = float(select_topq)
        self.require_positive_topq_u = bool(require_positive_topq_u)
        self.topq_min_samples = int(topq_min_samples)
        self.winsor_q_low = float(winsor_q_low)
        self.winsor_q_high = float(winsor_q_high)
        
        # Fitted attributes
        self.weights_: Optional[np.ndarray] = None
        self.model_names_: Optional[List[str]] = None
        self.best_params_: Optional[Dict] = None
        self.target_race_metrics_: Optional[Dict[str, Any]] = None
        self.cv_results_: Optional[pd.DataFrame] = None
        self.scaler_: Optional[PredictionScaler] = None
        self.ridge_pipeline_: Optional[Pipeline] = None
        self.limit_offset_pipeline_: Optional[Pipeline] = None
        self.limit_offset_features_: Optional[List[str]] = None
        self.oof_policy_pred_: Optional[np.ndarray] = None
        self.oof_limit_offset_pred_: Optional[np.ndarray] = None
        self.limit_offset_diag_: Optional[Dict[str, Any]] = None
        self.policy_model_bundle_: Optional[Dict[str, Any]] = None
        self.limit_offset_model_bundle_: Optional[Dict[str, Any]] = None
        self.feature_selection_diag_: Optional[Dict[str, Any]] = None
        self.offset_feature_selection_diag_: Optional[Dict[str, Any]] = None
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
        y_raw: np.ndarray,
        timestamps: np.ndarray | None,
        alpha: float,
        delta: float,
        gamma: float,
        groups: np.ndarray | None = None,
        symbols: np.ndarray | None = None,
    ) -> Dict:
        """Evaluate hyperparameters using purged cross-validation.
        
        Computes metrics using a realistic trading policy:
        1. Select top-k% predictions globally (across symbols and time)
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
            symbols: Optional array of symbols for diversity metrics
            
        Returns:
            Dictionary of evaluation metrics
        """
        from extreme_price_movements.purged_cv import PurgedKFold
        
        # Use purged CV to avoid lookahead bias
        # If timestamps available, use time-based purging (purge=5 means 5 seconds)
        # Otherwise use index-based purging (purge=5 means 5 samples)
        if timestamps is not None:
            # Time-based purging: purge 5 hours (18000 seconds) before test set
            pkf = PurgedKFold(n_splits=3, purge=43200, embargo=43200, times=timestamps)
        else:
            # Index-based purging: purge 5 samples before test set
            pkf = PurgedKFold(n_splits=3, purge=12, embargo=12)
        
        oof_preds = np.full(len(y), np.nan)
        oof_true_raw = np.full(len(y), np.nan)
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
            oof_true_raw[val_idx] = y_raw[val_idx]
            
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
                'PnL_per_day': 0.0,
                'Trades_per_day': 0.0,
                'Unique_Symbols_Selected': 0,
                'Unique_Symbols_Total': 0,
                'Sortino': 0.0,
                'MaxDD': 1.0,
                'IC': 0.0,
                'WinRate': 0.0,
            }
        
        pred = oof_preds[mask]
        true = y[mask]
        true_raw = oof_true_raw[mask]
        ts_masked = timestamps[mask] if timestamps is not None else None
        sym_masked = symbols[mask] if symbols is not None else None
        
        # Global ranking policy (across symbols AND time) with optional hard cap.
        effective_top_k_pct = float(self.top_k_pct)
        if self.top_k_hard_cap is not None:
            effective_top_k_pct_capped = min(effective_top_k_pct, float(self.top_k_hard_cap))
            if (effective_top_k_pct_capped < effective_top_k_pct) and (not self._top_k_cap_warned):
                tprint(
                    f"  top_k_pct capped by top_k_hard_cap: requested={effective_top_k_pct:.3f}, "
                    f"effective={effective_top_k_pct_capped:.3f}"
                )
                self._top_k_cap_warned = True
            effective_top_k_pct = effective_top_k_pct_capped
        k = max(1, int(effective_top_k_pct * len(pred)))
        selected_indices = np.argpartition(pred, -k)[-k:]

        # linear_5_15 sizing over globally selected ranks.
        sel_pred = pred[selected_indices]
        order = np.argsort(sel_pred)
        rank_local = np.empty(len(sel_pred), dtype=float)
        rank_local[order] = (np.arange(len(sel_pred), dtype=float) + 0.5) / max(len(sel_pred), 1)
        pos_frac = 0.05 + 0.10 * rank_local

        # Get returns for selected trades (use true_raw for financial metrics)
        selected_returns = true_raw[selected_indices]
        
        # Sort by timestamp if available for proper equity curve
        if ts_masked is not None:
            selected_ts = ts_masked[selected_indices]
            sort_order = np.argsort(selected_ts)
            selected_returns = selected_returns[sort_order]
            pos_frac = pos_frac[sort_order]
        else:
            sort_order = None
        
        # Compute base metrics (net of transaction costs)
        n_selected = len(selected_returns)
        net_returns = (selected_returns - self.cost_pct) * pos_frac
        total_pnl = float(np.sum(net_returns))
        
        # Determine frequency metrics
        import pandas as pd
        if ts_masked is not None and len(ts_masked) > 0:
            ts_conv = pd.to_datetime(ts_masked)
            n_days = (ts_conv.max() - ts_conv.min()).total_seconds() / 86400.0
            n_days = max(1.0/24.0, n_days)
            pnl_per_day = total_pnl / n_days
            trades_per_day = n_selected / n_days
        else:
            n_days = 1.0
            pnl_per_day = total_pnl
            trades_per_day = float(n_selected)

        # Aggregate to daily returns for risk metrics (Sortino/MaxDD)
        daily_returns = net_returns # Fallback
        if ts_masked is not None and len(selected_returns) > 0:
            try:
                selected_ts_sorted = ts_masked[selected_indices][sort_order] if sort_order is not None else ts_masked[selected_indices]
                dates_series = pd.to_datetime(selected_ts_sorted).date
                daily_df = pd.DataFrame({
                    'return': net_returns,
                    'date': dates_series
                })
                daily_sum_mapped = daily_df.groupby('date')['return'].sum()
                
                # Create full calendar series to include zero-return days
                unique_dates = np.unique(pd.to_datetime(ts_masked).date)
                full_daily = pd.Series(0.0, index=unique_dates)
                full_daily.update(daily_sum_mapped)
                daily_returns = full_daily.values
            except Exception:
                # Fallback to per-trade returns if date aggregation fails
                daily_returns = net_returns

        # Unique symbols metrics
        unique_symbols_selected = 0
        unique_symbols_total = 0
        if sym_masked is not None:
            unique_symbols_total = len(np.unique(sym_masked))
            if n_selected > 0:
                selected_symbols = sym_masked[selected_indices]
                unique_symbols_selected = len(np.unique(selected_symbols))
        
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
        # Refactor: IC is about price, not relative rank
        try:
            if np.std(pred) < 1e-12 or np.std(true_raw) < 1e-12:
                ic = 0.0
            else:
                ic = float(spearmanr(pred, true_raw).correlation)
            
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
            'PnL_total': total_pnl,
            'PnL_per_day': pnl_per_day,
            'Trades_per_day': trades_per_day,
            'Unique_Symbols_Selected': unique_symbols_selected,
            'Unique_Symbols_Total': unique_symbols_total,
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
            X = np.asarray(pred_wide.values, dtype=np.float32)
        else:
            # Wide format: one column per model
            self.model_names_ = list(oof_preds.columns)
            X = np.asarray(oof_preds.values, dtype=np.float32)
        
        # Compute gross and net returns
        # y_gross is for diagnostic metrics and IC (no cost subtraction)
        # y_net is for optimization targets (includes cost subtraction)
        self._top_k_cap_warned = False
        if labels is not None:
            y_arr = np.asarray(labels, dtype=np.float32)
            if self.returns_are_net:
                y_net = y_arr
                y_gross = (y_net + np.float32(self.cost_pct)).astype(np.float32, copy=False)
                tprint(f"  Using provided labels (net): mean={np.mean(y_net):.6f}, std={np.std(y_net):.6f}")
            else:
                y_gross = y_arr
                y_net = (y_gross - np.float32(self.cost_pct)).astype(np.float32, copy=False)
                tprint(f"  Using provided labels (gross): mean={np.mean(y_gross):.6f}, std={np.std(y_gross):.6f}")
        elif 'return' in trade_outcomes.columns:
            y_arr = np.asarray(trade_outcomes['return'].values, dtype=np.float32)
            if self.returns_are_net:
                y_net = y_arr
                y_gross = (y_net + np.float32(self.cost_pct)).astype(np.float32, copy=False)
                tprint(f"  Using returns from trade_outcomes (net): mean={np.mean(y_net):.6f}, std={np.std(y_net):.6f}")
            else:
                y_gross = y_arr
                y_net = (y_gross - np.float32(self.cost_pct)).astype(np.float32, copy=False)
                tprint(f"  Using returns from trade_outcomes (gross): mean={np.mean(y_gross):.6f}, std={np.std(y_gross):.6f}")
        elif all(c in trade_outcomes.columns for c in ['entry_price', 'exit_price', 'is_long']):
            y_gross = compute_trade_labels(
                trade_outcomes['entry_price'].values,
                trade_outcomes['exit_price'].values,
                trade_outcomes['is_long'].values,
                0.0, # GROSS returns
            ).astype(np.float32, copy=False)
            y_net = compute_trade_labels(
                trade_outcomes['entry_price'].values,
                trade_outcomes['exit_price'].values,
                trade_outcomes['is_long'].values,
                self.cost_pct, # NET returns
            ).astype(np.float32, copy=False)
            tprint(f"  Computed labels (gross): mean={np.mean(y_gross):.6f}, std={np.std(y_gross):.6f}")
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
        n = min(len(X), len(y_gross))
        X = X[:n]
        y_gross = y_gross[:n]
        y_net = y_net[:n]
        if timestamps is not None:
            timestamps = timestamps[:n]
        if groups is not None:
            groups = groups[:n]
        if symbols is not None:
            symbols = symbols[:n]

        # Handle NaN/Inf in predictions
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)
        y_gross = np.nan_to_num(y_gross, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)
        y_net = np.nan_to_num(y_net, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)
        
        # Run target representation race to find best y for this bucket
        tprint("  Running target representation race...")
        # Note: race expects gross returns for IC evaluation, but candidate targets built off y_net
        _u_policy = None
        if "u_policy_net" in trade_outcomes.columns:
            _u_policy = np.asarray(trade_outcomes["u_policy_net"].values, dtype=np.float32)
        elif "u_policy" in trade_outcomes.columns:
            _u_policy = np.asarray(trade_outcomes["u_policy"].values, dtype=np.float32)
        _trade_mask = np.asarray(trade_outcomes["trade_mask"].values, dtype=bool) if "trade_mask" in trade_outcomes.columns else np.ones(len(X), dtype=bool)
        if self.select_metric == "topq_u_policy" and _u_policy is None:
            tprint("  WARNING: sizer_select_metric='topq_u_policy' requested but u_policy_net missing. Falling back to 'ic'.")
            self.select_metric = "ic"
        tgt_name, y, race_log, race_diag = run_ridge_target_race(
            X, y_gross, symbols, timestamps,
            alpha=0.5, cost_pct=self.cost_pct,
            select_metric=self.select_metric,
            topq=self.select_topq,
            u_policy=_u_policy,
            require_positive_topq_u=self.require_positive_topq_u,
            topq_min_samples=self.topq_min_samples,
            trade_mask=_trade_mask,
        )
        for line in race_log:
            tprint(line)
        self.best_target_name_ = tgt_name
        self.target_race_metrics_ = race_diag
        if _u_policy is not None:
            tprint("  Using policy-aware utility labels (u_policy) as authoritative Ridge target")
            y = np.nan_to_num(_u_policy[:n], nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)
        
        # NOTE: Do NOT scale globally here - scaling is done per-fold in _evaluate_params
        # to prevent data leakage. The final scaler is fit after CV on all data.
        
        # Optuna Multi-Objective Optimization over hyperparameters
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        results = []
        
        def objective(trial):
            gamma = trial.suggest_float("gamma", self.gamma_range[0], self.gamma_range[1])
            alpha = trial.suggest_float("alpha", self.alpha_range[0], self.alpha_range[1], log=True)
            delta = trial.suggest_float("delta", self.delta_range[0], self.delta_range[1])
            
            metrics = self._evaluate_params(
                X, y, y_gross, timestamps, alpha, delta, gamma, groups, symbols
            )
            
            # Save all metrics into trial user_attrs for later dataframe construction
            for k, v in metrics.items():
                trial.set_user_attr(k, v)
            
            # Store for the dataframe
            results.append(metrics)
            
            # We want to maximize PnL_per_day, maximize Sortino, and minimize MaxDD
            # For Optuna's directions=["maximize", "maximize", "minimize"]:
            return (
                metrics.get("PnL_per_day", -999.0),
                metrics.get("Sortino", -99.0),
                metrics.get("MaxDD", 99.0)
            )

        n_trials = min(150, self.n_grid_points * self.n_grid_points * 4)
        tprint(f"  Evaluating {n_trials} hyperparameter combinations via Optuna...")
        
        class OptunaLogger:
            def __init__(self):
                self.best_pnl = -np.inf
                self.best_sortino = -np.inf
                
            def __call__(self, study, trial):
                pnl = trial.user_attrs.get("PnL_per_day", -999.0)
                sortino = trial.user_attrs.get("Sortino", -99.0)
                maxdd = trial.user_attrs.get("MaxDD", 99.0)
                trades_per_day = trial.user_attrs.get("Trades_per_day", 0.0)
                
                is_best = False
                msg = ""
                if pnl > self.best_pnl:
                    self.best_pnl = pnl
                    is_best = True
                    msg += "New Best PnL/Day! "
                if sortino > self.best_sortino:
                    self.best_sortino = sortino
                    is_best = True
                    msg += "New Best Sortino!"
                    
                if is_best:
                    tprint(f"    Trial {trial.number} {msg}PnL/Day={pnl:.6f}, Trades/Day={trades_per_day:.4f}, Sortino={sortino:.3f}, MaxDD={maxdd:.4f} | "
                           f"Params: alpha={trial.params.get('alpha'):.5f}, delta={trial.params.get('delta'):.3f}, gamma={trial.params.get('gamma'):.3f}")

        sampler = optuna.samplers.NSGAIISampler(seed=42)
        study = optuna.create_study(directions=["maximize", "maximize", "minimize"], sampler=sampler)
        study.optimize(objective, n_trials=n_trials, callbacks=[OptunaLogger()])
        
        # Create results DataFrame
        self.cv_results_ = pd.DataFrame(results)
        
        # Compute composite J z-score for selection
        # Use PnL_per_day (log-growth per unit time) instead of per trade
        self.cv_results_['J_zscore'] = composite_J_zscore(
            self.cv_results_,
            pnl_col='PnL_per_day',
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
        sample_weight = self._compute_sample_weights(y, float(self.best_params_.get('gamma', 1.0)))
        
        # Fold-safe feature pruning + ElasticNet tuning for Ridge
        base_feature_names = list(self.model_names_ or [])
        fs_diag_ridge = run_fold_safe_feature_pruning_and_elasticnet(
            X=X,
            y=y,
            feature_names=base_feature_names,
            timestamps=timestamps,
            outer_splits=4,
            inner_splits=4,
            top_q=max(0.05, min(0.30, float(self.select_topq))),
            max_samples=5000,
            random_state=int(self.random_state),
        )

        ridge_selected_features = [f for f in fs_diag_ridge.get("selected_features", []) if f in base_feature_names]
        if not ridge_selected_features:
            ridge_selected_features = base_feature_names
        ridge_selected_idx = np.asarray([base_feature_names.index(f) for f in ridge_selected_features], dtype=np.int32)
        X_ridge = np.asarray(X[:, ridge_selected_idx], dtype=np.float32, order='C')

        # New Feature Pruning for Tree Models (LGBM)
        from extreme_price_movements.feature_select.run import run_feature_selection
        from extreme_price_movements.feature_select.cv import CVConfig
        from extreme_price_movements.feature_select.scoring import UtilityConfig, FeatureSelectConfig
        import pandas as pd

        df_X = pd.DataFrame(X, columns=base_feature_names)
        time_s = pd.Series(timestamps) if timestamps is not None else None

        cv_cfg = CVConfig(n_splits=3, min_train_size=max(100, len(y)//4), val_size=max(100, len(y)//4))
        util_cfg = UtilityConfig(utility_mode="topq_mean", topq=max(0.05, min(0.30, float(self.select_topq))))
        fs_cfg = FeatureSelectConfig(min_features=5, n_repeats_perm=5, confirm_mode="single_seed_fast")

        lgbm_p = {
            "learning_rate": 0.05,
            "max_depth": 3,
            "n_estimators": 200,
            "early_stopping_rounds": 20
        }

        try:
            tree_fs_res = run_feature_selection(
                X=df_X, y=y, groups=None, time_index=time_s,
                model_kind="regression", quantile_alpha=None,
                cv_config=cv_cfg, lgbm_params=lgbm_p,
                utility_config=util_cfg, fs_config=fs_cfg,
                random_seed=int(self.random_state),
                output_dir="artifacts"
            )
            tree_selected_features = [f for f in tree_fs_res.selected_features if f in base_feature_names]
            if not tree_selected_features:
                tree_selected_features = base_feature_names
        except Exception as e:
            tprint(f"  WARNING: Tree feature selection failed ({e}), falling back to all features.")
            tree_selected_features = base_feature_names

        tree_selected_idx = np.asarray([base_feature_names.index(f) for f in tree_selected_features], dtype=np.int32)
        X_tree = np.asarray(X[:, tree_selected_idx], dtype=np.float32, order='C')

        # We will use self.model_names_ = (base_feature_names) but override X down the line
        # when passing into _run_policy_candidate
        self.model_names_ridge_ = ridge_selected_features
        self.model_names_tree_ = tree_selected_features
        self.model_names_ = base_feature_names

        self.feature_selection_diag_ = {"ridge": fs_diag_ridge}
        tprint(f"  Feature selection (sizer): kept {len(ridge_selected_features)} for Ridge, {len(tree_selected_features)} for Trees (out of {len(base_feature_names)})")

        from extreme_price_movements.purged_cv import PurgedKFold
        if timestamps is not None:
            pkf = PurgedKFold(n_splits=3, purge=43200, embargo=43200, times=timestamps)
        else:
            pkf = PurgedKFold(n_splits=3, purge=12, embargo=12)
        split_args = [X]
        if groups is not None:
            split_args.append(groups)
        cv_splits = list(pkf.split(*split_args))

        race_cfg = {}
        if hasattr(trade_outcomes, "attrs") and isinstance(trade_outcomes.attrs, dict):
            race_cfg = trade_outcomes.attrs.get("sizer_race_cfg", {}) or {}

        squash_fn = str((race_cfg or {}).get("sizer_race_squash_fn", "tanh")).lower()
        squash_k = float((race_cfg or {}).get("sizer_race_squash_k", 1.0))
        use_isotonic = bool((race_cfg or {}).get("sizer_race_use_isotonic", False))
        smoother_kinds = list((race_cfg or {}).get("sizer_race_smoothers", ["ridge", "huber"]))
        if not smoother_kinds:
            smoother_kinds = ["ridge", "huber"]
        top_frac = float((race_cfg or {}).get("sizer_race_top_frac", 0.30))
        top_frac = min(max(top_frac, 0.01), 0.95)
        top30_boost = float((race_cfg or {}).get("sizer_race_top30_boost", 2.0))
        use_two_pass = bool((race_cfg or {}).get("sizer_race_use_two_pass", True))
        require_sortino_top = bool((race_cfg or {}).get("sizer_race_require_sortino_top", False))
        pnl_top_floor = float((race_cfg or {}).get("sizer_race_min_pnl_top", 0.0))

        def _extract_aux_vec(df: pd.DataFrame, names: list[str], default: float = 0.0) -> np.ndarray:
            for n in names:
                if n in df.columns:
                    v = np.asarray(df[n].values[:len(y)], dtype=float)
                    return np.where(np.isfinite(v), v, default)
            return np.full(len(y), default, dtype=float)

        p_early_vec = _extract_aux_vec(oof_preds, ["early_inval", "oof_p_early_inval", "pred_early_inval"], default=0.0)
        mae_vec = _extract_aux_vec(oof_preds, ["mae_q70", "oof_log_mae_q70_hat", "mae_ret"], default=0.0)
        mfe_vec = _extract_aux_vec(oof_preds, ["mfe", "oof_log_mfe_hat", "mfe_ret"], default=0.0)
        if "oof_log_mae_q70_hat" in oof_preds.columns and "mae_q70" not in oof_preds.columns:
            mae_vec = np.expm1(np.clip(mae_vec, -20.0, 20.0))
        if "oof_log_mfe_hat" in oof_preds.columns and "mfe" not in oof_preds.columns:
            mfe_vec = np.expm1(np.clip(mfe_vec, -20.0, 20.0))
        mae_vec = np.maximum(mae_vec, 0.0)
        mfe_vec = np.maximum(mfe_vec, 0.0)

        def _phase1_weights(
            w_base: np.ndarray,
            p_early: np.ndarray,
            mae_q70: np.ndarray,
            mfe: np.ndarray | None = None,
            fold_train_idx: np.ndarray | None = None,
        ) -> np.ndarray:
            gamma_g = float((race_cfg or {}).get("sizer_phase1_gamma_g", 2.0))
            eps_gate = float((race_cfg or {}).get("sizer_phase1_eps_gate", 0.05))
            c_mae = float((race_cfg or {}).get("sizer_phase1_c_mae", 1.0))
            mfe_lambda = float((race_cfg or {}).get("sizer_phase1_mfe_lambda", 0.25))
            mfe_tau = float((race_cfg or {}).get("sizer_phase1_mfe_tau", 1.0))
            w_min = float((race_cfg or {}).get("sizer_phase1_w_min", 0.1))
            w_max = float((race_cfg or {}).get("sizer_phase1_w_max", 10.0))

            w = np.asarray(w_base, dtype=float).copy()
            p = np.clip(np.asarray(p_early, dtype=float), 0.0, 1.0)
            gate = np.maximum(eps_gate, 1.0 - p) ** gamma_g

            idx = np.arange(len(w)) if fold_train_idx is None else np.asarray(fold_train_idx, dtype=int)
            idx = idx[(idx >= 0) & (idx < len(w))]
            if len(idx) == 0:
                idx = np.arange(len(w))

            mae_train = np.asarray(mae_q70, dtype=float)[idx]
            mae_train = mae_train[np.isfinite(mae_train)]
            mae_med = float(np.median(mae_train)) if len(mae_train) else 1.0
            mae_med = max(mae_med, 1e-12)
            mae_n = np.clip(np.asarray(mae_q70, dtype=float) / mae_med, 0.0, 50.0)
            risk = 1.0 / (1.0 + c_mae * mae_n)

            opp = 1.0
            if mfe is not None:
                mfe_train = np.asarray(mfe, dtype=float)[idx]
                mfe_train = mfe_train[np.isfinite(mfe_train)]
                mfe_med = float(np.median(mfe_train)) if len(mfe_train) else 1.0
                mfe_med = max(mfe_med, 1e-12)
                mfe_n = np.clip(np.asarray(mfe, dtype=float) / mfe_med, 0.0, 50.0)
                opp = 1.0 + mfe_lambda * np.tanh((mfe_n - 1.0) / max(mfe_tau, 1e-6))

            w1 = w * gate * risk * opp
            w1 = np.clip(np.where(np.isfinite(w1), w1, w), w_min, w_max)
            w1 *= (np.mean(w) / (np.mean(w1) + 1e-12))
            return w1

        def _top_mask_from_score(score: np.ndarray, top_frac_local: float, order_index: np.ndarray) -> np.ndarray:
            n = len(score)
            k = max(1, int(np.ceil(float(top_frac_local) * max(n, 1))))
            # stable deterministic ranking: score desc, then original index asc
            ord_idx = np.lexsort((np.asarray(order_index, dtype=np.int64), -np.asarray(score, dtype=float)))
            keep = np.zeros(n, dtype=bool)
            keep[ord_idx[:k]] = True
            return keep

        def _apply_squash(v):
            vv = np.asarray(v, dtype=float)
            if squash_fn == "sigmoid":
                return np.clip(1.0 / (1.0 + np.exp(-squash_k * vv)), 0.0, 1.0)
            return np.clip(np.tanh(squash_k * vv), -1.0, 1.0)

        def _daily_metrics(y_part, size_part, ts_part):
            pnl = np.asarray(y_part, dtype=float) * np.asarray(size_part, dtype=float)
            if ts_part is None:
                daily = pnl
            else:
                d = pd.DataFrame({"ts": pd.to_datetime(ts_part), "pnl": pnl})
                d["day"] = d["ts"].dt.floor("D")
                daily = d.groupby("day", sort=True)["pnl"].sum().values
            if len(daily) == 0:
                return {"pnl_per_day": -1e9, "sortino": -1e9, "ulcer": 1e9, "tuw": 1.0}
            eq = np.cumsum(daily) + 1.0
            peak = np.maximum.accumulate(eq)
            dd = (eq - peak) / np.maximum(peak, 1e-12)
            downside = daily[daily < 0]
            downside_dev = float(np.sqrt(np.mean(downside * downside)) + 1e-12) if len(downside) else 1e-12
            sortino = float(np.mean(daily) / downside_dev)
            ulcer = float(np.sqrt(np.mean((dd * 100.0) ** 2)))
            tuw = float(np.mean(eq < peak))
            return {
                "pnl_per_day": float(np.mean(daily)),
                "sortino": sortino,
                "ulcer": ulcer,
                "tuw": tuw,
            }

        def _fit_base(name, X_tr, y_tr, X_va, y_va, w_tr):
            name = str(name)
            if name == "ridge":
                base = Pipeline([
                    ("scaler", StandardScaler(with_mean=True, with_std=True)),
                    ("model", Ridge(alpha=float((race_cfg or {}).get("race_alpha_ridge", 1.0)), fit_intercept=True, random_state=42)),
                ])
                base.fit(X_tr, y_tr, model__sample_weight=w_tr)
                return base
            if name == "et":
                base = ExtraTreesRegressor(
                    n_estimators=800, random_state=42, n_jobs=3, max_depth=5,
                    min_samples_leaf=80, min_samples_split=200, max_features="sqrt",
                    max_leaf_nodes=256, bootstrap=True, max_samples=0.7,
                    criterion="squared_error",
                )
                base.fit(X_tr, y_tr, sample_weight=w_tr)
                return base
            if name == "xgb":
                try:
                    import xgboost as xgb
                except Exception:
                    return None
                obj_func = make_confidence_conditional_regression_objective(
                    alpha=3.0,
                    threshold=0.0,
                    temperature=0.5,
                    lambda_conf=0.01,
                    use_magnitude=False
                )
                base = xgb.XGBRegressor(
                    n_estimators=3000,
                    learning_rate=0.03,
                    max_depth=5,
                    subsample=0.7,
                    colsample_bytree=0.7,
                    colsample_bylevel=0.8,
                    max_delta_step=2.0,
                    tree_method="hist",
                    random_state=42,
                    n_jobs=3,
                    objective=obj_func,
                )
                base.fit(
                    X_tr, y_tr,
                    sample_weight=w_tr,
                    eval_set=[(X_va, y_va)],
                    verbose=False,
                    early_stopping_rounds=100
                )
                return base
            return None

        def _build_smoother(kind):
            if str(kind) == "huber":
                from sklearn.linear_model import HuberRegressor
                return Pipeline([
                    ("scaler", StandardScaler()),
                    ("model", HuberRegressor(epsilon=1.35, alpha=float((race_cfg or {}).get("sizer_race_smoother_alpha", 1.0)), fit_intercept=True, max_iter=2000)),
                ])
            return Pipeline([
                ("scaler", StandardScaler()),
                ("model", Ridge(alpha=float((race_cfg or {}).get("sizer_race_smoother_alpha", 1.0)), fit_intercept=True, random_state=42)),
            ])

        def _run_policy_candidate(base_name, smoother_name):
            oof_size = np.full(len(y), np.nan)
            fold_rows = []
            for tr_idx, va_idx in cv_splits:
                X_tr, y_tr = X[tr_idx], y[tr_idx]
                X_va, y_va = X[va_idx], y[va_idx]
                w_full = _phase1_weights(sample_weight, p_early_vec, mae_vec, mfe_vec, fold_train_idx=tr_idx)
                w_tr = w_full[tr_idx] if w_full is not None else None

                if use_two_pass and top30_boost > 0.0:
                    # Inner K-Fold to generate OOS predictions for the training set
                    from sklearn.model_selection import KFold
                    inner_cv = KFold(n_splits=3, shuffle=False)
                    p_tr_oos = np.full(len(X_tr), np.nan, dtype=float)

                    for inner_tr, inner_va in inner_cv.split(X_tr):
                        X_inner_tr, y_inner_tr = X_tr[inner_tr], y_tr[inner_tr]
                        X_inner_va, y_inner_va = X_tr[inner_va], y_tr[inner_va]
                        w_inner_tr = w_tr[inner_tr] if w_tr is not None else None

                        inner_base = _fit_base(base_name, X_inner_tr, y_inner_tr, X_inner_va, y_inner_va, w_inner_tr)
                        if inner_base is not None:
                            p_tr_oos[inner_va] = np.asarray(inner_base.predict(X_inner_va), dtype=float)

                    # Fallback to in-sample if inner CV completely fails
                    valid_oos = np.isfinite(p_tr_oos)
                    if not valid_oos.all():
                        fallback_base = _fit_base(base_name, X_tr, y_tr, X_va, y_va, w_tr)
                        if fallback_base is not None:
                            p_tr_oos[~valid_oos] = np.asarray(fallback_base.predict(X_tr[~valid_oos]), dtype=float)

                    if np.isfinite(p_tr_oos).all():
                        # Smooth tail weighting using sigmoid
                        # s_i = score
                        # T = temperature (e.g., 0.25 * std(score))
                        # beta = boost strength
                        q_thresh = np.nanquantile(p_tr_oos, 1.0 - top_frac)
                        score_std = np.nanstd(p_tr_oos)
                        temperature = max(0.25 * score_std, 1e-6)

                        # sigmoid function: 1 / (1 + exp(-(s_i - q) / T))
                        sigmoid_weights = 1.0 / (1.0 + np.exp(-(p_tr_oos - q_thresh) / temperature))

                        w_tr2 = np.asarray(w_tr, dtype=float) * (1.0 + top30_boost * sigmoid_weights)
                        w_tr2 = np.clip(w_tr2, 0.1, 100.0)

                        # Retrain on full training set with smooth boosted weights
                        base = _fit_base(base_name, X_tr, y_tr, X_va, y_va, w_tr2)
                    else:
                        base = _fit_base(base_name, X_tr, y_tr, X_va, y_va, w_tr)
                else:
                    base = _fit_base(base_name, X_tr, y_tr, X_va, y_va, w_tr)

                if base is None:
                    return None

                p_tr = np.asarray(base.predict(X_tr), dtype=float)
                p_va = np.asarray(base.predict(X_va), dtype=float)

                # Apply top-30% calibration specifically for ET and XGB
                # This ensures the model's confidence distribution maps correctly
                # in the critical tail region.
                top_calibrator = None
                if base_name in ["et", "xgb"]:
                    # identify top 30% in training fold
                    top_mask_tr = _top_mask_from_score(p_tr, top_frac, tr_idx)
                    if np.any(top_mask_tr):
                        p_tr_top = p_tr[top_mask_tr]
                        y_tr_top = y_tr[top_mask_tr]
                        w_tr_top = w_tr[top_mask_tr] if w_tr is not None else None

                        try:
                            from sklearn.isotonic import IsotonicRegression
                            top_calibrator = IsotonicRegression(out_of_bounds="clip")
                            top_calibrator.fit(p_tr_top, y_tr_top, sample_weight=w_tr_top)

                            # transform inputs before they hit the smoother
                            p_tr = top_calibrator.predict(p_tr)
                            p_va = top_calibrator.predict(p_va)
                        except Exception:
                            top_calibrator = None

                smoother = _build_smoother(smoother_name)
                try:
                    smoother.fit(p_tr.reshape(-1, 1), y_tr, model__sample_weight=w_tr)
                except TypeError:
                    smoother.fit(p_tr.reshape(-1, 1), y_tr)
                s_tr = np.asarray(smoother.predict(p_tr.reshape(-1, 1)), dtype=float)
                s_va = np.asarray(smoother.predict(p_va.reshape(-1, 1)), dtype=float)

                # We still keep the secondary global isotonic block if use_isotonic is True
                if use_isotonic:
                    try:
                        from sklearn.isotonic import IsotonicRegression
                        iso = IsotonicRegression(out_of_bounds="clip")
                        iso.fit(s_tr, y_tr)
                        s_va = iso.predict(s_va)
                    except Exception:
                        pass

                size_va = _apply_squash(s_va)
                top_mask = _top_mask_from_score(s_va, top_frac, va_idx)
                size_va_top = np.where(top_mask, size_va, 0.0)
                oof_size[va_idx] = size_va

                ts_va = timestamps[va_idx] if timestamps is not None else None
                m_all = _daily_metrics(y_gross[va_idx], size_va, ts_va)
                m_top = _daily_metrics(y_gross[va_idx], size_va_top, ts_va)
                fold_rows.append({
                    "pnl_per_day_all": float(m_all["pnl_per_day"]),
                    "sortino_all": float(m_all["sortino"]),
                    "ulcer_all": float(m_all["ulcer"]),
                    "tuw_all": float(m_all["tuw"]),
                    "pnl_per_day_top": float(m_top["pnl_per_day"]),
                    "sortino_top": float(m_top["sortino"]),
                    "ulcer_top": float(m_top["ulcer"]),
                    "tuw_top": float(m_top["tuw"]),
                    "pnl_lift": float(m_top["pnl_per_day"] - m_all["pnl_per_day"]),
                    "n_top": int(np.sum(top_mask)),
                })

            if not fold_rows:
                return None
            agg = {
                "mu_pnl_all": float(np.mean([r["pnl_per_day_all"] for r in fold_rows])),
                "mu_sortino_all": float(np.mean([r["sortino_all"] for r in fold_rows])),
                "mu_ulcer_all": float(np.mean([r["ulcer_all"] for r in fold_rows])),
                "mu_tuw_all": float(np.mean([r["tuw_all"] for r in fold_rows])),
                "sigma_pnl_all": float(np.std([r["pnl_per_day_all"] for r in fold_rows])),
                "mu_pnl_top": float(np.mean([r["pnl_per_day_top"] for r in fold_rows])),
                "mu_sortino_top": float(np.mean([r["sortino_top"] for r in fold_rows])),
                "mu_ulcer_top": float(np.mean([r["ulcer_top"] for r in fold_rows])),
                "mu_tuw_top": float(np.mean([r["tuw_top"] for r in fold_rows])),
                "sigma_pnl_top": float(np.std([r["pnl_per_day_top"] for r in fold_rows])),
                "mu_pnl_lift": float(np.mean([r["pnl_lift"] for r in fold_rows])),
            }
            agg["stab_pen_top"] = float(agg["sigma_pnl_top"] / (abs(agg["mu_pnl_top"]) + 1e-12))
            agg["stab_pen_all"] = float(agg["sigma_pnl_all"] / (abs(agg["mu_pnl_all"]) + 1e-12))
            agg["passed_top_gate"] = bool(agg["mu_pnl_top"] > pnl_top_floor and ((agg["mu_sortino_top"] > 0.0) if require_sortino_top else True))
            return {"oof_size": oof_size, "fold_metrics": fold_rows, "agg": agg}

        race_results = {}
        for base_name in ["ridge", "et", "xgb"]:
            for sm_name in smoother_kinds:
                key = f"{base_name}+{sm_name}"
                out = _run_policy_candidate(base_name, sm_name)
                if out is not None:
                    race_results[key] = out

        if not race_results:
            raise RuntimeError("Sizer model race produced no valid candidates")

        cand_all = list(race_results.keys())
        gated = [k for k in cand_all if bool(race_results[k]["agg"].get("passed_top_gate", False))]
        cand = gated if gated else cand_all

        def _z(arr: np.ndarray) -> np.ndarray:
            arr = np.asarray(arr, dtype=float)
            return (arr - np.mean(arr)) / (np.std(arr) + 1e-12)

        mu_pnl_top = np.asarray([race_results[k]["agg"]["mu_pnl_top"] for k in cand], dtype=float)
        mu_sort_top = np.asarray([race_results[k]["agg"]["mu_sortino_top"] for k in cand], dtype=float)
        mu_ulc_top = np.asarray([race_results[k]["agg"]["mu_ulcer_top"] for k in cand], dtype=float)
        mu_tuw_top = np.asarray([race_results[k]["agg"]["mu_tuw_top"] for k in cand], dtype=float)
        mu_pnl_all = np.asarray([race_results[k]["agg"]["mu_pnl_all"] for k in cand], dtype=float)
        mu_sort_all = np.asarray([race_results[k]["agg"]["mu_sortino_all"] for k in cand], dtype=float)
        mu_ulc_all = np.asarray([race_results[k]["agg"]["mu_ulcer_all"] for k in cand], dtype=float)
        mu_tuw_all = np.asarray([race_results[k]["agg"]["mu_tuw_all"] for k in cand], dtype=float)
        stab_top = np.asarray([race_results[k]["agg"]["stab_pen_top"] for k in cand], dtype=float)
        stab_all = np.asarray([race_results[k]["agg"]["stab_pen_all"] for k in cand], dtype=float)

        scores = (
            1.5 * _z(mu_pnl_top)
            + 1.00 * _z(mu_sort_top)
            - 0.50 * _z(mu_ulc_top)
            - 0.50 * _z(mu_tuw_top)
            + 0.35 * _z(mu_pnl_all)
            + 0.25 * _z(mu_sort_all)
            - 0.15 * _z(mu_ulc_all)
            - 0.15 * _z(mu_tuw_all)
            - 0.40 * stab_top
            - 0.20 * stab_all
        )

        score_rows = []
        for k in cand_all:
            race_results[k]["agg"]["composite_score"] = float("-inf")
        for i, k in enumerate(cand):
            race_results[k]["agg"]["composite_score"] = float(scores[i])

        for k in cand_all:
            row = {"candidate": k, **race_results[k]["agg"]}
            _fm = race_results[k].get("fold_metrics", [])
            row["fold_pnl_all"] = [float(r["pnl_per_day_all"]) for r in _fm]
            row["fold_pnl_top"] = [float(r["pnl_per_day_top"]) for r in _fm]
            row["fold_sortino_all"] = [float(r["sortino_all"]) for r in _fm]
            row["fold_sortino_top"] = [float(r["sortino_top"]) for r in _fm]
            score_rows.append(row)
        self.cv_results_ = pd.DataFrame(score_rows)

        # Tie-breakers: higher pnl_top, higher sortino_top, lower ulcer_top, lower tuw_top
        ranked = sorted(
            cand,
            key=lambda k: (
                float(race_results[k]["agg"]["composite_score"]),
                float(race_results[k]["agg"]["mu_pnl_top"]),
                float(race_results[k]["agg"]["mu_sortino_top"]),
                -float(race_results[k]["agg"]["mu_ulcer_top"]),
                -float(race_results[k]["agg"]["mu_tuw_top"]),
            ),
            reverse=True,
        )
        winner_name = ranked[0]
        base_winner, smoother_winner = winner_name.split("+")
        tprint(f"  Sizer model race winner: {winner_name} score={race_results[winner_name]['agg']['composite_score']:.4f}")
        self.best_params_["race_winner"] = winner_name

        # Fit winner on full data for inference bundle with proper feature set
        X_final = X_ridge if base_winner == "ridge" else X_tree
        self.model_names_final_ = self.model_names_ridge_ if base_winner == "ridge" else self.model_names_tree_

        base_final = _fit_base(base_winner, X_final, y, X_final, y, sample_weight)
        pred_full = np.asarray(base_final.predict(X_final), dtype=float)

        # Apply top-30% calibration to final bundle for ET and XGB
        top_calibrator_final = None
        if base_winner in ["et", "xgb"]:
            # dummy order index using range
            top_mask_full = _top_mask_from_score(pred_full, top_frac, np.arange(len(pred_full)))
            if np.any(top_mask_full):
                try:
                    from sklearn.isotonic import IsotonicRegression
                    top_calibrator_final = IsotonicRegression(out_of_bounds="clip")
                    w_top = sample_weight[top_mask_full] if sample_weight is not None else None
                    top_calibrator_final.fit(pred_full[top_mask_full], y[top_mask_full], sample_weight=w_top)
                    pred_full = top_calibrator_final.predict(pred_full)
                except Exception:
                    top_calibrator_final = None

        smoother_final = _build_smoother(smoother_winner)
        try:
            smoother_final.fit(pred_full.reshape(-1, 1), y, model__sample_weight=sample_weight)
        except TypeError:
            smoother_final.fit(pred_full.reshape(-1, 1), y)

        iso_final = None
        if use_isotonic:
            try:
                from sklearn.isotonic import IsotonicRegression
                s_full = np.asarray(smoother_final.predict(pred_full.reshape(-1, 1)), dtype=float)
                iso_final = IsotonicRegression(out_of_bounds="clip")
                iso_final.fit(s_full, y)
            except Exception:
                iso_final = None

        self.policy_model_bundle_ = {
            "base_name": base_winner,
            "smoother_name": smoother_winner,
            "base_model": base_final,
            "top_calibrator": top_calibrator_final,
            "smoother_model": smoother_final,
            "isotonic_model": iso_final,
            "squash_fn": squash_fn,
            "squash_k": squash_k,
            "use_isotonic": bool(iso_final is not None),
            "race_results": {k: v["agg"] for k, v in race_results.items()},
        }
        self.oof_policy_pred_ = np.asarray(race_results[winner_name]["oof_size"], dtype=float)
        self.ridge_pipeline_ = None

        # proxy "weights" output for compatibility
        self.weights_ = np.zeros(len(self.model_names_), dtype=float)
        if base_winner == "ridge" and hasattr(base_final, "named_steps"):
            try:
                self.weights_ = np.asarray(base_final.named_steps["model"].coef_, dtype=float)
            except Exception:
                pass
        elif hasattr(base_final, "feature_importances_"):
            imp = np.asarray(base_final.feature_importances_, dtype=float)
            s_imp = float(np.sum(np.abs(imp))) + 1e-12
            self.weights_ = imp / s_imp
        # update model_names_ so compatibility is maintained
        self.model_names_ = self.model_names_final_
        self.scaler_ = None

        # Limit-offset model race (same folds/features/weights, no squash)
        k_col = None
        for candidate_col in ("k_star", "optimal_offset_ticks", "limit_offset_k", "target_offset_ticks"):
            if candidate_col in trade_outcomes.columns:
                k_col = candidate_col
                break
        self.oof_limit_offset_pred_ = None
        self.limit_offset_pipeline_ = None
        if k_col is not None:
            k_target = np.clip(np.nan_to_num(trade_outcomes[k_col].values[:n], nan=0.0), 0.0, 5.0)
        else:
            k_built = compute_optimal_limit_offset_labels(
                trade_outcomes.iloc[:n], tick_size=0.1, k_max=5, entry_fill_horizon_bars=4,
                max_hold_bars=48, tp_pct=0.005, sl_pct=0.0025, trailing_pct=0.0,
                cost_pct=self.cost_pct, eta=0.0, tie_break_smallest_k=True,
            )
            if k_built is not None:
                k_col = "k_star_built_from_policy"
                k_target = np.clip(np.nan_to_num(k_built, nan=0.0), 0.0, 5.0)

        if k_col is not None:
            offset_X = np.asarray(X, dtype=np.float32)
            if self.oof_policy_pred_ is not None and len(self.oof_policy_pred_) == len(offset_X):
                offset_X = np.column_stack([offset_X, np.asarray(self.oof_policy_pred_, dtype=np.float32)])
                offset_feature_names = list(self.model_names_) + ["sizer_score_oof"]
            else:
                offset_feature_names = list(self.model_names_)
            self.limit_offset_features_ = offset_feature_names

            def _run_offset_candidate(base_name, smoother_name):
                oof_k = np.full(len(k_target), np.nan)
                fold_mae = []
                for tr_idx, va_idx in cv_splits:
                    X_tr, y_tr = offset_X[tr_idx], k_target[tr_idx]
                    X_va, y_va = offset_X[va_idx], k_target[va_idx]
                    w_full = _phase1_weights(sample_weight, p_early_vec, mae_vec, mfe_vec, fold_train_idx=tr_idx)
                    w_tr = w_full[tr_idx] if w_full is not None else None
                    base = _fit_base(base_name, X_tr, y_tr, X_va, y_va, w_tr)
                    if base is None:
                        return None
                    p_tr = np.asarray(base.predict(X_tr), dtype=float)
                    p_va = np.asarray(base.predict(X_va), dtype=float)
                    smoother = _build_smoother(smoother_name)
                    try:
                        smoother.fit(p_tr.reshape(-1, 1), y_tr, model__sample_weight=w_tr)
                    except TypeError:
                        smoother.fit(p_tr.reshape(-1, 1), y_tr)
                    s_va = np.asarray(smoother.predict(p_va.reshape(-1, 1)), dtype=float)
                    if use_isotonic:
                        try:
                            from sklearn.isotonic import IsotonicRegression
                            s_tr = np.asarray(smoother.predict(p_tr.reshape(-1, 1)), dtype=float)
                            iso = IsotonicRegression(out_of_bounds="clip")
                            iso.fit(s_tr, y_tr)
                            s_va = iso.predict(s_va)
                        except Exception:
                            pass
                    pred = np.clip(s_va, 0.0, 5.0)
                    oof_k[va_idx] = pred
                    fold_mae.append(float(np.mean(np.abs(pred - y_va))))
                if not fold_mae:
                    return None
                return {"oof": oof_k, "mu_mae": float(np.mean(fold_mae)), "sigma_mae": float(np.std(fold_mae))}

            offset_race = {}
            for b in ["ridge", "et", "xgb"]:
                for sm in smoother_kinds:
                    key = f"{b}+{sm}"
                    out = _run_offset_candidate(b, sm)
                    if out is not None:
                        offset_race[key] = out
            if offset_race:
                offset_winner = min(offset_race.keys(), key=lambda k: (offset_race[k]["mu_mae"], offset_race[k]["sigma_mae"]))
                b_w, s_w = offset_winner.split("+")
                b_final = _fit_base(b_w, offset_X, k_target, offset_X, k_target, sample_weight)
                p_full = np.asarray(b_final.predict(offset_X), dtype=float)
                s_final = _build_smoother(s_w)
                try:
                    s_final.fit(p_full.reshape(-1, 1), k_target, model__sample_weight=sample_weight)
                except TypeError:
                    s_final.fit(p_full.reshape(-1, 1), k_target)
                self.limit_offset_model_bundle_ = {
                    "base_name": b_w,
                    "smoother_name": s_w,
                    "base_model": b_final,
                    "smoother_model": s_final,
                    "isotonic_model": None,
                    "race": {k: {"mu_mae": v["mu_mae"], "sigma_mae": v["sigma_mae"]} for k, v in offset_race.items()},
                    "winner": offset_winner,
                }
                self.oof_limit_offset_pred_ = np.clip(offset_race[offset_winner]["oof"], 0.0, 5.0)
                self.limit_offset_diag_ = {
                    "winner": offset_winner,
                    "target_column": k_col,
                    "race": self.limit_offset_model_bundle_["race"],
                }
                tprint(f"  Trained passive limit offset model race winner='{offset_winner}' target='{k_col}'")

        self.is_fitted_ = True

        # Store OOF inputs for downstream diagnostics / preds_metrics_computations
        self.oof_preds_ = oof_preds[self.model_names_].copy() if isinstance(oof_preds, pd.DataFrame) else pd.DataFrame(X, columns=self.model_names_)
        self.oof_targets_ = y.copy()
        self.oof_timestamps_ = np.asarray(timestamps).copy() if timestamps is not None else None
        self.oof_symbols_ = np.asarray(symbols).copy() if symbols is not None else None

        tprint(f"RidgePositionSizer.fit: Done. Ridge features={len(self.model_names_)}")

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
        
        if self.policy_model_bundle_ is not None:
            base = self.policy_model_bundle_.get("base_model")
            top_calibrator = self.policy_model_bundle_.get("top_calibrator")
            smoother = self.policy_model_bundle_.get("smoother_model")
            iso = self.policy_model_bundle_.get("isotonic_model")
            squash_fn = str(self.policy_model_bundle_.get("squash_fn", "tanh")).lower()
            squash_k = float(self.policy_model_bundle_.get("squash_k", 1.0))
            pred_base = np.asarray(base.predict(X), dtype=float)

            # Apply top-30% calibrator if available
            if top_calibrator is not None:
                pred_base = np.asarray(top_calibrator.predict(pred_base), dtype=float)

            s = np.asarray(smoother.predict(pred_base.reshape(-1, 1)), dtype=float)
            if iso is not None:
                s = np.asarray(iso.predict(s), dtype=float)
            if squash_fn == "sigmoid":
                return np.clip(1.0 / (1.0 + np.exp(-squash_k * s)), 0.0, 1.0)
            return np.clip(np.tanh(squash_k * s), -1.0, 1.0)
        if self.ridge_pipeline_ is not None:
            return self.ridge_pipeline_.predict(X)
        if self.scaler_ is not None and self.weights_ is not None:
            X = self.scaler_.transform(X)
            return X @ self.weights_
        if self.weights_ is not None:
            return X @ self.weights_
        raise RuntimeError("No fitted prediction backend found")

    def predict_limit_offset_ticks(self, model_preds: pd.DataFrame) -> np.ndarray:
        """Predict passive limit offset in ticks (clamped to [0, 5])."""
        if self.limit_offset_features_ is None:
            raise RuntimeError("Limit offset features missing")
        if "sizer_score_oof" in self.limit_offset_features_:
            base_features = [c for c in self.limit_offset_features_ if c != "sizer_score_oof"]
            x_base = np.nan_to_num(model_preds[base_features].values, nan=0.0, posinf=0.0, neginf=0.0)
            sizer_score = self.predict(model_preds)
            X = np.column_stack([x_base, sizer_score])
        else:
            X = np.nan_to_num(model_preds[self.limit_offset_features_].values, nan=0.0, posinf=0.0, neginf=0.0)
        if self.limit_offset_model_bundle_ is not None:
            base = self.limit_offset_model_bundle_.get("base_model")
            smoother = self.limit_offset_model_bundle_.get("smoother_model")
            iso = self.limit_offset_model_bundle_.get("isotonic_model")
            p = np.asarray(base.predict(X), dtype=float)
            s = np.asarray(smoother.predict(p.reshape(-1, 1)), dtype=float)
            if iso is not None:
                s = np.asarray(iso.predict(s), dtype=float)
            return np.clip(s, 0.0, 5.0)
        if self.limit_offset_pipeline_ is None:
            raise RuntimeError("Limit offset model not trained")
        return np.clip(self.limit_offset_pipeline_.predict(X), 0.0, 5.0)
    
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
            'target_race_metrics': getattr(self, 'target_race_metrics_', None),
            'gamma_range': self.gamma_range,
            'alpha_range': self.alpha_range,
            'delta_range': self.delta_range,
            'cost_pct': self.cost_pct,
            'sum_to_one': self.sum_to_one,
            'non_negative': self.non_negative,
            'top_k_pct': self.top_k_pct,
            'top_k_hard_cap': self.top_k_hard_cap,
            'returns_are_net': self.returns_are_net,
            'scaler_means': self.scaler_.means_.tolist() if self.scaler_ else None,
            'scaler_stds': self.scaler_.stds_.tolist() if self.scaler_ else None,
            'winsor_q_low': self.winsor_q_low,
            'winsor_q_high': self.winsor_q_high,
            'ridge_coef': self.ridge_pipeline_.named_steps['ridge'].coef_.tolist() if self.ridge_pipeline_ is not None else None,
            'ridge_intercept': float(self.ridge_pipeline_.named_steps['ridge'].intercept_) if self.ridge_pipeline_ is not None else None,
            'ridge_scaler_mean': self.ridge_pipeline_.named_steps['scaler'].mean_.tolist() if self.ridge_pipeline_ is not None else None,
            'ridge_scaler_scale': self.ridge_pipeline_.named_steps['scaler'].scale_.tolist() if self.ridge_pipeline_ is not None else None,
            'limit_offset_features': self.limit_offset_features_,
            'limit_offset_coef': self.limit_offset_pipeline_.named_steps['ridge'].coef_.tolist() if self.limit_offset_pipeline_ is not None else None,
            'limit_offset_intercept': float(self.limit_offset_pipeline_.named_steps['ridge'].intercept_) if self.limit_offset_pipeline_ is not None else None,
            'limit_offset_scaler_mean': self.limit_offset_pipeline_.named_steps['scaler'].mean_.tolist() if self.limit_offset_pipeline_ is not None else None,
            'limit_offset_scaler_scale': self.limit_offset_pipeline_.named_steps['scaler'].scale_.tolist() if self.limit_offset_pipeline_ is not None else None,
            'limit_offset_diag': self.limit_offset_diag_,
            'feature_selection_diag': self.feature_selection_diag_,
            'offset_feature_selection_diag': self.offset_feature_selection_diag_,
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
            top_k_hard_cap=save_dict.get('top_k_hard_cap', 0.30),
            returns_are_net=bool(save_dict.get('returns_are_net', True)),
            winsor_q_low=float(save_dict.get('winsor_q_low', 0.01)),
            winsor_q_high=float(save_dict.get('winsor_q_high', 0.99)),
        )
        
        instance.weights_ = np.array(save_dict['weights']) if save_dict['weights'] else None
        instance.model_names_ = save_dict['model_names']
        instance.best_params_ = save_dict['best_params']
        instance.target_race_metrics_ = save_dict.get('target_race_metrics')
        instance.is_fitted_ = True
        
        # Restore scaler
        if save_dict.get('scaler_means') is not None:
            instance.scaler_ = PredictionScaler()
            instance.scaler_.means_ = np.array(save_dict['scaler_means'])
            instance.scaler_.stds_ = np.array(save_dict['scaler_stds'])

        if save_dict.get('ridge_coef') is not None:
            rp = Pipeline([
                ('scaler', StandardScaler()),
                ('ridge', Ridge(alpha=float(instance.best_params_.get('alpha', 1.0)), fit_intercept=True, random_state=instance.random_state)),
            ])
            rp.named_steps['scaler'].mean_ = np.array(save_dict['ridge_scaler_mean'], dtype=float)
            rp.named_steps['scaler'].scale_ = np.array(save_dict['ridge_scaler_scale'], dtype=float)
            rp.named_steps['scaler'].var_ = rp.named_steps['scaler'].scale_ ** 2
            rp.named_steps['scaler'].n_features_in_ = len(rp.named_steps['scaler'].mean_)
            rp.named_steps['ridge'].coef_ = np.array(save_dict['ridge_coef'], dtype=float)
            rp.named_steps['ridge'].intercept_ = float(save_dict.get('ridge_intercept', 0.0))
            rp.named_steps['ridge'].n_features_in_ = len(rp.named_steps['ridge'].coef_)
            instance.ridge_pipeline_ = rp

        if save_dict.get('limit_offset_coef') is not None:
            lp = Pipeline([
                ('scaler', StandardScaler()),
                ('ridge', Ridge(alpha=float(instance.best_params_.get('alpha', 1.0)), fit_intercept=True, random_state=instance.random_state)),
            ])
            lp.named_steps['scaler'].mean_ = np.array(save_dict['limit_offset_scaler_mean'], dtype=float)
            lp.named_steps['scaler'].scale_ = np.array(save_dict['limit_offset_scaler_scale'], dtype=float)
            lp.named_steps['scaler'].var_ = lp.named_steps['scaler'].scale_ ** 2
            lp.named_steps['scaler'].n_features_in_ = len(lp.named_steps['scaler'].mean_)
            lp.named_steps['ridge'].coef_ = np.array(save_dict['limit_offset_coef'], dtype=float)
            lp.named_steps['ridge'].intercept_ = float(save_dict.get('limit_offset_intercept', 0.0))
            lp.named_steps['ridge'].n_features_in_ = len(lp.named_steps['ridge'].coef_)
            instance.limit_offset_pipeline_ = lp
            instance.limit_offset_features_ = save_dict.get('limit_offset_features')
        instance.limit_offset_diag_ = save_dict.get('limit_offset_diag')
        instance.feature_selection_diag_ = save_dict.get('feature_selection_diag')
        instance.offset_feature_selection_diag_ = save_dict.get('offset_feature_selection_diag')
        
        tprint(f"RidgePositionSizer loaded from {path}")
        
        return instance


# ═══════════════════════════════════════════════════════════════════════════════
# Pipeline Integration Function
# ═══════════════════════════════════════════════════════════════════════════════

def _compute_ridge_weight_diagnostics(
    weights: Dict[str, float],
    oof_preds: Optional[pd.DataFrame] = None,
    sizer: Optional["RidgePositionSizer"] = None,
) -> Dict[str, Any]:
    """Compute lightweight concentration diagnostics for ridge weight behavior."""
    if not weights:
        return {}
    names = list(weights.keys())
    w = np.asarray([float(weights[k]) for k in names], dtype=float)
    absw = np.abs(w)
    sum_abs = float(np.sum(absw))
    if sum_abs <= 0.0:
        p = np.zeros_like(absw)
    else:
        p = absw / sum_abs
    diag: Dict[str, Any] = {
        "weight_l1": sum_abs,
        "weight_l2": float(np.linalg.norm(w)),
        "weight_max_abs": float(np.max(absw)),
        "weight_top1_share": float(np.max(p)) if p.size else 0.0,
        "weight_top2_share": float(np.sum(np.sort(p)[-2:])) if p.size >= 2 else float(np.sum(p)),
        "weight_top3_share": float(np.sum(np.sort(p)[-3:])) if p.size >= 3 else float(np.sum(p)),
        "weight_effective_n_models": float(1.0 / np.sum(np.square(p))) if np.sum(np.square(p)) > 0 else 0.0,
        "weight_entropy": float(-np.sum(np.where(p > 0.0, p * np.log(p), 0.0))),
    }
    horizon_shares: Dict[str, float] = {}
    for h in ("H2", "H4", "H8"):
        m = np.array([f"_{h}" in n for n in names], dtype=bool)
        horizon_shares[h] = float(np.sum(absw[m]) / sum_abs) if sum_abs > 0 else 0.0
    diag["weight_share_by_horizon"] = horizon_shares

    # Correlation proxy with combined OOF signal (vectorized and cheap).
    if (
        oof_preds is not None
        and sizer is not None
        and getattr(sizer, "scaler_", None) is not None
        and getattr(sizer, "weights_", None) is not None
        and names
    ):
        present = [c for c in names if c in oof_preds.columns]
        if present:
            X_raw = np.nan_to_num(oof_preds[present].values, nan=0.0, posinf=0.0, neginf=0.0)
            X_scaled = sizer.scaler_.transform(X_raw)
            w_aligned = np.asarray([weights[c] for c in present], dtype=float)
            score = X_scaled @ w_aligned
            xc = X_scaled - X_scaled.mean(axis=0, keepdims=True)
            yc = score - float(score.mean())
            denom = np.sqrt(np.sum(xc * xc, axis=0)) * np.sqrt(np.sum(yc * yc))
            corr = np.zeros(len(present), dtype=float)
            valid = denom > 1e-12
            if np.any(valid):
                corr[valid] = (xc[:, valid].T @ yc) / denom[valid]
            contrib_df = pd.DataFrame({
                "model_name": present,
                "weight": w_aligned,
                "abs_weight": np.abs(w_aligned),
                "weight_share_abs": np.abs(w_aligned) / max(float(np.sum(np.abs(w_aligned))), 1e-12),
                "corr_with_combined_score": corr,
            }).sort_values("abs_weight", ascending=False)
            diag["top_model_contributors"] = contrib_df.head(15).to_dict(orient="records")
    return diag


def run_oof_grid_backtest(
    oof_df: pd.DataFrame,
    start_equity: float = 100000.0,
    fee_roundtrip: float = 0.002,
) -> pd.DataFrame:
    """Run a compact OOF backtest grid on sizer rankings and limit offsets.

    Expected columns include full path arrays for policy utility simulation:
    ts, asset, close, side, sizer_score_oof, opt_limit_offset_pct,
    future_highs, future_lows, future_closes, entry_price, is_long.
    """
    if oof_df.empty:
        return pd.DataFrame()
    df = oof_df.copy()
    req_cols = {
        "ts", "asset", "sizer_score_oof", "opt_limit_offset_pct",
        "future_opens", "future_highs", "future_lows", "future_closes", "entry_price", "is_long",
    }
    missing = [c for c in sorted(req_cols) if c not in df.columns]
    if missing:
        raise ValueError(
            "run_oof_grid_backtest requires policy-path columns and does not support "
            f"fwd_ret_H4 fallback. Missing: {missing}"
        )
    df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="coerce")
    df = df.dropna(subset=["ts", "asset", "sizer_score_oof"]).sort_values(["ts", "asset"])
    if df.empty:
        return pd.DataFrame()
    df["side"] = df.get("side", "LONG").fillna("LONG")
    df["close"] = np.maximum(df.get("close", 1.0).astype(float), 1e-9)
    df["opt_limit_offset_pct"] = np.maximum(df.get("opt_limit_offset_pct", 0.0).astype(float), 0.0)
    df["week_avg"] = df.groupby("asset")["sizer_score_oof"].transform(lambda s: s.rolling("7D", min_periods=1).mean())
    df["dev"] = df["sizer_score_oof"] - df["week_avg"]
    # Rank globally across symbols AND time (not cross-sectional per timestamp).
    df["rank_pct"] = df["dev"].rank(method="first", pct=True)
    # Hard policy cap: never trade outside top 30% by global rank.
    df = df[df["rank_pct"] >= 0.70].copy()
    if df.empty:
        return pd.DataFrame()

    def _size_linear_5_15(rank_pct: np.ndarray, threshold: float) -> np.ndarray:
        u = np.clip((rank_pct - threshold) / max(1.0 - threshold, 1e-9), 0.0, 1.0)
        return 0.05 + 0.10 * u

    def _size_convex_power(rank_pct: np.ndarray, threshold: float, p: float = 1.75) -> np.ndarray:
        u = np.clip((rank_pct - threshold) / max(1.0 - threshold, 1e-9), 0.0, 1.0)
        return 0.05 + 0.10 * np.power(u, float(p))

    def _size_concave_power(rank_pct: np.ndarray, threshold: float, p: float = 0.70) -> np.ndarray:
        u = np.clip((rank_pct - threshold) / max(1.0 - threshold, 1e-9), 0.0, 1.0)
        return 0.05 + 0.10 * np.power(u, float(max(p, 1e-6)))

    size_methods = {
        "linear_5_15": _size_linear_5_15,
        "convex_power": _size_convex_power,
        "concave_power": _size_concave_power,
    }

    # Phase 1: backtest all non-sizing params using linear_5_15 only.
    phase1 = []
    for q in (0.30, 0.10, 0.05):
        threshold = max(1.0 - q, 0.70)
        for offset_mode in ("optimizer", "fixed_0_15"):
            for ratio in ("2:1", "3:2", "4:2"):
                sel = df[df["rank_pct"] >= threshold].copy()
                if sel.empty:
                    continue
                frac = _size_linear_5_15(sel["rank_pct"].values, threshold)
                offset_pct = sel["opt_limit_offset_pct"] if offset_mode == "optimizer" else pd.Series(0.0015, index=sel.index)
                net = []
                frac_valid = []
                frac_all = _size_linear_5_15(sel["rank_pct"].values, threshold)
                for j, (_, r) in enumerate(sel.iterrows()):
                    try:
                        opens = np.asarray(r["future_opens"], dtype=float)
                        highs = np.asarray(r["future_highs"], dtype=float)
                        lows = np.asarray(r["future_lows"], dtype=float)
                        closes = np.asarray(r["future_closes"], dtype=float)
                        
                        if opens.size == 0 or highs.size == 0 or lows.size == 0 or closes.size == 0:
                            continue

                        entry_px_raw = float(r["entry_price"])
                        e_price = entry_px_raw - offset_pct.loc[r.name] * entry_px_raw
                        
                        sl_mult = float(r.get("label_policy_sl_atr_mult", np.nan))
                        tp_ratio = float(r.get("label_policy_tp_sl_ratio", np.nan))
                        atr_entry = float(r.get("atr_12_15m", np.nan))
                        if np.isfinite(sl_mult) and np.isfinite(tp_ratio) and np.isfinite(atr_entry):
                            sl_abs = max(sl_mult * max(atr_entry, 1e-9), 1e-9)
                            tp_abs = tp_ratio * sl_abs
                            sl_pct = sl_abs / max(entry_px_raw, 1e-9) # Use raw entry_px for SL/TP % calculation
                            tp_pct = tp_abs / max(entry_px_raw, 1e-9)
                        else:
                            tp_pct = 0.005
                            sl_pct = 0.0025
                        trailing_pct = float(r.get("label_policy_giveback_pct", 0.0))
                        max_bars = int(r.get("label_policy_max_hold_bars", 48))
                        
                        ut = _simulate_policy_utility_from_arrays(
                            entry_price=e_price,
                            is_long=bool(r["is_long"]),
                            future_opens=opens,
                            future_highs=highs,
                            future_lows=lows,
                            future_closes=closes,
                            tp_pct=tp_pct,
                            sl_pct=sl_pct,
                            trailing_pct=trailing_pct,
                            max_bars=min(max_bars, len(highs)),
                            cost_pct=fee_roundtrip,
                        )
                        net.append(float(ut))
                        frac_valid.append(float(frac_all[j]))
                    except Exception:
                        continue # Skip rows with malformed policy path data
                if not net:
                    continue
                net_arr = np.asarray(net, dtype=float)
                frac_arr = np.asarray(frac_valid, dtype=float)
                pnl = start_equity * frac_arr * net_arr
                trades = len(pnl)
                wins = int((pnl > 0).sum())
                days = max((sel["ts"].max() - sel["ts"].min()).days, 1)
                dd = np.minimum.accumulate(np.cumsum(pnl) - np.maximum.accumulate(np.cumsum(pnl)))
                phase1.append({
                    "phase": "phase1_non_sizing_grid",
                    "quantile": q,
                    "entry_offset_mode": offset_mode,
                    "tp_sl_ratio": ratio,
                    "sizing_mode": "linear_5_15",
                    "net_pnl": float(np.sum(pnl)),
                    "trades_per_day": float(trades / days),
                    "sortino": float(np.mean(net_arr) / (np.std(np.minimum(net_arr, 0.0)) + 1e-9) * np.sqrt(365)),
                    "maxdd": float(abs(dd.min())) if len(dd) else 0.0,
                    "ulcer": float(np.sqrt(np.mean(np.square(dd)))) if len(dd) else 0.0,
                    "tuw_max_days": 0.0,
                    "expectancy_per_trade": float(np.mean(pnl)) if trades else 0.0,
                    "win_rate": float(wins / max(trades, 1)),
                })
    if not phase1:
        return pd.DataFrame()
    phase1_df = pd.DataFrame(phase1)
    winner = phase1_df.loc[phase1_df["net_pnl"].idxmax()]
    win_q = float(winner["quantile"])
    win_threshold = max(1.0 - win_q, 0.70)
    win_offset_mode = str(winner["entry_offset_mode"])
    win_ratio = str(winner["tp_sl_ratio"])

    # Phase 2: keep winning non-sizing config and compare sizing families.
    sel = df[df["rank_pct"] >= win_threshold].copy()
    if sel.empty:
        return phase1_df
    # FIX #9: ensure offset_pct is always a Series so .loc[r.name] is safe.
    if win_offset_mode == "optimizer":
        offset_pct = sel["opt_limit_offset_pct"]
    else:
        offset_pct = pd.Series(0.0015, index=sel.index)
    row_nets = []
    kept_idx = []
    for j, (_, r) in enumerate(sel.iterrows()):
        try:
            opens = np.asarray(r["future_opens"], dtype=float)
            highs = np.asarray(r["future_highs"], dtype=float)
            lows = np.asarray(r["future_lows"], dtype=float)
            closes = np.asarray(r["future_closes"], dtype=float)
            
            if opens.size == 0 or highs.size == 0 or lows.size == 0 or closes.size == 0:
                continue

            entry_px_raw = float(r["entry_price"])
            e_price = entry_px_raw - offset_pct.loc[r.name] * entry_px_raw
            
            sl_mult = float(r.get("label_policy_sl_atr_mult", np.nan))
            tp_ratio = float(r.get("label_policy_tp_sl_ratio", np.nan))
            atr_entry = float(r.get("atr_12_15m", np.nan))
            if np.isfinite(sl_mult) and np.isfinite(tp_ratio) and np.isfinite(atr_entry):
                sl_abs = max(sl_mult * max(atr_entry, 1e-9), 1e-9)
                tp_abs = tp_ratio * sl_abs
                sl_pct = sl_abs / max(entry_px_raw, 1e-9)
                tp_pct = tp_abs / max(entry_px_raw, 1e-9)
            else:
                tp_pct = 0.005
                sl_pct = 0.0025
            trailing_pct = float(r.get("label_policy_giveback_pct", 0.0))
            max_bars = int(r.get("label_policy_max_hold_bars", 48))
            
            ut = _simulate_policy_utility_from_arrays(
                entry_price=e_price,
                is_long=bool(r["is_long"]),
                future_opens=opens,
                future_highs=highs,
                future_lows=lows,
                future_closes=closes,
                tp_pct=tp_pct,
                sl_pct=sl_pct,
                trailing_pct=trailing_pct,
                max_bars=min(max_bars, len(highs)),
                cost_pct=fee_roundtrip,
            )
            row_nets.append(float(ut))
            kept_idx.append(j)
        except Exception:
            continue # Skip rows with malformed policy path data
    if not row_nets:
        return phase1_df
    net_ref = np.asarray(row_nets, dtype=float)
    sel_kept = sel.iloc[kept_idx].copy()
    days = max((sel["ts"].max() - sel["ts"].min()).days, 1)
    phase2 = []
    for name, fn in size_methods.items():
        frac = np.asarray(fn(sel_kept["rank_pct"].values, win_threshold), dtype=float)
        pnl = start_equity * frac * net_ref
        trades = len(pnl)
        wins = int((pnl > 0).sum())
        dd = np.minimum.accumulate(np.cumsum(pnl) - np.maximum.accumulate(np.cumsum(pnl)))
        phase2.append({
            "phase": "phase2_sizing_compare",
            "quantile": win_q,
            "entry_offset_mode": win_offset_mode,
            "tp_sl_ratio": win_ratio,
            "sizing_mode": name,
            "net_pnl": float(np.sum(pnl)),
            "trades_per_day": float(trades / days),
            "sortino": float(np.mean(net_ref) / (np.std(np.minimum(net_ref, 0.0)) + 1e-9) * np.sqrt(365)),
            "maxdd": float(abs(dd.min())) if len(dd) else 0.0,
            "ulcer": float(np.sqrt(np.mean(np.square(dd)))) if len(dd) else 0.0,
            "tuw_max_days": 0.0,
            "expectancy_per_trade": float(np.mean(pnl)) if trades else 0.0,
            "win_rate": float(wins / max(trades, 1)),
        })
    return pd.concat([phase1_df, pd.DataFrame(phase2)], ignore_index=True)

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
    bucket_name: str | None = None,
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
            - top_k_hard_cap: Optional hard cap for top_k_pct during evaluation
            - returns_are_net: Whether provided returns/labels already include cost
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
    gamma_range = cfg.get('gamma_range', (0.0, 0.8))
    alpha_range = cfg.get('alpha_range', (1e-4, 1e-1))
    delta_range = cfg.get('delta_range', (0.5, 2.0))
    n_grid_points = cfg.get('n_grid_points', 10)
    cost_pct = cfg.get('cost_pct', 0.005)
    top_k_pct = cfg.get('top_k_pct', 0.30)
    top_k_hard_cap = cfg.get('top_k_hard_cap', 0.30)
    returns_are_net = bool(cfg.get('returns_are_net', True))
    
    # Initialize sizer
    policy_opt_meta = None
    if bool(cfg.get("label_policy_optimizer_enabled", True)):
        try:
            trade_outcomes, policy_opt_meta = optimize_label_policy(
                trade_outcomes=trade_outcomes,
                oof_preds=oof_preds,
                timestamps=timestamps,
                symbols=symbols,
                groups=groups,
                cfg=cfg,
                simulate_trade_exit_fn=simulate_trade_exit,
            )
            tprint("Label policy optimization completed before Ridge training")
        except Exception as e:
            tprint(f"WARNING: label policy optimizer failed, continuing with existing labels: {e}")

    sizer = RidgePositionSizer(
        gamma_range=gamma_range,
        alpha_range=alpha_range,
        delta_range=delta_range,
        n_grid_points=n_grid_points,
        cost_pct=cost_pct,
        top_k_pct=top_k_pct,
        top_k_hard_cap=top_k_hard_cap,
        returns_are_net=returns_are_net,
        select_metric=cfg.get('sizer_select_metric', 'topq_u_policy'),
        select_topq=float(cfg.get('sizer_topq', 0.30)),
        require_positive_topq_u=bool(cfg.get('sizer_require_positive_topq_u', True)),
        topq_min_samples=int(cfg.get('sizer_topq_min_samples', 50)),
        winsor_q_low=float(cfg.get('sizer_winsor_q_low', 0.01)),
        winsor_q_high=float(cfg.get('sizer_winsor_q_high', 0.99)),
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
        'target_race_metrics': getattr(sizer, 'target_race_metrics_', None),
        'n_models': len(weights),
        'n_trades': len(trade_outcomes),
        'sizer_uses_linear_5_15_training_eval': True,
    }
    if isinstance(policy_opt_meta, dict):
        metrics['label_policy_optimizer'] = policy_opt_meta
    ridge_diag = _compute_ridge_weight_diagnostics(weights=weights, oof_preds=oof_preds, sizer=sizer)
    if ridge_diag:
        metrics["weight_diagnostics"] = ridge_diag

    # Confirmation diagnostics for utility/offset model families and feature alignment.
    _util_bundle = getattr(sizer, "policy_model_bundle_", None) or {}
    _offset_bundle = getattr(sizer, "limit_offset_model_bundle_", None) or {}
    metrics["utility_policy_model_family"] = str(_util_bundle.get("base_name", "ridge"))
    metrics["utility_smoother_family"] = str(_util_bundle.get("smoother_name", "ridge"))
    metrics["offset_model_family"] = str(_offset_bundle.get("base_name", "ridge")) if (_offset_bundle or sizer.limit_offset_pipeline_ is not None) else None
    metrics["offset_smoother_family"] = str(_offset_bundle.get("smoother_name", "ridge")) if (_offset_bundle or sizer.limit_offset_pipeline_ is not None) else None
    metrics["sizer_feature_names"] = list(sizer.model_names_ or [])
    metrics["offset_feature_names"] = list(sizer.limit_offset_features_ or [])
    offset_base = [c for c in (sizer.limit_offset_features_ or []) if c != "sizer_score_oof"]
    metrics["offset_base_features_match_sizer_features"] = bool(offset_base == list(sizer.model_names_ or []))
    
    # Add CV results summary
    if sizer.cv_results_ is not None:
        best_idx = sizer.cv_results_['J_zscore'].idxmax()
        best_row = sizer.cv_results_.loc[best_idx]
        metrics['cv_best_pnl_total'] = float(best_row['PnL_total'])
        metrics['cv_best_pnl_per_day'] = float(best_row['PnL_per_day'])
        metrics['cv_best_trades_per_day'] = float(best_row['Trades_per_day'])
        metrics['cv_best_unique_symbols_selected'] = int(best_row['Unique_Symbols_Selected'])
        metrics['cv_best_sortino'] = float(best_row['Sortino'])
        metrics['cv_best_maxdd'] = float(best_row['MaxDD'])
        metrics['cv_best_ic'] = float(best_row['IC'])
        metrics['cv_best_winrate'] = float(best_row['WinRate'])
        metrics['cv_best_n_selected'] = int(best_row['N_selected'])
    
    data_root = resolve_data_root(cfg.get('data_root') if cfg else None)
    reports_dir = resolve_reports_dir(cfg.get('reports_root') if cfg else None)

    # Save model
    if save_model:
        run_id = run_id or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        model_dir = data_root / "models"
        model_dir.mkdir(parents=True, exist_ok=True)
        model_path = model_dir / f"ridge_position_sizer_{run_id}.json"
        sizer.save(model_path)
        metrics['model_path'] = str(model_path)
    
    # Save CV results
    if sizer.cv_results_ is not None:
        reports_dir.mkdir(parents=True, exist_ok=True)
        cv_path = reports_dir / f"ridge_position_sizer_cv_{run_id or 'latest'}.csv"
        sizer.cv_results_.to_csv(cv_path, index=False)
        metrics['cv_results_path'] = str(cv_path)
    _trm = getattr(sizer, "target_race_metrics_", None)
    if isinstance(_trm, dict):
        reports_dir.mkdir(parents=True, exist_ok=True)
        trm_path = reports_dir / f"ridge_position_sizer_target_race_{run_id or 'latest'}.json"
        with open(trm_path, "w") as f:
            json.dump(_trm, f, indent=2)
        metrics["target_race_metrics_path"] = str(trm_path)
        cand = _trm.get("candidate_metrics", [])
        if cand:
            pd.DataFrame(cand).to_csv(
                reports_dir / f"ridge_position_sizer_target_race_candidates_{run_id or 'latest'}.csv",
                index=False,
            )
        winner_mm = _trm.get("winner_model_metrics")
        if isinstance(winner_mm, dict) and winner_mm:
            wm_rows = []
            for mk, mv in winner_mm.items():
                if isinstance(mv, dict):
                    row = {"model_name": mk}
                    row.update(mv)
                    wm_rows.append(row)
            if wm_rows:
                pd.DataFrame(wm_rows).to_csv(
                    reports_dir / f"ridge_position_sizer_target_race_winner_models_{run_id or 'latest'}.csv",
                    index=False,
                )
    if ridge_diag:
        reports_dir.mkdir(parents=True, exist_ok=True)
        diag_path = reports_dir / f"ridge_position_sizer_weight_diag_{run_id or 'latest'}.json"
        with open(diag_path, "w") as f:
            json.dump(ridge_diag, f, indent=2)
        metrics["weight_diagnostics_path"] = str(diag_path)
        _top = ridge_diag.get("top_model_contributors")
        if _top:
            pd.DataFrame(_top).to_csv(
                reports_dir / f"ridge_position_sizer_weight_diag_top_{run_id or 'latest'}.csv",
                index=False,
            )

    # Export OOF parquet for preds_metrics_computations.py diagnostics
    # score = Ridge combined signal; fwd_ret_H4 = raw trade return (proxy for H4)
    if sizer.oof_preds_ is not None and sizer.oof_targets_ is not None:
        n_oof = len(sizer.oof_targets_)
        oof_score = sizer.predict(sizer.oof_preds_)
        oof_payload: dict = {
            "score": oof_score.astype(np.float32),
            "fwd_ret_H4": sizer.oof_targets_.astype(np.float32),
        }
        if sizer.oof_timestamps_ is not None and len(sizer.oof_timestamps_) == n_oof:
            oof_payload["ts"] = pd.to_datetime(sizer.oof_timestamps_)
        else:
            oof_payload["ts"] = pd.RangeIndex(n_oof)
        if sizer.oof_symbols_ is not None and len(sizer.oof_symbols_) == n_oof:
            oof_payload["asset"] = sizer.oof_symbols_.astype(str)
        else:
            oof_payload["asset"] = "ALL"
        if getattr(sizer, 'oof_policy_pred_', None) is not None:
            oof_payload["sizer_score_oof"] = np.asarray(sizer.oof_policy_pred_, dtype=np.float32)
        if getattr(sizer, 'oof_limit_offset_pred_', None) is not None:
            k_hat = np.asarray(sizer.oof_limit_offset_pred_, dtype=np.float32)
            oof_payload["opt_limit_offset_ticks"] = k_hat
            tick_size = float(cfg.get('TICK_SIZE', 0.1))
            close_proxy = np.maximum(np.asarray(trade_outcomes.get('entry_price', pd.Series(np.ones(n_oof))).values[:n_oof], dtype=float), 1e-9)
            oof_payload["opt_limit_offset_pct"] = ((tick_size * k_hat) / close_proxy).astype(np.float32)
        if 'is_long' in trade_outcomes.columns:
            oof_payload["side"] = np.where(np.asarray(trade_outcomes['is_long'].values[:n_oof], dtype=bool), "LONG", "SHORT")
        if 'entry_price' in trade_outcomes.columns:
            oof_payload["close"] = np.asarray(trade_outcomes['entry_price'].values[:n_oof], dtype=np.float32)
        # Required policy-path columns for OOF backtest (no proxy fallback supported).
        for col in (
            "future_opens", "future_highs", "future_lows", "future_closes",
            "entry_price", "is_long", "sizer_score_oof", "opt_limit_offset_pct",
            "label_policy_sl_atr_mult", "label_policy_tp_sl_ratio",
            "atr_12_15m", "label_policy_giveback_pct", "label_policy_max_hold_bars",
        ):
            if col in trade_outcomes.columns:
                oof_payload[col] = trade_outcomes[col].values[:n_oof]
        oof_df_out = pd.DataFrame(oof_payload)
        _oof_dir = data_root / "artifacts" / (run_id or "latest") / "ridge_sizer"
        _oof_dir.mkdir(parents=True, exist_ok=True)
        _oof_path = _oof_dir / "ridge_sizer_oof.parquet"
        oof_df_out.to_parquet(_oof_path, index=False)
        metrics['oof_parquet_path'] = str(_oof_path)
        tprint(f"Saved Ridge sizer OOF parquet to {_oof_path}")

        try:
            bt_df = run_oof_grid_backtest(oof_df_out)
            if not bt_df.empty:
                _bt_path = _oof_dir / "ridge_sizer_oof_backtest_grid.csv"
                bt_df.to_csv(_bt_path, index=False)
                metrics['oof_backtest_grid_path'] = str(_bt_path)
                tprint(f"Saved Ridge OOF backtest grid to {_bt_path}")

                # Extract and log Phase 2 sizing comparison metrics (post limit-offset opt)
                phase2_df = bt_df[bt_df["phase"] == "phase2_sizing_compare"]
                if not phase2_df.empty:
                    # Score function: PnL + Sortino factor
                    def _score(row):
                        return row["net_pnl"] + 10000.0 * row["sortino"]

                    phase2_df = phase2_df.copy()
                    phase2_df["_score"] = phase2_df.apply(_score, axis=1)
                    phase2_df = phase2_df.sort_values("_score", ascending=False)

                    best_row = phase2_df.iloc[0]
                    worst_row = phase2_df.iloc[-1]

                    tprint("=" * 80)
                    tprint("POSITION SIZING IMPACT (Post-Limit Offset Optimisation)")
                    tprint("=" * 80)
                    tprint(f"  Best Mode : {best_row['sizing_mode']}")
                    tprint(f"    PnL     : {best_row['net_pnl']:.2f}")
                    tprint(f"    Sortino : {best_row['sortino']:.3f}")
                    tprint(f"    MaxDD   : {best_row['maxdd']:.4f}")
                    tprint(f"  Worst Mode: {worst_row['sizing_mode']}")
                    tprint(f"    PnL     : {worst_row['net_pnl']:.2f}")
                    tprint(f"    Sortino : {worst_row['sortino']:.3f}")
                    tprint(f"    MaxDD   : {worst_row['maxdd']:.4f}")
                    tprint("=" * 80)

                    metrics['sizing_impact'] = {
                        "best": {
                            "mode": best_row["sizing_mode"],
                            "pnl": float(best_row["net_pnl"]),
                            "sortino": float(best_row["sortino"]),
                            "maxdd": float(best_row["maxdd"]),
                        },
                        "worst": {
                            "mode": worst_row["sizing_mode"],
                            "pnl": float(worst_row["net_pnl"]),
                            "sortino": float(worst_row["sortino"]),
                            "maxdd": float(worst_row["maxdd"]),
                        }
                    }

        except Exception as e:  # FIX #11: catch AttributeError and any other errors, not just ValueError
            metrics['oof_backtest_grid_error'] = str(e)
            tprint(f"WARNING: Ridge OOF backtest grid skipped: {type(e).__name__}: {e}")

    # Generate Trade Quality Diagnostic Plot
    try:
        from extreme_price_movements.trade_quality_diagnostics import generate_trade_quality_plot

        # Prepare data for diagnostic
        diag_df = trade_outcomes.copy()

        # Predict scores using fitted sizer (aligned with trade_outcomes)
        scores = sizer.predict(oof_preds)
        diag_df["score"] = scores

        # Determine output path
        b_name = bucket_name if bucket_name else "unknown_bucket"
        r_id = run_id if run_id else "latest"
        reports_dir.mkdir(parents=True, exist_ok=True)
        plot_path = reports_dir / f"trade_quality_{b_name}_{r_id}.png"

        generate_trade_quality_plot(
            df=diag_df,
            output_path=str(plot_path),
            bucket_label=f"{b_name} ({r_id})"
        )
        metrics['trade_quality_plot'] = str(plot_path)

    except Exception as e:
        tprint(f"WARNING: Failed to generate trade quality plot: {e}")

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

        # Copy aux diagnostic columns
        aux_cols = [
            "oof_u_hat", "oof_log_mae_q70_hat", "oof_log_mfe_hat", "oof_log_dur_hat",
            "mae_ret", "mfe_ret", "duration", "u_policy_net", "exit_code"
        ]
        for c in aux_cols:
            if c in oof_df.columns:
                outcomes[c] = oof_df[c].values

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
