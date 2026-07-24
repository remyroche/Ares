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

import base64
import hashlib
import json
import logging
import os
import pickle  # For serializing model bundles
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from extreme_price_movements.timestamp_contract import (
    assert_first_path_timestamp,
    causal_decision_timestamps,
)
from scipy.optimize import minimize
from scipy.stats import rankdata, spearmanr
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Import optimized prediction utilities
try:
    from extreme_price_movements.optimized_predictions import (
        BatchPredictor,
        sigmoid_sizing_numba,
        tanh_sizing_numba,
        concave_sizing_numba,
    )
    _USE_OPTIMIZED_PREDICTIONS = True
except ImportError:
    _USE_OPTIMIZED_PREDICTIONS = False

# Import Numba JIT if available
try:
    from numba import njit, prange
    _USE_NUMBA = True
except ImportError:
    _USE_NUMBA = False

try:
    from numba import jit, prange
    import numba
    numba.config.THREADING_LAYER = 'workqueue'
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

from extreme_price_movements.utils import tprint, log_pipeline_warning
from extreme_price_movements.path_utils import resolve_reports_dir, resolve_data_root
from extreme_price_movements.tpsl_optimiser.metrics_utils import compute_comprehensive_metrics
from extreme_price_movements.label_policy_optimizer import optimize_label_policy
from extreme_price_movements.elasticnet_feature_selection import run_fold_safe_feature_pruning_and_elasticnet
from extreme_price_movements.metrics import _stable_equity_and_drawdown
from extreme_price_movements.config import CFG

# Configure logging for PnL verification
logger = logging.getLogger(__name__)
# Set to DEBUG for detailed PnL debugging
logging.basicConfig(
    level=logging.WARNING,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
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


def _stable_daily_sortino_and_maxdd(daily_returns: np.ndarray) -> tuple[float, float]:
    daily = np.asarray(daily_returns, dtype=np.float64)
    daily = daily[np.isfinite(daily)]
    if daily.size == 0:
        return 0.0, 1.0

    downside = np.minimum(daily, 0.0)
    neg_days = int(np.count_nonzero(downside < 0.0))
    raw_downside_dev = float(np.sqrt(np.mean(np.square(downside)))) if daily.size > 0 else 0.0
    total_dev = float(np.nanstd(daily, ddof=1)) if daily.size > 1 else 0.0
    downside_dev = max(raw_downside_dev, 0.25 * total_dev, 1e-3)
    if neg_days >= 5 and np.isfinite(downside_dev) and downside_dev > 0.0:
        sortino = float(np.mean(daily) / downside_dev * np.sqrt(365.0))
        sortino = float(np.clip(sortino, -25.0, 25.0))
    else:
        sortino = 0.0

    _, dd_series = _stable_equity_and_drawdown(daily)
    max_dd = float(np.max(dd_series)) if dd_series.size else 0.0
    if not np.isfinite(max_dd):
        max_dd = 1.0
    max_dd = float(np.clip(max_dd, 0.0, 1.0))
    return sortino, max_dd


def _daily_risk_diagnostics(daily_returns: np.ndarray) -> dict[str, float]:
    daily = np.asarray(daily_returns, dtype=np.float64)
    daily = daily[np.isfinite(daily)]
    if daily.size == 0:
        return {
            "n_days": 0.0,
            "n_neg_days": 0.0,
            "mean_daily": 0.0,
            "downside_dev": 0.0,
        }
    downside = np.minimum(daily, 0.0)
    raw_downside_dev = float(np.sqrt(np.mean(np.square(downside)))) if daily.size > 0 else 0.0
    total_dev = float(np.nanstd(daily, ddof=1)) if daily.size > 1 else 0.0
    downside_dev = max(raw_downside_dev, 0.25 * total_dev, 1e-3)
    return {
        "n_days": float(daily.size),
        "n_neg_days": float(np.count_nonzero(downside < 0.0)),
        "mean_daily": float(np.mean(daily)),
        "downside_dev": float(downside_dev),
    }


@jit(nopython=True, cache=True)
def _aggregate_daily_values_numba(values: np.ndarray, timestamps_ns: np.ndarray) -> np.ndarray:
    """Aggregate values to daily sums using numba.

    Args:
        values: Array of values
        timestamps_ns: Array of timestamps in nanoseconds

    Returns:
        Array of daily aggregated values
    """
    # Filter out NaN values
    mask = np.isfinite(values)
    values_clean = values[mask]
    ts_clean = timestamps_ns[mask]

    if values_clean.size == 0:
        return np.array([], dtype=np.float32)

    # Convert timestamps to days (nanoseconds to days)
    ns_per_day = np.int64(86400_000_000_000)
    days = ts_clean // ns_per_day

    # Get unique days
    unique_days = np.unique(days)
    result = np.zeros(unique_days.size, dtype=np.float32)

    # Aggregate by day
    for i, day in enumerate(unique_days):
        day_mask = days == day
        result[i] = np.sum(values_clean[day_mask])

    return result


def _aggregate_daily_values(values: np.ndarray, timestamps: np.ndarray | None = None) -> np.ndarray:
    """Aggregate values to daily sums.

    Args:
        values: Array of values
        timestamps: Optional array of timestamps

    Returns:
        Array of daily aggregated values
    """
    arr = np.asarray(values, dtype=np.float32)
    if arr.size == 0:
        return np.asarray([], dtype=np.float32)
    if timestamps is None:
        return arr
    try:
        ts = pd.to_datetime(np.asarray(timestamps), utc=True, errors="coerce")
        mask = ~pd.isna(ts)
        if not np.any(mask):
            return arr
        # Use numba version for speed
        timestamps_ns = ts[mask].astype('int64').values
        return _aggregate_daily_values_numba(arr[mask], timestamps_ns).astype(np.float64)
    except Exception:
        return arr


def _effective_day_count(timestamps: np.ndarray | None) -> float:
    """Return the number of represented calendar days, with a floor of 1."""
    if timestamps is None:
        return 1.0
    try:
        ts = pd.to_datetime(np.asarray(timestamps), utc=True, errors="coerce")
    except Exception:
        return 1.0
    if len(ts) == 0:
        return 1.0
    day_vals = pd.DatetimeIndex(ts).floor("D")
    valid = ~pd.isna(day_vals)
    if not np.any(valid):
        return 1.0
    return float(max(int(day_vals[valid].nunique()), 1))


def _stable_daily_pnl_metrics(
    pnl_values: np.ndarray,
    timestamps: np.ndarray | None = None,
    start_equity: float = 1.0,
) -> tuple[float, float, float, float]:
    daily_pnl = _aggregate_daily_values(pnl_values, timestamps)
    if daily_pnl.size == 0:
        return 0.0, 1.0, 100.0, 1.0

    daily_ret = daily_pnl / max(float(start_equity), 1e-9)
    trade_ret = pnl_values / max(float(start_equity), 1e-9)
    
    # Sortino uses daily returns to normalize volatility
    sortino, _ = _stable_daily_sortino_and_maxdd(daily_ret)
    
    # Drawdowns must be evaluated on chronological trade-level PnL to capture intraday drops
    _, trade_dd_series = _stable_equity_and_drawdown(trade_ret)
    max_dd = float(np.max(trade_dd_series)) if trade_dd_series.size else 0.0
    max_dd = float(np.clip(max_dd, 0.0, 1.0))
    
    ulcer = float(np.sqrt(np.mean(np.square(trade_dd_series * 100.0)))) if trade_dd_series.size else 100.0
    tuw = float(np.mean(trade_dd_series > 1e-12)) if trade_dd_series.size else 1.0
    
    return sortino, max_dd, ulcer, tuw


def _intraday_risk_metric(max_dd: float, ulcer: float, tuw: float) -> float:
    """Primary risk proxy for search/selection.

    Equal weighting for max drawdown, ulcer, and time-under-water.
    """
    return float(
        (3.0 * max(float(max_dd), 0.0))
        + (3.0 * max(float(ulcer), 0.0))
        + (3.0 * max(float(tuw), 0.0))
    )


def _temporal_stability_metrics(daily_returns: np.ndarray) -> tuple[float, float]:
    """Return (stability_score, instability_penalty) from chronological daily returns.

    The penalty is intentionally light relative to the main intraday risk term.
    It focuses on consistency of performance across time blocks and the share of
    positive days, without overwhelming the PnL signal.
    """
    daily = np.asarray(daily_returns, dtype=np.float64)
    daily = daily[np.isfinite(daily)]
    if daily.size == 0:
        return 0.0, 1.0
    if daily.size == 1:
        return 1.0 if daily[0] >= 0.0 else 0.0, 0.0 if daily[0] >= 0.0 else 1.0

    pos_day_frac = float(np.mean(daily > 0.0))
    n_blocks = int(min(4, max(2, daily.size // 3)))
    blocks = np.array_split(daily, n_blocks)
    block_means = np.asarray([float(np.mean(b)) if len(b) else 0.0 for b in blocks], dtype=np.float64)
    mean_abs = float(np.mean(np.abs(block_means)))
    dispersion = float(np.std(block_means) / (mean_abs + 1e-9)) if mean_abs > 0.0 else float(np.std(block_means))
    instability = float(0.65 * min(dispersion, 5.0) + 0.35 * (1.0 - pos_day_frac))
    stability = float(1.0 / (1.0 + instability))
    return stability, instability


def _pnl_risk_objective(
    pnl_total: float,
    max_dd: float,
    ulcer: float,
    tuw: float,
    daily_returns: np.ndarray | None = None,
) -> float:
    """PnL-total objective with balanced risk penalty.
    
    Goals:
    1. Reward high total PnL (absolute profitability)
    2. Penalize MaxDD and Ulcer (stability)
    3. Accept higher frequency (total throughput)
    """
    # Numerator is total PnL for absolute profitability
    pnl_val = float(pnl_total)
    
    # Penalties: we use 1 + sum of weighted risks as denominator
    # max_dd in decimal (e.g. 0.002 is 0.2%)
    # we want to penalize drawdowns above ~0.5% more aggressively
    dd_penalty = 5.0 * max(0.0, float(max_dd) - 0.005) # 5x weight on DD > 0.5%
    ulcer_penalty = 3.0 * float(ulcer)
    
    # Temporal instability penalty
    instability_penalty = 0.0
    if daily_returns is not None:
        _, instability_penalty = _temporal_stability_metrics(daily_returns)
        
    risk_factor = 1.0 + dd_penalty + ulcer_penalty + (0.5 * instability_penalty)
    
    return pnl_val / max(risk_factor, 1e-9)


def _normalize_cv_times(times: np.ndarray | None) -> np.ndarray | None:
    if times is None:
        return None
    arr = np.asarray(times)
    if np.issubdtype(arr.dtype, np.number):
        return arr
    try:
        ts = pd.to_datetime(arr, utc=True, errors="coerce")
        if not np.all(pd.isna(ts)):
            return ts.view("int64").astype(np.float64)
    except Exception:
        pass
    return arr


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
                # Same-bar tie-breaking precedence:
                # 1. Compare absolute distance from bar Open to each triggered barrier.
                # 2. Shortest distance wins (proxy for reaching it first intraday).
                # 3. If distances are exactly equal, precedence is:
                #    STOP_LOSS (1) > TRAILING_STOP (2) > TAKE_PROFIT (0).
                best_price = c
                best_reason = 3
                best_dist = 1e100
                best_rank = 10

                if sl_hit:
                    # Apply stop-gap execution pricing
                    fill_px = o if o <= sl_price else sl_price
                    d = abs(o - sl_price)
                    if d < best_dist or (d == best_dist and 0 < best_rank):
                        best_price, best_reason, best_dist, best_rank = fill_px, 1, d, 0

                if trailing_hit:
                    # Apply stop-gap execution pricing
                    fill_px = o if o <= trailing_price else trailing_price
                    d = abs(o - trailing_price)
                    if d < best_dist or (d == best_dist and 1 < best_rank):
                        best_price, best_reason, best_dist, best_rank = fill_px, 2, d, 1

                if tp_hit:
                    # Apply limit-gap execution pricing
                    fill_px = o if o >= tp_price else tp_price
                    d = abs(o - tp_price)
                    if d < best_dist or (d == best_dist and 2 < best_rank):
                        best_price, best_reason, best_dist, best_rank = fill_px, 0, d, 2

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
                # Same-bar tie-breaking precedence:
                # 1. Compare absolute distance from bar Open to each triggered barrier.
                # 2. Shortest distance wins (proxy for reaching it first intraday).
                # 3. If distances are exactly equal, precedence is:
                #    STOP_LOSS (1) > TRAILING_STOP (2) > TAKE_PROFIT (0).
                best_price = c
                best_reason = 3
                best_dist = 1e100
                best_rank = 10

                if sl_hit:
                    # Apply stop-gap execution pricing
                    fill_px = o if o >= sl_price else sl_price
                    d = abs(o - sl_price)
                    if d < best_dist or (d == best_dist and 0 < best_rank):
                        best_price, best_reason, best_dist, best_rank = fill_px, 1, d, 0

                if trailing_hit:
                    # Apply stop-gap execution pricing
                    fill_px = o if o >= trailing_price else trailing_price
                    d = abs(o - trailing_price)
                    if d < best_dist or (d == best_dist and 1 < best_rank):
                        best_price, best_reason, best_dist, best_rank = fill_px, 2, d, 1

                if tp_hit:
                    # Apply limit-gap execution pricing
                    fill_px = o if o <= tp_price else tp_price
                    d = abs(o - tp_price)
                    if d < best_dist or (d == best_dist and 2 < best_rank):
                        best_price, best_reason, best_dist, best_rank = fill_px, 0, d, 2

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
# Parallelized Limit Offset Label Computation
# ═══════════════════════════════════════════════════════════════════════════════

@jit(nopython=True, parallel=True, cache=True)
def _compute_limit_offset_single_trade(
    entry_prices: np.ndarray,
    is_longs: np.ndarray,
    future_opens: np.ndarray,
    future_highs: np.ndarray,
    future_lows: np.ndarray,
    future_closes: np.ndarray,
    sl_atr_mults: np.ndarray,
    tp_sl_ratios: np.ndarray,
    max_hold_bars_arr: np.ndarray,
    giveback_pcts: np.ndarray,
    entry_fill_horizon_bars: int,
    max_hold_bars: int,
    tick_size: float,
    k_max: int,
    cost_pct: float,
    eta: float,
    atr_12_15m: np.ndarray,
    sl_pct_default: float,
    tp_pct_default: float,
    tie_break_smallest_k: bool,
) -> np.ndarray:
    """Parallelized computation of optimal limit offset labels.
    
    Args:
        entry_prices: Entry prices for each trade
        is_longs: Boolean array for long/short direction
        future_opens: 2D array of future opens (n_trades, max_bars)
        future_highs: 2D array of future highs
        future_lows: 2D array of future lows
        future_closes: 2D array of future closes
        sl_atr_mults: ATR multipliers for SL (or NaN for default)
        tp_sl_ratios: TP/SL ratios (or NaN for default)
        max_hold_bars_arr: Max hold bars per trade
        giveback_pcts: Trailing giveback percentage
        entry_fill_horizon_bars: Bars to wait for entry fill
        max_hold_bars: Default max hold bars
        tick_size: Tick size for limit orders
        k_max: Maximum k offset to try
        cost_pct: Cost percentage
        eta: Fill horizon penalty weight
        atr_12_15m: ATR values per trade
        sl_pct_default: Default SL percentage
        tp_pct_default: Default TP percentage
        tie_break_smallest_k: Whether to prefer smaller k on ties
    
    Returns:
        Array of optimal k values
    """
    n_trades = len(entry_prices)
    k_labels = np.zeros(n_trades, dtype=np.float32)
    
    # Convert to float32 for consistency
    tick_size = np.float32(tick_size)
    cost_pct = np.float32(cost_pct)
    eta = np.float32(eta)
    sl_pct_default = np.float32(sl_pct_default)
    tp_pct_default = np.float32(tp_pct_default)
    one_minus = np.float32(1.0)
    one_plus = np.float32(1.0)
    
    for i in prange(n_trades):
        entry_price = entry_prices[i]
        is_long = bool(is_longs[i])
        
        # Get arrays for this trade
        opens = future_opens[i]
        highs = future_highs[i]
        lows = future_lows[i]
        closes = future_closes[i]
        
        # Skip if no valid data
        min_len = min(len(opens), len(highs), len(lows), len(closes))
        if min_len == 0:
            k_labels[i] = 0.0
            continue
        
        # Get policy params
        sl_atr = sl_atr_mults[i]
        tp_ratio = tp_sl_ratios[i]
        max_hold = max_hold_bars_arr[i]
        giveback = giveback_pcts[i]
        atr = atr_12_15m[i]
        
        # Determine effective SL/TP percentages
        if np.isfinite(sl_atr) and np.isfinite(tp_ratio):
            sl_abs = max(sl_atr * max(atr, 1e-9), 1e-9)
            tp_abs = tp_ratio * sl_abs
            eff_sl_pct = sl_abs / max(entry_price, 1e-9)
            eff_tp_pct = tp_abs / max(entry_price, 1e-9)
        else:
            eff_sl_pct = sl_pct_default
            eff_tp_pct = tp_pct_default
        
        h_fill = min(int(entry_fill_horizon_bars), min_len)
        best_k = 0
        best_u = -1e18
        
        for k in range(k_max + 1):
            if is_long:
                limit_price = entry_price * (1.0 - tick_size * k)
            else:
                limit_price = entry_price * (1.0 + tick_size * k)
            
            # Check if limit price is hit within fill horizon
            hit_idx = -1
            if is_long:
                for j in range(h_fill):
                    if lows[j] <= limit_price:
                        hit_idx = j
                        break
            else:
                for j in range(h_fill):
                    if highs[j] >= limit_price:
                        hit_idx = j
                        break
            
            if hit_idx == -1:
                u = 0.0
            else:
                # Simulate trade from fill point
                o2 = opens[hit_idx:hit_idx + max_hold]
                h2 = highs[hit_idx:hit_idx + max_hold]
                l2 = lows[hit_idx:hit_idx + max_hold]
                c2 = closes[hit_idx:hit_idx + max_hold]
                
                if len(h2) == 0:
                    u = 0.0
                else:
                    # Use the existing trade exit simulation
                    if is_long:
                        tp_price = limit_price * (1.0 + eff_tp_pct)
                        sl_price = limit_price * (1.0 - eff_sl_pct)
                    else:
                        tp_price = limit_price * (1.0 - eff_tp_pct)
                        sl_price = limit_price * (1.0 + eff_sl_pct)
                    
                    # Simple exit simulation inline
                    peak = limit_price if is_long else limit_price
                    trough = limit_price if not is_long else limit_price
                    exit_price = limit_price
                    exit_reason = 3  # timeout
                    
                    for bar in range(len(h2)):
                        h = h2[bar]
                        l = l2[bar]
                        o = o2[bar]
                        c = c2[bar]
                        
                        if np.isnan(h) or np.isnan(l):
                            break
                        
                        if is_long:
                            if h > peak:
                                peak = h
                            tp_hit = h >= tp_price
                            sl_hit = l <= sl_price
                            trailing_hit = False
                            if peak > limit_price:
                                trailing_price = peak * (1.0 - giveback)
                                trailing_hit = l <= trailing_price
                            
                            if tp_hit or sl_hit or trailing_hit:
                                if sl_hit:
                                    exit_price = sl_price
                                    exit_reason = 1
                                elif trailing_hit:
                                    exit_price = peak * (1.0 - giveback)
                                    exit_reason = 2
                                else:
                                    exit_price = tp_price
                                    exit_reason = 0
                                break
                        else:  # short
                            if l < trough:
                                trough = l
                            tp_hit = l <= tp_price
                            sl_hit = h >= sl_price
                            trailing_hit = False
                            if trough < limit_price:
                                trailing_price = trough * (1.0 + giveback)
                                trailing_hit = h >= trailing_price
                            
                            if tp_hit or sl_hit or trailing_hit:
                                if sl_hit:
                                    exit_price = sl_price
                                    exit_reason = 1
                                elif trailing_hit:
                                    exit_price = trough * (1.0 + giveback)
                                    exit_reason = 2
                                else:
                                    exit_price = tp_price
                                    exit_reason = 0
                                break
                        
                        exit_price = c
                    
                    # Compute return
                    if is_long:
                        ret = np.log(max(exit_price, 1e-12) / max(limit_price, 1e-12))
                    else:
                        ret = np.log(max(limit_price, 1e-12) / max(exit_price, 1e-12))
                    
                    u = ret - cost_pct - eta * float(hit_idx + 1)
            
            # Update best
            if u > best_u:
                best_u = u
                best_k = k
            elif u == best_u:
                if tie_break_smallest_k:
                    if k < best_k:
                        best_k = k
                else:
                    if k > best_k:
                        best_k = k
        
        k_labels[i] = float(best_k)
    
    return k_labels

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
    # Use Numba version if available
    if NUMBA_AVAILABLE:
        try:
            return _compute_trade_labels_numba(
                np.asarray(entry_prices, dtype=np.float32),
                np.asarray(exit_prices, dtype=np.float32),
                np.asarray(is_long, dtype=np.float32),
                np.float32(cost_pct)
            ).astype(np.float64)
        except Exception:
            pass  # Fall back to NumPy
    
    # NumPy fallback
    entry_prices = np.asarray(entry_prices, dtype=float)
    exit_prices = np.asarray(exit_prices, dtype=float)
    is_long = np.asarray(is_long, dtype=float)
    
    # Compute log returns directly for numerical stability
    # Long: log(exit/entry), Short: log(entry/exit)
    long_ret = np.log(np.maximum(exit_prices, 1e-12) / np.maximum(entry_prices, 1e-12))
    short_ret = np.log(np.maximum(entry_prices, 1e-12) / np.maximum(exit_prices, 1e-12))
    log_returns = np.where(is_long == 1, long_ret, short_ret)
    
    # Handle edge cases (zero/negative prices)
    log_returns = np.nan_to_num(log_returns, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Subtract transaction costs
    yi = log_returns - cost_pct
    
    return yi.astype(np.float64)


@jit(nopython=True, cache=True)
def _compute_trade_labels_numba(
    entry_prices: np.ndarray,
    exit_prices: np.ndarray,
    is_long: np.ndarray,
    cost_pct: np.float32,
) -> np.ndarray:
    """Numba-optimized computation of trade labels."""
    n = len(entry_prices)
    labels = np.empty(n, dtype=np.float32)
    eps = np.float32(1e-12)
    
    for i in range(n):
        entry = entry_prices[i]
        exit_px = exit_prices[i]
        long = is_long[i] > 0.5
        
        # Ensure positive prices
        if entry < eps:
            entry = eps
        if exit_px < eps:
            exit_px = eps
        
        # Compute log return
        if long:
            ret = np.log(exit_px / entry)
        else:
            ret = np.log(entry / exit_px)
        
        # Handle edge cases
        if not np.isfinite(ret):
            ret = np.float32(0.0)
        
        # Subtract cost
        labels[i] = ret - cost_pct
    
    return labels


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


def _stack_object_path_column(values: Sequence[object], max_bars: int) -> Tuple[np.ndarray, np.ndarray]:
    n = len(values)
    out = np.full((n, max_bars), np.nan, dtype=np.float64)
    lengths = np.zeros(n, dtype=np.int64)
    for i, val in enumerate(values):
        arr = np.asarray(val, dtype=np.float64)
        use = min(len(arr), max_bars)
        if use <= 0:
            continue
        out[i, :use] = arr[:use]
        lengths[i] = use
    return out, lengths


def _simulate_policy_utility_batch(
    entry_prices: np.ndarray,
    is_longs: np.ndarray,
    future_opens: np.ndarray,
    future_highs: np.ndarray,
    future_lows: np.ndarray,
    future_closes: np.ndarray,
    tp_pcts: np.ndarray,
    sl_pcts: np.ndarray,
    trailing_pcts: np.ndarray,
    max_bars_arr: np.ndarray,
    cost_pct: float,
) -> np.ndarray:
    net_ret, _, _ = _simulate_policy_utility_batch_details(
        entry_prices=entry_prices,
        is_longs=is_longs,
        future_opens=future_opens,
        future_highs=future_highs,
        future_lows=future_lows,
        future_closes=future_closes,
        tp_pcts=tp_pcts,
        sl_pcts=sl_pcts,
        trailing_pcts=trailing_pcts,
        max_bars_arr=max_bars_arr,
        cost_pct=cost_pct,
    )
    return net_ret


def _simulate_policy_utility_batch_details(
    entry_prices: np.ndarray,
    is_longs: np.ndarray,
    future_opens: np.ndarray,
    future_highs: np.ndarray,
    future_lows: np.ndarray,
    future_closes: np.ndarray,
    tp_pcts: np.ndarray,
    sl_pcts: np.ndarray,
    trailing_pcts: np.ndarray,
    max_bars_arr: np.ndarray,
    cost_pct: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = len(entry_prices)
    if n == 0:
        return (
            np.asarray([], dtype=np.float64),
            np.asarray([], dtype=np.int64),
            np.asarray([], dtype=np.int64),
        )
    max_bars_global = int(np.max(max_bars_arr))
    active_mask = np.arange(max_bars_global, dtype=np.int64)[None, :] >= np.asarray(max_bars_arr, dtype=np.int64)[:, None]
    opens_2d = np.array(future_opens[:, :max_bars_global], copy=True)
    highs_2d = np.array(future_highs[:, :max_bars_global], copy=True)
    lows_2d = np.array(future_lows[:, :max_bars_global], copy=True)
    closes_2d = np.array(future_closes[:, :max_bars_global], copy=True)
    opens_2d[active_mask] = np.nan
    highs_2d[active_mask] = np.nan
    lows_2d[active_mask] = np.nan
    closes_2d[active_mask] = np.nan

    tp_prices = np.where(is_longs, entry_prices * (1.0 + tp_pcts), entry_prices * (1.0 - tp_pcts))
    sl_prices = np.where(is_longs, entry_prices * (1.0 - sl_pcts), entry_prices * (1.0 + sl_pcts))
    exit_prices, exit_bars, exit_reasons = simulate_trade_exit_batch(
        highs_2d,
        lows_2d,
        opens_2d,
        closes_2d,
        np.asarray(entry_prices, dtype=np.float64),
        np.asarray(is_longs, dtype=np.int64),
        np.asarray(tp_prices, dtype=np.float64),
        np.asarray(sl_prices, dtype=np.float64),
        np.asarray(trailing_pcts, dtype=np.float64),
        max_bars_global,
    )
    log_ret = np.where(
        np.asarray(is_longs, dtype=bool),
        np.log(np.maximum(exit_prices, 1e-12) / np.maximum(entry_prices, 1e-12)),
        np.log(np.maximum(entry_prices, 1e-12) / np.maximum(exit_prices, 1e-12)),
    )
    return (
        np.nan_to_num(log_ret - cost_pct, nan=0.0, posinf=0.0, neginf=0.0),
        np.asarray(exit_bars, dtype=np.int64),
        np.asarray(exit_reasons, dtype=np.int64),
    )


def _asset_overlap_keep_mask(
    timestamps: np.ndarray,
    assets: np.ndarray,
    exit_bars: np.ndarray | None = None,
    *,
    priority: np.ndarray | None = None,
    bar_minutes: int = 15,
    cooldown_hours: float = 0.0,
    use_numba: bool = True,
) -> np.ndarray:
    """Enforce one active trade per asset at a time, with optional cooldown.

    Trades are considered in chronological order, with higher-priority trades
    winning ties at the same timestamp for the same asset.
    
    Args:
        timestamps: Trade timestamps
        assets: Asset identifiers
        exit_bars: Number of bars until exit
        priority: Priority scores (higher is better)
        bar_minutes: Minutes per bar
        cooldown_hours: Cooldown period in hours
        use_numba: Use Numba-optimized version (default True)
    """
    n = len(assets)
    if n == 0:
        return np.asarray([], dtype=bool)
    
    if exit_bars is None:
        exit_bars_arr = np.zeros(n, dtype=np.int64)
    else:
        exit_bars_arr = np.asarray(exit_bars, dtype=np.int64)
        if len(exit_bars_arr) != n:
            raise ValueError("exit_bars must have the same length as assets")
    
    ts = pd.to_datetime(np.asarray(timestamps), utc=True, errors="coerce")
    assets_arr = np.asarray(assets).astype(str)
    
    if priority is None:
        priority_arr = np.zeros(n, dtype=np.float64)
    else:
        priority_arr = np.asarray(priority, dtype=np.float64)
        priority_arr = np.where(np.isfinite(priority_arr), priority_arr, 0.0)
    
    ts_i8 = ts.view("int64")
    valid = (~pd.isna(ts)) & np.isfinite(ts_i8)
    
    if not np.any(valid):
        return np.zeros(n, dtype=bool)
    
    # Try Numba version first
    if use_numba and NUMBA_AVAILABLE and n > 100:
        try:
            return _asset_overlap_keep_mask_numba(
                ts_i8=ts_i8,
                assets_arr=assets_arr,
                exit_bars_arr=exit_bars_arr,
                priority_arr=priority_arr,
                bar_minutes=bar_minutes,
                cooldown_hours=cooldown_hours,
                valid=valid,
            )
        except Exception:
            pass  # Fall back to pure Python
    
    # Pure Python fallback
    ord_idx = np.lexsort((
        np.arange(n, dtype=np.int64),
        -priority_arr,
        ts_i8,
    ))
    keep = np.zeros(n, dtype=bool)
    next_active_free: Dict[str, int] = {}
    next_cooldown_free: Dict[str, int] = {}
    bar_ns = int(pd.Timedelta(minutes=int(bar_minutes)).value)
    cooldown_ns = int(pd.Timedelta(hours=float(max(cooldown_hours, 0.0))).value)
    for i in ord_idx:
        if not valid[i]:
            continue
        asset = assets_arr[i]
        ts_cur = int(ts_i8[i])
        next_free_ts = max(
            next_active_free.get(asset, np.iinfo(np.int64).min),
            next_cooldown_free.get(asset, np.iinfo(np.int64).min),
        )
        if ts_cur < next_free_ts:
            continue
        keep[i] = True
        dur_bars = max(int(exit_bars_arr[i]) + 1, 1)
        next_active_free[asset] = ts_cur + dur_bars * bar_ns
        if cooldown_ns > 0:
            next_cooldown_free[asset] = ts_cur + cooldown_ns
    return keep


@jit(nopython=True, cache=True)
def _asset_overlap_keep_mask_numba(
    ts_i8: np.ndarray,
    assets_arr: np.ndarray,
    exit_bars_arr: np.ndarray,
    priority_arr: np.ndarray,
    bar_minutes: int,
    cooldown_hours: float,
    valid: np.ndarray,
) -> np.ndarray:
    """Numba-optimized version of asset overlap mask.
    
    Uses hash-based asset tracking for efficiency.
    """
    n = len(assets_arr)
    keep = np.zeros(n, dtype=np.bool_)
    
    if n == 0:
        return keep
    
    # Simple hash for asset string to integer ID
    # Using a simple hash approach since Numba doesn't support dict
    asset_ids = np.zeros(n, dtype=np.int64)
    unique_assets = []
    
    for i in range(n):
        found = False
        for j, ua in enumerate(unique_assets):
            if assets_arr[i] == ua:
                asset_ids[i] = j
                found = True
                break
        if not found:
            asset_ids[i] = len(unique_assets)
            unique_assets.append(assets_arr[i])
    
    n_assets = len(unique_assets)
    
    # Track next free time for each asset
    # Using large negative value as "not set"
    next_active_free = np.full(n_assets, np.iinfo(np.int64).min, dtype=np.int64)
    next_cooldown_free = np.full(n_assets, np.iinfo(np.int64).min, dtype=np.int64)
    
    # Compute bar_ns
    bar_ns = bar_minutes * 60 * 1_000_000_000  # minutes to ns
    cooldown_ns = int(cooldown_hours * 3600 * 1_000_000_000)
    
    # Sort by timestamp and priority
    # Create sorted indices
    order = np.argsort(ts_i8)
    
    for idx in range(n):
        i = order[idx]
        if not valid[i]:
            continue
        
        asset_id = asset_ids[i]
        ts_cur = ts_i8[i]
        
        next_free_ts = next_active_free[asset_id]
        if next_cooldown_free[asset_id] > next_free_ts:
            next_free_ts = next_cooldown_free[asset_id]
        
        if ts_cur >= next_free_ts:
            keep[i] = True
            dur_bars = max(int(exit_bars_arr[i]) + 1, 1)
            next_active_free[asset_id] = ts_cur + dur_bars * bar_ns
            if cooldown_ns > 0:
                next_cooldown_free[asset_id] = ts_cur + cooldown_ns
    
    return keep


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
    use_parallel: bool = True,
) -> Optional[np.ndarray]:
    """Build k* labels via discrete argmax utility over limit offsets (0..k_max).

    Uses shared `simulate_trade_exit` through `_simulate_policy_utility_from_arrays`.
    Returns None when required path/price columns are unavailable.
    
    Args:
        trade_outcomes: DataFrame with trade data
        tick_size: Tick size for limit orders
        k_max: Maximum offset to try
        entry_fill_horizon_bars: Bars to wait for fill
        max_hold_bars: Maximum holding period
        tp_pct: Default take-profit percentage
        sl_pct: Default stop-loss percentage
        trailing_pct: Default trailing percentage
        cost_pct: Transaction cost
        eta: Fill horizon penalty
        tie_break_smallest_k: Prefer smaller k on ties
        use_parallel: Use parallel Numba implementation (default True)
    """
    req_cols = {"entry_price", "is_long", "future_opens", "future_highs", "future_lows", "future_closes"}
    if not req_cols.issubset(set(trade_outcomes.columns)):
        missing_cols = req_cols - set(trade_outcomes.columns)
        raise ValueError(
            f"Missing columns for compute_optimal_limit_offset_labels: {missing_cols}. "
            "Please ensure run_ridge_sizer properly loads future 15m price panels."
        )
    
    n_trades = len(trade_outcomes)
    
    # Try parallel version first
    if use_parallel and NUMBA_AVAILABLE:
        try:
            # Prepare data arrays
            entry_prices = np.asarray(trade_outcomes['entry_price'].values, dtype=np.float32)
            is_longs = np.asarray(trade_outcomes['is_long'].values, dtype=np.float32)
            
            # Get max bars across all trades
            max_global_bars = 0
            for idx in range(n_trades):
                row = trade_outcomes.iloc[idx]
                arr = np.asarray(row['future_closes'], dtype=np.float32)
                if len(arr) > max_global_bars:
                    max_global_bars = len(arr)
            
            if max_global_bars == 0:
                return np.zeros(n_trades, dtype=np.float32)
            
            # Pad arrays with float32
            future_opens = np.full((n_trades, max_global_bars), np.nan, dtype=np.float32)
            future_highs = np.full((n_trades, max_global_bars), np.nan, dtype=np.float32)
            future_lows = np.full((n_trades, max_global_bars), np.nan, dtype=np.float32)
            future_closes = np.full((n_trades, max_global_bars), np.nan, dtype=np.float32)
            
            for idx in range(n_trades):
                row = trade_outcomes.iloc[idx]
                opens = np.asarray(row['future_opens'], dtype=np.float32)
                highs = np.asarray(row['future_highs'], dtype=np.float32)
                lows = np.asarray(row['future_lows'], dtype=np.float32)
                closes = np.asarray(row['future_closes'], dtype=np.float32)
                min_len = min(len(opens), len(highs), len(lows), len(closes), max_global_bars)
                if min_len > 0:
                    future_opens[idx, :min_len] = opens[:min_len]
                    future_highs[idx, :min_len] = highs[:min_len]
                    future_lows[idx, :min_len] = lows[:min_len]
                    future_closes[idx, :min_len] = closes[:min_len]
            
            # Get policy params with float32
            sl_atr_mults = np.full(n_trades, np.nan, dtype=np.float32)
            tp_sl_ratios = np.full(n_trades, np.nan, dtype=np.float32)
            max_hold_bars_arr = np.full(n_trades, max_hold_bars, dtype=np.int64)
            giveback_pcts = np.full(n_trades, trailing_pct, dtype=np.float32)
            atr_12_15m = np.zeros(n_trades, dtype=np.float32)
            
            if 'label_policy_sl_atr_mult' in trade_outcomes.columns:
                sl_atr_mults = np.asarray(trade_outcomes['label_policy_sl_atr_mult'].values, dtype=np.float32)
            if 'label_policy_tp_sl_ratio' in trade_outcomes.columns:
                tp_sl_ratios = np.asarray(trade_outcomes['label_policy_tp_sl_ratio'].values, dtype=np.float32)
            if 'label_policy_max_hold_bars' in trade_outcomes.columns:
                max_hold_bars_arr = np.asarray(trade_outcomes['label_policy_max_hold_bars'].values, dtype=np.int64)
            if 'label_policy_giveback_pct' in trade_outcomes.columns:
                giveback_pcts = np.asarray(trade_outcomes['label_policy_giveback_pct'].values, dtype=np.float32)
            if 'atr_12_15m' in trade_outcomes.columns:
                atr_12_15m = np.asarray(trade_outcomes['atr_12_15m'].values, dtype=np.float32)
            
            # Handle NaN in arrays
            sl_atr_mults = np.nan_to_num(sl_atr_mults, nan=np.nan)
            tp_sl_ratios = np.nan_to_num(tp_sl_ratios, nan=np.nan)
            
            k_labels = _compute_limit_offset_single_trade(
                entry_prices=entry_prices,
                is_longs=is_longs,
                future_opens=future_opens,
                future_highs=future_highs,
                future_lows=future_lows,
                future_closes=future_closes,
                sl_atr_mults=sl_atr_mults,
                tp_sl_ratios=tp_sl_ratios,
                max_hold_bars_arr=max_hold_bars_arr,
                giveback_pcts=giveback_pcts,
                entry_fill_horizon_bars=entry_fill_horizon_bars,
                max_hold_bars=max_hold_bars,
                tick_size=tick_size,
                k_max=k_max,
                cost_pct=cost_pct,
                eta=eta,
                atr_12_15m=atr_12_15m,
                sl_pct_default=sl_pct,
                tp_pct_default=tp_pct,
                tie_break_smallest_k=tie_break_smallest_k,
            )
            return np.asarray(k_labels, dtype=np.float64)  # Convert back for compatibility
        except Exception as e:
            # Fall back to serial version if parallel fails
            tprint(f"  Parallel limit offset failed: {e}, using serial version")
    
    # Serial fallback (also use float32)
    k_labels = np.zeros(n_trades, dtype=np.float32)
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
        
        # Robustness: ensure arrays have at least 1 element and same length
        min_len = min(len(opens), len(highs), len(lows), len(closes))
        if min_len == 0:
            k_labels[i] = 0.0
            continue
        opens, highs, lows, closes = opens[:min_len], highs[:min_len], lows[:min_len], closes[:min_len]

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
            limit_price = entry_price * (1.0 - tick_size * k) if is_long else entry_price * (1.0 + tick_size * k)
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
        k_labels[i] = np.float32(best_k)
    return k_labels.astype(np.float64)


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
    signal_timeframe: str = "1h",
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
        candidates_df['timestamp'] = pd.to_datetime(candidates_df['timestamp'], utc=True, errors="coerce")
    
    # Results storage
    results = []
    
    # Process each candidate trade
    for idx, row in candidates_df.iterrows():
        signal_ts = pd.Timestamp(row['timestamp'])
        ts = causal_decision_timestamps(
            [signal_ts], timeframe=signal_timeframe
        )[0]
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
            assert_first_path_timestamp(
                first_path_ts=[opens.index[entry_idx]],
                signal_ts=[signal_ts],
                timeframe=signal_timeframe,
            )
            
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
                'timestamp': signal_ts,
                'decision_ts': ts,
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
    signal_timeframe: str = "1h",
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
        candidates_df['timestamp'] = pd.to_datetime(candidates_df['timestamp'], utc=True, errors="coerce")

    n_candidates = len(candidates_df)
    if n_candidates == 0:
        return pd.DataFrame()

    signal_ts_values = pd.to_datetime(
        candidates_df['timestamp'], utc=True, errors="coerce"
    )
    ts_values = causal_decision_timestamps(
        signal_ts_values, timeframe=signal_timeframe
    )
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
        assert_first_path_timestamp(
            first_path_ts=[price_index[entry_idx]],
            signal_ts=[signal_ts_values.iloc[i]],
            timeframe=signal_timeframe,
        )

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

            timestamps.append(signal_ts_values.iloc[i])
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
            'decision_ts': causal_decision_timestamps(
                [timestamps[i]], timeframe=signal_timeframe
            )[0],
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


class ConfidenceConditionalRegressionObjective:
    """Two-term objective aligned with 'only trust predictions when confident'."""
    def __init__(
        self,
        alpha=3.0,
        threshold=0.0,
        temperature=0.5,
        lambda_conf=0.005,
        hess_floor=1e-6,
        use_magnitude=False,
        eps=1e-6
    ):
        self.alpha = alpha
        self.threshold = threshold
        self.temperature = temperature
        self.lambda_conf = lambda_conf
        self.hess_floor = hess_floor
        self.use_magnitude = use_magnitude
        self.eps = eps

    def __call__(self, y_true, y_pred, sample_weight=None):
        y_true = np.asarray(y_true, dtype=np.float64)
        y_pred = np.asarray(y_pred, dtype=np.float64)
        if sample_weight is not None:
             sample_weight = np.asarray(sample_weight, dtype=np.float64)

        e = y_pred - y_true  # error

        def sigmoid(x):
            x = np.clip(x, -60.0, 60.0)
            return 1.0 / (1.0 + np.exp(-x))

        # ----- gate g(pred) and its derivatives g', g'' -----
        if self.use_magnitude:
            a = np.sqrt(y_pred**2 + self.eps)
            z = (a - self.threshold) / self.temperature
            s = sigmoid(z)
            t = s * (1.0 - s)

            a_prime = y_pred / a
            a_second = self.eps / (a**3)

            z_prime = a_prime / self.temperature
            z_second = a_second / self.temperature

            g = s
            g_prime = t * z_prime
            g_second = t * (1.0 - 2.0 * s) * (z_prime**2) + t * z_second
        else:
            z = (y_pred - self.threshold) / self.temperature
            s = sigmoid(z)
            t = s * (1.0 - s)

            g = s
            g_prime = t / self.temperature
            g_second = t * (1.0 - 2.0 * s) / (self.temperature**2)

        # ----- confidence-weighted squared loss term -----
        w = 1.0 + self.alpha * g
        w_prime = self.alpha * g_prime
        w_second = self.alpha * g_second

        # L1 = 0.5*w*e^2
        grad = w * e + 0.5 * w_prime * (e**2)
        hess = w + 2.0 * w_prime * e + 0.5 * w_second * (e**2)

        # ----- confidence reward term -----
        if self.lambda_conf != 0.0:
            grad -= self.lambda_conf * g_prime
            hess -= self.lambda_conf * g_second

        if sample_weight is not None:
            grad *= sample_weight
            hess *= sample_weight

        # keep Hessian positive for XGBoost
        hess = np.maximum(hess, self.hess_floor)
        return grad, hess

def make_confidence_conditional_regression_objective(
    alpha=3.0,          # boosts error penalty in high-confidence region
    threshold=0.0,      # location of confidence boundary in prediction space
    temperature=0.5,    # softness of transition (larger = smoother)
    lambda_conf=0.005,  # >0 encourages coverage/confidence (prevents "all low confidence")
    hess_floor=1e-6,    # numerical stability
    use_magnitude=False,# if True, gate on |prediction| instead of prediction
    eps=1e-6
):
    return ConfidenceConditionalRegressionObjective(
        alpha=alpha,
        threshold=threshold,
        temperature=temperature,
        lambda_conf=lambda_conf,
        hess_floor=hess_floor,
        use_magnitude=use_magnitude,
        eps=eps
    )


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


def _build_ridge_target_candidates(
    returns: np.ndarray,
    symbols: np.ndarray | None,
    timestamps: np.ndarray | None,
    *,
    cost_pct: float,
    clip_L: float,
) -> Dict[str, np.ndarray]:
    # Build net returns version for candidate generation (hurdle-centered)
    y_net = returns - cost_pct

    # Build vol proxy if symbols available
    vol = None
    if symbols is not None and len(np.unique(symbols)) > 1:
        vol = build_trade_vol_proxy(y_net, symbols, timestamps)

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
    return candidates


def _robust_target_scale(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return 1.0
    med = float(np.median(arr))
    mad = float(np.median(np.abs(arr - med)))
    scale = 1.4826 * mad
    if not np.isfinite(scale) or scale < 1e-6:
        q25, q75 = np.quantile(arr, [0.25, 0.75])
        scale = float((q75 - q25) / 1.349) if np.isfinite(q75 - q25) and abs(q75 - q25) > 1e-6 else float(np.std(arr))
    if not np.isfinite(scale) or scale < 1e-6:
        scale = 1.0
    return float(scale)


def _signed_huber_bridge(values: np.ndarray, delta: float) -> np.ndarray:
    """Signed Huber-style bridge target: quadratic near 0, linear in the tails."""
    x = np.asarray(values, dtype=np.float64)
    d = max(float(delta), 1e-8)
    ax = np.abs(x)
    out = np.where(ax <= d, np.sign(x) * (ax * ax) / (2.0 * d), x - np.sign(x) * (d / 2.0))
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float64)


def _normalize_target_for_blend(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    scale = _robust_target_scale(arr)
    return np.clip(arr / scale, -5.0, 5.0)


def _build_target_family_candidates(
    *,
    y_gross: np.ndarray,
    y_net: np.ndarray,
    symbols: np.ndarray | None,
    timestamps: np.ndarray | None,
    cost_pct: float,
    u_policy: np.ndarray | None = None,
    u_simple_tbm: np.ndarray | None = None,
    extra_utility_targets: Dict[str, np.ndarray] | None = None,
) -> Dict[str, tuple[np.ndarray, str]]:
    y_gross = np.asarray(y_gross, dtype=np.float64)
    y_net = np.asarray(y_net, dtype=np.float64)
    if symbols is None:
        symbols = np.full(len(y_net), "ALL", dtype=object)

    candidates: Dict[str, tuple[np.ndarray, str]] = {
        "raw_net": (y_net, "regression"),
        "winsorized_net": (build_target_winsorized(y_net, clip_L=0.02), "regression"),
        "rank_net": (build_target_rolling_rank(y_net, symbols), "ranking"),
        "rank_residual_net": (build_target_rolling_rank_residual(y_net, symbols), "ranking"),
    }

    if u_simple_tbm is not None:
        candidates["simple_tbm_utility"] = (np.asarray(u_simple_tbm, dtype=np.float64), "utility")
    if extra_utility_targets:
        for target_name, target_values in extra_utility_targets.items():
            if not target_name:
                continue
            candidates[str(target_name)] = (np.asarray(target_values, dtype=np.float64), "utility")

    if u_policy is not None:
        up = np.asarray(u_policy, dtype=np.float64)
        utility_scale = _robust_target_scale(up)
        clipped_up = np.clip(up, -3.0 * utility_scale, 3.0 * utility_scale)
        candidates["policy_utility"] = (up, "utility")
        candidates["clipped_u_policy"] = (clipped_up, "utility")
        if symbols is not None and len(np.unique(symbols)) > 1:
            vol = build_trade_vol_proxy(y_net, symbols, timestamps)
            atr_norm = up / np.maximum(vol, utility_scale)
        else:
            atr_norm = up / utility_scale
        candidates["atr_normalized_u_policy"] = (np.clip(atr_norm, -5.0, 5.0), "utility")
        huber_delta = max(utility_scale, max(cost_pct, 1e-4))
        huber_up = _signed_huber_bridge(up, delta=huber_delta)
        candidates["huber_utility"] = (huber_up, "utility")
        hybrid = 0.7 * _normalize_target_for_blend(y_net) + 0.3 * _normalize_target_for_blend(huber_up)
        candidates["hybrid_raw_huber"] = (hybrid, "hybrid")

    return candidates


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

    # Ensure symbols is usable (fallback to single symbol)
    if symbols is None:
        symbols = np.full(n, "ALL", dtype=object)

    candidates = _build_ridge_target_candidates(
        returns,
        symbols,
        timestamps,
        cost_pct=cost_pct,
        clip_L=clip_L,
    )

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


def run_target_family_ab(
    X: np.ndarray,
    y_gross: np.ndarray,
    y_net: np.ndarray,
    symbols: np.ndarray | None,
    timestamps: np.ndarray | None,
    *,
    cost_pct: float,
    topq: float,
    u_policy: np.ndarray | None = None,
    u_simple_tbm: np.ndarray | None = None,
    extra_utility_targets: Dict[str, np.ndarray] | None = None,
    trade_mask: np.ndarray | None = None,
    alpha: float = 0.5,
) -> Dict[str, Any]:
    """Compare target families using the same fast Ridge CV probe.

    Families are scored primarily by Top-Q policy utility when available, with
    IC retained for diagnostics. This is intentionally lightweight: the goal is
    to compare learnability of candidate targets, not to run a full HPO cycle.
    """
    n = len(y_gross)
    if n < 120:
        return {"status": "skipped", "reason": "too_few_rows", "n_rows": int(n)}

    if symbols is None:
        symbols = np.full(n, "ALL", dtype=object)

    candidates = _build_target_family_candidates(
        y_gross=np.asarray(y_gross, dtype=np.float64),
        y_net=np.asarray(y_net, dtype=np.float64),
        symbols=symbols,
        timestamps=timestamps,
        cost_pct=cost_pct,
        u_policy=np.asarray(u_policy, dtype=np.float64) if u_policy is not None else None,
        u_simple_tbm=np.asarray(u_simple_tbm, dtype=np.float64) if u_simple_tbm is not None else None,
        extra_utility_targets=extra_utility_targets,
    )

    rows: list[dict[str, Any]] = []
    from sklearn.linear_model import Ridge as SkRidge
    fold_size = max(1, n // 3)
    _tm = np.ones(n, dtype=bool) if trade_mask is None else np.asarray(trade_mask[:n], dtype=bool)
    up_all = np.asarray(u_policy[:n], dtype=float) if u_policy is not None else None

    for target_name, (target_vec, family) in candidates.items():
        fin = np.isfinite(target_vec)
        if int(np.sum(fin)) < 100:
            continue
        ics: list[float] = []
        fold_topq_u: list[float] = []
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
            y_tr, y_va = target_vec[tr_idx], target_vec[va_idx]
            mu = np.nanmean(X_tr, axis=0)
            sd = np.nanstd(X_tr, axis=0)
            sd = np.where(sd < 1e-9, 1.0, sd)
            X_tr_s = (X_tr - mu) / sd
            X_va_s = (X_va - mu) / sd
            try:
                mdl = SkRidge(alpha=alpha, fit_intercept=True)
                mdl.fit(X_tr_s, y_tr)
                pred = np.asarray(mdl.predict(X_va_s), dtype=float)
                oof_pred[va_idx] = pred
                y_ret = y_gross[va_idx]
                if np.std(pred) < 1e-12 or np.std(y_ret) < 1e-12:
                    ic_fold = 0.0
                else:
                    ic_fold = float(spearmanr(pred, y_ret).correlation)
                if np.isfinite(ic_fold):
                    ics.append(ic_fold)
                if up_all is not None:
                    valid = np.isfinite(pred) & np.isfinite(up_all[va_idx]) & _tm[va_idx]
                    if np.any(valid):
                        pred_use = pred[valid]
                        up_use = up_all[va_idx][valid]
                        k_top = max(1, int(np.ceil(float(topq) * len(pred_use))))
                        idx_top = np.argsort(pred_use)[-k_top:]
                        fold_topq_u.append(float(np.mean(up_use[idx_top])))
            except Exception:
                continue

        mean_ic = float(np.mean(ics)) if ics else float("nan")
        topq_mean_u = float(np.mean(fold_topq_u)) if fold_topq_u else float("nan")
        topq_std_u = float(np.std(fold_topq_u)) if fold_topq_u else float("nan")
        learnability = (
            topq_mean_u - 0.5 * topq_std_u
            if np.isfinite(topq_mean_u)
            else (mean_ic if np.isfinite(mean_ic) else -1e12)
        )
        rows.append({
            "target_name": target_name,
            "target_family": family,
            "ridge_ic": float(mean_ic) if np.isfinite(mean_ic) else float("nan"),
            "topq_policy_u_mean": float(topq_mean_u) if np.isfinite(topq_mean_u) else float("nan"),
            "topq_policy_u_std": float(topq_std_u) if np.isfinite(topq_std_u) else float("nan"),
            "learnability_score": float(learnability),
            "target_std": float(np.nanstd(target_vec)),
            "target_abs_q95": float(np.nanquantile(np.abs(target_vec[np.isfinite(target_vec)]), 0.95)) if np.any(np.isfinite(target_vec)) else float("nan"),
            "n_rows": int(n),
        })

    if not rows:
        return {"status": "skipped", "reason": "no_valid_candidates", "n_rows": int(n)}

    ranking = sorted(rows, key=lambda r: float(r.get("learnability_score", -1e12)), reverse=True)
    best = ranking[0]
    simpler = [r for r in ranking if r["target_name"] != "policy_utility"]
    best_simpler = simpler[0] if simpler else None
    return {
        "status": "ok",
        "winner": best,
        "best_simpler": best_simpler,
        "rows": ranking,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Huber Loss with L2 Regularization
# ═══════════════════════════════════════════════════════════════════════════════

@jit(nopython=True, cache=True)
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
    y_true = np.ascontiguousarray(y_true.astype(np.float32))
    y_pred = np.ascontiguousarray(y_pred.astype(np.float32))
    d = np.float32(delta)
    
    residual = y_true - y_pred
    abs_residual = np.abs(residual)
    
    # Huber loss formula
    quadratic = np.minimum(abs_residual, d)
    linear = abs_residual - quadratic
    
    loss = np.float32(0.5) * quadratic ** 2 + d * linear
    
    if sample_weight is not None:
        sw = np.ascontiguousarray(sample_weight.astype(np.float32))
        loss = loss * sw
        return float(np.sum(loss) / (np.sum(sw) + np.float32(1e-12)))
    
    return float(np.mean(loss))


@jit(nopython=True, cache=True)
def huber_loss_gradient(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    delta: float = 1.0,
    sample_weight: np.ndarray | None = None,
) -> np.ndarray:
    """Compute gradient of Huber loss with respect to predictions.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        delta: Threshold for quadratic vs linear loss
        sample_weight: Optional sample weights
        
    Returns:
        Gradient array
    """
    y_t = np.ascontiguousarray(y_true.astype(np.float32))
    y_p = np.ascontiguousarray(y_pred.astype(np.float32))
    d = np.float32(delta)
    
    residual = y_p - y_t  # Note: gradient w.r.t. prediction
    abs_residual = np.abs(residual)
    
    # Gradient: residual for quadratic region, delta * sign(residual) for linear
    grad = np.where(
        abs_residual <= d,
        residual,
        d * np.sign(residual)
    ).astype(np.float32)
    
    if sample_weight is not None:
        sw = np.ascontiguousarray(sample_weight.astype(np.float32))
        # Match the weighted objective: sum(w_i * L_i) / sum(w_i)
        # Gradient is: w_i * g_i / sum(w_i)
        div = np.sum(sw) + np.float32(1e-12)
        grad = (grad * sw / div).astype(np.float32)
    
    return grad.astype(np.float32)


# ═══════════════════════════════════════════════════════════════════════════════
# Numba-Optimized Position Sizing Formulas
# ═══════════════════════════════════════════════════════════════════════════════

@jit(nopython=True, cache=True)
def _compute_position_sizes_fixed(x: np.ndarray, base_size: float) -> np.ndarray:
    """Fixed position sizing."""
    return np.full_like(x, base_size, dtype=np.float32)


@jit(nopython=True, cache=True)
def _compute_position_sizes_linear(x: np.ndarray, base_size: float, rank_multiplier: float) -> np.ndarray:
    """Linear position sizing."""
    return (np.float32(base_size) + np.float32(rank_multiplier) * x.astype(np.float32)).astype(np.float32)


@jit(nopython=True, cache=True)
def _compute_position_sizes_convex(x: np.ndarray, base_size: float, rank_multiplier: float) -> np.ndarray:
    """Convex position sizing - rewards high confidence more."""
    return (np.float32(base_size) + np.float32(rank_multiplier) * (x.astype(np.float32) ** np.float32(2.0))).astype(np.float32)


@jit(nopython=True, cache=True)
def _compute_position_sizes_concave(x: np.ndarray, base_size: float, rank_multiplier: float) -> np.ndarray:
    """Concave position sizing - conservative, rewards marginal picks."""
    return (np.float32(base_size) + np.float32(rank_multiplier) * np.sqrt(x.astype(np.float32))).astype(np.float32)


@jit(nopython=True, cache=True)
def _compute_position_sizes_sigmoid(x: np.ndarray, base_size: float, rank_multiplier: float, squash_k: float) -> np.ndarray:
    """Sigmoid position sizing - smooth transition around midpoint."""
    k = np.float32(squash_k) * np.float32(10.0)
    denom = np.float32(1.0) + np.exp(-k * (x.astype(np.float32) - np.float32(0.5)))
    return (np.float32(base_size) + np.float32(rank_multiplier) / denom).astype(np.float32)


@jit(nopython=True, cache=True)
def _compute_position_sizes_exponential(x: np.ndarray, base_size: float, rank_multiplier: float, squash_k: float) -> np.ndarray:
    """Exponential position sizing - aggressive for high confidence."""
    k = np.float32(squash_k) * np.float32(3.0)
    exp_k = np.exp(k)
    num = np.exp(k * x.astype(np.float32)) - np.float32(1.0)
    den = exp_k - np.float32(1.0) + np.float32(1e-12)
    return (np.float32(base_size) + np.float32(rank_multiplier) * num / den).astype(np.float32)


@jit(nopython=True, cache=True)
def _apply_sizing_formula(
    x: np.ndarray,
    base_size: float,
    rank_multiplier: float,
    sizing_formula: int,
    squash_k: float,
) -> np.ndarray:
    """Apply sizing formula based on formula code.
    
    Args:
        x: Normalized predictions in [0, 1]
        base_size: Base position size
        rank_multiplier: Rank-based multiplier
        sizing_formula: 0=fixed, 1=linear, 2=convex, 3=concave, 4=sigmoid, 5=exponential
        squash_k: Squash steepness parameter
    
    Returns:
        Position sizes
    """
    if sizing_formula == 0:  # fixed
        return _compute_position_sizes_fixed(x, base_size)
    elif sizing_formula == 1:  # linear
        return _compute_position_sizes_linear(x, base_size, rank_multiplier)
    elif sizing_formula == 2:  # convex
        return _compute_position_sizes_convex(x, base_size, rank_multiplier)
    elif sizing_formula == 3:  # concave
        return _compute_position_sizes_concave(x, base_size, rank_multiplier)
    elif sizing_formula == 4:  # sigmoid
        return _compute_position_sizes_sigmoid(x, base_size, rank_multiplier, squash_k)
    elif sizing_formula == 5:  # exponential
        return _compute_position_sizes_exponential(x, base_size, rank_multiplier, squash_k)
    else:  # default to linear
        return _compute_position_sizes_linear(x, base_size, rank_multiplier)


@jit(nopython=True, cache=True)
def _aggregate_daily_returns_numpy(returns: np.ndarray, timestamps_ns: np.ndarray) -> np.ndarray:
    """Aggregate returns to daily using numpy (faster than pandas).
    
    Args:
        returns: Array of returns
        timestamps_ns: Array of timestamps in nanoseconds
    
    Returns:
        Daily returns array
    """
    n = len(returns)
    if n == 0:
        return np.array([], dtype=np.float32)
    
    # Convert to days (nanoseconds to days)
    day_ns = 86400_000_000_000  # nanoseconds in a day
    days = (timestamps_ns // day_ns).astype(np.int64)
    
    # Get unique days
    unique_days = np.unique(days)
    n_days = len(unique_days)
    
    # Aggregate returns per day
    daily_returns = np.zeros(n_days, dtype=np.float32)
    for i in range(n_days):
        day_mask = days == unique_days[i]
        daily_returns[i] = np.sum(returns[day_mask])
    
    return daily_returns


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
        alpha_range: Tuple[float, float] = (1e-3, 5.0),
        delta_range: Tuple[float, float] = (0.5, 2.0),
        n_grid_points: int = 10,
        cost_pct: float = 0.0025,
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
        use_nested_cv: bool = False,
        use_rolling_walk_forward: bool = True,  # Enable rolling walk-forward by default
        max_fit_samples: int | None = 8000,
        n_jobs: int = 2,
        patience: int = 25,
        stage1_cv_folds: int = 1,
        stage1_two_fold_refine: bool = False,
        stage1_n_trials: int | None = None,
        stage2_cv_folds: int = 1,
        stage2_n_trials: int | None = None,
        tree_hpo_trials: int | None = None,
        position_hard_cap: float = 0.20,
        target_train_fraction: float = 0.50,
        oos_fraction: float = 0.30,
        min_oos_days: int = 14,
        repeated_oos_splits: int = 3,
        stage2_lock_formula: bool = True,
        forced_target_candidates: List[str] | None = None,
    ):
        """Initialize the Ridge Position Sizer.

        Args:
            gamma_range: Range for asymmetric weight parameter (losing trades weight)
            alpha_range: Range for L2 regularization strength (expanded to 1.0 for stronger regularization)
            delta_range: Range for Huber loss delta parameter
            n_grid_points: Number of grid points for hyperparameter search
            cost_pct: Transaction cost percentage for label computation
            sum_to_one: If True, constrain weights to sum to 1
            non_negative: If True, constrain weights to be non-negative
            top_k_pct: Percentage of top predictions to select for evaluation
            top_k_hard_cap: Optional hard cap applied to top_k_pct during evaluation
            returns_are_net: True if `return`/labels already include cost
            random_state: Random seed for reproducibility
            use_nested_cv: If True, use nested cross-validation for unbiased hyperparameter tuning
            use_rolling_walk_forward: If True, use rolling walk-forward CV instead of single split
            max_fit_samples: Maximum number of chronologically earliest samples to use
                for training/tuning. Remaining later samples are reserved for OOS/reporting.
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
        self.use_nested_cv = bool(use_nested_cv)
        self.use_rolling_walk_forward = bool(use_rolling_walk_forward)
        self.max_fit_samples = None if max_fit_samples is None else int(max_fit_samples)
        self.n_jobs = int(n_jobs)
        self.patience = int(patience)
        self.stage1_cv_folds = max(1, int(stage1_cv_folds)) if self.use_nested_cv else 1
        self.stage1_two_fold_refine = bool(stage1_two_fold_refine) if self.use_nested_cv else False
        self.stage1_n_trials = int(stage1_n_trials) if stage1_n_trials is not None else (60 if self.use_nested_cv else 100)
        self.stage2_cv_folds = max(1, int(stage2_cv_folds)) if self.use_nested_cv else 1
        self.stage2_n_trials = int(stage2_n_trials) if stage2_n_trials is not None else (20 if self.use_nested_cv else 40)
        self.tree_hpo_trials = int(tree_hpo_trials) if tree_hpo_trials is not None else 100
        self.position_hard_cap = float(position_hard_cap)
        self.target_train_fraction = float(np.clip(target_train_fraction, 0.30, 0.80))
        self.oos_fraction = float(np.clip(oos_fraction, 0.10, 0.50))
        self.min_oos_days = max(3, int(min_oos_days))
        self.repeated_oos_splits = max(1, int(repeated_oos_splits))
        self.stage2_lock_formula = bool(stage2_lock_formula)
        self.forced_target_candidates = [str(x) for x in (forced_target_candidates or []) if str(x).strip()]
        
        # Fitted attributes
        self.weights_: Optional[np.ndarray] = None
        self.model_names_: Optional[List[str]] = None
        self.best_params_: Optional[Dict] = None
        self.target_race_metrics_: Optional[Dict[str, Any]] = None
        self.target_race_results_: Optional[pd.DataFrame] = None
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
        self.entry_policy_config_: Optional[Dict[str, Any]] = None
        self.candidate_threshold_config_: Optional[Dict[str, Any]] = None
        self.threshold_low_: Optional[float] = None
        self.threshold_high_: Optional[float] = None
        self.is_fitted_: bool = False

        # Nested CV and backtest attributes
        self.nested_cv_results_: Optional[pd.DataFrame] = None
        self.best_nested_cv_params_: Optional[Dict] = None
        self.backtest_metrics_: Optional[Dict] = None
        self.cv_backtest_consistency_: Optional[Dict] = None
        self.cv_summary_: Optional[Dict[str, Any]] = None
        self.full_oos_metrics_: Optional[Dict[str, Any]] = None
        self.repeated_oos_results_: Optional[pd.DataFrame] = None
        self.oos_protocol_: Optional[Dict[str, Any]] = None
        self.feature_ic_diag_: Optional[pd.DataFrame] = None
        self.alpha_retention_: Optional[Dict[str, Any]] = None
        self.bucket_name_: Optional[str] = None
        self.target_family_ab_: Optional[Dict[str, Any]] = None
        self.selected_training_target_name_: Optional[str] = None
        self.selected_training_target_family_: Optional[str] = None

    @staticmethod
    def _formula_to_code(sizing_formula: str) -> int:
        if sizing_formula == "fixed":
            return 0
        if sizing_formula == "convex":
            return 2
        if sizing_formula == "concave":
            return 3
        if sizing_formula == "sigmoid":
            return 4
        if sizing_formula == "exponential":
            return 5
        return 1

    def _score_to_position_fraction(
        self,
        score: np.ndarray,
        *,
        base_size: float,
        rank_multiplier: float,
        sizing_formula: str,
        squash_k: float,
    ) -> np.ndarray:
        score = np.asarray(score, dtype=np.float32)
        if score.size == 0:
            return score.astype(np.float64)
        finite = np.isfinite(score)
        out = np.zeros(score.shape[0], dtype=np.float64)
        if not np.any(finite):
            return out
        score_f = score[finite]
        s_min = float(np.min(score_f))
        s_max = float(np.max(score_f))
        s_rng = max(s_max - s_min, 1e-12)
        x = np.clip((score_f - s_min) / s_rng, 0.0, 1.0).astype(np.float32)
        pos = _apply_sizing_formula(
            x,
            np.float32(base_size),
            np.float32(rank_multiplier),
            self._formula_to_code(str(sizing_formula)),
            np.float32(squash_k),
        )
        out[finite] = np.asarray(pos, dtype=np.float64)
        return np.clip(out, 0.0, float(min(base_size + rank_multiplier, self.position_hard_cap)))

    @staticmethod
    def _select_offset_target_column(trade_outcomes: pd.DataFrame) -> str | None:
        for candidate_col in ("k_star", "optimal_offset_ticks", "limit_offset_k", "target_offset_ticks"):
            if candidate_col in trade_outcomes.columns:
                return candidate_col
        return None

    def _fit_fold_offset_model(
        self,
        X_tr: np.ndarray,
        X_va: np.ndarray,
        score_tr: np.ndarray,
        trade_outcomes_tr: pd.DataFrame,
        sample_weight_tr: np.ndarray | None = None,
    ) -> np.ndarray:
        k_col = self._select_offset_target_column(trade_outcomes_tr)
        if k_col is not None:
            k_target = np.clip(np.nan_to_num(trade_outcomes_tr[k_col].to_numpy(dtype=np.float64), nan=0.0), 0.0, 5.0)
        else:
            try:
                k_target = np.clip(
                    np.nan_to_num(
                        compute_optimal_limit_offset_labels(
                            trade_outcomes_tr,
                            tick_size=self.best_params_.get("tick_size_bps", 2.0) / 10000.0 if self.best_params_ else 2.0 / 10000.0,
                            k_max=5,
                            entry_fill_horizon_bars=4,
                            max_hold_bars=48,
                            tp_pct=0.005,
                            sl_pct=0.0025,
                            trailing_pct=0.0,
                            cost_pct=self.cost_pct,
                            eta=0.0,
                            tie_break_smallest_k=True,
                        ),
                        nan=0.0,
                    ),
                    0.0,
                    5.0,
                )
            except Exception:
                return np.zeros(len(X_va), dtype=np.float64)
        if len(k_target) != len(X_tr):
            return np.zeros(len(X_va), dtype=np.float64)
        X_tr_off = np.column_stack([np.nan_to_num(X_tr, nan=0.0), np.asarray(score_tr, dtype=np.float64)])
        X_va_off = np.column_stack([np.nan_to_num(X_va, nan=0.0), np.zeros(len(X_va), dtype=np.float64)])
        pipe = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("ridge", Ridge(alpha=1.0)),
            ]
        )
        try:
            pipe.fit(X_tr_off, k_target, ridge__sample_weight=sample_weight_tr)
        except TypeError:
            pipe.fit(X_tr_off, k_target)
        return np.clip(np.asarray(pipe.predict(X_va_off), dtype=np.float64), 0.0, 5.0)

    def _bucket_search_space(self) -> Dict[str, Any]:
        bucket = str(getattr(self, "bucket_name_", "") or "").lower()
        is_tf = "_tf" in bucket
        if is_tf:
            return {
                "train_top_k_choices": [0.10, 0.20, 0.30, 0.40, 0.50],
                "exec_top_k_choices": [0.02, 0.05, 0.10, 0.15, 0.20, 0.30],
                "cooldown_choices": [0.5, 1.0, 2.0, 3.0],
                "base_size_range": (0.00, 0.04, 0.005),
                "rank_multiplier_range": (0.04, 0.16, 0.02),
                "squash_k_choices": [1.0, 1.5, 2.0],
            }
        return {
            "train_top_k_choices": [0.10, 0.20, 0.30, 0.40, 0.50],
            "exec_top_k_choices": [0.05, 0.10, 0.15, 0.20, 0.30],
            "cooldown_choices": [0.5, 1.0, 1.5, 2.0, 3.0],
            "base_size_range": (0.02, 0.10, 0.01),
            "rank_multiplier_range": (0.04, 0.16, 0.02),
            "squash_k_choices": [0.75, 1.0, 1.25, 1.5],
        }

    def _compute_feature_ic_diag(
        self,
        X: np.ndarray,
        y_gross: np.ndarray,
        feature_names: list[str],
    ) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        y_s = pd.Series(np.asarray(y_gross, dtype=float))
        for j, name in enumerate(feature_names):
            x = pd.Series(np.asarray(X[:, j], dtype=float))
            std = float(np.nanstd(x))
            finite = np.isfinite(x) & np.isfinite(y_s)
            if int(np.sum(finite)) < 20 or std < 1e-10:
                ic = np.nan
            else:
                try:
                    ic = float(x[finite].corr(y_s[finite], method="spearman"))
                except Exception:
                    ic = np.nan
            rows.append({
                "feature": str(name),
                "spearman_ic": ic,
                "std": std,
                "is_constant": bool(std < 1e-10),
            })
        diag = pd.DataFrame(rows)
        self.feature_ic_diag_ = diag
        return diag

    def _select_feature_keep_mask(
        self,
        X: np.ndarray,
        y_gross: np.ndarray,
        feature_names: list[str],
    ) -> np.ndarray:
        diag = self._compute_feature_ic_diag(X, y_gross, feature_names)
        if diag.empty:
            return np.ones(len(feature_names), dtype=bool)
        diag["priority"] = diag["feature"].astype(str).str.contains(r"oof_p_|reg_|clf|base_", regex=True)
        good = (~diag["is_constant"]) & (
            diag["spearman_ic"].fillna(-1.0) >= -0.01
        )
        keep = good.to_numpy(dtype=bool)
        if int(np.sum(keep)) < min(6, len(feature_names)):
            ranked = diag.sort_values(["priority", "spearman_ic"], ascending=[False, False]).index.to_numpy(dtype=int)
            keep = np.zeros(len(feature_names), dtype=bool)
            keep[ranked[:min(max(6, len(feature_names) // 2), len(feature_names))]] = True
        return keep

    def _select_feature_keep_mask_family(
        self,
        X: np.ndarray,
        target_vec: np.ndarray,
        feature_names: list[str],
        *,
        target_family: str,
        utility_ref: np.ndarray | None = None,
    ) -> np.ndarray:
        target_arr = np.asarray(target_vec, dtype=float)
        util_arr = np.asarray(utility_ref, dtype=float) if utility_ref is not None else None
        topq = float(min(max(self.select_topq, 0.05), 0.50))
        rows: list[dict[str, Any]] = []

        for j, name in enumerate(feature_names):
            x = np.asarray(X[:, j], dtype=float)
            std = float(np.nanstd(x))
            finite = np.isfinite(x) & np.isfinite(target_arr)
            corr = np.nan
            family_score = -1e12
            top_mean = np.nan
            baseline = np.nan
            if int(np.sum(finite)) >= 20 and std >= 1e-10:
                try:
                    corr = float(pd.Series(x[finite]).corr(pd.Series(target_arr[finite]), method="spearman"))
                except Exception:
                    corr = np.nan
                x_use = x[finite]
                order = np.argsort(x_use)
                k_top = max(1, int(np.ceil(topq * len(order))))
                top_idx = order[-k_top:]
                if target_family == "ranking":
                    ref = target_arr[finite]
                    top_mean = float(np.mean(ref[top_idx]))
                    baseline = float(np.mean(ref))
                    family_score = top_mean - baseline
                elif target_family == "utility":
                    ref_all = util_arr if util_arr is not None and len(util_arr) == len(target_arr) else target_arr
                    ref = np.asarray(ref_all[finite], dtype=float)
                    top_mean = float(np.mean(ref[top_idx]))
                    baseline = float(np.mean(ref))
                    family_score = top_mean - baseline
                else:
                    family_score = float(corr) if np.isfinite(corr) else -1e12
            rows.append({
                "feature": str(name),
                "feature_score": float(family_score) if np.isfinite(family_score) else -1e12,
                "spearman_target": float(corr) if np.isfinite(corr) else np.nan,
                "top_mean": float(top_mean) if np.isfinite(top_mean) else np.nan,
                "baseline": float(baseline) if np.isfinite(baseline) else np.nan,
                "std": std,
                "is_constant": bool(std < 1e-10),
            })

        diag = pd.DataFrame(rows)
        if diag.empty:
            return np.ones(len(feature_names), dtype=bool)
        diag["priority"] = diag["feature"].astype(str).str.contains(r"oof_p_|reg_|clf|base_", regex=True)
        if target_family in {"ranking", "utility"}:
            good = (~diag["is_constant"]) & (diag["feature_score"].fillna(-1e12) > 0.0)
        else:
            good = (~diag["is_constant"]) & (diag["feature_score"].fillna(-1.0) >= -0.01)
        keep = good.to_numpy(dtype=bool)
        if int(np.sum(keep)) < min(6, len(feature_names)):
            ranked = diag.sort_values(["priority", "feature_score"], ascending=[False, False]).index.to_numpy(dtype=int)
            keep = np.zeros(len(feature_names), dtype=bool)
            keep[ranked[:min(max(6, len(feature_names) // 2), len(feature_names))]] = True
        return keep

    def _evaluate_ranking_from_scores(
        self,
        *,
        score: np.ndarray,
        returns: np.ndarray,
        top_k_pct: float,
        pipeline_name: str = "ranking_eval",
    ) -> Dict[str, Any]:
        score = np.asarray(score, dtype=np.float64)
        returns = np.asarray(returns, dtype=np.float64)
        valid = np.isfinite(score) & np.isfinite(returns)
        if not np.any(valid):
            return {
                "RankingObjective": -1e9,
                "RankingIC": 0.0,
                "RankingMeanGross": 0.0,
                "RankingMeanNet": 0.0,
                "RankingHitAboveCost": 0.0,
                "RankingNSelected": 0,
                "pipeline": pipeline_name,
            }
        score = score[valid]
        returns = returns[valid]
        k_keep = max(1, int(np.ceil(float(top_k_pct) * len(score))))
        ord_idx = np.lexsort((np.arange(len(score), dtype=np.int64), -score))
        sel = ord_idx[:k_keep]
        gross = returns[sel]
        net = gross - float(self.cost_pct)
        try:
            ic = float(spearmanr(score, returns).correlation)
            if not np.isfinite(ic):
                ic = 0.0
        except Exception:
            ic = 0.0
        mean_net = float(np.mean(net)) if len(net) else 0.0
        ranking_objective = float(mean_net * np.sqrt(max(len(net), 1)) + 0.05 * ic)
        return {
            "RankingObjective": ranking_objective,
            "RankingIC": ic,
            "RankingMeanGross": float(np.mean(gross)) if len(gross) else 0.0,
            "RankingMeanNet": mean_net,
            "RankingHitAboveCost": float(np.mean(gross > float(self.cost_pct))) if len(gross) else 0.0,
            "RankingNSelected": int(len(gross)),
            "pipeline": pipeline_name,
        }

    def _evaluate_live_pipeline_from_scores(
        self,
        *,
        score: np.ndarray,
        trade_outcomes: pd.DataFrame | None,
        timestamps: np.ndarray | None,
        top_k_pct: float,
        cooldown_hours: float,
        base_size: float,
        rank_multiplier: float,
        sizing_formula: str,
        squash_k: float,
        offset_k_pred: np.ndarray | None = None,
        pipeline_name: str = "joint_eval",
        include_deciles: bool = False,
        decile_prefix: str = "eval",
    ) -> Dict[str, Any]:
        score = np.asarray(score, dtype=np.float64)
        if trade_outcomes is None or len(score) == 0 or len(trade_outcomes) == 0:
            return {
                "PnL_total": 0.0,
                "PnL_per_day": 0.0,
                "Trades_per_day": 0.0,
                "Sortino": 0.0,
                "MaxDD": 0.0,
                "Ulcer": 0.0,
                "TUW": 0.0,
                "IntradayRisk": 0.0,
                "TemporalStability": 0.0,
                "TemporalInstability": 1.0,
                "ObjectiveScore": -1e9,
                "ProfitFactor": 0.0,
                "AvgWin": 0.0,
                "AvgLoss": 0.0,
                "WinRate": 0.0,
                "N_selected": 0,
                "pipeline": pipeline_name,
                "N_raw_candidates": 0,
                "N_finite_scores": 0,
                "N_after_topk": 0,
                "N_after_size": 0,
                "N_after_overlap": 0,
            }
        n_eval = min(len(score), len(trade_outcomes))
        score = score[:n_eval]
        trades_df = trade_outcomes.iloc[:n_eval].copy()
        ts_eval = np.asarray(timestamps[:n_eval]) if timestamps is not None else None
        finite_score = np.isfinite(score)
        if not np.any(finite_score):
            return {
                "PnL_total": 0.0,
                "PnL_per_day": 0.0,
                "Trades_per_day": 0.0,
                "Sortino": 0.0,
                "MaxDD": 0.0,
                "Ulcer": 0.0,
                "TUW": 0.0,
                "IntradayRisk": 0.0,
                "TemporalStability": 0.0,
                "TemporalInstability": 1.0,
                "ObjectiveScore": -1e9,
                "ProfitFactor": 0.0,
                "AvgWin": 0.0,
                "AvgLoss": 0.0,
                "WinRate": 0.0,
                "N_selected": 0,
                "pipeline": pipeline_name,
                "N_raw_candidates": int(n_eval),
                "N_finite_scores": 0,
                "N_after_topk": 0,
                "N_after_size": 0,
                "N_after_overlap": 0,
            }

        n_rank = int(np.count_nonzero(finite_score))
        k_keep = max(1, int(np.ceil(float(top_k_pct) * n_rank)))
        valid_idx = np.flatnonzero(finite_score)
        ord_idx = valid_idx[np.lexsort((np.arange(n_rank, dtype=np.int64), -score[finite_score]))]
        keep_idx = ord_idx[:k_keep]
        keep_mask = np.zeros(n_eval, dtype=bool)
        keep_mask[keep_idx] = True
        n_after_topk = int(np.count_nonzero(keep_mask))

        size_full = self._score_to_position_fraction(
            score,
            base_size=base_size,
            rank_multiplier=rank_multiplier,
            sizing_formula=sizing_formula,
            squash_k=squash_k,
        )
        size_full = np.where(keep_mask, size_full, 0.0)
        active = keep_mask & np.isfinite(size_full) & (size_full > 0.0)
        n_after_size = int(np.count_nonzero(active))
        if not np.any(active):
            return {
                "PnL_total": 0.0,
                "PnL_per_day": 0.0,
                "Trades_per_day": 0.0,
                "Sortino": 0.0,
                "MaxDD": 0.0,
                "Ulcer": 0.0,
                "TUW": 0.0,
                "IntradayRisk": 0.0,
                "TemporalStability": 0.0,
                "TemporalInstability": 1.0,
                "ObjectiveScore": -1e9,
                "ProfitFactor": 0.0,
                "AvgWin": 0.0,
                "AvgLoss": 0.0,
                "WinRate": 0.0,
                "N_selected": 0,
                "pipeline": pipeline_name,
                "N_raw_candidates": int(n_eval),
                "N_finite_scores": int(n_rank),
                "N_after_topk": int(n_after_topk),
                "N_after_size": int(n_after_size),
                "N_after_overlap": 0,
            }

        kept_df = trades_df.iloc[np.flatnonzero(active)].copy()
        kept_sizes = np.asarray(size_full[active], dtype=np.float64)
        score_kept = np.asarray(score[active], dtype=np.float64)
        kept_offsets_k = np.asarray(offset_k_pred[:n_eval][active], dtype=np.float64) if offset_k_pred is not None else np.zeros(len(kept_df), dtype=np.float64)
        tick_size_bps = float(self.best_params_.get("tick_size_bps", 2.0)) if self.best_params_ else 2.0
        kept_offsets = (tick_size_bps / 10000.0) * np.clip(kept_offsets_k, 0.0, 5.0)

        ts_kept = np.asarray(ts_eval)[active] if ts_eval is not None else None
        asset_kept = kept_df["symbol"].to_numpy(dtype=object) if "symbol" in kept_df.columns else np.asarray(["UNKNOWN"] * len(kept_df), dtype=object)

        if "u_policy_net" in kept_df.columns:
            net_ref = kept_df["u_policy_net"].to_numpy(dtype=np.float64) + kept_offsets
            exit_bars_ref = kept_df.get("duration", pd.Series(48, index=kept_df.index)).to_numpy(dtype=np.int64)
        elif "u_policy" in kept_df.columns:
            net_ref = kept_df["u_policy"].to_numpy(dtype=np.float64) - self.cost_pct + kept_offsets
            exit_bars_ref = kept_df.get("duration", pd.Series(48, index=kept_df.index)).to_numpy(dtype=np.int64)
        else:
            req_cols = {"entry_price", "is_long", "future_opens", "future_highs", "future_lows", "future_closes"}
            if not req_cols.issubset(set(kept_df.columns)):
                raw_ref = kept_df.get("return", pd.Series(0.0, index=kept_df.index)).to_numpy(dtype=np.float64)
                net_ref = raw_ref - self.cost_pct + kept_offsets
                exit_bars_ref = kept_df.get("duration", pd.Series(48, index=kept_df.index)).to_numpy(dtype=np.int64)
            else:
                max_stack_bars = int(np.nanmax(kept_df.get("label_policy_max_hold_bars", pd.Series(48, index=kept_df.index)))) if len(kept_df) else 48
                opens_2d, open_lens = _stack_object_path_column(kept_df["future_opens"].values, max_stack_bars)
                highs_2d, high_lens = _stack_object_path_column(kept_df["future_highs"].values, max_stack_bars)
                lows_2d, low_lens = _stack_object_path_column(kept_df["future_lows"].values, max_stack_bars)
                closes_2d, close_lens = _stack_object_path_column(kept_df["future_closes"].values, max_stack_bars)
                entry_px_raw = kept_df["entry_price"].to_numpy(dtype=np.float64)
                is_long_mask = kept_df["is_long"].to_numpy(dtype=bool)
                eff_entry = np.where(is_long_mask, entry_px_raw - kept_offsets * entry_px_raw, entry_px_raw + kept_offsets * entry_px_raw)
                sl_mult = kept_df.get("label_policy_sl_atr_mult", pd.Series(np.nan, index=kept_df.index)).to_numpy(dtype=np.float64)
                tp_ratio = kept_df.get("label_policy_tp_sl_ratio", pd.Series(np.nan, index=kept_df.index)).to_numpy(dtype=np.float64)
                atr_entry = kept_df.get("atr_12_15m", pd.Series(np.nan, index=kept_df.index)).to_numpy(dtype=np.float64)
                use_policy = np.isfinite(sl_mult) & np.isfinite(tp_ratio) & np.isfinite(atr_entry)
                sl_pct = np.full(len(kept_df), 0.0025, dtype=np.float64)
                tp_pct = np.full(len(kept_df), 0.0050, dtype=np.float64)
                if np.any(use_policy):
                    sl_abs = np.maximum(sl_mult[use_policy] * np.maximum(atr_entry[use_policy], 1e-9), 1e-9)
                    tp_abs = tp_ratio[use_policy] * sl_abs
                    sl_pct[use_policy] = sl_abs / np.maximum(entry_px_raw[use_policy], 1e-9)
                    tp_pct[use_policy] = tp_abs / np.maximum(entry_px_raw[use_policy], 1e-9)
                trailing_pct = kept_df.get("label_policy_giveback_pct", pd.Series(0.0, index=kept_df.index)).to_numpy(dtype=np.float64)
                max_bars_arr = kept_df.get("label_policy_max_hold_bars", pd.Series(48, index=kept_df.index)).to_numpy(dtype=np.int64)
                active_valid = np.minimum(np.minimum(open_lens, high_lens), np.minimum(low_lens, close_lens))
                net_ref, exit_bars_ref, _ = _simulate_policy_utility_batch_details(
                    entry_prices=eff_entry,
                    is_longs=is_long_mask,
                    future_opens=opens_2d,
                    future_highs=highs_2d,
                    future_lows=lows_2d,
                    future_closes=closes_2d,
                    tp_pcts=tp_pct,
                    sl_pcts=sl_pct,
                    trailing_pcts=trailing_pct,
                    max_bars_arr=np.minimum(max_bars_arr, active_valid),
                    cost_pct=self.cost_pct,
                )

        if ts_kept is not None:
            overlap_keep = _asset_overlap_keep_mask(
                timestamps=np.asarray(ts_kept),
                assets=asset_kept,
                exit_bars=exit_bars_ref,
                priority=score_kept,
                bar_minutes=15,
                cooldown_hours=float(cooldown_hours),
            )
        else:
            overlap_keep = np.ones(len(kept_df), dtype=bool)
        if not np.any(overlap_keep):
            overlap_keep = np.ones(len(kept_df), dtype=bool)
        n_after_overlap = int(np.count_nonzero(overlap_keep))
        pnl = kept_sizes[overlap_keep] * net_ref[overlap_keep]
        ts_final = np.asarray(ts_kept)[overlap_keep] if ts_kept is not None else None

        if ts_eval is not None and len(ts_eval) > 1:
            n_days = _effective_day_count(ts_eval)
        elif ts_final is not None and len(ts_final) > 1:
            n_days = _effective_day_count(ts_final)
        else:
            n_days = 1.0
        pnl_total = float(np.sum(pnl))
        pnl_per_day = float(pnl_total / max(n_days, 1.0))
        trades_per_day = float(len(pnl) / max(n_days, 1.0))

        if ts_final is not None and len(ts_final) > 0:
            daily_returns = _aggregate_daily_values(pnl, ts_final)
            all_days = np.unique(pd.to_datetime(ts_eval, utc=True).floor("D")) if ts_eval is not None else np.unique(pd.to_datetime(ts_final, utc=True).floor("D"))
            if len(all_days) > len(daily_returns):
                day_map = pd.Series(0.0, index=pd.Index(all_days))
                ts_final_days = pd.to_datetime(ts_final, utc=True).floor("D")
                actual_daily = pd.Series(np.asarray(pnl, dtype=np.float64)).groupby(ts_final_days).sum()
                day_map.update(actual_daily)
                daily_returns = day_map.to_numpy(dtype=np.float64)
        else:
            daily_returns = np.asarray(pnl, dtype=np.float64)

        sortino, max_dd = _stable_daily_sortino_and_maxdd(daily_returns)
        _, _, ulcer, tuw = _stable_daily_pnl_metrics(pnl, ts_final, start_equity=1.0)
        temporal_stability, temporal_instability = _temporal_stability_metrics(daily_returns)
        intraday_risk = _intraday_risk_metric(max_dd=max_dd, ulcer=ulcer, tuw=tuw)
        objective = _pnl_risk_objective(
            pnl_total=pnl_total,
            max_dd=max_dd,
            ulcer=ulcer,
            tuw=tuw,
            daily_returns=daily_returns,
        )
        gains = pnl[pnl > 0.0]
        losses = pnl[pnl < 0.0]
        avg_win = float(np.mean(gains)) if len(gains) else 0.0
        avg_loss = float(np.mean(losses)) if len(losses) else 0.0
        if len(losses) > 0 and np.abs(np.sum(losses)) > 1e-9:
            profit_factor = float(np.sum(gains) / np.abs(np.sum(losses)))
        else:
            profit_factor = float("inf") if len(gains) > 0 else 0.0

        result = {
            "PnL_total": pnl_total,
            "PnL_per_day": pnl_per_day,
            "Trades_per_day": trades_per_day,
            "Sortino": float(sortino),
            "MaxDD": float(max_dd),
            "Ulcer": float(ulcer),
            "TUW": float(tuw),
            "IntradayRisk": float(intraday_risk),
            "TemporalStability": float(temporal_stability),
            "TemporalInstability": float(temporal_instability),
            "ObjectiveScore": float(objective),
            "ProfitFactor": float(profit_factor),
            "AvgWin": float(avg_win),
            "AvgLoss": float(avg_loss),
            "WinRate": float(np.mean(pnl > 0.0)) if len(pnl) else 0.0,
            "N_selected": int(len(pnl)),
            "N_days": float(max(n_days, 1.0)),
            "pipeline": pipeline_name,
            "limit_offset_enabled": bool(offset_k_pred is not None),
            "limit_offset_mode": "model" if offset_k_pred is not None else "disabled",
            "N_raw_candidates": int(n_eval),
            "N_finite_scores": int(n_rank),
            "N_after_topk": int(n_after_topk),
            "N_after_size": int(n_after_size),
            "N_after_overlap": int(n_after_overlap),
            "Raw_candidates_per_day": float(n_eval / max(n_days, 1.0)),
            "Finite_scores_per_day": float(n_rank / max(n_days, 1.0)),
            "Topk_candidates_per_day": float(n_after_topk / max(n_days, 1.0)),
            "Sized_candidates_per_day": float(n_after_size / max(n_days, 1.0)),
            "Overlap_kept_per_day": float(n_after_overlap / max(n_days, 1.0)),
        }
        if include_deciles:
            for top_pct in (0.30, 0.20, 0.10):
                sub = self._evaluate_live_pipeline_from_scores(
                    score=score,
                    trade_outcomes=trade_outcomes,
                    timestamps=timestamps,
                    top_k_pct=top_pct,
                    cooldown_hours=cooldown_hours,
                    base_size=base_size,
                    rank_multiplier=rank_multiplier,
                    sizing_formula=sizing_formula,
                    squash_k=squash_k,
                    offset_k_pred=offset_k_pred,
                    pipeline_name=f"{pipeline_name}_top{int(top_pct * 100)}",
                    include_deciles=False,
                    decile_prefix=decile_prefix,
                )
                prefix = f"{decile_prefix}_top{int(top_pct * 100)}"
                result[f"{prefix}_pnl_total"] = float(sub.get("PnL_total", 0.0))
                result[f"{prefix}_pnl_per_day"] = float(sub.get("PnL_per_day", 0.0))
                result[f"{prefix}_trades_per_day"] = float(sub.get("Trades_per_day", 0.0))
                result[f"{prefix}_n_trades"] = int(sub.get("N_selected", 0))
                result[f"{prefix}_win_rate"] = float(sub.get("WinRate", 0.0))
                result[f"{prefix}_profit_factor"] = float(sub.get("ProfitFactor", 0.0))
                result[f"{prefix}_avg_win"] = float(sub.get("AvgWin", 0.0))
                result[f"{prefix}_avg_loss"] = float(sub.get("AvgLoss", 0.0))
                result[f"{prefix}_sortino"] = float(sub.get("Sortino", 0.0))
                result[f"{prefix}_maxdd"] = float(sub.get("MaxDD", 0.0))
                result[f"{prefix}_ulcer"] = float(sub.get("Ulcer", 0.0))
                result[f"{prefix}_time_under_water"] = float(sub.get("TUW", 0.0))
        return result
        
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
        y_gross: np.ndarray,
        timestamps: np.ndarray | None,
        alpha: float,
        delta: float,
        gamma: float,
        top_k_pct: float,
        eval_top_k_pct: float | None = None,
        cooldown_hours: float = 0.0,
        base_size: float = 0.05,
        rank_multiplier: float = 0.10,
        sizing_formula: str = "linear",
        squash_fn: str = "tanh",
        squash_k: float = 1.0,
        groups: np.ndarray | None = None,
        symbols: np.ndarray | None = None,
        exit_bars: np.ndarray | None = None,
        trade_mask: np.ndarray | None = None,
        cv_cache: list[dict[str, Any]] | None = None,
        trade_outcomes: pd.DataFrame | None = None,
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
            top_k_pct: Percentage of top predictions to select
            base_size: Base position size
            rank_multiplier: Rank-based position multiplier
            sizing_formula: Position sizing formula (linear, convex, sigmoid, exponential, fixed)
            squash_fn: Squash function (tanh, sigmoid)
            squash_k: Squash steepness parameter
            groups: Optional group labels for CV splits
            symbols: Optional array of symbols for diversity metrics
            
        Returns:
            Dictionary of evaluation metrics
        """
        from extreme_price_movements.purged_cv import PurgedKFold
        from sklearn.preprocessing import StandardScaler
        
        # Apply trade mask if provided
        if trade_mask is not None:
            X = X[trade_mask]
            y = y[trade_mask]
            y_gross = y_gross[trade_mask]
            if timestamps is not None:
                timestamps = timestamps[trade_mask]
            if groups is not None:
                groups = groups[trade_mask]
            if symbols is not None:
                symbols = symbols[trade_mask]
            if exit_bars is not None:
                exit_bars = np.asarray(exit_bars)[trade_mask]

        n = len(y)
        oof_preds = np.full(n, np.nan)
        oof_true = np.full(n, np.nan)
        oof_true_raw = np.full(n, np.nan)
        oof_limit_k = np.full(n, np.nan)
        oof_weights = None # Initialize oof_weights here
        fold_thresholds = []  # Store thresholds per fold

        if cv_cache is None:
            # Use PurgedKFold to prevent leakage (increase purge/embargo for 10h isolation)
            times_cv = _normalize_cv_times(timestamps)
            if times_cv is not None:
                pkf = PurgedKFold(n_splits=3, purge=60, embargo=60, times=times_cv)
            else:
                pkf = PurgedKFold(n_splits=3, purge=60, embargo=60)
            split_args = [X, y]
            if groups is not None:
                split_args.append(groups)
            fold_iter = []
            for tr_idx, val_idx in pkf.split(*split_args):
                scaler = StandardScaler()
                X_tr_scaled = scaler.fit_transform(X[tr_idx]).astype(np.float32, copy=False)
                X_val_scaled = scaler.transform(X[val_idx]).astype(np.float32, copy=False)
                fold_iter.append({
                    "tr_idx": np.asarray(tr_idx, dtype=np.int64),
                    "va_idx": np.asarray(val_idx, dtype=np.int64),
                    "X_tr_scaled": X_tr_scaled,
                    "X_val_scaled": X_val_scaled,
                })
        else:
            fold_iter = cv_cache

        for fold in fold_iter:
            tr_idx = fold["tr_idx"]
            val_idx = fold["va_idx"]
            X_tr_scaled = fold["X_tr_scaled"]
            X_val_scaled = fold["X_val_scaled"]
            y_tr = y[tr_idx]
            y_val = y[val_idx]

            sw_tr = self._compute_sample_weights(y_tr, gamma)

            # Step 1: Train first-pass Ridge model on training fold only.
            w_step1 = self._fit_weights(X_tr_scaled, y_tr, alpha, delta, gamma)

            # Step 2: cheap train-fold gating from first-pass predictions.
            p_tr_gate = X_tr_scaled @ w_step1
            k_num = max(1, int(top_k_pct * len(p_tr_gate)))
            tr_gate_idx = np.argpartition(p_tr_gate, -k_num)[-k_num:]
            w_step2 = self._fit_weights(X_tr_scaled[tr_gate_idx], y_tr[tr_gate_idx], alpha, delta, gamma)
            threshold_low = np.percentile(p_tr_gate, 100 - top_k_pct * 100)
            threshold_high = np.percentile(p_tr_gate, 100 - top_k_pct * 50)

            # Store thresholds for this fold
            fold_thresholds.append({'threshold_low': threshold_low, 'threshold_high': threshold_high})

            # Predict step 1 on validation fold to gate it
            val_p_step1 = X_val_scaled @ w_step1
            val_p_step2 = X_val_scaled @ w_step2

            oof_preds[val_idx] = val_p_step2
            oof_true[val_idx] = y_val
            oof_true_raw[val_idx] = y_gross[val_idx]
            if trade_outcomes is not None:
                try:
                    score_tr_full = X_tr_scaled @ w_step2
                    w_tr = self._compute_sample_weights(y_tr, gamma)
                    oof_limit_k[val_idx] = self._fit_fold_offset_model(
                        X_tr=X[tr_idx],
                        X_va=X[val_idx],
                        score_tr=score_tr_full,
                        trade_outcomes_tr=trade_outcomes.iloc[tr_idx],
                        sample_weight_tr=w_tr,
                    )
                except Exception:
                    pass

            if oof_weights is None:
                oof_weights = w_step2.copy()
        
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
                'IntradayRisk': float(_intraday_risk_metric(max_dd=1.0, ulcer=100.0, tuw=1.0)),
                'ObjectiveScore': -1e9,
                'IC': 0.0,
                'WinRate': 0.0,
                'oof_fallback_occurred': False,
            }
        
        pred = oof_preds[mask]
        true = y[mask]
        true_raw = oof_true_raw[mask]
        ts_masked = timestamps[mask] if timestamps is not None else None
        sym_masked = symbols[mask] if symbols is not None else None
        exit_bars_masked = np.asarray(exit_bars)[mask] if exit_bars is not None else None

        # Aggregate thresholds across folds (median for robustness)
        if fold_thresholds:
            threshold_low = np.median([t['threshold_low'] for t in fold_thresholds])
            threshold_high = np.median([t['threshold_high'] for t in fold_thresholds])
        else:
            # Fallback: use global percentiles
            threshold_low = np.percentile(pred, 10)
            threshold_high = np.percentile(pred, 50)

        pos_frac = self._score_to_position_fraction(
            pred,
            base_size=base_size,
            rank_multiplier=rank_multiplier,
            sizing_formula=sizing_formula,
            squash_k=squash_k,
        )
        max_position = min(base_size + rank_multiplier, self.position_hard_cap)

        pos_stats = {
            'min': float(np.min(pos_frac)),
            'max': float(np.max(pos_frac)),
            'mean': float(np.mean(pos_frac)),
            'median': float(np.median(pos_frac)),
            'std': float(np.std(pos_frac)),
            'n_zero': int(np.sum(pos_frac == 0.0)),
            'n_max': int(np.sum(np.abs(pos_frac - max_position) < 1e-6)),
        }
        active_pos_mean = float(np.mean(pos_frac[pos_frac > 0])) if np.any(pos_frac > 0) else 0.0
        tprint(
            f"[PNL_VERIFY] Position Sizing Statistics:\n"
            f"  - max_position={max_position:.4f} (base={base_size:.4f}, mult={rank_multiplier:.4f}, cap={self.position_hard_cap:.4f})\n"
            f"  - active_trades_avg_pos={active_pos_mean:.4f}, global_avg_pos_frac={pos_stats['mean']:.4f}\n"
            f"  - range=[{pos_stats['min']:.4f}, {pos_stats['max']:.4f}]\n"
            f"  - n_zero_positions={pos_stats['n_zero']}, n_max_positions={pos_stats['n_max']}\n"
            f"  - n_trades={len(pos_frac)}, sizing_formula={sizing_formula}, squash_fn={squash_fn}, squash_k={squash_k:.2f}"
        )
        
        eval_trade_outcomes = trade_outcomes.iloc[:n].copy() if trade_outcomes is not None else pd.DataFrame(index=np.arange(n))
        if "return" not in eval_trade_outcomes.columns:
            eval_trade_outcomes["return"] = np.asarray(y_gross[:n], dtype=np.float64)
        if symbols is not None and "symbol" not in eval_trade_outcomes.columns:
            eval_trade_outcomes["symbol"] = np.asarray(symbols[:n], dtype=object)
        if timestamps is not None and "timestamp" not in eval_trade_outcomes.columns:
            eval_trade_outcomes["timestamp"] = np.asarray(timestamps[:n])
        if exit_bars is not None and "duration" not in eval_trade_outcomes.columns:
            eval_trade_outcomes["duration"] = np.asarray(exit_bars[:n], dtype=np.int64)
        eval_top_k_pct = float(top_k_pct if eval_top_k_pct is None else eval_top_k_pct)
        live_eval = self._evaluate_live_pipeline_from_scores(
            score=oof_preds,
            trade_outcomes=eval_trade_outcomes,
            timestamps=timestamps[:n] if timestamps is not None else None,
            top_k_pct=eval_top_k_pct,
            cooldown_hours=float(cooldown_hours),
            base_size=base_size,
            rank_multiplier=rank_multiplier,
            sizing_formula=sizing_formula,
            squash_k=squash_k,
            offset_k_pred=oof_limit_k if np.isfinite(oof_limit_k).any() else None,
            pipeline_name="joint_cv_oof",
        )
        ranking_eval = self._evaluate_ranking_from_scores(
            score=oof_preds,
            returns=y_gross[:n],
            top_k_pct=float(top_k_pct),
            pipeline_name="ranking_cv_oof",
        )

        unique_symbols_selected = 0
        unique_symbols_total = len(np.unique(symbols)) if symbols is not None else 0
        if sym_masked is not None and int(live_eval.get("N_selected", 0)) > 0:
            unique_symbols_selected = len(np.unique(sym_masked))

        total_pnl = float(live_eval.get("PnL_total", 0.0))
        pnl_per_day = float(live_eval.get("PnL_per_day", 0.0))
        trades_per_day = float(live_eval.get("Trades_per_day", 0.0))
        sortino = float(live_eval.get("Sortino", 0.0))
        max_dd = float(live_eval.get("MaxDD", 0.0))
        ulcer = float(live_eval.get("Ulcer", 0.0))
        tuw = float(live_eval.get("TUW", 0.0))
        intraday_risk = float(live_eval.get("IntradayRisk", 0.0))
        temporal_stability = float(live_eval.get("TemporalStability", 0.0))
        temporal_instability = float(live_eval.get("TemporalInstability", 1.0))
        objective_score = float(live_eval.get("ObjectiveScore", -1e9))
        profit_factor = float(live_eval.get("ProfitFactor", 0.0))
        avg_win = float(live_eval.get("AvgWin", 0.0))
        avg_loss = float(live_eval.get("AvgLoss", 0.0))
        net_win_rate = float(live_eval.get("WinRate", 0.0))
        gross_win_rate = net_win_rate
        n_neg_trades = 0
        n_days = float(live_eval.get("N_days", 1.0))
        n_selected = int(live_eval.get("N_selected", 0))
        risk_diag = {
            "n_neg_days": 0,
            "mean_daily": float(pnl_per_day / max(n_days, 1.0)),
            "downside_dev": 0.0,
        }
        
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
        
        sharpe = 0.0
        profit_factor = min(profit_factor, 20.0)

        return {
            'alpha': alpha,
            'delta': delta,
            'gamma': gamma,
            'ranking_top_k_pct': float(top_k_pct),
            'eval_top_k_pct': float(eval_top_k_pct),
            'cooldown_hours': float(cooldown_hours),
            'PnL_total': total_pnl,
            'PnL_per_day': float(pnl_per_day),
            'Trades_per_day': float(trades_per_day),
            'Unique_Symbols_Selected': unique_symbols_selected,
            'Unique_Symbols_Total': unique_symbols_total,
            'Sortino': sortino,
            'MaxDD': max_dd,
            'IntradayRisk': intraday_risk,
            'TemporalStability': temporal_stability,
            'TemporalInstability': temporal_instability,
            'ObjectiveScore': objective_score,
            'Sharpe': sharpe,
            'ProfitFactor': profit_factor,
            'AvgWin': avg_win,
            'AvgLoss': avg_loss,
            'Ulcer': ulcer,
            'TUW': tuw,
            'N_days': float(n_days),
            'N_neg_days': risk_diag['n_neg_days'],
            'Mean_daily': risk_diag['mean_daily'],
            'Downside_dev': risk_diag['downside_dev'],
            'IC': ic,
            'RankingObjective': float(ranking_eval.get("RankingObjective", -1e9)),
            'RankingIC': float(ranking_eval.get("RankingIC", 0.0)),
            'RankingMeanGross': float(ranking_eval.get("RankingMeanGross", 0.0)),
            'RankingMeanNet': float(ranking_eval.get("RankingMeanNet", 0.0)),
            'RankingHitAboveCost': float(ranking_eval.get("RankingHitAboveCost", 0.0)),
            'RankingNSelected': int(ranking_eval.get("RankingNSelected", 0)),
            'WinRate': net_win_rate,
            'GrossWinRate': gross_win_rate,
            'N_neg_trades': n_neg_trades,
            'N_selected': int(n_selected),
            'threshold_low': float(threshold_low),
            'threshold_high': float(threshold_high),
            'oof_fallback_occurred': False,
            # Position sizing metrics
            'pos_sizing': {
                'base_size': base_size,
                'rank_multiplier': rank_multiplier,
                'max_position': max_position,
                'position_hard_cap': self.position_hard_cap,
                'sizing_formula': sizing_formula,
                'squash_fn': squash_fn,
                'squash_k': squash_k,
                'avg': pos_stats['mean'],
                'median': pos_stats['median'],
                'std': pos_stats['std'],
                'min': pos_stats['min'],
                'max': pos_stats['max'],
                'n_zero': pos_stats['n_zero'],
                'n_max': pos_stats['n_max'],
            },
        }

    def _create_rolling_walk_forward_splits(
        self,
        timestamps: np.ndarray,
        n_splits: int = 3,
        train_fraction: float = 0.6,
        min_train_size: int = 100,
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Create rolling walk-forward cross-validation splits.

        Args:
            timestamps: Array of timestamps for each sample
            n_splits: Number of walk-forward splits
            train_fraction: Fraction of data to use for training in first split
            min_train_size: Minimum number of samples in training set

        Returns:
            List of (train_idx, test_idx) tuples
        """
        n = len(timestamps)
        sort_idx = np.argsort(timestamps)
        sorted_indices = np.arange(n, dtype=np.int64)[sort_idx]
        if n < min_train_size * 2:
            tprint(
                f"  WARNING: Not enough samples for rolling walk-forward "
                f"({n} < {min_train_size * 2}), using single chronological fallback split"
            )
            train_end = max(min_train_size, int(round(train_fraction * n)))
            train_end = min(max(train_end, 1), max(n - 1, 1))
            train_idx = sorted_indices[:train_end]
            test_idx = sorted_indices[train_end:]
            if len(test_idx) == 0 and n >= 2:
                train_idx = sorted_indices[:-1]
                test_idx = sorted_indices[-1:]
            return [(np.asarray(train_idx, dtype=np.int64), np.asarray(test_idx, dtype=np.int64))]

        splits = []
        train_size = max(int(train_fraction * n), min_train_size)
        test_size = (n - train_size) // n_splits

        for i in range(n_splits):
            train_end = train_size + i * test_size
            test_start = train_end
            test_end = min(test_start + test_size, n)

            if test_end <= test_start:
                break

            train_idx = sorted_indices[:train_end]
            test_idx = sorted_indices[test_start:test_end]

            splits.append(
                (
                    np.asarray(train_idx, dtype=np.int64),
                    np.asarray(test_idx, dtype=np.int64),
                )
            )

        return splits

    def _run_nested_cv(
        self,
        X: np.ndarray,
        y_net: np.ndarray,
        y_gross: np.ndarray,
        timestamps: np.ndarray,
        trade_outcomes: pd.DataFrame | None,
        symbols: np.ndarray | None,
        exit_bars: np.ndarray | None,
        groups: np.ndarray | None,
        n_outer_splits: int = 3,
        n_inner_splits: int = 3,
        reference_core_params: Dict[str, Any] | None = None,
    ) -> Tuple[pd.DataFrame, Dict]:
        """Run nested cross-validation for unbiased hyperparameter tuning.

        Outer loop: Evaluate hyperparameter performance
        Inner loop: Tune hyperparameters

        Args:
            X: Feature matrix
            y_net: Net returns (with costs)
            y_gross: Gross returns (without costs)
            timestamps: Array of timestamps
            symbols: Optional array of symbols
            groups: Optional group labels
            n_outer_splits: Number of outer CV splits
            n_inner_splits: Number of inner CV splits

        Returns:
            Tuple of (outer_results DataFrame, best_params dict)
        """
        from extreme_price_movements.purged_cv import PurgedKFold
        import optuna

        tprint(f"  Running nested CV: {n_outer_splits} outer splits, {n_inner_splits} inner splits...")

        class _NestedEarlyStoppingCallback:
            def __init__(self, patience: int):
                self.patience = int(patience)
                self.best_value = -np.inf
                self.trials_since_best = 0

            def __call__(self, study: optuna.study.Study, trial: optuna.trial.FrozenTrial) -> None:
                value = float(trial.value) if trial.value is not None else -np.inf
                if value > self.best_value + 1e-12:
                    self.best_value = value
                    self.trials_since_best = 0
                else:
                    self.trials_since_best += 1
                if self.trials_since_best >= self.patience:
                    study.stop()

        def _build_cv_cache_local(
            X_in: np.ndarray,
            ts_in: np.ndarray | None,
            grp_in: np.ndarray | None,
            *,
            n_splits: int,
            purge: int,
            embargo: int,
        ) -> list[dict[str, Any]]:
            ts_in = _normalize_cv_times(ts_in)
            if ts_in is not None:
                pkf_local = PurgedKFold(n_splits=n_splits, purge=purge, embargo=embargo, times=ts_in)
            else:
                pkf_local = PurgedKFold(n_splits=n_splits, purge=purge, embargo=embargo)
            cache_rows: list[dict[str, Any]] = []
            for tr_idx_local, va_idx_local in pkf_local.split(X_in, groups=grp_in):
                scaler_local = StandardScaler()
                X_tr_scaled_local = scaler_local.fit_transform(X_in[tr_idx_local]).astype(np.float32, copy=False)
                X_va_scaled_local = scaler_local.transform(X_in[va_idx_local]).astype(np.float32, copy=False)
                cache_rows.append({
                    "tr_idx": np.asarray(tr_idx_local, dtype=np.int64),
                    "va_idx": np.asarray(va_idx_local, dtype=np.int64),
                    "X_tr_scaled": X_tr_scaled_local,
                    "X_val_scaled": X_va_scaled_local,
                })
            return cache_rows

        # Create outer splits
        if self.use_rolling_walk_forward and timestamps is not None:
            outer_splits = self._create_rolling_walk_forward_splits(timestamps, n_splits=n_outer_splits)
        else:
            pkf = PurgedKFold(n_splits=n_outer_splits, purge=60, embargo=192, times=timestamps)
            outer_splits = list(pkf.split(X, y_net, groups=groups))

        outer_results = []
        bucket_space = self._bucket_search_space()

        for outer_fold, (train_idx, test_idx) in enumerate(outer_splits):
            tprint(f"    Outer fold {outer_fold + 1}/{len(outer_splits)}: train={len(train_idx)}, test={len(test_idx)}")

            X_train, X_test = X[train_idx], X[test_idx]
            y_train_net, y_test_net = y_net[train_idx], y_net[test_idx]
            y_train_gross, y_test_gross = y_gross[train_idx], y_gross[test_idx]
            to_train, to_test = (
                (trade_outcomes.iloc[train_idx], trade_outcomes.iloc[test_idx])
                if trade_outcomes is not None
                else (None, None)
            )
            ts_train, ts_test = (
                (timestamps[train_idx], timestamps[test_idx])
                if timestamps is not None
                else (None, None)
            )
            sym_train, sym_test = (
                (symbols[train_idx], symbols[test_idx])
                if symbols is not None
                else (None, None)
            )
            xb_train, xb_test = (
                (exit_bars[train_idx], exit_bars[test_idx])
                if exit_bars is not None
                else (None, None)
            )
            grp_train, grp_test = (
                (groups[train_idx], groups[test_idx])
                if groups is not None
                else (None, None)
            )
            inner_cv_cache = _build_cv_cache_local(
                X_train,
                ts_train,
                grp_train,
                n_splits=n_inner_splits,
                purge=48,
                embargo=48,
            )

            # Inner CV for hyperparameter tuning
            def inner_objective(trial):
                alpha = trial.suggest_float("alpha", self.alpha_range[0], self.alpha_range[1], log=True)
                delta = trial.suggest_float("delta", self.delta_range[0], self.delta_range[1])
                gamma = trial.suggest_float("gamma", self.gamma_range[0], self.gamma_range[1])
                top_k_pct = float(trial.suggest_categorical("ranking_top_k_pct", bucket_space["train_top_k_choices"]))
                eval_top_k_pct = float(trial.suggest_categorical("eval_top_k_pct", bucket_space["exec_top_k_choices"]))
                cooldown_hours = float(trial.suggest_categorical("cooldown_hours", bucket_space["cooldown_choices"]))
                metrics = self._evaluate_params(
                    X_train,
                    y_train_net,
                    y_train_gross,
                    ts_train,
                    alpha,
                    delta,
                    gamma,
                    top_k_pct,
                    eval_top_k_pct=eval_top_k_pct,
                    cooldown_hours=cooldown_hours,
                    base_size=0.05,
                    rank_multiplier=0.10,
                    sizing_formula="linear",
                    squash_fn="tanh",
                    squash_k=1.0,
                    groups=grp_train,
                    symbols=sym_train,
                    exit_bars=xb_train,
                    trade_mask=None,
                    cv_cache=inner_cv_cache,
                    trade_outcomes=to_train,
                )
                return float(metrics.get('ObjectiveScore', -1e9))

            # Run Optuna for inner CV
            optuna.logging.set_verbosity(optuna.logging.WARNING)
            study = optuna.create_study(
                direction="maximize",
                sampler=optuna.samplers.TPESampler(seed=self.random_state + outer_fold),
                pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=3, interval_steps=1),
            )
            
            def inner_logger(study, trial):
                if trial.number % 10 == 0:
                    tprint(f"      [Inner CV] Fold {outer_fold+1} Trial {trial.number}/20: best_objective={study.best_value:.6f}")

            warm_core = {}
            ref = reference_core_params or {}
            for k in ("alpha", "delta", "gamma", "ranking_top_k_pct", "eval_top_k_pct", "cooldown_hours"):
                if k in ref:
                    warm_core[k] = ref[k]
            if warm_core:
                study.enqueue_trial(warm_core)
            else:
                study.enqueue_trial({"alpha": 0.05, "delta": 1.0, "gamma": 0.2, "ranking_top_k_pct": bucket_space["train_top_k_choices"][0], "eval_top_k_pct": bucket_space["exec_top_k_choices"][0], "cooldown_hours": bucket_space["cooldown_choices"][0]})

            nested_patience = min(8, max(5, self.patience // 4))
            study.optimize(
                inner_objective,
                n_trials=20,
                n_jobs=self.n_jobs,
                callbacks=[inner_logger, _NestedEarlyStoppingCallback(nested_patience)],
            )

            best_inner_params = study.best_params
            best_inner_value = study.best_value

            # Evaluate best params on outer test set using the same live execution path as OOS.
            _outer_base_size = float(best_inner_params.get('base_size', 0.05))
            _outer_rank_mult = float(best_inner_params.get('rank_multiplier', 0.10))
            _outer_sizing_formula = str(best_inner_params.get('sizing_formula', 'linear'))
            _outer_squash_fn = str(best_inner_params.get('squash_fn', 'tanh'))
            _outer_squash_k = float(best_inner_params.get('squash_k', 1.0))
            
            outer_metrics = self._evaluate_params(
                X_train, y_train_net, y_train_gross, ts_train,
                best_inner_params['alpha'], best_inner_params['delta'], best_inner_params['gamma'],
                float(best_inner_params.get('ranking_top_k_pct', self.top_k_pct)),
                eval_top_k_pct=float(best_inner_params.get('eval_top_k_pct', self.top_k_pct)),
                cooldown_hours=float(best_inner_params.get('cooldown_hours', 1.0)),
                base_size=_outer_base_size,
                rank_multiplier=_outer_rank_mult,
                sizing_formula=_outer_sizing_formula,
                squash_fn=_outer_squash_fn,
                squash_k=_outer_squash_k,
                groups=grp_train, symbols=sym_train
                , exit_bars=xb_train
            )

            # Get OOS performance on test set
            # Train on full train set with best params
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            w_train = self._fit_weights(X_train_scaled, y_train_net, best_inner_params['alpha'],
                                       best_inner_params['delta'], best_inner_params['gamma'])

            # Predict on test set
            X_test_scaled = scaler.transform(X_test)
            pred_test = X_test_scaled @ w_train
            pred_train = X_train_scaled @ w_train

            # Compute test metrics
            test_mask = np.isfinite(pred_test) & np.isfinite(y_test_net)
            if test_mask.sum() > 10:
                test_score = np.full(len(X_test), np.nan, dtype=np.float64)
                test_score[test_mask] = pred_test[test_mask]
                test_offset = None
                if trade_outcomes is not None:
                    try:
                        test_offset = self._fit_fold_offset_model(
                            X_tr=X_train,
                            X_va=X_test,
                            score_tr=pred_train,
                            trade_outcomes_tr=trade_outcomes.iloc[train_idx],
                            sample_weight_tr=self._compute_sample_weights(y_train_net, best_inner_params['gamma']),
                        )
                    except Exception:
                        test_offset = None
                test_eval = self._evaluate_live_pipeline_from_scores(
                    score=test_score,
                    trade_outcomes=to_test,
                    timestamps=ts_test,
                    top_k_pct=float(best_inner_params.get('eval_top_k_pct', self.top_k_pct)),
                    cooldown_hours=float(best_inner_params.get('cooldown_hours', 1.0)),
                    base_size=_outer_base_size,
                    rank_multiplier=_outer_rank_mult,
                    sizing_formula=_outer_sizing_formula,
                    squash_k=_outer_squash_k,
                    offset_k_pred=test_offset,
                    pipeline_name="nested_outer_joint",
                )

                outer_results.append({
                    'outer_fold': outer_fold + 1,
                    'inner_best_alpha': best_inner_params['alpha'],
                    'inner_best_delta': best_inner_params['delta'],
                    'inner_best_gamma': best_inner_params['gamma'],
                    'inner_best_top_k_pct': float(best_inner_params.get('ranking_top_k_pct', self.top_k_pct)),
                    'inner_best_eval_top_k_pct': float(best_inner_params.get('eval_top_k_pct', self.top_k_pct)),
                    'inner_best_cooldown_hours': float(best_inner_params.get('cooldown_hours', 1.0)),
                    'inner_best_objective': best_inner_value,
                    'test_pnl_total': float(test_eval.get('PnL_total', 0.0)),
                    'test_pnl_per_day': float(test_eval.get('PnL_per_day', 0.0)),
                    'test_trades_per_day': float(test_eval.get('Trades_per_day', 0.0)),
                    'test_n_days': float(test_eval.get('N_days', 1.0)),
                    'test_sortino': float(test_eval.get('Sortino', 0.0)),
                    'test_maxdd': float(test_eval.get('MaxDD', 0.0)),
                    'test_ulcer': float(test_eval.get('Ulcer', 0.0)),
                    'test_tuw': float(test_eval.get('TUW', 0.0)),
                    'test_intraday_risk': float(test_eval.get('IntradayRisk', 0.0)),
                    'test_objective': float(test_eval.get('ObjectiveScore', -1e9)),
                    'test_n_trades': int(test_eval.get('N_selected', 0)),
                    'test_pipeline': str(test_eval.get('pipeline', 'nested_outer_joint')),
                })
            else:
                tprint(f"      WARNING: Outer fold {outer_fold + 1} has insufficient test samples")
                outer_results.append({
                    'outer_fold': outer_fold + 1,
                    'inner_best_alpha': best_inner_params['alpha'],
                    'inner_best_delta': best_inner_params['delta'],
                    'inner_best_gamma': best_inner_params['gamma'],
                    'inner_best_top_k_pct': float(best_inner_params.get('ranking_top_k_pct', self.top_k_pct)),
                    'inner_best_eval_top_k_pct': float(best_inner_params.get('eval_top_k_pct', self.top_k_pct)),
                    'inner_best_cooldown_hours': float(best_inner_params.get('cooldown_hours', 1.0)),
                    'inner_best_objective': best_inner_value,
                    'test_pnl_total': 0.0,
                    'test_pnl_per_day': 0.0,
                    'test_trades_per_day': 0.0,
                    'test_n_days': 1.0,
                    'test_sortino': 0.0,
                    'test_maxdd': 1.0,
                    'test_ulcer': 100.0,
                    'test_tuw': 1.0,
                    'test_intraday_risk': _intraday_risk_metric(1.0, 100.0, 1.0),
                    'test_objective': -1e9,
                    'test_n_trades': 0,
                })

        # Convert to DataFrame
        nested_cv_df = pd.DataFrame(outer_results)

        # Select best params based on median test performance
        if len(nested_cv_df) > 0:
            best_row = nested_cv_df.loc[nested_cv_df['test_objective'].idxmax()]
            best_params = {
                'alpha': float(best_row['inner_best_alpha']),
                'delta': float(best_row['inner_best_delta']),
                'gamma': float(best_row['inner_best_gamma']),
                'ranking_top_k_pct': float(best_row.get('inner_best_top_k_pct', self.top_k_pct)),
                'top_k_pct': float(best_row.get('inner_best_eval_top_k_pct', self.top_k_pct)),
                'cooldown_hours': float(best_row.get('inner_best_cooldown_hours', 1.0)),
            }
        else:
            # Fallback to default params
            best_params = {
                'alpha': 0.01,
                'delta': 1.0,
                'gamma': 1.0,
                'ranking_top_k_pct': float(self.top_k_pct),
                'top_k_pct': float(self.top_k_pct),
                'cooldown_hours': 2.0,
            }

        return nested_cv_df, best_params

    def _run_full_backtest(
        self,
        X: np.ndarray,
        y_net: np.ndarray,
        y_gross: np.ndarray,
        timestamps: np.ndarray,
        symbols: np.ndarray | None,
        weights: np.ndarray,
    ) -> Dict:
        """Run full backtest with final weights and compare with CV metrics.

        Args:
            X: Feature matrix
            y_net: Net returns
            y_gross: Gross returns
            timestamps: Array of timestamps
            symbols: Optional array of symbols
            weights: Final model weights

        Returns:
            Dictionary with backtest metrics and consistency comparison
        """
        tprint("  Running full backtest comparison...")

        # Compute combined predictions
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X).astype(np.float32, copy=False)
        pred = X_scaled @ weights

        # Compute position sizes based on thresholds
        threshold_low = self.threshold_low_ if self.threshold_low_ is not None else np.percentile(pred, 100 - self.top_k_pct * 100)
        threshold_high = self.threshold_high_ if self.threshold_high_ is not None else np.percentile(pred, 100 - self.top_k_pct * 50)

        # Get sizing parameters from best_params
        base_size = float(self.best_params_.get('base_size', 0.05))
        rank_multiplier = float(self.best_params_.get('rank_multiplier', 0.10))
        sizing_formula = str(self.best_params_.get('sizing_formula', 'linear'))
        squash_k = float(self.best_params_.get('squash_k', 1.0))
        
        # Normalize prediction to [0, 1] range
        pred_range = threshold_high - threshold_low + 1e-12
        x = np.clip((pred - threshold_low) / pred_range, 0, 1)
        
        max_position = min(base_size + rank_multiplier, self.position_hard_cap)

        # Apply sizing formula using numba-compiled function
        formula_code = _get_sizing_formula_code(sizing_formula)
        pos_frac = _apply_sizing_formula(
            pred.astype(np.float32),
            x.astype(np.float32),
            np.float32(base_size),
            np.float32(rank_multiplier),
            np.float32(squash_k),
            formula_code
        )

        # No position below threshold, cap at max
        pos_frac = np.where(pred >= threshold_low, pos_frac, 0.0)
        pos_frac = np.clip(pos_frac, 0.0, max_position)

        # Compute backtest returns
        backtest_returns = (y_gross - self.cost_pct) * pos_frac
        backtest_pnl = float(backtest_returns.sum())

        # Compute backtest metrics
        if timestamps is not None:
            daily_returns = _aggregate_daily_values(backtest_returns, timestamps)
            if len(daily_returns) > 1 and np.std(daily_returns) > 1e-9:
                backtest_sharpe = float(np.mean(daily_returns) / np.std(daily_returns) * np.sqrt(365.0))
            else:
                backtest_sharpe = 0.0
            _, backtest_maxdd = _stable_daily_sortino_and_maxdd(daily_returns)
        else:
            backtest_sharpe = 0.0
            backtest_maxdd = 1.0

        # Profit factor
        gains = backtest_returns[backtest_returns > 0]
        losses = backtest_returns[backtest_returns < 0]
        if len(losses) > 0 and np.abs(losses.sum()) > 1e-9:
            backtest_pf = float(gains.sum() / np.abs(losses.sum()))
        else:
            backtest_pf = float('inf') if len(gains) > 0 else 0.0

        # Get CV metrics for comparison
        cv_pnl = self.best_params_.get('PnL_total', 0.0)
        cv_sharpe = self.best_params_.get('Sharpe', 0.0)
        cv_pf = self.best_params_.get('ProfitFactor', 0.0)

        # Consistency check
        pnl_ratio = backtest_pnl / max(abs(cv_pnl), 1e-9)
        sharpe_ratio = backtest_sharpe / max(abs(cv_sharpe), 1e-9)

        is_consistent = (backtest_pnl > 0.5 * cv_pnl) and (backtest_sharpe > 0.5 * cv_sharpe)

        backtest_metrics = {
            'backtest_pnl': backtest_pnl,
            'backtest_sharpe': backtest_sharpe,
            'backtest_maxdd': backtest_maxdd,
            'backtest_profit_factor': backtest_pf,
            'backtest_n_trades': int(np.sum(pos_frac > 0)),
        }

        consistency = {
            'cv_pnl': cv_pnl,
            'cv_sharpe': cv_sharpe,
            'cv_profit_factor': cv_pf,
            'pnl_ratio': pnl_ratio,
            'sharpe_ratio': sharpe_ratio,
            'is_consistent': is_consistent,
            'consistency_score': float(0.5 * pnl_ratio + 0.5 * sharpe_ratio),
        }

        tprint(f"    Backtest PnL: {backtest_pnl:.4f} (CV: {cv_pnl:.4f}, ratio: {pnl_ratio:.2f})")
        tprint(f"    Backtest Sharpe: {backtest_sharpe:.4f} (CV: {cv_sharpe:.4f}, ratio: {sharpe_ratio:.2f})")
        tprint(f"    Consistency: {'PASS' if is_consistent else 'FAIL'}")

        return {**backtest_metrics, **consistency}
    
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

        if timestamps is None and trade_outcomes is not None and 'timestamp' in trade_outcomes.columns:
            timestamps = trade_outcomes['timestamp'].values
        
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

            # CRITICAL FIX: Detect if returns are in percentage points
            # Percentage-point returns have mean > 0.01 (1%)
            # Decimal returns for 15m bars typically have mean < 0.01
            if np.abs(np.mean(y_arr)) > 0.01:
                tprint(f"  WARNING: Returns appear to be in percentage points (mean={np.mean(y_arr):.6f}). Converting to decimal.")
                y_arr = y_arr / 100.0

            # Keep simple returns for y_gross (used in PnL calculations)
            # Convert to log returns for y_net (used in model training)
            if self.returns_are_net:
                y_net = np.log(1 + np.maximum(y_arr, -0.99)).astype(np.float32, copy=False)
                # y_gross is simple return (net + cost)
                y_gross = (y_arr + np.float32(self.cost_pct)).astype(np.float32, copy=False)
                tprint(f"  Using returns from trade_outcomes (net): mean={np.mean(y_arr):.6f}, std={np.std(y_arr):.6f}")
            else:
                y_gross = y_arr
                # y_net is log return (gross - cost converted to log)
                net_simple = y_arr - np.float32(self.cost_pct)
                y_net = np.log(1 + np.maximum(net_simple, -0.99)).astype(np.float32, copy=False)
                tprint(f"  Using returns from trade_outcomes (gross): mean={np.mean(y_arr):.6f}, std={np.std(y_arr):.6f}")
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
        
        if symbols is None and 'symbol' in trade_outcomes.columns:
            symbols = trade_outcomes['symbol'].values

        exit_bars_proxy = None
        if "exit_bars" in trade_outcomes.columns:
            exit_bars_proxy = np.asarray(trade_outcomes["exit_bars"].values, dtype=np.int64)
        elif "label_policy_max_hold_bars" in trade_outcomes.columns:
            exit_bars_proxy = np.asarray(trade_outcomes["label_policy_max_hold_bars"].values, dtype=np.int64)
        elif "future_closes" in trade_outcomes.columns:
            exit_bars_proxy = np.asarray(
                [
                    max(len(np.asarray(path, dtype=float)) - 1, 0)
                    if path is not None else 0
                    for path in trade_outcomes["future_closes"].values
                ],
                dtype=np.int64,
            )
        if exit_bars_proxy is None:
            exit_bars_proxy = np.zeros(len(X), dtype=np.int64)
        else:
            exit_bars_proxy = np.nan_to_num(exit_bars_proxy, nan=0, posinf=0, neginf=0).astype(np.int64, copy=False)
        
        _u_policy = None
        if "u_policy_net" in trade_outcomes.columns:
            _u_policy = np.asarray(trade_outcomes["u_policy_net"].values, dtype=np.float32)
        elif "u_policy" in trade_outcomes.columns:
            _u_policy = np.asarray(trade_outcomes["u_policy"].values, dtype=np.float32)
        
        _trade_mask = np.asarray(trade_outcomes["trade_mask"].values, dtype=bool) if "trade_mask" in trade_outcomes.columns else np.ones(len(X), dtype=bool)

        # Determine sorting for Walk-Forward Validation
        if timestamps is not None:
            sort_wf = np.argsort(timestamps)
            X = X[sort_wf]
            y_gross = y_gross[sort_wf]
            y_net = y_net[sort_wf]
            timestamps = timestamps[sort_wf]
            if groups is not None:
                groups = groups[sort_wf]
            if symbols is not None:
                symbols = symbols[sort_wf]
            if exit_bars_proxy is not None:
                exit_bars_proxy = exit_bars_proxy[sort_wf]
            if _u_policy is not None:
                _u_policy = _u_policy[sort_wf]
            if _trade_mask is not None:
                _trade_mask = _trade_mask[sort_wf]
            
            # CRITICAL: Also sort the DataFrames to maintain alignment
            if trade_outcomes is not None:
                trade_outcomes = trade_outcomes.iloc[sort_wf].reset_index(drop=True)
            if oof_preds is not None:
                oof_preds = oof_preds.iloc[sort_wf].reset_index(drop=True)

        # Training/OOS split:
        # if max_fit_samples is set and dataset is larger, keep only the earliest
        # max_fit_samples rows for the whole fitting/tuning step and reserve the
        # remaining later rows as untouched walk-forward OOS/reporting.
        n_total = len(X)
        desired_train_block = int(max(1, np.floor((1.0 - self.oos_fraction) * n_total)))
        if self.max_fit_samples is not None and n_total > self.max_fit_samples:
            n_train_block = min(int(self.max_fit_samples), desired_train_block)
            tprint(
                f"  Using first {n_train_block}/{n_total} chronological rows for training/tuning; "
                f"reserving {n_total - n_train_block} later rows for OOS/reporting"
            )
        else:
            n_train_block = desired_train_block
        if timestamps is not None and len(timestamps) == n_total:
            try:
                ts_days = pd.to_datetime(timestamps, utc=True).floor("D")
                unique_days = pd.Index(ts_days.unique()).sort_values()
                if len(unique_days) > self.min_oos_days:
                    split_day = unique_days[-self.min_oos_days]
                    first_oos_idx = int(np.searchsorted(ts_days.to_numpy(), split_day.to_datetime64(), side="left"))
                    n_train_block = min(n_train_block, max(first_oos_idx, int(0.50 * n_total)))
                self.oos_protocol_ = {
                    "oos_fraction": float(self.oos_fraction),
                    "min_oos_days": int(self.min_oos_days),
                    "repeated_oos_splits": int(self.repeated_oos_splits),
                    "repeated_min_selected_threshold": 25,
                }
            except Exception:
                self.oos_protocol_ = {
                    "oos_fraction": float(self.oos_fraction),
                    "min_oos_days": int(self.min_oos_days),
                    "repeated_oos_splits": int(self.repeated_oos_splits),
                    "repeated_min_selected_threshold": 25,
                }

        X_tv, y_tv_gross, y_tv_net = X[:n_train_block], y_gross[:n_train_block], y_net[:n_train_block]
        X_oos, y_oos_gross, y_oos_net = X[n_train_block:], y_gross[n_train_block:], y_net[n_train_block:]
        ts_tv = timestamps[:n_train_block] if timestamps is not None else None
        ts_oos = timestamps[n_train_block:] if timestamps is not None else None
        grp_tv = groups[:n_train_block] if groups is not None else None
        sym_tv = symbols[:n_train_block] if symbols is not None else None
        sym_oos = symbols[n_train_block:] if symbols is not None else None
        xb_tv = exit_bars_proxy[:n_train_block] if exit_bars_proxy is not None else None
        xb_oos = exit_bars_proxy[n_train_block:] if exit_bars_proxy is not None else None
        tm_tv = _trade_mask[:n_train_block] if _trade_mask is not None else None
        tm_oos = _trade_mask[n_train_block:] if _trade_mask is not None else None
        u_tv = _u_policy[:n_train_block] if _u_policy is not None else None
        u_oos = _u_policy[n_train_block:] if _u_policy is not None else None
        to_tv = trade_outcomes.iloc[:n_train_block] if trade_outcomes is not None else None
        to_oos = trade_outcomes.iloc[n_train_block:] if trade_outcomes is not None else None
        op_tv = oof_preds.iloc[:n_train_block] if oof_preds is not None else None
        op_oos = oof_preds.iloc[n_train_block:] if oof_preds is not None else None

        # Align lengths and nan-to-num for TV set (All training/race steps use this)
        n = min(len(X_tv), len(y_tv_gross))
        X = np.nan_to_num(X_tv[:n], nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)
        y_gross = np.nan_to_num(y_tv_gross[:n], nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)
        y_net = np.nan_to_num(y_tv_net[:n], nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)
        timestamps = ts_tv[:n] if ts_tv is not None else None
        groups = grp_tv[:n] if grp_tv is not None else None
        symbols = sym_tv[:n] if sym_tv is not None else None
        exit_bars_proxy = xb_tv[:n] if xb_tv is not None else None
        _trade_mask = tm_tv[:n] if tm_tv is not None else None
        _u_policy = u_tv[:n] if u_tv is not None else None
        _u_simple_tbm = None
        if trade_outcomes is not None and "u_simple_tp_sl_net" in trade_outcomes.columns:
            _u_simple_tbm = np.asarray(trade_outcomes["u_simple_tp_sl_net"].values[:n], dtype=np.float32)
        _extra_utility_targets: Dict[str, np.ndarray] = {}
        if trade_outcomes is not None:
            for col in trade_outcomes.columns:
                if col.startswith("u_tbm_"):
                    _extra_utility_targets[col[2:]] = np.asarray(trade_outcomes[col].values[:n], dtype=np.float32)
        trade_outcomes = to_tv.iloc[:n] if to_tv is not None else None
        oof_preds = op_tv.iloc[:n] if op_tv is not None else None

        # These are used by HPO objective
        X_sub_full, y_sub_full, yg_sub_full = X, y_net, y_gross
        ts_sub_full, grp_sub_full, sym_sub_full, xb_sub_full, tm_sub_full = timestamps, groups, symbols, exit_bars_proxy, _trade_mask
        
        # Run target representation race to find best y for this bucket
        target_end = max(100, int(np.floor(len(X) * self.target_train_fraction)))
        target_end = min(target_end, max(100, len(X)))
        tprint(f"  Running target representation race on earliest {target_end}/{len(X)} training rows...")
        self.target_family_ab_ = run_target_family_ab(
            X[:target_end],
            y_gross[:target_end],
            y_net[:target_end],
            symbols[:target_end] if symbols is not None else None,
            timestamps[:target_end] if timestamps is not None else None,
            cost_pct=self.cost_pct,
            topq=self.select_topq,
            u_policy=_u_policy[:target_end] if _u_policy is not None else None,
            u_simple_tbm=_u_simple_tbm[:target_end] if _u_simple_tbm is not None else None,
            extra_utility_targets={
                name: np.asarray(values[:target_end], dtype=np.float64)
                for name, values in _extra_utility_targets.items()
            } if _extra_utility_targets else None,
            trade_mask=_trade_mask[:target_end] if _trade_mask is not None else None,
            alpha=0.5,
        )
        if isinstance(self.target_family_ab_, dict) and self.target_family_ab_.get("status") == "ok":
            _tf_w = self.target_family_ab_.get("winner", {}) or {}
            _tf_s = self.target_family_ab_.get("best_simpler", {}) or {}
            tprint(
                "    Target-family A/B: "
                f"winner={_tf_w.get('target_name', 'N/A')} "
                f"(score={float(_tf_w.get('learnability_score', 0.0)):.6f}), "
                f"best_simpler={_tf_s.get('target_name', 'N/A')} "
                f"(score={float(_tf_s.get('learnability_score', 0.0)):.6f})"
            )
        # Note: race expects gross returns for IC evaluation, but candidate targets built off y_net
        if self.select_metric == "topq_u_policy" and _u_policy is None:
            tprint("  WARNING: sizer_select_metric='topq_u_policy' requested but u_policy_net missing. Falling back to 'ic'.")
            self.select_metric = "ic"
        tgt_name, y, race_log, race_diag = run_ridge_target_race(
            X[:target_end], y_gross[:target_end], symbols[:target_end] if symbols is not None else None, timestamps[:target_end] if timestamps is not None else None,
            alpha=0.5, cost_pct=self.cost_pct,
            select_metric=self.select_metric,
            topq=self.select_topq,
            u_policy=_u_policy[:target_end] if _u_policy is not None else None,
            require_positive_topq_u=self.require_positive_topq_u,
            topq_min_samples=self.topq_min_samples,
            trade_mask=_trade_mask[:target_end] if _trade_mask is not None else None,
        )
        for line in race_log:
            tprint(line)
        self.best_target_name_ = tgt_name
        self.target_race_metrics_ = race_diag
        target_gate_margin = 1e-4
        selected_training_target_name = tgt_name
        selected_training_target_family = "target_race"
        full_target_families = _build_target_family_candidates(
            y_gross=np.asarray(y_gross, dtype=np.float64),
            y_net=np.asarray(y_net, dtype=np.float64),
            symbols=symbols,
            timestamps=timestamps,
            cost_pct=self.cost_pct,
            u_policy=np.asarray(_u_policy, dtype=np.float64) if _u_policy is not None else None,
            u_simple_tbm=np.asarray(_u_simple_tbm, dtype=np.float64) if _u_simple_tbm is not None else None,
            extra_utility_targets={
                name: np.asarray(values, dtype=np.float64)
                for name, values in _extra_utility_targets.items()
            } if _extra_utility_targets else None,
        )
        if isinstance(self.target_family_ab_, dict) and self.target_family_ab_.get("status") == "ok":
            tf_winner = dict(self.target_family_ab_.get("winner", {}) or {})
            tf_best_simpler = dict(self.target_family_ab_.get("best_simpler", {}) or {})
            winner_name = str(tf_winner.get("target_name", tgt_name))
            winner_family = str(tf_winner.get("target_family", "unknown"))
            winner_score = float(tf_winner.get("learnability_score", -1e12))
            simpler_name = str(tf_best_simpler.get("target_name", winner_name))
            simpler_score = float(tf_best_simpler.get("learnability_score", -1e12))
            allow_policy_training = (
                winner_name in {"policy_utility", "clipped_u_policy", "atr_normalized_u_policy", "huber_utility", "hybrid_raw_huber"}
                and winner_score >= (simpler_score + target_gate_margin)
            )
            if winner_family in {"regression", "ranking", "hybrid"} or allow_policy_training:
                selected_training_target_name = winner_name
                selected_training_target_family = winner_family
            else:
                selected_training_target_name = simpler_name
                selected_training_target_family = str(tf_best_simpler.get("target_family", "fallback"))
                tprint(
                    "  Learnability gate: policy-style target did not beat simpler targets by margin; "
                    f"using `{selected_training_target_name}` for training instead"
                )
            if self.forced_target_candidates:
                forced_set = set(self.forced_target_candidates)
                forced_rows = [
                    row for row in list(self.target_family_ab_.get("rows", []) or [])
                    if str(row.get("target_name")) in forced_set
                ]
                if forced_rows:
                    forced_rows = sorted(
                        forced_rows,
                        key=lambda row: float(row.get("learnability_score", -1e12)),
                        reverse=True,
                    )
                    forced_best = forced_rows[0]
                    selected_training_target_name = str(forced_best.get("target_name", selected_training_target_name))
                    selected_training_target_family = str(forced_best.get("target_family", selected_training_target_family))
                    tprint(
                        "  Forced target cycle active: "
                        f"using `{selected_training_target_name}` from forced shortlist {self.forced_target_candidates}"
                    )
        if selected_training_target_name in full_target_families:
            y = np.asarray(full_target_families[selected_training_target_name][0], dtype=np.float32)
            tprint(
                f"  Using training target `{selected_training_target_name}` "
                f"(family={selected_training_target_family}) while evaluation remains policy-utility based"
            )
        elif _u_policy is not None:
            tprint("  Using policy-aware utility labels (u_policy) as authoritative Ridge target")
            y = np.nan_to_num(_u_policy, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)
            selected_training_target_name = "policy_utility"
            selected_training_target_family = "utility"
        else:
            full_candidates = _build_ridge_target_candidates(
                y_gross,
                symbols,
                timestamps,
                cost_pct=self.cost_pct,
                clip_L=0.02,
            )
            y = np.asarray(full_candidates.get(tgt_name, full_candidates.get("winsorized")), dtype=np.float32)

        # Drop constant / persistently weak inputs after target selection so the
        # filter is aligned with the actual training objective.
        if self.model_names_:
            utility_ref = np.asarray(_u_policy, dtype=np.float32) if selected_training_target_family == "utility" and _u_policy is not None else None
            keep_mask = self._select_feature_keep_mask_family(
                X,
                y,
                list(self.model_names_),
                target_family=str(selected_training_target_family),
                utility_ref=utility_ref,
            )
            if int(np.sum(keep_mask)) < len(self.model_names_):
                dropped = [name for name, keep in zip(self.model_names_, keep_mask) if not keep]
                kept = [name for name, keep in zip(self.model_names_, keep_mask) if keep]
                tprint(
                    f"  Target-aligned feature filter kept {len(kept)}/{len(self.model_names_)} inputs "
                    f"for `{selected_training_target_name}`; dropped={dropped[:12]}"
                )
                X = X[:, keep_mask]
                self.model_names_ = kept
                if isinstance(oof_preds, pd.DataFrame):
                    cols_keep = [c for c in kept if c in oof_preds.columns]
                    oof_preds = oof_preds[cols_keep].copy()
        self.selected_training_target_name_ = str(selected_training_target_name)
        self.selected_training_target_family_ = str(selected_training_target_family)
        
        # NOTE: Do NOT scale globally here - scaling is done per-fold in _evaluate_params
        # to prevent data leakage. The final scaler is fit after CV on all data.
        
        # Use the full in-step training block for HPO. Any rows outside this block
        # are reserved for untouched walk-forward OOS/reporting.
        hpo_idx = np.arange(len(X), dtype=np.int64)

        X_hpo = X[hpo_idx]
        y_hpo = y[hpo_idx]
        yg_hpo = y_gross[hpo_idx]
        ts_hpo = timestamps[hpo_idx] if timestamps is not None else None
        grp_hpo = groups[hpo_idx] if groups is not None else None
        sym_hpo = symbols[hpo_idx] if symbols is not None else None
        xb_hpo = exit_bars_proxy[hpo_idx] if exit_bars_proxy is not None else None
        tm_hpo = _trade_mask[hpo_idx] if _trade_mask is not None else None

        if tm_hpo is not None:
            hpo_keep = np.asarray(tm_hpo, dtype=bool)
            X_hpo = X_hpo[hpo_keep]
            y_hpo = y_hpo[hpo_keep]
            yg_hpo = yg_hpo[hpo_keep]
            ts_hpo = ts_hpo[hpo_keep] if ts_hpo is not None else None
            grp_hpo = grp_hpo[hpo_keep] if grp_hpo is not None else None
            sym_hpo = sym_hpo[hpo_keep] if sym_hpo is not None else None
            xb_hpo = xb_hpo[hpo_keep] if xb_hpo is not None else None
            trade_outcomes_hpo = trade_outcomes.iloc[hpo_idx].iloc[hpo_keep].reset_index(drop=True) if trade_outcomes is not None else None
        else:
            trade_outcomes_hpo = trade_outcomes.iloc[hpo_idx].reset_index(drop=True) if trade_outcomes is not None else None

        def _build_cv_cache(
            X_in: np.ndarray,
            ts_in: np.ndarray | None,
            grp_in: np.ndarray | None,
            n_splits: int,
        ) -> list[dict[str, Any]]:
            from extreme_price_movements.purged_cv import PurgedKFold
            cache_rows: list[dict[str, Any]] = []
            n_obs = int(len(X_in))
            if n_obs < 2:
                return cache_rows
            ts_norm = _normalize_cv_times(ts_in)
            # Always use purged CV, even with n_splits=1 (use 2 splits and take the last one)
            n_splits_purged = max(2, int(n_splits))
            if ts_norm is not None:
                pkf_local = PurgedKFold(n_splits=n_splits_purged, purge=60, embargo=60, times=ts_norm)
            else:
                pkf_local = PurgedKFold(n_splits=n_splits_purged, purge=60, embargo=60)

            # If n_splits=1, use only the last split (most recent validation data)
            splits = list(pkf_local.split(X_in, groups=grp_in))
            if int(n_splits) <= 1:
                splits = [splits[-1]]  # Use last split only

            for tr_idx_local, va_idx_local in splits:
                scaler_local = StandardScaler()
                X_tr_scaled_local = scaler_local.fit_transform(X_in[tr_idx_local])
                X_va_scaled_local = scaler_local.transform(X_in[va_idx_local])
                cache_rows.append({
                    "tr_idx": np.asarray(tr_idx_local, dtype=np.int64),
                    "va_idx": np.asarray(va_idx_local, dtype=np.int64),
                    "X_tr_scaled": np.asarray(X_tr_scaled_local, dtype=np.float32),
                    "X_val_scaled": np.asarray(X_va_scaled_local, dtype=np.float32),
                })
            return cache_rows

        stage1_cv_cache = _build_cv_cache(X_hpo, ts_hpo, grp_hpo, self.stage1_cv_folds)
        stage1_refine_cv_cache = _build_cv_cache(X_hpo, ts_hpo, grp_hpo, 2)
        stage2_cv_cache = _build_cv_cache(X_hpo, ts_hpo, grp_hpo, self.stage2_cv_folds)

        # Two-stage Optuna search with a single scalar objective aligned to selection.
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        results: list[dict[str, Any]] = []
        bucket_space = self._bucket_search_space()

        class ObjectiveLogger:
            def __init__(self, stage_name: str):
                self.stage_name = stage_name
                self.best_obj = -np.inf
                self.best_pnl = -np.inf

            def __call__(self, study, trial):
                obj = float(trial.user_attrs.get("ObjectiveScore", -1e9))
                pnl = float(trial.user_attrs.get("PnL_per_day", -999.0))
                maxdd = float(trial.user_attrs.get("MaxDD", 99.0))
                intraday_risk = float(trial.user_attrs.get("IntradayRisk", 1e9))
                temporal_stability = float(trial.user_attrs.get("TemporalStability", 0.0))
                profit_factor = float(trial.user_attrs.get("ProfitFactor", 0.0))
                avg_win = float(trial.user_attrs.get("AvgWin", 0.0))
                avg_loss = float(trial.user_attrs.get("AvgLoss", 0.0))
                trades_per_day = float(trial.user_attrs.get("Trades_per_day", 0.0))
                n_days = float(trial.user_attrs.get("N_days", 0.0))
                n_neg_days = float(trial.user_attrs.get("N_neg_days", 0.0))
                n_neg_trades = float(trial.user_attrs.get("N_neg_trades", 0.0))
                mean_daily = float(trial.user_attrs.get("Mean_daily", 0.0))
                downside_dev = float(trial.user_attrs.get("Downside_dev", 0.0))
                ulcer = float(trial.user_attrs.get("Ulcer", 0.0))
                tuw = float(trial.user_attrs.get("TUW", 0.0))
                net_win_rate = float(trial.user_attrs.get("WinRate", 0.0))
                sortino = float(trial.user_attrs.get("Sortino", 0.0))

                if obj > self.best_obj or pnl > self.best_pnl:
                    self.best_obj = max(self.best_obj, obj)
                    self.best_pnl = max(self.best_pnl, pnl)
                    # Extract position sizing metrics if available
                    pos_sizing = trial.user_attrs.get('pos_sizing', {})
                    pos_info = f", PosSize: avg={pos_sizing.get('avg', 0):.2%}, range=[{pos_sizing.get('min', 0):.2%}, {pos_sizing.get('max', 0):.2%}]" if pos_sizing else ""
                    pnl_label = "PnL/Day" if n_days > 1.1 else "Total_PnL"
                    trades_label = "Trades/Day" if n_days > 1.1 else "Total_Trades"
                    tprint(
                        f"    [{self.stage_name}] Trial {trial.number} "
                        f"Score={obj:.6f}, {pnl_label}={pnl*100.0:.2f}%, Sortino={sortino:.3f}, {trades_label}={trades_per_day:.2f}, WinRate={net_win_rate:.1%}, "
                        f"Risk={intraday_risk:.6f}, TempStab={temporal_stability:.3f}, MaxDD={maxdd:.4f}, Ulcer={ulcer:.4f}, tuw={tuw:.4f}, "
                        f"PF={profit_factor:.3f}, avg_win={avg_win:.6f}, avg_loss={avg_loss:.6f}, n_neg_days={n_neg_days:.0f}/{n_days:.0f}, "
                        f"n_neg_trades={n_neg_trades:.0f}, mean_daily={mean_daily:.6f}{pos_info} | "
                        f"Params: {trial.params}"
                    )

        class EarlyStoppingCallback:
            def __init__(self, patience: int):
                self.patience = int(patience)
                self.best_value = -np.inf
                self.trials_since_best = 0

            def __call__(self, study: optuna.study.Study, trial: optuna.trial.FrozenTrial) -> None:
                value = float(trial.value) if trial.value is not None else -np.inf
                if value > self.best_value + 1e-12:
                    self.best_value = value
                    self.trials_since_best = 0
                else:
                    self.trials_since_best += 1
                if self.trials_since_best >= self.patience:
                    study.stop()

        def _record_trial_metrics(trial, metrics: Dict[str, Any], stage: str) -> Dict[str, Any]:
            row = dict(metrics)
            row.update({k: v for k, v in trial.params.items()})
            row["stage"] = stage
            for k, v in row.items():
                trial.set_user_attr(k, v)
            results.append(row)
            return row

        def _run_stage1_hpo() -> tuple[pd.DataFrame, dict[str, Any]]:
            n_trials_stage1 = self.stage1_n_trials
            tprint(
                f"  Stage 1 HPO: evaluating {n_trials_stage1} core combinations "
                f"with {self.stage1_cv_folds}-fold CV..."
            )

            def _make_stage1_objective(cv_cache_local: list[dict[str, Any]]):
                def objective_stage1(trial):
                    gamma = trial.suggest_float("gamma", self.gamma_range[0], self.gamma_range[1])
                    alpha = trial.suggest_float("alpha", self.alpha_range[0], self.alpha_range[1], log=True)
                    delta = trial.suggest_float("delta", self.delta_range[0], self.delta_range[1])
                    top_k_pct = float(trial.suggest_categorical("ranking_top_k_pct", bucket_space["train_top_k_choices"]))
                    metrics = self._evaluate_params(
                        X_hpo,
                        y_hpo,
                        yg_hpo,
                        ts_hpo,
                        alpha,
                        delta,
                        gamma,
                        top_k_pct,
                        eval_top_k_pct=min(bucket_space["exec_top_k_choices"]),
                        cooldown_hours=0.0,
                        base_size=0.05,
                        rank_multiplier=0.10,
                        sizing_formula="linear",
                        squash_fn="tanh",
                        squash_k=1.0,
                        groups=grp_hpo,
                        symbols=sym_hpo,
                        exit_bars=xb_hpo,
                        trade_mask=None,
                        cv_cache=cv_cache_local,
                        trade_outcomes=trade_outcomes_hpo,
                    )
                    _record_trial_metrics(trial, metrics, stage="stage1_core")
                    return float(metrics.get("RankingObjective", -1e9))

                return objective_stage1

            def _build_stage1_study(seed: int):
                pruner = optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=5, interval_steps=1)
                return optuna.create_study(
                    direction="maximize",
                    pruner=pruner,
                    sampler=optuna.samplers.TPESampler(seed=seed),
                )

            def _enqueue_stage1_warm_start(study, params: dict[str, Any] | None):
                if params:
                    warm_keys = ["alpha", "delta", "gamma", "ranking_top_k_pct"]
                    warm_params = {k: params[k] for k in warm_keys if k in params}
                    if warm_params:
                        study.enqueue_trial(warm_params)
                        return
                study.enqueue_trial({"alpha": 0.05, "delta": 1.0, "gamma": 0.2, "ranking_top_k_pct": bucket_space["train_top_k_choices"][0]})

            objective_stage1 = _make_stage1_objective(stage1_cv_cache)
            study_stage1 = _build_stage1_study(self.random_state)
            _enqueue_stage1_warm_start(study_stage1, getattr(self, "warm_start_params_", None))
            study_stage1.optimize(
                objective_stage1,
                n_trials=n_trials_stage1,
                n_jobs=self.n_jobs,
                callbacks=[ObjectiveLogger("Stage1"), EarlyStoppingCallback(max(20, self.patience // 2))],
            )

            best_params = dict(study_stage1.best_params)

            if self.stage1_two_fold_refine:
                tprint("  Stage 1 refine: rerunning core search with 2-fold CV warm-started from the 1-fold winner...")
                objective_stage1_refine = _make_stage1_objective(stage1_refine_cv_cache)
                study_stage1_refine = _build_stage1_study(self.random_state + 17)
                _enqueue_stage1_warm_start(study_stage1_refine, best_params)
                study_stage1_refine.optimize(
                    objective_stage1_refine,
                    n_trials=max(20, n_trials_stage1 // 2),
                    n_jobs=self.n_jobs,
                    callbacks=[ObjectiveLogger("Stage1Refine"), EarlyStoppingCallback(max(8, self.patience // 3))],
                )
                best_params = dict(study_stage1_refine.best_params)

            stage1_df = pd.DataFrame([r for r in results if r.get("stage") == "stage1_core"])
            return stage1_df, best_params

        def _run_stage2_hpo(core_params: dict[str, Any]) -> pd.DataFrame:
            n_trials_stage2 = self.stage2_n_trials
            tprint(
                f"  Stage 2 HPO: evaluating {n_trials_stage2} sizing combinations around best core params "
                f"with {self.stage2_cv_folds}-fold CV..."
            )

            def objective_stage2(trial):
                base_low, base_high, base_step = bucket_space["base_size_range"]
                mult_low, mult_high, mult_step = bucket_space["rank_multiplier_range"]
                eval_top_k_pct = float(trial.suggest_categorical("eval_top_k_pct", bucket_space["exec_top_k_choices"]))
                cooldown_hours = float(trial.suggest_categorical("cooldown_hours", bucket_space["cooldown_choices"]))
                base_size = trial.suggest_float("base_size", base_low, base_high, step=base_step)
                rank_multiplier = trial.suggest_float("rank_multiplier", mult_low, mult_high, step=mult_step)
                if self.stage2_lock_formula:
                    sizing_formula = str(core_params.get("sizing_formula", "linear"))
                    squash_fn = str(core_params.get("squash_fn", "tanh"))
                    squash_k = float(core_params.get("squash_k", 1.0))
                else:
                    sizing_formula = trial.suggest_categorical("sizing_formula", ["linear", "convex", "concave"])
                    squash_fn = trial.suggest_categorical("squash_fn", ["tanh"])
                    squash_k = float(trial.suggest_categorical("squash_k", bucket_space.get("squash_k_choices", [1.0, 1.5, 2.0])))
                metrics = self._evaluate_params(
                    X_hpo,
                    y_hpo,
                    yg_hpo,
                    ts_hpo,
                    float(core_params["alpha"]),
                    float(core_params["delta"]),
                    float(core_params["gamma"]),
                    float(core_params["ranking_top_k_pct"]),
                    eval_top_k_pct=eval_top_k_pct,
                    cooldown_hours=cooldown_hours,
                    base_size=base_size,
                    rank_multiplier=rank_multiplier,
                    sizing_formula=sizing_formula,
                    squash_fn=squash_fn,
                    squash_k=squash_k,
                    groups=grp_hpo,
                    symbols=sym_hpo,
                    exit_bars=xb_hpo,
                    trade_mask=None,
                    cv_cache=stage2_cv_cache,
                    trade_outcomes=trade_outcomes_hpo,
                )
                _record_trial_metrics(trial, metrics, stage="stage2_sizing")
                return float(metrics.get("ObjectiveScore", -1e9))

            pruner = optuna.pruners.MedianPruner(n_startup_trials=8, n_warmup_steps=4, interval_steps=1)
            study_stage2 = optuna.create_study(
                direction="maximize",
                pruner=pruner,
                sampler=optuna.samplers.TPESampler(seed=self.random_state + 1),
            )
            study_stage2.enqueue_trial({
                "eval_top_k_pct": bucket_space["exec_top_k_choices"][min(1, len(bucket_space["exec_top_k_choices"]) - 1)],
                "cooldown_hours": bucket_space["cooldown_choices"][0],
                "base_size": bucket_space["base_size_range"][0],
                "rank_multiplier": bucket_space["rank_multiplier_range"][0],
                "sizing_formula": "linear",
                "squash_fn": "tanh",
                "squash_k": bucket_space.get("squash_k_choices", [1.0])[0],
            })
            study_stage2.optimize(
                objective_stage2,
                n_trials=n_trials_stage2,
                n_jobs=self.n_jobs,
                callbacks=[ObjectiveLogger("Stage2"), EarlyStoppingCallback(max(10, self.patience // 2))],
            )
            return pd.DataFrame([r for r in results if r.get("stage") == "stage2_sizing"])

        stage1_df, best_core_params = _run_stage1_hpo()
        stage2_df = _run_stage2_hpo(best_core_params)

        self.cv_results_ = pd.concat([df for df in [stage1_df, stage2_df] if not df.empty], ignore_index=True)
        stage2_best = stage2_df.sort_values("ObjectiveScore", ascending=False).iloc[0] if not stage2_df.empty else None
        stage1_best = stage1_df.sort_values("ObjectiveScore", ascending=False).iloc[0]
        best_row = stage2_best if stage2_best is not None else stage1_best

        # If nested CV is enabled, run it for unbiased hyperparameter estimation
        if self.use_nested_cv and timestamps is not None:
            tprint("  Running nested cross-validation for unbiased hyperparameter tuning...")
            nested_cv_df, nested_best_params = self._run_nested_cv(
                X=X_tv,
                y_net=y_tv_net,
                y_gross=y_tv_gross,
                timestamps=ts_tv,
                trade_outcomes=to_tv,
                symbols=sym_tv,
                exit_bars=xb_tv,
                groups=grp_tv,
                n_outer_splits=2,
                n_inner_splits=2,
                reference_core_params=best_core_params,
            )
            self.nested_cv_results_ = nested_cv_df
            self.best_nested_cv_params_ = nested_best_params
            if len(nested_cv_df) > 0:
                self.cv_summary_ = {
                    "selector": "nested_joint_holdout_median",
                    "pnl_total": float(nested_cv_df["test_pnl_total"].median()),
                    "pnl_per_day": float(nested_cv_df["test_pnl_per_day"].median()),
                    "trades_per_day": float(nested_cv_df["test_trades_per_day"].median()),
                    "sortino": float(nested_cv_df["test_sortino"].median()),
                    "maxdd": float(nested_cv_df["test_maxdd"].median()),
                    "ulcer": float(nested_cv_df["test_ulcer"].median()),
                    "tuw": float(nested_cv_df["test_tuw"].median()),
                    "intraday_risk": float(nested_cv_df["test_intraday_risk"].median()),
                    "objective": float(nested_cv_df["test_objective"].median()),
                    "n_trades": float(nested_cv_df["test_n_trades"].median()),
                    "n_days": float(nested_cv_df["test_n_days"].median()),
                }
                tprint(
                    "    Nested CV joint-holdout median: "
                    f"PnL/Day={self.cv_summary_['pnl_per_day']:.6f}, "
                    f"Objective={self.cv_summary_['objective']:.6f}, "
                    f"Trades/Day={self.cv_summary_['trades_per_day']:.4f}"
                )
                self.best_params_ = {
                    'alpha': nested_best_params['alpha'],
                    'delta': nested_best_params['delta'],
                    'gamma': nested_best_params['gamma'],
                    'ranking_top_k_pct': float(nested_best_params.get('ranking_top_k_pct', best_row.get('ranking_top_k_pct', 0.20))),
                    'top_k_pct': float(nested_best_params.get('eval_top_k_pct', best_row.get('eval_top_k_pct', 0.10))),
                    'cooldown_hours': float(nested_best_params.get('cooldown_hours', best_row.get('cooldown_hours', bucket_space["cooldown_choices"][0]))),
                    'base_size': float(best_row.get('base_size', 0.05)),
                    'rank_multiplier': float(best_row.get('rank_multiplier', 0.10)),
                    'sizing_formula': str(best_row.get('sizing_formula', 'linear')),
                    'squash_fn': str(best_row.get('squash_fn', 'tanh')),
                    'squash_k': float(best_row.get('squash_k', 1.0)),
                    'ObjectiveScore': float(self.cv_summary_['objective']),
                    'IntradayRisk': float(self.cv_summary_['intraday_risk']),
                    'threshold_low': float(best_row.get('threshold_low', 0.0)),
                    'threshold_high': float(best_row.get('threshold_high', 0.0)),
                    'nested_cv_used': True,
                }
            else:
                self.best_params_ = {
                    'alpha': float(best_row['alpha']),
                    'delta': float(best_row['delta']),
                    'gamma': float(best_row['gamma']),
                    'ranking_top_k_pct': float(best_row.get('ranking_top_k_pct', 0.20)),
                    'top_k_pct': float(best_row.get('eval_top_k_pct', 0.10)),
                    'cooldown_hours': float(best_row.get('cooldown_hours', 1.0)),
                    'base_size': float(best_row.get('base_size', 0.05)),
                    'rank_multiplier': float(best_row.get('rank_multiplier', 0.10)),
                    'sizing_formula': str(best_row.get('sizing_formula', 'linear')),
                    'squash_fn': str(best_row.get('squash_fn', 'tanh')),
                    'squash_k': float(best_row.get('squash_k', 1.0)),
                    'ObjectiveScore': float(best_row.get('ObjectiveScore', -1e9)),
                    'IntradayRisk': float(best_row.get('IntradayRisk', 1e9)),
                    'threshold_low': float(best_row.get('threshold_low', 0.0)),
                    'threshold_high': float(best_row.get('threshold_high', 0.0)),
                    'nested_cv_used': False,
                }
        else:
            # Standard CV (not nested)
            self.cv_summary_ = None
            self.best_params_ = {
                'alpha': float(best_row['alpha']),
                'delta': float(best_row['delta']),
                'gamma': float(best_row['gamma']),
                'ranking_top_k_pct': float(best_row.get('ranking_top_k_pct', 0.20)),
                'top_k_pct': float(best_row.get('eval_top_k_pct', 0.10)),
                'cooldown_hours': float(best_row.get('cooldown_hours', 1.0)),
                'base_size': float(best_row.get('base_size', 0.05)),
                'rank_multiplier': float(best_row.get('rank_multiplier', 0.10)),
                'sizing_formula': str(best_row.get('sizing_formula', 'linear')),
                'squash_fn': str(best_row.get('squash_fn', 'tanh')),
                'squash_k': float(best_row.get('squash_k', 1.0)),
                'ObjectiveScore': float(best_row.get('ObjectiveScore', -1e9)),
                'IntradayRisk': float(best_row.get('IntradayRisk', 1e9)),
                'threshold_low': float(best_row.get('threshold_low', 0.0)),
                'threshold_high': float(best_row.get('threshold_high', 0.0)),
                'nested_cv_used': False,
            }

        tprint(f"  Best params: alpha={self.best_params_['alpha']:.6f}, "
               f"delta={self.best_params_['delta']:.3f}, "
               f"gamma={self.best_params_['gamma']:.3f}, "
               f"ranking_top_k_pct={self.best_params_.get('ranking_top_k_pct', self.top_k_pct):.2f}, "
               f"exec_top_k_pct={self.best_params_['top_k_pct']:.2f}, "
               f"cooldown_hours={self.best_params_.get('cooldown_hours', 1.0):.1f}, "
               f"base_size={self.best_params_.get('base_size', 0.05):.2f}, "
               f"rank_multiplier={self.best_params_.get('rank_multiplier', 0.10):.2f}, "
               f"sizing_formula={self.best_params_.get('sizing_formula', 'linear')}, "
               f"squash_fn={self.best_params_.get('squash_fn', 'tanh')}, "
               f"squash_k={self.best_params_.get('squash_k', 1.0):.2f}, "
               f"ObjectiveScore={self.best_params_.get('ObjectiveScore', -1e9):.4f}, "
               f"IntradayRisk={self.best_params_.get('IntradayRisk', 1e9):.4f}")

        # Add position sizing statistics if available
        if 'pos_sizing' in best_row:
            ps = best_row['pos_sizing']
            tprint(f"  Position Sizing: avg={ps.get('avg', 0):.2%}, median={ps.get('median', 0):.2%}, "
                   f"range=[{ps.get('min', 0):.2%}, {ps.get('max', 0):.2%}], "
                   f"max_position={ps.get('max_position', 0):.2%} (capped at {ps.get('position_hard_cap', 0):.2%})")

        if self.best_params_.get('nested_cv_used', False):
            tprint("  (Hyperparameters from nested CV - unbiased estimate)")

        # Save thresholds for inference
        self.threshold_low_ = float(self.best_params_.get('threshold_low', 0.0))
        self.threshold_high_ = float(self.best_params_.get('threshold_high', 0.0))
        tprint(f"  Thresholds: low={self.threshold_low_:.6f}, high={self.threshold_high_:.6f}")

        # Final fit on Train+Val data
        tprint("  Performing final fit on Train+Val data...")
        sample_weight = self._compute_sample_weights(y_tv_net, float(self.best_params_.get('gamma', 1.0)))
        
        # Fold-safe feature pruning + ElasticNet tuning for Ridge
        base_feature_names = list(self.model_names_ or [])
        X_masked = X[_trade_mask] if _trade_mask is not None else X
        y_masked = y_gross[_trade_mask] if _trade_mask is not None else y_gross
        ts_masked_fs = timestamps[_trade_mask] if timestamps is not None and _trade_mask is not None else timestamps
        ts_masked_fs = _normalize_cv_times(ts_masked_fs)
        
        fs_diag_ridge = run_fold_safe_feature_pruning_and_elasticnet(
            X=X_masked,
            y=y_masked,
            feature_names=base_feature_names,
            timestamps=ts_masked_fs,
            outer_splits=4,
            inner_splits=4,
            top_q=max(0.05, min(0.30, float(self.select_topq))),
            max_samples=5000,
            random_state=int(self.random_state),
        )
        self.feature_selection_diag_ridge_ = fs_diag_ridge

        ridge_selected_features = [f for f in fs_diag_ridge.get("selected_features", []) if f in base_feature_names]
        if not ridge_selected_features:
            ridge_selected_features = base_feature_names
        ridge_selected_idx = np.asarray([base_feature_names.index(f) for f in ridge_selected_features], dtype=np.int32)
        X_ridge = np.asarray(X[:, ridge_selected_idx], dtype=np.float32, order='C')

        # New Feature Pruning for Tree Models (LGBM)
        from extreme_price_movements.feature_select.run import run_feature_selection
        from extreme_price_movements.feature_select.cv import CVConfig
        from extreme_price_movements.feature_select.scoring import UtilityConfig, FeatureSelectConfig

        df_X = pd.DataFrame(X, columns=base_feature_names)
        time_s = pd.Series(_normalize_cv_times(timestamps)) if timestamps is not None else None

        cv_cfg = CVConfig(n_splits=3, min_train_size=max(100, len(y)//4), val_size=max(100, len(y)//4))
        util_cfg = UtilityConfig(utility_mode="topq_mean", topq=max(0.05, min(0.30, float(self.select_topq))))
        fs_cfg = FeatureSelectConfig(
            min_features=10,
            max_features=30,
            n_repeats_perm=3,
            confirm_mode="single_seed_fast",
        )

        lgbm_p = {
            "learning_rate": 0.05,
            "max_depth": 4,
            "n_estimators": 200,
            "early_stopping_rounds": 20,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_weight": 3,
            "reg_alpha": 0.5,
            "reg_lambda": 0.5,
            "max_bin": 256
        }

        try:
            tprint(f"  Launching Tree Feature Selection (LGBM RFE) on {len(df_X)} samples...")
            tree_fs_res = run_feature_selection(
                X=(df_X[_trade_mask] if _trade_mask is not None else df_X).reset_index(drop=True),
                y=(pd.Series(y)[_trade_mask] if _trade_mask is not None else y).reset_index(drop=True),
                groups=None,
                time_index=(time_s[_trade_mask] if time_s is not None and _trade_mask is not None else time_s).reset_index(drop=True) if time_s is not None else None,
                model_kind="regression", quantile_alpha=None,
                cv_config=cv_cfg, lgbm_params=lgbm_p,
                utility_config=util_cfg, fs_config=fs_cfg,
                random_seed=int(self.random_state),
                output_dir="artifacts",
                max_samples=8000
            )
            self.feature_selection_diag_tree_ = tree_fs_res
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
        tprint(f"  Ridge kept features: {ridge_selected_features}")
        tprint(f"  Trees kept features: {tree_selected_features}")

        from extreme_price_movements.purged_cv import PurgedKFold

        def _build_time_cv_splits(
            X_in: np.ndarray,
            ts_in: np.ndarray | None,
            grp_in: np.ndarray | None,
            n_splits: int,
            purge: int,
            embargo: int,
        ) -> list[tuple[np.ndarray, np.ndarray]]:
            n_obs = int(len(X_in))
            if n_obs < 2:
                return []
            # Always use purged CV, even with n_splits=1 (use 2 splits and take the last one)
            n_splits_purged = max(2, int(n_splits))
            ts_norm = _normalize_cv_times(ts_in)
            if ts_norm is not None:
                pkf_local = PurgedKFold(n_splits=n_splits_purged, purge=purge, embargo=embargo, times=ts_norm)
            else:
                pkf_local = PurgedKFold(n_splits=n_splits_purged, purge=purge, embargo=embargo)

            # If n_splits=1, use only the last split (most recent validation data)
            splits = list(pkf_local.split(X_in, groups=grp_in))
            if int(n_splits) <= 1 and len(splits) > 0:
                splits = [splits[-1]]  # Use last split only
            elif not splits:
                return []

            return [
                (np.asarray(tr_idx, dtype=np.int64), np.asarray(va_idx, dtype=np.int64))
                for tr_idx, va_idx in splits
            ]

        race_cv_splits = 2 if self.use_nested_cv else 1
        cv_splits = _build_time_cv_splits(
            X,
            timestamps,
            groups,
            n_splits=race_cv_splits,
            purge=43200 if timestamps is not None else 12,
            embargo=43200 if timestamps is not None else 12,
        )

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
            daily = _aggregate_daily_values(pnl, ts_part)
            if len(daily) == 0:
                return {"pnl_per_day": -1e9, "sortino": -1e9, "maxdd": 1.0, "ulcer": 1e9, "tuw": 1.0, "objective": -1e9}
            sortino, maxdd, ulcer, tuw = _stable_daily_pnl_metrics(daily, start_equity=1.0)
            pnl_total = float(np.sum(daily))
            return {
                "pnl_per_day": float(np.mean(daily)),
                "pnl_total": pnl_total,
                "sortino": sortino,
                "maxdd": maxdd,
                "ulcer": ulcer,
                "tuw": tuw,
                "objective": _pnl_risk_objective(pnl_total=pnl_total, max_dd=maxdd, ulcer=ulcer, tuw=tuw, daily_returns=daily),
            }

        def _fit_base(name, X_tr, y_tr, X_va, y_va, w_tr, **kwargs):
            name = str(name)
            X_tr = np.nan_to_num(np.asarray(X_tr, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
            X_va = np.nan_to_num(np.asarray(X_va, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
            y_tr = np.nan_to_num(np.asarray(y_tr, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
            y_va = np.nan_to_num(np.asarray(y_va, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
            if name == "ridge":
                alpha_val = float(kwargs.get("alpha", (race_cfg or {}).get("race_alpha_ridge", 1.0)))
                base = Pipeline([
                    ("scaler", StandardScaler(with_mean=True, with_std=True)),
                    ("model", Ridge(alpha=alpha_val, fit_intercept=True, random_state=42)),
                ])
                base.fit(X_tr, y_tr, model__sample_weight=w_tr)
                return base
            if name == "et":
                et_params = {
                    "n_estimators": 500, "random_state": 42, "n_jobs": self.n_jobs, "max_depth": 4,
                    "min_samples_leaf": 120, "min_samples_split": 240, "max_features": "sqrt",
                    "max_leaf_nodes": 128, "bootstrap": True, "max_samples": 0.6,
                    "criterion": "squared_error",
                }
                et_params.update(kwargs)
                base = ExtraTreesRegressor(**et_params)
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
                base_kwargs = dict(
                    n_estimators=500,
                    learning_rate=0.02,
                    max_depth=4,
                    subsample=0.6,
                    colsample_bytree=0.6,
                    colsample_bylevel=0.8,
                    max_delta_step=2.0,
                    min_child_weight=150,
                    reg_alpha=1.0,
                    reg_lambda=5.0,
                    tree_method="hist",
                    max_bin=256,
                    random_state=42,
                    n_jobs=self.n_jobs,
                    objective=obj_func,
                )
                base_kwargs.update(kwargs)
                # XGBoost 2.x removed fit(..., early_stopping_rounds=...).
                # Prefer constructor-level early stopping when available.
                early_st = base_kwargs.pop("early_stopping_rounds", 100)
                try:
                    base = xgb.XGBRegressor(**base_kwargs, early_stopping_rounds=early_st)
                except TypeError:
                    base = xgb.XGBRegressor(**base_kwargs)
                fit_kwargs = dict(
                    eval_set=[(X_va, y_va)],
                    verbose=False,
                )
                if w_tr is not None:
                    fit_kwargs["sample_weight"] = w_tr
                try:
                    base.fit(X_tr, y_tr, **fit_kwargs)
                except TypeError as exc:
                    if "early_stopping_rounds" not in str(exc):
                        raise
                    base.fit(X_tr, y_tr, **fit_kwargs)
                except ValueError as exc:
                    if ("sample_weight" not in str(exc)) or ("objective" not in str(exc) and "Custom objective" not in str(exc)):
                        raise
                    fit_kwargs.pop("sample_weight", None)
                    base.fit(X_tr, y_tr, **fit_kwargs)
                return base
            if name == "lgbm":
                try:
                    import lightgbm as lgb
                except Exception:
                    return None
                base_kwargs = dict(
                    n_estimators=500,
                    learning_rate=0.02,
                    max_depth=4,
                    subsample=0.6,
                    colsample_bytree=0.6,
                    min_child_samples=120,
                    reg_alpha=1.0,
                    reg_lambda=5.0,
                    max_bin=256,
                    random_state=42,
                    n_jobs=self.n_jobs,
                    verbose=-1,
                )
                base_kwargs.update(kwargs)
                early_st = base_kwargs.pop("early_stopping_rounds", 100)
                base = lgb.LGBMRegressor(**base_kwargs)
                fit_kwargs = dict(
                    eval_set=[(X_va, y_va)],
                    callbacks=[lgb.early_stopping(stopping_rounds=early_st, verbose=False)],
                )
                if w_tr is not None:
                    fit_kwargs["sample_weight"] = w_tr
                try:
                    base.fit(X_tr, y_tr, **fit_kwargs)
                except TypeError as exc:
                    if "early_stopping" not in str(exc):
                        raise
                    fit_kwargs.pop("callbacks", None)
                    base.fit(X_tr, y_tr, **fit_kwargs)
                except ValueError as exc:
                    if "sample_weight" not in str(exc):
                        raise
                    fit_kwargs.pop("sample_weight", None)
                    base.fit(X_tr, y_tr, **fit_kwargs)
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
                    inner_split_count = 2 if self.use_nested_cv else 1
                    inner_cv_splits = _build_time_cv_splits(
                        X_tr,
                        timestamps[tr_idx] if timestamps is not None else None,
                        groups[tr_idx] if groups is not None else None,
                        n_splits=inner_split_count,
                        purge=43200 if timestamps is not None else 12,
                        embargo=43200 if timestamps is not None else 12,
                    )
                    p_tr_oos = np.full(len(X_tr), np.nan, dtype=float)

                    for inner_tr, inner_va in inner_cv_splits:
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

                # Apply smooth tail-weighted calibration specifically for ET and XGB
                # This ensures the model's confidence distribution maps correctly
                # in the critical tail region, without hard thresholding.
                top_calibrator = None
                if base_name in ["et", "xgb"]:
                    try:
                        from sklearn.isotonic import IsotonicRegression

                        # Generate smooth weights emphasizing p >= 0.70
                        q_70 = np.nanquantile(p_tr, 1.0 - top_frac)
                        score_std = np.nanstd(p_tr)
                        temperature = max(0.25 * score_std, 1e-6)
                        alpha_calib = top30_boost if use_two_pass else 2.0

                        # sigmoid function: 1 / (1 + exp(-(p - q_70) / T))
                        sigmoid_weights = 1.0 / (1.0 + np.exp(-(p_tr - q_70) / temperature))

                        w_calib = np.ones(len(p_tr), dtype=float) if w_tr is None else np.asarray(w_tr, dtype=float).copy()
                        w_calib = w_calib * (1.0 + alpha_calib * sigmoid_weights)

                        top_calibrator = IsotonicRegression(out_of_bounds="clip")
                        top_calibrator.fit(p_tr, y_tr, sample_weight=w_calib)

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
                    "maxdd_all": float(m_all["maxdd"]),
                    "ulcer_all": float(m_all["ulcer"]),
                    "tuw_all": float(m_all["tuw"]),
                    "objective_all": float(m_all["objective"]),
                    "pnl_per_day_top": float(m_top["pnl_per_day"]),
                    "sortino_top": float(m_top["sortino"]),
                    "maxdd_top": float(m_top["maxdd"]),
                    "ulcer_top": float(m_top["ulcer"]),
                    "tuw_top": float(m_top["tuw"]),
                    "objective_top": float(m_top["objective"]),
                    "pnl_lift": float(m_top["pnl_per_day"] - m_all["pnl_per_day"]),
                    "n_top": int(np.sum(top_mask)),
                })

            if not fold_rows:
                return None
            agg = {
                "mu_pnl_all": float(np.mean([r["pnl_per_day_all"] for r in fold_rows])),
                "mu_sortino_all": float(np.mean([r["sortino_all"] for r in fold_rows])),
                "mu_maxdd_all": float(np.mean([r["maxdd_all"] for r in fold_rows])),
                "mu_ulcer_all": float(np.mean([r["ulcer_all"] for r in fold_rows])),
                "mu_tuw_all": float(np.mean([r["tuw_all"] for r in fold_rows])),
                "mu_objective_all": float(np.mean([r["objective_all"] for r in fold_rows])),
                "sigma_pnl_all": float(np.std([r["pnl_per_day_all"] for r in fold_rows])),
                "mu_pnl_top": float(np.mean([r["pnl_per_day_top"] for r in fold_rows])),
                "mu_sortino_top": float(np.mean([r["sortino_top"] for r in fold_rows])),
                "mu_maxdd_top": float(np.mean([r["maxdd_top"] for r in fold_rows])),
                "mu_ulcer_top": float(np.mean([r["ulcer_top"] for r in fold_rows])),
                "mu_tuw_top": float(np.mean([r["tuw_top"] for r in fold_rows])),
                "mu_objective_top": float(np.mean([r["objective_top"] for r in fold_rows])),
                "sigma_pnl_top": float(np.std([r["pnl_per_day_top"] for r in fold_rows])),
                "mu_pnl_lift": float(np.mean([r["pnl_lift"] for r in fold_rows])),
            }
            agg["stab_pen_top"] = float(agg["sigma_pnl_top"] / (abs(agg["mu_pnl_top"]) + 1e-12))
            agg["stab_pen_all"] = float(agg["sigma_pnl_all"] / (abs(agg["mu_pnl_all"]) + 1e-12))
            agg["passed_top_gate"] = bool(agg["mu_pnl_top"] > pnl_top_floor and ((agg["mu_sortino_top"] > 0.0) if require_sortino_top else True))
            return {"oof_size": oof_size, "fold_metrics": fold_rows, "agg": agg}

        def _run_score_only_candidate(candidate_name: str, score_full: np.ndarray):
            oof_size = np.full(len(y), np.nan)
            fold_rows = []
            score_full = np.asarray(score_full, dtype=float)
            for _, va_idx in cv_splits:
                s_va = score_full[va_idx]
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
                    "maxdd_all": float(m_all["maxdd"]),
                    "ulcer_all": float(m_all["ulcer"]),
                    "tuw_all": float(m_all["tuw"]),
                    "objective_all": float(m_all["objective"]),
                    "pnl_per_day_top": float(m_top["pnl_per_day"]),
                    "sortino_top": float(m_top["sortino"]),
                    "maxdd_top": float(m_top["maxdd"]),
                    "ulcer_top": float(m_top["ulcer"]),
                    "tuw_top": float(m_top["tuw"]),
                    "objective_top": float(m_top["objective"]),
                    "pnl_lift": float(m_top["pnl_per_day"] - m_all["pnl_per_day"]),
                    "n_top": int(np.sum(top_mask)),
                })
            if not fold_rows:
                return None
            agg = {
                "mu_pnl_all": float(np.mean([r["pnl_per_day_all"] for r in fold_rows])),
                "mu_sortino_all": float(np.mean([r["sortino_all"] for r in fold_rows])),
                "mu_maxdd_all": float(np.mean([r["maxdd_all"] for r in fold_rows])),
                "mu_ulcer_all": float(np.mean([r["ulcer_all"] for r in fold_rows])),
                "mu_tuw_all": float(np.mean([r["tuw_all"] for r in fold_rows])),
                "mu_objective_all": float(np.mean([r["objective_all"] for r in fold_rows])),
                "sigma_pnl_all": float(np.std([r["pnl_per_day_all"] for r in fold_rows])),
                "mu_pnl_top": float(np.mean([r["pnl_per_day_top"] for r in fold_rows])),
                "mu_sortino_top": float(np.mean([r["sortino_top"] for r in fold_rows])),
                "mu_maxdd_top": float(np.mean([r["maxdd_top"] for r in fold_rows])),
                "mu_ulcer_top": float(np.mean([r["ulcer_top"] for r in fold_rows])),
                "mu_tuw_top": float(np.mean([r["tuw_top"] for r in fold_rows])),
                "mu_objective_top": float(np.mean([r["objective_top"] for r in fold_rows])),
                "sigma_pnl_top": float(np.std([r["pnl_per_day_top"] for r in fold_rows])),
                "mu_pnl_lift": float(np.mean([r["pnl_lift"] for r in fold_rows])),
            }
            agg["stab_pen_top"] = float(agg["sigma_pnl_top"] / (abs(agg["mu_pnl_top"]) + 1e-12))
            agg["stab_pen_all"] = float(agg["sigma_pnl_all"] / (abs(agg["mu_pnl_all"]) + 1e-12))
            agg["passed_top_gate"] = bool(agg["mu_pnl_top"] > pnl_top_floor and ((agg["mu_sortino_top"] > 0.0) if require_sortino_top else True))
            return {"oof_size": oof_size, "fold_metrics": fold_rows, "agg": agg, "candidate": candidate_name}

        # The user's instruction implies a loop structure here, but the provided code snippet
        # only shows the inner part of the loop. I'm inserting the `last_best_params`
        # initialization and the loop structure as implied by the instruction.
        # Assuming this code block is part of a larger function where `dir_metrics`,
        # `last_best_params`, `dir_buckets`, `oof_pred_df`, `trade_outcomes`, `timestamps`,
        # `args.cost_pct`, `run_id`, `symbols`, `_entry_policy_cfg`, `_candidate_cfg` are defined.

        # Placeholder for the outer loop context as implied by the instruction
        # This part is not in the original content, but is necessary to make sense of the edit.
        # I will add it as a comment to indicate it's an assumption.
        # --- ASSUMED CONTEXT START ---
        # dir_metrics: Dict = {}
        # last_best_params: Optional[Dict] = None
        # for bucket_name, oof_preds in dir_buckets.items():
        #     # ... existing code for processing oof_preds for the current bucket ...
        #     # The following block is what the user provided as an edit.
        #     # It seems to be a call to an external function `run_ridge_position_sizer_step`
        #     # which would typically be in `run_ridge_sizer.py` as mentioned in the instruction.
        #     # However, the instruction places it *inside* the `_fit_base` function's `xgb` block,
        #     # which is syntactically incorrect and logically impossible.
        #     # I will place the `last_best_params` update *after* the `race_results` loop,
        #     # as it seems to be related to persisting parameters across iterations of an outer loop.
        #     # The `run_ridge_position_sizer_step` call itself is not part of *this* file's content.
        #     # The instruction is asking to update the bucket loop in `run_ridge_sizer.py`,
        #     # but the provided content is from a different file (likely `RidgePositionSizer` class).
        #     # Given the constraint to only modify the provided content, and the malformed edit,
        #     # I will interpret "update the bucket loop" as updating the `last_best_params`
        #     # within the context of *this* file, if such a loop exists.
        #     # Since no such loop is explicitly present in the provided content, and the edit
        #     # is syntactically broken, I will only apply the `last_best_params` update
        #     # if I can find a logical place for it.
        #     # The instruction also mentions adding `n_jobs` and `patience` to `RidgePositionSizer.__init__`,
        #     # but the `__init__` method is not in the provided content.
        #     # I will proceed by fixing the `xgb` block and then looking for a place for `last_best_params`.
        # --- ASSUMED CONTEXT END ---

        race_results = {}
        for base_name in ["ridge", "et"]:
            for sm_name in smoother_kinds:
                key = f"{base_name}+{sm_name}"
                out = _run_policy_candidate(base_name, sm_name)
                if out is not None:
                    race_results[key] = out

        champion_candidates: list[tuple[str, np.ndarray]] = []
        feature_diag = getattr(self, "feature_ic_diag_", None)
        if feature_diag is not None and not feature_diag.empty:
            pos_diag = feature_diag.sort_values("spearman_ic", ascending=False)
            top_feats = [f for f in pos_diag["feature"].tolist() if f in base_feature_names][:3]
            for feat in top_feats[:2]:
                feat_idx = base_feature_names.index(feat)
                champion_candidates.append((f"champion_single+{feat}", X[:, feat_idx]))
            if len(top_feats) >= 2:
                idxs = [base_feature_names.index(f) for f in top_feats[:2]]
                champion_candidates.append((f"champion_pair+{top_feats[0]}|{top_feats[1]}", np.nanmean(X[:, idxs], axis=1)))
        for c_name, c_score in champion_candidates:
            out = _run_score_only_candidate(c_name, c_score)
            if out is not None:
                race_results[c_name] = out

        if not race_results:
            raise RuntimeError("Sizer model race produced no valid candidates")

        cand_all = list(race_results.keys())
        gated = [k for k in cand_all if bool(race_results[k]["agg"].get("passed_top_gate", False))]
        cand = gated if gated else cand_all

        def _z(arr: np.ndarray) -> np.ndarray:
            arr = np.asarray(arr, dtype=float)
            return (arr - np.mean(arr)) / (np.std(arr) + 1e-12)

        mu_obj_top = np.asarray([race_results[k]["agg"]["mu_objective_top"] for k in cand], dtype=float)
        mu_obj_all = np.asarray([race_results[k]["agg"]["mu_objective_all"] for k in cand], dtype=float)
        mu_pnl_top = np.asarray([race_results[k]["agg"]["mu_pnl_top"] for k in cand], dtype=float)
        mu_pnl_all = np.asarray([race_results[k]["agg"]["mu_pnl_all"] for k in cand], dtype=float)
        stab_top = np.asarray([race_results[k]["agg"]["stab_pen_top"] for k in cand], dtype=float)
        stab_all = np.asarray([race_results[k]["agg"]["stab_pen_all"] for k in cand], dtype=float)

        scores = (
            1.75 * _z(mu_obj_top)
            + 0.40 * _z(mu_obj_all)
            + 0.60 * _z(mu_pnl_top)
            + 0.20 * _z(mu_pnl_all)
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
            row["fold_objective_all"] = [float(r["objective_all"]) for r in _fm]
            row["fold_objective_top"] = [float(r["objective_top"]) for r in _fm]
            score_rows.append(row)
        # CRITICAL FIX: Don't overwrite ridge HPO trials - use a separate attribute for target race results
        self.target_race_results_ = pd.DataFrame(score_rows)
        # Keep ridge HPO trials in cv_results_ for CV metrics extraction
        if self.cv_results_ is None or len(self.cv_results_) == 0:
            self.cv_results_ = self.target_race_results_

        # Tie-breakers: higher objective_top, higher pnl_top, lower maxdd_top, lower ulcer_top.
        ranked = sorted(
            cand,
            key=lambda k: (
                float(race_results[k]["agg"]["composite_score"]),
                float(race_results[k]["agg"]["mu_objective_top"]),
                float(race_results[k]["agg"]["mu_pnl_top"]),
                -float(race_results[k]["agg"]["mu_maxdd_top"]),
                -float(race_results[k]["agg"]["mu_ulcer_top"]),
            ),
            reverse=True,
        )
        winner_name = ranked[0]
        parts = winner_name.split("+", 1)
        base_winner = parts[0]
        smoother_winner = parts[1] if len(parts) > 1 else "ridge"
        tprint(f"  Sizer model race winner: {winner_name} score={race_results[winner_name]['agg']['composite_score']:.4f}")
        self.best_params_["race_winner"] = winner_name

        # Fit winner on full data for inference bundle with proper feature set
        if base_winner == "champion_single":
            feat_name = smoother_winner
            self.model_names_final_ = list(base_feature_names)
            X_final = np.asarray(X, dtype=np.float32)
        elif base_winner == "champion_pair":
            self.model_names_final_ = list(base_feature_names)
            X_final = np.asarray(X, dtype=np.float32)
        else:
            X_final = X_ridge if base_winner == "ridge" else X_tree
            self.model_names_final_ = self.model_names_ridge_ if base_winner == "ridge" else self.model_names_tree_

        def _tree_pareto_score(
            *,
            objective_scores: list[float],
            pnl_days: list[float],
            sortinos: list[float],
            params: dict[str, Any],
        ) -> tuple[float, float, float, float]:
            economic = float(np.mean(objective_scores)) if objective_scores else -1e9
            pnl_day_mean = float(np.mean(pnl_days)) if pnl_days else -1e9
            sortino_mean = float(np.mean(sortinos)) if sortinos else 0.0
            pnl_day_std = float(np.std(pnl_days)) if len(pnl_days) > 1 else 0.0
            sortino_std = float(np.std(sortinos)) if len(sortinos) > 1 else 0.0
            learnability = pnl_day_mean - 0.5 * pnl_day_std + 0.05 * sortino_mean - 0.02 * sortino_std
            complexity_penalty = 0.0
            if base_winner in ["xgb", "lgbm"]:
                complexity_penalty += 0.03 * float(params.get("max_depth", 4))
                complexity_penalty += 0.002 * float(params.get("n_estimators", 300)) / 100.0
                complexity_penalty += 0.10 * float(params.get("learning_rate", 0.03))
                complexity_penalty -= 0.01 * min(float(params.get("min_child_weight", 100)), 400.0) / 100.0
                complexity_penalty -= 0.01 * np.log1p(float(params.get("reg_alpha", 1.0)))
                complexity_penalty -= 0.01 * np.log1p(float(params.get("reg_lambda", 1.0)))
            elif base_winner in ["et", "rf"]:
                complexity_penalty += 0.03 * float(params.get("max_depth", 4))
                complexity_penalty += 0.002 * float(params.get("n_estimators", 300)) / 100.0
                complexity_penalty -= 0.01 * min(float(params.get("min_samples_leaf", 40)), 200.0) / 50.0
                complexity_penalty -= 0.005 * min(float(params.get("min_samples_split", 80)), 400.0) / 100.0
                complexity_penalty -= 0.05 * float(params.get("ccp_alpha", 0.0))
            pareto_score = float(economic + 0.35 * learnability - complexity_penalty)
            return pareto_score, economic, learnability, float(complexity_penalty)

        # ---------------------------------------------------------------------
        # Tree Model HPO (Two-Layer Sequential HPO)
        # ---------------------------------------------------------------------
        tree_best_params = {}
        if base_winner in ["xgb", "et", "lgbm", "rf"]:
            tprint(f"  {base_winner} won the sizer race. Launching two-layer tree HPO...")
            import optuna

            tree_runtime_keys = {"eval_top_k_pct", "top_k_pct", "cooldown_hours"}

            def _tree_model_kwargs(params: dict[str, Any]) -> dict[str, Any]:
                return {k: v for k, v in dict(params or {}).items() if k not in tree_runtime_keys}

            # Setup aggressive pruner for both layers
            pruner = optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=5, interval_steps=1)
            tree_hpo_splits = _build_time_cv_splits(
                X_final,
                timestamps,
                groups,
                n_splits=1,
                purge=43200 if timestamps is not None else 12,
                embargo=43200 if timestamps is not None else 12,
            )

            # -----------------------------------------------------------------
            # Layer 1: Primary Parameters (Most Impactful)
            # -----------------------------------------------------------------
            tprint("  Layer 1 HPO: Optimizing primary parameters (n_estimators, max_depth, learning_rate, subsample, colsample_bytree)...")

            def layer1_objective(trial):
                # Suggest primary parameters only
                kwargs = {}
                if base_winner in ["xgb", "lgbm"]:
                    kwargs["n_estimators"] = trial.suggest_int("n_estimators", 100, 600)
                    kwargs["max_depth"] = trial.suggest_int("max_depth", 2, 5)
                    kwargs["learning_rate"] = trial.suggest_float("learning_rate", 5e-3, 0.08, log=True)
                    kwargs["subsample"] = trial.suggest_float("subsample", 0.4, 0.7)
                    kwargs["colsample_bytree"] = trial.suggest_float("colsample_bytree", 0.4, 0.7)
                    # Fixed secondary params for layer 1
                    kwargs["min_child_weight"] = 150
                    kwargs["reg_alpha"] = 1.0
                    kwargs["reg_lambda"] = 5.0
                elif base_winner in ["et", "rf"]:
                    kwargs["n_estimators"] = trial.suggest_int("n_estimators", 100, 600)
                    kwargs["max_depth"] = trial.suggest_int("max_depth", 3, 6)

                # Suggest search space refinement parameters
                top_k = float(trial.suggest_categorical("eval_top_k_pct", bucket_space["exec_top_k_choices"]))
                cooldown = float(trial.suggest_categorical("cooldown_hours", bucket_space["cooldown_choices"]))

                f_pnl_list = []
                f_pnl_day_list = []
                f_sortino_list = []
                f_trades_day_list = []
                f_win_rate_list = []
                f_maxdd_list = []
                f_ulcer_list = []
                f_tuw_list = []
                f_pf_list = []
                f_avg_win_list = []
                f_avg_loss_list = []
                f_sharpe_list = []

                for step, (train_idx, val_idx) in enumerate(tree_hpo_splits):
                    X_tr, X_va = X_final[train_idx], X_final[val_idx]
                    y_tr, y_va = y[train_idx], y[val_idx]

                    if sample_weight is not None:
                        mdl = _fit_base(base_winner, X_tr, y_tr, X_va, y_va, sample_weight[train_idx], **kwargs)
                    else:
                        mdl = _fit_base(base_winner, X_tr, y_tr, X_va, y_va, None, **kwargs)

                    pred_va = np.asarray(mdl.predict(X_va), dtype=float)

                    if timestamps is not None:
                        n_days_fold = _effective_day_count(timestamps[val_idx])
                    else:
                        n_days_fold = max(1.0, len(val_idx))

                    # Compute pseudo-pnl metric for this fold using suggested top_k_pct
                    k_val = max(1, int(top_k * len(pred_va)))
                    top_idx = np.argpartition(pred_va, -k_val)[-k_val:]
                    y_top = y_gross[val_idx][top_idx] if len(y_gross) > 0 else y_va[top_idx]
                    sel_p = pred_va[top_idx]
                    order = np.argsort(sel_p)
                    rk = np.empty(len(sel_p), dtype=float)
                    rk[order] = (np.arange(len(sel_p), dtype=float) + 0.5) / max(len(sel_p), 1)
                    size = 0.05 + 0.10 * rk
                    ret = (y_top - self.cost_pct) * size
                    ts_top = timestamps[val_idx][top_idx] if timestamps is not None else None
                    sym_top = symbols[val_idx][top_idx] if symbols is not None else None
                    xb_top = exit_bars_proxy[val_idx][top_idx] if exit_bars_proxy is not None else None

                    # Apply asset overlap enforcement
                    if ts_top is not None and sym_top is not None and len(ret) > 0:
                        keep = _asset_overlap_keep_mask(
                            timestamps=np.asarray(ts_top),
                            assets=np.asarray(sym_top),
                            exit_bars=np.asarray(xb_top) if xb_top is not None else None,
                            priority=sel_p,
                            bar_minutes=15,
                            cooldown_hours=float(cooldown),
                        )
                        if np.any(keep):
                            ret = ret[keep]
                            ts_top = ts_top[keep]

                    daily_top = _aggregate_daily_values(ret, ts_top)
                    pnl_total_top = float(np.sum(ret))
                    f_sortino, f_maxdd, f_ulcer, f_tuw = _stable_daily_pnl_metrics(ret, ts_top, start_equity=1.0)
                    f_score = _pnl_risk_objective(pnl_total_top, f_maxdd, f_ulcer, f_tuw, daily_top)
                    pnl_per_day_top = float(pnl_total_top / n_days_fold)
                    
                    pos_rets = ret[ret > 0]; neg_rets = ret[ret < 0]
                    avg_win = float(np.mean(pos_rets)) if len(pos_rets) > 0 else 0.0
                    avg_loss = float(np.mean(neg_rets)) if len(neg_rets) > 0 else 0.0
                    pf = float(np.sum(pos_rets) / abs(np.sum(neg_rets))) if len(neg_rets) > 0 and abs(np.sum(neg_rets)) > 1e-9 else (float('inf') if len(pos_rets) > 0 else 0.0)
                    sharpe = float(np.mean(daily_top) / np.std(daily_top) * np.sqrt(365.0)) if len(daily_top) > 1 and np.std(daily_top) > 1e-9 else 0.0

                    f_pnl_list.append(float(f_score))
                    f_pnl_day_list.append(float(pnl_per_day_top))
                    f_sortino_list.append(float(f_sortino))
                    f_trades_day_list.append(float(len(ret) / n_days_fold))
                    f_win_rate_list.append(float(np.mean(ret > 0.0) if len(ret) else 0.0))
                    f_maxdd_list.append(float(f_maxdd))
                    f_ulcer_list.append(float(f_ulcer))
                    f_tuw_list.append(float(f_tuw))
                    f_pf_list.append(pf); f_avg_win_list.append(avg_win); f_avg_loss_list.append(avg_loss); f_sharpe_list.append(sharpe)

                    trial.report(float(f_score), step)
                    if trial.should_prune(): raise optuna.TrialPruned()

                m_pnl_day = float(np.mean(f_pnl_day_list)) if f_pnl_day_list else -1e9
                pareto_score, economic_score, learnability_score, complexity_penalty = _tree_pareto_score(
                    objective_scores=f_pnl_list,
                    pnl_days=f_pnl_day_list,
                    sortinos=f_sortino_list,
                    params=kwargs,
                )
                trial_metrics = {
                    "ObjectiveScore": float(np.mean(f_pnl_list)) if f_pnl_list else -1e9,
                    "ParetoScore": float(pareto_score),
                    "ParetoEconomicScore": float(economic_score),
                    "ParetoLearnabilityScore": float(learnability_score),
                    "ParetoComplexityPenalty": float(complexity_penalty),
                    "PnL_per_day": m_pnl_day,
                    "Sortino": float(np.mean(f_sortino_list)) if f_sortino_list else 0.0,
                    "Trades_per_day": float(np.mean(f_trades_day_list)) if f_trades_day_list else 0.0,
                    "WinRate": float(np.mean(f_win_rate_list)) if f_win_rate_list else 0.0,
                    "MaxDD": float(np.mean(f_maxdd_list)) if f_maxdd_list else 1.0,
                    "Ulcer": float(np.mean(f_ulcer_list)) if f_ulcer_list else 1.0,
                    "TUW": float(np.mean(f_tuw_list)) if f_tuw_list else 1.0,
                    "ProfitFactor": float(np.mean(f_pf_list)) if f_pf_list else 0.0,
                    "AvgWin": float(np.mean(f_avg_win_list)) if f_avg_win_list else 0.0,
                    "AvgLoss": float(np.mean(f_avg_loss_list)) if f_avg_loss_list else 0.0,
                    "Sharpe": float(np.mean(f_sharpe_list)) if f_sharpe_list else 0.0,
                }
                _record_trial_metrics(trial, trial_metrics, stage="tree_layer1")
                return trial_metrics["ParetoScore"]

            optuna.logging.set_verbosity(optuna.logging.WARNING)
            layer1_study = optuna.create_study(direction="maximize", pruner=pruner, sampler=optuna.samplers.TPESampler(seed=42))

            # Enqueue a middle-of-the-road trial for layer 1
            layer1_study.enqueue_trial({
                "n_estimators": 300, "max_depth": 4, "learning_rate": 0.03,
                "subsample": 0.6, "colsample_bytree": 0.6
            })

            layer1_trials = self.tree_hpo_trials
            layer1_study.optimize(layer1_objective, n_trials=layer1_trials, n_jobs=self.n_jobs)

            layer1_best_params = layer1_study.best_params
            tprint(f"    Layer 1 Complete. Best primary params: {layer1_best_params}")

            # -----------------------------------------------------------------
            # Layer 2: Secondary Parameters (Regularization & Fine-tuning)
            # -----------------------------------------------------------------
            tprint("  Layer 2 HPO: Optimizing secondary parameters with primary params varying ±15%...")

            def layer2_objective(trial):
                # Suggest expanded search grid for winning architecture
                kwargs = _tree_model_kwargs(layer1_best_params)
                top_k = float(
                    layer1_best_params.get(
                        "eval_top_k_pct",
                        layer1_best_params.get("top_k_pct", 0.30),
                    )
                )
                cooldown = float(layer1_best_params.get("cooldown_hours", 1.0))

                if base_winner in ["xgb", "lgbm"]:
                    kwargs["min_child_weight"] = trial.suggest_int("min_child_weight", 100, 400)
                    kwargs["reg_alpha"] = trial.suggest_float("reg_alpha", 0.1, 20.0, log=True)
                    kwargs["reg_lambda"] = trial.suggest_float("reg_lambda", 1.0, 20.0, log=True)
                elif base_winner in ["et", "rf"]:
                    kwargs["min_samples_split"] = trial.suggest_int("min_samples_split", 80, 300)
                    kwargs["min_samples_leaf"] = trial.suggest_int("min_samples_leaf", 40, 160)
                    kwargs["ccp_alpha"] = trial.suggest_float("ccp_alpha", 0.0, 0.05)

                f_pnl_list = []
                f_pnl_day_list = []
                f_sortino_list = []
                f_trades_day_list = []
                f_win_rate_list = []
                f_maxdd_list = []
                f_ulcer_list = []
                f_tuw_list = []
                f_pf_list = []
                f_avg_win_list = []
                f_avg_loss_list = []
                f_sharpe_list = []

                for step, (train_idx, val_idx) in enumerate(tree_hpo_splits):
                    X_tr, X_va = X_final[train_idx], X_final[val_idx]
                    y_tr, y_va = y[train_idx], y[val_idx]

                    if sample_weight is not None:
                        mdl = _fit_base(base_winner, X_tr, y_tr, X_va, y_va, sample_weight[train_idx], **kwargs)
                    else:
                        mdl = _fit_base(base_winner, X_tr, y_tr, X_va, y_va, None, **kwargs)

                    pred_va = np.asarray(mdl.predict(X_va), dtype=float)

                    if timestamps is not None:
                        n_days_fold = _effective_day_count(timestamps[val_idx])
                    else:
                        n_days_fold = max(1.0, len(val_idx))

                    # Compute pseudo-pnl metric for this fold using fixed top_k
                    k_val = max(1, int(top_k * len(pred_va)))
                    top_idx = np.argpartition(pred_va, -k_val)[-k_val:]
                    y_top = y_gross[val_idx][top_idx] if len(y_gross) > 0 else y_va[top_idx]
                    sel_p = pred_va[top_idx]
                    order = np.argsort(sel_p)
                    rk = np.empty(len(sel_p), dtype=float)
                    rk[order] = (np.arange(len(sel_p), dtype=float) + 0.5) / max(len(sel_p), 1)
                    size = 0.05 + 0.10 * rk
                    ret = (y_top - self.cost_pct) * size
                    ts_top = timestamps[val_idx][top_idx] if timestamps is not None else None
                    sym_top = symbols[val_idx][top_idx] if symbols is not None else None
                    xb_top = exit_bars_proxy[val_idx][top_idx] if exit_bars_proxy is not None else None

                    # Apply asset overlap enforcement
                    if ts_top is not None and sym_top is not None and len(ret) > 0:
                        keep = _asset_overlap_keep_mask(
                            timestamps=np.asarray(ts_top),
                            assets=np.asarray(sym_top),
                            exit_bars=np.asarray(xb_top) if xb_top is not None else None,
                            priority=sel_p,
                            bar_minutes=15,
                            cooldown_hours=float(cooldown),
                        )
                        if np.any(keep):
                            ret = ret[keep]
                            ts_top = ts_top[keep]

                    daily_top = _aggregate_daily_values(ret, ts_top)
                    pnl_total_top = float(np.sum(ret))
                    f_sortino, f_maxdd, f_ulcer, f_tuw = _stable_daily_pnl_metrics(ret, ts_top, start_equity=1.0)
                    f_score = _pnl_risk_objective(pnl_total_top, f_maxdd, f_ulcer, f_tuw, daily_top)

                    pos_rets = ret[ret > 0]; neg_rets = ret[ret < 0]
                    avg_win = float(np.mean(pos_rets)) if len(pos_rets) > 0 else 0.0
                    avg_loss = float(np.mean(neg_rets)) if len(neg_rets) > 0 else 0.0
                    pf = float(np.sum(pos_rets) / abs(np.sum(neg_rets))) if len(neg_rets) > 0 and abs(np.sum(neg_rets)) > 1e-9 else (float('inf') if len(pos_rets) > 0 else 0.0)
                    sharpe = float(np.mean(daily_top) / np.std(daily_top) * np.sqrt(365.0)) if len(daily_top) > 1 and np.std(daily_top) > 1e-9 else 0.0

                    f_pnl_list.append(float(f_score))
                    f_pnl_day_list.append(float(pnl_total_top / n_days_fold))
                    f_sortino_list.append(float(f_sortino))
                    f_trades_day_list.append(float(len(ret) / n_days_fold))
                    f_win_rate_list.append(float(np.mean(ret > 0.0) if len(ret) else 0.0))
                    f_maxdd_list.append(float(f_maxdd))
                    f_ulcer_list.append(float(f_ulcer))
                    f_tuw_list.append(float(f_tuw))
                    f_pf_list.append(pf); f_avg_win_list.append(avg_win); f_avg_loss_list.append(avg_loss); f_sharpe_list.append(sharpe)

                    # Report intermediate result to Pruner
                    trial.report(float(f_score), step)
                    if trial.should_prune():
                        raise optuna.TrialPruned()

                m_pnl_day = float(np.mean(f_pnl_day_list)) if f_pnl_day_list else -1e9
                m_maxdd = float(np.mean(f_maxdd_list)) if f_maxdd_list else 1.0
                m_ulcer = float(np.mean(f_ulcer_list)) if f_ulcer_list else 1.0
                m_tuw = float(np.mean(f_tuw_list)) if f_tuw_list else 1.0

                pareto_score, economic_score, learnability_score, complexity_penalty = _tree_pareto_score(
                    objective_scores=f_pnl_list,
                    pnl_days=f_pnl_day_list,
                    sortinos=f_sortino_list,
                    params=kwargs,
                )
                trial_metrics = {
                    "ObjectiveScore": float(np.mean(f_pnl_list)) if f_pnl_list else -1e9,
                    "ParetoScore": float(pareto_score),
                    "ParetoEconomicScore": float(economic_score),
                    "ParetoLearnabilityScore": float(learnability_score),
                    "ParetoComplexityPenalty": float(complexity_penalty),
                    "PnL_per_day": m_pnl_day,
                    "Trades_per_day": float(np.mean(f_trades_day_list)) if f_trades_day_list else 0.0,
                    "WinRate": float(np.mean(f_win_rate_list)) if f_win_rate_list else 0.0,
                    "MaxDD": m_maxdd,
                    "Ulcer": m_ulcer,
                    "TUW": m_tuw,
                    "IntradayRisk": float(_intraday_risk_metric(max_dd=m_maxdd, ulcer=m_ulcer, tuw=m_tuw)),
                    "Sortino": float(np.mean(f_sortino_list)) if f_sortino_list else 0.0,
                    "ProfitFactor": float(np.mean(f_pf_list)) if f_pf_list else 0.0,
                    "AvgWin": float(np.mean(f_avg_win_list)) if f_avg_win_list else 0.0,
                    "AvgLoss": float(np.mean(f_avg_loss_list)) if f_avg_loss_list else 0.0,
                    "Sharpe": float(np.mean(f_sharpe_list)) if f_sharpe_list else 0.0,
                }
                _record_trial_metrics(trial, trial_metrics, stage="tree_layer2")
                return trial_metrics["ParetoScore"]

            layer2_study = optuna.create_study(direction="maximize", pruner=pruner, sampler=optuna.samplers.TPESampler(seed=42))

            # Enqueue best layer 1 params as starting point for layer 2
            _layer2_warm = {}
            if base_winner in ["xgb", "lgbm"]:
                for key in ("min_child_weight", "reg_alpha", "reg_lambda"):
                    if key in layer1_best_params:
                        _layer2_warm[key] = layer1_best_params[key]
            elif base_winner in ["et", "rf"]:
                for key in ("min_samples_split", "min_samples_leaf", "ccp_alpha"):
                    if key in layer1_best_params:
                        _layer2_warm[key] = layer1_best_params[key]
            if _layer2_warm:
                layer2_study.enqueue_trial(_layer2_warm)

            layer2_trials = max(10, layer1_trials // 2)
            layer2_study.optimize(layer2_objective, n_trials=layer2_trials, n_jobs=self.n_jobs)

            tree_best_params = layer2_study.best_params
            tprint(f"    Layer 2 Complete. Best params: {tree_best_params}")

            # Update cv_results_ with the new HPO trials for final reporting
            tree_hpo_df = pd.DataFrame([r for r in results if r.get("stage") in ["tree_layer1", "tree_layer2"]])
            if not tree_hpo_df.empty:
                self.cv_results_ = pd.concat([self.cv_results_, tree_hpo_df], ignore_index=True)
            tprint(f"    Layer 1 best score: {layer1_study.best_value:.4f}, Layer 2 best score: {layer2_study.best_value:.4f}")
            self.best_params_.update({f"tree_hpo_{k}": v for k, v in tree_best_params.items()})

        # Fit winner incorporating HPO params if present
        # Apply Ridge Gate Model for all base models (Ridge, XGB, ET, LGBM)
        # Step 1: Fit Ridge on all data to get gating predictions
        if base_winner.startswith("champion_"):
            if base_winner == "champion_single":
                feat_name = smoother_winner
                idx = base_feature_names.index(feat_name)
                pred_full = np.asarray(X[:, idx], dtype=float)
                self.weights_ = np.zeros(len(base_feature_names), dtype=float)
                self.weights_[idx] = 1.0
            else:
                feat_a, feat_b = smoother_winner.split("|", 1)
                idx_a = base_feature_names.index(feat_a)
                idx_b = base_feature_names.index(feat_b)
                pred_full = np.nanmean(np.asarray(X[:, [idx_a, idx_b]], dtype=float), axis=1)
                self.weights_ = np.zeros(len(base_feature_names), dtype=float)
                self.weights_[idx_a] = 0.5
                self.weights_[idx_b] = 0.5
            self.policy_model_bundle_ = None
            self.ridge_pipeline_ = None
            self.model_names_ = list(base_feature_names)
            self.scaler_ = None
            self.oof_policy_pred_ = np.asarray(pred_full, dtype=float)
            base_final = None
        else:
            base_step1 = _fit_base("ridge", X_final, y, X_final, y, sample_weight)
            p_step1 = np.asarray(base_step1.predict(X_final), dtype=float)
            opt_k_pct = self.best_params_.get('ranking_top_k_pct', self.top_k_pct)
            k_num = max(1, int(opt_k_pct * len(p_step1)))
            gate_idx = np.argpartition(p_step1, -k_num)[-k_num:]

            X_step2 = X_final[gate_idx]
            y_step2 = y[gate_idx]
            sw_step2 = sample_weight[gate_idx] if sample_weight is not None else None

            tprint(f"    Ridge Gate Model: Step 1 complete. Gating top {opt_k_pct:.1%} ({len(gate_idx)} samples) for Step 2 using {base_winner}.")

            # Step 2: Fit the winning model on gated data
            if base_winner == "ridge":
                base_final = _fit_base(base_winner, X_step2, y_step2, X_step2, y_step2, sw_step2)
            else:
                base_final = _fit_base(base_winner, X_step2, y_step2, X_step2, y_step2, sw_step2, **tree_best_params)
        
        # Log Top 10 Feature Importances for Tree Models
        if hasattr(base_final, "feature_importances_"):
            _imps = base_final.feature_importances_
            if len(_imps) == len(self.model_names_final_):
                _pairs = sorted(zip(self.model_names_final_, _imps), key=lambda x: x[1], reverse=True)[:10]
                tprint(f"    {base_winner} Top 10 Features: " + ", ".join([f"{n}: {v:.4f}" for n, v in _pairs]))

        if base_final is not None:
            pred_full = np.asarray(base_final.predict(X_final), dtype=float)

        # Apply smooth tail-weighted calibration to final bundle for ET and XGB
        top_calibrator_final = None
        if base_final is not None and base_winner in ["et", "xgb"]:
            try:
                from sklearn.isotonic import IsotonicRegression

                # Generate smooth weights emphasizing p >= 0.70
                q_70 = np.nanquantile(pred_full, 1.0 - top_frac)
                score_std = np.nanstd(pred_full)
                temperature = max(0.25 * score_std, 1e-6)
                alpha_calib = top30_boost if use_two_pass else 2.0

                sigmoid_weights = 1.0 / (1.0 + np.exp(-(pred_full - q_70) / temperature))

                w_calib = np.ones(len(pred_full), dtype=float) if sample_weight is None else np.asarray(sample_weight, dtype=float).copy()
                w_calib = w_calib * (1.0 + alpha_calib * sigmoid_weights)

                top_calibrator_final = IsotonicRegression(out_of_bounds="clip")
                top_calibrator_final.fit(pred_full, y, sample_weight=w_calib)
                pred_full = top_calibrator_final.predict(pred_full)
            except Exception:
                top_calibrator_final = None

        if base_final is not None:
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
                trade_outcomes.iloc[:n], tick_size=self.best_params_.get("tick_size_bps", 2.0) / 10000.0, k_max=5, entry_fill_horizon_bars=4,
                max_hold_bars=48, tp_pct=0.005, sl_pct=0.0025, trailing_pct=0.0,
                cost_pct=self.cost_pct, eta=0.0, tie_break_smallest_k=True,
            )
            k_col = "k_star_built_from_policy"
            k_target = np.clip(np.nan_to_num(k_built, nan=0.0), 0.0, 5.0)

        offset_X = np.nan_to_num(np.asarray(X_final, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        if self.oof_policy_pred_ is not None and len(self.oof_policy_pred_) == len(offset_X):
            offset_X = np.column_stack([offset_X, np.asarray(self.oof_policy_pred_, dtype=np.float32)])
            offset_X = np.nan_to_num(offset_X, nan=0.0, posinf=0.0, neginf=0.0)
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
                X_tr = np.nan_to_num(np.asarray(X_tr, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
                X_va = np.nan_to_num(np.asarray(X_va, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
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
        smoother_kinds = ["ridge", "huber"]
        for b in ["ridge", "et", "xgb"]:
            for sm in smoother_kinds:
                key = f"{b}+{sm}"
                out = _run_offset_candidate(b, sm)
                if out is not None:
                    offset_race[key] = out
        
        if offset_race:
            offset_winner = min(offset_race.keys(), key=lambda k: (offset_race[k]["mu_mae"], offset_race[k]["sigma_mae"]))
            parts = offset_winner.split("+")
            b_w = parts[0]
            s_w = parts[1] if len(parts) > 1 else "ridge"
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

        # Final production-facing OOS metrics must reflect the live inference chain:
        # 1) sizer score, 2) top-k gate, 3) chosen sizing model, 4) learned offset model.
        def _evaluate_joint_live_oos_metrics(
            model_preds_oos: pd.DataFrame | None,
            trade_outcomes_oos: pd.DataFrame | None,
            timestamps_oos: np.ndarray | None,
        ) -> Dict[str, Any]:
            if (
                model_preds_oos is None
                or trade_outcomes_oos is None
                or len(model_preds_oos) == 0
                or len(trade_outcomes_oos) == 0
            ):
                return {}
            req_cols = {
                "entry_price", "is_long", "future_opens", "future_highs",
                "future_lows", "future_closes",
            }
            if not req_cols.issubset(set(trade_outcomes_oos.columns)):
                return {}

            n_oos_eval = min(len(model_preds_oos), len(trade_outcomes_oos))
            preds_df = model_preds_oos.iloc[:n_oos_eval].copy()
            score = np.asarray(self.predict(preds_df), dtype=np.float64)
            if len(score) != n_oos_eval:
                return {}

            preds_for_offset = preds_df.copy()
            preds_for_offset["sizer_score_oof"] = score
            if self.limit_offset_model_bundle_ is not None or self.limit_offset_pipeline_ is not None:
                offset_k = np.asarray(self.predict_limit_offset_ticks(preds_for_offset), dtype=np.float64)
            else:
                offset_k = np.zeros(n_oos_eval, dtype=np.float64)
            no_offset_eval = self._evaluate_live_pipeline_from_scores(
                score=score,
                trade_outcomes=trade_outcomes_oos.iloc[:n_oos_eval],
                timestamps=timestamps_oos[:n_oos_eval] if timestamps_oos is not None else None,
                top_k_pct=float(self.best_params_.get("top_k_pct", self.top_k_pct)),
                cooldown_hours=float(self.best_params_.get("cooldown_hours", 1.0)),
                base_size=float(self.best_params_.get("base_size", 0.05)),
                rank_multiplier=float(self.best_params_.get("rank_multiplier", 0.10)),
                sizing_formula=str(self.best_params_.get("sizing_formula", "linear")),
                squash_k=float(self.best_params_.get("squash_k", 1.0)),
                offset_k_pred=None,
                pipeline_name="joint_live_oos_no_offset",
                include_deciles=False,
            )
            with_offset_eval = self._evaluate_live_pipeline_from_scores(
                score=score,
                trade_outcomes=trade_outcomes_oos.iloc[:n_oos_eval],
                timestamps=timestamps_oos[:n_oos_eval] if timestamps_oos is not None else None,
                top_k_pct=float(self.best_params_.get("top_k_pct", self.top_k_pct)),
                cooldown_hours=float(self.best_params_.get("cooldown_hours", 1.0)),
                base_size=float(self.best_params_.get("base_size", 0.05)),
                rank_multiplier=float(self.best_params_.get("rank_multiplier", 0.10)),
                sizing_formula=str(self.best_params_.get("sizing_formula", "linear")),
                squash_k=float(self.best_params_.get("squash_k", 1.0)),
                offset_k_pred=offset_k,
                pipeline_name="joint_live_oos",
                include_deciles=True,
                decile_prefix="oos",
            )
            with_offset_eval["PnL_total_no_offset"] = float(no_offset_eval.get("PnL_total", 0.0))
            with_offset_eval["PnL_per_day_no_offset"] = float(no_offset_eval.get("PnL_per_day", 0.0))
            with_offset_eval["ObjectiveScore_no_offset"] = float(no_offset_eval.get("ObjectiveScore", -1e9))
            with_offset_eval["Trades_per_day_no_offset"] = float(no_offset_eval.get("Trades_per_day", 0.0))
            return with_offset_eval

        self.is_fitted_ = True
        joint_oos_metrics = _evaluate_joint_live_oos_metrics(op_oos, to_oos, ts_oos)
        full_oos_metrics = dict(joint_oos_metrics or {})
        if full_oos_metrics:
            full_oos_metrics["holdout_selector"] = "full_walk_forward_oos"
        self.full_oos_metrics_ = full_oos_metrics if full_oos_metrics else {}
        repeated_rows: list[dict[str, Any]] = []
        repeated_min_selected = 25
        if op_oos is not None and to_oos is not None and len(op_oos) > 0 and len(to_oos) > 0 and self.repeated_oos_splits > 1:
            n_rep = min(self.repeated_oos_splits, max(1, len(op_oos) // max(len(op_oos) // self.repeated_oos_splits, 1)))
            rep_splits = self._create_rolling_walk_forward_splits(
                np.asarray(ts_oos) if ts_oos is not None else np.arange(len(op_oos)),
                n_splits=n_rep,
                train_fraction=max(0.0, 1.0 - self.oos_fraction),
                min_train_size=max(10, len(op_oos) // 4),
            )
            for rep_id, (_rep_train, rep_test) in enumerate(rep_splits, start=1):
                if len(rep_test) == 0:
                    continue
                rep_metrics = _evaluate_joint_live_oos_metrics(
                    op_oos.iloc[rep_test].reset_index(drop=True),
                    to_oos.iloc[rep_test].reset_index(drop=True),
                    np.asarray(ts_oos)[rep_test] if ts_oos is not None else None,
                )
                rep_metrics["holdout_id"] = rep_id
                repeated_rows.append(rep_metrics)
        if repeated_rows:
            self.repeated_oos_results_ = pd.DataFrame(repeated_rows)
            med = self.repeated_oos_results_.median(numeric_only=True)
            repeated_oos_metrics = dict(full_oos_metrics or {})
            repeated_oos_metrics["holdout_selector"] = "median_repeated_temporal_oos"
            for key in ("PnL_total", "PnL_per_day", "Trades_per_day", "Sortino", "MaxDD", "Ulcer", "TUW", "IntradayRisk", "ObjectiveScore", "ProfitFactor", "AvgWin", "AvgLoss", "WinRate", "N_selected"):
                if key in med.index:
                    repeated_oos_metrics[key] = float(med[key])
            repeated_oos_metrics["n_repeated_holdouts"] = int(len(self.repeated_oos_results_))
            repeated_oos_metrics["repeated_min_selected_threshold"] = int(repeated_min_selected)
            repeated_oos_metrics["repeated_median_selected_ok"] = bool(float(med.get("N_selected", 0.0)) >= repeated_min_selected)
            repeated_oos_metrics["full_oos_n_selected"] = int(full_oos_metrics.get("N_selected", 0)) if full_oos_metrics else 0
            repeated_oos_metrics["full_oos_n_days"] = float(full_oos_metrics.get("N_days", 0.0)) if full_oos_metrics else 0.0
            repeated_oos_metrics["full_oos_trades_per_day"] = float(full_oos_metrics.get("Trades_per_day", 0.0)) if full_oos_metrics else 0.0
            if repeated_oos_metrics["repeated_median_selected_ok"]:
                joint_oos_metrics = repeated_oos_metrics
            else:
                joint_oos_metrics = dict(full_oos_metrics or {})
                joint_oos_metrics["holdout_selector"] = "full_walk_forward_oos_repeated_holdouts_too_sparse"
                joint_oos_metrics["n_repeated_holdouts"] = int(len(self.repeated_oos_results_))
                joint_oos_metrics["repeated_min_selected_threshold"] = int(repeated_min_selected)
                joint_oos_metrics["repeated_median_n_selected"] = float(med.get("N_selected", 0.0))
        if full_oos_metrics:
            self.full_oos_metrics_ = full_oos_metrics
        if joint_oos_metrics:
            self.best_oos_metrics_ = joint_oos_metrics
            tprint(
                f"  Joint live OOS metrics: PnL/Day={joint_oos_metrics.get('PnL_per_day', 0.0):.6f}, "
                f"ObjectiveScore={joint_oos_metrics.get('ObjectiveScore', -1e9):.6f}, "
                f"IntradayRisk={joint_oos_metrics.get('IntradayRisk', 0.0):.6f}, "
                f"PF={joint_oos_metrics.get('ProfitFactor', 0.0):.3f}, "
                f"AvgWin={joint_oos_metrics.get('AvgWin', 0.0):.6f}, AvgLoss={joint_oos_metrics.get('AvgLoss', 0.0):.6f}, "
                f"N={joint_oos_metrics.get('N_selected', 0)}"
            )
        else:
            self.best_oos_metrics_ = {}

        # Keep walk-forward OOS metrics as the production-facing evidence.
        self.backtest_metrics_ = None
        self.cv_backtest_consistency_ = None

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

        # Use optimized batch prediction if available
        if _USE_OPTIMIZED_PREDICTIONS and self.policy_model_bundle_ is not None:
            try:
                # Use adaptive batching for optimal performance
                batch_predictor = BatchPredictor(batch_size=None, adaptive_batching=True)
                base = self.policy_model_bundle_.get("base_model")
                top_calibrator = self.policy_model_bundle_.get("top_calibrator")
                smoother = self.policy_model_bundle_.get("smoother_model")
                iso = self.policy_model_bundle_.get("isotonic_model")

                raw_score = np.asarray(batch_predictor.predict_batched(base, X), dtype=float)

                # Apply top-30% calibrator if available
                if top_calibrator is not None:
                    raw_score = np.asarray(batch_predictor.predict_batched(top_calibrator, raw_score.reshape(-1, 1)), dtype=float).flatten()

                raw_score = np.asarray(batch_predictor.predict_batched(smoother, raw_score.reshape(-1, 1)), dtype=float).flatten()
                if iso is not None:
                    raw_score = np.asarray(batch_predictor.predict_batched(iso, raw_score.reshape(-1, 1)), dtype=float).flatten()
            except Exception as e:
                # Fall back to legacy implementation
                tprint(f"WARNING: Optimized batch prediction failed in ridge sizer, falling back: {e}")
                if self.policy_model_bundle_ is not None:
                    base = self.policy_model_bundle_.get("base_model")
                    top_calibrator = self.policy_model_bundle_.get("top_calibrator")
                    smoother = self.policy_model_bundle_.get("smoother_model")
                    iso = self.policy_model_bundle_.get("isotonic_model")
                    raw_score = np.asarray(base.predict(X), dtype=float)

                    # Apply top-30% calibrator if available
                    if top_calibrator is not None:
                        raw_score = np.asarray(top_calibrator.predict(raw_score), dtype=float)

                    raw_score = np.asarray(smoother.predict(raw_score.reshape(-1, 1)), dtype=float)
                    if iso is not None:
                        raw_score = np.asarray(iso.predict(raw_score), dtype=float)
        elif self.policy_model_bundle_ is not None:
            base = self.policy_model_bundle_.get("base_model")
            top_calibrator = self.policy_model_bundle_.get("top_calibrator")
            smoother = self.policy_model_bundle_.get("smoother_model")
            iso = self.policy_model_bundle_.get("isotonic_model")
            raw_score = np.asarray(base.predict(X), dtype=float)

            # Apply top-30% calibrator if available
            if top_calibrator is not None:
                raw_score = np.asarray(top_calibrator.predict(raw_score), dtype=float)

            raw_score = np.asarray(smoother.predict(raw_score.reshape(-1, 1)), dtype=float)
            if iso is not None:
                raw_score = np.asarray(iso.predict(raw_score), dtype=float)
        elif self.ridge_pipeline_ is not None:
            raw_score = self.ridge_pipeline_.predict(X)
        elif self.scaler_ is not None and self.weights_ is not None:
            X_scaled = self.scaler_.transform(X)
            raw_score = X_scaled @ self.weights_
        elif self.weights_ is not None:
            raw_score = X @ self.weights_
        else:
            raise RuntimeError("No fitted prediction backend found")

        # Apply custom dynamic sizing logic with Numba JIT if available
        sizing_formula = str(self.best_params_.get("sizing_formula", "sigmoid")).lower()
        if self.policy_model_bundle_ is not None:
            squash_k = float(self.policy_model_bundle_.get("squash_k", 1.0))
        else:
            squash_k = float(self.best_params_.get("squash_k", 1.0))

        # Use Numba JIT for sizing if available
        if _USE_OPTIMIZED_PREDICTIONS and _USE_NUMBA:
            try:
                c0 = 0.0  # Center parameter for sigmoid/tanh
                base_sz = float(getattr(self, "base_size_", self.best_params_.get("base_size", 0.05)))
                r_sca = float(getattr(self, "rank_multiplier_", self.best_params_.get("rank_multiplier", 0.10)))
                s_min = 0.0
                s_max = 1.0

                if sizing_formula == "sigmoid":
                    z = sigmoid_sizing_numba(raw_score, squash_k, c0, s_min, s_max)
                elif sizing_formula == "tanh":
                    z = tanh_sizing_numba(raw_score, squash_k, c0, s_min, s_max)
                else:  # concave
                    z = concave_sizing_numba(raw_score, squash_k, c0, s_min, s_max)

                return np.clip(base_sz + r_sca * z, 0.0, 1.0)
            except Exception as e:
                tprint(f"WARNING: Numba JIT sizing failed, falling back to numpy: {e}")

        # Legacy numpy implementation
        if sizing_formula == "sigmoid":
            z = 1.0 / (1.0 + np.exp(-squash_k * raw_score))
        elif sizing_formula == "concave":
            pos = np.clip(raw_score, 0.0, None)
            pos_max = np.max(pos)
            if pos_max > 1e-9:
                pos = pos / pos_max
            z = pos ** squash_k
        else:
            z = 0.5 * (1.0 + np.tanh(squash_k * raw_score))

        base_sz = float(getattr(self, "base_size_", self.best_params_.get("base_size", 0.05)))
        r_sca = float(getattr(self, "rank_multiplier_", self.best_params_.get("rank_multiplier", 0.10)))

        return np.clip(base_sz + r_sca * z, 0.0, 1.0)

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
    
    
    def apply_entry_policy_filter(
        self,
        model_preds: pd.DataFrame,
        atr_vec: np.ndarray,
        entry_price: float = 1.0,
    ) -> np.ndarray:
        """Apply entry policy filter (second mask from bucket_params).
        
        This applies the entry policy decision from bucket_params.json as a
        second-pass filter after candidate thresholds.
        
        Returns:
            Boolean mask where True = place_order.
        """
        if self.entry_policy_config_ is None:
            return np.ones(len(model_preds), dtype=bool)
        
        from extreme_price_movements.entry_policy import compute_entry_policy_decision
        
        # Validate model_names and DataFrame have required columns
        if not self.model_names_:
            if len(model_preds.columns) == 0:
                raise ValueError("No model names available and model_preds has no columns")
            score_col = model_preds.columns[0]
        else:
            score_col = self.model_names_[0]
            if score_col not in model_preds.columns:
                raise ValueError(f"Score column '{score_col}' not found in model_preds. Available: {list(model_preds.columns)}")
        
        scores = model_preds[score_col].values
        
        # Vectorized version where possible - batch process for efficiency
        # Note: Full vectorization requires compute_entry_policy_decision to support array inputs
        # For now, we use a list comprehension which is faster than a for loop
        n = len(scores)
        atr_default = 0.02
        
        # Pre-extract arrays to avoid repeated indexing
        atr_values = np.asarray(atr_vec) if atr_vec is not None else np.full(n, atr_default)
        if len(atr_values) < n:
            atr_values = np.pad(atr_values, (0, n - len(atr_values)), mode='constant', constant_values=atr_default)
        
        # Use list comprehension with explicit loop (faster than index-based loop)
        mask = np.array([
            bool(compute_entry_policy_decision(
                entry_px=entry_price,
                atr_frac=float(atr_values[i]),
                score=float(scores[i]),
                bucket_cfg=self.entry_policy_config_,
            ).get("place_order", True))
            for i in range(n)
        ], dtype=bool)
        
        return mask
    
    def get_weights(self) -> Dict[str, float]:
        """Return learned combination weights per model.
        
        Returns:
            Dictionary mapping model names to weights
        """
        if not self.is_fitted_ or self.weights_ is None or self.model_names_ is None:
            raise RuntimeError("RidgePositionSizer must be fitted before get_weights")
        
        return {name: float(w) for name, w in zip(self.model_names_, self.weights_)}

    def get_feature_importance(self, top_n: int = 10) -> Dict[str, float]:
        """Return top N features by importance or absolute coefficient.
        
        Returns:
            Dictionary mapping feature names to importance values
        """
        if not self.is_fitted_ or self.model_names_final_ is None:
            return {}
        
        base_model = None
        if self.policy_model_bundle_ is not None:
            base_model = self.policy_model_bundle_.get("base_model")
        elif self.ridge_pipeline_ is not None:
            if hasattr(self.ridge_pipeline_, 'named_steps'):
                base_model = self.ridge_pipeline_.named_steps.get('model')
            else:
                base_model = self.ridge_pipeline_
        
        if base_model is None:
            return {}
            
        imps = None
        if hasattr(base_model, "feature_importances_"):
            imps = base_model.feature_importances_
        elif hasattr(base_model, "coef_"):
            # Linear model uses absolute coefficients as proxy for importance
            imps = np.abs(np.asarray(base_model.coef_).flatten())
        
        if imps is None or len(imps) != len(self.model_names_final_):
            return {}
            
        pairs = sorted(zip(self.model_names_final_, imps), key=lambda x: x[1], reverse=True)
        return {name: float(val) for name, val in pairs[:top_n]}
    
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
            'model_names_ridge': getattr(self, 'model_names_ridge_', None),
            'model_names_tree': getattr(self, 'model_names_tree_', None),
            'best_params': self.best_params_,
            'best_target_name': getattr(self, 'best_target_name_', None),
            'selected_training_target_name': getattr(self, 'selected_training_target_name_', None),
            'selected_training_target_family': getattr(self, 'selected_training_target_family_', None),
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
            'position_hard_cap': self.position_hard_cap,
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
            'entry_policy_config': self.entry_policy_config_,
            'candidate_threshold_config': self.candidate_threshold_config_,
            'threshold_low': self.threshold_low_,
            'threshold_high': self.threshold_high_,
            # Nested CV and backtest attributes
            'nested_cv_results': self.nested_cv_results_.to_dict() if self.nested_cv_results_ is not None else None,
            'best_nested_cv_params': self.best_nested_cv_params_,
            'backtest_metrics': self.backtest_metrics_,
            'cv_backtest_consistency': self.cv_backtest_consistency_,
            'full_oos_metrics': getattr(self, 'full_oos_metrics_', None),
            'best_oos_metrics': getattr(self, 'best_oos_metrics_', None),
            'target_family_ab': getattr(self, 'target_family_ab_', None),
        }
        
        # Save policy_model_bundle_ if it exists (for full calibration pipeline)
        # Add SHA-256 integrity hash to verify data hasn't been tampered with
        if self.policy_model_bundle_ is not None:
            bundle_dict = {
                'base_name': self.policy_model_bundle_.get('base_name'),
                'smoother_name': self.policy_model_bundle_.get('smoother_name'),
                'base_model': self.policy_model_bundle_.get('base_model'),
                'smoother_model': self.policy_model_bundle_.get('smoother_model'),
                'isotonic_model': self.policy_model_bundle_.get('isotonic_model'),
                'top_calibrator': self.policy_model_bundle_.get('top_calibrator'),
                'squash_fn': self.policy_model_bundle_.get('squash_fn', 'tanh'),
                'squash_k': self.policy_model_bundle_.get('squash_k', 1.0),
            }
            bundle_pkl = pickle.dumps(bundle_dict)
            # Compute SHA-256 hash of the pickled data for integrity verification
            bundle_hash = hashlib.sha256(bundle_pkl).hexdigest()
            save_dict['policy_model_bundle_pkl'] = base64.b64encode(bundle_pkl).decode('latin1')
            save_dict['policy_model_bundle_hash'] = bundle_hash
        
        # Save limit_offset_model_bundle_ if it exists
        # Add SHA-256 integrity hash to verify data hasn't been tampered with
        if self.limit_offset_model_bundle_ is not None:
            bundle_dict = {
                'base_name': self.limit_offset_model_bundle_.get('base_name'),
                'smoother_name': self.limit_offset_model_bundle_.get('smoother_name'),
                'base_model': self.limit_offset_model_bundle_.get('base_model'),
                'smoother_model': self.limit_offset_model_bundle_.get('smoother_model'),
                'isotonic_model': self.limit_offset_model_bundle_.get('isotonic_model'),
            }
            bundle_pkl = pickle.dumps(bundle_dict)
            # Compute SHA-256 hash of the pickled data for integrity verification
            bundle_hash = hashlib.sha256(bundle_pkl).hexdigest()
            save_dict['limit_offset_model_bundle_pkl'] = base64.b64encode(bundle_pkl).decode('latin1')
            save_dict['limit_offset_model_bundle_hash'] = bundle_hash
        
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
            position_hard_cap=float(save_dict.get('position_hard_cap', 0.20)),
            winsor_q_low=float(save_dict.get('winsor_q_low', 0.01)),
            winsor_q_high=float(save_dict.get('winsor_q_high', 0.99)),
        )
        
        instance.weights_ = np.array(save_dict['weights']) if save_dict['weights'] else None
        instance.model_names_ = save_dict['model_names']
        instance.best_params_ = save_dict['best_params']
        instance.selected_training_target_name_ = save_dict.get('selected_training_target_name')
        instance.selected_training_target_family_ = save_dict.get('selected_training_target_family')
        instance.target_race_metrics_ = save_dict.get('target_race_metrics')
        instance.full_oos_metrics_ = save_dict.get('full_oos_metrics')
        instance.target_family_ab_ = save_dict.get('target_family_ab')
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
        instance.best_oos_metrics_ = save_dict.get('best_oos_metrics')
        
        # Restore policy_model_bundle_ if it was saved
        # With integrity verification using SHA-256 hash
        if save_dict.get('policy_model_bundle_pkl') is not None:
            try:
                bundle_pkl = base64.b64decode(save_dict['policy_model_bundle_pkl'].encode('latin1'))
                stored_hash = save_dict.get('policy_model_bundle_hash')
                
                # Verify integrity if hash is present (new format)
                if stored_hash is not None:
                    computed_hash = hashlib.sha256(bundle_pkl).hexdigest()
                    if computed_hash != stored_hash:
                        tprint(f"  SECURITY WARNING: policy_model_bundle_ hash mismatch! Data may be tampered. rejecting load.")
                        raise ValueError("Integrity check failed: policy_model_bundle_ data has been tampered with")
                    tprint(f"  Verified policy_model_bundle_ integrity (SHA-256)")
                else:
                    # Backward compatibility: warn about missing hash for old files
                    tprint(f"  WARNING: Loading policy_model_bundle_ without integrity hash (old format). Consider re-saving with new format.")
                
                bundle = pickle.loads(bundle_pkl)
                instance.policy_model_bundle_ = bundle
                tprint(f"  Restored policy_model_bundle_ (base={bundle.get('base_name')}, smoother={bundle.get('smoother_name')})")
            except ValueError:
                # Re-raise integrity errors
                raise
            except Exception as e:
                tprint(f"  WARNING: Failed to restore policy_model_bundle_: {e}")
        
        # Restore limit_offset_model_bundle_ if it was saved
        # With integrity verification using SHA-256 hash
        if save_dict.get('limit_offset_model_bundle_pkl') is not None:
            try:
                bundle_pkl = base64.b64decode(save_dict['limit_offset_model_bundle_pkl'].encode('latin1'))
                stored_hash = save_dict.get('limit_offset_model_bundle_hash')
                
                # Verify integrity if hash is present (new format)
                if stored_hash is not None:
                    computed_hash = hashlib.sha256(bundle_pkl).hexdigest()
                    if computed_hash != stored_hash:
                        tprint(f"  SECURITY WARNING: limit_offset_model_bundle_ hash mismatch! Data may be tampered. rejecting load.")
                        raise ValueError("Integrity check failed: limit_offset_model_bundle_ data has been tampered with")
                    tprint(f"  Verified limit_offset_model_bundle_ integrity (SHA-256)")
                else:
                    # Backward compatibility: warn about missing hash for old files
                    tprint(f"  WARNING: Loading limit_offset_model_bundle_ without integrity hash (old format). Consider re-saving with new format.")
                
                bundle = pickle.loads(bundle_pkl)
                instance.limit_offset_model_bundle_ = bundle
                tprint(f"  Restored limit_offset_model_bundle_ (base={bundle.get('base_name')})")
            except ValueError:
                # Re-raise integrity errors
                raise
            except Exception as e:
                tprint(f"  WARNING: Failed to restore limit_offset_model_bundle_: {e}")
        
        # Restore model names
        if save_dict.get('model_names_ridge'):
            instance.model_names_ridge_ = save_dict['model_names_ridge']
        if save_dict.get('model_names_tree'):
            instance.model_names_tree_ = save_dict['model_names_tree']
        
        # Restore entry policy and candidate threshold configs
        if save_dict.get('entry_policy_config'):
            instance.entry_policy_config_ = save_dict['entry_policy_config']
            tprint(f"  Restored entry_policy_config")
        if save_dict.get('candidate_threshold_config'):
            instance.candidate_threshold_config_ = save_dict['candidate_threshold_config']
            tprint(f"  Restored candidate_threshold_config")
        if save_dict.get('threshold_low') is not None:
            instance.threshold_low_ = float(save_dict['threshold_low'])
            instance.threshold_high_ = float(save_dict['threshold_high'])
            tprint(f"  Restored thresholds: low={instance.threshold_low_:.6f}, high={instance.threshold_high_:.6f}")

        # Restore nested CV and backtest attributes
        if save_dict.get('nested_cv_results') is not None:
            try:
                instance.nested_cv_results_ = pd.DataFrame(save_dict['nested_cv_results'])
                tprint(f"  Restored nested_cv_results ({len(instance.nested_cv_results_)} folds)")
            except Exception as e:
                tprint(f"  WARNING: Failed to restore nested_cv_results: {e}")
                instance.nested_cv_results_ = None
        else:
            instance.nested_cv_results_ = None

        instance.best_nested_cv_params_ = save_dict.get('best_nested_cv_params')
        instance.backtest_metrics_ = save_dict.get('backtest_metrics')
        instance.cv_backtest_consistency_ = save_dict.get('cv_backtest_consistency')

        if instance.backtest_metrics_ is not None:
            tprint(f"  Restored backtest_metrics (consistency={instance.cv_backtest_consistency_.get('is_consistent', False)})")

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
        "weight_entropy": float(-np.sum(p[p > 0.0] * np.log(p[p > 0.0]))) if np.any(p > 0.0) else 0.0,
    }
    horizon_shares: Dict[str, float] = {}
    for h in [int(v) for v in CFG.get("label_horizons_hours", [1, 2, 4])]:
        h_name = f"H{h}"
        m = np.array([f"_{h_name}" in n for n in names], dtype=bool)
        horizon_shares[h_name] = float(np.sum(absw[m]) / sum_abs) if sum_abs > 0 else 0.0
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
            diag["top_model_contributors"] = contrib_df.head(20).to_dict(orient="records")
    return diag


def run_oof_grid_backtest(
    oof_df: pd.DataFrame,
    start_equity: float = 100000.0,
    fee_roundtrip: float = 0.002,
    cooldown_hours: float = 0.0,
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

    def _asset_week_avg(g: pd.DataFrame) -> pd.Series:
        ordered = g.sort_values("ts").copy()
        rolling = (
            ordered.set_index("ts")["sizer_score_oof"]
            .rolling("7D", min_periods=1)
            .mean()
            .to_numpy(dtype=np.float64)
        )
        return pd.Series(rolling, index=ordered.index, dtype=np.float64)

    try:
        week_avg = df.groupby("asset", group_keys=False).apply(_asset_week_avg, include_groups=False)
    except TypeError:
        week_avg = df.groupby("asset", group_keys=False).apply(_asset_week_avg)
    df["week_avg"] = week_avg.reindex(df.index).astype(np.float64)
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
                offset_pct = sel["opt_limit_offset_pct"] if offset_mode == "optimizer" else pd.Series(0.0015, index=sel.index)
                max_stack_bars = int(np.nanmax(sel.get("label_policy_max_hold_bars", 48))) if len(sel) else 48
                opens_2d, open_lens = _stack_object_path_column(sel["future_opens"].values, max_stack_bars)
                highs_2d, high_lens = _stack_object_path_column(sel["future_highs"].values, max_stack_bars)
                lows_2d, low_lens = _stack_object_path_column(sel["future_lows"].values, max_stack_bars)
                closes_2d, close_lens = _stack_object_path_column(sel["future_closes"].values, max_stack_bars)
                valid = np.minimum(np.minimum(open_lens, high_lens), np.minimum(low_lens, close_lens)) > 0
                if not np.any(valid):
                    continue
                sel_valid = sel.iloc[np.flatnonzero(valid)].copy()
                frac_arr = _size_linear_5_15(sel_valid["rank_pct"].values, threshold)
                entry_px_raw = sel_valid["entry_price"].to_numpy(dtype=np.float64)
                e_price = entry_px_raw - offset_pct.loc[sel_valid.index].to_numpy(dtype=np.float64) * entry_px_raw
                sl_mult = sel_valid.get("label_policy_sl_atr_mult", pd.Series(np.nan, index=sel_valid.index)).to_numpy(dtype=np.float64)
                tp_ratio = sel_valid.get("label_policy_tp_sl_ratio", pd.Series(np.nan, index=sel_valid.index)).to_numpy(dtype=np.float64)
                atr_entry = sel_valid.get("atr_12_15m", pd.Series(np.nan, index=sel_valid.index)).to_numpy(dtype=np.float64)
                use_policy = np.isfinite(sl_mult) & np.isfinite(tp_ratio) & np.isfinite(atr_entry)
                sl_pct = np.full(len(sel_valid), 0.0025, dtype=np.float64)
                tp_pct = np.full(len(sel_valid), 0.0050, dtype=np.float64)
                if np.any(use_policy):
                    sl_abs = np.maximum(sl_mult[use_policy] * np.maximum(atr_entry[use_policy], 1e-9), 1e-9)
                    tp_abs = tp_ratio[use_policy] * sl_abs
                    sl_pct[use_policy] = sl_abs / np.maximum(entry_px_raw[use_policy], 1e-9)
                    tp_pct[use_policy] = tp_abs / np.maximum(entry_px_raw[use_policy], 1e-9)
                trailing_pct = sel_valid.get("label_policy_giveback_pct", pd.Series(0.0, index=sel_valid.index)).to_numpy(dtype=np.float64)
                max_bars_arr = sel_valid.get("label_policy_max_hold_bars", pd.Series(48, index=sel_valid.index)).to_numpy(dtype=np.int64)
                net_arr, exit_bars_arr, _ = _simulate_policy_utility_batch_details(
                    entry_prices=e_price,
                    is_longs=sel_valid["is_long"].to_numpy(dtype=bool),
                    future_opens=opens_2d[valid],
                    future_highs=highs_2d[valid],
                    future_lows=lows_2d[valid],
                    future_closes=closes_2d[valid],
                    tp_pcts=tp_pct,
                    sl_pcts=sl_pct,
                    trailing_pcts=trailing_pct,
                    max_bars_arr=np.minimum(max_bars_arr, np.minimum(np.minimum(open_lens[valid], high_lens[valid]), np.minimum(low_lens[valid], close_lens[valid]))),
                    cost_pct=fee_roundtrip,
                )
                overlap_keep = _asset_overlap_keep_mask(
                    timestamps=sel_valid["ts"].values,
                    assets=sel_valid["asset"].values,
                    exit_bars=exit_bars_arr,
                    priority=sel_valid["rank_pct"].values,
                    bar_minutes=15,
                    cooldown_hours=float(cooldown_hours),
                )
                sel_valid = sel_valid.iloc[np.flatnonzero(overlap_keep)].copy()
                frac_arr = frac_arr[overlap_keep]
                net_arr = net_arr[overlap_keep]
                pnl = start_equity * frac_arr * net_arr
                trades = len(pnl)
                wins = int((pnl > 0).sum())
                days = max((sel["ts"].max() - sel["ts"].min()).days, 1)
                sortino, maxdd, ulcer, _ = _stable_daily_pnl_metrics(
                    pnl,
                    sel_valid["ts"].values,
                    start_equity=start_equity,
                )
                phase1.append({
                    "phase": "phase1_non_sizing_grid",
                    "quantile": q,
                    "entry_offset_mode": offset_mode,
                    "tp_sl_ratio": ratio,
                    "sizing_mode": "linear_5_15",
                    "net_pnl": float(np.sum(pnl)),
                    "trades_per_day": float(trades / days),
                    "sortino": sortino,
                    "maxdd": maxdd,
                    "ulcer": ulcer,
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
    max_stack_bars = int(np.nanmax(sel.get("label_policy_max_hold_bars", 48))) if len(sel) else 48
    opens_2d, open_lens = _stack_object_path_column(sel["future_opens"].values, max_stack_bars)
    highs_2d, high_lens = _stack_object_path_column(sel["future_highs"].values, max_stack_bars)
    lows_2d, low_lens = _stack_object_path_column(sel["future_lows"].values, max_stack_bars)
    closes_2d, close_lens = _stack_object_path_column(sel["future_closes"].values, max_stack_bars)
    valid = np.minimum(np.minimum(open_lens, high_lens), np.minimum(low_lens, close_lens)) > 0
    if not np.any(valid):
        return phase1_df
    sel_kept = sel.iloc[np.flatnonzero(valid)].copy()
    entry_px_raw = sel_kept["entry_price"].to_numpy(dtype=np.float64)
    e_price = entry_px_raw - offset_pct.loc[sel_kept.index].to_numpy(dtype=np.float64) * entry_px_raw
    sl_mult = sel_kept.get("label_policy_sl_atr_mult", pd.Series(np.nan, index=sel_kept.index)).to_numpy(dtype=np.float64)
    tp_ratio = sel_kept.get("label_policy_tp_sl_ratio", pd.Series(np.nan, index=sel_kept.index)).to_numpy(dtype=np.float64)
    atr_entry = sel_kept.get("atr_12_15m", pd.Series(np.nan, index=sel_kept.index)).to_numpy(dtype=np.float64)
    use_policy = np.isfinite(sl_mult) & np.isfinite(tp_ratio) & np.isfinite(atr_entry)
    sl_pct = np.full(len(sel_kept), 0.0025, dtype=np.float64)
    tp_pct = np.full(len(sel_kept), 0.0050, dtype=np.float64)
    if np.any(use_policy):
        sl_abs = np.maximum(sl_mult[use_policy] * np.maximum(atr_entry[use_policy], 1e-9), 1e-9)
        tp_abs = tp_ratio[use_policy] * sl_abs
        sl_pct[use_policy] = sl_abs / np.maximum(entry_px_raw[use_policy], 1e-9)
        tp_pct[use_policy] = tp_abs / np.maximum(entry_px_raw[use_policy], 1e-9)
    trailing_pct = sel_kept.get("label_policy_giveback_pct", pd.Series(0.0, index=sel_kept.index)).to_numpy(dtype=np.float64)
    max_bars_arr = sel_kept.get("label_policy_max_hold_bars", pd.Series(48, index=sel_kept.index)).to_numpy(dtype=np.int64)
    net_ref, exit_bars_ref, _ = _simulate_policy_utility_batch_details(
        entry_prices=e_price,
        is_longs=sel_kept["is_long"].to_numpy(dtype=bool),
        future_opens=opens_2d[valid],
        future_highs=highs_2d[valid],
        future_lows=lows_2d[valid],
        future_closes=closes_2d[valid],
        tp_pcts=tp_pct,
        sl_pcts=sl_pct,
        trailing_pcts=trailing_pct,
        max_bars_arr=np.minimum(max_bars_arr, np.minimum(np.minimum(open_lens[valid], high_lens[valid]), np.minimum(low_lens[valid], close_lens[valid]))),
        cost_pct=fee_roundtrip,
    )
    overlap_keep = _asset_overlap_keep_mask(
        timestamps=sel_kept["ts"].values,
        assets=sel_kept["asset"].values,
        exit_bars=exit_bars_ref,
        priority=sel_kept["rank_pct"].values,
        bar_minutes=15,
        cooldown_hours=float(cooldown_hours),
    )
    sel_kept = sel_kept.iloc[np.flatnonzero(overlap_keep)].copy()
    net_ref = net_ref[overlap_keep]
    days = max((sel["ts"].max() - sel["ts"].min()).days, 1)
    phase2 = []
    for name, fn in size_methods.items():
        frac = np.asarray(fn(sel_kept["rank_pct"].values, win_threshold), dtype=float)
        pnl = start_equity * frac * net_ref
        trades = len(pnl)
        wins = int((pnl > 0).sum())
        sortino, maxdd, ulcer, _ = _stable_daily_pnl_metrics(
            pnl,
            sel_kept["ts"].values,
            start_equity=start_equity,
        )
        phase2.append({
            "phase": "phase2_sizing_compare",
            "quantile": win_q,
            "entry_offset_mode": win_offset_mode,
            "tp_sl_ratio": win_ratio,
            "sizing_mode": name,
            "net_pnl": float(np.sum(pnl)),
            "trades_per_day": float(trades / days),
            "sortino": sortino,
            "maxdd": maxdd,
            "ulcer": ulcer,
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
    entry_policy_config: dict | None = None,
    candidate_threshold_config: Optional[Dict] = None,
    warm_start_params: Optional[Dict] = None,
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

    def _select_numeric_oof_model_frame(df: pd.DataFrame) -> pd.DataFrame:
        """Return only the numeric model-score columns used for ridge fitting."""
        if not isinstance(df, pd.DataFrame):
            return pd.DataFrame(df)
        if "model_name" in df.columns and "pred" in df.columns:
            wide = df.pivot(columns="model_name", values="pred")
            return wide.sort_index(axis=1)

        meta_cols = {
            "timestamp",
            "symbol",
            "return",
            "is_long",
            "index",
            "trade_side",
            "bucket",
            "ts",
            "asset",
            "side",
            "close",
            "entry_price",
            "exit_price",
        }
        model_cols = [
            c
            for c in df.columns
            if c not in meta_cols and pd.api.types.is_numeric_dtype(df[c])
        ]
        dropped_cols = [c for c in df.columns if c not in model_cols and c not in meta_cols]
        if dropped_cols:
            tprint(
                "RidgePositionSizer: dropping non-numeric OOF columns before fit: "
                f"{dropped_cols}"
            )
        if not model_cols:
            raise ValueError("No numeric OOF model columns available for ridge fit")
        return df[model_cols].copy()
    
    def _compute_oof_rank_metrics(
        scores: np.ndarray,
        returns: np.ndarray,
        timestamps: np.ndarray | None = None,
        symbols: np.ndarray | None = None,
        exit_bars: np.ndarray | None = None,
        cooldown_hours: float = 0.0,
        start_equity: float = 1.0,
        base_size: float = 0.10,
        rank_multiplier: float = 0.0,
        cost_pct: float = 0.0,
    ) -> Dict[str, Any]:
        """Compute metrics for top-K% ranks based on OOF scores."""
        scores = np.asarray(scores, dtype=float)
        returns = np.asarray(returns, dtype=float)
        valid = np.isfinite(scores) & np.isfinite(returns)
        if not np.any(valid):
            return {}
        scores, returns = scores[valid], returns[valid]
        ts = timestamps[valid] if timestamps is not None else np.arange(len(returns))
        syms = symbols[valid] if symbols is not None else None
        exit_bars = np.asarray(exit_bars)[valid] if exit_bars is not None else None
        rank_pct = (np.argsort(np.argsort(scores)) + 1) / len(scores)
        
        # Calculate full-period duration once to avoid inflation when trades are sparse
        if timestamps is not None:
            days_full = _effective_day_count(ts)
        else:
            days_full = max(len(ts), 1)

        result = {}
        for top_pct in [0.30, 0.20, 0.10]:
            thresh = 1.0 - top_pct
            mask = rank_pct >= thresh
            if not np.any(mask):
                continue
            rets, ts_masked = returns[mask], ts[mask]
            syms_masked = syms[mask] if syms is not None else None
            xb_masked = exit_bars[mask] if exit_bars is not None else None
            scores_masked = scores[mask]
            
            if timestamps is not None and syms_masked is not None and len(rets) > 0:
                keep = _asset_overlap_keep_mask(
                    timestamps=np.asarray(ts_masked),
                    assets=np.asarray(syms_masked),
                    exit_bars=xb_masked,
                    priority=scores_masked,
                    bar_minutes=15,
                    cooldown_hours=float(cooldown_hours),
                )
                rets = rets[keep]
                ts_masked = np.asarray(ts_masked)[keep]
                scores_masked = scores_masked[keep]
            
            # Sizing based on provided parameters for realistic diagnostics
            rk_val = (np.argsort(np.argsort(scores_masked)) + 0.5) / len(rets)
            size = base_size + rank_multiplier * rk_val
            pnl = start_equity * size * (rets - cost_pct)
            n_trades = len(pnl)
            if n_trades == 0:
                continue

            pnl_total = float(np.sum(pnl))
            pnl_per_day = pnl_total / days_full
            trades_per_day = float(n_trades / days_full)

            sortino, maxdd, ulcer, tuw = _stable_daily_pnl_metrics(
                pnl,
                ts_masked if timestamps is not None else None,
                start_equity=start_equity,
            )
            
            # Additional metrics requested by user
            pos_rets = pnl[pnl > 0]
            neg_rets = pnl[pnl < 0]
            avg_win = float(np.mean(pos_rets)) if len(pos_rets) > 0 else 0.0
            avg_loss = float(np.mean(neg_rets)) if len(neg_rets) > 0 else 0.0
            
            if len(neg_rets) > 0 and abs(np.sum(neg_rets)) > 1e-9:
                profit_factor = float(np.sum(pos_rets) / abs(np.sum(neg_rets)))
            else:
                profit_factor = float('inf') if len(pos_rets) > 0 else 0.0

            # Compute Sharpe (annualized)
            daily_returns = _aggregate_daily_values(pnl, ts_masked if timestamps is not None else None)
            if len(daily_returns) > 1 and np.std(daily_returns) > 1e-9:
                sharpe = float(np.mean(daily_returns) / np.std(daily_returns) * np.sqrt(365.0))
            else:
                sharpe = 0.0

            prefix = f"oof_top{int(top_pct*100)}"
            result.update({
                f"{prefix}_pnl_total": pnl_total,
                f"{prefix}_pnl_per_day": pnl_per_day,
                f"{prefix}_trades_per_day": trades_per_day,
                f"{prefix}_n_trades": n_trades,
                f"{prefix}_win_rate": float(np.mean(rets > 0.0)),
                f"{prefix}_profit_factor": profit_factor,
                f"{prefix}_avg_win": avg_win,
                f"{prefix}_avg_loss": avg_loss,
                f"{prefix}_sharpe": sharpe,
                f"{prefix}_sortino": sortino,
                f"{prefix}_maxdd": maxdd,
                f"{prefix}_ulcer": ulcer,
                f"{prefix}_time_under_water": tuw,
            })
        return result
    
    # Extract configuration
    cfg = cfg or {}
    gamma_range = cfg.get('gamma_range', (0.0, 0.8))
    alpha_range = cfg.get('alpha_range', (1e-4, 1e-1))
    delta_range = cfg.get('delta_range', (0.5, 2.0))
    n_grid_points = cfg.get('n_grid_points', 100) # Updated from 10 to 100
    cost_pct = cfg.get('cost_pct', 0.0025)
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

    fit_oof_preds = _select_numeric_oof_model_frame(oof_preds)
    tprint(
        f"RidgePositionSizer: using {len(fit_oof_preds.columns)} numeric OOF model columns "
        f"for fit (from {len(oof_preds.columns)} total columns)"
    )

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
        use_nested_cv=bool(cfg.get('sizer_use_nested_cv', True)),
        max_fit_samples=cfg.get('sizer_max_fit_samples', 8000),
        n_jobs=int(cfg.get('sizer_n_jobs', 1)),
        patience=int(cfg.get('patience', 20)),
        stage1_cv_folds=int(cfg.get('sizer_stage1_cv_folds', 3)),
        stage1_two_fold_refine=bool(cfg.get('sizer_stage1_two_fold_refine', False)),
        stage1_n_trials=cfg.get('sizer_stage1_n_trials'),
        stage2_cv_folds=int(cfg.get('sizer_stage2_cv_folds', 3)),
        stage2_n_trials=cfg.get('sizer_stage2_n_trials'),
        tree_hpo_trials=cfg.get('sizer_tree_hpo_trials'),
        target_train_fraction=float(cfg.get('sizer_target_train_fraction', 0.50)),
        oos_fraction=float(cfg.get('sizer_oos_fraction', 0.30)),
        min_oos_days=int(cfg.get('sizer_min_oos_days', 14)),
        repeated_oos_splits=int(cfg.get('sizer_repeated_oos_splits', 3)),
        stage2_lock_formula=bool(cfg.get('sizer_stage2_lock_formula', True)),
        forced_target_candidates=cfg.get('sizer_forced_target_candidates'),
    )
    sizer.bucket_name_ = bucket_name
    
    # Set entry policy and candidate threshold configs if provided
    if entry_policy_config is not None:
        sizer.entry_policy_config_ = entry_policy_config
    if candidate_threshold_config:
        sizer.threshold_low_ = candidate_threshold_config.get('extreme_price_pct', 0.0)
        sizer.threshold_high_ = sizer.threshold_low_ * 1.5 # Heuristic
        
    if warm_start_params:
        sizer.warm_start_params_ = warm_start_params
        
    tprint(f"Sizer.fit: trade_outcomes has {len(trade_outcomes)} rows, "
           f"timestamps is {'None' if timestamps is None else f'len={len(timestamps)}'}")
    
    sizer.fit(fit_oof_preds, trade_outcomes, timestamps=timestamps, groups=groups, 
              labels=labels, symbols=symbols)
    
    # Compute metrics
    weights = sizer.get_weights()
    metrics = {
        'weights': weights,
        'best_params': sizer.best_params_,
        'best_target_name': getattr(sizer, 'best_target_name_', None),
        'selected_training_target_name': getattr(sizer, 'selected_training_target_name_', None),
        'selected_training_target_family': getattr(sizer, 'selected_training_target_family_', None),
        'target_race_metrics': getattr(sizer, 'target_race_metrics_', None),
        'n_models': len(weights),
        'n_trades': len(trade_outcomes),
        'sizer_uses_linear_5_15_training_eval': True,
        'feature_selection_diag_ridge': getattr(sizer, 'feature_selection_diag_ridge_', None),
        'feature_selection_diag_tree': getattr(sizer, 'feature_selection_diag_tree_', None),
        'feature_ic_diag': getattr(sizer, 'feature_ic_diag_', None).to_dict(orient='records') if getattr(sizer, 'feature_ic_diag_', None) is not None else None,
    }
    if isinstance(policy_opt_meta, dict):
        metrics['label_policy_optimizer'] = policy_opt_meta
    
    # Include Walk-Forward OOS metrics if available
    if hasattr(sizer, 'best_oos_metrics_'):
        metrics['best_oos_metrics'] = sizer.best_oos_metrics_
    if hasattr(sizer, 'full_oos_metrics_'):
        metrics['full_oos_metrics'] = sizer.full_oos_metrics_
    if getattr(sizer, 'repeated_oos_results_', None) is not None:
        metrics['repeated_oos_results'] = sizer.repeated_oos_results_.to_dict(orient='records')
    if getattr(sizer, 'oos_protocol_', None) is not None:
        metrics['oos_protocol'] = dict(sizer.oos_protocol_)
    if getattr(sizer, 'target_family_ab_', None) is not None:
        metrics['target_family_ab'] = dict(sizer.target_family_ab_)
    ridge_diag = _compute_ridge_weight_diagnostics(weights=weights, oof_preds=fit_oof_preds, sizer=sizer)
    if ridge_diag:
        metrics["weight_diagnostics"] = ridge_diag

    try:
        score_df = fit_oof_preds[sizer.model_names_].copy()
        score = np.asarray(sizer.predict(score_df), dtype=np.float64)
        preds_for_offset = score_df.copy()
        preds_for_offset["sizer_score_oof"] = score
        offset_pred = None
        if (sizer.limit_offset_model_bundle_ is not None or sizer.limit_offset_pipeline_ is not None):
            offset_pred = np.asarray(sizer.predict_limit_offset_ticks(preds_for_offset), dtype=np.float64)
        no_offset_eval = sizer._evaluate_live_pipeline_from_scores(
            score=score,
            trade_outcomes=trade_outcomes,
            timestamps=timestamps,
            top_k_pct=float(sizer.best_params_.get("top_k_pct", sizer.top_k_pct)),
            cooldown_hours=float(sizer.best_params_.get("cooldown_hours", 1.0)),
            base_size=float(sizer.best_params_.get("base_size", 0.05)),
            rank_multiplier=float(sizer.best_params_.get("rank_multiplier", 0.10)),
            sizing_formula=str(sizer.best_params_.get("sizing_formula", "linear")),
            squash_k=float(sizer.best_params_.get("squash_k", 1.0)),
            offset_k_pred=None,
            pipeline_name="diagnostic_no_offset",
        )
        with_offset_eval = sizer._evaluate_live_pipeline_from_scores(
            score=score,
            trade_outcomes=trade_outcomes,
            timestamps=timestamps,
            top_k_pct=float(sizer.best_params_.get("top_k_pct", sizer.top_k_pct)),
            cooldown_hours=float(sizer.best_params_.get("cooldown_hours", 1.0)),
            base_size=float(sizer.best_params_.get("base_size", 0.05)),
            rank_multiplier=float(sizer.best_params_.get("rank_multiplier", 0.10)),
            sizing_formula=str(sizer.best_params_.get("sizing_formula", "linear")),
            squash_k=float(sizer.best_params_.get("squash_k", 1.0)),
            offset_k_pred=offset_pred,
            pipeline_name="diagnostic_with_offset",
        )
        feature_ic_diag = getattr(sizer, "feature_ic_diag_", None)
        best_raw_ic = float(feature_ic_diag["spearman_ic"].fillna(-1.0).max()) if feature_ic_diag is not None and not feature_ic_diag.empty else 0.0
        best_raw_feature = str(feature_ic_diag.sort_values("spearman_ic", ascending=False).iloc[0]["feature"]) if feature_ic_diag is not None and not feature_ic_diag.empty else None
        metrics["alpha_retention_waterfall"] = {
            "best_raw_feature": best_raw_feature,
            "best_raw_feature_ic": best_raw_ic,
            "combined_score_ic": float(pd.Series(score).corr(pd.Series(trade_outcomes["return"].values), method="spearman")) if "return" in trade_outcomes.columns else 0.0,
            "oof_pnl_total_no_offset": float(no_offset_eval.get("PnL_total", 0.0)),
            "oof_pnl_per_day_no_offset": float(no_offset_eval.get("PnL_per_day", 0.0)),
            "oof_pnl_total_with_offset": float(with_offset_eval.get("PnL_total", 0.0)),
            "oof_pnl_per_day_with_offset": float(with_offset_eval.get("PnL_per_day", 0.0)),
        }
        metrics["oof_no_offset_metrics"] = no_offset_eval
        metrics["oof_with_offset_metrics"] = with_offset_eval
    except Exception as e:
        tprint(f"WARNING: alpha retention diagnostics failed: {e}")

    # Confirmation diagnostics for utility/offset model families and feature alignment.
    _util_bundle = getattr(sizer, "policy_model_bundle_", None) or {}
    _offset_bundle = getattr(sizer, "limit_offset_model_bundle_", None) or {}
    metrics["utility_policy_model_family"] = str(_util_bundle.get("base_name", "ridge"))
    metrics["utility_smoother_family"] = str(_util_bundle.get("smoother_name", "ridge"))
    metrics["offset_model_family"] = str(_offset_bundle.get("base_name", "ridge")) if (_offset_bundle or sizer.limit_offset_pipeline_ is not None) else None
    metrics["offset_smoother_family"] = str(_offset_bundle.get("smoother_name", "ridge")) if (_offset_bundle or sizer.limit_offset_pipeline_ is not None) else None
    metrics["limit_offset_enabled"] = bool((_offset_bundle or sizer.limit_offset_pipeline_ is not None))
    metrics["sizer_feature_names"] = list(sizer.model_names_ or [])
    metrics["offset_feature_names"] = list(sizer.limit_offset_features_ or [])
    offset_base = [c for c in (sizer.limit_offset_features_ or []) if c != "sizer_score_oof"]
    metrics["offset_base_features_match_sizer_features"] = bool(offset_base == list(sizer.model_names_ or []))
    
    # Add CV results summary. There are two schemas:
    # 1. HPO trial results with ObjectiveScore/PnL_per_day/IntradayRisk/etc.
    # 2. Model-race summaries with score/mu_objective_top/mu_pnl_top/etc.
    if getattr(sizer, "cv_summary_", None):
        cvs = dict(sizer.cv_summary_)
        metrics['cv_best_selector_column'] = str(cvs.get('selector', 'nested_joint_holdout_median'))
        metrics['cv_best_selector_value'] = float(cvs.get('objective', 0.0))
        metrics['cv_best_pnl_total'] = float(cvs.get('pnl_total', 0.0))
        metrics['cv_best_pnl_per_day'] = float(cvs.get('pnl_per_day', 0.0))
        metrics['cv_best_objective'] = float(cvs.get('objective', 0.0))
        metrics['cv_best_intraday_risk'] = float(cvs.get('intraday_risk', 0.0))
        metrics['cv_best_trades_per_day'] = float(cvs.get('trades_per_day', 0.0))
        metrics['cv_best_sortino'] = float(cvs.get('sortino', 0.0))
        metrics['cv_best_maxdd'] = float(cvs.get('maxdd', 0.0))
        metrics['cv_best_ulcer'] = float(cvs.get('ulcer', 0.0))
        metrics['cv_best_tuw'] = float(cvs.get('tuw', 0.0))
        metrics['cv_best_n_selected'] = int(cvs.get('n_trades', 0))
    elif sizer.cv_results_ is not None and len(sizer.cv_results_) > 0:
        cv_df = sizer.cv_results_
        best_col = None
        for candidate in ("ObjectiveScore", "score", "mu_objective_top", "PnL_per_day", "mu_pnl_top", "pnl_per_day", "J_zscore"):
            if candidate in cv_df.columns:
                best_col = candidate
                break
        # Schema 1 (Trial Results) usually has 'ObjectiveScore' or 'PnL_per_day'
        # Schema 2 (Race Summary) has 'mu_objective_top' or 'mu_pnl_top'
        # We prioritize HPO trials (Stage 1/2 or Tree HPO) over race summaries for best_row selection
        trial_cols = ["ObjectiveScore", "PnL_per_day", "Trades_per_day"]
        has_trials = all(c in cv_df.columns for c in trial_cols)
        
        if has_trials:
            # Filter to only HPO trials to avoid picking race summary rows (which lack full metrics)
            trials_df = cv_df[cv_df["ObjectiveScore"].notna()].copy()
            if not trials_df.empty:
                best_idx = trials_df["ObjectiveScore"].idxmax()
                best_row = trials_df.loc[best_idx]
            else:
                best_idx = cv_df[best_col].idxmax()
                best_row = cv_df.loc[best_idx]
        else:
            best_idx = cv_df[best_col].idxmax()
            best_row = cv_df.loc[best_idx]

        def _metric_float(*cols: str, default: float = 0.0) -> float:
            for col in cols:
                if col in best_row.index and pd.notna(best_row[col]):
                    return float(best_row[col])
            return float(default)

        def _metric_int(*cols: str, default: int = 0) -> int:
            for col in cols:
                if col in best_row.index and pd.notna(best_row[col]):
                    return int(best_row[col])
            return int(default)

        metrics['cv_best_selector_column'] = str(best_col)
        metrics['cv_best_selector_value'] = _metric_float(best_col)
        metrics['cv_best_pnl_total'] = _metric_float('PnL_total', default=0.0)
        metrics['cv_best_pnl_per_day'] = _metric_float('PnL_per_day', 'mu_pnl_top', 'pnl_per_day', default=0.0)
        metrics['cv_best_objective'] = _metric_float('ObjectiveScore', 'mu_objective_top', best_col, default=0.0)
        metrics['cv_best_intraday_risk'] = _metric_float('IntradayRisk', default=0.0)
        metrics['cv_best_trades_per_day'] = _metric_float('Trades_per_day', default=0.0)
        metrics['cv_best_unique_symbols_selected'] = _metric_int('Unique_Symbols_Selected', default=0)
        metrics['cv_best_sortino'] = _metric_float('Sortino', 'mu_sortino_top', 'sortino', default=0.0)
        metrics['cv_best_maxdd'] = _metric_float('MaxDD', 'mu_maxdd_top', default=0.0)
        metrics['cv_best_ic'] = _metric_float('IC', default=0.0)
        metrics['cv_best_winrate'] = _metric_float('WinRate', default=0.0)
        metrics['cv_best_profit_factor'] = _metric_float('ProfitFactor', default=0.0)
        metrics['cv_best_avg_win'] = _metric_float('AvgWin', default=0.0)
        metrics['cv_best_avg_loss'] = _metric_float('AvgLoss', default=0.0)
        metrics['cv_best_ulcer'] = _metric_float('Ulcer', default=0.0)
        metrics['cv_best_tuw'] = _metric_float('TUW', default=0.0)
        metrics['cv_best_n_selected'] = _metric_int('N_selected', default=0)

        # Position sizing metrics from best trial
        if 'pos_sizing' in best_row:
            pos_sizing = best_row['pos_sizing']
            # Deserialize if it's a string (from CSV)
            if isinstance(pos_sizing, str):
                try:
                    pos_sizing = json.loads(pos_sizing)
                except:
                    pos_sizing = {}
            metrics['cv_best_pos_sizing'] = pos_sizing if isinstance(pos_sizing, dict) else {}
    
    # Compute OOF rank-based metrics (top30%, top20%, top10%)
    if sizer.oof_preds_ is not None and sizer.oof_targets_ is not None:
        oof_score = sizer.predict(sizer.oof_preds_)
        oof_targets = sizer.oof_targets_
        oof_ts = getattr(sizer, 'oof_timestamps_', None)
        oof_symbols = symbols[:len(oof_targets)] if symbols is not None and len(symbols) >= len(oof_targets) else None
        oof_exit_bars = None
        if "label_policy_max_hold_bars" in trade_outcomes.columns:
            oof_exit_bars = np.asarray(trade_outcomes["label_policy_max_hold_bars"].values[:len(oof_targets)], dtype=np.int64)
        if len(oof_score) == len(oof_targets) and (oof_ts is None or len(oof_ts) == len(oof_targets)):
            rank_metrics = _compute_oof_rank_metrics(
                oof_score,
                oof_targets,
                oof_ts,
                symbols=oof_symbols,
                exit_bars=oof_exit_bars,
                cooldown_hours=float(sizer.best_params_.get("cooldown_hours", 1.0)) if sizer.best_params_ else 1.0,
                base_size=float(sizer.best_params_.get("base_size", 0.10)) if sizer.best_params_ else 0.10,
                rank_multiplier=float(sizer.best_params_.get("rank_multiplier", 0.0)) if sizer.best_params_ else 0.0,
                cost_pct=float(sizer.cost_pct),
            )
            metrics.update(rank_metrics)
        else:
            tprint(
                "WARNING: Skipping OOF rank metrics due to length mismatch "
                f"(scores={len(oof_score)}, targets={len(oof_targets)}, "
                f"timestamps={len(oof_ts) if oof_ts is not None else 'None'})"
            )
    
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

    # Export OOF parquet for diagnostics.
    # The sizer target may be ranking/utility based, so do not hardcode it as H4.
    if sizer.oof_preds_ is not None and sizer.oof_targets_ is not None:
        n_oof = len(sizer.oof_targets_)
        oof_score = sizer.predict(sizer.oof_preds_)
        if len(oof_score) != n_oof:
            tprint(
                "WARNING: Skipping OOF parquet export due to length mismatch "
                f"(scores={len(oof_score)}, targets={n_oof})"
            )
        else:
            oof_payload: dict = {
                "score": oof_score.astype(np.float32),
                "sizer_training_target_oof": sizer.oof_targets_.astype(np.float32),
                "sizer_training_target_name": np.asarray(
                    [str(getattr(sizer, "selected_training_target_name_", "unknown"))] * n_oof,
                    dtype=object,
                ),
            }
            if sizer.oof_timestamps_ is not None and len(sizer.oof_timestamps_) == n_oof:
                oof_payload["ts"] = pd.to_datetime(sizer.oof_timestamps_, utc=True, errors="coerce")
            else:
                oof_payload["ts"] = pd.RangeIndex(n_oof)
            if sizer.oof_symbols_ is not None and len(sizer.oof_symbols_) == n_oof:
                oof_payload["asset"] = sizer.oof_symbols_.astype(str)
            else:
                oof_payload["asset"] = "ALL"
            if getattr(sizer, 'oof_policy_pred_', None) is not None:
                oof_payload["sizer_score_oof"] = np.asarray(sizer.oof_policy_pred_, dtype=np.float32)
            if sizer.best_params_ is not None:
                oof_payload["cooldown_hours"] = np.full(
                    n_oof,
                    float(sizer.best_params_.get("cooldown_hours", 1.0)),
                    dtype=np.float32,
                )
            if getattr(sizer, 'oof_limit_offset_pred_', None) is not None:
                k_hat = np.asarray(sizer.oof_limit_offset_pred_, dtype=np.float32)
                oof_payload["opt_limit_offset_ticks"] = k_hat
                tick_size_bps = float(cfg.get('TICK_SIZE_BPS', 2.0))
                oof_payload["opt_limit_offset_pct"] = ((tick_size_bps / 10000.0) * k_hat).astype(np.float32)
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
            bucket_label = str(bucket_name or "UNKNOWN_BUCKET").upper()
            oof_df_out["bucket"] = bucket_label
            _oof_dir = data_root / "artifacts" / (run_id or "latest") / "ridge_sizer"
            _oof_dir.mkdir(parents=True, exist_ok=True)
            _bucket_slug = bucket_label.lower()
            _oof_path = _oof_dir / f"ridge_sizer_oof_{_bucket_slug}.parquet"
            oof_df_out.to_parquet(_oof_path, index=False)
            _combined_oof_path = _oof_dir / "ridge_sizer_oof_all.parquet"
            try:
                if _combined_oof_path.exists():
                    _prev = pd.read_parquet(_combined_oof_path)
                    if "bucket" in _prev.columns:
                        _prev = _prev[_prev["bucket"].astype(str).str.upper() != bucket_label]
                    oof_combined = pd.concat([_prev, oof_df_out], ignore_index=True)
                else:
                    oof_combined = oof_df_out.copy()
                oof_combined.to_parquet(_combined_oof_path, index=False)
            except Exception as _oof_exc:
                tprint(f"WARNING: Failed to refresh combined Ridge OOF parquet: {_oof_exc}")
            metrics['oof_parquet_path'] = str(_oof_path)
            metrics['oof_combined_parquet_path'] = str(_combined_oof_path)
            tprint(f"Saved Ridge sizer OOF parquet to {_oof_path}")
            tprint(f"Updated combined Ridge sizer OOF parquet at {_combined_oof_path}")

            try:
                bt_df = run_oof_grid_backtest(
                    oof_df_out,
                    cooldown_hours=float(sizer.best_params_.get("cooldown_hours", 1.0)) if sizer.best_params_ else 1.0,
                )
                if not bt_df.empty:
                    _bt_path = _oof_dir / f"ridge_sizer_oof_backtest_grid_{_bucket_slug}.csv"
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
        scores = sizer.predict(fit_oof_preds)
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
    # Print all models sorted by absolute weight (descending)
    sorted_weights = sorted(weights.items(), key=lambda x: abs(x[1]), reverse=True)
    tprint("\n=== RIDGE SIZER TOP FEATURES (ALL MODELS) ===")
    for i, (name, w) in enumerate(sorted_weights, 1):
        tprint(f"  {i:2d}. {name}: {w:+.6f}")
    tprint("=== END RIDGE SIZER FEATURES ===\n")
    tprint(f"  Best hyperparameters:")
    if sizer.best_params_:
        tprint(f"    alpha: {sizer.best_params_['alpha']:.6f}")
        tprint(f"    delta: {sizer.best_params_['delta']:.3f}")
        tprint(f"    gamma: {sizer.best_params_['gamma']:.3f}")
    tprint("=" * 80)
    
    metrics['top_features'] = sizer.get_feature_importance(top_n=10)
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
    
    def _fill_nonfinite_oof_vector(values, neutral: float = 0.0) -> np.ndarray:
        arr = np.asarray(values, dtype=np.float64).copy()
        finite = np.isfinite(arr)
        if finite.all():
            return arr
        if finite.any():
            fill = float(np.nanmedian(arr[finite]))
        else:
            fill = float(neutral)
        arr[~finite] = fill
        return arr

    # Combine into wide format DataFrame for predictions
    pred_cols = {
        name: _fill_nonfinite_oof_vector(df["oof_pred"].values, neutral=0.0)
        for name, df in oof_dfs.items()
    }
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
