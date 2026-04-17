"""
simple_offset_generator.py

Generates limit order offsets based on Ridge confidence from simple_position_sizer.py.

Flow:
1. Takes results from simple_position_sizer.py (optimal threshold, wallet settings)
2. Filters trades above the optimal confidence threshold from profit_proxy_table_
3. Applies same position sizing (linear/convex wallet % allocation from sizer)
4. Computes ideal offset in ATR% (0.1%, 0.2%, 0.3%) based on Ridge confidence score
5. Generates metrics with fill rate deltas comparing to baseline (no offset)

Key Design:
- Uses optimal threshold selected by simple_position_sizer.py's evaluate_selection_profit_proxy()
- Offset expressed in ATR% terms (not ticks) for consistency
- Fill rate delta tracked as primary execution quality metric
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from numba import jit, prange
from scipy.stats import spearmanr

from extreme_price_movements.metrics import _stable_equity_and_drawdown
from extreme_price_movements.run_ridge_sizer import (
    load_base_oof_predictions,
    load_meta_oof_predictions,
    load_trade_outcomes,
)
from extreme_price_movements.simple_position_sizer import (
    SimpleHeadRidgeSizer,
    clean_and_standardize,
    collect_ridge_head_columns,
    detect_meta_head_keys,
    evaluate_selection_profit_proxy,
    walk_forward_temporal_splits,
)

logger = logging.getLogger(__name__)


# NOTE: This function is currently unused. The probabilistic fallback path uses
# direct np.exp() calculation instead (line ~830). Retained for potential future use.
@jit(nopython=True, cache=True)
def _compute_fill_probability(
    offset_ticks: np.ndarray,
    confidence_scores: np.ndarray,
    base_fill_prob: float = 0.9,
    sensitivity: float = 0.2,
) -> np.ndarray:
    """
    Compute fill probability as function of offset and confidence.

    Higher confidence + larger offset = higher fill probability.
    Returns fill probability in [0, 1].
    """
    n = len(offset_ticks)
    fill_probs = np.zeros(n, dtype=np.float32)

    for i in range(n):
        # Base fill probability decreases with larger offset (harder to fill)
        # But increases with confidence (we're more willing to wait)
        offset_penalty = 1.0 / (1.0 + sensitivity * offset_ticks[i])
        confidence_boost = 0.5 + 0.5 * confidence_scores[i]  # Map to [0.5, 1.0]
        fill_probs[i] = base_fill_prob * offset_penalty * confidence_boost

    return np.clip(fill_probs, 0.1, 0.99)


@jit(nopython=True, cache=True)
def _simulate_offset_execution_atr(
    entry_prices: np.ndarray,
    is_longs: np.ndarray,
    atr_values: np.ndarray,
    future_opens: np.ndarray,
    future_highs: np.ndarray,
    future_lows: np.ndarray,
    offset_atr_pct: np.ndarray,  # Offset as ATR% (e.g., 0.001 = 0.1%)
    max_wait_bars: int = 4,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Simulate limit order execution with offset expressed as ATR%.

    Args:
        offset_atr_pct: Offset as fraction of ATR (e.g., 0.001 = 0.1% of ATR)

    Returns:
        - executed: bool array indicating if order filled
        - fill_prices: price at which order filled (entry_price if not executed)
        - fill_bars: bars waited until fill (-1 if not filled)
        - entry_improvements: price improvement as fraction (for return adjustment)
    """
    n = len(entry_prices)
    executed = np.zeros(n, dtype=np.bool_)
    fill_prices = entry_prices.copy()
    fill_bars = np.full(n, -1, dtype=np.int32)
    entry_improvements = np.zeros(n, dtype=np.float64)

    for i in range(n):
        entry = entry_prices[i]
        is_long = is_longs[i]
        atr = atr_values[i] if i < len(atr_values) else entry * 0.001  # Fallback

        # Compute offset in price terms from ATR%
        offset_price = atr * offset_atr_pct[i]

        # Compute limit price based on direction and offset
        if is_long:
            limit_price = entry - offset_price
        else:
            limit_price = entry + offset_price

        # Check for fill in next bars
        max_b = min(max_wait_bars, len(future_opens[i]))
        for b in range(max_b):
            # For longs: fill if low <= limit_price
            # For shorts: fill if high >= limit_price
            if is_long:
                if future_lows[i, b] <= limit_price:
                    executed[i] = True
                    fill_prices[i] = limit_price
                    fill_bars[i] = b
                    # Entry improvement: bought below market = positive
                    entry_improvements[i] = (entry - limit_price) / entry
                    break
            else:
                if future_highs[i, b] >= limit_price:
                    executed[i] = True
                    fill_prices[i] = limit_price
                    fill_bars[i] = b
                    # Entry improvement: sold above market = positive
                    entry_improvements[i] = (limit_price - entry) / entry
                    break

    return executed, fill_prices, fill_bars, entry_improvements


def compute_offset_atr_pct_from_confidence(
    confidence_scores: np.ndarray,
    base_offset_atr_pct: float = 0.001,  # 0.1% default
    max_offset_atr_pct: float = 0.003,  # 0.3% default
    confidence_threshold: float = 0.0,  # Use optimal from sizer, not hardcoded
    scaling: str = "linear",
) -> np.ndarray:
    """
    Compute ideal offset in ATR% based on Ridge confidence.

    Args:
        confidence_scores: Normalized confidence scores [0, 1] from sizer
        base_offset_atr_pct: Minimum offset as ATR% (e.g., 0.001 = 0.1%)
        max_offset_atr_pct: Maximum offset as ATR% (e.g., 0.003 = 0.3%)
        confidence_threshold: Min confidence to apply offset (from sizer optimal threshold)
        scaling: "linear", "convex", or "concave"

    Returns:
        Array of offset values as ATR% (e.g., 0.001 = 0.1%)
    """
    # Normalize scores to [0, 1] if needed
    if confidence_scores.max() > 1.0 or confidence_scores.min() < 0.0:
        conf_norm = (confidence_scores - confidence_scores.min()) / (
            confidence_scores.max() - confidence_scores.min() + 1e-9
        )
    else:
        conf_norm = confidence_scores.copy()

    # Apply threshold mask
    above_threshold = conf_norm >= confidence_threshold

    # Scale offset based on confidence level above threshold
    if confidence_threshold >= 1.0:
        offset_scale = np.zeros_like(conf_norm)
    else:
        if scaling == "linear":
            offset_scale = (conf_norm - confidence_threshold) / (
                1.0 - confidence_threshold + 1e-9
            )
        elif scaling == "convex":
            # Aggressive scaling: high confidence gets max offset quickly
            offset_scale = (
                (conf_norm - confidence_threshold) / (1.0 - confidence_threshold + 1e-9)
            ) ** 0.5
        elif scaling == "concave":
            # Conservative scaling: requires very high confidence for max offset
            offset_scale = (
                (conf_norm - confidence_threshold) / (1.0 - confidence_threshold + 1e-9)
            ) ** 2.0
        else:
            offset_scale = conf_norm - confidence_threshold

    offset_scale = np.clip(offset_scale, 0.0, 1.0)

    # Compute offset in ATR%
    offsets = np.where(
        above_threshold,
        base_offset_atr_pct + offset_scale * (max_offset_atr_pct - base_offset_atr_pct),
        np.full_like(
            conf_norm, base_offset_atr_pct
        ),  # Even low confidence gets base offset
    )

    return offsets


def compute_offset_raw_return_from_confidence(
    confidence_scores: np.ndarray,
    expected_returns: np.ndarray,
    base_offset_ret: float = 0.0001,  # 1 bps minimum
    max_offset_ret: float = 0.001,  # 10 bps maximum
    confidence_threshold: float = 0.0,
    scaling: str = "linear",
    invert: bool = False,  # If True: higher confidence = LOWER offset
) -> np.ndarray:
    """
    Compute offset in raw return terms (price improvement as % of entry).

    Args:
        confidence_scores: Normalized confidence scores [0, 1]
        expected_returns: Expected return per trade (for scaling context)
        base_offset_ret: Minimum offset in return terms (e.g., 0.0001 = 1 bps)
        max_offset_ret: Maximum offset in return terms (e.g., 0.001 = 10 bps)
        confidence_threshold: Min confidence to apply offset
        scaling: "linear", "convex", "concave"
        invert: If True, higher confidence gets LOWER offset (tighter to market)

    Returns:
        Array of offset values as raw return (entry price improvement)
    """
    # Normalize confidence to [0, 1]
    if confidence_scores.max() > 1.0 or confidence_scores.min() < 0.0:
        conf_norm = (confidence_scores - confidence_scores.min()) / (
            confidence_scores.max() - confidence_scores.min() + 1e-9
        )
    else:
        conf_norm = confidence_scores.copy()

    # Scale offset based on confidence
    if confidence_threshold >= 1.0:
        offset_scale = np.zeros_like(conf_norm)
    else:
        if scaling == "linear":
            offset_scale = (conf_norm - confidence_threshold) / (
                1.0 - confidence_threshold + 1e-9
            )
        elif scaling == "convex":
            offset_scale = (
                (conf_norm - confidence_threshold) / (1.0 - confidence_threshold + 1e-9)
            ) ** 0.5
        elif scaling == "concave":
            offset_scale = (
                (conf_norm - confidence_threshold) / (1.0 - confidence_threshold + 1e-9)
            ) ** 2.0
        else:
            offset_scale = conf_norm - confidence_threshold

    offset_scale = np.clip(offset_scale, 0.0, 1.0)

    if invert:
        # Invert: higher confidence = lower offset (tighter to market)
        # Low confidence (scale=0) gets max_offset, high confidence (scale=1) gets base_offset
        offsets = max_offset_ret - offset_scale * (max_offset_ret - base_offset_ret)
    else:
        # Normal: higher confidence = higher offset (further from market)
        offsets = base_offset_ret + offset_scale * (max_offset_ret - base_offset_ret)

    return offsets


def adjust_returns_for_offset(
    original_returns: np.ndarray,
    is_longs: np.ndarray,
    entry_price_improvements: np.ndarray,
) -> np.ndarray:
    """
    Adjust trade returns for better entry price from offset.

    Key insight: If you enter at a better price:
    - For LONGS: limit price below market → entry is lower → gains are LARGER, losses are SMALLER
    - For SHORTS: limit price above market → entry is higher → gains are LARGER, losses are SMALLER

    Args:
        original_returns: Raw returns from original entry price
        is_longs: Boolean array True=long, False=short
        entry_price_improvements: Price improvement as fraction (e.g., 0.001 = 0.1% better entry)

    Returns:
        Adjusted returns accounting for better entry
    """
    # For both longs and shorts, better entry means adding the improvement to returns
    # Long: bought lower → extra gain on exit
    # Short: sold higher → extra gain on exit
    adjusted = original_returns + entry_price_improvements
    return adjusted


def apply_position_sizing_with_offset(
    returns: np.ndarray,
    confidence_scores: np.ndarray,
    executed: np.ndarray,
    wallet_range: Tuple[float, float] = (0.05, 0.15),
    sizing_mode: str = "linear",
    cost_pct: float = 0.003,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply position sizing to returns, accounting for execution.

    Args:
        returns: Raw net returns per trade
        confidence_scores: Confidence scores for sizing
        executed: Boolean array indicating if trade was executed
        wallet_range: (min_wallet%, max_wallet%) allocation
        sizing_mode: "linear", "convex", or "fixed"
        cost_pct: Transaction cost percentage

    Returns:
        - sized_returns: Returns after position sizing and costs
        - sizes: Position sizes applied
    """
    n = len(returns)

    # Only size executed trades
    executed_returns = returns.copy()
    executed_returns[~executed] = 0.0  # No return if not executed

    # Apply transaction costs to executed trades
    executed_returns[executed] -= cost_pct

    # Compute position sizes based on confidence
    if sizing_mode == "linear":
        # Linear interpolation from wallet_range[0] to wallet_range[1]
        sorted_idx = np.argsort(confidence_scores)
        sizes = np.zeros(n)
        sizes[sorted_idx] = np.linspace(wallet_range[0], wallet_range[1], n)
    elif sizing_mode == "convex":
        # Convex: higher confidence gets disproportionately larger size
        sorted_idx = np.argsort(confidence_scores)
        linear = np.linspace(0, 1, n)
        convex = linear**0.5  # Square root for convexity
        sizes = wallet_range[0] + convex * (wallet_range[1] - wallet_range[0])
        sized_array = np.zeros(n)
        sized_array[sorted_idx] = sizes
        sizes = sized_array
    elif sizing_mode == "concave":
        # Concave: size increases slowly at first then rapidly
        sorted_idx = np.argsort(confidence_scores)
        linear = np.linspace(0, 1, n)
        concave = linear**2.0
        sizes = wallet_range[0] + concave * (wallet_range[1] - wallet_range[0])
        sized_array = np.zeros(n)
        sized_array[sorted_idx] = sizes
        sizes = sized_array
    else:
        # Fixed size at max
        sizes = np.full(n, wallet_range[1])

    # Zero out sizes for non-executed trades
    sizes[~executed] = 0.0

    sized_returns = executed_returns * sizes

    return sized_returns, sizes


def evaluate_offset_strategy(
    returns: np.ndarray,
    timestamps: np.ndarray,
    confidence_scores: np.ndarray,
    offset_atr_pct: np.ndarray,  # Changed from offset_ticks to ATR%
    executed: np.ndarray,
    fill_bars: np.ndarray,
    wallet_range: Tuple[float, float] = (0.05, 0.15),
    sizing_mode: str = "linear",
    cost_pct: float = 0.003,
    n_days: float = 365.0,
    baseline_fill_rate: float = 1.0,  # For fill rate delta calculation
) -> Dict[str, Any]:
    """
    Evaluate strategy with offset applied.

    Returns comprehensive metrics including fill rate delta vs baseline.
    """
    # Basic info
    n_trades = len(returns)
    executed_mask = executed
    n_executed = np.sum(executed_mask)
    fill_rate = n_executed / n_trades if n_trades > 0 else 0.0

    if n_executed == 0:
        return {
            "n_trades": 0,
            "n_executed": 0,
            "fill_rate": 0.0,
            "fill_rate_delta": -1.0,
            "net_pnl": 0.0,
            "hit_rate": 0.0,
            "profit_factor": 0.0,
            "sortino": 0.0,
            "max_drawdown": 0.0,
        }

    # KEY METRIC: Fill rate delta vs baseline (market orders = 100% fill)
    fill_rate_delta = fill_rate - baseline_fill_rate
    fill_rate_delta_pct = fill_rate_delta * 100  # Convert to percentage points

    # Calculate hit_rate and profit_factor on RAW returns (before sizing)
    # This gives true trade performance, not distorted by position sizing
    raw_executed_rets = returns[executed_mask]
    hit_rate = np.mean(raw_executed_rets > 0) if len(raw_executed_rets) > 0 else 0.0

    # Profit factor on raw returns
    gross_profit_raw = np.sum(raw_executed_rets[raw_executed_rets > 0])
    gross_loss_raw = np.abs(np.sum(raw_executed_rets[raw_executed_rets < 0]))
    profit_factor = (
        gross_profit_raw / gross_loss_raw
        if gross_loss_raw > 1e-9
        else float(gross_profit_raw)
    )

    # Apply position sizing for PnL-based metrics
    sized_returns, sizes = apply_position_sizing_with_offset(
        returns, confidence_scores, executed, wallet_range, sizing_mode, cost_pct
    )

    # Filter to executed trades for metrics
    executed_mask_sized = executed & (sizes > 0)
    executed_rets = sized_returns[executed_mask_sized]
    executed_ts = timestamps[executed_mask_sized] if timestamps is not None else None

    net_pnl = np.sum(executed_rets)

    # Compute Sortino-like metric (downside deviation)
    rets = executed_rets
    downside_rets = rets[rets < 0]
    if len(downside_rets) > 1:
        downside_std = np.std(downside_rets)
    else:
        downside_std = 1e-6  # Small positive to avoid division by zero (need >1 sample for meaningful std)
    sortino = np.mean(rets) / downside_std

    # Drawdown
    _, dd_series = _stable_equity_and_drawdown(executed_rets)
    max_drawdown = np.max(dd_series) if len(dd_series) > 0 else 0.0

    # Execution quality
    avg_fill_bars = (
        np.mean(fill_bars[executed_mask]) if np.sum(executed_mask) > 0 else 0.0
    )
    avg_offset_atr_pct = (
        np.mean(offset_atr_pct[executed_mask]) if np.sum(executed_mask) > 0 else 0.0
    )

    # Per-trade metrics
    pnl_per_trade_bps = (net_pnl / n_executed) * 10000 if n_executed > 0 else 0.0
    trades_per_day = n_executed / n_days if n_days > 0 else 0.0

    # Cost of non-execution (opportunity cost)
    missed_trades = ~executed
    missed_returns = returns[missed_trades]
    opportunity_cost = (
        np.sum(missed_returns[missed_returns > 0]) if len(missed_returns) > 0 else 0.0
    )

    return {
        "n_trades": int(n_trades),
        "n_executed": int(n_executed),
        "fill_rate": float(fill_rate),
        "fill_rate_delta": float(fill_rate_delta),
        "fill_rate_delta_pct": float(fill_rate_delta_pct),
        "net_pnl": float(net_pnl),
        "hit_rate": float(hit_rate),
        "profit_factor": float(profit_factor),
        "sortino": float(sortino),
        "max_drawdown": float(max_drawdown),
        "avg_fill_bars": float(avg_fill_bars),
        "avg_offset_atr_pct": float(avg_offset_atr_pct),
        "pnl_per_trade_bps": float(pnl_per_trade_bps),
        "trades_per_day": float(trades_per_day),
        "mean_position_size": float(np.mean(sizes[executed_mask])),
        "opportunity_cost": float(opportunity_cost),  # Positive returns we missed
        "n_missed": int(np.sum(missed_trades)),
    }


def generate_offset_comparison(
    baseline_metrics: Dict[str, Any],
    offset_metrics: Dict[str, Any],
    strategy_name: str = "strategy",
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Generate comparison table between baseline and offset strategy.

    Includes fill_rate_delta as primary execution quality metric.
    """
    metrics_to_compare = [
        "net_pnl",
        "hit_rate",
        "profit_factor",
        "sortino",
        "max_drawdown",
        "n_executed",
        "fill_rate",
        "pnl_per_trade_bps",
        "fill_rate_delta",
        "fill_rate_delta_pct",
        "opportunity_cost",
    ]

    rows = []
    for metric in metrics_to_compare:
        base_val = baseline_metrics.get(metric, 0.0)
        offset_val = offset_metrics.get(metric, 0.0)

        # For fill_rate_delta, lower (less negative) is better
        if metric == "fill_rate_delta":
            is_better = offset_val > base_val  # Less negative = better
        elif metric == "opportunity_cost":
            is_better = offset_val < base_val  # Lower opportunity cost = better
        else:
            is_better = (
                offset_val > base_val
                if metric not in ["max_drawdown"]
                else offset_val < base_val
            )

        if base_val != 0:
            improvement = (offset_val - base_val) / abs(base_val) * 100
        else:
            improvement = 0.0 if offset_val == 0 else float("inf")

        rows.append(
            {
                "metric": metric,
                "baseline": base_val,
                "with_offset": offset_val,
                "improvement_pct": improvement,
                "better": "✓" if is_better else "✗" if not is_better else "=",
            }
        )

    df = pd.DataFrame(rows)

    # Summary with fill rate delta as key metric
    n_better = sum(1 for r in rows if r["better"] == "✓")
    n_worse = sum(1 for r in rows if r["better"] == "✗")

    summary = {
        "strategy": strategy_name,
        "metrics_better": n_better,
        "metrics_worse": n_worse,
        "net_pnl_delta": offset_metrics.get("net_pnl", 0)
        - baseline_metrics.get("net_pnl", 0),
        "sortino_delta": offset_metrics.get("sortino", 0)
        - baseline_metrics.get("sortino", 0),
        "fill_rate_delta_pct": offset_metrics.get("fill_rate_delta_pct", 0),
        "opportunity_cost": offset_metrics.get("opportunity_cost", 0),
        "n_missed": offset_metrics.get("n_missed", 0),
    }

    return df, summary


def run_simple_offset_generator_from_sizer(
    sizer_results: Dict[str, Any],
    trade_outcomes: pd.DataFrame,
    future_prices: Optional[Dict[str, np.ndarray]] = None,
    base_offset_atr_pct: Optional[float] = None,
    max_offset_atr_pct: Optional[float] = None,
    base_offset_ret: float = 0.0001,
    max_offset_ret: float = 0.001,
    use_raw_return_offset: bool = False,
    invert_offset: bool = False,
    offset_scaling: str = "linear",
    cost_pct: float = 0.003,
    max_wait_bars: int = 4,
    use_atr_values: Optional[np.ndarray] = None,
    barrier_conf_alpha: float = 0.5,
) -> Dict[str, Any]:
    """
    Main orchestrator for offset generation - takes simple_position_sizer.py results directly.

    Uses the optimal threshold and wallet settings from sizer_results['profit_proxy_table_']

    Key insight: With offset entry, gains are LARGER and losses are SMALLER because you
    enter at a better price (lower for longs, higher for shorts).

    Args:
        sizer_results: Output dict from run_simple_position_sizer() containing:
            - 'ridge_sizer_scores_': OOF Ridge predictions
            - 'ridge_profit_proxy_table_': Profit proxy with optimal threshold
            - 'ridge_opt_rets_': Optimal returns from baseline
            - 'ridge_opt_ts_': Optimal timestamps
        trade_outcomes: DataFrame with trade outcome data including 'entry_price', 'is_long'
        future_prices: Optional dict with 'opens', 'highs', 'lows', 'atr' arrays
        base_offset_atr_pct: Minimum offset as ATR% (0.001 = 0.1%). Use None for raw return mode.
        max_offset_atr_pct: Maximum offset as ATR% (0.003 = 0.3%). Use None for raw return mode.
        base_offset_ret: Minimum offset in raw return terms (1 bps = 0.0001)
        max_offset_ret: Maximum offset in raw return terms (10 bps = 0.001)
        use_raw_return_offset: If True, use raw return offsets instead of ATR%
        invert_offset: If True, higher confidence gets LOWER offset (tighter to market)
        offset_scaling: "linear", "convex", "concave"
        cost_pct: Transaction cost %
        max_wait_bars: Max bars to wait for limit fill
        use_atr_values: Optional ATR values per trade (for offset computation)

    Returns:
        Dictionary with baseline metrics, offset metrics, comparison table, and diagnostics.
    """
    # Determine mode: ATR% or raw return
    if use_raw_return_offset or (
        base_offset_atr_pct is None and max_offset_atr_pct is None
    ):
        use_raw_return_offset = True
        logger.info("Using RAW RETURN based offsets")
    else:
        logger.info(
            f"Using ATR% offsets: base={base_offset_atr_pct}, max={max_offset_atr_pct}"
        )
    # 1. Extract from sizer results

    # We may be forcing a specific model's scores to be used.
    # Check if this is passed via kwargs or in the sizer_results directly
    # If not specified, default to best_simple_score_
    force_model = sizer_results.get("_force_model_", None)

    if force_model == "ridge":
        ridge_scores = sizer_results.get("ridge_sizer_scores_")
        profit_proxy_df = sizer_results.get("ridge_profit_proxy_table_")
    elif force_model == "et":
        ridge_scores = sizer_results.get("et_sizer_scores_")
        profit_proxy_df = sizer_results.get("et_profit_proxy_table_")
    else:
        # Auto mode
        ridge_scores = sizer_results.get("best_simple_score_")
        # Use ridge_profit_proxy_table_ if Ridge was used, otherwise use profit_proxy_table_
        # (Though best_simple_score_ could also be ET, we should check best_simple_score_name_)
        best_name = sizer_results.get("best_simple_score_name_")
        if best_name == "ExtraTrees_Head_Sizer":
            profit_proxy_df = sizer_results.get("et_profit_proxy_table_")
        else:
            profit_proxy_df = sizer_results.get("ridge_profit_proxy_table_")

        if profit_proxy_df is None or profit_proxy_df.empty:
            profit_proxy_df = sizer_results.get("profit_proxy_table_")
    baseline_rets = sizer_results.get("opt_rets_")
    baseline_ts = sizer_results.get("opt_ts_")

    if ridge_scores is None:
        logger.error("No best_simple_score_ found in sizer_results")
        return {"error": "No sizer scores available"}

    n_samples = len(ridge_scores)

    # 2. Get optimal threshold and wallet settings from profit_proxy_table_
    if profit_proxy_df is not None and not profit_proxy_df.empty:
        # Find optimal row (marked by is_optimal or highest wallet_pnl)
        if "is_optimal" in profit_proxy_df.columns:
            opt_row = profit_proxy_df[profit_proxy_df["is_optimal"]].iloc[0]
        else:
            opt_idx = profit_proxy_df["wallet_pnl"].idxmax()
            opt_row = profit_proxy_df.loc[opt_idx]

        opt_frac = opt_row["selection_frac"]
        wallet_range = (
            opt_row.get("wallet_min", 0.05),
            opt_row.get("wallet_max", 0.15),
        )
        sizing_mode = opt_row.get("sizing_mode", "linear")
    else:
        # Defaults if no profit proxy table
        opt_frac = 0.2
        wallet_range = (0.05, 0.15)
        sizing_mode = "linear"

    # 3. Determine threshold and filter trades
    k_opt = max(1, int(n_samples * opt_frac))
    above_thresh_idx = np.argpartition(ridge_scores, -k_opt)[-k_opt:]

    # Sort threshold idx by score for proper confidence ranking (low to high)
    above_thresh_idx = above_thresh_idx[np.argsort(ridge_scores[above_thresh_idx])]

    # Normalize confidence scores to [0, 1] using sigmoid (for ALL samples first)
    confidence_scores = 1.0 / (1.0 + np.exp(-ridge_scores))

    _oof_p_tp = sizer_results.get("oof_p_tp_")
    if _oof_p_tp is not None and len(_oof_p_tp) == n_samples:
        p_tp_full = np.asarray(_oof_p_tp, dtype=np.float32)
        blended = (
            barrier_conf_alpha * confidence_scores
            + (1.0 - barrier_conf_alpha) * p_tp_full
        )
        confidence_scores = np.clip(blended, 0.0, 1.0).astype(np.float32)

    # Then extract for thresholded trades
    thresh_confidence = confidence_scores[above_thresh_idx]

    # Get returns for thresholded trades
    # Need to extract returns aligned with the sizer output
    if (
        baseline_rets is not None
        and len(baseline_rets) > 0
        and len(baseline_rets) == k_opt
    ):
        # Use the baseline returns directly - they're already filtered
        thresh_returns = baseline_rets
        thresh_ts = (
            baseline_ts if baseline_ts is not None else np.arange(len(baseline_rets))
        )
    else:
        # Extract from trade_outcomes using the above_thresh_idx
        if "net_return" in trade_outcomes.columns:
            full_returns = trade_outcomes["net_return"].values
        elif "return" in trade_outcomes.columns:
            full_returns = trade_outcomes["return"].values
        else:
            full_returns = np.zeros(n_samples)

        thresh_returns = full_returns[above_thresh_idx]

        if "timestamp" in trade_outcomes.columns:
            full_ts = pd.to_datetime(trade_outcomes["timestamp"]).values
            thresh_ts = full_ts[above_thresh_idx]
        else:
            thresh_ts = np.arange(len(above_thresh_idx))

    # 4. Baseline metrics (market orders - 100% fill rate)
    baseline_metrics = evaluate_offset_strategy(
        returns=thresh_returns,
        timestamps=thresh_ts,
        confidence_scores=thresh_confidence,
        offset_atr_pct=np.zeros(len(thresh_returns)),  # No offset
        executed=np.ones(len(thresh_returns), dtype=bool),  # All executed
        fill_bars=np.zeros(len(thresh_returns)),
        wallet_range=wallet_range,
        sizing_mode=sizing_mode,
        cost_pct=cost_pct,
        baseline_fill_rate=1.0,
    )

    # 5. Compute offset based on confidence (ATR% or raw return)
    if use_raw_return_offset:
        offset_raw = compute_offset_raw_return_from_confidence(
            confidence_scores=thresh_confidence,
            expected_returns=thresh_returns,
            base_offset_ret=base_offset_ret,
            max_offset_ret=max_offset_ret,
            confidence_threshold=0.0,
            scaling=offset_scaling,
            invert=invert_offset,
        )
        # Convert to ATR% for simulation (estimate ATR as 0.1% of entry)
        entry_prices = (
            trade_outcomes["entry_price"].values[above_thresh_idx]
            if "entry_price" in trade_outcomes.columns
            else np.ones(len(thresh_returns))
        )
        atr_estimate = entry_prices * 0.001  # 0.1% of price as ATR estimate
        offset_atr_pct = offset_raw / atr_estimate
    else:
        # Default ATR% mode
        base_offset = base_offset_atr_pct if base_offset_atr_pct is not None else 0.001
        max_offset = max_offset_atr_pct if max_offset_atr_pct is not None else 0.003
        offset_atr_pct = compute_offset_atr_pct_from_confidence(
            confidence_scores=thresh_confidence,
            base_offset_atr_pct=base_offset,
            max_offset_atr_pct=max_offset,
            confidence_threshold=0.0,
            scaling=offset_scaling,
        )
        offset_raw = None  # Will compute later from fill results

    # 6. Simulate execution with offset
    entry_improvements = np.zeros(
        len(thresh_returns)
    )  # Price improvement from better entry

    if use_atr_values is None and future_prices is not None and "atr" in future_prices:
        use_atr_values = future_prices["atr"][above_thresh_idx]

    if future_prices is not None and all(
        k in future_prices for k in ["opens", "highs", "lows"]
    ):
        # Get entry prices and directions
        entry_prices = (
            trade_outcomes["entry_price"].values[above_thresh_idx]
            if "entry_price" in trade_outcomes.columns
            else np.ones(len(thresh_returns))
        )
        if "is_long" not in trade_outcomes.columns:
            raise ValueError(
                "Missing required 'is_long' column in trade_outcomes. "
                "Direction is critical for correct limit order placement."
            )
        is_longs = trade_outcomes["is_long"].values[above_thresh_idx]

        # Use ATR values if provided, else estimate from entry price
        if use_atr_values is not None:
            atr_vals = (
                use_atr_values[above_thresh_idx]
                if len(use_atr_values) == n_samples
                else use_atr_values
            )
        else:
            atr_vals = entry_prices * 0.001  # Estimate 0.1% as default ATR

        (
            executed,
            fill_prices,
            fill_bars,
            entry_improvements,
        ) = _simulate_offset_execution_atr(
            entry_prices=entry_prices,
            is_longs=is_longs,
            atr_values=atr_vals,
            future_opens=future_prices["opens"][above_thresh_idx],
            future_highs=future_prices["highs"][above_thresh_idx],
            future_lows=future_prices["lows"][above_thresh_idx],
            offset_atr_pct=offset_atr_pct,
            max_wait_bars=max_wait_bars,
        )

        # If using raw return offset, use actual improvements, otherwise compute from ATR%
        if use_raw_return_offset and offset_raw is not None:
            # Use the actual entry improvements from simulation
            pass  # entry_improvements already set

        # Adjust returns for better entry price
        # Gains are LARGER, losses are SMALLER with offset
        adjusted_returns = adjust_returns_for_offset(
            original_returns=thresh_returns,
            is_longs=is_longs,
            entry_price_improvements=entry_improvements,
        )
    else:
        # Probabilistic fill model based on offset size
        if "is_long" not in trade_outcomes.columns:
            raise ValueError(
                "Missing required 'is_long' column in trade_outcomes. "
                "Direction is critical for correct limit order placement."
            )
        is_longs = trade_outcomes["is_long"].values[above_thresh_idx]

        if use_raw_return_offset and offset_raw is not None:
            fill_probs = np.exp(-offset_raw * 1000)  # Exponential decay with raw offset
        else:
            fill_probs = np.exp(-offset_atr_pct * 100)  # Exponential decay with ATR%
        fill_probs = np.clip(fill_probs, 0.3, 0.95)
        # Set seed for reproducibility when future_prices unavailable
        rng = np.random.default_rng(seed=42)
        executed = rng.random(len(thresh_returns)) < fill_probs
        fill_bars = np.where(
            executed, rng.integers(0, max_wait_bars, len(thresh_returns)), -1
        )

        # Estimate entry improvements for non-simulation case
        entry_improvements = np.where(
            executed, offset_atr_pct * 0.001, 0.0
        )  # Rough estimate
        adjusted_returns = adjust_returns_for_offset(
            original_returns=thresh_returns,
            is_longs=is_longs,
            entry_price_improvements=entry_improvements,
        )

    # 7. Offset strategy metrics (using adjusted returns for better entry)
    offset_metrics = evaluate_offset_strategy(
        returns=adjusted_returns,  # Use adjusted returns!
        timestamps=thresh_ts,
        confidence_scores=thresh_confidence,
        offset_atr_pct=offset_atr_pct,
        executed=executed,
        fill_bars=fill_bars,
        wallet_range=wallet_range,
        sizing_mode=sizing_mode,
        cost_pct=cost_pct,
        baseline_fill_rate=1.0,
    )

    # 8. Generate comparison
    comparison_df, summary = generate_offset_comparison(
        baseline_metrics, offset_metrics, strategy_name="ridge_atr_offset"
    )

    # 9. Additional diagnostics
    diagnostics = {
        "n_total_trades": n_samples,
        "n_above_threshold": len(above_thresh_idx),
        "threshold_frac": opt_frac,
        "wallet_range": wallet_range,
        "sizing_mode": sizing_mode,
        "use_raw_return_offset": use_raw_return_offset,
        "avg_offset_atr_pct": np.mean(offset_atr_pct),
        "max_offset_atr_pct": np.max(offset_atr_pct),
        "avg_entry_improvement": np.mean(entry_improvements),
        "total_entry_improvement": np.sum(entry_improvements),
        "adjusted_return_boost": np.sum(adjusted_returns) - np.sum(thresh_returns),
        "offset_distribution_atr_pct": pd.Series(offset_atr_pct).describe().to_dict(),
        "entry_improvement_distribution": pd.Series(entry_improvements)
        .describe()
        .to_dict(),
        "fill_rate_by_offset": pd.DataFrame(
            {
                "offset_atr_pct": offset_atr_pct,
                "executed": executed,
            }
        )
        .groupby("offset_atr_pct")["executed"]
        .mean()
        .to_dict()
        if len(offset_atr_pct) > 0
        else {},
    }

    return {
        "baseline_metrics": baseline_metrics,
        "offset_metrics": offset_metrics,
        "comparison_table": comparison_df,
        "summary": summary,
        "profit_proxy_table": profit_proxy_df,
        "diagnostics": diagnostics,
        "above_threshold_idx": above_thresh_idx,
        "offset_atr_pct": offset_atr_pct,
        "confidence_scores": thresh_confidence,
        "executed": executed,
        "fill_bars": fill_bars if "fill_bars" in locals() else None,
    }


def build_policy_path_state_bundle(
    trade_outcomes: pd.DataFrame,
    selected_idx: Optional[np.ndarray] = None,
    k_recent: int = 3,
) -> Dict[str, np.ndarray]:
    """Materialize vectorized path-state arrays used by policy optimisation and OOS replay."""
    n_total = len(trade_outcomes)
    if selected_idx is None:
        selected_idx = np.arange(n_total, dtype=np.int64)
    idx = np.asarray(selected_idx, dtype=np.int64)

    def _col(name: str, default: float = 0.0) -> np.ndarray:
        if name in trade_outcomes.columns:
            return np.asarray(trade_outcomes[name].values, dtype=np.float32)[idx]
        return np.full(len(idx), default, dtype=np.float32)

    returns = _col("return", 0.0)
    mfe = np.maximum(_col("mfe_ret", np.abs(returns)), 0.0)
    mae = np.maximum(_col("mae_ret", np.abs(np.minimum(returns, 0.0))), 0.0)
    duration = np.maximum(_col("duration", 4.0), 1.0)

    atr_raw = _col("atr_12_15m", 0.0)
    sl_atr_mult = _col("label_policy_sl_atr_mult", 0.0)
    has_label_policy = np.any(atr_raw > 1e-6) and np.any(sl_atr_mult > 1e-6)
    has_atr = np.any(atr_raw > 1e-6)
    if has_label_policy:
        barrier_pct = np.clip(atr_raw * sl_atr_mult, 0.001, 0.2).astype(np.float32)
    elif has_atr:
        barrier_pct = np.clip(atr_raw * 2.0, 0.001, 0.2).astype(np.float32)
    else:
        barrier_pct = np.clip(np.maximum(mae * 2.5, 1e-4), 0.005, 0.2).astype(np.float32)

    k = max(1, int(k_recent))
    ae_vel = mae / np.maximum(duration, float(k))
    delta_mfe = mfe / np.maximum(duration, float(k))
    delta_mae = mae / np.maximum(duration, float(k))
    pressure = delta_mae / (delta_mfe + 1e-6)
    path_quality = delta_mfe - delta_mae
    progress_per_bar = np.maximum(returns, 0.0) / np.maximum(
        duration * barrier_pct, 1e-6
    )

    asym_raw = np.log((mfe + 1e-6) / (mae + 1e-6)).astype(np.float32)
    trend = _col("trend_strength_percentile", 0.0)
    choppiness = _col("choppiness", 0.0)
    confidence = _col("oof_u_hat", 0.0)
    p_tp = _col("oof_p_tp", 0.33)
    p_sl = _col("oof_p_sl", 0.33)
    p_time = _col("oof_p_time", 0.34)

    timestamps = np.asarray(
        trade_outcomes.get("timestamp", pd.Series(np.arange(n_total))).values
    )[idx]

    tp_sl_ratio = _col("label_policy_tp_sl_ratio", 2.5)

    return {
        "returns": returns.astype(np.float32),
        "mfe_ret": mfe.astype(np.float32),
        "mae_ret": mae.astype(np.float32),
        "delta_mfe": delta_mfe.astype(np.float32),
        "delta_mae": delta_mae.astype(np.float32),
        "bars_since_entry": duration.astype(np.int32),
        "barrier_pct": barrier_pct,
        "label_tp_sl_ratio": tp_sl_ratio,
        "has_label_policy": np.float32(1.0 if has_label_policy else 0.0),
        "AE_vel": ae_vel.astype(np.float32),
        "pressure": pressure.astype(np.float32),
        "path_quality": path_quality.astype(np.float32),
        "progress_per_bar": progress_per_bar.astype(np.float32),
        "asym_raw": asym_raw,
        "trend": trend.astype(np.float32),
        "choppiness": choppiness.astype(np.float32),
        "confidence": confidence.astype(np.float32),
        "p_tp": p_tp.astype(np.float32),
        "p_sl": p_sl.astype(np.float32),
        "p_time": p_time.astype(np.float32),
        "timestamps": timestamps,
    }


# Keep old function for backward compatibility
run_simple_offset_generator = run_simple_offset_generator_from_sizer


def run_offset_sweep_analysis_from_sizer(
    sizer_results: Dict[str, Any],
    trade_outcomes: pd.DataFrame,
    future_prices: Optional[Dict[str, np.ndarray]] = None,
    base_offset_atr_pcts: List[float] = [0.0005, 0.001, 0.0015],  # 0.05%, 0.1%, 0.15%
    max_offset_atr_pcts: List[float] = [0.002, 0.003, 0.005],  # 0.2%, 0.3%, 0.5%
    offset_scalings: List[str] = ["linear", "convex", "concave"],
    use_atr_values: Optional[np.ndarray] = None,
    **kwargs,
) -> pd.DataFrame:
    """
    Run comprehensive sweep over ATR% offset parameters.

    Uses sizer_results directly (no recomputation). Sweeps:
    - Base offset ATR% (minimum offset)
    - Max offset ATR% (maximum offset at high confidence)
    - Offset scaling curve (linear/convex/concave)

    Returns DataFrame with all combinations and their metrics.
    """
    results = []

    for base_atr in base_offset_atr_pcts:
        for max_atr in max_offset_atr_pcts:
            if max_atr <= base_atr:
                continue  # Skip invalid ranges
            for scaling in offset_scalings:
                try:
                    result = run_simple_offset_generator_from_sizer(
                        sizer_results=sizer_results,
                        trade_outcomes=trade_outcomes,
                        future_prices=future_prices,
                        base_offset_atr_pct=base_atr,
                        max_offset_atr_pct=max_atr,
                        offset_scaling=scaling,
                        use_atr_values=use_atr_values,
                        **kwargs,
                    )

                    row = {
                        "base_offset_atr_pct": base_atr,
                        "max_offset_atr_pct": max_atr,
                        "offset_scaling": scaling,
                        "baseline_pnl": result["baseline_metrics"]["net_pnl"],
                        "offset_pnl": result["offset_metrics"]["net_pnl"],
                        "pnl_improvement": result["offset_metrics"]["net_pnl"]
                        - result["baseline_metrics"]["net_pnl"],
                        "pnl_improvement_pct": (
                            result["offset_metrics"]["net_pnl"]
                            - result["baseline_metrics"]["net_pnl"]
                        )
                        / (abs(result["baseline_metrics"]["net_pnl"]) + 1e-9)
                        * 100,
                        "baseline_sortino": result["baseline_metrics"]["sortino"],
                        "offset_sortino": result["offset_metrics"]["sortino"],
                        "sortino_delta": result["offset_metrics"]["sortino"]
                        - result["baseline_metrics"]["sortino"],
                        "fill_rate": result["offset_metrics"]["fill_rate"],
                        "fill_rate_delta_pct": result["offset_metrics"][
                            "fill_rate_delta_pct"
                        ],
                        "n_executed": result["offset_metrics"]["n_executed"],
                        "n_missed": result["offset_metrics"]["n_missed"],
                        "opportunity_cost": result["offset_metrics"][
                            "opportunity_cost"
                        ],
                        "avg_offset_atr_pct": result["diagnostics"][
                            "avg_offset_atr_pct"
                        ],
                        "threshold_frac": result["diagnostics"]["threshold_frac"],
                    }
                    results.append(row)

                    logger.info(
                        f"Sweep: base={base_atr:.4f}, max={max_atr:.4f}, scale={scaling} | "
                        f"FillRate={row['fill_rate']:.2%}, FillDelta={row['fill_rate_delta_pct']:.1f}pp, "
                        f"PnLΔ={row['pnl_improvement']:.4f}"
                    )

                except Exception as e:
                    logger.warning(
                        f"Failed for base={base_atr}, max={max_atr}, scale={scaling}: {e}"
                    )

    df = pd.DataFrame(results)

    # Add ranking by PnL improvement adjusted for fill rate loss
    if not df.empty:
        # Score = PnL improvement - penalty for fill rate loss
        df["adjusted_score"] = (
            df["pnl_improvement"] - 0.01 * df["fill_rate_delta_pct"].abs()
        )
        df = df.sort_values("adjusted_score", ascending=False).reset_index(drop=True)

    return df


# Backward compatibility alias
run_offset_sweep_analysis = run_offset_sweep_analysis_from_sizer


if __name__ == "__main__":
    # Example usage - generates mock data for demonstration
    logging.basicConfig(level=logging.INFO)

    logger.info("Running simple_offset_generator.py with synthetic data...")

    # Generate synthetic test data
    np.random.seed(42)
    n_samples = 1000

    # Synthetic features (Ridge head predictions) - use prefixes that detect_meta_head_keys expects
    feature_dict = {
        "base_h_0": np.random.randn(n_samples) * 0.5 + 0.1,
        "base_h_1": np.random.randn(n_samples) * 0.5 + 0.05,
        "base_h_2": np.random.randn(n_samples) * 0.5,
    }

    # Synthetic returns (some predictive signal)
    signal = 0.3 * feature_dict["base_h_0"] + 0.2 * feature_dict["base_h_1"]
    y_raw_net_return = signal + np.random.randn(n_samples) * 0.02

    # Synthetic trade outcomes
    trade_outcomes = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=n_samples, freq="15min"),
            "entry_price": np.ones(n_samples) * 100.0,
            "is_long": np.random.choice([True, False], n_samples),
            "net_return": y_raw_net_return,
        }
    )

    timestamps = trade_outcomes["timestamp"].values

    # Step 1: Run simple position sizer to get thresholds and wallet settings
    logger.info("Step 1: Running simple position sizer...")

    from extreme_price_movements.simple_position_sizer import run_simple_position_sizer

    sizer_results = run_simple_position_sizer(
        feature_dict=feature_dict,
        trade_outcomes=trade_outcomes,
        y_raw_net_return=y_raw_net_return,
        y_downside=np.abs(np.minimum(y_raw_net_return, 0)),
        timestamps=timestamps,
    )

    # Check sizer results
    if "error" in sizer_results:
        logger.error(f"Sizer failed: {sizer_results['error']}")
    else:
        logger.info(
            f"Sizer completed. Found {len(sizer_results.get('ridge_profit_proxy_table_', []))} profit proxy configurations"
        )

    # Step 2a: Test ATR% offset mode
    logger.info("\nStep 2a: Testing ATR% based offsets...")

    offset_results_atr = run_simple_offset_generator_from_sizer(
        sizer_results=sizer_results,
        trade_outcomes=trade_outcomes,
        base_offset_atr_pct=0.001,  # 0.1%
        max_offset_atr_pct=0.003,  # 0.3%
        offset_scaling="linear",
    )

    print("\n=== ATR% Mode: Comparison Table (Baseline vs Offset) ===")
    print(offset_results_atr["comparison_table"].to_string())

    print("\n=== ATR% Mode: Diagnostics ===")
    for k, v in offset_results_atr["diagnostics"].items():
        if k not in [
            "offset_distribution_atr_pct",
            "fill_rate_by_offset",
            "entry_improvement_distribution",
        ]:
            print(f"  {k}: {v}")

    # Step 2b: Test RAW RETURN offset mode
    logger.info("\nStep 2b: Testing RAW RETURN based offsets...")

    offset_results_raw = run_simple_offset_generator_from_sizer(
        sizer_results=sizer_results,
        trade_outcomes=trade_outcomes,
        use_raw_return_offset=True,  # Enable raw return mode
        base_offset_ret=0.0001,  # 1 bps min
        max_offset_ret=0.0005,  # 5 bps max
        offset_scaling="linear",
    )

    print("\n=== RAW RETURN Mode: Comparison Table (Baseline vs Offset) ===")
    print(offset_results_raw["comparison_table"].to_string())

    print("\n=== RAW RETURN Mode: Key Metrics ===")
    print(
        f"  Adjusted Return Boost: {offset_results_raw['diagnostics'].get('adjusted_return_boost', 0):.6f}"
    )
    print(
        f"  Avg Entry Improvement: {offset_results_raw['diagnostics'].get('avg_entry_improvement', 0):.6f}"
    )

    # Step 3: Run sweep analysis on ATR%
    logger.info("\nStep 3: Running ATR% sweep analysis...")

    sweep_df = run_offset_sweep_analysis_from_sizer(
        sizer_results=sizer_results,
        trade_outcomes=trade_outcomes,
        base_offset_atr_pcts=[0.0005, 0.001, 0.0015],
        max_offset_atr_pcts=[0.002, 0.003],
    )

    print("\n=== Top 5 ATR% Sweep Results (by adjusted score) ===")
    print(sweep_df.head().to_string())

    logger.info("\n=== Done ===")
