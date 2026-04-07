"""Limit Order Price Estimation using MAE/MFE predictions.

This module provides functions to estimate optimal limit order prices
based on meta model head predictions (MAE, MFE, Utility).

Key Features:
- MAE-based offset estimation (conservative fill for volatile markets)
- MFE-based offset adjustment (aggressive fill for trending markets)
- Fill probability modeling
- Support for both entry and exit limit orders
- Proper high/low price handling for long/short positions
"""

from __future__ import annotations

import numpy as np
from typing import Dict, Optional, Tuple, Union


def clip_offset_to_bounds(offset: float, cfg: Optional[Dict] = None) -> float:
    """Clips an offset based on the semantic contract bounds."""
    cfg = cfg or {}
    min_offset = cfg.get("limit_offset_min", cfg.get("limit_offset_min_bps", 5.0))
    max_offset = cfg.get("limit_offset_max", cfg.get("limit_offset_max_bps", 50.0))
    return float(np.clip(offset, min_offset, max_offset))

def predict_offset(
    mae_hat: float,
    mfe_hat: float,
    u_hat: float,
    confidence: float = 0.5,
    fee_market: float = 0.0025,
    fee_limit: float = 0.0010,
    cfg: Optional[Dict] = None,
) -> float:
    """Predict execution offset. Validates contract and routes to heuristic logic."""
    cfg = cfg or {}
    mode = cfg.get("limit_offset_mode", "heuristic")

    if mode == "disabled":
        return 0.0

    if mode == "ml":
        # The ML path is disabled by default via config, but if reached here unexpectedly
        # (e.g. bypass), fallback to heuristic but warn.
        import logging
        logging.getLogger(__name__).warning(
            "ML offset mode accessed via direct heuristic call. "
            "Falling back to heuristic estimation temporarily."
        )

    return estimate_entry_limit_offset(
        mae_hat=mae_hat,
        mfe_hat=mfe_hat,
        u_hat=u_hat,
        confidence=confidence,
        fee_market=fee_market,
        fee_limit=fee_limit,
        cfg=cfg,
    )

def estimate_entry_limit_offset(
    mae_hat: float,
    mfe_hat: float,
    u_hat: float,
    confidence: float = 0.5,
    fee_market: float = 0.0025,
    fee_limit: float = 0.0010,
    cfg: Optional[Dict] = None,
) -> float:
    """Estimate optimal limit order offset in bps for entry.
    
    Args:
        mae_hat: Predicted MAE (adverse excursion) as fraction (e.g., 0.01 = 1%)
        mfe_hat: Predicted MFE (favorable excursion) as fraction (e.g., 0.02 = 2%)
        u_hat: Predicted utility score (typically -1 to 1)
        confidence: Prediction confidence [0, 1], higher = more confident
        fee_market: Market order fee (fraction)
        fee_limit: Limit order fee (fraction)
        cfg: Optional configuration dict with override values
    
    Returns:
        Optimal limit offset in bps
    
    Strategy:
        - Higher MAE → wider offset (more conservative fill)
        - Higher MFE → tighter offset (more likely to capture move)
        - Higher confidence → tighter offset (trust prediction more)
    """
    cfg = cfg or {}
    
    min_offset = cfg.get("limit_offset_min", cfg.get("limit_offset_min_bps", 5.0))
    max_offset = cfg.get("limit_offset_max", cfg.get("limit_offset_max_bps", 50.0))
    
    # Convert fees to bps
    fee_market_bps = fee_market * 10000
    fee_limit_bps = fee_limit * 10000
    fee_savings_bps = fee_market_bps - fee_limit_bps  # 15 bps typically
    
    # Convert predictions to fractions if in bps
    mae_hat = _normalize_to_fraction(mae_hat)
    mfe_hat = _normalize_to_fraction(mfe_hat)
    
    # Base offset from MAE prediction (scaled down to be reasonable)
    # MAE of 2% = 200 bps, use 10% of that = 20 bps base
    mae_offset = mae_hat * 1000  # Scale MAE to bps range
    
    # MFE bonus: reduce offset if high favorable excursion expected
    # Higher MFE = more likely to move in our favor = can be tighter
    mfe_bonus = mfe_hat * 300  # Scale MFE to reduce offset
    
    # Utility adjustment: high positive utility → tighter offset
    u_bonus = u_hat * 10 if u_hat > 0 else 0
    
    # Confidence adjustment: higher confidence = tighter offset
    confidence_penalty = (1.0 - confidence) * 15
    
    # Calculate raw offset
    offset_bps = mae_offset - mfe_bonus + u_bonus - confidence_penalty
    
    # Apply fee savings threshold
    # If fee savings > expected adverse move, use minimum offset
    expected_adverse = mae_hat * 500  # 50% of MAE in bps
    if fee_savings_bps > expected_adverse:
        # Fee savings exceed expected adverse move, use minimum
        offset_bps = min_offset
    
    # Clamp to bounds
    return clip_offset_to_bounds(offset_bps, cfg)


def estimate_exit_limit_offset(
    mfe_hat: float,
    duration_hat: float,
    profit_locked: float,
    mae_hat: float = 0.0,
    cfg: Optional[Dict] = None,
) -> float:
    """Estimate optimal limit order offset for exit in bps.
    
    Args:
        mfe_hat: Predicted MFE (favorable excursion) as fraction
        duration_hat: Predicted duration (in bars)
        profit_locked: Current unrealized profit as fraction (e.g., 0.02 = 2%)
        mae_hat: Predicted MAE as fraction (for risk adjustment)
        cfg: Optional configuration dict
    
    Returns:
        Optimal exit limit offset in bps
    """
    cfg = cfg or {}
    
    min_offset = cfg.get("limit_offset_min", cfg.get("limit_offset_min_bps", 5.0))
    max_offset = cfg.get("limit_offset_max", cfg.get("limit_offset_max_bps", 50.0))
    
    # Convert to fractions
    mfe_hat = _normalize_to_fraction(mfe_hat)
    profit_locked = _normalize_to_fraction(profit_locked)
    
    # Base offset
    base_offset = 10.0  # 10 bps default
    
    # Reduce offset as profit increases (capture more profit)
    if profit_locked > 0.01:  # > 1% profit
        profit_reduction = min(profit_locked * 500, 10.0)  # Up to 10 bps reduction
        offset_bps = base_offset - profit_reduction
    else:
        # In loss or small profit: wider offset to avoid premature exit
        # Use duration as proxy for confidence
        confidence = 1.0 / (1.0 + duration_hat * 0.1)
        offset_bps = base_offset + (1 - confidence) * 10  # Up to +10 bps
    
    # MAE adjustment: if high MAE expected, be more conservative (wider offset)
    if mae_hat > 0:
        mae_adjustment = mae_hat * 200  # Scale MAE to bps
        offset_bps += mae_adjustment
    
    # MFE adjustment: if high MFE expected, tighter offset
    mfe_adjustment = -mfe_hat * 200 if mfe_hat > 0 else 0
    offset_bps += mfe_adjustment
    
    return clip_offset_to_bounds(offset_bps, cfg)


def estimate_fill_probability(
    offset_bps: float,
    mae_hat: float,
    vol_regime: float = 0.5,
    liquidity: float = 0.5,
    cfg: Optional[Dict] = None,
) -> float:
    """Estimate probability of limit order filling.
    
    Args:
        offset_bps: Limit offset in bps
        mae_hat: Predicted MAE (adverse excursion) as fraction
        vol_regime: Current volatility regime [0=low, 1=high]
        liquidity: Liquidity score [0=illiquid, 1=liquid]
        cfg: Optional configuration dict
    
    Returns:
        Estimated fill probability [0, 1]
    """
    cfg = cfg or {}
    
    vol_weight = cfg.get("limit_fill_vol_regime_weight", 0.3)
    liq_bonus = cfg.get("limit_fill_liquidity_bonus", 0.2)
    
    mae_hat = _normalize_to_fraction(mae_hat)
    
    # Offset ratio: how offset compares to expected adverse move
    mae_bps = mae_hat * 10000
    offset_ratio = offset_bps / max(mae_bps, 1e-6)
    
    # Base fill probability from offset ratio
    # Higher offset = higher fill probability
    base_prob = 1.0 - np.exp(-offset_ratio * 0.5)
    
    # Volatility penalty: high vol = lower fill prob
    vol_penalty = 1.0 - vol_regime * vol_weight
    
    # Liquidity bonus: higher liquidity = higher fill prob
    liq_factor = 0.8 + liquidity * liq_bonus
    
    fill_prob = base_prob * vol_penalty * liq_factor
    
    return float(np.clip(fill_prob, 0.0, 1.0))


def compute_limit_order_ev(
    fill_prob: float,
    fee_savings: float,
    expected_fill_pnl: float,
    expected_missed_pnl: float = 0.0,
    fill_rate_penalty: float = 0.0,
) -> float:
    """Compute expected value of using limit order vs market order.
    
    Args:
        fill_prob: Probability of order filling
        fee_savings: Fee saved by using limit instead of market (fraction)
        expected_fill_pnl: Expected PnL if filled at limit price
        expected_missed_pnl: Expected PnL if not filled (missed move)
        fill_rate_penalty: Cost of re-entering later (fraction)
    
    Returns:
        Expected value difference (limit vs market)
    """
    ev = (
        fill_prob * (expected_fill_pnl + fee_savings) +
        (1 - fill_prob) * (expected_missed_pnl - fill_rate_penalty)
    )
    return float(ev)


def get_limit_price_for_order(
    signal_price: float,
    offset_bps: float,
    is_long: bool,
    high_price: Optional[float] = None,
    low_price: Optional[float] = None,
) -> float:
    """Calculate limit order price using proper high/low for long/short.
    
    For LONG positions:
        - We want to buy below signal price
        - Check if low price is below our limit price (fill opportunity)
        - If high_price < limit_price, we get filled
    
    For SHORT positions:
        - We want to sell above signal price
        - Check if high price is above our limit price (fill opportunity)
        - If low_price > limit_price, we get filled
    
    Args:
        signal_price: The signal/reference price
        offset_bps: Limit offset in bps (positive = better price for us)
        is_long: True for long position, False for short
        high_price: Current bar high (for short fill check)
        low_price: Current bar low (for long fill check)
    
    Returns:
        The limit price we'd like to get filled at
    """
    offset_pct = offset_bps / 10000.0
    
    if is_long:
        # Long: buy below signal = signal * (1 - offset)
        limit_price = signal_price * (1.0 - offset_pct)
    else:
        # Short: sell above signal = signal * (1 + offset)
        limit_price = signal_price * (1.0 + offset_pct)
    
    return float(limit_price)


def check_limit_order_fill(
    limit_price: float,
    is_long: bool,
    high_price: float,
    low_price: float,
    open_price: float,
) -> Tuple[bool, float]:
    """Check if limit order would fill at current bar prices.
    
    Args:
        limit_price: Our limit order price
        is_long: True for long, False for short
        high_price: Bar high price
        low_price: Bar low price  
        open_price: Bar open price (for gap analysis)
    
    Returns:
        Tuple of (did_fill, fill_price)
            - did_fill: True if order filled
            - fill_price: Price at which fill occurred
    """
    if is_long:
        # Long fills when price drops to or below limit price
        # Check: did low go below our limit?
        fill_price = min(low_price, open_price)  # Use worse of open/low
        did_fill = low_price <= limit_price
    else:
        # Short fills when price rises to or above limit price
        # Check: did high go above our limit?
        fill_price = max(high_price, open_price)  # Use worse of open/high
        did_fill = high_price >= limit_price
    
    return bool(did_fill), float(fill_price)


def simulate_trade_with_limit_order(
    signal_price: float,
    offset_bps_entry: float,
    offset_bps_exit: float,
    is_long: bool,
    high_prices: np.ndarray,
    low_prices: np.ndarray,
    close_prices: np.ndarray,
    tp_mult: float = 2.0,
    sl_mult: float = 1.0,
    atr: float = 0.02,
    fee_entry: float = 0.001,
    fee_exit: float = 0.001,
    max_bars: int = 24,
) -> Dict:
    """Simulate a trade with limit order entry and exit.
    
    This function simulates the full trade lifecycle with limit orders:
    1. Place entry limit order at offset from signal price
    2. Check if/when entry fills using high/low prices
    3. Place exit limit order at offset from entry (trailing or fixed)
    4. Check if/when exit fills
    
    Args:
        signal_price: Price when signal triggered
        offset_bps_entry: Entry limit offset in bps
        offset_bps_exit: Exit limit offset in bps
        is_long: True for long position
        high_prices: Array of high prices after signal
        low_prices: Array of low prices after signal
        close_prices: Array of close prices
        tp_mult: TP distance as multiple of ATR
        sl_mult: SL distance as multiple of ATR
        atr: ATR value (as fraction of price)
        fee_entry: Entry fee (fraction)
        fee_exit: Exit fee (fraction)
        max_bars: Maximum holding periods in bars
    
    Returns:
        Dict with trade results including fill info, PnL, etc.
    """
    # Calculate entry limit price
    entry_limit_price = get_limit_price_for_order(
        signal_price, offset_bps_entry, is_long, 
        high_price=high_prices[0] if len(high_prices) > 0 else signal_price,
        low_price=low_prices[0] if len(low_prices) > 0 else signal_price,
    )
    
    # Calculate TP/SL prices (relative to signal for consistency)
    # Note: In practice, these could be relative to fill price (Scenario A)
    # or relative to signal price (Scenario B)
    if is_long:
        tp_price = signal_price * (1 + tp_mult * atr)
        sl_price = signal_price * (1 - sl_mult * atr)
    else:
        tp_price = signal_price * (1 - tp_mult * atr)
        sl_price = signal_price * (1 + sl_mult * atr)
    
    # Simulate entry fill
    entry_fill_price = None
    entry_bar = None
    for i in range(min(max_bars, len(high_prices))):
        did_fill, fill_price = check_limit_order_fill(
            entry_limit_price, is_long, 
            high_prices[i], low_prices[i], 
            close_prices[i] if i < len(close_prices) else signal_price
        )
        if did_fill:
            entry_fill_price = fill_price
            entry_bar = i
            break
    
    # If no fill, return missed trade
    if entry_fill_price is None:
        return {
            "filled": False,
            "entry_fill_price": None,
            "exit_fill_price": None,
            "exit_bar": None,
            "exit_reason": "not_filled",
            "return": 0.0,
            "return_net": 0.0,
            "fee_total": 0.0,
        }
    
    # Recalculate TP/SL relative to fill price (Scenario A: new barriers)
    if is_long:
        tp_price_from_fill = entry_fill_price * (1 + tp_mult * atr)
        sl_price_from_fill = entry_fill_price * (1 - sl_mult * atr)
    else:
        tp_price_from_fill = entry_fill_price * (1 - tp_mult * atr)
        sl_price_from_fill = entry_fill_price * (1 + sl_mult * atr)
    
    # Calculate exit limit price (could be dynamic based on profit)
    # For simplicity, use fixed offset from entry
    exit_limit_price = get_limit_price_for_order(
        entry_fill_price, offset_bps_exit, not is_long,  # Opposite direction for exit
        high_price=high_prices[entry_bar] if entry_bar < len(high_prices) else entry_fill_price,
        low_price=low_prices[entry_bar] if entry_bar < len(low_prices) else entry_fill_price,
    )
    
    # Simulate exit (check TP, SL, then limit order)
    exit_fill_price = None
    exit_reason = "timeout"
    
    for i in range(entry_bar + 1, min(max_bars, len(high_prices))):
        # First check TP/SL
        if is_long:
            if high_prices[i] >= tp_price_from_fill:
                exit_fill_price = tp_price_from_fill
                exit_reason = "tp_hit"
                break
            if low_prices[i] <= sl_price_from_fill:
                exit_fill_price = sl_price_from_fill
                exit_reason = "sl_hit"
                break
        else:
            if low_prices[i] <= tp_price_from_fill:
                exit_fill_price = tp_price_from_fill
                exit_reason = "tp_hit"
                break
            if high_prices[i] >= sl_price_from_fill:
                exit_fill_price = sl_price_from_fill
                exit_reason = "sl_hit"
                break
        
        # Check exit limit order fill
        did_exit_fill, exit_p = check_limit_order_fill(
            exit_limit_price, not is_long,  # Opposite side
            high_prices[i], low_prices[i],
            close_prices[i] if i < len(close_prices) else entry_fill_price
        )
        if did_exit_fill:
            exit_fill_price = exit_p
            exit_reason = "limit_exit"
            break
    
    # Timeout case
    if exit_fill_price is None:
        exit_fill_price = close_prices[min(max_bars - 1, len(close_prices) - 1)]
        exit_bar = max_bars - 1
        exit_reason = "timeout"
    
    # Calculate returns
    if is_long:
        gross_return = (exit_fill_price / entry_fill_price) - 1.0
    else:
        gross_return = (entry_fill_price / exit_fill_price) - 1.0
    
    # Net of fees
    fee_total = fee_entry + fee_exit
    net_return = gross_return - fee_total
    
    return {
        "filled": True,
        "entry_fill_price": float(entry_fill_price),
        "exit_fill_price": float(exit_fill_price),
        "entry_bar": int(entry_bar) if entry_bar is not None else None,
        "exit_bar": int(i) if exit_fill_price is not None else None,
        "exit_reason": exit_reason,
        "gross_return": float(gross_return),
        "return": float(net_return),
        "fee_total": float(fee_total),
    }


def _normalize_to_fraction(value: float) -> float:
    """Normalize a value to fraction if it appears to be in bps or percent.
    
    Args:
        value: Input value
    
    Returns:
        Normalized fraction
    """
    # If value > 1, assume it's in bps (e.g., 25 = 0.0025)
    if value > 1.0:
        return value / 10000.0
    
    # If value > 0.1, assume it's in percent (e.g., 0.5 = 0.005)
    if value > 0.1:
        return value / 100.0
    
    # Already in fraction
    return float(value)


# ============================================================================
# PnL Comparison Metrics
# ============================================================================


def compute_pnl_comparison_metrics(
    trades: pd.DataFrame,
    entry_offset_bps: float = 20.0,
    exit_offset_bps: float = 20.0,
    mae_hat_col: str = "mae_hat",
    mfe_hat_col: str = "mfe_hat",
    is_long_col: str = "is_long",
    entry_price_col: str = "entry_price",
    exit_price_col: str = "exit_price",
    fee_limit_entry: float = 0.001,  # 10 bps
    fee_limit_exit: float = 0.001,
    fee_market_entry: float = 0.0025,  # 25 bps
    fee_market_exit: float = 0.0025,
) -> Dict:
    """Compute PnL comparison metrics for different execution strategies.
    
    This function compares:
    1. Solution A: Limit orders with new TP/SL (barriers recalculated from fill price)
    2. Solution B: Limit orders with same barrier distances (offset reduces effective RR)
    3. Market Orders: No limit offset, higher fees
    
    Args:
        trades: DataFrame with trade data
        entry_offset_bps: Entry limit offset in bps
        exit_offset_bps: Exit limit offset in bps
        mae_hat_col: Column name for predicted MAE
        mfe_hat_col: Column name for predicted MFE
        is_long_col: Column name for position direction
        entry_price_col: Column name for entry price
        exit_price_col: Column name for exit price
        fee_limit_entry: Fee for limit order entry
        fee_limit_exit: Fee for limit order exit
        fee_market_entry: Fee for market order entry
        fee_market_exit: Fee for market order exit
    
    Returns:
        Dict with comparison metrics
    """
    n = len(trades)
    if n == 0:
        return {"error": "No trades to compare"}
    
    is_long = trades[is_long_col].values
    entry_px = trades[entry_price_col].values
    exit_px = trades[exit_price_col].values
    
    # Get MAE/MFE predictions if available
    mae_hat = trades.get(mae_hat_col, pd.Series(np.zeros(n))).values
    mfe_hat = trades.get(mfe_hat_col, pd.Series(np.zeros(n))).values
    
    # Calculate baseline PnL (current implementation)
    gross_ret = np.where(
        is_long == 1,
        (exit_px / entry_px) - 1.0,
        (entry_px / exit_px) - 1.0
    )
    
    # Current fees (limit for entry, market for exit - old logic)
    baseline_fee = fee_limit_entry + fee_market_exit
    pnl_baseline = gross_ret - baseline_fee
    
    # === Solution A: Limit entry + Limit exit with new TP/SL ===
    # New barriers calculated from fill price
    # This assumes we get better fills on both entry and exit
    entry_offset = entry_offset_bps / 10000.0
    exit_offset = exit_offset_bps / 10000.0
    
    # Improved entry price (assume fills at limit)
    entry_px_a = np.where(
        is_long == 1,
        entry_px * (1.0 - entry_offset),
        entry_px * (1.0 + entry_offset)
    )
    
    # Improved exit price (assume fills at limit)  
    exit_px_a = np.where(
        is_long == 1,
        exit_px * (1.0 + exit_offset),
        exit_px * (1.0 - exit_offset)
    )
    
    gross_ret_a = np.where(
        is_long == 1,
        (exit_px_a / entry_px_a) - 1.0,
        (entry_px_a / exit_px_a) - 1.0
    )
    
    fee_a = fee_limit_entry + fee_limit_exit
    pnl_solution_a = gross_ret_a - fee_a
    
    # === Solution B: Limit entry with same barrier distance ===
    # Offset reduces effective R:R since barriers stay at same price level
    # Entry improves but TP/SL distances don't change
    entry_px_b = np.where(
        is_long == 1,
        entry_px * (1.0 - entry_offset),
        entry_px * (1.0 + entry_offset)
    )
    
    # Exit stays at same price (no improvement)
    exit_px_b = exit_px
    
    gross_ret_b = np.where(
        is_long == 1,
        (exit_px_b / entry_px_b) - 1.0,
        (entry_px_b / exit_px_b) - 1.0
    )
    
    fee_b = fee_limit_entry + fee_market_exit
    pnl_solution_b = gross_ret_b - fee_b
    
    # === Market Order Baseline ===
    # No offset, higher fees
    pnl_market = gross_ret - (fee_market_entry + fee_market_exit)
    
    # === Compute Metrics ===
    def _metrics(pnl):
        return {
            "mean": float(np.mean(pnl)),
            "std": float(np.std(pnl)),
            "median": float(np.median(pnl)),
            "win_rate": float(np.mean(pnl > 0)),
            "pnl_total": float(np.sum(pnl)),
            "sharpe": float(np.mean(pnl) / np.std(pnl)) * np.sqrt(252) if np.std(pnl) > 0 else 0.0,
        }
    
    metrics = {
        "n_trades": n,
        "baseline": _metrics(pnl_baseline),
        "solution_a": _metrics(pnl_solution_a),
        "solution_b": _metrics(pnl_solution_b),
        "market_order": _metrics(pnl_market),
        "diff_a_vs_baseline": float(np.mean(pnl_solution_a - pnl_baseline)),
        "diff_b_vs_baseline": float(np.mean(pnl_solution_b - pnl_baseline)),
        "diff_a_vs_market": float(np.mean(pnl_solution_a - pnl_market)),
        "diff_b_vs_market": float(np.mean(pnl_solution_b - pnl_market)),
        "fee_savings_entry": (fee_market_entry - fee_limit_entry) * n,
        "fee_savings_exit": (fee_market_exit - fee_limit_exit) * n,
    }
    
    return metrics


def compute_exit_limit_fill_impact(
    trades: pd.DataFrame,
    exit_filled_col: str = "exit_filled_via_limit",
    exit_price_col: str = "exit_price",
    entry_price_col: str = "entry_price",
    is_long_col: str = "is_long",
) -> Dict:
    """Compute the impact of exit limit order fills.
    
    Args:
        trades: DataFrame with trade data including exit_filled_via_limit
        exit_filled_col: Column indicating if exit was filled via limit
        exit_price_col: Column for exit price
        entry_price_col: Column for entry price
        is_long_col: Column for position direction
    
    Returns:
        Dict with exit limit impact metrics
    """
    if exit_filled_col not in trades.columns:
        return {"error": f"Column {exit_filled_col} not found"}
    
    n = len(trades)
    is_long = trades[is_long_col].values
    entry_px = trades[entry_price_col].values
    exit_px = trades[exit_price_col].values
    exit_filled = trades[exit_filled_col].values
    
    # Calculate PnL
    gross_ret = np.where(
        is_long == 1,
        (exit_px / entry_px) - 1.0,
        (entry_px / exit_px) - 1.0
    )
    
    filled_mask = exit_filled == True
    not_filled_mask = exit_filled == False
    
    metrics = {
        "total_trades": n,
        "exit_limit_filled": int(np.sum(filled_mask)),
        "exit_limit_not_filled": int(np.sum(not_filled_mask)),
        "fill_rate": float(np.mean(filled_mask)),
    }
    
    if np.sum(filled_mask) > 0:
        metrics["mean_pnl_filled"] = float(np.mean(gross_ret[filled_mask]))
    if np.sum(not_filled_mask) > 0:
        metrics["mean_pnl_not_filled"] = float(np.mean(gross_ret[not_filled_mask]))
    
    return metrics


# ============================================================================
# Configuration helpers
# ============================================================================


def get_fee_for_order_type(
    order_type: str = "market",
    side: str = "entry",
    cfg: Optional[Dict] = None,
) -> float:
    """Get the appropriate fee based on order type and side.
    
    Args:
        order_type: "market" or "limit"
        side: "entry" or "exit"
        cfg: Configuration dict
    
    Returns:
        Fee as fraction (e.g., 0.0025 = 0.25%)
    """
    cfg = cfg or {}
    
    if order_type == "market":
        if side == "entry":
            return cfg.get("fee_bps_market", 25.0) / 10000.0
        else:  # exit
            return cfg.get("fee_bps_market_exit", 25.0) / 10000.0
    else:  # limit
        if side == "entry":
            return cfg.get("fee_bps_limit_entry", 10.0) / 10000.0
        else:  # exit
            return cfg.get("fee_bps_limit_exit", 10.0) / 10000.0


def create_limit_order_config(
    mae_hat: float,
    mfe_hat: float,
    u_hat: float,
    confidence: float,
    vol_regime: float = 0.5,
    liquidity: float = 0.5,
    profit_locked: float = 0.0,
    duration_hat: float = 1.0,
    cfg: Optional[Dict] = None,
) -> Dict:
    """Create a complete limit order configuration from predictions.
    
    This is the main entry point for integrating MAE/MFE-based limit
    order pricing into the trading system.
    
    Args:
        mae_hat: Predicted MAE (adverse excursion)
        mfe_hat: Predicted MFE (favorable excursion)
        u_hat: Predicted utility
        confidence: Prediction confidence
        vol_regime: Current volatility regime
        liquidity: Current liquidity
        profit_locked: Current unrealized profit (for exits)
        duration_hat: Predicted time to TP
        cfg: Configuration dict
    
    Returns:
        Dict with limit order parameters:
            - entry_offset_bps: Entry limit offset
            - exit_offset_bps: Exit limit offset
            - fill_prob_entry: Entry fill probability
            - fill_prob_exit: Exit fill probability
            - fee_entry: Entry fee
            - fee_exit: Exit fee
    """
    cfg = cfg or {}
    
    # Get fees
    fee_market = get_fee_for_order_type("market", "entry", cfg)
    fee_limit_entry = get_fee_for_order_type("limit", "entry", cfg)
    fee_limit_exit = get_fee_for_order_type("limit", "exit", cfg)
    
    # Estimate entry offset
    entry_offset = predict_offset(
        mae_hat=mae_hat,
        mfe_hat=mfe_hat,
        u_hat=u_hat,
        confidence=confidence,
        fee_market=fee_market,
        fee_limit=fee_limit_entry,
        cfg=cfg,
    )
    
    # Estimate exit offset
    exit_offset = estimate_exit_limit_offset(
        mfe_hat=mfe_hat,
        duration_hat=duration_hat,
        profit_locked=profit_locked,
        mae_hat=mae_hat,
        cfg=cfg,
    )
    
    # Estimate fill probabilities
    fill_prob_entry = estimate_fill_probability(
        offset_bps=entry_offset,
        mae_hat=mae_hat,
        vol_regime=vol_regime,
        liquidity=liquidity,
        cfg=cfg,
    )
    
    # For exit, use smaller offset and higher expected fill
    fill_prob_exit = estimate_fill_probability(
        offset_bps=exit_offset,
        mae_hat=mae_hat * 0.5,  # Less adverse after entry
        vol_regime=vol_regime,
        liquidity=liquidity,
        cfg=cfg,
    )
    
    return {
        "entry_offset_bps": entry_offset,
        "exit_offset_bps": exit_offset,
        "fill_prob_entry": fill_prob_entry,
        "fill_prob_exit": fill_prob_exit,
        "fee_entry": fee_limit_entry,
        "fee_exit": fee_limit_exit,
        "fee_market_entry": fee_market,
        "fee_market_exit": get_fee_for_order_type("market", "exit", cfg),
    }
