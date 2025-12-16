
import numpy as np
import pandas as pd
from typing import Optional, Dict, List, Tuple, Union

def calc_prob_touch_sl(
    mu: float,
    sigma: float,
    sl_dist: float,
    tp_dist: float,
    direction: int = 1  # 1 for Long, -1 for Short
) -> float:
    """
    Calculates the probability of touching the Stop Loss (SL) before the Take Profit (TP),
    assuming a Geometric Brownian Motion (GBM) with drift mu and volatility sigma.

    Robust implementation managing numerical stability and directionality.
    
    Args:
        mu: Drift of the process (e.g., annualized return). 
        sigma: Volatility of the process. Must be > 0.
        sl_dist: Distance to Stop Loss (absolute value, > 0). e.g. 0.01 for 1%.
        tp_dist: Distance to Take Profit (absolute value, > 0). e.g. 0.02 for 2%.
        direction: Trade direction (+1 or -1).
        
    Returns:
        float: Probability [0.0, 1.0] of hitting SL before TP.
    """
    # 1. Effective Drift relative to trade direction
    # If Long (+1): positive mu is good.
    # If Short (-1): positive mu is bad (drift against you).
    # We transform problem space so "Up" is "Winning" (towards TP).
    # If Short: Drift -> -Drift.
    
    mu_eff_trade = mu * float(direction)
    
    # 2. Geometric adjustment (log-space drift)
    # d(ln x) = (mu - 0.5*sigma^2)dt
    sigma_clamped = max(float(sigma), 1e-6)
    variance_term = 0.5 * (sigma_clamped ** 2)
    drift_log = mu_eff_trade - variance_term
    
    # 3. Ratio for first passage (Brownian in log space)
    # Use the canonical shifted-interval formula.
    # Let lower barrier at -sl, upper at +tp, start at 0.
    # Shift to [0, sl+tp] with start at x=sl.
    ratio = 2.0 * drift_log / (sigma_clamped ** 2)

    sl = float(max(sl_dist, 0.0))
    tp = float(max(tp_dist, 0.0))
    L = sl + tp
    if not np.isfinite(L) or L <= 1e-12:
        return 1.0

    # 4. Handle Martingale Case (Drift ~ 0)
    if abs(ratio) < 1e-6:
        return float(tp / L)

    # 5. Robust exponentials
    MAX_EXP = 50.0
    arg_x = np.clip(-ratio * sl, -MAX_EXP, MAX_EXP)
    arg_L = np.clip(-ratio * L, -MAX_EXP, MAX_EXP)
    exp_x = np.exp(arg_x)
    exp_L = np.exp(arg_L)

    denom = 1.0 - exp_L
    if abs(float(denom)) < 1e-12:
        return 1.0

    p_hit_tp_first = (1.0 - exp_x) / denom
    p_hit_sl_first = 1.0 - p_hit_tp_first
    return float(np.clip(p_hit_sl_first, 0.0, 1.0))


def calc_prob_touch_sl_vec(
    mu: np.ndarray,
    sigma: np.ndarray,
    sl_dist: np.ndarray,
    tp_dist: np.ndarray,
    direction: int = 1,
) -> np.ndarray:
    mu_arr = np.asarray(mu, dtype=float)
    sigma_arr = np.asarray(sigma, dtype=float)
    sl_arr = np.asarray(sl_dist, dtype=float)
    tp_arr = np.asarray(tp_dist, dtype=float)

    mu_eff_trade = mu_arr * float(direction)
    sigma_clamped = np.maximum(sigma_arr, 1e-6)
    drift_log = mu_eff_trade - 0.5 * (sigma_clamped ** 2)
    ratio = 2.0 * drift_log / (sigma_clamped ** 2)

    sl = np.maximum(sl_arr, 0.0)
    tp = np.maximum(tp_arr, 0.0)
    L = sl + tp

    out = np.ones_like(L, dtype=float)
    valid = np.isfinite(L) & (L > 1e-12) & np.isfinite(ratio)
    if not np.any(valid):
        return out

    # martingale fallback
    small = valid & (np.abs(ratio) < 1e-6)
    if np.any(small):
        out[small] = np.clip(tp[small] / L[small], 0.0, 1.0)

    reg = valid & (~small)
    if np.any(reg):
        MAX_EXP = 50.0
        arg_x = np.clip(-ratio[reg] * sl[reg], -MAX_EXP, MAX_EXP)
        arg_L = np.clip(-ratio[reg] * L[reg], -MAX_EXP, MAX_EXP)
        exp_x = np.exp(arg_x)
        exp_L = np.exp(arg_L)
        denom = 1.0 - exp_L
        safe = np.abs(denom) > 1e-12
        p_tp = np.zeros_like(denom, dtype=float)
        p_tp[safe] = (1.0 - exp_x[safe]) / denom[safe]
        p_sl = 1.0 - p_tp
        out[reg] = np.clip(p_sl, 0.0, 1.0)

    out = np.where(np.isfinite(out), out, 1.0)
    return out

def compute_moe_weights(
    regime_params: Dict[str, float],
    market_state: Dict[str, float],
    expert_names: List[str]
) -> np.ndarray:
    """
    Computes dynamic weights for the Mixture of Experts (MoE) committee.
    Includes explicit normalization and thresholding.
    
    Args:
        regime_params: HPO params including thresholds.
        market_state: Current state dict (keys: 'adx', 'vol_ratio').
        expert_names: List of expert names.
        
    Returns:
        np.ndarray: Normalized weights sum to 1.0.
    """
    n_experts = len(expert_names)
    weights = np.ones(n_experts, dtype=float)
    
    # Extract state
    adx = market_state.get('adx', 20.0)
    vol_ratio = market_state.get('vol_ratio', 1.0)
    
    # Extract HPO params
    trend_dom = regime_params.get('moe_trend_dominance', 0.0)
    scalp_dom = regime_params.get('moe_scalp_dominance', 0.0)
    vol_sens  = regime_params.get('moe_vol_sensitivity', 0.0)
    
    # Thresholds (HPO or Default)
    # Backward compatible: prefer quantile-style knobs if market_state provides distribution stats.
    thr_trend_high = float(regime_params.get('moe_trend_threshold', 25.0))
    thr_trend_low = float(regime_params.get('moe_chop_threshold', 20.0))

    try:
        q_trend = float(regime_params.get('moe_adx_trend_q', np.nan))
    except Exception:
        q_trend = float('nan')
    try:
        q_chop = float(regime_params.get('moe_adx_chop_q', np.nan))
    except Exception:
        q_chop = float('nan')
    try:
        q_vol = float(regime_params.get('moe_vol_spike_q', np.nan))
    except Exception:
        q_vol = float('nan')

    # Optional distribution info (can be passed by callers)
    try:
        adx_q05 = float(market_state.get('adx_q05', np.nan))
        adx_q50 = float(market_state.get('adx_q50', np.nan))
        adx_q95 = float(market_state.get('adx_q95', np.nan))
    except Exception:
        adx_q05 = adx_q50 = adx_q95 = float('nan')

    try:
        vol_q50 = float(market_state.get('vol_ratio_q50', np.nan))
        vol_q95 = float(market_state.get('vol_ratio_q95', np.nan))
    except Exception:
        vol_q50 = vol_q95 = float('nan')

    def _interp_quantile(q: float, lo: float, mid: float, hi: float) -> float:
        if (not np.isfinite(q)) or (not np.isfinite(lo)) or (not np.isfinite(mid)) or (not np.isfinite(hi)):
            return float('nan')
        q = float(np.clip(q, 0.0, 1.0))
        if q <= 0.5:
            return float(lo + (mid - lo) * (q / 0.5))
        return float(mid + (hi - mid) * ((q - 0.5) / 0.5))

    try:
        thr_trend_q = _interp_quantile(q_trend, adx_q05, adx_q50, adx_q95)
        thr_chop_q = _interp_quantile(q_chop, adx_q05, adx_q50, adx_q95)
        if np.isfinite(thr_trend_q):
            thr_trend_high = float(thr_trend_q)
        if np.isfinite(thr_chop_q):
            thr_trend_low = float(thr_chop_q)
        if thr_trend_high < thr_trend_low + 1e-6:
            thr_trend_high = float(thr_trend_low + 1.0)
    except Exception:
        pass
    
    # Expert Names for helper
    # exp_names = ["Scalp_L", "Scalp_S", "Swing_L", "Swing_S", "Trend_L", "Trend_S"]
    # FUTURE: ["Breakout_L", "VWAP_Rev_L", "VolShock_S", ...]
    
    # 1. Trend Regime Logic
    # Group Experts
    trend_experts = []
    scalp_experts = []
    swing_experts = []
    # future_breakout_experts = []
    # future_vwap_experts = []
    
    for i, name in enumerate(expert_names):
        name_lower = name.lower()
        if 'trend' in name_lower: trend_experts.append(i)
        elif 'scalp' in name_lower: scalp_experts.append(i)
        elif 'swing' in name_lower: swing_experts.append(i)
        # elif 'breakout' in name_lower: future_breakout_experts.append(i)
        # elif 'vwap' in name_lower: future_vwap_experts.append(i)
        
    is_strong_trend = adx > thr_trend_high
    is_choppy = adx < thr_trend_low
    
    if is_strong_trend and trend_dom > 0.01:
        # Boost Trend, Penalize others
        factor = 1.0 + trend_dom * 2.0
        for i in trend_experts: weights[i] *= factor
        # Penalize others
        p_factor = max(0.1, 1.0 - trend_dom)
        for i in scalp_experts + swing_experts: weights[i] *= p_factor
            
    elif is_choppy and scalp_dom > 0.01:
        # Boost Scalp/Swing, Penalize Trend
        factor = 1.0 + scalp_dom * 2.0
        for i in scalp_experts + swing_experts: weights[i] *= factor
        # Penalize Trend
        p_factor = max(0.1, 1.0 - scalp_dom)
        for i in trend_experts: weights[i] *= p_factor
        
    # 2. Volatility Logic
    thr_vol = 1.5
    try:
        if np.isfinite(q_vol) and np.isfinite(vol_q50) and np.isfinite(vol_q95):
            thr_vol_q = _interp_quantile(q_vol, vol_q50, vol_q50, vol_q95)
            if np.isfinite(thr_vol_q) and float(thr_vol_q) > 0:
                thr_vol = float(thr_vol_q)
    except Exception:
        pass

    if vol_ratio > thr_vol and vol_sens > 0.01:
        # Boost Swing (wider stops), Penalize Scalp (tight stops)
        factor = 1.0 + vol_sens
        for i in swing_experts: weights[i] *= factor
        
        p_factor = max(0.1, 1.0 - vol_sens * 0.5)
        for i in scalp_experts: weights[i] *= p_factor
                 
    # 3. Normalization
    w_sum = np.sum(weights)
    if w_sum < 1e-9:
        return np.ones(n_experts) / n_experts
        
    return weights / w_sum


# =============================================================================
# NEW EXPERTS: Breakout/Expansion, VWAP Reversion, Volatility Shock
# =============================================================================
# Each expert outputs (score, confidence) per event:
#   - score: signed float in [-1, +1], direction of signal
#   - confidence: float in [0, 1], strength/reliability of signal
# =============================================================================


def compute_breakout_expansion_expert(
    market_data: pd.DataFrame,
    event_idx: pd.DatetimeIndex,
    lookback: int = 20,
    squeeze_lookback: int = 10,
    direction: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Breakout / Range-Expansion expert.
    
    When it helps: transition out of chop into trend, post-compression.
    Signals: Donchian breakout, BB squeeze → expansion, range percentile.
    Gate tie-in: high choppiness + rising trend_strength or rising vol_level → boost.
    
    Args:
        market_data: OHLCV DataFrame with 'high', 'low', 'close' columns.
        event_idx: DatetimeIndex of events to score.
        lookback: Donchian channel lookback.
        squeeze_lookback: BB squeeze detection lookback.
        direction: +1 for long, -1 for short.
        
    Returns:
        (scores, confidences): Arrays of shape (len(event_idx),).
    """
    n_events = len(event_idx)
    scores = np.zeros(n_events, dtype=float)
    confidences = np.zeros(n_events, dtype=float)
    
    if market_data is None or len(market_data) < lookback + 5:
        return scores, confidences
    
    try:
        high = market_data['high'].astype(float)
        low = market_data['low'].astype(float)
        close = market_data['close'].astype(float)
        
        # Donchian Channel
        donchian_high = high.rolling(lookback).max()
        donchian_low = low.rolling(lookback).min()
        donchian_mid = (donchian_high + donchian_low) / 2.0
        donchian_width = donchian_high - donchian_low
        
        # Bollinger Band width (proxy for squeeze)
        bb_std = close.rolling(squeeze_lookback).std()
        bb_width = 2.0 * bb_std
        
        # Range percentile: where is close within recent range?
        range_pct = (close - donchian_low) / (donchian_width + 1e-10)
        
        # Squeeze detection: BB width relative to Donchian width
        squeeze_ratio = bb_width / (donchian_width + 1e-10)
        squeeze_expanding = squeeze_ratio.diff(3)  # positive = expansion
        
        # Breakout detection
        breakout_up = (close > donchian_high.shift(1)).astype(float)
        breakout_down = (close < donchian_low.shift(1)).astype(float)
        
        # Reindex to events
        range_pct_ev = range_pct.reindex(event_idx).fillna(0.5).values
        squeeze_exp_ev = squeeze_expanding.reindex(event_idx).fillna(0.0).values
        breakout_up_ev = breakout_up.reindex(event_idx).fillna(0.0).values
        breakout_down_ev = breakout_down.reindex(event_idx).fillna(0.0).values
        
        # Score: combine breakout direction + range position
        # Long direction: breakout_up is positive, breakout_down is negative
        # Short direction: invert
        raw_score = (breakout_up_ev - breakout_down_ev) + 0.5 * (range_pct_ev - 0.5)
        raw_score = raw_score * float(direction)
        scores = np.clip(raw_score, -1.0, 1.0)
        
        # Confidence: higher when squeeze is expanding (post-compression breakout)
        squeeze_conf = np.clip(squeeze_exp_ev * 10.0, 0.0, 1.0)
        breakout_conf = np.abs(breakout_up_ev - breakout_down_ev)
        confidences = np.clip(0.3 + 0.4 * squeeze_conf + 0.3 * breakout_conf, 0.0, 1.0)
        
    except Exception:
        pass
    
    return scores, confidences


def compute_vwap_reversion_expert(
    market_data: pd.DataFrame,
    event_idx: pd.DatetimeIndex,
    zscore_lookback: int = 20,
    direction: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Mean-Reversion-to-VWAP / Liquidity-reversion expert.
    
    When it helps: chop / "sticky price" regimes, intraday pullbacks.
    Signals: distance-to-VWAP, zscore of (close−VWAP), volume imbalance proxies.
    Gate tie-in: high choppiness + low trend_strength → boost.
    
    Args:
        market_data: OHLCV DataFrame with 'close', 'volume' columns.
        event_idx: DatetimeIndex of events to score.
        zscore_lookback: Lookback for z-score calculation.
        direction: +1 for long, -1 for short.
        
    Returns:
        (scores, confidences): Arrays of shape (len(event_idx),).
    """
    n_events = len(event_idx)
    scores = np.zeros(n_events, dtype=float)
    confidences = np.zeros(n_events, dtype=float)
    
    if market_data is None or len(market_data) < zscore_lookback + 5:
        return scores, confidences
    
    try:
        close = market_data['close'].astype(float)
        volume = market_data.get('volume')
        if volume is None:
            volume = pd.Series(1.0, index=close.index)
        volume = volume.astype(float).replace(0, np.nan).fillna(1.0)
        
        # VWAP calculation (cumulative for session, or rolling)
        # Use rolling VWAP as proxy (more general)
        typical_price = close  # simplified; could use (H+L+C)/3
        cum_vol = volume.rolling(zscore_lookback).sum()
        cum_tp_vol = (typical_price * volume).rolling(zscore_lookback).sum()
        vwap = cum_tp_vol / (cum_vol + 1e-10)
        
        # Distance to VWAP
        dist_to_vwap = close - vwap
        dist_std = dist_to_vwap.rolling(zscore_lookback).std()
        zscore_vwap = dist_to_vwap / (dist_std + 1e-10)
        
        # Reindex to events
        zscore_ev = zscore_vwap.reindex(event_idx).fillna(0.0).values
        
        # Score: mean reversion signal
        # If price is above VWAP (zscore > 0), expect reversion down → short signal
        # If price is below VWAP (zscore < 0), expect reversion up → long signal
        # For long direction: negative zscore is bullish (buy the dip)
        # For short direction: positive zscore is bearish (sell the rip)
        raw_score = -zscore_ev * float(direction)  # invert: below VWAP = long signal
        raw_score = np.clip(raw_score / 2.0, -1.0, 1.0)  # scale down
        scores = raw_score
        
        # Confidence: higher when zscore is extreme (clear deviation)
        abs_z = np.abs(zscore_ev)
        confidences = np.clip(abs_z / 3.0, 0.1, 1.0)  # z=3 → conf=1.0
        
    except Exception:
        pass
    
    return scores, confidences


def compute_vol_shock_expert(
    market_data: pd.DataFrame,
    event_idx: pd.DatetimeIndex,
    vol_lookback: int = 20,
    shock_threshold: float = 1.5,
    direction: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Volatility Shock / "Event" expert.
    
    When it helps: sudden vol spikes where swing/trend logic lags.
    Signals: vol-of-vol, ATR jump, realized vol percentile crossing, gap size.
    Gate tie-in: high vol_level or rapid vol_level change → boost.
    
    Args:
        market_data: OHLCV DataFrame with 'high', 'low', 'close' columns.
        event_idx: DatetimeIndex of events to score.
        vol_lookback: Lookback for volatility calculation.
        shock_threshold: Multiplier above which vol is considered "shocked".
        direction: +1 for long, -1 for short.
        
    Returns:
        (scores, confidences): Arrays of shape (len(event_idx),).
    """
    n_events = len(event_idx)
    scores = np.zeros(n_events, dtype=float)
    confidences = np.zeros(n_events, dtype=float)
    
    if market_data is None or len(market_data) < vol_lookback + 5:
        return scores, confidences
    
    try:
        high = market_data['high'].astype(float)
        low = market_data['low'].astype(float)
        close = market_data['close'].astype(float)
        
        # ATR calculation
        tr = pd.concat([
            high - low,
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs()
        ], axis=1).max(axis=1)
        atr = tr.rolling(vol_lookback).mean()
        atr_std = atr.rolling(vol_lookback).std()
        
        # Vol-of-vol: how much is ATR changing?
        vol_of_vol = atr_std / (atr + 1e-10)
        
        # ATR jump: current ATR vs recent median
        atr_median = atr.rolling(vol_lookback).median()
        atr_ratio = atr / (atr_median + 1e-10)
        
        # Gap detection
        gap = (close - close.shift(1)).abs() / (atr + 1e-10)
        
        # Shock detection: ATR ratio above threshold
        is_shock = (atr_ratio > shock_threshold).astype(float)
        
        # Price direction during shock
        price_dir = np.sign(close - close.shift(1))
        
        # Reindex to events
        is_shock_ev = is_shock.reindex(event_idx).fillna(0.0).values
        price_dir_ev = price_dir.reindex(event_idx).fillna(0.0).values
        atr_ratio_ev = atr_ratio.reindex(event_idx).fillna(1.0).values
        vol_of_vol_ev = vol_of_vol.reindex(event_idx).fillna(0.0).values
        gap_ev = gap.reindex(event_idx).fillna(0.0).values
        
        # Score: during vol shock, follow the momentum (vol expansion often continues)
        # Direction alignment: if shock + price moving up → long signal
        raw_score = is_shock_ev * price_dir_ev * float(direction)
        scores = np.clip(raw_score, -1.0, 1.0)
        
        # Confidence: higher when shock is clear (high ATR ratio, high vol-of-vol)
        shock_intensity = np.clip((atr_ratio_ev - 1.0) / 2.0, 0.0, 1.0)
        vov_conf = np.clip(vol_of_vol_ev * 5.0, 0.0, 0.5)
        gap_conf = np.clip(gap_ev / 2.0, 0.0, 0.3)
        confidences = np.clip(shock_intensity + vov_conf + gap_conf, 0.0, 1.0)
        
        # Zero confidence when no shock
        confidences = confidences * is_shock_ev
        
    except Exception:
        pass
    
    return scores, confidences


def compute_new_experts_matrix(
    market_data: pd.DataFrame,
    event_idx: pd.DatetimeIndex,
    direction: int = 1,
    breakout_lookback: int = 20,
    vwap_lookback: int = 20,
    vol_lookback: int = 20,
    verbose: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute all 3 new experts and return combined matrices.
    
    Returns:
        (scores_matrix, confidence_matrix): Shape (n_events, 3).
        Column order: [breakout, vwap_reversion, vol_shock]
    """
    from src.utils.tprint import tprint_info, tprint_warning
    
    n_events = len(event_idx)
    scores_mat = np.zeros((n_events, 3), dtype=float)
    conf_mat = np.zeros((n_events, 3), dtype=float)
    
    expert_stats = {}
    
    # Breakout/Expansion expert
    try:
        s0, c0 = compute_breakout_expansion_expert(
            market_data, event_idx, lookback=breakout_lookback, direction=direction
        )
        scores_mat[:, 0] = s0
        conf_mat[:, 0] = c0
        expert_stats["breakout"] = {
            "n_positive": int(np.sum(s0 > 0)),
            "n_negative": int(np.sum(s0 < 0)),
            "n_zero": int(np.sum(s0 == 0)),
            "mean_conf": float(np.mean(c0)),
            "max_conf": float(np.max(c0)) if c0.size > 0 else 0.0,
        }
    except Exception as e:
        if verbose:
            tprint_warning(f"   [new_experts] breakout expert failed: {e}")
    
    # VWAP Reversion expert
    try:
        s1, c1 = compute_vwap_reversion_expert(
            market_data, event_idx, zscore_lookback=vwap_lookback, direction=direction
        )
        scores_mat[:, 1] = s1
        conf_mat[:, 1] = c1
        expert_stats["vwap_rev"] = {
            "n_positive": int(np.sum(s1 > 0)),
            "n_negative": int(np.sum(s1 < 0)),
            "n_zero": int(np.sum(s1 == 0)),
            "mean_conf": float(np.mean(c1)),
            "max_conf": float(np.max(c1)) if c1.size > 0 else 0.0,
        }
    except Exception as e:
        if verbose:
            tprint_warning(f"   [new_experts] vwap_rev expert failed: {e}")
    
    # Vol Shock expert
    try:
        s2, c2 = compute_vol_shock_expert(
            market_data, event_idx, vol_lookback=vol_lookback, direction=direction
        )
        scores_mat[:, 2] = s2
        conf_mat[:, 2] = c2
        expert_stats["vol_shock"] = {
            "n_positive": int(np.sum(s2 > 0)),
            "n_negative": int(np.sum(s2 < 0)),
            "n_zero": int(np.sum(s2 == 0)),
            "mean_conf": float(np.mean(c2)),
            "max_conf": float(np.max(c2)) if c2.size > 0 else 0.0,
        }
    except Exception as e:
        if verbose:
            tprint_warning(f"   [new_experts] vol_shock expert failed: {e}")
    
    # Log summary
    if verbose and expert_stats:
        for name, stats in expert_stats.items():
            frac_active = (stats["n_positive"] + stats["n_negative"]) / max(n_events, 1)
            tprint_info(
                f"   [new_experts] {name}: "
                f"+={stats['n_positive']}, -={stats['n_negative']}, 0={stats['n_zero']} "
                f"(active={frac_active:.1%}, mean_conf={stats['mean_conf']:.3f})"
            )
    
    return scores_mat, conf_mat


# Expert name constants for the new experts
NEW_EXPERT_NAMES = ["breakout", "vwap_rev", "vol_shock"]


def get_new_expert_diagnostics(
    scores_mat: np.ndarray,
    conf_mat: np.ndarray,
    expert_names: List[str] = None,
) -> Dict[str, Any]:
    """
    Generate diagnostic dict for new experts (for reporting).
    
    Args:
        scores_mat: (n_events, 3) scores matrix.
        conf_mat: (n_events, 3) confidence matrix.
        expert_names: Names for each column.
        
    Returns:
        Dict with per-expert statistics.
    """
    if expert_names is None:
        expert_names = NEW_EXPERT_NAMES
    
    diag = {}
    n_events = scores_mat.shape[0] if scores_mat.ndim == 2 else 0
    
    for i, name in enumerate(expert_names):
        if i >= scores_mat.shape[1]:
            break
        s = scores_mat[:, i]
        c = conf_mat[:, i]
        
        n_pos = int(np.sum(s > 0))
        n_neg = int(np.sum(s < 0))
        n_zero = int(np.sum(s == 0))
        
        diag[f"{name}_n_positive"] = n_pos
        diag[f"{name}_n_negative"] = n_neg
        diag[f"{name}_n_zero"] = n_zero
        diag[f"{name}_active_rate"] = float((n_pos + n_neg) / max(n_events, 1))
        diag[f"{name}_mean_score"] = float(np.mean(s))
        diag[f"{name}_std_score"] = float(np.std(s))
        diag[f"{name}_mean_conf"] = float(np.mean(c))
        diag[f"{name}_max_conf"] = float(np.max(c)) if c.size > 0 else 0.0
        
        # Correlation between score and confidence
        if np.std(s) > 1e-10 and np.std(c) > 1e-10:
            diag[f"{name}_score_conf_corr"] = float(np.corrcoef(np.abs(s), c)[0, 1])
        else:
            diag[f"{name}_score_conf_corr"] = 0.0
    
    return diag
