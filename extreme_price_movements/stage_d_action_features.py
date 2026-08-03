"""Causal, vectorised features for the Stage-D clear-event action layer.

The path supplied to :func:`path_to_clear_features` must end at the completed
bar on which the hurdle first cleared.  This module has no labels, action
fills, model code, or policy logic.  Liquidity-like quantities derived from
OHLCV retain ``_proxy`` in their names.
"""
from __future__ import annotations

import hashlib
from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd


A1_FEATURES = [
    "completed_bars_to_clear", "mfe_observed_bps", "mae_observed_bps",
    "mfe_to_mae_ratio", "mae_before_clear_bps", "distance_to_observed_mfe_bps",
    "giveback_from_observed_mfe_bps", "fraction_observed_mfe_surrendered",
    "time_since_observed_mfe_minutes", "time_since_observed_mae_minutes",
    "path_efficiency", "sum_absolute_returns_bps", "directional_consistency",
    "direction_changes", "max_counter_direction_move_bps",
    "max_same_direction_continuation_bps", "return_slope_bps_per_bar",
    "return_slope_r2", "short_vs_full_path_slope", "return_acceleration_into_clear",
    "clear_single_jump_fraction",
]
A2_FEATURES = [
    "latest_candle_body_to_range", "latest_close_location", "latest_side_aligned_wick_rejection",
    "rolling_side_wick_imbalance", "failed_breakout_count", "breakout_rejection_intensity",
    "distance_from_recent_extreme_bps", "recency_of_recent_extreme_minutes",
    "new_side_extremes_since_entry", "range_expansion_into_clear", "range_compression_before_clear",
    "compression_to_expansion_transition", "post_impulse_rejection",
    "fraction_clear_move_latest_bar", "jump_concentration",
]
A3_FEATURES = [
    "volume_since_entry", "volume_acceleration", "volume_persistence", "volume_z_at_clear",
    "latest_bars_volume_fraction", "signed_volume_proxy", "cumulative_signed_volume_proxy",
    "obv_change_proxy", "obv_slope_proxy", "price_volume_correlation",
    "return_volume_correlation", "volume_weighted_path_efficiency",
    "volume_confirmed_continuation", "high_volume_low_efficiency_churn",
    "volume_climax", "volume_shock_age_minutes", "volume_shock_decay",
    "range_per_unit_volume_proxy", "absolute_return_per_unit_volume_proxy",
]
A4_FEATURES = [
    "realised_volatility", "side_adverse_semivolatility", "short_full_volatility_ratio",
    "volatility_of_volatility", "atr_change_since_entry", "range_expansion_ratio",
    "squared_return_autocorrelation", "jump_frequency", "extreme_bar_frequency",
    "volatility_shock_magnitude", "time_since_volatility_shock_minutes",
    "volatility_shock_decay", "return_per_unit_volatility", "path_efficiency_conditional_on_volatility",
]
A5_FEATURES = [
    "market_return_since_entry", "market_recent_action_return", "side_aligned_breadth",
    "return_breadth", "return_dispersion", "volatility_dispersion", "volume_breadth",
    "candidate_cross_sectional_return_rank", "candidate_volume_rank",
    "candidate_residual_return_vs_market", "market_beta", "asset_move_vs_market_move",
    "breadth_confirmation", "isolated_move_indicator", "leader_laggard_status",
    "change_in_breadth_since_entry", "change_in_dispersion_since_entry",
    "eligible_universe_size",
]
A9_FEATURES = [
    "path_efficiency_x_volume_persistence", "path_efficiency_x_breadth_confirmation",
    "return_acceleration_x_wick_rejection", "volume_climax_x_low_path_efficiency",
    "volatility_climax_x_wick_rejection", "isolated_move_x_volume_climax",
    "giveback_x_time_since_mfe", "time_to_clear_x_path_efficiency",
]


def _div(a: float | np.ndarray, b: float | np.ndarray, floor: float = 1e-12):
    return np.asarray(a) / np.maximum(np.abs(np.asarray(b)), floor)


def decode_completed_path(raw: Any, stop_index: int) -> dict[str, np.ndarray]:
    """Decode exactly bars ``0..stop_index``; later values are never returned."""
    import json

    payload = json.loads(raw) if isinstance(raw, str) else raw
    if not 0 <= int(stop_index) < len(payload["timestamp"]):
        raise ValueError("invalid completed-path stop index")
    end = int(stop_index) + 1
    result = {}
    for name in ("timestamp", "open", "high", "low", "close"):
        if name not in payload:
            raise ValueError(f"path lacks {name}")
        dtype = np.int64 if name == "timestamp" else np.float64
        result[name] = np.asarray(payload[name][:end], dtype=dtype)
    if "volume" in payload:
        result["volume"] = np.asarray(payload["volume"][:end], dtype=np.float64)
    n = {len(value) for value in result.values()}
    if n != {end} or not all(np.isfinite(result[k]).all() for k in ("open", "high", "low", "close")):
        raise ValueError("invalid completed path")
    return result


def _slope_r2(values: np.ndarray) -> tuple[float, float]:
    if len(values) < 2:
        return 0.0, 0.0
    x = np.arange(len(values), dtype=float)
    xc, yc = x - x.mean(), values - values.mean()
    den = float(xc @ xc)
    slope = float(xc @ yc / den) if den else 0.0
    yden = float(yc @ yc)
    r2 = float((xc @ yc) ** 2 / (den * yden)) if den and yden else 0.0
    return slope, float(np.clip(r2, 0.0, 1.0))


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 3 or np.std(a) <= 1e-15 or np.std(b) <= 1e-15:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def path_to_clear_features(raw: Any, *, stop_index: int, side: str, entry_price: float) -> dict[str, float]:
    """Calculate A1--A4 from completed one-minute bars through first clear."""
    p = decode_completed_path(raw, stop_index)
    sign = 1.0 if side == "long" else -1.0 if side == "short" else np.nan
    if not np.isfinite(sign) or not np.isfinite(entry_price) or entry_price <= 0:
        raise ValueError("invalid side/entry")
    o, h, l, c = (p[k] for k in ("open", "high", "low", "close"))
    volume_available = "volume" in p and np.isfinite(p["volume"]).all()
    v = p["volume"] if volume_available else np.full(len(c), np.nan)
    n = len(c)
    side_close = sign * (c / entry_price - 1.0) * 1e4
    side_high = (h / entry_price - 1.0) * 1e4 if sign > 0 else (1.0 - l / entry_price) * 1e4
    side_low = (l / entry_price - 1.0) * 1e4 if sign > 0 else (1.0 - h / entry_price) * 1e4
    mfe, mae = float(np.max(side_high)), float(np.min(side_low))
    ret = np.diff(np.log(np.r_[entry_price, c])) * sign * 1e4
    abs_sum = float(np.abs(ret).sum())
    path_eff = float(abs(side_close[-1]) / max(abs_sum, 1e-12))
    slopes = _slope_r2(side_close)
    short_n = min(5, n)
    short_slope = _slope_r2(side_close[-short_n:])[0]
    prev_slope = _slope_r2(side_close[:-short_n] if n > short_n + 1 else side_close[: max(1, n // 2)])[0]
    bar_range = np.maximum(h - l, 0.0)
    body = c - o
    close_loc = _div(c - l, bar_range)
    upper = np.maximum(h - np.maximum(o, c), 0.0)
    lower = np.maximum(np.minimum(o, c) - l, 0.0)
    side_wick_reject = _div(upper if sign > 0 else lower, bar_range)
    side_wick_support = _div(lower if sign > 0 else upper, bar_range)
    prior_extreme = np.maximum.accumulate(h) if sign > 0 else np.minimum.accumulate(l)
    failed = ((h > np.r_[h[0], np.maximum.accumulate(h[:-1])]) & (c < np.r_[h[0], np.maximum.accumulate(h[:-1])])) if sign > 0 else ((l < np.r_[l[0], np.minimum.accumulate(l[:-1])]) & (c > np.r_[l[0], np.minimum.accumulate(l[:-1])]))
    new_extreme = np.r_[True, np.diff(prior_extreme) * sign > 0]
    recent_window = min(30, n)
    extreme_idx = int(np.argmax(h) if sign > 0 else np.argmin(l))
    range_mean_first = float(np.mean(bar_range[: max(1, n // 3)]))
    range_mean_last = float(np.mean(bar_range[-min(5, n):]))
    range_before = float(np.mean(bar_range[max(0, n - 10): max(1, n - 3)])) if n > 3 else float(np.mean(bar_range))
    signed_volume = np.sign(c - o) * v
    side_signed_volume = signed_volume * sign
    vol_first = float(np.mean(v[: max(1, n // 3)]))
    vol_last = float(np.mean(v[-min(5, n):]))
    vol_mean, vol_std = float(np.mean(v)), float(np.std(v))
    volume_z = float((v[-1] - vol_mean) / max(vol_std, 1e-12))
    shock = np.flatnonzero(v > vol_mean + 2 * vol_std)
    shock_age = float(n - 1 - shock[-1]) if len(shock) else float(n)
    rv = float(np.sqrt(np.mean(ret * ret)))
    adverse_rv = float(np.sqrt(np.mean(np.minimum(ret, 0.0) ** 2)))
    short_rv = float(np.sqrt(np.mean(ret[-min(5, n):] ** 2)))
    rolling_vol = (pd.Series(ret).rolling(min(5, n), min_periods=min(2, n)).std().dropna().to_numpy() if n else np.array([]))
    typical_range = float(np.median(bar_range))
    jump_cut = max(float(np.median(np.abs(ret)) * 3.0), 1e-12)
    vol_shock = np.flatnonzero(bar_range > np.mean(bar_range) + 2 * np.std(bar_range))
    vol_shock_age = float(n - 1 - vol_shock[-1]) if len(vol_shock) else float(n)
    latest_move = float(abs(ret[-1])) if len(ret) else 0.0
    max_run_same = max_run_counter = 0.0
    run_same = run_counter = 0.0
    for value in ret:
        if value >= 0: run_same += value; run_counter = 0.0
        else: run_counter += -value; run_same = 0.0
        max_run_same, max_run_counter = max(max_run_same, run_same), max(max_run_counter, run_counter)
    result = {
        "_path_last_bar_open_ns": int(p["timestamp"][-1]),
        "completed_bars_to_clear": float(n), "mfe_observed_bps": mfe, "mae_observed_bps": mae,
        "mfe_to_mae_ratio": float(mfe / max(abs(mae), 1e-12)), "mae_before_clear_bps": mae,
        "distance_to_observed_mfe_bps": float(mfe - side_close[-1]),
        "giveback_from_observed_mfe_bps": float(mfe - side_close[-1]),
        "fraction_observed_mfe_surrendered": float((mfe - side_close[-1]) / max(abs(mfe), 1e-12)),
        "time_since_observed_mfe_minutes": float(n - 1 - int(np.argmax(side_high))),
        "time_since_observed_mae_minutes": float(n - 1 - int(np.argmin(side_low))),
        "path_efficiency": path_eff, "sum_absolute_returns_bps": abs_sum,
        "directional_consistency": float(np.mean(ret >= 0)),
        "direction_changes": float(np.sum(np.sign(ret[1:]) != np.sign(ret[:-1]))) if n > 1 else 0.0,
        "max_counter_direction_move_bps": max_run_counter, "max_same_direction_continuation_bps": max_run_same,
        "return_slope_bps_per_bar": slopes[0], "return_slope_r2": slopes[1],
        "short_vs_full_path_slope": float(short_slope / max(abs(slopes[0]), 1e-12)),
        "return_acceleration_into_clear": float(short_slope - prev_slope),
        "clear_single_jump_fraction": float(np.max(np.maximum(ret, 0.0)) / max(side_close[-1], 1e-12)),
        "latest_candle_body_to_range": float(abs(body[-1]) / max(bar_range[-1], 1e-12)),
        "latest_close_location": float(close_loc[-1]), "latest_side_aligned_wick_rejection": float(side_wick_reject[-1]),
        "rolling_side_wick_imbalance": float(np.mean(side_wick_support - side_wick_reject)),
        "failed_breakout_count": float(failed.sum()),
        "breakout_rejection_intensity": float(np.sum(side_wick_reject * failed)),
        "distance_from_recent_extreme_bps": float((mfe - side_close[-1])),
        "recency_of_recent_extreme_minutes": float(n - 1 - extreme_idx),
        "new_side_extremes_since_entry": float(new_extreme.sum()),
        "range_expansion_into_clear": float(range_mean_last / max(range_mean_first, 1e-12)),
        "range_compression_before_clear": float(range_before / max(range_mean_first, 1e-12)),
        "compression_to_expansion_transition": float(range_mean_last / max(range_before, 1e-12)),
        "post_impulse_rejection": float(side_wick_reject[-1] * latest_move / max(abs_sum, 1e-12)),
        "fraction_clear_move_latest_bar": float(max(ret[-1], 0.0) / max(side_close[-1], 1e-12)),
        "jump_concentration": float(np.max(np.abs(ret)) / max(abs_sum, 1e-12)),
        "volume_since_entry": float(v.sum()), "volume_acceleration": float(vol_last / max(vol_first, 1e-12) - 1),
        "volume_persistence": float(vol_last / max(vol_mean, 1e-12)), "volume_z_at_clear": volume_z,
        "latest_bars_volume_fraction": float(v[-min(5, n):].sum() / max(v.sum(), 1e-12)),
        "signed_volume_proxy": float(side_signed_volume[-1]), "cumulative_signed_volume_proxy": float(side_signed_volume.sum()),
        "obv_change_proxy": float(side_signed_volume.sum()), "obv_slope_proxy": _slope_r2(np.cumsum(side_signed_volume))[0],
        "price_volume_correlation": _corr(side_close, v), "return_volume_correlation": _corr(ret, v),
        "volume_weighted_path_efficiency": float(path_eff * np.average(np.maximum(ret, 0), weights=np.maximum(v, 1e-12)) / max(np.mean(np.abs(ret)), 1e-12)),
        "volume_confirmed_continuation": float(path_eff * max(volume_z, 0.0)),
        "high_volume_low_efficiency_churn": float(max(volume_z, 0.0) * (1 - np.clip(path_eff, 0, 1))),
        "volume_climax": float(max(volume_z, 0.0)), "volume_shock_age_minutes": shock_age,
        "volume_shock_decay": float(np.exp(-shock_age / max(n, 1))),
        "range_per_unit_volume_proxy": float(bar_range.sum() / max(v.sum(), 1e-12)),
        "absolute_return_per_unit_volume_proxy": float(abs_sum / max(v.sum(), 1e-12)),
        "realised_volatility": rv, "side_adverse_semivolatility": adverse_rv,
        "short_full_volatility_ratio": float(short_rv / max(rv, 1e-12)),
        "volatility_of_volatility": float(np.std(rolling_vol)) if len(rolling_vol) else 0.0,
        "atr_change_since_entry": float(range_mean_last / max(range_mean_first, 1e-12) - 1),
        "range_expansion_ratio": float(bar_range[-1] / max(typical_range, 1e-12)),
        "squared_return_autocorrelation": _corr(ret[1:] ** 2, ret[:-1] ** 2),
        "jump_frequency": float(np.mean(np.abs(ret) > jump_cut)),
        "extreme_bar_frequency": float(np.mean(bar_range > np.mean(bar_range) + 2 * np.std(bar_range))),
        "volatility_shock_magnitude": float(max((bar_range[-1] - np.mean(bar_range)) / max(np.std(bar_range), 1e-12), 0.0)),
        "time_since_volatility_shock_minutes": vol_shock_age,
        "volatility_shock_decay": float(np.exp(-vol_shock_age / max(n, 1))),
        "return_per_unit_volatility": float(side_close[-1] / max(rv, 1e-12)),
        "path_efficiency_conditional_on_volatility": float(path_eff / max(rv, 1e-12)),
        "side_return_since_entry_bps": float(side_close[-1]), "raw_return_since_entry": float(np.log(c[-1] / entry_price)),
        "recent_raw_return": float(np.sum(np.diff(np.log(np.r_[entry_price, c]))[-min(5, n):])),
        "path_volume_mean": vol_mean,
    }
    if not volume_available:
        for name in A3_FEATURES:
            result[name] = np.nan
        result["path_volume_mean"] = np.nan
    return {name: (int(value) if name == "_path_last_bar_open_ns" else float(value)) for name, value in result.items()}


def batch_path_to_clear_features(raw_paths: list[Any], *, stop_indices: np.ndarray, sides: np.ndarray, entry_prices: np.ndarray) -> pd.DataFrame:
    """Bounded NumPy batch implementation of price-only A1/A2/A4.

    JSON parsing is necessarily per payload, but all numerical path work is
    vectorised across the batch.  The only time loop tracks consecutive runs
    across the fixed 720-column matrix while updating every row at once.
    """
    import json

    b = len(raw_paths)
    if not (len(stop_indices) == len(sides) == len(entry_prices) == b):
        raise ValueError("batch inputs differ in length")
    payloads = [json.loads(raw) if isinstance(raw, str) else raw for raw in raw_paths]
    width = max(int(i) for i in stop_indices) + 1
    ts = np.stack([np.asarray(p["timestamp"][:width], dtype=np.int64) for p in payloads])
    o = np.stack([np.asarray(p["open"][:width], dtype=float) for p in payloads])
    h = np.stack([np.asarray(p["high"][:width], dtype=float) for p in payloads])
    l = np.stack([np.asarray(p["low"][:width], dtype=float) for p in payloads])
    c = np.stack([np.asarray(p["close"][:width], dtype=float) for p in payloads])
    n = np.asarray(stop_indices, dtype=int) + 1
    pos = np.arange(width)[None, :]
    mask = pos < n[:, None]
    sign = np.where(np.asarray(sides) == "long", 1.0, np.where(np.asarray(sides) == "short", -1.0, np.nan))
    entry = np.asarray(entry_prices, dtype=float)
    if not np.isfinite(sign).all() or not np.isfinite(entry).all() or (entry <= 0).any():
        raise ValueError("invalid batch side/entry")
    row = np.arange(b)
    last = n - 1
    side_close = sign[:, None] * (c / entry[:, None] - 1) * 1e4
    side_high = np.where(sign[:, None] > 0, (h / entry[:, None] - 1) * 1e4, (1 - l / entry[:, None]) * 1e4)
    side_low = np.where(sign[:, None] > 0, (l / entry[:, None] - 1) * 1e4, (1 - h / entry[:, None]) * 1e4)
    log_prev = np.concatenate([np.log(entry)[:, None], np.log(c[:, :-1])], axis=1)
    ret = sign[:, None] * (np.log(c) - log_prev) * 1e4
    ret = np.where(mask, ret, np.nan)
    abs_sum = np.nansum(np.abs(ret), axis=1)
    final_ret = side_close[row, last]
    mfe = np.nanmax(np.where(mask, side_high, np.nan), axis=1)
    mae = np.nanmin(np.where(mask, side_low, np.nan), axis=1)
    path_eff = np.abs(final_ret) / np.maximum(abs_sum, 1e-12)

    def slopes(values: np.ndarray, use: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        count = use.sum(axis=1).astype(float)
        x = np.broadcast_to(np.arange(width, dtype=float), values.shape)
        sx = np.sum(np.where(use, x, 0), axis=1); sy = np.sum(np.where(use, values, 0), axis=1)
        sxx = np.sum(np.where(use, x*x, 0), axis=1); syy = np.sum(np.where(use, values*values, 0), axis=1)
        sxy = np.sum(np.where(use, x*values, 0), axis=1)
        num = count*sxy-sx*sy; dx = count*sxx-sx*sx; dy = count*syy-sy*sy
        slope = np.divide(num, dx, out=np.zeros_like(num), where=dx != 0)
        r2 = np.divide(num*num, dx*dy, out=np.zeros_like(num), where=(dx*dy) != 0)
        return slope, np.clip(r2, 0, 1)
    slope, slope_r2 = slopes(side_close, mask)
    short_start = np.maximum(n - 5, 0)
    short_mask = mask & (pos >= short_start[:, None])
    short_slope, _ = slopes(side_close, short_mask)
    preceding_mask = mask & (pos < short_start[:, None])
    fallback_n = np.maximum(1, n // 2)
    preceding_mask = np.where((preceding_mask.sum(axis=1) >= 2)[:, None], preceding_mask, mask & (pos < fallback_n[:, None]))
    prior_slope, _ = slopes(side_close, preceding_mask)

    bar_range = np.maximum(h-l, 0); body = c-o
    close_loc = (c-l)/np.maximum(bar_range, 1e-12)
    upper = np.maximum(h-np.maximum(o,c),0); lower=np.maximum(np.minimum(o,c)-l,0)
    reject = np.where(sign[:,None]>0,upper,lower)/np.maximum(bar_range,1e-12)
    support = np.where(sign[:,None]>0,lower,upper)/np.maximum(bar_range,1e-12)
    prior_h = np.maximum.accumulate(h, axis=1); prior_l=np.minimum.accumulate(l,axis=1)
    ph=np.concatenate([h[:,:1],prior_h[:,:-1]],axis=1); pl=np.concatenate([l[:,:1],prior_l[:,:-1]],axis=1)
    failed=np.where(sign[:,None]>0,(h>ph)&(c<ph),(l<pl)&(c>pl)) & mask
    extreme=np.where(sign[:,None]>0,prior_h,prior_l)
    new_ext=np.ones_like(mask); new_ext[:,1:]=(np.diff(extreme,axis=1)*sign[:,None]>0); new_ext &= mask
    extreme_values=np.where(mask,np.where(sign[:,None]>0,h,-l),np.nan)
    extreme_idx=np.nanargmax(extreme_values,axis=1)
    first_count=np.maximum(1,n//3)
    first_mask=mask&(pos<first_count[:,None]); last5=mask&(pos>=np.maximum(n-5,0)[:,None])
    pre_mask=mask&(pos>=np.maximum(n-10,0)[:,None])&(pos<np.maximum(1,n-3)[:,None])
    pre_mask=np.where((n<=3)[:,None],mask,pre_mask)
    pre_mask=np.where((pre_mask.sum(axis=1)>0)[:,None],pre_mask,mask)
    mean_mask=lambda x,m: np.sum(np.where(m,x,0),axis=1)/np.maximum(m.sum(axis=1),1)
    range_first=mean_mask(bar_range,first_mask); range_last=mean_mask(bar_range,last5); range_pre=mean_mask(bar_range,pre_mask)
    median_range=np.nanmedian(np.where(mask,bar_range,np.nan),axis=1)
    median_abs=np.nanmedian(np.abs(ret),axis=1); jump_cut=np.maximum(median_abs*3,1e-12)
    rv=np.sqrt(np.nanmean(ret*ret,axis=1)); adverse=np.sqrt(np.nanmean(np.minimum(ret,0)**2,axis=1))
    short_rv=np.sqrt(mean_mask(np.nan_to_num(ret)**2,last5))
    range_mean=mean_mask(bar_range,mask); range_std=np.sqrt(mean_mask((bar_range-range_mean[:,None])**2,mask))
    vol_event=(bar_range>range_mean[:,None]+2*range_std[:,None])&mask
    event_pos=np.where(vol_event,pos,-1); last_event=event_pos.max(axis=1); vol_age=np.where(last_event>=0,last-last_event,n).astype(float)
    jump_freq=mean_mask((np.abs(ret)>jump_cut[:,None]).astype(float),mask)
    extreme_freq=mean_mask((bar_range>range_mean[:,None]+2*range_std[:,None]).astype(float),mask)
    # Vectorised adjacent squared-return correlation.
    def row_corr(a: np.ndarray, z: np.ndarray, valid: np.ndarray) -> np.ndarray:
        cnt=valid.sum(axis=1); am=mean_mask(a,valid); zm=mean_mask(z,valid)
        ac=np.where(valid,a-am[:,None],0); zc=np.where(valid,z-zm[:,None],0)
        den=np.sqrt(np.sum(ac*ac,axis=1)*np.sum(zc*zc,axis=1))
        return np.divide(np.sum(ac*zc,axis=1),den,out=np.zeros(b),where=(den>0)&(cnt>=3))
    ac_mask=mask[:,1:]; sq_corr=row_corr(np.nan_to_num(ret[:,1:])**2,np.nan_to_num(ret[:,:-1])**2,ac_mask)
    # Consecutive same/counter direction; loop is over fixed time, vectorised across rows.
    same_run=np.zeros(b); counter_run=np.zeros(b); max_same=np.zeros(b); max_counter=np.zeros(b)
    for j in range(width):
        active=mask[:,j]; value=np.nan_to_num(ret[:,j]); positive=value>=0
        same_run=np.where(active&positive,same_run+value,0); counter_run=np.where(active&~positive,counter_run-value,0)
        max_same=np.maximum(max_same,same_run); max_counter=np.maximum(max_counter,counter_run)
    direction_changes=np.sum(mask[:,1:]&(np.sign(np.nan_to_num(ret[:,1:]))!=np.sign(np.nan_to_num(ret[:,:-1]))),axis=1)
    out = pd.DataFrame({
        "_path_last_bar_open_ns": ts[row,last], "completed_bars_to_clear": n.astype(float), "mfe_observed_bps": mfe,
        "mae_observed_bps": mae, "mfe_to_mae_ratio": mfe/np.maximum(np.abs(mae),1e-12), "mae_before_clear_bps": mae,
        "distance_to_observed_mfe_bps": mfe-final_ret, "giveback_from_observed_mfe_bps": mfe-final_ret,
        "fraction_observed_mfe_surrendered": (mfe-final_ret)/np.maximum(np.abs(mfe),1e-12),
        "time_since_observed_mfe_minutes": last-np.nanargmax(np.where(mask,side_high,np.nan),axis=1),
        "time_since_observed_mae_minutes": last-np.nanargmin(np.where(mask,side_low,np.nan),axis=1),
        "path_efficiency": path_eff, "sum_absolute_returns_bps": abs_sum, "directional_consistency": mean_mask((ret>=0).astype(float),mask),
        "direction_changes": direction_changes, "max_counter_direction_move_bps": max_counter, "max_same_direction_continuation_bps": max_same,
        "return_slope_bps_per_bar": slope, "return_slope_r2": slope_r2, "short_vs_full_path_slope": short_slope/np.maximum(np.abs(slope),1e-12),
        "return_acceleration_into_clear": short_slope-prior_slope, "clear_single_jump_fraction": np.nanmax(np.maximum(ret,0),axis=1)/np.maximum(final_ret,1e-12),
        "latest_candle_body_to_range": np.abs(body[row,last])/np.maximum(bar_range[row,last],1e-12), "latest_close_location": close_loc[row,last],
        "latest_side_aligned_wick_rejection": reject[row,last], "rolling_side_wick_imbalance": mean_mask(support-reject,mask),
        "failed_breakout_count": failed.sum(axis=1), "breakout_rejection_intensity": np.sum(np.where(mask,reject*failed,0),axis=1),
        "distance_from_recent_extreme_bps": mfe-final_ret, "recency_of_recent_extreme_minutes": last-extreme_idx,
        "new_side_extremes_since_entry": new_ext.sum(axis=1), "range_expansion_into_clear": range_last/np.maximum(range_first,1e-12),
        "range_compression_before_clear": range_pre/np.maximum(range_first,1e-12), "compression_to_expansion_transition": range_last/np.maximum(range_pre,1e-12),
        "post_impulse_rejection": reject[row,last]*np.abs(ret[row,last])/np.maximum(abs_sum,1e-12),
        "fraction_clear_move_latest_bar": np.maximum(ret[row,last],0)/np.maximum(final_ret,1e-12), "jump_concentration": np.nanmax(np.abs(ret),axis=1)/np.maximum(abs_sum,1e-12),
        "realised_volatility": rv, "side_adverse_semivolatility": adverse, "short_full_volatility_ratio": short_rv/np.maximum(rv,1e-12),
        "volatility_of_volatility": np.nan, "atr_change_since_entry": range_last/np.maximum(range_first,1e-12)-1,
        "range_expansion_ratio": bar_range[row,last]/np.maximum(median_range,1e-12), "squared_return_autocorrelation": sq_corr,
        "jump_frequency": jump_freq, "extreme_bar_frequency": extreme_freq,
        "volatility_shock_magnitude": np.maximum((bar_range[row,last]-range_mean)/np.maximum(range_std,1e-12),0),
        "time_since_volatility_shock_minutes": vol_age, "volatility_shock_decay": np.exp(-vol_age/np.maximum(n,1)),
        "return_per_unit_volatility": final_ret/np.maximum(rv,1e-12), "path_efficiency_conditional_on_volatility": path_eff/np.maximum(rv,1e-12),
        "side_return_since_entry_bps": final_ret, "raw_return_since_entry": np.log(c[row,last]/entry),
        "recent_raw_return": np.sum(np.where(last5,np.log(c)-log_prev,0),axis=1), "path_volume_mean": np.nan,
    })
    # Vol-of-vol uses a small fixed rolling-width calculation across the batch.
    roll_std=[]
    for j in range(width):
        start=max(0,j-4); segment=ret[:,start:j+1]; valid_segment=np.isfinite(segment)
        count=valid_segment.sum(axis=1); mean=np.nansum(segment,axis=1)/np.maximum(count,1)
        variance=np.nansum(np.where(valid_segment,(segment-mean[:,None])**2,0),axis=1)/np.maximum(count-1,1)
        roll_std.append(np.where(mask[:,j]&(count>=2),np.sqrt(variance),np.nan))
    roll_matrix=np.stack(roll_std,axis=1)
    finite=np.isfinite(roll_matrix); roll_count=finite.sum(axis=1); roll_mean=np.nansum(roll_matrix,axis=1)/np.maximum(roll_count,1)
    out["volatility_of_volatility"] = np.sqrt(np.nansum(np.where(finite,(roll_matrix-roll_mean[:,None])**2,0),axis=1)/np.maximum(roll_count,1))
    for name in A3_FEATURES: out[name]=np.nan
    return out


def build_market_context_snapshots(bars: pd.DataFrame, requested_cutoffs: pd.Series) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build synchronized, outcome-independent A5 snapshots from completed bars.

    ``bars`` is the full eligible symbol universe, not the action population.
    Only exact completed hourly cutoffs requested by action rows are retained.
    Twelve causal duration buckets approximate the 1--12h entry-to-action
    interval without ever looking after a cutoff.
    """
    required = {"ts", "symbol", "close", "volume"}
    if required.difference(bars.columns):
        raise ValueError(f"market bars lack {sorted(required.difference(bars.columns))}")
    frame = bars.loc[:, list(required)].copy()
    frame["ts"] = pd.to_datetime(frame.ts, utc=True, errors="raise")
    frame = frame.sort_values(["symbol", "ts"], kind="stable").drop_duplicates(["symbol", "ts"], keep="last")
    symbol = frame.symbol
    log_close = np.log(pd.to_numeric(frame.close, errors="coerce").where(lambda x: x > 0))
    volume = pd.to_numeric(frame.volume, errors="coerce").where(lambda x: x >= 0)
    one_ret = log_close.groupby(symbol, observed=True).diff()
    market_one = one_ret.groupby(frame.ts, observed=True).transform("median")
    rolling_cov = one_ret.groupby(symbol, observed=True).transform(lambda x: x.rolling(24, min_periods=8).cov(market_one.loc[x.index]))
    rolling_var = market_one.groupby(symbol, observed=True).transform(lambda x: x.rolling(24, min_periods=8).var())
    frame["market_beta"] = rolling_cov / rolling_var.replace(0, np.nan)
    volume_ratio = volume / volume.groupby(symbol, observed=True).transform(lambda x: x.rolling(24, min_periods=3).mean()).replace(0, np.nan)
    frame["candidate_volume_rank"] = volume_ratio.groupby(frame.ts, observed=True).rank(pct=True)
    frame["volume_breadth"] = volume_ratio.groupby(frame.ts, observed=True).transform(lambda x: (x > x.median()).mean())
    frame["vol24"] = one_ret.groupby(symbol, observed=True).transform(lambda x: np.sqrt(x.pow(2).rolling(24, min_periods=3).mean()))
    frame["volatility_dispersion"] = frame.vol24.groupby(frame.ts, observed=True).transform("std")
    for horizon in range(1, 13):
        ret = log_close.groupby(symbol, observed=True).diff(horizon)
        frame[f"ret_{horizon}"] = ret
        frame[f"market_{horizon}"] = ret.groupby(frame.ts, observed=True).transform("median")
        frame[f"breadth_{horizon}"] = (ret > 0).groupby(frame.ts, observed=True).transform("mean")
        frame[f"dispersion_{horizon}"] = ret.groupby(frame.ts, observed=True).transform("std")
        frame[f"rank_{horizon}"] = ret.groupby(frame.ts, observed=True).rank(pct=True)
    cutoffs = pd.DatetimeIndex(pd.to_datetime(requested_cutoffs, utc=True).unique())
    frame = frame.loc[frame.ts.isin(cutoffs)].copy()
    grouped = frame.groupby("ts", observed=True)
    membership = grouped.symbol.agg(lambda x: "\n".join(sorted(set(map(str, x))))).reset_index(name="members")
    membership["eligible_universe_membership_sha256"] = membership.members.map(lambda x: hashlib.sha256(x.encode()).hexdigest())
    membership["eligible_universe_size"] = membership.members.map(lambda x: len(x.splitlines()) if x else 0)
    return frame, membership.drop(columns="members")


def latest_completed_hour_open(timestamps: pd.Series) -> pd.Series:
    """Bar-open timestamp of the latest hourly bar closed by each decision."""
    values = pd.to_datetime(timestamps, utc=True, errors="raise")
    return values.dt.floor("h") - pd.Timedelta(hours=1)


def join_market_context_features(actions: pd.DataFrame, snapshots: pd.DataFrame, membership: pd.DataFrame) -> pd.DataFrame:
    """Join a frozen action row to its latest completed market snapshot."""
    out = actions.copy()
    out["market_source_bar_open_ts"] = latest_completed_hour_open(out.action_decision_ts)
    out["market_feature_available_ts"] = out.market_source_bar_open_ts + pd.Timedelta(hours=1)
    out["market_entry_source_bar_open_ts"] = latest_completed_hour_open(out.entry_ts)
    out["market_horizon_hours"] = ((out.market_source_bar_open_ts - out.market_entry_source_bar_open_ts) / pd.Timedelta(hours=1)).clip(1, 12).astype(int)
    base = ["ts", "symbol", "market_beta", "candidate_volume_rank", "volume_breadth", "volatility_dispersion"]
    wide = snapshots[base + [f"{stem}_{h}" for stem in ("ret", "market", "breadth", "dispersion", "rank") for h in range(1, 13)]].rename(columns={"ts": "market_source_bar_open_ts", "symbol": "source_symbol"})
    out = out.merge(wide, on=["market_source_bar_open_ts", "source_symbol"], how="left", validate="many_to_one")
    idx = np.arange(len(out)); horizon = out.market_horizon_hours.to_numpy(int)
    def choose(stem: str) -> np.ndarray:
        matrix = out[[f"{stem}_{h}" for h in range(1, 13)]].to_numpy(float)
        return matrix[idx, horizon - 1]
    out["market_return_since_entry"] = choose("market")
    out["market_recent_action_return"] = out["market_1"]
    out["return_breadth"] = choose("breadth")
    out["return_dispersion"] = choose("dispersion")
    out["candidate_cross_sectional_return_rank"] = choose("rank")
    asset_return = choose("ret")
    out["candidate_residual_return_vs_market"] = asset_return - out.market_return_since_entry
    out["asset_move_vs_market_move"] = _div(asset_return, out.market_return_since_entry)
    sign = out.side.map({"long": 1.0, "short": -1.0})
    out["side_aligned_breadth"] = np.where(sign > 0, out.return_breadth, 1 - out.return_breadth)
    out["breadth_confirmation"] = np.where(asset_return * sign > 0, out.side_aligned_breadth, -out.side_aligned_breadth)
    out["isolated_move_indicator"] = (out.candidate_residual_return_vs_market.abs() > out.return_dispersion).astype(float)
    out["leader_laggard_status"] = 2 * out.candidate_cross_sectional_return_rank - 1
    entry_stats = snapshots[["ts", "symbol", "breadth_1", "dispersion_1"]].rename(columns={"ts": "market_entry_source_bar_open_ts", "symbol": "source_symbol", "breadth_1": "entry_return_breadth_1h", "dispersion_1": "entry_return_dispersion_1h"})
    out = out.merge(entry_stats, on=["market_entry_source_bar_open_ts", "source_symbol"], how="left", validate="many_to_one")
    out["change_in_breadth_since_entry"] = out.return_breadth - out.entry_return_breadth_1h
    out["change_in_dispersion_since_entry"] = out.return_dispersion - out.entry_return_dispersion_1h
    out = out.merge(membership.rename(columns={"ts": "market_source_bar_open_ts"}), on="market_source_bar_open_ts", how="left", validate="many_to_one")
    out["eligible_universe_size"] = out.eligible_universe_size.astype("Int64")
    drop = [f"{stem}_{h}" for stem in ("ret", "market", "breadth", "dispersion", "rank") for h in range(1, 13)]
    return out.drop(columns=[*drop, "entry_return_breadth_1h", "entry_return_dispersion_1h"])


def add_a9_composites(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["path_efficiency_x_volume_persistence"] = out.path_efficiency * out.volume_persistence
    out["path_efficiency_x_breadth_confirmation"] = out.path_efficiency * out.breadth_confirmation
    out["return_acceleration_x_wick_rejection"] = out.return_acceleration_into_clear * out.latest_side_aligned_wick_rejection
    out["volume_climax_x_low_path_efficiency"] = out.volume_climax * (1 - out.path_efficiency.clip(0, 1))
    out["volatility_climax_x_wick_rejection"] = out.volatility_shock_magnitude * out.latest_side_aligned_wick_rejection
    out["isolated_move_x_volume_climax"] = out.isolated_move_indicator * out.volume_climax
    out["giveback_x_time_since_mfe"] = out.giveback_from_observed_mfe_bps * out.time_since_observed_mfe_minutes
    out["time_to_clear_x_path_efficiency"] = out.time_to_clear_minutes * out.path_efficiency
    return out
