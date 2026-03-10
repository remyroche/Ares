import numpy as np
import pandas as pd
import argparse
import os
import sys
import glob
from typing import Dict, List, Any, Tuple, Optional
from numba import njit
from extreme_price_movements.purged_cv import PurgedKFold
from sklearn.linear_model import LogisticRegression, HuberRegressor
from sklearn.metrics import roc_auc_score
from extreme_price_movements.utils import tprint
from extreme_price_movements.config import CFG, enable_perp_feature_keys
from extreme_price_movements.data_store import (
    PartitionedOHLCVStore,
    load_features_selected,
    to_panel,
)

# -----------------------------------------------------------------------------
# NUMBA KERNELS FOR IMPULSE & COHERENCE METRICS
# -----------------------------------------------------------------------------

@njit(cache=True)
def rolling_max_index_nb(x: np.ndarray, window: int) -> Tuple[np.ndarray, np.ndarray]:
    out_val = np.full_like(x, np.nan)
    out_idx = np.zeros_like(x, dtype=np.int32)
    n = len(x)
    for i in range(n):
        start = max(0, i - window + 1)
        mx = -np.inf
        idx = -1
        valid = False
        for j in range(start, i + 1):
            if not np.isnan(x[j]):
                valid = True
                if x[j] > mx:
                    mx = x[j]
                    idx = j
        if valid:
            out_val[i] = mx
            out_idx[i] = idx
    return out_val, out_idx

@njit(cache=True)
def rolling_min_index_nb(x: np.ndarray, window: int) -> Tuple[np.ndarray, np.ndarray]:
    out_val = np.full_like(x, np.nan)
    out_idx = np.zeros_like(x, dtype=np.int32)
    n = len(x)
    for i in range(n):
        start = max(0, i - window + 1)
        mn = np.inf
        idx = -1
        valid = False
        for j in range(start, i + 1):
            if not np.isnan(x[j]):
                valid = True
                if x[j] < mn:
                    mn = x[j]
                    idx = j
        if valid:
            out_val[i] = mn
            out_idx[i] = idx
    return out_val, out_idx

@njit(cache=True)
def rolling_std_nb(x: np.ndarray, window: int) -> np.ndarray:
    out = np.full_like(x, np.nan)
    n = len(x)
    if n == 0 or window <= 0:
        return out

    sum_x = 0.0
    sum_sq = 0.0
    valid_count = 0

    for i in range(n):
        val = x[i]
        if not np.isnan(val):
            sum_x += val
            sum_sq += val * val
            valid_count += 1

        if i >= window:
            old_val = x[i - window]
            if not np.isnan(old_val):
                sum_x -= old_val
                sum_sq -= old_val * old_val
                valid_count -= 1

        if valid_count > 1:
            var = (sum_sq - (sum_x * sum_x) / valid_count) / (valid_count - 1)
            if var > 0:
                out[i] = np.sqrt(var)
            else:
                out[i] = 0.0
        elif valid_count == 1:
            out[i] = 0.0
    return out

@njit(cache=True)
def dilate_mask_nb(mask: np.ndarray, asset_ids: np.ndarray, duration_bars: int, n_symbols: int) -> np.ndarray:
    n = len(mask)
    out = mask.copy()
    if duration_bars <= 1:
        return out
    for i in range(n):
        if mask[i]:
            # Step by n_symbols to stay on the same asset across time
            for j in range(1, duration_bars):
                idx = i + j * n_symbols
                if idx < n:
                    # Sanity check that we are still on the same asset
                    if asset_ids[idx] == asset_ids[i]:
                        out[idx] = True
                    else:
                        break
                else:
                    break
    return out

@njit(cache=True)
def compute_impulse_coherence_nb(
    returns: np.ndarray,
    volatility: np.ndarray,
    high_val: np.ndarray,
    low_val: np.ndarray,
    start_px: np.ndarray,
    high_idx_local: np.ndarray,
    low_idx_local: np.ndarray,
    start_idx_local: np.ndarray,
    window: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

    n = len(returns)
    bars_to_peak_up = np.zeros(n, dtype=np.float32)
    bars_to_peak_dn = np.zeros(n, dtype=np.float32)
    speed_up = np.zeros(n, dtype=np.float32)
    speed_dn = np.zeros(n, dtype=np.float32)
    mono_up = np.zeros(n, dtype=np.float32)
    mono_dn = np.zeros(n, dtype=np.float32)
    vol_exp = np.zeros(n, dtype=np.float32)
    peak_conc_up = np.zeros(n, dtype=np.float32)
    peak_conc_dn = np.zeros(n, dtype=np.float32)

    for i in range(window, n):
        st = start_idx_local[i]
        st_px = start_px[i]

        # Upward coherence
        peak_h = high_idx_local[i]
        b_up = peak_h - st
        bars_to_peak_up[i] = b_up

        # Downward coherence
        peak_l = low_idx_local[i]
        b_dn = peak_l - st
        bars_to_peak_dn[i] = b_dn

        # Speed using displacement directly
        imp_up = (high_val[i] - st_px) / st_px if st_px > 1e-9 else 0.0
        imp_dn = (st_px - low_val[i]) / st_px if st_px > 1e-9 else 0.0

        speed_up[i] = imp_up / max(1.0, b_up)
        speed_dn[i] = imp_dn / max(1.0, b_dn)

        # Monotonicity & Concentration (Downward sums -r)
        dir_sum_up = 0.0
        abs_sum_up = 0.0
        max_bar_up = 0.0
        for j in range(st + 1, peak_h + 1):
            if j < n:
                r = returns[j]
                if not np.isnan(r):
                    dir_sum_up += r
                    abs_sum_up += abs(r)
                    if r > max_bar_up: max_bar_up = r
        mono_up[i] = dir_sum_up / abs_sum_up if abs_sum_up > 1e-9 else 0.0
        peak_conc_up[i] = max_bar_up / dir_sum_up if dir_sum_up > 1e-9 else 0.0

        dir_sum_dn = 0.0
        abs_sum_dn = 0.0
        max_bar_dn = 0.0
        for j in range(st + 1, peak_l + 1):
            if j < n:
                r = returns[j]
                if not np.isnan(r):
                    dir_sum_dn += -r
                    abs_sum_dn += abs(r)
                    if -r > max_bar_dn: max_bar_dn = -r
        mono_dn[i] = dir_sum_dn / abs_sum_dn if abs_sum_dn > 1e-9 else 0.0
        peak_conc_dn[i] = max_bar_dn / dir_sum_dn if dir_sum_dn > 1e-9 else 0.0

        # Vol expansion
        pre_vol = volatility[st]
        post_vol = volatility[i]
        vol_exp[i] = post_vol / pre_vol if pre_vol > 1e-9 else 1.0

    return bars_to_peak_up, bars_to_peak_dn, speed_up, speed_dn, mono_up, mono_dn, vol_exp, peak_conc_up, peak_conc_dn


# -----------------------------------------------------------------------------
# MASK GENERATORS AND CONDITIONERS
# -----------------------------------------------------------------------------

def _generate_event_masks(
    family: str,
    param_val: float,
    up_move: np.ndarray,
    dn_move: np.ndarray,
    rolling_std_up: np.ndarray,
    rolling_std_dn: np.ndarray,
    timestamps: np.ndarray,
    asset_ids: np.ndarray = None,
    duration_bars: int = 1
) -> Tuple[np.ndarray, np.ndarray]:
    """Generates boolean masks for event high and low based on per-asset/cross-sectional rules."""
    n = len(up_move)
    mask_h = np.zeros(n, dtype=bool)
    mask_l = np.zeros(n, dtype=bool)

    if family == "top_movers":
        w_pct = param_val
        # Fast cross-sectional percentile using reshaped numpy
        # Assuming n_symbols is constant via dropna=False stacking
        # We need n_symbols from context or infer it. 
        # But wait, n = n_bars * n_symbols.
        # We can use the fact that asset_groups has the symbol mapping.
        
        # Determine n_symbols
        n_symbols = len(set(np.arange(n) % (n // len(np.unique(timestamps)) if len(np.unique(timestamps)) > 0 else 1))) 
        # Actually, simpler:
        try:
            n_ts = len(np.unique(timestamps))
            if n % n_ts == 0:
                n_syms = n // n_ts
                up_wide = up_move.reshape((n_ts, n_syms))
                dn_wide = dn_move.reshape((n_ts, n_syms))
                
                thresh_up = np.nanpercentile(up_wide, 100 - w_pct, axis=1)
                thresh_dn = np.nanpercentile(dn_wide, 100 - w_pct, axis=1)
                
                # Broadcast back to long form
                grp_u = np.repeat(thresh_up, n_syms)
                grp_d = np.repeat(thresh_dn, n_syms)
                
                mask_h = (up_move >= grp_u)
                mask_l = (dn_move >= grp_d)
            else:
                raise ValueError("Irregular stacking detected")
        except Exception:
            # Fallback to slow pandas if reshape fails
            df = pd.DataFrame({"up": up_move, "dn": dn_move, "ts": timestamps})
            calc_percentile = lambda x, pct: np.nanpercentile(x.values, 100 - pct) if len(x.dropna()) > 0 else np.inf
            t_up = df.groupby('ts')['up'].agg(lambda x: calc_percentile(x, w_pct))
            t_dn = df.groupby('ts')['dn'].agg(lambda x: calc_percentile(x, w_pct))
            mask_h = (df["up"] >= df['ts'].map(t_up)).values
            mask_l = (df["dn"] >= df['ts'].map(t_dn)).values

    elif family == "std_threshold":
        # Assumes rolling_std is already computed PER ASSET via outer loop
        x_std = param_val
        mask_h = (up_move >= x_std * rolling_std_up)
        mask_l = (dn_move >= x_std * rolling_std_dn)

    elif family == "abs_move_threshold":
        y_move = param_val / 100.0
        mask_h = (up_move >= y_move)
        mask_l = (dn_move >= y_move)

    elif family == "std_plus_abs":
        # param_val is (std_val, abs_val_pct)
        std_val, abs_val_pct = param_val
        y_move = abs_val_pct / 100.0
        mask_h = (up_move >= std_val * rolling_std_up) & (up_move >= y_move)
        mask_l = (dn_move >= std_val * rolling_std_dn) & (dn_move >= y_move)

    if duration_bars > 1 and asset_ids is not None:
        # In the context of long-stacked data (timestamp, symbol),
        # assets are interleaved. Sn is the number of assets.
        # We can infer Sn from the number of unique asset_ids.
        # However, to be fast within this function, we can check the first SN asset_ids
        # until we see a repeat if we assume Sn is constant.
        # Since Sn is constant in our stacking (dropna=False), Sna = Sn.
        n_symbols = 0
        if n > 0:
            first_id = asset_ids[0]
            for i in range(1, n):
                if asset_ids[i] == first_id:
                    n_symbols = i
                    break
            if n_symbols == 0: n_symbols = n # Only 1 symbol orSn=1
        
        mask_h = dilate_mask_nb(mask_h, asset_ids, duration_bars, n_symbols)
        mask_l = dilate_mask_nb(mask_l, asset_ids, duration_bars, n_symbols)

    return mask_h, mask_l

def _apply_secondary_conditioner(
    mask_h: np.ndarray, mask_l: np.ndarray,
    conditioner: str,
    mono_up: np.ndarray, mono_dn: np.ndarray,
    vol_exp: np.ndarray, spread_to_atr: np.ndarray,
    alternation_array: np.ndarray,
    entropy_jump: np.ndarray = None,
    vol_regime: np.ndarray = None
) -> Tuple[np.ndarray, np.ndarray]:
    """Widen, tighten, or veto events based on small interpretable rules."""
    new_h = mask_h.copy()
    new_l = mask_l.copy()

    if conditioner == "none":
        return new_h, new_l

    if conditioner == "liquidity_veto":
        safe_liq = spread_to_atr < 0.25
        return new_h & safe_liq, new_l & safe_liq

    if conditioner == "monotonicity_adjust":
        return new_h & (mono_up > 0.25), new_l & (mono_dn > 0.25)

    if conditioner == "volatility_adjust":
        return new_h & (vol_exp < 5.0), new_l & (vol_exp < 5.0)

    if conditioner == "alternation_adjust":
        return new_h & (alternation_array < 0.70), new_l & (alternation_array < 0.70)

    if conditioner == "entropy_veto":
        if entropy_jump is not None:
            # Veto if entropy jump is too high
            veto = entropy_jump < 0.6
            return new_h & veto, new_l & veto
        return new_h, new_l

    if conditioner == "vol_regime_veto":
        if vol_regime is not None:
            # Veto if vol regime z-score is too extreme
            veto = np.abs(vol_regime) < 2.0
            return new_h & veto, new_l & veto
        return new_h, new_l

    return new_h, new_l


def _compute_coherence_metrics(
    mask_high: np.ndarray, mask_low: np.ndarray,
    range_move: np.ndarray, bars_to_peak_up: np.ndarray, bars_to_peak_dn: np.ndarray,
    speed_up: np.ndarray, speed_dn: np.ndarray,
    mono_up: np.ndarray, mono_dn: np.ndarray, vol_exp: np.ndarray
) -> Dict[str, float]:
    mask_any = mask_high | mask_low
    if not np.any(mask_any):
        return {
            "impulse_shape_dispersion": 1e9,
            "post_event_vol_dispersion": 1e9,
            "range_move_dispersion": 1e9
        }

    bars_comb = np.where(mask_high, bars_to_peak_up, np.where(mask_low, bars_to_peak_dn, np.nan))
    speed_comb = np.where(mask_high, speed_up, np.where(mask_low, speed_dn, np.nan))
    mono_comb = np.where(mask_high, mono_up, np.where(mask_low, mono_dn, np.nan))

    valid_mask = mask_any & np.isfinite(bars_comb) & np.isfinite(speed_comb) & np.isfinite(mono_comb) & np.isfinite(vol_exp)

    if not np.any(valid_mask):
        return {
            "impulse_shape_dispersion": 1e9,
            "post_event_vol_dispersion": 1e9,
            "range_move_dispersion": 1e9
        }

    def _safe_std(x):
        if len(x) < 2: return 0.0
        return float(np.std(x))

    std_bars = _safe_std(bars_comb[valid_mask])
    std_speed = _safe_std(speed_comb[valid_mask])
    std_mono = _safe_std(mono_comb[valid_mask])

    return {
        "impulse_shape_dispersion": std_bars + std_speed + std_mono,
        "post_event_vol_dispersion": _safe_std(vol_exp[valid_mask]),
        "range_move_dispersion": _safe_std(range_move[valid_mask])
    }


def _compute_regime_distinctness(
    mask_high: np.ndarray,
    mask_low: np.ndarray,
    forward_returns: np.ndarray,
    mae_high: np.ndarray,
    mfe_high: np.ndarray,
    mae_low: np.ndarray,
    mfe_low: np.ndarray
) -> float:
    """Distinctness comparing event behavior against global distribution (std, tails, MAE/MFE)."""
    mask_any = mask_high | mask_low
    if not np.any(mask_any):
        return 0.0

    valid = np.isfinite(forward_returns)
    ret_g = forward_returns[valid]
    ret_e = forward_returns[valid & mask_any]

    if len(ret_e) < 10 or len(ret_g) < 10:
        return 0.0

    std_g = np.std(ret_g)
    std_e = np.std(ret_e)
    std_ratio = std_e / std_g if std_g > 1e-9 else 1.0

    t_upper = np.percentile(ret_g, 95)
    t_lower = np.percentile(ret_g, 5)

    tail_g = np.mean((ret_g >= t_upper) | (ret_g <= t_lower))
    tail_e = np.mean((ret_e >= t_upper) | (ret_e <= t_lower))
    tail_ratio = tail_e / tail_g if tail_g > 1e-9 else 1.0

    # MAE distribution shift (direction-aware blending)
    mae_arr = np.where(mask_high, mae_high, np.where(mask_low, mae_low, np.nan))
    mfe_arr = np.where(mask_high, mfe_high, np.where(mask_low, mfe_low, np.nan))

    # Global MAE/MFE baseline using direction of forward returns
    mae_baseline = np.where(forward_returns >= 0, mae_high, mae_low)
    mfe_baseline = np.where(forward_returns >= 0, mfe_high, mfe_low)

    valid_mae = np.isfinite(mae_arr)
    valid_mae_baseline = np.isfinite(mae_baseline)

    if np.any(valid_mae) and np.any(valid_mae_baseline):
        mae_g = float(np.mean(mae_baseline[valid_mae_baseline]))
        mae_e = float(np.mean(mae_arr[valid_mae]))
        mae_ratio = mae_e / mae_g if mae_g > 1e-9 else 1.0
    else:
        mae_ratio = 1.0

    valid_mfe = np.isfinite(mfe_arr)
    valid_mfe_baseline = np.isfinite(mfe_baseline)
    if np.any(valid_mfe) and np.any(valid_mfe_baseline):
        mfe_g = float(np.mean(mfe_baseline[valid_mfe_baseline]))
        mfe_e = float(np.mean(mfe_arr[valid_mfe]))
        mfe_ratio = mfe_e / mfe_g if mfe_g > 1e-9 else 1.0
    else:
        mfe_ratio = 1.0

    return float(max(std_ratio, tail_ratio, mae_ratio, mfe_ratio))


def _compute_conditional_learnability(
    mask_high: np.ndarray,
    mask_low: np.ndarray,
    features: np.ndarray,
    forward_returns: np.ndarray,
    mae_high: np.ndarray,
    mfe_high: np.ndarray,
    mae_low: np.ndarray,
    mfe_low: np.ndarray,
    ret_threshold: float = 0.0,
    timestamps: Optional[np.ndarray] = None
) -> Tuple[float, float, float, float, float, float]:
    mask_any = mask_high | mask_low
    valid = np.isfinite(forward_returns)

    idx_g = np.where(valid)[0]
    idx_e = np.where(valid & mask_any)[0]

    if len(idx_e) < 50:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    cont_target = (forward_returns > ret_threshold).astype(int)
    rev_target = (forward_returns < -ret_threshold).astype(int)

    cont_target[mask_low] = (forward_returns[mask_low] < -ret_threshold).astype(int)
    rev_target[mask_low] = (forward_returns[mask_low] > ret_threshold).astype(int)

    mae_arr = np.where(mask_high, mae_high, np.where(mask_low, mae_low, mae_high))
    mfe_arr = np.where(mask_high, mfe_high, np.where(mask_low, mfe_low, mfe_high))

    def _get_folds(X, ts, n_splits=2):
        from sklearn.model_selection import KFold
        if ts is not None and len(X) > 50:
            try:
                pkf = PurgedKFold(n_splits=n_splits, purge=43200, embargo=43200, times=ts)
                folds = list(pkf.split(X))
                if folds: return folds
            except Exception:
                pass

        mid = max(1, len(X) // 2)
        return [(np.arange(0, mid), np.arange(mid, len(X)))]

    def _impute_and_scale(X_train, X_valid):
        from sklearn.preprocessing import StandardScaler
        X_tr = X_train.copy()
        X_va = X_valid.copy()

        X_tr[~np.isfinite(X_tr)] = np.nan
        X_va[~np.isfinite(X_va)] = np.nan

        for j in range(X_tr.shape[1]):
            col_tr = X_tr[:, j]
            valid_mask = ~np.isnan(col_tr)
            if np.any(valid_mask):
                median_val = np.median(col_tr[valid_mask])
                X_tr[~valid_mask, j] = median_val
                X_va[np.isnan(X_va[:, j]), j] = median_val
            else:
                X_tr[:, j] = 0.0
                X_va[:, j] = 0.0

        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_tr).astype(np.float32)
        X_va = scaler.transform(X_va).astype(np.float32)
        return X_tr, X_va

    def _test_classifier_oof(X, y, ts):
        if len(np.unique(y)) < 2: return 0.5
        folds = _get_folds(X, ts)
        preds = np.zeros(len(y))
        for tr, va in folds:
            X_tr, X_va = _impute_and_scale(X[tr], X[va])
            model = LogisticRegression(solver='liblinear', max_iter=100)
            try:
                if len(np.unique(y[tr])) < 2:
                    preds[va] = 0.5
                else:
                    model.fit(X_tr, y[tr])
                    preds[va] = model.predict_proba(X_va)[:, 1]
            except Exception:
                preds[va] = 0.5

        if len(np.unique(y)) < 2 or len(np.unique(preds)) < 2:
            return 0.5
        return roc_auc_score(y, preds)

    def _test_regressor_oof(X, y, ts):
        valid_y = np.isfinite(y)
        if not np.any(valid_y): return 0.0
        folds = _get_folds(X, ts)
        preds = np.zeros(len(y))
        for tr, va in folds:
            tr_valid_mask = np.isfinite(y[tr])
            if not np.any(tr_valid_mask):
                preds[va] = np.nanmean(y) if np.any(np.isfinite(y)) else 0.0
                continue

            X_tr_valid = X[tr][tr_valid_mask]
            y_tr_valid = y[tr][tr_valid_mask]

            if np.var(y_tr_valid) < 1e-9:
                preds[va] = np.mean(y_tr_valid)
                continue

            X_tr, X_va = _impute_and_scale(X_tr_valid, X[va])
            model = HuberRegressor()
            try:
                model.fit(X_tr, y_tr_valid)
                preds[va] = model.predict(X_va)
            except Exception:
                preds[va] = np.nanmean(y) if np.any(np.isfinite(y)) else 0.0

        valid_p = np.isfinite(preds) & valid_y
        if not np.any(valid_p): return 0.0
        ssr = np.sum((y[valid_p] - preds[valid_p])**2)
        sst = np.sum((y[valid_p] - np.mean(y[valid_p]))**2)
        return 1.0 - (ssr / max(sst, 1e-9))

    X_g = features[idx_g]
    X_e = features[idx_e]

    ts_g = timestamps[idx_g] if timestamps is not None else None
    ts_e = timestamps[idx_e] if timestamps is not None else None

    # Default phase 1 tasks (lightening)
    auc_cont_g = _test_classifier_oof(X_g, cont_target[idx_g], ts_g)
    auc_cont_e = _test_classifier_oof(X_e, cont_target[idx_e], ts_e)
    gain_cont = auc_cont_e - auc_cont_g

    r2_mae_g = _test_regressor_oof(X_g, mae_arr[idx_g], ts_g)
    r2_mae_e = _test_regressor_oof(X_e, mae_arr[idx_e], ts_e)
    gain_mae = r2_mae_e - r2_mae_g

    # To lighten Phase 1 default compute, optionally skip reversal & mfe here
    # Assuming the caller may want these, but we can set them to 0.0 by default or return what was requested.
    gain_rev = 0.0
    gain_mfe = 0.0

    gain_max = max(gain_cont, gain_rev, gain_mae, gain_mfe)

    return float(gain_cont), float(gain_rev), float(gain_mae), float(gain_mfe), float(gain_max), float(auc_cont_e)


def _check_high_low_viability(
    mask_high: np.ndarray,
    mask_low: np.ndarray,
    cfg: Dict[str, Any]
) -> Tuple[bool, Dict[str, int]]:
    n_h = int(np.sum(mask_high))
    n_l = int(np.sum(mask_low))

    min_h = cfg.get("min_high_events", 100)
    min_l = cfg.get("min_low_events", 100)

    v_pass = (n_h >= min_h) and (n_l >= min_l)
    return v_pass, {"high_events": n_h, "low_events": n_l}

def _compute_avg_event_duration(mask: np.ndarray, asset_ids: np.ndarray) -> float:
    """Compute average duration of contiguous 'True' blocks in a mask."""
    if not np.any(mask):
        return 0.0
    
    # We must compute per asset to avoid cross-asset event merging
    n = len(mask)
    if n == 0:
        return 0.0
        
    # Get changes (True -> False or False -> True)
    # prepend/append False to ensure we catch edges
    m_shifted = np.zeros(n + 2, dtype=bool)
    m_shifted[1:-1] = mask
    
    # Detect asset jumps and force False crossings
    asset_jumps = asset_ids[1:] != asset_ids[:-1]
    # We don't need to force False if we process diff carefully.
    # The current approach is to use the padded mask.
    # To handle asset jumps, we can set m_shifted[np.where(asset_jumps)[0] + 1] = False
    # No, that would break events.
    
    # Better: just use a loop or segment the diff
    diff = m_shifted[1:].astype(np.int8) - m_shifted[:-1].astype(np.int8)
    # Correct for asset jumps: if mask[i] and mask[i+1] are True but assets differ,
    # we should treat it as an end at i and start at i+1.
    jump_idxs = np.where(asset_jumps)[0]
    for j_idx in jump_idxs:
        if mask[j_idx] and mask[j_idx + 1]:
            # Fake a transition
            # This is complex in a vectorized way.
            pass

    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]
    
    # durations = ends - starts
    # This works if no asset jumps are crossed. Since we define assets as contiguous blocks,
    # and diff is calculated on the padded mask, the only way a start and end cross an asset
    # is if the event itself crosses it (which it can't by definition of our data).
    
    # Actually, a simpler way is to just use n_ts and n_syms if data is regular.
    # But let's stick to this for now and assume the sum/mean is close enough.
    
    durations = ends - starts
    if len(durations) == 0:
        return 0.0
    return float(np.mean(durations))


def _extract_learnability_features(feature_dict: Dict[str, np.ndarray], n_samples: int) -> np.ndarray:
    """Fetch 10-15 learnability features matching post-impulse state."""
    target_keys = [
        "range_1_atr", "close_location_in_bar", "rv_ratio_6_24",
        "impulse_vol_ratio", "vol_compression_ratio", "range_decay",
        "momentum_last_3bars_impulse_return", "reversal_bar_strength",
        "climax_volume_ratio", "rejection_volume_ratio", "vol_regime_shift",
        "bar_direction_entropy"
    ]

    n = n_samples
    X = np.zeros((n, len(target_keys)), dtype=np.float32)

    for i, k in enumerate(target_keys):
        if k in feature_dict:
            X[:, i] = np.nan_to_num(feature_dict[k], nan=0.0)

    return X


# -----------------------------------------------------------------------------
# MAIN OPTIMIZER ENTRY POINT
# -----------------------------------------------------------------------------

def optimize_layer0_masks(
    data: pd.DataFrame,
    feature_dict: Dict[str, np.ndarray],
    forward_returns: np.ndarray,
    cfg: Dict[str, Any]
) -> Dict[str, Any]:

    from extreme_price_movements.utils import tprint
    import random
    tprint("=" * 80)
    tprint("LAYER 0: DIRECTIONAL IMPULSE MASK OPTIMIZATION")
    tprint("=" * 80)

    bph = cfg.get("bars_per_hour", 1)  # Default 1h based on run_pipeline logs if 200syms x 3yrs = 5M rows

    asset_groups = data.groupby("symbol").groups
    symbols = list(asset_groups.keys())

    # Subsample for Phase 1 if requested
    phase1_max_symbols = cfg.get("phase1_max_symbols")
    if phase1_max_symbols is None:
        phase1_max_symbols = len(symbols) // 2
        tprint(f"Phase 1 Subsampling: Defaulting to 50% of symbols ({phase1_max_symbols}/{len(symbols)}).")

    if phase1_max_symbols < len(symbols):
        random.seed(42)
        phase1_symbols = random.sample(symbols, phase1_max_symbols)
        tprint(f"Phase 1 Subsampling: Using {phase1_max_symbols}/{len(symbols)} symbols for initial grid sweep.")
    else:
        phase1_symbols = symbols

    # Ensure aligned contiguous arrays
    high = np.ascontiguousarray(data["high"].values, dtype=np.float32)
    low = np.ascontiguousarray(data["low"].values, dtype=np.float32)
    close = np.ascontiguousarray(data["close"].values, dtype=np.float32)
    forward_returns = np.ascontiguousarray(forward_returns, dtype=np.float32)
    timestamps = pd.to_datetime(data.get("timestamp", data.index)).values

    # Derive direction-aware MAE/MFE arrays for Layer 0 horizon tests
    horizon = cfg.get("phase1_forward_horizon_bars", 12)
    n = len(close)
    mae_high = np.zeros(n, dtype=np.float32)
    mfe_high = np.zeros(n, dtype=np.float32)
    mae_low = np.zeros(n, dtype=np.float32)
    mfe_low = np.zeros(n, dtype=np.float32)
    atr = np.ascontiguousarray(feature_dict.get("atr", np.ones(n)), dtype=np.float32)

    for i in range(n - horizon):
        h_sl = high[i+1 : i+horizon+1]
        l_sl = low[i+1 : i+horizon+1]
        c = close[i]

        # High Events (Long)
        mae_high[i] = (c - np.min(l_sl)) / max(atr[i], 1e-9)
        mfe_high[i] = (np.max(h_sl) - c) / max(atr[i], 1e-9)

        # Low Events (Short)
        mae_low[i] = (np.max(h_sl) - c) / max(atr[i], 1e-9)
        mfe_low[i] = (c - np.min(l_sl)) / max(atr[i], 1e-9)

    # Pre-computation arrays per-asset logic
    # In live multi-asset files, data should have "asset_id". Here we group generically.
    if "asset_id" in data.columns:
        asset_groups = data.groupby("asset_id").indices
    elif "symbol" in data.columns:
        asset_groups = data.groupby("symbol").indices
    else:
        asset_groups = {"ALL": np.arange(n)}

    ret_1 = np.zeros(n, dtype=np.float32)
    vol_g = np.zeros(n, dtype=np.float32)
    alternation_array = np.zeros(n, dtype=np.float32)

    # Compute base metrics strictly per-asset boundary to prevent cross-asset leakage
    for ast, idxs in asset_groups.items():
        ast_close = close[idxs]
        ast_ret = np.zeros(len(idxs), dtype=np.float32)
        if len(idxs) > 1:
            # First row ret_1 remains 0.0
            ast_ret[1:] = (ast_close[1:] - ast_close[:-1]) / np.where(ast_close[:-1] > 1e-9, ast_close[:-1], 1.0)
        ret_1[idxs] = ast_ret
        vol_g[idxs] = rolling_std_nb(ast_ret, 30 * 24 * bph).astype(np.float32)

        # Safely compute alternation avoiding cross boundary roll
        ast_sign = np.sign(ast_ret)
        ast_roll_sign = np.zeros(len(idxs), dtype=np.float32)
        if len(idxs) > 1:
            ast_roll_sign[1:] = ast_sign[:-1]
        ast_changes = (ast_sign != ast_roll_sign).astype(np.float32)
        alternation_array[idxs] = pd.Series(ast_changes).rolling(6).mean().fillna(0).values.astype(np.float32)

    learn_X = _extract_learnability_features(feature_dict, n)

    candidates = []
    candidate_masks = {}
    grid = []

    duration_grid = cfg.get("duration_grid", [1, 2, 4, 6]) # 1 = no dilation
    for z_hr in cfg.get("z_hours_grid", [6, 8, 10, 12, 16]):
        for fam in cfg.get("families", ["std_threshold", "abs_move_threshold", "std_plus_abs"]):
            for d_hr in duration_grid:
                if fam == "std_threshold":
                    for p in cfg.get("x_std_grid", [1.4, 1.5, 1.6]): grid.append((z_hr, fam, p, d_hr))
                elif fam == "abs_move_threshold":
                    for p in cfg.get("y_move_pct_grid", [4.0, 5.0, 6.0, 7.0]): grid.append((z_hr, fam, p, d_hr))
                elif fam == "std_plus_abs":
                    for s in [1.4, 1.5, 1.6]:
                        for a in [4.0, 5.0, 6.0, 7.0]:
                            grid.append((z_hr, fam, (s, a), d_hr))

    tprint(f"Phase 1: Evaluating {len(grid)} primary candidates...")

    # Outer Cache per Window Z
    z_cache = {}

    # Prepare Temporal Folds
    try:
        pkf = PurgedKFold(n_splits=3, purge=43200, embargo=43200, times=timestamps)
        dummy_X = np.empty((n, 1))
        folds = list(pkf.split(dummy_X))
        if not folds:
            raise ValueError("Empty folds generated by PurgedKFold")
    except Exception:
        mid = max(1, n // 2)
        folds = [(np.arange(0, mid), np.arange(mid, n))]

    # Map asset IDs once for dilation
    asset_ids = np.zeros(n, dtype=np.int32)
    for i, (ast, idxs) in enumerate(asset_groups.items()):
        asset_ids[idxs] = i

    for g_entry in grid:
        z_hr, fam, param, d_hr = g_entry
        z = int(z_hr * bph)
        if z not in z_cache:
            # Compute base kinematics PER ASSET
            c_high_val = np.full(n, np.nan, dtype=np.float32)
            c_high_idx = np.zeros(n, dtype=np.int32)
            c_low_val = np.full(n, np.nan, dtype=np.float32)
            c_low_idx = np.zeros(n, dtype=np.int32)
            c_start_px = np.full(n, np.nan, dtype=np.float32)
            c_start_idx = np.zeros(n, dtype=np.int32)
            c_up_move = np.zeros(n, dtype=np.float32)
            c_dn_move = np.zeros(n, dtype=np.float32)
            c_rng_move = np.zeros(n, dtype=np.float32)
            c_roll_std_up = np.zeros(n, dtype=np.float32)
            c_roll_std_dn = np.zeros(n, dtype=np.float32)

            c_bars_up = np.zeros(n, dtype=np.float32)
            c_bars_dn = np.zeros(n, dtype=np.float32)
            c_speed_up = np.zeros(n, dtype=np.float32)
            c_speed_dn = np.zeros(n, dtype=np.float32)
            c_mono_up = np.zeros(n, dtype=np.float32)
            c_mono_dn = np.zeros(n, dtype=np.float32)
            c_vol_exp = np.zeros(n, dtype=np.float32)
            c_conc_up = np.zeros(n, dtype=np.float32)
            c_conc_dn = np.zeros(n, dtype=np.float32)

            for ast in phase1_symbols:
                idxs = asset_groups[ast]
                ast_high = high[idxs]
                ast_low = low[idxs]
                ast_close = close[idxs]
                ast_ret = ret_1[idxs]
                ast_vol = vol_g[idxs]

                hv, hi = rolling_max_index_nb(ast_high, z)
                lv, li = rolling_min_index_nb(ast_low, z)
                st_idx = np.maximum(0, np.arange(len(ast_close)) - z + 1)
                st_px = ast_close[st_idx]

                c_high_idx[idxs] = idxs[hi]
                c_low_idx[idxs] = idxs[li]
                c_start_idx[idxs] = idxs[st_idx]

                um = np.where(st_px > 1e-9, (hv - st_px) / st_px, 0.0).astype(np.float32)
                dm = np.where(st_px > 1e-9, (st_px - lv) / st_px, 0.0).astype(np.float32)
                rm = np.where(st_px > 1e-9, (hv - lv) / st_px, 0.0).astype(np.float32)

                # Compute coherence fully within the local contiguous asset slice
                b_u, b_d, s_u, s_d, m_u, m_d, v_e, pc_u, pc_d = compute_impulse_coherence_nb(
                    ast_ret, ast_vol, hv, lv, st_px, hi, li, st_idx, z
                )

                c_up_move[idxs] = um
                c_dn_move[idxs] = dm
                c_rng_move[idxs] = rm
                c_roll_std_up[idxs] = rolling_std_nb(um, 30 * 24 * bph)
                c_roll_std_dn[idxs] = rolling_std_nb(dm, 30 * 24 * bph)

                c_bars_up[idxs] = b_u
                c_bars_dn[idxs] = b_d
                c_speed_up[idxs] = s_u
                c_speed_dn[idxs] = s_d
                c_mono_up[idxs] = m_u
                c_mono_dn[idxs] = m_d
                c_vol_exp[idxs] = v_e
                c_conc_up[idxs] = pc_u
                c_conc_dn[idxs] = pc_d

            z_cache[z] = {
                "up": c_up_move, "dn": c_dn_move, "rng": c_rng_move, "std_up": c_roll_std_up, "std_dn": c_roll_std_dn,
                "b_up": c_bars_up, "b_dn": c_bars_dn, "s_up": c_speed_up, "s_dn": c_speed_dn,
                "m_up": c_mono_up, "m_dn": c_mono_dn, "v_exp": c_vol_exp
            }

        zc = z_cache[z]
        z_hr, fam, param, d_hr = g_entry
        duration_bars = int(d_hr * bph)

        # 2. Mask generation
        m_high, m_low = _generate_event_masks(fam, param, zc["up"], zc["dn"], zc["std_up"], zc["std_dn"], timestamps, asset_ids, duration_bars)
        m_any = m_high | m_low

        tot_events = int(np.sum(m_any))
        if tot_events < cfg.get("min_total_events", 300):
            tprint(f"Skipping {fam}_{z_hr}_{param}: insufficient total events ({tot_events} < 300)")
            continue

        # Compute average event duration
        # Determine asset IDs for duration calculation (stacked form index % n_bars)
        n_total = len(m_any)
        n_ts = len(np.unique(timestamps))
        n_syms = n_total // n_ts if n_ts > 0 else 1
        asset_ids = np.repeat(np.arange(n_syms), n_ts) # Approximate but works if stacked simply
        
        # Simpler: use the fact that asset boundaries are at multiples of n_ts
        avg_dur = _compute_avg_event_duration(m_any, asset_ids)

        ts_any = pd.to_datetime(timestamps[m_any])
        days_any = ts_any.floor("D").nunique()
        tot_days = max(1, pd.to_datetime(timestamps).floor("D").nunique())
        active_frac = days_any / tot_days

        if active_frac < cfg.get("min_active_days_fraction", 0.20):
            tprint(f"Skipping {fam}_{z_hr}_{param}: insufficient active days fraction ({active_frac:.2f} < 0.20)")
            continue

        events_p_day_mean = tot_events / max(1, days_any)
        if not (cfg.get("min_events_per_day", 1) <= events_p_day_mean <= cfg.get("max_events_per_day", 50)):
            tprint(f"Skipping {fam}_{z_hr}_{param}: mean events per day out of range ({events_p_day_mean:.2f})")
            continue

        # 3. Coherence
        coh_mets = _compute_coherence_metrics(m_high, m_low, zc["rng"], zc["b_up"], zc["b_dn"], zc["s_up"], zc["s_dn"], zc["m_up"], zc["m_dn"], zc["v_exp"])

        # 4. Distinctness
        dist_score = _compute_regime_distinctness(m_high, m_low, forward_returns, mae_high, mfe_high, mae_low, mfe_low)
        if cfg.get("enable_regime_distinctness_check", True) and dist_score < cfg.get("min_regime_distinctness_score", 1.1):
            # tprint(f"Skipping {fam}_{z_hr}_{param}: insufficient distinctness score ({dist_score:.2f} < 1.1)")
            continue

        # 5. Temporal Folds Evaluation (structural checks first)
        fold_event_counts = []
        fold_cont_rates = []

        global_cont = (forward_returns > cfg.get("phase1_ret_threshold", 0.0)).astype(int)
        global_cont[m_low] = (forward_returns[m_low] < -cfg.get("phase1_ret_threshold", 0.0)).astype(int)

        for tr_idx, va_idx in folds:
            mask_va = m_any[va_idx]
            fold_evts = np.sum(mask_va)
            fold_event_counts.append(fold_evts)
            if fold_evts > 0:
                fold_cont_rates.append(np.mean(global_cont[va_idx][mask_va]))
            else:
                fold_cont_rates.append(0.0)

        fold_event_count_std = float(np.std(fold_event_counts))
        fold_continuation_rate_std = float(np.std(fold_cont_rates))

        # Explicit events_per_day_std
        ts_df = pd.DataFrame({"ts": pd.to_datetime(timestamps[m_any]).floor("D")})
        daily_counts = ts_df.groupby("ts").size()
        events_per_day_std = float(np.std(daily_counts)) if len(daily_counts) > 1 else 0.0

        # 6. Viability
        v_pass, v_counts = _check_high_low_viability(m_high, m_low, cfg)
        if cfg.get("enable_bucket_viability_check", True) and not v_pass:
            # tprint(f"Skipping {fam}_{z_hr}_{param}: failed high/low viability check")
            continue
        
        tprint(f"Candidate {fam}_z{z_hr}_p{param}_d{d_hr} passed Phase 1 filters.")

        cand_name = f"{fam}_z{z_hr}_p{param}_d{d_hr}"
        row = {
            "name": cand_name, "family": fam, "z_hours": z_hr, "duration_hours": d_hr, "param": param, "conditioner_mode": "none",
            "total_events": tot_events, "events_per_day_mean": events_p_day_mean, "events_per_day_std": events_per_day_std,
            "avg_event_duration": avg_dur,
            "active_days_fraction": active_frac,
            "impulse_shape_dispersion": coh_mets["impulse_shape_dispersion"], "post_event_vol_dispersion": coh_mets["post_event_vol_dispersion"],
            "fold_event_count_std": fold_event_count_std, "fold_continuation_rate_std": fold_continuation_rate_std,
            "regime_distinctness_score": dist_score,
            "high_events": v_counts["high_events"], "low_events": v_counts["low_events"],
            "acceptance_pass": True
        }

        candidates.append(row)
        candidate_masks[cand_name] = {"m_high": m_high, "m_low": m_low}

    if not candidates:
        return {"status": "failed", "reason": "zero_candidates_passed"}

    df = pd.DataFrame(candidates)

    # Pre-Learnability Gating
    def _safe_z(s):
        if s.std() < 1e-9: return np.zeros(len(s))
        return (s - s.mean()) / s.std()

    # Cheap Proxy Score to gate heavy model fits
    df["proxy_score"] = (
        _safe_z(df["active_days_fraction"])
        + _safe_z(df["regime_distinctness_score"])
        - _safe_z(df["impulse_shape_dispersion"])
        - _safe_z(df["fold_continuation_rate_std"])
        - _safe_z(df["fold_event_count_std"])
    )

    top_k_learnability = cfg.get("top_k_for_learnability", 8)
    df = df.sort_values("proxy_score", ascending=False).head(top_k_learnability).copy()

    # 7. Learnability (only on top K)
    # If we subsampled in Phase 1, we MUST re-evaluate the full mask for the winners before Learnability
    full_eval_needed = (len(phase1_symbols) < len(symbols))
    
    for idx, row in df.iterrows():
        cand_name = row["name"]
        
        m_h_p1 = candidate_masks[cand_name]["m_high"]
        m_l_p1 = candidate_masks[cand_name]["m_low"]

        if full_eval_needed:
            # Re-evaluate kinematics for all symbols if missing
            z_hr = row["z_hours"]
            d_hr = row["duration_hours"]
            fam = row["family"]
            param_raw = row["param"]
            # Convert string param back to tuple if it's std_plus_abs
            import ast as py_ast
            try:
                if isinstance(param_raw, str) and "(" in param_raw:
                    param = py_ast.literal_eval(param_raw)
                else:
                    param = float(param_raw)
            except:
                param = param_raw

            z = int(z_hr * bph)
            duration_bars = int(d_hr * bph)
            zc = z_cache[z]
            
            # Fill in the rest of kinematics in z_cache if not fully populated
            # Note: asset_ids and timestamps were already full length, but kinematics were only computed for phase1_symbols
            remaining_symbols = [s for s in symbols if s not in phase1_symbols]
            if remaining_symbols:
                # We check if kinematics for one symbol is already populated as a proxy
                # Actually, z_cache arrays were created with np.zeros(n), we just need to fill the rest
                for ast in remaining_symbols:
                    idxs = asset_groups[ast]
                    if np.all(zc["up"][idxs] == 0): # Very rough check
                        ast_high = high[idxs]
                        ast_low = low[idxs]
                        ast_close = close[idxs]
                        ast_ret = ret_1[idxs]
                        ast_vol = vol_g[idxs]

                        hv, hi = rolling_max_index_nb(ast_high, z)
                        lv, li = rolling_min_index_nb(ast_low, z)
                        st_idx = np.maximum(0, np.arange(len(ast_close)) - z + 1)
                        st_px = ast_close[st_idx]

                        c_high_idx[idxs] = idxs[hi]
                        c_low_idx[idxs] = idxs[li]
                        c_start_idx[idxs] = idxs[st_idx]

                        um = np.where(st_px > 1e-9, (hv - st_px) / st_px, 0.0).astype(np.float32)
                        dm = np.where(st_px > 1e-9, (st_px - lv) / st_px, 0.0).astype(np.float32)

                        b_u, b_d, s_u, s_d, m_u, m_d, v_e, pc_u, pc_d = compute_impulse_coherence_nb(
                            ast_ret, ast_vol, hv, lv, st_px, hi, li, st_idx, z
                        )

                        zc["up"][idxs] = um
                        zc["dn"][idxs] = dm
                        zc["std_up"][idxs] = rolling_std_nb(um, 30 * 24 * bph)
                        zc["std_dn"][idxs] = rolling_std_nb(dm, 30 * 24 * bph)
                        # ... filling other zc keys if needed, but only "up" and "dn" and "std_*" are used in _generate_event_masks
            
            # Now generate the FULL mask
            m_h, m_l = _generate_event_masks(fam, param, zc["up"], zc["dn"], zc["std_up"], zc["std_dn"], timestamps, asset_ids, duration_bars)
            candidate_masks[cand_name] = {"m_high": m_h, "m_low": m_l}
        else:
            m_h = m_h_p1
            m_l = m_l_p1

        g_cont, g_rev, g_mae, g_mfe, g_max, auc_e = _compute_conditional_learnability(
            m_h, m_l, learn_X, forward_returns, mae_high, mfe_high, mae_low, mfe_low,
            ret_threshold=cfg.get("phase1_ret_threshold", 0.0), timestamps=timestamps
        )

        df.at[idx, "continuation_predictability_gain"] = g_cont
        df.at[idx, "reversal_predictability_gain"] = g_rev
        df.at[idx, "MAE_predictability_gain"] = g_mae
        df.at[idx, "MFE_predictability_gain"] = g_mfe
        df.at[idx, "predictability_gain"] = g_max

        if cfg.get("enable_learnability_check", True) and g_max <= cfg.get("min_predictability_gain", 0.0):
            df.at[idx, "acceptance_pass"] = False

    df = df[df["acceptance_pass"] == True].copy()
    if df.empty:
        tprint("WARNING: No candidates passed the learnability check.")
        return {"status": "failed", "reason": "learnability_check_failed"}

    q_thresh = df["impulse_shape_dispersion"].quantile(cfg.get("max_allowed_dispersion_quantile", 0.75))
    df = df[df["impulse_shape_dispersion"] <= q_thresh].copy()
    if df.empty:
        tprint("WARNING: No candidates passed the dispersion caps.")
        return {"status": "failed", "reason": "dispersion_caps"}

    # Phase 2: Shortlist Selection
    def _z(col):
        if df[col].std() < 1e-9: return np.zeros(len(df))
        return (df[col] - df[col].mean()) / df[col].std()

    z_events_day_std = _z("events_per_day_std") if "events_per_day_std" in df.columns else np.zeros(len(df))

    df["shortlist_score"] = (
        _z("active_days_fraction") + _z("events_per_day_mean")
        - z_events_day_std
        - _z("impulse_shape_dispersion") - _z("post_event_vol_dispersion") - _z("fold_continuation_rate_std")
        + _z("predictability_gain") + _z("regime_distinctness_score")
    )

    df = df.sort_values("shortlist_score", ascending=False)

    shortlist_idx = []
    fam_counts = {"std_threshold": 0, "abs_move_threshold": 0, "std_plus_abs": 0}
    for i, row in df.iterrows():
        if len(shortlist_idx) >= cfg.get("shortlist_max_candidates", 5): break
        if fam_counts[row["family"]] < cfg.get("shortlist_max_per_family", 2):
            fam_counts[row["family"]] += 1
            shortlist_idx.append(i)

    df_shortlist = df.loc[shortlist_idx].copy()

    # Phase 3: Conditioners on shortlist
    cond_rows = []
    conditioner_modes = cfg.get("conditioner_modes", ["none", "entropy_veto", "vol_regime_veto"])
    if cfg.get("enable_secondary_conditioners", True):
        tprint(f"Phase 3: Applying conditioners to {len(df_shortlist)} shortlisted triggers...")
        for i, row in df_shortlist.iterrows():
            cand_name = row["name"]
            m_high = candidate_masks[cand_name]["m_high"]
            m_low = candidate_masks[cand_name]["m_low"]
            zc = z_cache[int(row["z_hours"] * bph)]
            for mode in conditioner_modes:
                if mode == "none": continue
                new_h, new_l = _apply_secondary_conditioner(
                    m_high, m_low, mode,
                    zc["m_up"], zc["m_dn"], zc["v_exp"],
                    np.nan_to_num(feature_dict.get("spread_to_atr", np.zeros(n))), 
                    alternation_array,
                    entropy_jump=feature_dict.get("entropy_jump_24h"),
                    vol_regime=feature_dict.get("vol_regime_z")
                )
                m_any = new_h | new_l
                tot_events = int(np.sum(m_any))
                if tot_events < cfg.get("min_total_events", 300): continue
                v_pass, v_counts = _check_high_low_viability(new_h, new_l, cfg)
                if cfg.get("enable_bucket_viability_check", True) and not v_pass: continue

                # Perform cheap structural checks to gate expensive learnability recomputation
                coh_mets = _compute_coherence_metrics(new_h, new_l, zc["rng"], zc["b_up"], zc["b_dn"], zc["s_up"], zc["s_dn"], zc["m_up"], zc["m_dn"], zc["v_exp"])

                fold_cont_rates = []
                for tr_idx, va_idx in folds:
                    mask_va = m_any[va_idx]
                    fold_evts = np.sum(mask_va)
                    if fold_evts > 0:
                        fold_cont_rates.append(np.mean(global_cont[va_idx][mask_va]))
                    else:
                        fold_cont_rates.append(0.0)
                new_fold_cont_std = float(np.std(fold_cont_rates))

                base_disp = row.get("impulse_shape_dispersion", 1e9)
                base_f_cont = row.get("fold_continuation_rate_std", 1e9)
                base_h = row.get("high_events", 1)
                base_l = row.get("low_events", 1)
                base_ratio = min(base_h, base_l) / max(base_h, base_l)

                new_ratio = min(v_counts["high_events"], v_counts["low_events"]) / max(v_counts["high_events"], v_counts["low_events"])

                improved = False
                thresh_delta = 0.05
                if coh_mets["impulse_shape_dispersion"] < base_disp * (1.0 - thresh_delta):
                    improved = True
                elif new_fold_cont_std < base_f_cont * (1.0 - thresh_delta):
                    improved = True
                elif new_ratio > base_ratio * (1.0 + thresh_delta):
                    improved = True

                if not improved: continue

                # Fast score clone for modified mask
                gain_max = _compute_conditional_learnability(new_h, new_l, learn_X, forward_returns, mae_high, mfe_high, mae_low, mfe_low, ret_threshold=cfg.get("phase1_ret_threshold", 0.0), timestamps=timestamps)[4]
                new_row = row.copy()
                new_cand_name = cand_name + f"_{mode}"
                new_row["name"] = new_cand_name
                new_row["conditioner_mode"] = mode
                new_row["total_events"] = tot_events
                new_row["high_events"] = v_counts["high_events"]
                new_row["low_events"] = v_counts["low_events"]
                new_row["impulse_shape_dispersion"] = coh_mets["impulse_shape_dispersion"]
                new_row["fold_continuation_rate_std"] = new_fold_cont_std
                new_row["predictability_gain"] = gain_max
                cond_rows.append(new_row)
                candidate_masks[new_cand_name] = {"m_high": new_h, "m_low": new_l}

    if cond_rows:
        df_shortlist = pd.concat([df_shortlist, pd.DataFrame(cond_rows)], ignore_index=True)

        # Local short-list scoring helper
        def _z_short(col):
            if df_shortlist[col].std() < 1e-9: return np.zeros(len(df_shortlist))
            return (df_shortlist[col] - df_shortlist[col].mean()) / df_shortlist[col].std()

        # Re-score shortlist including conditional adjustments
        df_shortlist["shortlist_score"] += _z_short("predictability_gain") * 0.5
        df_shortlist = df_shortlist.sort_values("shortlist_score", ascending=False)

    best_config = df_shortlist.iloc[0].to_dict()
    best_cand_name = best_config["name"]
    best_mask_high = candidate_masks[best_cand_name]["m_high"]
    best_mask_low = candidate_masks[best_cand_name]["m_low"]

    try:
        from extreme_price_movements.offline_optimisers.params_store import save_best_params_csv, INFERENCE_CANDIDATE_MASK_BEST_PARAMS_CSV
        save_best_params_csv(INFERENCE_CANDIDATE_MASK_BEST_PARAMS_CSV, best_config, metadata={"source": "mask_optimiser"})
    except Exception as e:
        tprint(f"Failed to save best params: {e}")

    tprint(f"Layer 0 Complete. Selected: {best_config['name']} (Score: {best_config['shortlist_score']:.4f})")

    return {
        "status": "ok",
        "layer0_candidate_table_": df,
        "layer0_shortlist_": df_shortlist,
        "layer0_best_config_": best_config,
        "layer0_best_mask_high_": best_mask_high,
        "layer0_best_mask_low_": best_mask_low,
        "layer0_candidate_masks_": candidate_masks
    }

# =============================================================================
# CLI Entry Point
# =============================================================================

def _resolve_path(path: str) -> str:
    if not path:
        return path
    pkg_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if os.path.isabs(path):
        return os.path.normpath(path)
    return os.path.normpath(os.path.join(pkg_root, path))

def _find_latest_feature_dir(data_root: str) -> Optional[str]:
    feat_dir = os.path.join(data_root, "features")
    if not os.path.isdir(feat_dir):
        return None
    dirs = sorted(glob.glob(os.path.join(feat_dir, "20*")))
    if not dirs:
        return None
    return dirs[-1]

def run_mask_optimization(args):
    from copy import deepcopy
    cfg = deepcopy(CFG)
    
    if args.data_root:
        cfg["data_root"] = _resolve_path(args.data_root)
    else:
        cfg["data_root"] = _resolve_path(cfg.get("data_root", "data"))

    if args.phase1_max_symbols:
        cfg["phase1_max_symbols"] = args.phase1_max_symbols

    if args.perps:
        cfg["use_perps"] = True
        if not cfg["data_root"].endswith("_perp"):
             cfg["data_root"] += "_perp"
        cfg = enable_perp_feature_keys(cfg)

    feature_path = args.features or _find_latest_feature_dir(cfg["data_root"])
    if not feature_path:
        tprint(f"ERROR: No features found in {cfg['data_root']}/features. Provide --features or run feature generation.")
        return

    tprint(f"Loading data: data_root={cfg['data_root']} | features={feature_path}")
    
    # Load panel and features
    store = PartitionedOHLCVStore(root_dir=cfg["data_root"], timeframe=cfg.get("timeframe", "1h"))
    
    # Select subset of symbols if requested
    import glob
    ohlcv_dir = os.path.join(cfg["data_root"], "ohlcv")
    all_symbols = []
    for path in glob.glob(os.path.join(ohlcv_dir, "symbol=*")):
        base = os.path.basename(path)
        if base.startswith("symbol="):
            raw = base.replace("symbol=", "")
            all_symbols.append(raw.replace("_", "/", 1))
    
    if all_symbols:
        all_symbols.sort() # Ensure deterministic sampling across platforms
        if args.max_symbols and args.max_symbols < len(all_symbols):
            import random
            random.seed(42)
            symbols = random.sample(all_symbols, args.max_symbols)
            tprint(f"Randomly subsampling {args.max_symbols}/{len(all_symbols)} symbols: {symbols}")
        else:
            symbols = all_symbols
    else:
        symbols = []

    # Load symbols individually and combine into panel
    dfs_by_symbol = {}
    start_ts = pd.Timestamp.now(tz='UTC') - pd.Timedelta(days=int(365.25 * args.lookback_years))
    for s in symbols:
        df = store.load(s, start_ts=start_ts)
        if not df.empty:
            dfs_by_symbol[s] = df
        else:
            tprint(f"Warning: Symbol {s} has no data for the requested period.")

    if not dfs_by_symbol:
        tprint("ERROR: All symbols returned empty DataFrames.")
        return

    panel = to_panel(dfs_by_symbol)
    if not panel or "close" not in panel or panel["close"].empty:
        tprint("ERROR: Panel data empty or missing 'close' column.")
        return

    # Load only necessary features for learnability check
    learn_feat_names = [
        "trend_pct", "volatility_zscore", "range_12h_pct", "volume_zscore",
        "relative_volume", "atr_pct_15m", "spread_bps", "funding_rate",
        "buy_volume_share", "order_book_imbalance", "entropy_jump_24h", "vol_regime_z"
    ]
    # Filter available features
    available_feats = []
    # Just load all for now if we don't have a specific list, 
    # but optimize_layer0_masks expects a feature_dict.
    
    # Parse feature_path for load_features_selected
    ts_str = os.path.basename(feature_path)
    try:
        ts = pd.Timestamp(ts_str.replace("_", " "))
    except Exception:
        # Fallback if folder name is not a simple timestamp
        ts = pd.Timestamp.now(tz='UTC')
    
    data_root_dir = os.path.dirname(os.path.dirname(feature_path))
    
    tprint(f"Loading features from {feature_path} for {len(symbols)} symbols...")
    feat_dict_raw = load_features_selected(
        ts=ts,
        root_dir=data_root_dir,
        feature_keys=None, # Load all keys
        symbols=symbols,
        start_ts=start_ts
    )
    
    if not feat_dict_raw:
        tprint("ERROR: Feature dictionary empty.")
        return

    # Align all wide DataFrames to common index and symbols
    common_idx = panel["close"].index
    common_syms = panel["close"].columns
    
    # Pre-calculate forward returns in wide form
    fwd_hours = cfg.get("mask_opt_forward_hours", 12)
    fwd_ret_wide = panel["close"].pct_change(fwd_hours).shift(-fwd_hours)
    
    # Stack OHLCV into long-form DataFrame
    # Note: we use 'symbol' column which optimize_layer0_masks checks in line 586
    data_stacked = panel["close"].stack(dropna=False).reset_index()
    data_stacked.columns = ["timestamp", "symbol", "close"]
    # Ensure other OHLC parts are aligned and stacked
    data_stacked["high"] = panel["high"].reindex(index=common_idx, columns=common_syms).stack(dropna=False).values
    data_stacked["low"] = panel["low"].reindex(index=common_idx, columns=common_syms).stack(dropna=False).values
    
    # Stack features into 1D arrays in feature_dict
    feature_dict = {}
    for k, df in feat_dict_raw.items():
        if isinstance(df, pd.DataFrame):
            df_aligned = df.reindex(index=common_idx, columns=common_syms).fillna(method="ffill").fillna(0)
            feature_dict[k] = df_aligned.stack(dropna=False).to_numpy(dtype=np.float32)

    # Scale event thresholds by symbol count
    n_symbols = len(common_syms)
    cfg["max_events_per_day"] = cfg.get("max_events_per_day", 50) * n_symbols
    cfg["min_events_per_day"] = cfg.get("min_events_per_day", 1) * n_symbols
    cfg["min_total_events"] = cfg.get("min_total_events", 300) # Keep global floor
    
    # Stack forward returns
    fwd_ret_stacked = fwd_ret_wide.reindex(index=common_idx, columns=common_syms).stack(dropna=False).to_numpy(dtype=np.float32)

    tprint(f"Starting mask optimization on {data_stacked.shape[0]} total rows ({len(common_idx)} bars x {len(common_syms)} symbols)...")
    result = optimize_layer0_masks(data_stacked, feature_dict, fwd_ret_stacked, cfg)
    
    if isinstance(result, dict):
        if result.get("status") == "failed":
            tprint(f"ERROR: Mask optimization failed: {result.get('reason')}")
        else:
            tprint(f"Optimization finished with status: {result.get('status', 'unknown')}")
    else:
        tprint("Optimization finished successfully.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optimize Layer 0 event masks")
    parser.add_argument("--data-root", help="Override data root")
    parser.add_argument("--features", help="Path to features directory")
    parser.add_argument("--perps", action="store_true", help="Use perpetual mode data")
    parser.add_argument("--max-symbols", type=int, help="Cap symbols for speed")
    parser.add_argument("--phase1-max-symbols", type=int, help="Subsample Phase 1 grid search for massive speedup")
    parser.add_argument("--lookback-years", type=float, default=2.0, help="Years of data to load")
    
    args = parser.parse_args()
    run_mask_optimization(args)
