import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple, Optional
from numba import njit
from extreme_price_movements.purged_cv import PurgedKFold
from sklearn.linear_model import LogisticRegression, HuberRegressor
from sklearn.metrics import roc_auc_score

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
    for i in range(n):
        start = max(0, i - window + 1)
        slice_x = x[start:i+1]
        valid_count = 0
        mean = 0.0
        for val in slice_x:
            if not np.isnan(val):
                mean += val
                valid_count += 1
        if valid_count > 1:
            mean /= valid_count
            var = 0.0
            for val in slice_x:
                if not np.isnan(val):
                    var += (val - mean)**2
            out[i] = np.sqrt(var / (valid_count - 1))
        elif valid_count == 1:
            out[i] = 0.0
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
    timestamps: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Generates boolean masks for event high and low based on per-asset/cross-sectional rules."""
    n = len(up_move)
    mask_h = np.zeros(n, dtype=bool)
    mask_l = np.zeros(n, dtype=bool)

    if family == "top_movers":
        w_pct = param_val
        df = pd.DataFrame({"up": up_move, "dn": dn_move, "ts": timestamps})

        def calc_percentile(x, pct):
            vals = x.dropna().values
            if len(vals) == 0:
                return np.inf
            return np.nanpercentile(vals, 100 - pct)

        thresh_up = df.groupby('ts')['up'].agg(lambda x: calc_percentile(x, w_pct))
        thresh_dn = df.groupby('ts')['dn'].agg(lambda x: calc_percentile(x, w_pct))

        grp_u = df['ts'].map(thresh_up)
        grp_d = df['ts'].map(thresh_dn)

        mask_h = (df["up"] >= grp_u).values.astype(bool)
        mask_l = (df["dn"] >= grp_d).values.astype(bool)

    elif family == "std_threshold":
        # Assumes rolling_std is already computed PER ASSET via outer loop
        x_std = param_val
        mask_h = (up_move >= x_std * rolling_std_up)
        mask_l = (dn_move >= x_std * rolling_std_dn)

    elif family == "abs_move_threshold":
        y_move = param_val / 100.0
        mask_h = (up_move >= y_move)
        mask_l = (dn_move >= y_move)

    return mask_h, mask_l

def _apply_secondary_conditioner(
    mask_h: np.ndarray, mask_l: np.ndarray,
    conditioner: str,
    mono_up: np.ndarray, mono_dn: np.ndarray,
    vol_exp: np.ndarray, spread_to_atr: np.ndarray,
    alternation_array: np.ndarray
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
    tprint("=" * 80)
    tprint("LAYER 0: DIRECTIONAL IMPULSE MASK OPTIMIZATION")
    tprint("=" * 80)

    bph = cfg.get("bars_per_hour", 4) # Default 15m

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
        vol_g[idxs] = rolling_std_nb(ast_ret, 24 * bph).astype(np.float32)

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

    for z_hr in cfg.get("z_hours_grid", [8, 16]):
        for fam in cfg.get("families", ["top_movers", "std_threshold", "abs_move_threshold"]):
            if fam == "top_movers":
                for p in cfg.get("top_w_pct_grid", [4, 8]): grid.append((z_hr, fam, p))
            elif fam == "std_threshold":
                for p in cfg.get("x_std_grid", [1.5, 1.8]): grid.append((z_hr, fam, p))
            elif fam == "abs_move_threshold":
                for p in cfg.get("y_move_pct_grid", [4.0, 6.0]): grid.append((z_hr, fam, p))

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

    for (z_hr, fam, param) in grid:
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

            for ast, idxs in asset_groups.items():
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
                c_roll_std_up[idxs] = rolling_std_nb(um, 24 * bph)
                c_roll_std_dn[idxs] = rolling_std_nb(dm, 24 * bph)

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

        # 2. Mask generation
        m_high, m_low = _generate_event_masks(fam, param, zc["up"], zc["dn"], zc["std_up"], zc["std_dn"], timestamps)
        m_any = m_high | m_low

        tot_events = int(np.sum(m_any))
        if tot_events < cfg.get("min_total_events", 300): continue

        ts_any = pd.to_datetime(timestamps[m_any])
        days_any = ts_any.floor("D").nunique()
        tot_days = max(1, pd.to_datetime(timestamps).floor("D").nunique())
        active_frac = days_any / tot_days

        if active_frac < cfg.get("min_active_days_fraction", 0.20): continue

        events_p_day_mean = tot_events / max(1, days_any)
        if not (cfg.get("min_events_per_day", 1) <= events_p_day_mean <= cfg.get("max_events_per_day", 50)): continue

        # 3. Coherence
        coh_mets = _compute_coherence_metrics(m_high, m_low, zc["rng"], zc["b_up"], zc["b_dn"], zc["s_up"], zc["s_dn"], zc["m_up"], zc["m_dn"], zc["v_exp"])

        # 4. Distinctness
        dist_score = _compute_regime_distinctness(m_high, m_low, forward_returns, mae_high, mfe_high, mae_low, mfe_low)
        if cfg.get("enable_regime_distinctness_check", True) and dist_score < cfg.get("min_regime_distinctness_score", 1.1): continue

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
        if cfg.get("enable_bucket_viability_check", True) and not v_pass: continue

        cand_name = f"{fam}_z{z_hr}_p{param}"
        row = {
            "name": cand_name, "family": fam, "z_hours": z_hr, "param": param, "conditioner_mode": "none",
            "total_events": tot_events, "events_per_day_mean": events_p_day_mean, "events_per_day_std": events_per_day_std,
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
    for idx, row in df.iterrows():
        cand_name = row["name"]
        m_h = candidate_masks[cand_name]["m_high"]
        m_l = candidate_masks[cand_name]["m_low"]

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
        return {"status": "failed", "reason": "learnability_check_failed"}

    q_thresh = df["impulse_shape_dispersion"].quantile(cfg.get("max_allowed_dispersion_quantile", 0.75))
    df = df[df["impulse_shape_dispersion"] <= q_thresh].copy()
    if df.empty: return {"status": "failed", "reason": "dispersion_caps"}

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
    fam_counts = {"top_movers": 0, "std_threshold": 0, "abs_move_threshold": 0}
    for i, row in df.iterrows():
        if len(shortlist_idx) >= cfg.get("shortlist_max_candidates", 5): break
        if fam_counts[row["family"]] < cfg.get("shortlist_max_per_family", 2):
            fam_counts[row["family"]] += 1
            shortlist_idx.append(i)

    df_shortlist = df.loc[shortlist_idx].copy()

    # Phase 3: Conditioners on shortlist
    cond_rows = []
    if cfg.get("enable_secondary_conditioners", True):
        tprint(f"Phase 3: Applying conditioners to {len(df_shortlist)} shortlisted triggers...")
        for i, row in df_shortlist.iterrows():
            cand_name = row["name"]
            m_high = candidate_masks[cand_name]["m_high"]
            m_low = candidate_masks[cand_name]["m_low"]
            zc = z_cache[int(row["z_hours"] * bph)]
            for mode in cfg.get("conditioner_modes", ["none"]):
                if mode == "none": continue
                new_h, new_l = _apply_secondary_conditioner(
                    m_high, m_low, mode,
                    zc["m_up"], zc["m_dn"], zc["v_exp"],
                    np.nan_to_num(feature_dict.get("spread_to_atr", np.zeros(n))), alternation_array
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
