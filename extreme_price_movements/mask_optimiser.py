import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple
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
    high_idx: np.ndarray,
    low_idx: np.ndarray,
    start_idx: np.ndarray,
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
        st = start_idx[i]
        st_px = start_px[i]

        # Upward coherence
        peak_h = high_idx[i]
        b_up = peak_h - st
        bars_to_peak_up[i] = b_up

        # Downward coherence
        peak_l = low_idx[i]
        b_dn = peak_l - st
        bars_to_peak_dn[i] = b_dn

        # Speed (displacement over bars)
        imp_up = (high_val[i] - st_px) / st_px if st_px > 1e-9 else 0.0
        imp_dn = (st_px - low_val[i]) / st_px if st_px > 1e-9 else 0.0

        speed_up[i] = imp_up / max(1.0, b_up)
        speed_dn[i] = imp_dn / max(1.0, b_dn)

        # Monotonicity & Concentration
        dir_sum_up = 0.0
        abs_sum_up = 0.0
        max_bar_up = 0.0
        for j in range(st + 1, peak_h + 1):
            if j < n:
                r = returns[j]
                if not np.isnan(r):
                    dir_sum_up += r
                    abs_sum_up += abs(r)
                    if r > max_bar_up:
                        max_bar_up = r
        mono_up[i] = dir_sum_up / abs_sum_up if abs_sum_up > 1e-9 else 0.0
        peak_conc_up[i] = max_bar_up / dir_sum_up if dir_sum_up > 1e-9 else 0.0

        dir_sum_dn = 0.0
        abs_sum_dn = 0.0
        max_bar_dn = 0.0
        for j in range(st + 1, peak_l + 1):
            if j < n:
                r = returns[j]
                if not np.isnan(r):
                    dir_sum_dn += -r  # accumulate -r for downward trends
                    abs_sum_dn += abs(r)
                    if -r > max_bar_dn:
                        max_bar_dn = -r
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
    rolling_std: np.ndarray,
    timestamps: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Generates boolean masks for event high and low based on per-asset/cross-sectional rules."""
    n = len(up_move)
    mask_h = np.zeros(n, dtype=bool)
    mask_l = np.zeros(n, dtype=bool)

    if family == "top_movers":
        w_pct = param_val
        # Cross-sectional rank by timestamp avoiding index corruption
        df = pd.DataFrame({"up": up_move, "dn": dn_move, "ts": timestamps})

        # Calculate thresholds per timestamp group and map them back securely
        grp_u = df.groupby("ts")["up"].transform(lambda x: np.nanpercentile(x, 100 - w_pct) if x.notna().any() else np.inf)
        grp_d = df.groupby("ts")["dn"].transform(lambda x: np.nanpercentile(x, 100 - w_pct) if x.notna().any() else np.inf)

        mask_h = (df["up"] >= grp_u).values.astype(bool)
        mask_l = (df["dn"] >= grp_d).values.astype(bool)

    elif family == "std_threshold":
        # Assumes rolling_std is already computed PER ASSET via outer loop
        x_std = param_val
        thresh = x_std * rolling_std
        mask_h = (up_move >= thresh)
        mask_l = (dn_move >= thresh)

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
    ret_1: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Widen, tighten, or veto events based on small interpretable rules."""
    new_h = mask_h.copy()
    new_l = mask_l.copy()

    if conditioner == "none":
        return new_h, new_l

    if conditioner == "liquidity_veto":
        # Absolute hard veto if cost relative to ATR is astronomical
        safe_liq = spread_to_atr < 0.25
        return new_h & safe_liq, new_l & safe_liq

    if conditioner == "monotonicity_adjust":
        # Tighten: drop events if monotonicity is garbage (i.e. very choppy chop)
        return new_h & (mono_up > 0.25), new_l & (mono_dn > 0.25)

    if conditioner == "volatility_adjust":
        # Tighten: drop events if it's just a generalized vol explosion without directionality
        # Vol expansion > 5x is usually pure chaos
        return new_h & (vol_exp < 5.0), new_l & (vol_exp < 5.0)

    if conditioner == "alternation_adjust":
        # Alternation approximation: drop if consecutive bars constantly reverse sign
        roll_alt = pd.Series(np.sign(ret_1) != np.roll(np.sign(ret_1), 1)).rolling(6).mean().values
        return new_h & (roll_alt < 0.70), new_l & (roll_alt < 0.70)

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
    mask_any: np.ndarray,
    forward_returns: np.ndarray,
    mae_atr: np.ndarray,
    mfe_atr: np.ndarray
) -> float:
    """Distinctness comparing event behavior against global distribution (std, tails, MAE/MFE)."""
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

    # MAE distribution shift
    mae_g = np.nanmean(mae_atr)
    mae_e = np.nanmean(mae_atr[mask_any])
    mae_ratio = mae_e / mae_g if mae_g > 1e-9 else 1.0

    return float(max(std_ratio, tail_ratio, mae_ratio))


def _compute_conditional_learnability(
    mask_high: np.ndarray,
    mask_low: np.ndarray,
    features: np.ndarray,
    forward_returns: np.ndarray,
    mae_atr: np.ndarray,
    mfe_atr: np.ndarray,
    ret_threshold: float = 0.0
) -> Tuple[float, float, float, float, float, float]:
    """
    Tests predictability inside the event regime using tiny simple models.
    Compares event predictability against global background.
    """
    mask_any = mask_high | mask_low
    valid = np.isfinite(forward_returns) & np.isfinite(mae_atr)

    idx_g = np.where(valid)[0]
    idx_e = np.where(valid & mask_any)[0]

    if len(idx_e) < 50:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    # Build direction-aware classification targets
    # For the global baseline, we assume a generic positive-trend target
    # so the global model has a valid baseline classification task to learn against.
    # Inside the event mask, we flip the polarity based on direction.
    cont_target = (forward_returns > ret_threshold).astype(int)
    rev_target = (forward_returns < -ret_threshold).astype(int)

    # Override only within the downward impulse mask (Low events)
    cont_target[mask_low] = (forward_returns[mask_low] < -ret_threshold).astype(int)
    rev_target[mask_low] = (forward_returns[mask_low] > ret_threshold).astype(int)

    def _test_classifier(X, y):
        if len(np.unique(y)) < 2: return 0.5
        model = LogisticRegression(max_iter=100)
        try:
            model.fit(X, y)
            preds = model.predict_proba(X)[:, 1]
            return roc_auc_score(y, preds)
        except Exception:
            return 0.5

    def _test_regressor(X, y):
        model = HuberRegressor()
        try:
            model.fit(X, y)
            preds = model.predict(X)
            # R2 proxy
            ssr = np.sum((y - preds)**2)
            sst = np.sum((y - np.mean(y))**2)
            return 1.0 - (ssr / max(sst, 1e-9))
        except Exception:
            return 0.0

    X_g = np.nan_to_num(features[idx_g])
    X_e = np.nan_to_num(features[idx_e])

    # Compare Continuation (Classification)
    auc_cont_g = _test_classifier(X_g, cont_target[idx_g])
    auc_cont_e = _test_classifier(X_e, cont_target[idx_e])
    gain_cont = auc_cont_e - auc_cont_g

    # Compare Reversal (Classification)
    auc_rev_g = _test_classifier(X_g, rev_target[idx_g])
    auc_rev_e = _test_classifier(X_e, rev_target[idx_e])
    gain_rev = auc_rev_e - auc_rev_g

    # Compare MAE (Regression)
    r2_mae_g = _test_regressor(X_g, mae_atr[idx_g])
    r2_mae_e = _test_regressor(X_e, mae_atr[idx_e])
    gain_mae = r2_mae_e - r2_mae_g

    # Compare MFE (Regression)
    r2_mfe_g = _test_regressor(X_g, mfe_atr[idx_g])
    r2_mfe_e = _test_regressor(X_e, mfe_atr[idx_e])
    gain_mfe = r2_mfe_e - r2_mfe_g

    gain_max = max(gain_cont, gain_rev, gain_mae, gain_mfe)

    return float(gain_cont), float(gain_rev), float(gain_mae), float(gain_mfe), float(gain_max), float(auc_cont_e)


def _check_bucket_viability(
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


def _extract_learnability_features(feature_dict: Dict[str, np.ndarray]) -> np.ndarray:
    """Fetch 10-15 learnability features matching post-impulse state."""
    # Ordered keys aligning with prompt spec priorities
    target_keys = [
        "range_1_atr", "close_location_in_bar", "rv_ratio_6_24",
        "impulse_vol_ratio", "vol_compression_ratio", "range_decay",
        "momentum_last_3bars_impulse_return", "reversal_bar_strength",
        "climax_volume_ratio", "rejection_volume_ratio", "vol_regime_shift",
        "bar_direction_entropy"
    ]

    n = len(next(iter(feature_dict.values())))
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
    timestamps = pd.to_datetime(data.get("timestamp", data.index)).values

    # Derive mock MAE/MFE arrays locally for cheap Layer 0 horizon tests
    horizon = cfg.get("phase1_forward_horizon_bars", 12)
    n = len(close)
    mae_arr = np.zeros(n, dtype=np.float32)
    mfe_arr = np.zeros(n, dtype=np.float32)
    atr = np.ascontiguousarray(feature_dict.get("atr", np.ones(n)), dtype=np.float32)

    # Cheap rolling horizon proxy
    for i in range(n - horizon):
        h_sl = high[i+1 : i+horizon+1]
        l_sl = low[i+1 : i+horizon+1]
        c = close[i]
        mae_arr[i] = (c - np.min(l_sl)) / max(atr[i], 1e-9)
        mfe_arr[i] = (np.max(h_sl) - c) / max(atr[i], 1e-9)

    # Global baselines
    ret_1 = np.where(np.roll(close, 1) > 0, (close - np.roll(close, 1)) / np.roll(close, 1), 0).astype(np.float32)
    ret_1[0] = 0.0
    vol_g = rolling_std_nb(ret_1, 24 * bph)

    learn_X = _extract_learnability_features(feature_dict)

    # Pre-computation arrays per-asset logic
    # In live multi-asset files, data should have "asset_id". Here we group generically.
    if "asset_id" in data.columns:
        asset_groups = data.groupby("asset_id").indices
    elif "symbol" in data.columns:
        asset_groups = data.groupby("symbol").indices
    else:
        asset_groups = {"ALL": np.arange(n)}

    candidates = []
    grid = []

    for z_hr in cfg.get("z_hours_grid", [8, 12, 16]):
        for fam in cfg.get("families", ["top_movers", "std_threshold", "abs_move_threshold"]):
            if fam == "top_movers":
                for p in cfg.get("top_w_pct_grid", [4, 6, 8]): grid.append((z_hr, fam, p))
            elif fam == "std_threshold":
                for p in cfg.get("x_std_grid", [1.4, 1.6, 1.8]): grid.append((z_hr, fam, p))
            elif fam == "abs_move_threshold":
                for p in cfg.get("y_move_pct_grid", [4.0, 5.5, 7.0]): grid.append((z_hr, fam, p))

    tprint(f"Phase 1: Evaluating {len(grid)} primary candidates...")

    # Outer Cache per Window Z
    z_cache = {}

    # Prepare Temporal Folds
    pkf = PurgedKFold(n_splits=3, purge=43200, embargo=43200, times=timestamps)
    dummy_X = np.empty((n, 1))
    folds = list(pkf.split(dummy_X))
    if not folds:
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

            for ast, idxs in asset_groups.items():
                ast_high = high[idxs]
                ast_low = low[idxs]
                ast_close = close[idxs]

                hv, hi = rolling_max_index_nb(ast_high, z)
                lv, li = rolling_min_index_nb(ast_low, z)
                st_idx = np.maximum(0, np.arange(len(ast_close)) - z + 1)
                st_px = ast_close[st_idx]

                um = np.where(st_px > 1e-9, (hv - st_px) / st_px, 0.0).astype(np.float32)
                dm = np.where(st_px > 1e-9, (st_px - lv) / st_px, 0.0).astype(np.float32)
                rm = np.where(st_px > 1e-9, (hv - lv) / st_px, 0.0).astype(np.float32)

                c_high_val[idxs] = hv
                c_high_idx[idxs] = hi + idxs[0] # globalize
                c_low_val[idxs] = lv
                c_low_idx[idxs] = li + idxs[0]
                c_start_px[idxs] = st_px
                c_start_idx[idxs] = st_idx + idxs[0]
                c_up_move[idxs] = um
                c_dn_move[idxs] = dm
                c_rng_move[idxs] = rm
                c_roll_std_up[idxs] = rolling_std_nb(um, 24 * bph) # Standardize over trailing day

            b_up, b_dn, s_up, s_dn, m_up, m_dn, v_exp, p_conc_up, p_conc_dn = compute_impulse_coherence_nb(
                ret_1, vol_g, c_high_val, c_low_val, c_start_px, c_high_idx, c_low_idx, c_start_idx, z
            )

            z_cache[z] = {
                "up": c_up_move, "dn": c_dn_move, "rng": c_rng_move, "std": c_roll_std_up,
                "b_up": b_up, "b_dn": b_dn, "s_up": s_up, "s_dn": s_dn,
                "m_up": m_up, "m_dn": m_dn, "v_exp": v_exp
            }

        zc = z_cache[z]

        # 2. Mask generation
        m_high, m_low = _generate_event_masks(fam, param, zc["up"], zc["dn"], zc["std"], timestamps)
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
        dist_score = _compute_regime_distinctness(m_any, forward_returns, mae_arr, mfe_arr)
        if cfg.get("enable_regime_distinctness_check", True) and dist_score < cfg.get("min_regime_distinctness_score", 1.1): continue

        # 5. Learnability
        g_cont, g_rev, g_mae, g_mfe, g_max, auc_e = _compute_conditional_learnability(
            m_high, m_low, learn_X, forward_returns, mae_arr, mfe_arr, ret_threshold=cfg.get("phase1_ret_threshold", 0.0)
        )
        if cfg.get("enable_learnability_check", True) and g_max <= cfg.get("min_predictability_gain", 0.0): continue

        # 6. Temporal Folds Evaluation
        fold_event_counts = []
        fold_cont_rates = []

        # Determine global continuation target for fold check
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

        # 7. Viability
        v_pass, v_counts = _check_bucket_viability(m_high, m_low, cfg)
        if cfg.get("enable_bucket_viability_check", True) and not v_pass: continue

        row = {
            "name": f"{fam}_z{z_hr}_p{param}", "family": fam, "z_hours": z_hr, "param": param, "conditioner_mode": "none",
            "total_events": tot_events, "events_per_day_mean": events_p_day_mean, "active_days_fraction": active_frac,
            "impulse_shape_dispersion": coh_mets["impulse_shape_dispersion"], "post_event_vol_dispersion": coh_mets["post_event_vol_dispersion"],
            "fold_event_count_std": fold_event_count_std, "fold_continuation_rate_std": fold_continuation_rate_std,
            "regime_distinctness_score": dist_score, "continuation_predictability_gain": g_cont, "reversal_predictability_gain": g_rev,
            "MAE_predictability_gain": g_mae, "MFE_predictability_gain": g_mfe, "predictability_gain": g_max,
            "high_events": v_counts["high_events"], "low_events": v_counts["low_events"],
            "acceptance_pass": True, "m_high": m_high, "m_low": m_low
        }
        candidates.append(row)

    if not candidates:
        return {"status": "failed", "reason": "zero_candidates_passed"}

    df = pd.DataFrame(candidates)

    q_thresh = df["impulse_shape_dispersion"].quantile(cfg.get("max_allowed_dispersion_quantile", 0.75))
    df = df[df["impulse_shape_dispersion"] <= q_thresh].copy()
    if df.empty: return {"status": "failed", "reason": "dispersion_caps"}

    # Phase 2: Shortlist Selection
    def _z(col):
        if df[col].std() < 1e-9: return np.zeros(len(df))
        return (df[col] - df[col].mean()) / df[col].std()

    df["shortlist_score"] = (
        _z("active_days_fraction") + _z("events_per_day_mean")
        - _z("events_per_day_std") if "events_per_day_std" in df.columns else 0.0
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
            zc = z_cache[int(row["z_hours"] * bph)]
            for mode in cfg.get("conditioner_modes", ["none"]):
                if mode == "none": continue
                new_h, new_l = _apply_secondary_conditioner(
                    row["m_high"], row["m_low"], mode,
                    zc["m_up"], zc["m_dn"], zc["v_exp"],
                    np.nan_to_num(feature_dict.get("spread_to_atr", np.zeros(n))), ret_1
                )
                m_any = new_h | new_l
                tot_events = int(np.sum(m_any))
                if tot_events < cfg.get("min_total_events", 300): continue
                v_pass, v_counts = _check_bucket_viability(new_h, new_l, cfg)
                if cfg.get("enable_bucket_viability_check", True) and not v_pass: continue

                # Fast score clone for modified mask
                gain_max = _compute_conditional_learnability(new_h, new_l, learn_X, forward_returns, mae_arr, mfe_arr, cfg.get("phase1_ret_threshold", 0.0))[4]
                new_row = row.copy()
                new_row["name"] += f"_{mode}"
                new_row["conditioner_mode"] = mode
                new_row["total_events"] = tot_events
                new_row["high_events"] = v_counts["high_events"]
                new_row["low_events"] = v_counts["low_events"]
                new_row["predictability_gain"] = gain_max
                new_row["m_high"] = new_h
                new_row["m_low"] = new_l
                cond_rows.append(new_row)

    if cond_rows:
        df_shortlist = pd.concat([df_shortlist, pd.DataFrame(cond_rows)], ignore_index=True)
        # Re-score shortlist including conditional adjustments
        df_shortlist["shortlist_score"] += _z("predictability_gain") * 0.5 # Bump those with better gain after conditioning
        df_shortlist = df_shortlist.sort_values("shortlist_score", ascending=False)

    best_config = df_shortlist.iloc[0].to_dict()
    best_mask_high = best_config.pop("m_high")
    best_mask_low = best_config.pop("m_low")

    df = df.drop(columns=["m_high", "m_low"])
    df_shortlist = df_shortlist.drop(columns=["m_high", "m_low"])

    tprint(f"Layer 0 Complete. Selected: {best_config['name']} (Score: {best_config['shortlist_score']:.4f})")

    return {
        "status": "ok",
        "layer0_candidate_table_": df,
        "layer0_shortlist_": df_shortlist,
        "layer0_best_config_": best_config,
        "layer0_best_mask_high_": best_mask_high,
        "layer0_best_mask_low_": best_mask_low,
    }
