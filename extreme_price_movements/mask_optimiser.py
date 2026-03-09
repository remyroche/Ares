import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple
from numba import njit
from extreme_price_movements.purged_cv import PurgedKFold

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
    high_idx: np.ndarray,
    low_idx: np.ndarray,
    start_idx: np.ndarray,
    window: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

    n = len(returns)
    bars_to_peak_up = np.zeros(n, dtype=np.float32)
    bars_to_peak_dn = np.zeros(n, dtype=np.float32)
    speed_up = np.zeros(n, dtype=np.float32)
    speed_dn = np.zeros(n, dtype=np.float32)
    mono_up = np.zeros(n, dtype=np.float32)
    mono_dn = np.zeros(n, dtype=np.float32)
    vol_exp = np.zeros(n, dtype=np.float32)

    for i in range(window, n):
        st = start_idx[i]

        # Upward coherence
        peak_h = high_idx[i]
        bars_to_peak_up[i] = peak_h - st

        # Downward coherence
        peak_l = low_idx[i]
        bars_to_peak_dn[i] = peak_l - st

        # Speed
        speed_up[i] = returns[peak_h] / max(1.0, bars_to_peak_up[i]) if bars_to_peak_up[i] > 0 else 0.0
        speed_dn[i] = returns[peak_l] / max(1.0, bars_to_peak_dn[i]) if bars_to_peak_dn[i] > 0 else 0.0

        # Monotonicity = directional sum / abs sum
        dir_sum_up = 0.0
        abs_sum_up = 0.0
        for j in range(st + 1, peak_h + 1):
            if j < n:
                r = returns[j]
                if not np.isnan(r):
                    dir_sum_up += r
                    abs_sum_up += abs(r)
        mono_up[i] = dir_sum_up / abs_sum_up if abs_sum_up > 1e-9 else 0.0

        dir_sum_dn = 0.0
        abs_sum_dn = 0.0
        for j in range(st + 1, peak_l + 1):
            if j < n:
                r = returns[j]
                if not np.isnan(r):
                    dir_sum_dn += r
                    abs_sum_dn += abs(r)
        mono_dn[i] = dir_sum_dn / abs_sum_dn if abs_sum_dn > 1e-9 else 0.0

        # Vol expansion
        pre_vol = volatility[st]
        post_vol = volatility[i]
        vol_exp[i] = post_vol / pre_vol if pre_vol > 1e-9 else 1.0

    return bars_to_peak_up, bars_to_peak_dn, speed_up, speed_dn, mono_up, mono_dn, vol_exp, vol_exp


# -----------------------------------------------------------------------------
# MASK GENERATORS
# -----------------------------------------------------------------------------

def _generate_event_masks(
    family: str,
    param_val: float,
    up_move: np.ndarray,
    dn_move: np.ndarray,
    rolling_std: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Generates boolean masks for event high and low."""
    if family == "top_movers":
        w_pct = param_val
        valid_u = up_move[np.isfinite(up_move)]
        valid_d = dn_move[np.isfinite(dn_move)]
        u_thresh = np.percentile(valid_u, 100 - w_pct) if len(valid_u) else np.inf
        d_thresh = np.percentile(valid_d, 100 - w_pct) if len(valid_d) else np.inf
        return (up_move >= u_thresh), (dn_move >= d_thresh)

    elif family == "std_threshold":
        x_std = param_val
        thresh = x_std * rolling_std
        return (up_move >= thresh), (dn_move >= thresh)

    elif family == "abs_move_threshold":
        y_move = param_val / 100.0
        return (up_move >= y_move), (dn_move >= y_move)

    return np.zeros_like(up_move, dtype=bool), np.zeros_like(dn_move, dtype=bool)


def _compute_coherence_metrics(
    mask_high: np.ndarray, mask_low: np.ndarray,
    range_move: np.ndarray, bars_to_peak_up: np.ndarray, bars_to_peak_dn: np.ndarray,
    speed_up: np.ndarray, speed_dn: np.ndarray,
    mono_up: np.ndarray, mono_dn: np.ndarray, vol_exp: np.ndarray
) -> Dict[str, float]:
    """Computes dispersion properties of the impulse geometry."""
    metrics = {}

    mask_any = mask_high | mask_low
    if not np.any(mask_any):
        return {
            "impulse_shape_dispersion": 1e9,
            "post_event_vol_dispersion": 1e9,
            "range_move_dispersion": 1e9
        }

    # Pool UP/DN coherence features conditionally
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

    # Standardization inside event
    def _safe_std(x):
        if len(x) < 2: return 0.0
        return float(np.std(x))

    std_bars = _safe_std(bars_comb[valid_mask])
    std_speed = _safe_std(speed_comb[valid_mask])
    std_mono = _safe_std(mono_comb[valid_mask])

    # Composite shape dispersion
    metrics["impulse_shape_dispersion"] = std_bars + std_speed + std_mono
    metrics["post_event_vol_dispersion"] = _safe_std(vol_exp[valid_mask])
    metrics["range_move_dispersion"] = _safe_std(range_move[valid_mask])

    return metrics


def _compute_regime_distinctness(
    mask_any: np.ndarray,
    forward_returns: np.ndarray
) -> Tuple[float, float, float]:
    """Calculates distinctness vs global background."""
    if not np.any(mask_any):
        return 0.0, 0.0, 0.0

    valid = np.isfinite(forward_returns)
    ret_g = forward_returns[valid]
    ret_e = forward_returns[valid & mask_any]

    if len(ret_e) < 10 or len(ret_g) < 10:
        return 0.0, 0.0, 0.0

    std_g = np.std(ret_g)
    std_e = np.std(ret_e)

    std_ratio = std_e / std_g if std_g > 1e-9 else 1.0

    # Tail probability (outer 10%)
    t_upper = np.percentile(ret_g, 95)
    t_lower = np.percentile(ret_g, 5)

    tail_g = np.mean((ret_g >= t_upper) | (ret_g <= t_lower))
    tail_e = np.mean((ret_e >= t_upper) | (ret_e <= t_lower))

    tail_ratio = tail_e / tail_g if tail_g > 1e-9 else 1.0

    score = max(std_ratio, tail_ratio)
    return float(score), float(std_ratio), float(tail_ratio)


def _compute_conditional_learnability(
    mask_any: np.ndarray,
    features: np.ndarray,
    forward_returns: np.ndarray
) -> Tuple[float, float, float]:
    """Measures if maximum feature correlation improves inside the event mask."""
    if not np.any(mask_any):
        return 0.0, 0.0, 0.0

    valid = np.isfinite(forward_returns)
    ret_g = forward_returns[valid]
    ret_e = forward_returns[valid & mask_any]

    if len(ret_e) < 50 or len(ret_g) < 50:
        return 0.0, 0.0, 0.0

    feat_g = features[valid]
    feat_e = features[valid & mask_any]

    max_corr_g = 0.0
    max_corr_e = 0.0

    for col in range(feat_g.shape[1]):
        fg = feat_g[:, col]
        fe = feat_e[:, col]
        if np.var(fg) > 1e-9:
            cg = abs(np.corrcoef(fg, ret_g)[0, 1])
            if not np.isnan(cg) and cg > max_corr_g:
                max_corr_g = cg
        if np.var(fe) > 1e-9:
            ce = abs(np.corrcoef(fe, ret_e)[0, 1])
            if not np.isnan(ce) and ce > max_corr_e:
                max_corr_e = ce

    gain = max_corr_e - max_corr_g
    return float(max_corr_e), float(max_corr_g), float(gain)

def _check_bucket_viability(
    mask_high: np.ndarray,
    mask_low: np.ndarray,
    is_long: np.ndarray,
    timestamps: np.ndarray,
    cfg: Dict[str, Any]
) -> Tuple[bool, Dict[str, int]]:
    """Checks downstream limits."""
    n_lh = np.sum(mask_high & is_long)
    n_sh = np.sum(mask_high & ~is_long)
    n_ll = np.sum(mask_low & is_long)
    n_sl = np.sum(mask_low & ~is_long)

    min_tot = cfg.get("min_bucket_samples_total", 100)

    counts = {
        "long_high": int(n_lh),
        "short_high": int(n_sh),
        "long_low": int(n_ll),
        "short_low": int(n_sl)
    }

    # We require ALL 4 buckets to be viable, or else it breaks the symmetrical pipeline expectation.
    viability = all(c >= min_tot for c in counts.values())
    return viability, counts

# -----------------------------------------------------------------------------
# MAIN OPTIMIZER ENTRY POINT
# -----------------------------------------------------------------------------

def optimize_layer0_masks(
    data: pd.DataFrame,
    features: np.ndarray,
    forward_returns: np.ndarray,
    cfg: Dict[str, Any]
) -> Dict[str, Any]:

    from extreme_price_movements.utils import tprint
    tprint("=" * 80)
    tprint("LAYER 0: DIRECTIONAL IMPULSE MASK OPTIMIZATION")
    tprint("=" * 80)

    high = data["high"].values.astype(np.float32)
    low = data["low"].values.astype(np.float32)
    close = data["close"].values.astype(np.float32)
    is_long = data["is_long"].values.astype(bool)
    timestamps = pd.to_datetime(data.get("timestamp", data.index)).values

    # Precompute global baselines
    ret_1 = np.where(np.roll(close, 1) > 0, (close - np.roll(close, 1)) / np.roll(close, 1), 0)
    ret_1[0] = 0.0
    vol_g = rolling_std_nb(ret_1, 24)

    candidates = []

    # Generate the parameter grid from config
    grid = []
    for z in cfg.get("z_hours_grid", [8, 12, 16]):
        for w in cfg.get("top_w_pct_grid", [4, 6, 8]):
            grid.append((z, "top_movers", w))
        for x in cfg.get("x_std_grid", [1.4, 1.6, 1.8]):
            grid.append((z, "std_threshold", x))
        for y in cfg.get("y_move_pct_grid", [4.0, 5.5, 7.0]):
            grid.append((z, "abs_move_threshold", y))

    tprint(f"Evaluating {len(grid)} candidates...")

    for (z, fam, param) in grid:

        # 1. Kinematics
        high_val, high_idx = rolling_max_index_nb(high, z)
        low_val, low_idx = rolling_min_index_nb(low, z)

        start_idx = np.maximum(0, np.arange(len(close)) - z + 1)
        start_px = close[start_idx]

        up_move = np.where(start_px > 1e-9, (high_val - start_px) / start_px, 0.0)
        dn_move = np.where(start_px > 1e-9, (start_px - low_val) / start_px, 0.0)
        rng_move = np.where(start_px > 1e-9, (high_val - low_val) / start_px, 0.0)

        # 2. Mask generation
        m_high, m_low = _generate_event_masks(fam, param, up_move, dn_move, vol_g)
        m_any = m_high | m_low

        tot_events = int(np.sum(m_any))
        if tot_events < cfg.get("min_total_events", 300):
            continue

        # Active days
        ts_any = pd.to_datetime(timestamps[m_any])
        days_any = ts_any.floor("D").nunique()
        tot_days = pd.to_datetime(timestamps).floor("D").nunique()
        active_frac = days_any / max(1, tot_days)

        if active_frac < cfg.get("min_active_days_fraction", 0.20):
            continue

        events_p_day_mean = tot_events / max(1, days_any)

        if not (cfg.get("min_events_per_day", 1) <= events_p_day_mean <= cfg.get("max_events_per_day", 50)):
            continue

        # 3. Coherence
        b_up, b_dn, s_up, s_dn, m_up, m_dn, v_exp, _ = compute_impulse_coherence_nb(
            ret_1, vol_g, high_idx, low_idx, start_idx, z
        )

        coh_mets = _compute_coherence_metrics(
            m_high, m_low, rng_move, b_up, b_dn, s_up, s_dn, m_up, m_dn, v_exp
        )

        # 4. Distinctness
        dist_score, dist_std, dist_tail = _compute_regime_distinctness(m_any, forward_returns)
        if cfg.get("enable_regime_distinctness_check", True) and dist_score < 1.1:
            continue

        # 5. Learnability
        l_evt, l_glob, l_gain = _compute_conditional_learnability(m_any, features, forward_returns)
        if cfg.get("enable_learnability_check", True) and l_gain <= 0:
            continue

        # 6. Bucket Viability
        v_pass, v_counts = _check_bucket_viability(m_high, m_low, is_long, timestamps, cfg)
        if cfg.get("enable_bucket_viability_check", True) and not v_pass:
            continue

        # Create row
        row = {
            "name": f"{fam}_z{z}_p{param}",
            "family": fam,
            "z_hours": z,
            "param": param,
            "total_events": tot_events,
            "events_per_day_mean": events_p_day_mean,
            "active_days_fraction": active_frac,
            "impulse_shape_dispersion": coh_mets["impulse_shape_dispersion"],
            "post_event_vol_dispersion": coh_mets["post_event_vol_dispersion"],
            "regime_distinctness_score": dist_score,
            "conditional_learnability_event": l_evt,
            "conditional_learnability_global": l_glob,
            "conditional_learnability_gain": l_gain,
            "long_high_count": v_counts["long_high"],
            "short_high_count": v_counts["short_high"],
            "long_low_count": v_counts["long_low"],
            "short_low_count": v_counts["short_low"],
            "bucket_viability_pass": v_pass,
            "layer0_acceptance_pass": True,
            "m_high": m_high,
            "m_low": m_low
        }
        candidates.append(row)

    if not candidates:
        tprint("  [ERROR] No Layer 0 candidates passed acceptance criteria.")
        return {
            "status": "failed",
            "reason": "zero_candidates_passed"
        }

    df = pd.DataFrame(candidates)

    # 7. Heterogeneity Caps (Dispersion Filter)
    q_thresh = df["impulse_shape_dispersion"].quantile(cfg.get("max_allowed_dispersion_quantile", 0.75))
    df = df[df["impulse_shape_dispersion"] <= q_thresh].copy()

    if df.empty:
         tprint("  [ERROR] All candidates failed dispersion caps.")
         return {"status": "failed", "reason": "dispersion_caps"}

    # 8. Shortlist Selection (Standardized Composite)
    def _z(col):
        if df[col].std() < 1e-9: return np.zeros(len(df))
        return (df[col] - df[col].mean()) / df[col].std()

    df["shortlist_score"] = (
        _z("active_days_fraction") + _z("events_per_day_mean")
        - _z("impulse_shape_dispersion") - _z("post_event_vol_dispersion")
        + _z("conditional_learnability_gain") + _z("regime_distinctness_score")
    )

    df = df.sort_values("shortlist_score", ascending=False)

    shortlist_idx = []
    fam_counts = {"top_movers": 0, "std_threshold": 0, "abs_move_threshold": 0}
    max_tot = cfg.get("shortlist_max_candidates", 5)
    max_fam = cfg.get("shortlist_max_per_family", 2)

    for i, row in df.iterrows():
        if len(shortlist_idx) >= max_tot:
            break
        if fam_counts[row["family"]] < max_fam:
            fam_counts[row["family"]] += 1
            shortlist_idx.append(i)

    df_shortlist = df.loc[shortlist_idx].copy()

    best_config = df_shortlist.iloc[0].to_dict()
    best_mask_high = best_config.pop("m_high")
    best_mask_low = best_config.pop("m_low")

    # Remove masks from dataframes for storage
    df = df.drop(columns=["m_high", "m_low"])
    df_shortlist = df_shortlist.drop(columns=["m_high", "m_low"])

    tprint(f"Layer 0 Optimization Complete. Best Candidate: {best_config['name']} (Score: {best_config['shortlist_score']:.4f})")

    return {
        "status": "ok",
        "layer0_candidate_table_": df,
        "layer0_shortlist_": df_shortlist,
        "layer0_best_config_": best_config,
        "layer0_best_mask_high_": best_mask_high,
        "layer0_best_mask_low_": best_mask_low,
    }
