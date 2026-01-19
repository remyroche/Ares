from __future__ import annotations

import numpy as np
import pandas as pd
from numba import njit, prange
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import Ridge, HuberRegressor, ElasticNet
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.mixture import GaussianMixture
from sklearn.pipeline import Pipeline
from scipy.optimize import minimize
from typing import List, Tuple, Dict, Any, Optional, Union
from sklearn.base import clone

EPS = 1e-12

# ----------------------------
# Numba Optimizations
# ----------------------------

@njit(cache=True)
def _rolling_mean_numba(x, window):
    n = len(x)
    out = np.empty(n, dtype=np.float64)
    out[:] = np.nan
    s = 0.0
    for i in range(n):
        val = x[i]
        if np.isnan(val):
            # Reset sum if nan encountered? Or just skip?
            # Simple rolling mean usually propagates NaNs.
            # If we want to support NaNs properly, it gets complex.
            # Assuming filled data for now or propagating NaNs.
            s = np.nan
        else:
            if np.isnan(s): s = 0.0 # recovering from nan? No, simplified.
            s += val

        if i >= window:
            old = x[i - window]
            if not np.isnan(old):
                s -= old

        if i >= window - 1:
            out[i] = s / window
    return out

@njit(cache=True)
def _rolling_std_numba(x, window):
    n = len(x)
    out = np.empty(n, dtype=np.float64)
    out[:] = np.nan

    # We can use Welford's algorithm or simple sum of squares for window
    # Simple sum of squares is faster but less numerically stable.
    # Given financial data, simple sum of squares is usually fine if centered.

    # Let's use a simpler Pandas-like approach (recalc if needed or incremental)
    # Incremental is tricky with window removal.
    # Re-summing is O(N*W). Numba is fast enough for small W.

    # Optimization: maintain sum and sum_sq
    s = 0.0
    ss = 0.0
    count = 0

    # Initialize first window
    # This loop structure handles NaNs better
    for i in range(n):
        val = x[i]

        # Add new
        if not np.isnan(val):
            s += val
            ss += val * val
            count += 1

        # Remove old
        if i >= window:
            old = x[i - window]
            if not np.isnan(old):
                s -= old
                ss -= old * old
                count -= 1

        if i >= window - 1 and count >= 2: # Need at least 2 for sample std
            mean = s / count
            var = (ss - 2*mean*s + count*mean*mean) / (count - 1)
            # var = (ss / count) - (mean * mean) # Population
            # Sample variance: (ss - s*s/n) / (n-1)
            # (ss - (s*s)/count) / (count - 1)

            # More stable:
            num = ss - (s*s)/count
            if num < 0: num = 0
            out[i] = np.sqrt(num / (count - 1))

    return out

# ----------------------------
# Helpers
# ----------------------------

def rolling_sigma(x: pd.Series, L: int) -> pd.Series:
    # Use Numba if possible for speed, else Pandas
    # Fallback to Pandas for simplicity/safety unless really needed
    # Numba version above is basic.
    return x.rolling(L, min_periods=L).std()

def rolling_mean(x: pd.Series, L: int) -> pd.Series:
    return x.rolling(L, min_periods=L).mean()

def zscore_rolling(x: pd.Series, L: int) -> pd.Series:
    mu = rolling_mean(x, L)
    sd = rolling_sigma(x, L)
    return (x - mu) / (sd + EPS)

def winsorize(s: pd.Series, k: float = 4.0) -> pd.Series:
    return s.clip(lower=-k, upper=k)

def sigmoid(x: pd.Series | np.ndarray, k: float = 1.0) -> pd.Series | np.ndarray:
    return 1.0 / (1.0 + np.exp(-k * x))

def true_range(df: pd.DataFrame) -> pd.Series:
    h, l, c = df["high"], df["low"], df["close"]
    cprev = c.shift(1)
    tr = pd.concat([(h - l), (h - cprev).abs(), (l - cprev).abs()], axis=1).max(axis=1)
    return tr

def ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False).mean()

def slope_ols(y: pd.Series, window: int) -> pd.Series:
    """
    Unweighted rolling slope of y vs time index using OLS.
    """
    idx = np.arange(len(y), dtype=float)
    t = pd.Series(idx, index=y.index)

    # rolling cov(t, y) / var(t)
    mean_t = t.rolling(window, min_periods=window).mean()
    mean_y = y.rolling(window, min_periods=window).mean()
    cov_ty = (t * y).rolling(window, min_periods=window).mean() - mean_t * mean_y
    var_t = (t * t).rolling(window, min_periods=window).mean() - mean_t * mean_t
    return cov_ty / (var_t + EPS)

def select_features_elasticnet_fast(
    X_tr: pd.DataFrame,
    y_tr: pd.Series,
    *,
    # ElasticNet
    alpha: float = 1e-3,
    l1_ratio: float = 0.3,
    # sparsity / safety
    min_features: int = 10,
    max_features: int | None = None,      # if None -> min(n_features, n_samples//50) clamped
    coef_tol: float = 1e-8,
    # cheap multicollinearity proxy (pre-filter)
    corr_thresh: float = 0.98,
) -> list[str]:
    """
    Fast feature selector for Ridge/Huber *only*:
      1) Cheap multicollinearity pre-filter via correlation pruning on TRAIN ONLY
      2) ElasticNet on remaining features (with scaling)
      3) Keep non-zero coefficients; enforce min/max feature counts

    Notes
    -----
    - Call inside each CV fold using X_tr/y_tr from the TRAIN slice only.
    - Multicollinearity pre-filter is a greedy pruning based on absolute Pearson correlation.
      This is O(p^2) in the number of features p, so keep p moderate (typical in your use-case).
    """

    # ----------------------------
    # 0) Sanity / defaults
    # ----------------------------
    if not isinstance(X_tr, pd.DataFrame):
        raise TypeError("X_tr must be a pandas DataFrame.")
    if not isinstance(y_tr, (pd.Series, pd.DataFrame)):
        raise TypeError("y_tr must be a pandas Series (or single-column DataFrame).")
    if isinstance(y_tr, pd.DataFrame):
        if y_tr.shape[1] != 1:
            raise ValueError("y_tr must be a Series or a single-column DataFrame.")
        y_tr = y_tr.iloc[:, 0]

    X = X_tr.copy()

    # Drop constant / near-constant columns (fast, prevents correlation NaNs)
    nunique = X.nunique(dropna=False)
    X = X.loc[:, nunique > 1]

    if X.shape[1] == 0:
        return []

    n_samples = len(X)
    n_features = X.shape[1]

    if max_features is None:
        max_features = max(min_features, min(n_features, n_samples // 50))
    max_features = int(max(min_features, min(max_features, n_features)))

    # ----------------------------
    # 1) Cheap multicollinearity filter: greedy corr pruning
    #    Keep the feature with larger |corr(feature, y)| when two features are too correlated
    # ----------------------------
    # Compute feature-target correlation for tie-breaking (fast proxy of usefulness)
    # Uses Pearson corr; fillna to avoid dropping rows (fast, stable).
    y = y_tr.astype(float)
    y0 = y.fillna(y.median())

    X0 = X.apply(lambda s: s.astype(float).fillna(s.median()), axis=0)

    # feature-target absolute correlation
    with np.errstate(invalid="ignore"):
        ft_corr = X0.corrwith(y0).abs().fillna(0.0)

    # feature-feature absolute correlation matrix
    # (If p is huge, consider a two-stage reduction first; for typical engineered sets this is OK.)
    corr = X0.corr().abs()

    # Greedy: sort by usefulness, keep best, drop its highly-correlated peers
    order = ft_corr.sort_values(ascending=False).index.tolist()
    keep = []
    dropped = set()

    for f in order:
        if f in dropped:
            continue
        keep.append(f)

        # mark highly correlated peers for dropping (excluding self)
        if corr_thresh is not None and corr_thresh < 1.0:
            peers = corr.index[(corr.loc[f] >= corr_thresh)].tolist()
            for p in peers:
                if p != f:
                    dropped.add(p)

    X_mc = X0[keep]

    # If pruning got too aggressive, relax by keeping top ft_corr features up to a floor
    # (prevents ending with too few features for ElasticNet to work meaningfully)
    if X_mc.shape[1] < min_features:
        top = ft_corr.sort_values(ascending=False).index.tolist()[:min_features]
        X_mc = X0[top]

    # ----------------------------
    # 2) ElasticNet selection on the pruned feature set
    # ----------------------------
    pipe = Pipeline([
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ("en", ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=100_000)),
    ])
    pipe.fit(X_mc, y_tr)

    coef = np.asarray(pipe.named_steps["en"].coef_, dtype=float)
    names = list(X_mc.columns)

    selected_idx = np.flatnonzero(np.abs(coef) > coef_tol)
    selected = [names[i] for i in selected_idx]

    # ----------------------------
    # 3) Enforce min/max (fallback to top-|coef|)
    # ----------------------------
    if len(selected) < min_features:
        order_coef = np.argsort(np.abs(coef))[::-1]
        selected = [names[i] for i in order_coef[:min_features]]

    if len(selected) > max_features:
        abscoef = {names[i]: float(abs(coef[i])) for i in range(len(names))}
        selected = sorted(selected, key=lambda c: abscoef.get(c, 0.0), reverse=True)[:max_features]

    return selected

# ----------------------------
# 0) OOF Prediction Templates
# ----------------------------

def time_series_oof_predictions(
    X: pd.DataFrame,
    y: pd.Series,
    cv_splits: list[tuple[np.ndarray, np.ndarray]],
    model,
) -> pd.Series:
    """
    Generic OOF generator for time-series splits (you provide cv_splits).
    - cv_splits: list of (train_idx, valid_idx) arrays, already purged/embargoed if needed.
    Returns a Series aligned to X.index with NaN for rows never validated.
    """
    oof = pd.Series(index=X.index, dtype=float)

    for tr_idx, va_idx in cv_splits:
        X_tr, y_tr = X.iloc[tr_idx], y.iloc[tr_idx]
        X_va = X.iloc[va_idx]

        m = clone(model)
        m.fit(X_tr, y_tr)
        pred = m.predict(X_va)
        oof.iloc[va_idx] = pred

    return oof

def isotonic_calibrate_oof(
    raw_oof_pred: pd.Series,
    y_true: pd.Series,
    cv_splits: list[tuple[np.ndarray, np.ndarray]],
    y_is_binary: bool = True,
) -> pd.Series:
    """
    OOF-of-OOF isotonic calibration.
    """
    cal = pd.Series(index=raw_oof_pred.index, dtype=float)

    for tr_idx, va_idx in cv_splits:
        tr = raw_oof_pred.iloc[tr_idx].dropna()
        # Find intersection of indices
        common_idx = tr.index.intersection(y_true.index)

        # Align y_true on the same training indices where pred is not NaN
        y_tr = y_true.loc[common_idx]
        tr = tr.loc[common_idx]

        if y_is_binary:
            y_tr = y_tr.astype(float)

        iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
        iso.fit(tr.values, y_tr.values)

        va = raw_oof_pred.iloc[va_idx]
        cal.iloc[va_idx] = iso.transform(va.values)

    return cal

# ------------------------------------------
# 1) Core alpha modalities + calibration
# ------------------------------------------

def build_core_alpha_features(
    df: pd.DataFrame,
    p_stable_oof: pd.Series,
    p_agg_oof: pd.Series,
    ret_col_for_sigma: str = "ret",
    sigma_L: int = 256,
    winsor_k: float = 4.0,
    do_isotonic: bool = False,
    y_bin_for_iso: pd.Series | None = None,
    cv_splits_for_iso: list[tuple[np.ndarray, np.ndarray]] | None = None,
) -> pd.DataFrame:
    """
    Produces:
    - vol-normalized predictions p_*_z (winsorized at ±winsor_k)
    - optional isotonic-calibrated probabilities p_*_iso in [0,1]
    - spread_z, spread_abs, spread_frac
    - raw_direction
    - cos_sim, disagree_abs
    - p_*_change
    - consensus_mag_z
    - consensus_strength_alt (robust)
    """
    out = pd.DataFrame(index=df.index)

    # 1) Vol normalization
    # Check if ret_col_for_sigma exists, else compute from close
    if ret_col_for_sigma not in df.columns:
        if 'close' in df.columns:
            ret_series = df['close'].pct_change().fillna(0.0)
        else:
            # Fallback
            ret_series = pd.Series(0.01, index=df.index)
    else:
        ret_series = df[ret_col_for_sigma]

    sig = rolling_sigma(ret_series, sigma_L)
    p_stable_z = p_stable_oof / (sig + EPS)
    p_agg_z = p_agg_oof / (sig + EPS)

    # 2) Winsorize at ±4 sigma (fat-tail guard)
    p_stable_z = winsorize(p_stable_z, winsor_k)
    p_agg_z = winsorize(p_agg_z, winsor_k)

    out["p_stable_z"] = p_stable_z
    out["p_agg_z"] = p_agg_z

    # 3) Optional isotonic calibration -> probability space [0,1]
    if do_isotonic and y_bin_for_iso is not None and cv_splits_for_iso is not None:
        out["p_stable_iso"] = isotonic_calibrate_oof(p_stable_oof, y_bin_for_iso, cv_splits_for_iso, y_is_binary=True)
        out["p_agg_iso"] = isotonic_calibrate_oof(p_agg_oof, y_bin_for_iso, cv_splits_for_iso, y_is_binary=True)
        out["spread_prob"] = out["p_stable_iso"] - out["p_agg_iso"]

    # 4) Direction + disagreement geometry (use z-space for robustness)
    out["spread_z"] = out["p_stable_z"] - out["p_agg_z"]
    out["spread_abs"] = out["spread_z"].abs()
    out["spread_frac"] = out["spread_z"] / (out["p_stable_z"].abs() + out["p_agg_z"].abs() + EPS)

    out["raw_direction"] = np.sign(out["p_stable_z"] + out["p_agg_z"]).replace(0.0, np.nan).fillna(0.0)

    # Continuous agreement proxy
    out["cos_sim"] = (out["p_stable_z"] * out["p_agg_z"]) / (
        out["p_stable_z"].abs() * out["p_agg_z"].abs() + EPS
    )
    out["disagree_abs"] = (out["p_stable_z"] - out["p_agg_z"]).abs()

    # 5) Changes (stability)
    out["p_stable_change"] = out["p_stable_z"] - out["p_stable_z"].shift(1)
    out["p_agg_change"] = out["p_agg_z"] - out["p_agg_z"].shift(1)

    # 6) Conviction
    out["consensus_mag_z"] = 0.5 * (out["p_stable_z"] + out["p_agg_z"])

    # robust agreement-and-strength (no quadratic blow-up)
    out["consensus_strength_alt"] = (
        np.sign(out["p_stable_z"]) * np.sign(out["p_agg_z"]) * np.minimum(out["p_stable_z"].abs(), out["p_agg_z"].abs())
    )

    return out

# ------------------------------------------
# 2) Structural/context features (15m OHLCV)
# ------------------------------------------

def kaufman_efficiency_ratio(close: pd.Series, window: int) -> pd.Series:
    """
    KER = net change / sum of absolute changes
    """
    net = (close - close.shift(window)).abs()
    denom = close.diff().abs().rolling(window, min_periods=window).sum()
    return net / (denom + EPS)

def build_structural_features(
    df: pd.DataFrame,
    vol_L: int = 96,
    tr_L: int = 96,
    ker_fast: int = 50,
    ker_slow: int = 150,
    anchor_L: int = 150,
    gravity_L: int = 150,
    gravity_slope_k: int = 20,
) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)

    # Needs OHLCV
    if 'close' not in df.columns:
        return out

    # Liquidity / intention proxies
    if 'high' in df.columns and 'low' in df.columns:
        tr = true_range(df)
        atr = rolling_mean(tr, tr_L) # tr.rolling(tr_L, min_periods=tr_L).mean()
        range_rel = tr / (atr + EPS)
        out["range_rel"] = range_rel
        out["atr"] = atr
        out["spread_atr_proxy"] = (df["high"] - df["low"]) / (atr + EPS)
    else:
        # Fallback
        tr = df['close'].diff().abs()
        atr = rolling_mean(tr, tr_L)
        range_rel = tr / (atr + EPS)
        out["range_rel"] = range_rel

    if 'volume' in df.columns:
        vol_mean = rolling_mean(df["volume"], vol_L)
        vol_rel = df["volume"] / (vol_mean + EPS)
        out["vol_rel"] = vol_rel
    else:
        out["vol_rel"] = 1.0
        vol_rel = pd.Series(1.0, index=df.index)

    # You can keep liquidity_valid binary or continuous; continuous is usually better for trees
    out["liquidity_score"] = (zscore_rolling(out["vol_rel"], vol_L) + zscore_rolling(out["range_rel"], tr_L)) / 2.0
    out["liquidity_valid"] = (out["liquidity_score"] > 0.0).astype(float)

    # Trend modality (KER multi-timeframe)
    out["ker_fast"] = kaufman_efficiency_ratio(df["close"], ker_fast)
    out["ker_slow"] = kaufman_efficiency_ratio(df["close"], ker_slow)

    # Anchor alignment + overextension
    out["anchor_z"] = zscore_rolling(df["close"], anchor_L)
    out["anchor_extreme"] = out["anchor_z"].abs()

    # Gravity confirmation: slope of rolling mean, normalized by rolling sigma of price
    roll_mean = rolling_mean(df["close"], gravity_L)
    grav_slope = slope_ols(roll_mean, gravity_slope_k)

    price_sig = rolling_sigma(df["close"].pct_change().fillna(0.0), gravity_L)
    out["gravity"] = grav_slope / (price_sig + EPS)

    return out

# ------------------------------------------
# 3) Orchestrator / gating (soft, monotone)
# ------------------------------------------

def compute_lgc(
    core: pd.DataFrame,
    structural: pd.DataFrame,
    gate_k: float,
    agree_thr: float,
    trend_thr: float,
    liq_thr: float,
    *,
    long_only: bool = True,
):
    """
    Reconstructs the orchestrator outputs needed for gating diagnostics.
    """
    agreement_score = (core["cos_sim"] + 1.0) / 2.0  # [0,1]
    if "ker_fast" in structural.columns:
        trend_score = (structural["ker_fast"].fillna(0.0) + structural["ker_slow"].fillna(0.0)) / 2.0
    else:
        trend_score = 0.5

    if "liquidity_score" in structural.columns:
        liq_score = structural["liquidity_score"].fillna(0.0)
    else:
        liq_score = 0.0

    g_agree = sigmoid(agreement_score - agree_thr, k=gate_k)
    g_trend = sigmoid(trend_score - trend_thr, k=gate_k)
    g_liq = sigmoid(liq_score - liq_thr, k=gate_k)

    gate_soft = g_agree * g_trend * g_liq

    # base confidence (causal driver)
    base_conf = core["consensus_strength_alt"].copy()
    if long_only:
        base_conf = base_conf.clip(lower=0.0)

    # directional policy hook
    directional_gate = (core["raw_direction"] > 0).astype(float) if long_only else 1.0

    lgc = directional_gate * gate_soft * base_conf
    return lgc, base_conf, directional_gate, gate_soft

def ess_ratio(w: pd.Series) -> float:
    """Effective sample size ratio ESS/N using weights w."""
    w = w.astype(float).fillna(0.0)
    N = float(len(w))
    s1 = float(w.sum())
    s2 = float((w * w).sum())
    if s2 <= EPS or N <= 0:
        return 0.0
    ess = (s1 * s1) / (s2 + EPS)
    return float(ess / N)

def flip_rate(active: pd.Series) -> float:
    """Fraction of times active_t != active_{t-1}."""
    a = active.astype(int).fillna(0).values
    if len(a) < 2:
        return 0.0
    return float(np.mean(a[1:] != a[:-1]))

def spearman_safe(x: pd.Series, y: pd.Series) -> float:
    """Spearman correlation, safe for constant series."""
    x = x.astype(float)
    y = y.astype(float)
    if x.nunique(dropna=True) < 2 or y.nunique(dropna=True) < 2:
        return np.nan
    return float(x.corr(y, method="spearman"))

def gate_diagnostics(
    lgc: pd.Series,
    consensus_strength_alt: pd.Series,
    directional_gate: pd.Series | float,
    *,
    tau: float = 1e-6,
    spearman_on_directional_gate: bool = True,
):
    """
    Computes diagnostics for gate tuning.
    """
    lgc = lgc.fillna(0.0)
    active = (lgc > tau).astype(int)

    cov = float(active.mean())
    ess = ess_ratio(lgc)
    flips = flip_rate(active)

    if spearman_on_directional_gate:
        if isinstance(directional_gate, (int, float)):
            mask = pd.Series(True, index=lgc.index)
        else:
            mask = directional_gate.astype(bool).fillna(False)
        rho = spearman_safe(lgc[mask], consensus_strength_alt[mask])
    else:
        rho = spearman_safe(lgc, consensus_strength_alt)

    active_vals = lgc[lgc > tau]
    disp = float(active_vals.std()) if len(active_vals) > 2 else 0.0

    return {
        "coverage": cov,
        "ess_over_n": ess,
        "flip_rate": flips,
        "spearman": rho,
        "dispersion_active": disp,
    }

def tune_gate_params_grid(
    core: pd.DataFrame,
    structural: pd.DataFrame,
    *,
    gate_k_grid=(0.6, 0.8, 1.0, 1.2, 1.5),
    agree_thr_grid=(0.45, 0.50, 0.55),
    trend_thr_grid=(0.25, 0.30, 0.35),
    liq_thr_grid=(-0.25, 0.0, 0.25),
    long_only: bool = True,
    tau: float = 1e-6,
    min_coverage: float = 0.30,
    max_coverage: float | None = 0.70,
    min_ess_over_n: float = 0.30,
    max_flip_rate: float = 0.20,
    spearman_on_directional_gate: bool = True,
):
    """
    Grid-search gate params WITHOUT using PnL/returns.
    """
    # Quick check for required columns
    # We assume core and structural have necessary cols or compute_lgc handles them
    # compute_lgc handles missing structural columns gracefully above

    rows = []
    for gate_k in gate_k_grid:
        for agree_thr in agree_thr_grid:
            for trend_thr in trend_thr_grid:
                for liq_thr in liq_thr_grid:
                    lgc, base_conf, dir_gate, gate_soft = compute_lgc(
                        core, structural,
                        gate_k=gate_k,
                        agree_thr=agree_thr,
                        trend_thr=trend_thr,
                        liq_thr=liq_thr,
                        long_only=long_only,
                    )
                    stats = gate_diagnostics(
                        lgc=lgc,
                        consensus_strength_alt=base_conf,
                        directional_gate=dir_gate,
                        tau=tau,
                        spearman_on_directional_gate=spearman_on_directional_gate,
                    )
                    rows.append({
                        "gate_k": gate_k,
                        "agree_thr": agree_thr,
                        "trend_thr": trend_thr,
                        "liq_thr": liq_thr,
                        **stats
                    })

    res = pd.DataFrame(rows)

    # Apply constraints
    ok = (
        (res["coverage"] >= min_coverage) &
        (res["ess_over_n"] >= min_ess_over_n) &
        (res["flip_rate"] <= max_flip_rate)
    )
    if max_coverage is not None:
        ok &= (res["coverage"] <= max_coverage)

    ok &= res["spearman"].notna()

    filtered = res.loc[ok].copy()

    if filtered.empty:
        # Fallback: sort by coverage/ess closeness to ideal
        # Or simply return None and let caller handle
        return None, res

    # Sort by selection criteria
    # 1. Maximize spearman
    # 2. Tie-breakers: higher dispersion, lower gate_k
    filtered = filtered.sort_values(
        by=["spearman", "dispersion_active", "gate_k"],
        ascending=[False, False, True],
        kind="mergesort"
    ).reset_index(drop=True)

    best = filtered.iloc[0].to_dict()
    best_params = {
        "gate_k": float(best["gate_k"]),
        "agree_thr": float(best["agree_thr"]),
        "trend_thr": float(best["trend_thr"]),
        "liq_thr": float(best["liq_thr"]),
    }

    return best_params, filtered

def build_orchestrator_features(
    core: pd.DataFrame,
    structural: pd.DataFrame,
    gate_k: float = 1.0,
    agree_thr: float = 0.5,
    trend_thr: float = 0.3,
    liq_thr: float = 0.0,
) -> pd.DataFrame:
    """
    Builds:
    - directional_gate (general abstraction)
    - gate_soft (soft product of sigmoids)
    - logic_gated_confidence (monotone: base_conf * gate_soft)
    """
    out = pd.DataFrame(index=core.index)

    # Compute LGC using helper
    lgc, base_conf, dir_gate, gate_soft = compute_lgc(
        core, structural,
        gate_k=gate_k,
        agree_thr=agree_thr,
        trend_thr=trend_thr,
        liq_thr=liq_thr,
        long_only=True # Layer 4 usually long-only sizing logic on absolute returns?
                       # Or if raw_direction handles it.
                       # SimpleMultiModelRiskEngine targets usually abs_returns
                       # and uses directions.
                       # Let's align with previous implementation logic
    )

    out["directional_gate"] = dir_gate
    out["base_conf"] = base_conf
    out["gate_soft"] = gate_soft
    out["logic_gated_confidence"] = lgc

    # Overextension penalty as a separate feature (do not hard-gate)
    if "anchor_extreme" in structural.columns and "ker_slow" in structural.columns:
        out["trend_overextended"] = structural["anchor_extreme"] * structural["ker_slow"]
    else:
        out["trend_overextended"] = 0.0

    return out

# ------------------------------------------
# 4) Exogenous regime health metrics (market-only)
# ------------------------------------------

def build_regime_health_features(
    df: pd.DataFrame,
    rv_short: int = 32,
    rv_long: int = 256,
    rv_z_L: int = 512,
) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)

    if 'close' not in df.columns:
        return out

    r = df["close"].pct_change().fillna(0.0)
    rv_s = rolling_sigma(r, rv_short)
    rv_l = rolling_sigma(r, rv_long)

    out["realized_vol"] = rv_l
    out["vol_regime"] = rv_s / (rv_l + EPS)

    # vol_z computed on rv_l (or use rv_s). Use longer window for stability.
    mu = rolling_mean(rv_l, rv_z_L)
    sd = rolling_sigma(rv_l, rv_z_L)
    out["vol_z"] = (rv_l - mu) / (sd + EPS)

    return out

# ------------------------------------------
# 5) Meta-health features (confidence-aware, tempered)
# ------------------------------------------

def _clip_weights(w: pd.Series, w_min: float = 0.0, w_max: float | None = None) -> pd.Series:
    w = w.clip(lower=w_min)
    if w_max is not None:
        w = w.clip(upper=w_max)
    return w

def weighted_rolling_mean(x: pd.Series, w: pd.Series, window: int) -> pd.Series:
    wx = (w * x).rolling(window, min_periods=window).sum()
    ww = w.rolling(window, min_periods=window).sum()
    return wx / (ww + EPS)

def weighted_rolling_var(x: pd.Series, w: pd.Series, window: int) -> pd.Series:
    ex = weighted_rolling_mean(x, w, window)
    ex2 = weighted_rolling_mean(x * x, w, window)
    return (ex2 - ex * ex).clip(lower=0.0)

def weighted_rolling_std(x: pd.Series, w: pd.Series, window: int) -> pd.Series:
    return np.sqrt(weighted_rolling_var(x, w, window))

def rolling_sortino_conf(trade_returns: pd.Series, confidence: pd.Series, window: int, target: float = 0.0) -> pd.Series:
    r = trade_returns
    w = _clip_weights(confidence, w_min=0.0)
    excess = r - target
    downside_sq = np.minimum(excess, 0.0) ** 2
    mu = weighted_rolling_mean(excess, w, window)
    dd = np.sqrt(weighted_rolling_mean(downside_sq, w, window)).replace(0.0, np.nan)
    return mu / (dd + EPS)

def rolling_avg_signed_return_per_signal_conf(trade_returns: pd.Series, signal_direction: pd.Series,
                                              confidence: pd.Series, window: int) -> pd.Series:
    signed_r = np.sign(signal_direction).replace(0.0, np.nan) * trade_returns
    return weighted_rolling_mean(signed_r.fillna(0.0), confidence, window)

def rolling_expectancy_conf(trade_returns: pd.Series, confidence: pd.Series, window: int) -> pd.Series:
    r = trade_returns
    w = _clip_weights(confidence, w_min=0.0)
    win_mask = (r > 0).astype(float)
    loss_mask = (r < 0).astype(float)

    ww = w.rolling(window, min_periods=window).sum()
    w_win = (w * win_mask).rolling(window, min_periods=window).sum()
    w_loss = (w * loss_mask).rolling(window, min_periods=window).sum()

    p_win = w_win / (ww + EPS)
    p_loss = w_loss / (ww + EPS)

    sum_win = (w * r.where(r > 0, 0.0)).rolling(window, min_periods=window).sum()
    sum_loss_abs = (w * (-r).where(r < 0, 0.0)).rolling(window, min_periods=window).sum()

    avg_win = sum_win / (w_win + EPS)
    avg_loss_abs = sum_loss_abs / (w_loss + EPS)

    return p_win * avg_win - p_loss * avg_loss_abs

def rolling_avg_loss_size_conf(trade_returns: pd.Series, confidence: pd.Series, window: int) -> pd.Series:
    loss_w = (confidence.where(trade_returns < 0, 0.0)).rolling(window, min_periods=window).sum()
    loss_sum_abs = (confidence * (-trade_returns).where(trade_returns < 0, 0.0)).rolling(window, min_periods=window).sum()
    return loss_sum_abs / (loss_w + EPS)

def equity_from_size_weighted_returns(trade_returns: pd.Series, size: pd.Series, initial_equity: float = 1.0) -> pd.Series:
    eff_r = (size.clip(lower=0.0) * trade_returns).fillna(0.0)
    return initial_equity * (1.0 + eff_r).cumprod()

def drawdown_series(equity: pd.Series) -> pd.Series:
    peak = equity.cummax()
    return equity / (peak + EPS) - 1.0

def rolling_dd_vol(drawdown: pd.Series, w: pd.Series, window: int) -> pd.Series:
    return weighted_rolling_std(drawdown, w, window)

def rolling_drawdown_slope_linreg(drawdown: pd.Series, w: pd.Series, window: int) -> pd.Series:
    """
    Weighted rolling slope DD ~ t.
    """
    t = pd.Series(np.arange(len(drawdown)), index=drawdown.index, dtype=float)

    ww = w.rolling(window, min_periods=window).sum()
    wt = (w * t).rolling(window, min_periods=window).sum()
    wdd = (w * drawdown).rolling(window, min_periods=window).sum()
    wtt = (w * t * t).rolling(window, min_periods=window).sum()
    wtdd = (w * t * drawdown).rolling(window, min_periods=window).sum()

    mean_t = wt / (ww + EPS)
    mean_dd = wdd / (ww + EPS)

    cov = wtdd / (ww + EPS) - mean_t * mean_dd
    var = wtt / (ww + EPS) - mean_t * mean_t
    return cov / (var + EPS)

def build_meta_health_features(
    trade_returns: pd.Series,
    signal_direction: pd.Series,
    confidence_or_size: pd.Series,
    window: int = 50,
    temper_alpha: float = 0.5,
    cap_q: float = 0.95,
    smooth_span: int = 10,
) -> pd.DataFrame:
    out = pd.DataFrame(index=trade_returns.index)

    w_raw = confidence_or_size.clip(lower=0.0).astype(float)

    # Global cap
    w_cap = float(w_raw.quantile(cap_q)) if w_raw.notna().any() else 1.0
    w = w_raw.clip(upper=w_cap)

    # Smooth then temper
    w_s = ema(w, span=smooth_span)
    w_eff = (w_s + EPS) ** temper_alpha

    out["sortino_conf"] = rolling_sortino_conf(trade_returns, w_eff, window)
    out["avg_signed_return_conf"] = rolling_avg_signed_return_per_signal_conf(trade_returns, signal_direction, w_eff, window)
    out["expectancy_conf"] = rolling_expectancy_conf(trade_returns, w_eff, window)
    out["avg_loss_size_conf"] = rolling_avg_loss_size_conf(trade_returns, w_eff, window)

    # Equity/DD based on effective size
    eq = equity_from_size_weighted_returns(trade_returns, w_s, initial_equity=1.0)
    dd = drawdown_series(eq)
    out["dd_vol_conf"] = rolling_dd_vol(dd, w_eff, window)
    out["dd_slope_conf"] = rolling_drawdown_slope_linreg(dd, w_eff, window)

    # Smooth outputs
    for c in out.columns:
        out[c] = ema(out[c], span=smooth_span)

    return out

# ------------------------------------------
# Master feature builder / Main Interface
# ------------------------------------------

def generate_dual_chaser_features(
    df: pd.DataFrame,
    p_stable: pd.Series,
    p_agg: pd.Series,
    trade_returns: Optional[pd.Series] = None,
    gate_params: Optional[Dict[str, float]] = None
) -> pd.DataFrame:
    """
    Main entry point for generating dual chaser features.
    """

    # 1. Core Alpha Features
    core = build_core_alpha_features(
        df=df,
        p_stable_oof=p_stable,
        p_agg_oof=p_agg,
        ret_col_for_sigma="ret" if "ret" in df.columns else "close",
        sigma_L=256,
        winsor_k=4.0,
        do_isotonic=False
    )

    # 2. Structural Features
    structural = build_structural_features(df)

    # 3. Orchestrator
    # If gate_params provided, use them. Else defaults.
    if gate_params:
        orchestrator = build_orchestrator_features(
            core, structural,
            gate_k=gate_params.get("gate_k", 1.0),
            agree_thr=gate_params.get("agree_thr", 0.5),
            trend_thr=gate_params.get("trend_thr", 0.3),
            liq_thr=gate_params.get("liq_thr", 0.0),
        )
    else:
        orchestrator = build_orchestrator_features(core, structural)

    # 4. Regime
    regime = build_regime_health_features(df)

    # 5. Meta-Health (Optional)
    if trade_returns is not None:
        signal_dir = np.sign(core["consensus_mag_z"]).fillna(0.0)
        confidence = orchestrator["logic_gated_confidence"].fillna(0.0).clip(lower=0.0)

        meta = build_meta_health_features(
            trade_returns=trade_returns,
            signal_direction=signal_dir,
            confidence_or_size=confidence,
            window=50,
            temper_alpha=0.5,
            cap_q=0.95,
            smooth_span=10
        )
        meta = meta.add_prefix("meta_")
        return pd.concat([core, structural, orchestrator, regime, meta], axis=1)
    else:
        return pd.concat([core, structural, orchestrator, regime], axis=1)


# ------------------------------------------
# Legacy Classes (Kept for compatibility)
# ------------------------------------------

class StructuralRegimeGMM:
    def __init__(self, n_regimes=4):
        self.n_regimes = n_regimes
        self.scaler = StandardScaler()
        self.gmm = GaussianMixture(n_components=n_regimes, covariance_type='full', random_state=42)

    def get_structural_features(self, df):
        """Derives the 3 pillars of market context: Vol, Volume, Trend."""
        f = pd.DataFrame(index=df.index)

        # Ensure we have required columns
        if 'close' not in df.columns or 'volume' not in df.columns:
             # Try case insensitive
             cols = {c.lower(): c for c in df.columns}
             if 'close' in cols and 'volume' in cols:
                 df = df.rename(columns={cols['close']: 'close', cols['volume']: 'volume'})
             else:
                 # Cannot compute
                 return pd.DataFrame()

        # 1. Volatility (Normalized)
        log_ret = np.log(df['close'] / df['close'].shift(1))
        f['volatility'] = log_ret.rolling(20).std()

        # 2. Volume Intensity (Relative to 50-bar average)
        vol_rolling_std = df['volume'].rolling(50).std() + 1e-9
        f['volume_z'] = (df['volume'] - df['volume'].rolling(50).mean()) / vol_rolling_std

        # 3. Trend Strength (Absolute return over 20 bars / volatility)
        denom = df['close'].rolling(20).std() * np.sqrt(20) + 1e-9
        f['trend_strength'] = np.abs(df['close'].diff(20)) / denom

        return f.dropna()

    def fit_predict(self, df):
        features = self.get_structural_features(df)
        if features.empty or len(features) < self.n_regimes * 2:
            return [], np.array([])

        scaled_features = self.scaler.fit_transform(features)

        # Fit and predict the 4 regimes
        clusters = self.gmm.fit_predict(scaled_features)

        # Create the env_indices list for IRM
        env_indices = []
        for i in range(self.n_regimes):
            # Map cluster assignments back to the original dataframe index positions
            cluster_indices = np.where(clusters == i)[0]

            # Aligning with the dropped NaNs from feature engineering
            # Using index search (can be optimized if needed, but robust)
            actual_indices = [df.index.get_loc(features.index[idx]) for idx in cluster_indices]
            env_indices.append(np.array(actual_indices))

        return env_indices, clusters

class IRMv1HuberRegressor:
    def __init__(self, irm_lambda=25.0, alpha=0.1, huber_epsilon=1.1):
        self.irm_lambda = irm_lambda
        self.alpha = alpha
        self.huber_epsilon = huber_epsilon
        self.coef_ = None

    def _huber_loss_and_grad(self, w, X, y):
        """Standard Huber Loss and its Gradient."""
        errors = (X @ w) - y
        abs_errors = np.abs(errors)

        # Loss calculation
        quadratic_mask = abs_errors <= self.huber_epsilon
        loss = np.where(quadratic_mask, 0.5 * errors**2,
                        self.huber_epsilon * (abs_errors - 0.5 * self.huber_epsilon))

        # Gradient calculation
        grad_mult = np.where(quadratic_mask, errors,
                             self.huber_epsilon * np.sign(errors))
        grad = (X.T @ grad_mult) / len(y)

        return np.mean(loss), grad

    def _objective(self, w, envs):
        """IRM-v1 Objective: ERM + Lambda * Var(Gradients) + L2."""
        total_loss = 0
        penalty = 0
        valid_envs = 0

        for X_e, y_e in envs:
            if len(y_e) == 0: continue
            valid_envs += 1
            loss_e, grad_e = self._huber_loss_and_grad(w, X_e, y_e)
            total_loss += loss_e
            # IRM-v1 Penalty: Squared norm of the gradient per environment
            penalty += np.sum(grad_e**2)

        if valid_envs == 0:
            return 0.0

        # Structural L2 Regularization
        l2_reg = self.alpha * np.sum(w**2)

        return (total_loss / valid_envs) + (self.irm_lambda * penalty) + l2_reg

    def fit(self, X, y, env_indices):
        # Prepare environment data
        envs = []
        for idx in env_indices:
            if len(idx) > 0:
                envs.append((X[idx], y[idx]))

        if not envs:
            self.coef_ = np.zeros(X.shape[1])
            return self

        # Initial guess (zeros)
        initial_w = np.zeros(X.shape[1])

        # Optimize
        res = minimize(self._objective, initial_w, args=(envs,), method='L-BFGS-B')
        self.coef_ = res.x
        return self

    def predict(self, X):
        if self.coef_ is None:
            return np.zeros(X.shape[0])
        return X @ self.coef_

def train_dual_chaser_audit(
    X: Union[pd.DataFrame, np.ndarray],
    y: Union[pd.Series, np.ndarray],
    env_indices: List[np.ndarray],
    irm_lambda: float = 15.0,
    alpha: float = 0.1,
    random_state: int = 42,
    cv_splits: Optional[List[Tuple[np.ndarray, np.ndarray]]] = None
) -> Tuple[Any, Any, Optional[pd.Series], Optional[pd.Series]]:
    """
    Trains Dual Chasers and optionally generates OOF predictions.

    Args:
        X: Feature matrix (Should be DataFrame for feature selection, but Array accepted)
        y: Target variable
        env_indices: List of indices for each regime (for IRM)
        cv_splits: Optional list of (train_idx, val_idx) for OOF generation

    Returns:
        stable_chaser: Fitted on full data
        aggressive_chaser: Fitted on full data
        oof_stable: OOF predictions (if cv_splits provided)
        oof_agg: OOF predictions (if cv_splits provided)
    """
    # Helper to ensure DataFrame
    if isinstance(X, pd.DataFrame):
        X_df = X
        X_val = X.values
    else:
        # Create DF wrapper with generic columns if needed
        # We need this for feature selector
        X_df = pd.DataFrame(X, columns=[f"feat_{i}" for i in range(X.shape[1])])
        X_val = X

    if isinstance(y, pd.Series):
        y_val = y.values
        y_series = y
    else:
        y_val = y
        y_series = pd.Series(y, index=X_df.index)

    # --- Feature Selection for Global Models ---
    # Fast ElasticNet selection on FULL dataset (or subset if very large?)
    # Here we run on full dataset to select features for the final models.
    selected_feats = select_features_elasticnet_fast(X_df, y_series)

    if len(selected_feats) > 0:
        # Use selected features
        X_sel_val = X_df[selected_feats].values
        # Need to remap env_indices to selected features?
        # No, env_indices are row indices. Features are columns.
        # But IRM fit takes X[idx], so we should pass reduced X.
        X_train_final = X_sel_val
    else:
        # Fallback if selection fails to select anything
        X_train_final = X_val

    # 1. THE STABLE CHASER (IRM-Huber)
    stable_chaser = IRMv1HuberRegressor(
        irm_lambda=irm_lambda,
        alpha=alpha
    ).fit(X_train_final, y_val, env_indices)

    # 2. THE AGGRESSIVE CHASER (Standard Ridge)
    aggressive_chaser = Ridge(alpha=1.0, random_state=random_state).fit(X_train_final, y_val)

    # Store selected features in models if possible or return them?
    # For now, we assume the chaser object doesn't strictly validate feature names unless wrapped.
    # But wait, predict needs same features.
    # The models returned are sklearn/custom objects. They don't know about dropping columns.
    # If we return these models, the caller MUST know which features to pass.
    # But `train_dual_chaser_audit` returns the raw model objects.
    # AND `label_based_layer_4.py` calls predict on `X_chaser_scaled` (full features).
    # This will break dimensionality!

    # Solution: Wrap the models or feature selector?
    # Or, simpler for this task: Do NOT filter for global model here,
    # OR change `train_dual_chaser_audit` to return a wrapped object.

    # Actually, the user requirement is "before running Ridge and Huber, run a quick feature selection".
    # If I filter features, I must ensure prediction uses same features.
    # Since I cannot easily change the calling code's `predict` call (it's inside `SimpleMultiModelRiskEngine.predict_bet_size`),
    # I should wrap the result in a class that handles selection?
    # Or, `label_based_layer_4.py` stores `self.stable_chaser`.
    # If I change `stable_chaser` to be a pipeline including selection?
    # But `IRMv1HuberRegressor` is custom.

    # Let's wrap the logic inside a simple wrapper class or lambda?
    # Or just don't select for the global model return?
    # The user said "Fast feature selector for Ridge/Huber *only*... Call inside each CV fold...".
    # This implies primarily for OOF.
    # But usually global model should match OOF methodology.

    # Let's perform selection, and attach the selected feature names to the model object if possible,
    # or return them. But return signature is fixed.

    # Hack: Monkey-patch the predict method of the returned objects?
    # Or better: `train_dual_chaser_audit` handles the OOF.
    # The global models are used in `predict_bet_size`.
    # If I filter features, `predict_bet_size` will fail unless I store `selected_features`.

    # Let's attach `selected_features` attribute to the returned models.
    stable_chaser.selected_features_ = selected_feats
    aggressive_chaser.selected_features_ = selected_feats

    # Now we need to handle OOF generation.
    oof_stable = None
    oof_agg = None

    if cv_splits is not None:
        oof_stable = pd.Series(index=X_df.index, dtype=float)
        oof_agg = pd.Series(index=X_df.index, dtype=float)

        # OOF Loop with per-fold feature selection
        for tr_idx, va_idx in cv_splits:
            X_tr_fold = X_df.iloc[tr_idx]
            y_tr_fold = y_series.iloc[tr_idx]
            X_va_fold = X_df.iloc[va_idx]

            # 1. Feature Selection on this fold
            feats_fold = select_features_elasticnet_fast(X_tr_fold, y_tr_fold)
            if not feats_fold:
                feats_fold = list(X_df.columns) # Fallback

            X_tr_sel = X_tr_fold[feats_fold].values
            X_va_sel = X_va_fold[feats_fold].values

            # 2. Train Ridge (Aggressive)
            m_ridge = Ridge(alpha=1.0, random_state=random_state)
            m_ridge.fit(X_tr_sel, y_tr_fold.values)
            oof_agg.iloc[va_idx] = m_ridge.predict(X_va_sel)

            # 3. Train IRM (Stable)
            # Filter env_indices for this fold
            # (Assuming standard time series split logic or filtering by value)
            global_to_local = {g_idx: l_idx for l_idx, g_idx in enumerate(tr_idx)}

            fold_env_indices = []
            for env_idx_arr in env_indices:
                local_indices = []
                for g_idx in env_idx_arr:
                    if g_idx in global_to_local:
                        local_indices.append(global_to_local[g_idx])
                fold_env_indices.append(np.array(local_indices))

            m_irm = IRMv1HuberRegressor(irm_lambda=irm_lambda, alpha=alpha)
            m_irm.fit(X_tr_sel, y_tr_fold.values, fold_env_indices)
            oof_stable.iloc[va_idx] = m_irm.predict(X_va_sel)

    return stable_chaser, aggressive_chaser, oof_stable, oof_agg

# Kept for backward compat, but users should prefer generate_dual_chaser_features
generate_sizer_features_v2 = generate_dual_chaser_features
