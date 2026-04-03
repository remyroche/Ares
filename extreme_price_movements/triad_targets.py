"""
Triad Targets Module for Regime Mining.

This module implements the triad target system for identifying predictive regimes
in price movements. The two targets (EFF, VAME) are independent continuous
regressands bounded in [0, 1].

Target Definitions:
- target_eff: Trend Efficiency - measures how efficiently price moves toward a final level
- target_vame: Expansion Sustainability - measures sustainability of price expansion

Critical Constraints:
- No future-on-future persistence
- No full future range as drawdown proxy for VAME
- No simplex-normalization of targets
- Targets remain independent continuous regressands in [0, 1]
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from extreme_price_movements.utils import tprint


# Constants
TRIAD_TARGET_NAMES: List[str] = ["target_eff", "target_vame"]


def _per_symbol_shift(series: pd.Series, symbol: pd.Series, periods: int) -> pd.Series:
    """
    Apply shift operation per symbol to avoid cross-symbol data leakage.
    
    Parameters
    ----------
    series : pd.Series
        Series to shift
    symbol : pd.Series
        Symbol column for grouping
    periods : int
        Number of periods to shift (negative for forward shift)
    
    Returns
    -------
    pd.Series
        Shifted series with NaN at symbol boundaries
    """
    result = pd.Series(np.nan, index=series.index, dtype=series.dtype)
    for _, idx in symbol.groupby(symbol).groups.items():
        result.iloc[idx] = series.iloc[idx].shift(periods)
    return result


def _per_symbol_rolling_apply(
    series: pd.Series,
    symbol: pd.Series,
    window: int,
    func,
    shift_periods: int = 0,
    min_periods: int = 1,
) -> pd.Series:
    """
    Apply rolling operation per symbol, then optionally shift.
    
    Parameters
    ----------
    series : pd.Series
        Series to apply rolling operation to
    symbol : pd.Series
        Symbol column for grouping
    window : int
        Rolling window size
    func : callable
        Function to apply (e.g., .max(), .min(), .sum())
    shift_periods : int
        Periods to shift after rolling (negative for forward shift)
    min_periods : int
        Minimum periods required for rolling calculation
    
    Returns
    -------
    pd.Series
        Result with NaN at symbol boundaries and where insufficient data
    """
    result = pd.Series(np.nan, index=series.index, dtype=np.float64)
    for _, idx in symbol.groupby(symbol).groups.items():
        idx_arr = np.asarray(idx)
        rolled = series.iloc[idx_arr].rolling(window, min_periods=min_periods)
        applied = func(rolled)
        if shift_periods != 0:
            applied = applied.shift(shift_periods)
        result.iloc[idx_arr] = applied
    return result


def _per_symbol_rolling_argmax(
    series: pd.Series,
    symbol: pd.Series,
    window: int,
    shift_periods: int = 0,
) -> pd.Series:
    """
    Compute rolling argmax per symbol.
    
    Parameters
    ----------
    series : pd.Series
        Series to compute argmax on
    symbol : pd.Series
        Symbol column for grouping
    window : int
        Rolling window size
    shift_periods : int
        Periods to shift after rolling (negative for forward shift)
    
    Returns
    -------
    pd.Series
        Argmax values (relative index within window)
    """
    result = pd.Series(np.nan, index=series.index, dtype=np.float64)
    for _, idx in symbol.groupby(symbol).groups.items():
        idx_arr = np.asarray(idx)
        rolled = series.iloc[idx_arr].rolling(window, min_periods=1)
        applied = rolled.apply(np.argmax, raw=True)
        if shift_periods != 0:
            applied = applied.shift(shift_periods)
        result.iloc[idx_arr] = applied
    return result


def _per_symbol_rolling_argmin(
    series: pd.Series,
    symbol: pd.Series,
    window: int,
    shift_periods: int = 0,
) -> pd.Series:
    """
    Compute rolling argmin per symbol.
    
    Parameters
    ----------
    series : pd.Series
        Series to compute argmin on
    symbol : pd.Series
        Symbol column for grouping
    window : int
        Rolling window size
    shift_periods : int
        Periods to shift after rolling (negative for forward shift)
    
    Returns
    -------
    pd.Series
        Argmin values (relative index within window)
    """
    result = pd.Series(np.nan, index=series.index, dtype=np.float64)
    for _, idx in symbol.groupby(symbol).groups.items():
        idx_arr = np.asarray(idx)
        rolled = series.iloc[idx_arr].rolling(window, min_periods=1)
        applied = rolled.apply(np.argmin, raw=True)
        if shift_periods != 0:
            applied = applied.shift(shift_periods)
        result.iloc[idx_arr] = applied
    return result


def sigmoid(x: np.ndarray | float) -> np.ndarray | float:
    """
    Compute sigmoid function element-wise.

    Parameters
    ----------
    x : np.ndarray or float
        Input values

    Returns
    -------
    np.ndarray or float
        Sigmoid of input: 1 / (1 + exp(-x))
    """
    return 1 / (1 + np.exp(-x))


def rolling_percentile_rank(
    series: pd.Series, window: int, min_periods: int = 100
) -> pd.Series:
    """
    Compute strictly backward-looking rolling empirical percentile rank.

    Parameters
    ----------
    series : pd.Series
        Input data series
    window : int
        Rolling window size
    min_periods : int
        Minimum number of observations in window required to have a value

    Returns
    -------
    pd.Series
        Percentile rank in [0, 1]. NaNs are propagated.
    """
    # Use pandas rank method on rolling window. We use pct=True.
    # However, rolling().rank() returns the rank of the *current* element
    # compared to the window ending at the current element, which is exactly
    # a backward-looking empirical percentile.
    return series.rolling(window, min_periods=min_periods).rank(pct=True)


def _per_symbol_rolling_percentile_rank(
    series: pd.Series,
    symbol: pd.Series,
    window: int,
    min_periods: int = 100,
) -> pd.Series:
    """Compute strictly backward-looking rolling percentile rank per symbol."""
    result = pd.Series(np.nan, index=series.index, dtype=np.float64)
    positions = np.arange(len(symbol))
    for sym in symbol.unique():
        mask = (symbol == sym).values
        idx_arr = positions[mask]
        vals = series.iloc[idx_arr]
        res = rolling_percentile_rank(vals, window=window, min_periods=min_periods)
        result.iloc[idx_arr] = res.values
    return result


def _log_target_diagnostics(df: pd.DataFrame, target_name: str) -> None:
    """Emit compact diagnostics for a computed target column."""
    values = df[target_name].to_numpy(dtype=np.float64)
    finite = values[np.isfinite(values)]
    valid_count = int(finite.size)
    total_count = int(values.size)
    valid_pct = 100.0 * valid_count / max(total_count, 1)
    if valid_count == 0:
        tprint(
            f"{target_name}: valid=0/{total_count} (0.0%) | mean=nan std=nan min=nan max=nan"
        )
        return
    tprint(
        f"{target_name}: valid={valid_count}/{total_count} ({valid_pct:.2f}%) | "
        f"mean={float(finite.mean()):.6f} std={float(finite.std()):.6f} "
        f"min={float(finite.min()):.6f} max={float(finite.max()):.6f}"
    )


def harmonic_mean(a: np.ndarray | float, b: np.ndarray | float) -> np.ndarray | float:
    """
    Compute harmonic mean of two values/arrays.

    Parameters
    ----------
    a : np.ndarray or float
        First value(s)
    b : np.ndarray or float
        Second value(s)

    Returns
    -------
    np.ndarray or float
        Harmonic mean: (2 * a * b) / (a + b + epsilon)
    """
    return (2 * a * b) / (a + b + 1e-9)


def compute_target_revert_weighted(
    close_wide: pd.DataFrame,
    anchor_wide: pd.DataFrame,
    horizon: int,
    min_dev_frac: float = 0.01,        # 1.0%
    strong_dev_frac: float = 0.015,    # 1.5%
    strong_k: float = 80.0,            # saturation speed above 1.5%
    eps: float = 1e-12,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Anchor-reversion target with explicit preference for larger initial dislocations.

    Design:
    - dislocations < 1.0%: not scored
    - 1.0% to 1.5%: lower scores by amplitude weight
    - 1.5%+: majority of score mass lives here
    """
    close = close_wide.to_numpy(dtype=np.float64)
    anchor = anchor_wide.to_numpy(dtype=np.float64)

    T, N = close.shape

    reasons = np.full((T, N), "valid", dtype=object)

    # Initial deviation
    dev0 = close - anchor
    abs_dev0 = np.abs(dev0)
    dev_frac = abs_dev0 / np.maximum(np.abs(close), eps)

    # Eligibility gate
    valid_start = dev_frac >= min_dev_frac

    # Forward horizon alignment
    close_H = np.full((T, N), np.nan, dtype=np.float64)
    anchor_H = np.full((T, N), np.nan, dtype=np.float64)
    close_H[:-horizon] = close[horizon:]
    anchor_H[:-horizon] = anchor[horizon:]

    devH = close_H - anchor_H
    abs_devH = np.abs(devH)

    # Recovery fraction
    recovered = np.maximum(0.0, abs_dev0 - abs_devH)
    recovery_frac = recovered / np.maximum(abs_dev0, eps)
    recovery_frac = np.clip(recovery_frac, 0.0, 1.0)

    # Speed score: first bar where 50% of initial deviation is recovered
    fwd_close = np.full((T, horizon, N), np.nan, dtype=np.float64)
    fwd_anchor = np.full((T, horizon, N), np.nan, dtype=np.float64)
    for k in range(1, horizon + 1):
        fwd_close[:-k, k - 1, :] = close[k:]
        fwd_anchor[:-k, k - 1, :] = anchor[k:]

    fwd_abs_dev = np.abs(fwd_close - fwd_anchor)
    fwd_recovery_frac = (abs_dev0[:, None, :] - fwd_abs_dev) / np.maximum(abs_dev0[:, None, :], eps)
    fwd_recovery_frac = np.clip(fwd_recovery_frac, 0.0, 1.0)

    hit_50 = fwd_recovery_frac >= 0.50
    any_hit_50 = np.any(hit_50, axis=1)

    first_hit_idx = np.argmax(hit_50, axis=1) + 1
    first_hit_idx = first_hit_idx.astype(np.float64)
    first_hit_idx[~any_hit_50] = np.nan

    speed_score = 1.0 / np.sqrt(first_hit_idx)
    speed_score[~any_hit_50] = 0.0

    # Base revert quality
    base = np.sqrt(recovery_frac * speed_score)

    # Amplitude weight
    amp_weight = np.zeros((T, N), dtype=np.float64)

    # 1.0% to 1.5%: weakly weighted
    mid_mask = (dev_frac >= min_dev_frac) & (dev_frac < strong_dev_frac)
    x_mid = (dev_frac[mid_mask] - min_dev_frac) / max(strong_dev_frac - min_dev_frac, eps)
    amp_weight[mid_mask] = 0.4 * (x_mid ** 2)

    # 1.5%+: strong region, saturating toward 1
    hi_mask = dev_frac >= strong_dev_frac
    x_hi = dev_frac[hi_mask] - strong_dev_frac
    amp_weight[hi_mask] = 0.5 + 0.5 * (1.0 - np.exp(-strong_k * x_hi))

    # Final target
    target = base * amp_weight
    target = np.clip(target, 0.0, 1.0)

    # Invalidate rows without enough forward horizon or insufficient initial dislocation
    target[-horizon:] = np.nan

    reasons[~valid_start] = "outside_support_mask"
    reasons[-horizon:] = "horizon_exceeded"

    target[~valid_start] = np.nan

    return (
        pd.DataFrame(
            target.astype(np.float32),
            index=close_wide.index,
            columns=close_wide.columns,
        ),
        pd.DataFrame(
            reasons,
            index=close_wide.index,
            columns=close_wide.columns,
        )
    )


def tail_weighted_geometric_mean(
    a: np.ndarray | float,
    b: np.ndarray | float,
    tail_boost: float = 1.15,
    center: float = 0.5,
) -> np.ndarray | float:
    """
    Compute geometric mean with gentle tail boost for selective regime discovery.

    Amplifies extreme values (near 0 or 1) while compressing middle values.
    This makes SELECTIVE regimes score higher than broad regimes.

    Parameters
    ----------
    a : np.ndarray or float
        First value(s) in [0, 1]
    b : np.ndarray or float
        Second value(s) in [0, 1]
    tail_boost : float
        Amplification factor for tails (default 1.15 = gentle 15% boost)
        - 1.0 = no boost (standard geometric mean)
        - 1.15 = gentle boost (recommended)
        - 1.5 = moderate boost
        - 2.0 = aggressive boost
    center : float
        Center point around which to apply boost (default 0.5)

    Returns
    -------
    np.ndarray or float
        Tail-weighted geometric mean in [0, 1]
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)

    a_clipped = np.clip(a, 1e-9, 1.0)
    b_clipped = np.clip(b, 1e-9, 1.0)
    geometric = np.sqrt(a_clipped * b_clipped)
    distance = np.abs(geometric - center)
    boost_factor = 1.0 + (tail_boost - 1.0) * (distance / center)
    boosted = geometric * boost_factor

    result = np.where(
        geometric >= center,
        np.clip(boosted, center, 1.0),
        np.clip(boosted, 0.0, center),
    )
    return result


def compute_rolling_surprisal(
    target_values: np.ndarray,
    lookback: int = 500,
    min_samples: int = 50,
    eps: float = 1e-9,
    log_base: str = "bits",
    two_sided: bool = True,
    smooth_window: int | None = None,
) -> np.ndarray:
    """
    Compute rolling empirical surprisal of target values.

    Higher surprisal means the current value is rarer under recent history.
    """
    x = np.asarray(target_values, dtype=np.float64)
    n = len(x)
    surprisal = np.full(n, np.nan, dtype=np.float64)

    if lookback <= 0:
        raise ValueError("lookback must be > 0")
    if min_samples <= 0:
        raise ValueError("min_samples must be > 0")
    if eps <= 0:
        raise ValueError("eps must be > 0")
    if log_base not in {"bits", "nats"}:
        raise ValueError("log_base must be either 'bits' or 'nats'")

    log_fn = np.log2 if log_base == "bits" else np.log

    for i in range(n):
        current = x[i]
        if np.isnan(current):
            continue

        start = max(0, i - lookback)
        window = x[start:i]
        valid_window = window[np.isfinite(window)]

        if len(valid_window) < min_samples:
            continue

        sorted_window = np.sort(valid_window)
        p = np.searchsorted(sorted_window, current, side="right") / len(sorted_window)
        p = float(np.clip(p, eps, 1.0 - eps))

        tail_prob = min(p, 1.0 - p)
        if two_sided:
            tail_prob *= 2.0
        tail_prob = float(np.clip(tail_prob, eps, 1.0))
        surprisal[i] = -log_fn(tail_prob)

    if smooth_window is not None and smooth_window > 1:
        surprisal = (
            pd.Series(surprisal)
            .rolling(window=smooth_window, min_periods=1)
            .mean()
            .to_numpy(dtype=np.float64)
        )

    return surprisal


def scale_surprisal(
    surprisal: np.ndarray, reference_bits: float = 3.0, clip: bool = True
) -> np.ndarray:
    """Convert surprisal to a bounded [0, 1] multiplier."""
    if reference_bits <= 0:
        raise ValueError("reference_bits must be > 0")
    out = np.asarray(surprisal, dtype=np.float64) / reference_bits
    if clip:
        out = np.clip(out, 0.0, 1.0)
    return out


def apply_surprisal_to_targets(
    df: pd.DataFrame,
    target_cols: list[str],
    symbol_col: str = "symbol",
    lookback: int = 500,
    min_samples: int = 50,
    eps: float = 1e-9,
    log_base: str = "bits",
    two_sided: bool = True,
    smooth_window: int | None = 5,
    feature_suffix: str = "_surprisal",
    blend_weight: float = 0.2,
    reference_bits: float = 3.0,
) -> pd.DataFrame:
    """
    Add surprisal features and apply a weak multiplicative surprisal modulation.

    The surprisal term is first mapped onto multiplier scale around 1.0:
        surprisal_multiplier = 1.0 + (scaled_surprisal - 0.5)

    Then each target is updated as:
        target *= (1 - blend_weight) * 1.0 + blend_weight * surprisal_multiplier
    """
    if not 0.0 <= blend_weight <= 1.0:
        raise ValueError("blend_weight must be in [0, 1]")

    out = df.copy()

    for col in target_cols:
        if col not in out.columns:
            raise KeyError(f"Missing target column: {col}")

        surprisal = np.full(len(out), np.nan, dtype=np.float64)
        if symbol_col in out.columns:
            positions = np.arange(len(out))
            for sym in out[symbol_col].unique():
                mask = (out[symbol_col] == sym).values
                idx_arr = positions[mask]
                values = out.iloc[idx_arr][col].to_numpy(dtype=np.float64)
                surprisal[idx_arr] = compute_rolling_surprisal(
                    values,
                    lookback=lookback,
                    min_samples=min_samples,
                    eps=eps,
                    log_base=log_base,
                    two_sided=two_sided,
                    smooth_window=smooth_window,
                )
        else:
            surprisal = compute_rolling_surprisal(
                out[col].to_numpy(dtype=np.float64),
                lookback=lookback,
                min_samples=min_samples,
                eps=eps,
                log_base=log_base,
                two_sided=two_sided,
                smooth_window=smooth_window,
            )

        out[f"{col}{feature_suffix}"] = surprisal.astype(np.float32)
        scaled = scale_surprisal(surprisal, reference_bits=reference_bits, clip=True)
        scaled = np.nan_to_num(scaled, nan=0.0)
        base = out[col].to_numpy(dtype=np.float64)
        surprisal_multiplier = 1.0 + (scaled - 0.5)
        weak_multiplier = ((1.0 - blend_weight) * 1.0) + (
            blend_weight * surprisal_multiplier
        )
        out[col] = np.clip(base * weak_multiplier, 0.0, 1.0).astype(np.float32)

    return out


def get_bounded_triad(
    df: pd.DataFrame,
    n: int = 24,
    percentile_lookback: int = 2000,
    lookback_vol_baseline: int = 100,
    min_history_percentile: int = 100,
    use_tail_weighting: bool = True,
    tail_boost: float = 1.15,
    use_surprisal_selectivity: bool = True,
    surprisal_lookback: int = 500,
    surprisal_min_samples: int = 50,
    surprisal_eps: float = 1e-9,
    surprisal_smooth_window: int | None = 5,
    surprisal_reference_bits: float = 3.0,
    surprisal_blend_weight: float = 0.2,
) -> pd.DataFrame:
    """
    Compute bounded triad targets for regime mining.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns: 'close', 'high', 'low', 'atr', 'volume'
    n : int
        Horizon in bars (default 24)
    percentile_lookback : int
        Rolling window for percentile normalization (default 2000)
    lookback_vol_baseline : int
        Rolling window for volume baseline (default 100)
    min_history_percentile : int
        Minimum history required before yielding valid percentile (default 100)
    use_tail_weighting : bool
        If True, use tail-weighted geometric mean for target combination.
        If False, use standard harmonic mean (default True)
    tail_boost : float
        Tail amplification factor (default 1.15 = gentle 15% boost)
        Only used when use_tail_weighting=True
    use_surprisal_selectivity : bool
        If True, add rolling surprisal features and apply a weak blended multiplier
        to the triad targets (default True)
    surprisal_lookback : int
        Rolling lookback used for surprisal computation (default 500)
    surprisal_min_samples : int
        Minimum historical samples required before surprisal is emitted (default 50)
    surprisal_eps : float
        Numerical stability constant for surprisal computation
    surprisal_smooth_window : int | None
        Optional trailing smoothing window applied to surprisal
    surprisal_reference_bits : float
        Scaling constant used to convert surprisal into a [0, 1] multiplier
    surprisal_blend_weight : float
        Weak target blending weight for the surprisal multiplier

    Returns
    -------
    pd.DataFrame
        Original df with added columns: target_eff, target_vame

    CRITICAL CONSTRAINTS:
    - Do NOT use future-on-future persistence like: p_eff = s_eff.rolling(n).mean().shift(-n)
    - Do NOT use full future range as a drawdown proxy for VAME.
    - Do NOT simplex-normalize the three targets.
    - Do NOT add a 4th target.
    - Targets should remain independent continuous regressands in [0, 1].
    """
    if "symbol" not in df.columns:
        raise ValueError("get_bounded_triad requires a 'symbol' column for per-symbol causality")

    grouped = df.groupby("symbol", sort=False)

    # Forward-looking price metrics
    fwd_close = grouped["close"].transform(lambda x: x.shift(-n))

    fwd_high_max = grouped["high"].transform(lambda x: x.rolling(n).max().shift(-n))
    fwd_low_min = grouped["low"].transform(lambda x: x.rolling(n).min().shift(-n))

    # Excursion calculations
    up_exc = fwd_high_max - df["close"]
    down_exc = df["close"] - fwd_low_min
    max_excursion = np.maximum(up_exc, down_exc)

    # Finding the index of the max and min values within the forward window.
    # Note: argmax/argmin return the relative index (0 to n-1)
    # NaN propagation handled implicitly by pandas.
    fwd_high_argmax = grouped["high"].transform(
        lambda x: x.rolling(n).apply(np.argmax, raw=True).shift(-n)
    )
    fwd_low_argmin = grouped["low"].transform(
        lambda x: x.rolling(n).apply(np.argmin, raw=True).shift(-n)
    )

    # The max absolute excursion could be up or down
    is_up_max = up_exc >= down_exc
    max_excursion_idx = np.where(is_up_max, fwd_high_argmax, fwd_low_argmin)

    # Convert the relative index to a time-to-extreme score
    # Earlier extreme (index near 0) -> score near 1
    # Later extreme (index near n-1) -> score near 0
    time_to_extreme_score = 1.0 - (max_excursion_idx / max(n - 1, 1))
    # Replace NaNs that resulted from calculation (like insufficient forward window size)
    time_to_extreme_score = np.nan_to_num(time_to_extreme_score, nan=0.0)

    # Final displacement
    final_disp = (fwd_close - df["close"]).abs()

    # 1) Trend Efficiency - using anchor-reversion implementation
    # Pivot to wide format for vectorized computation
    close_wide = df.pivot(columns="symbol", values="close")

    # Compute anchor as backward-looking moving average
    anchor_wide = close_wide.rolling(window=n, min_periods=1).mean()

    # Compute target_eff using anchor-reversion implementation
    target_eff_wide, target_eff_reasons_wide = compute_target_revert_weighted(
        close_wide=close_wide,
        anchor_wide=anchor_wide,
        horizon=n,
    )

    # Convert back to long format and assign back to DataFrame.
    # Note: reset_index(level="symbol", drop=True) is safer than reindex(df.index)
    # when dealing with RangeIndex and MultiIndex stacking in pandas 2.x.
    if df.index.has_duplicates or isinstance(df.index, pd.RangeIndex):
        time_col_name = target_eff_wide.index.name or "index"

        eff_stacked = target_eff_wide.stack().reset_index()
        eff_stacked.columns = [time_col_name, "symbol", "target_eff"]

        reason_stacked = target_eff_reasons_wide.stack().reset_index()
        reason_stacked.columns = [time_col_name, "symbol", "target_eff_reason"]

        df_reset = df.reset_index(names=time_col_name)
        df_merged = df_reset.merge(eff_stacked, on=[time_col_name, "symbol"], how="left")
        df_merged = df_merged.merge(reason_stacked, on=[time_col_name, "symbol"], how="left")

        df["target_eff"] = df_merged["target_eff"].values
        df["target_eff_reason"] = df_merged["target_eff_reason"].values
    else:
        # MultiIndex creation aligns the stack back to the original df index correctly.
        eff_series = target_eff_wide.stack()
        reason_series = target_eff_reasons_wide.stack()

        # When pivot happens, original unique index values form the row index.
        # stack() creates a MultiIndex (time_index, symbol)
        mi = df.set_index(["symbol"], append=True).index
        df["target_eff"] = eff_series.reindex(mi).values
        df["target_eff_reason"] = reason_series.reindex(mi).values

    _log_target_diagnostics(df, "target_eff")

    # 2) Elasticity
    ela_excursion_normalized = max_excursion / (df["atr"] + 1e-9)
    s_ela = _per_symbol_rolling_percentile_rank(
        ela_excursion_normalized,
        df["symbol"],
        window=percentile_lookback,
        min_periods=min_history_percentile,
    )
    s_ela = s_ela.fillna(0.5)

    p_ela = 1 - (
        final_disp / (max_excursion + 1e-9)
    )

    # 3) Expansion Sustainability
    # IMPORTANT: use dominant excursion direction, not final close sign
    direction = np.where(up_exc >= down_exc, 1, -1)

    worst_move_against = np.where(
        direction > 0,
        df["close"] - fwd_low_min,
        fwd_high_max - df["close"]
    )

    vol_ratio = max_excursion / (df["atr"] + 1e-9)

    # Convert vol_ratio to a [0, 1] scale using a backward-looking empirical percentile rank
    s_vame = _per_symbol_rolling_percentile_rank(
        vol_ratio,
        df["symbol"],
        window=percentile_lookback,
        min_periods=min_history_percentile,
    )
    s_vame = s_vame.fillna(0.5)  # safe default before enough history

    p_vame = 1 - (
        worst_move_against / (max_excursion + 1e-9)
    )

    # Participation Confirmation
    # Baseline volume over the past window
    vol_baseline = grouped["volume"].transform(
        lambda x: x.rolling(
            lookback_vol_baseline, min_periods=min_history_percentile
        ).median()
    )
    # Expected volume over the forward horizon
    expected_vol = vol_baseline * n
    # Actual volume sum over forward horizon
    fwd_vol_sum = grouped["volume"].transform(lambda x: x.rolling(n).sum().shift(-n))

    participation_ratio = fwd_vol_sum / (expected_vol + 1e-9)
    participation_confirmation = _per_symbol_rolling_percentile_rank(
        participation_ratio,
        df["symbol"],
        window=percentile_lookback,
        min_periods=min_history_percentile,
    )
    participation_confirmation = participation_confirmation.fillna(0.5)

    # In VAME, the relevant extreme is the max excursion in the dominant direction,
    # which is already what is_up_max defines.
    # Therefore, time_to_extreme_score correctly captures the time to the VAME extreme.
    if use_tail_weighting:
        vame_intermediate_1 = tail_weighted_geometric_mean(
            s_vame.clip(0, 1),
            p_vame.clip(0, 1),
            tail_boost=tail_boost,
        )
        vame_intermediate_2 = tail_weighted_geometric_mean(
            participation_confirmation.clip(0, 1),
            np.clip(time_to_extreme_score, 0, 1),
            tail_boost=tail_boost,
        )
        vame_combined = tail_weighted_geometric_mean(
            vame_intermediate_1,
            vame_intermediate_2,
            tail_boost=tail_boost,
        )
    else:
        vame_combined = harmonic_mean(
            harmonic_mean(s_vame.clip(0, 1), p_vame.clip(0, 1)),
            harmonic_mean(
                participation_confirmation.clip(0, 1),
                np.clip(time_to_extreme_score, 0, 1),
            ),
        )

    vame_reasons = np.full(len(df), "valid", dtype=object)

    # We apply horizon check for VAME
    # Check if fwd_close was NaN but current close was not, roughly indicating horizon exceeded
    fwd_close_nan = fwd_close.isna() & ~df["close"].isna()
    vame_reasons[fwd_close_nan] = "horizon_exceeded"

    df["target_vame"] = np.nan_to_num(vame_combined, nan=0.0)

    # Apply horizon NaNs
    df.loc[fwd_close_nan, "target_vame"] = np.nan
    df["target_vame_reason"] = vame_reasons

    _log_target_diagnostics(df, "target_vame")

    if use_surprisal_selectivity:
        df = apply_surprisal_to_targets(
            df=df,
            target_cols=TRIAD_TARGET_NAMES,
            symbol_col="symbol",
            lookback=surprisal_lookback,
            min_samples=surprisal_min_samples,
            eps=surprisal_eps,
            log_base="bits",
            two_sided=True,
            smooth_window=surprisal_smooth_window,
            reference_bits=surprisal_reference_bits,
            blend_weight=surprisal_blend_weight,
        )

    return df


def compute_triad_targets_for_horizons(
    df: pd.DataFrame,
    horizons: List[int],
    atr_col: str = "atr",
    percentile_lookback: int = 2000,
    lookback_vol_baseline: int = 100,
    min_history_percentile: int = 100,
    use_tail_weighting: bool = True,
    tail_boost: float = 1.15,
    use_surprisal_selectivity: bool = True,
    surprisal_lookback: int = 500,
    surprisal_min_samples: int = 50,
    surprisal_eps: float = 1e-9,
    surprisal_smooth_window: int | None = 5,
    surprisal_reference_bits: float = 3.0,
    surprisal_blend_weight: float = 0.2,
) -> Dict[int, pd.DataFrame]:
    """
    Compute triad targets for multiple horizons.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns: 'close', 'high', 'low', 'volume', and the specified ATR column
    horizons : List[int]
        List of horizon values (in bars) to compute targets for
    atr_col : str
        Name of the ATR column (default 'atr')
    use_tail_weighting : bool
        If True, use tail-weighted geometric mean (default True)
    tail_boost : float
        Tail amplification factor (default 1.15)
    use_surprisal_selectivity : bool
        If True, add surprisal companion features and apply weak blended weighting
    surprisal_lookback : int
        Rolling lookback used for surprisal computation
    surprisal_min_samples : int
        Minimum historical samples required before surprisal is emitted
    surprisal_eps : float
        Numerical stability constant for surprisal computation
    surprisal_smooth_window : int | None
        Optional trailing smoothing window applied to surprisal
    surprisal_reference_bits : float
        Scaling constant used to convert surprisal into a [0, 1] multiplier
    surprisal_blend_weight : float
        Weak target blending weight for the surprisal multiplier

    Returns
    -------
    Dict[int, pd.DataFrame]
        Dictionary mapping horizon -> DataFrame with target columns.
        Each DataFrame contains the original data plus target columns
        suffixed with the horizon (e.g., 'target_eff_24', 'target_vame_24')
    """
    results: Dict[int, pd.DataFrame] = {}

    for horizon in horizons:
        # Create a copy to avoid modifying the original
        df_copy = df.copy()

        # Ensure ATR column is correctly named for get_bounded_triad
        if atr_col != "atr" and atr_col in df_copy.columns:
            df_copy["atr"] = df_copy[atr_col]
        elif "atr" not in df_copy.columns:
            raise ValueError(f"ATR column '{atr_col}' not found in DataFrame")

        # Compute targets for this horizon
        df_with_targets = get_bounded_triad(
            df_copy,
            n=horizon,
            percentile_lookback=percentile_lookback,
            lookback_vol_baseline=lookback_vol_baseline,
            min_history_percentile=min_history_percentile,
            use_tail_weighting=use_tail_weighting,
            tail_boost=tail_boost,
            use_surprisal_selectivity=use_surprisal_selectivity,
            surprisal_lookback=surprisal_lookback,
            surprisal_min_samples=surprisal_min_samples,
            surprisal_eps=surprisal_eps,
            surprisal_smooth_window=surprisal_smooth_window,
            surprisal_reference_bits=surprisal_reference_bits,
            surprisal_blend_weight=surprisal_blend_weight,
        )

        # Rename target columns with horizon suffix
        rename_map = {
            "target_eff": f"target_eff_{horizon}",
            "target_vame": f"target_vame_{horizon}",
            "target_eff_surprisal": f"target_eff_surprisal_{horizon}",
            "target_vame_surprisal": f"target_vame_surprisal_{horizon}",
            "target_eff_reason": f"target_eff_reason_{horizon}",
            "target_vame_reason": f"target_vame_reason_{horizon}",
        }
        rename_map = {k: v for k, v in rename_map.items() if k in df_with_targets.columns}
        df_with_targets = df_with_targets.rename(columns=rename_map)

        # Store only the target columns plus original data
        target_cols = list(rename_map.values())
        results[horizon] = df_with_targets[target_cols]

    return results


def compute_target_diagnostics(
    df: pd.DataFrame,
    target_names: List[str] = TRIAD_TARGET_NAMES
) -> Dict:
    """
    Compute target quality diagnostics.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the target columns
    target_names : List[str]
        List of target column names to analyze

    Returns
    -------
    Dict
        Dictionary containing:
        - per_target_stats: {target: {mean, std, p05, p25, p50, p75, p95, nonzero_frac}}
        - cross_target_corr: correlation matrix between targets
        - quality_flags: list of any quality issues detected
    """
    per_target_stats: Dict[str, Dict] = {}
    quality_flags: List[str] = []

    for target in target_names:
        if target not in df.columns:
            quality_flags.append(f"Missing target column: {target}")
            continue

        # Drop NaN values for statistics
        target_data = df[target].dropna()

        if len(target_data) == 0:
            quality_flags.append(f"No valid data for target: {target}")
            per_target_stats[target] = {
                "mean": np.nan,
                "std": np.nan,
                "p05": np.nan,
                "p25": np.nan,
                "p50": np.nan,
                "p75": np.nan,
                "p95": np.nan,
                "nonzero_frac": np.nan
            }
            continue

        # Compute statistics
        stats = {
            "mean": float(target_data.mean()),
            "std": float(target_data.std()),
            "p05": float(target_data.quantile(0.05)),
            "p25": float(target_data.quantile(0.25)),
            "p50": float(target_data.quantile(0.50)),
            "p75": float(target_data.quantile(0.75)),
            "p95": float(target_data.quantile(0.95)),
            "nonzero_frac": float((target_data > 0).mean())
        }
        per_target_stats[target] = stats

        # Check for degenerate distributions
        if stats["std"] < 1e-6:
            quality_flags.append(f"Near-zero variance for target: {target}")

        # Check for extreme concentration at boundaries
        near_zero_frac = (target_data < 0.01).mean()
        near_one_frac = (target_data > 0.99).mean()
        if near_zero_frac > 0.9:
            quality_flags.append(f"Target {target} concentrated near 0 ({near_zero_frac:.1%})")
        if near_one_frac > 0.9:
            quality_flags.append(f"Target {target} concentrated near 1 ({near_one_frac:.1%})")

    # Compute cross-target correlation
    available_targets = [t for t in target_names if t in df.columns]
    cross_target_corr: np.ndarray | None = None

    if len(available_targets) > 1:
        target_df = df[available_targets].dropna()
        if len(target_df) > 1:
            cross_target_corr = target_df.corr().values

            # Check for excessive correlation
            if cross_target_corr is not None:
                n_targets = len(available_targets)
                for i in range(n_targets):
                    for j in range(i + 1, n_targets):
                        corr_val = abs(cross_target_corr[i, j])
                        if corr_val > 0.85:
                            quality_flags.append(
                                f"High correlation ({corr_val:.3f}) between "
                                f"{available_targets[i]} and {available_targets[j]}"
                            )

    return {
        "per_target_stats": per_target_stats,
        "cross_target_corr": cross_target_corr,
        "quality_flags": quality_flags
    }


def validate_target_quality(
    diagnostics: Dict,
    variance_threshold: float = 1e-6,
    correlation_threshold: float = 0.85
) -> Tuple[bool, List[str]]:
    """
    Validate target quality based on diagnostics.

    Parameters
    ----------
    diagnostics : Dict
        Output from compute_target_diagnostics()
    variance_threshold : float
        Minimum acceptable variance (default 1e-6)
    correlation_threshold : float
        Maximum acceptable cross-target correlation (default 0.85)

    Returns
    -------
    Tuple[bool, List[str]]
        (is_valid, list_of_issues)
        
    Flags:
    - Variance near zero
    - Excessive cross-target correlation
    - Degenerate distributions
    """
    issues: List[str] = []

    # Check per-target statistics
    per_target_stats = diagnostics.get("per_target_stats", {})
    for target, stats in per_target_stats.items():
        std = stats.get("std", np.nan)
        if np.isnan(std):
            issues.append(f"Cannot compute variance for target: {target}")
        elif std < variance_threshold:
            issues.append(
                f"Variance ({std:.2e}) below threshold ({variance_threshold:.2e}) "
                f"for target: {target}"
            )

        # Check for degenerate distributions (concentrated at boundaries)
        mean = stats.get("mean", np.nan)
        p05 = stats.get("p05", np.nan)
        p95 = stats.get("p95", np.nan)

        if not np.isnan(mean):
            if mean < 0.01:
                issues.append(f"Target {target} mean ({mean:.4f}) near zero boundary")
            elif mean > 0.99:
                issues.append(f"Target {target} mean ({mean:.4f}) near one boundary")

        if not np.isnan(p05) and not np.isnan(p95):
            range_val = p95 - p05
            if range_val < 0.1:
                issues.append(
                    f"Target {target} has narrow interquartile range ({range_val:.4f})"
                )

    # Check cross-target correlation
    cross_target_corr = diagnostics.get("cross_target_corr")
    target_names = list(per_target_stats.keys())

    if cross_target_corr is not None and len(target_names) > 1:
        n_targets = len(target_names)
        for i in range(n_targets):
            for j in range(i + 1, n_targets):
                if i < cross_target_corr.shape[0] and j < cross_target_corr.shape[1]:
                    corr_val = abs(cross_target_corr[i, j])
                    if corr_val > correlation_threshold:
                        issues.append(
                            f"Excessive correlation ({corr_val:.3f}) between "
                            f"{target_names[i]} and {target_names[j]} "
                            f"(threshold: {correlation_threshold})"
                        )

    # Include any pre-computed quality flags
    quality_flags = diagnostics.get("quality_flags", [])
    issues.extend(quality_flags)

    is_valid = len(issues) == 0
    return is_valid, issues
