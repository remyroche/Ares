"""Bagged probability aggregation and evaluation utilities.

This module centralizes combination logic for ensembles of per-bag
probabilities (e.g. Diversity-Defense specialists or simple bagging).
It is intentionally stateless so it can be reused from both the
meta-labeling backtests and the Analyst base-layer diagnostics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class BagAggregationConfig:
    """Configuration of which bagging combinations to compute.

    Methods are identified by simple string keys so that both
    configuration files and code can reference them without importing
    an Enum. The defaults cover the main recipes we care about; callers
    can always request a subset.

    We include 10+ distinct aggregation schemes to allow rich
    diagnostics over the same underlying bag predictions.
    """

    methods: Sequence[str] = (
        "mean",                 # simple average
        "lower",                # mean - std
        "mean_minus_std",       # alias for lower
        "mean_minus_std_1_5",   # mean - 1.5 * std
        "mean_plus_std",        # mean + std (upper confidence)
        "median",               # robust central tendency
        "trimmed_mean_10",      # drop top/bottom 10% bags
        "q10",                  # 10th percentile
        "q25",                  # 25th percentile
        "q75",                  # 75th percentile
        "q90",                  # 90th percentile
        "max",                  # max bag prob
        "min",                  # min bag prob
        "mad_consensus",        # MAD-weighted consensus in [0, 1]
    )


def _compute_basic_stats(preds_mat: np.ndarray) -> Dict[str, np.ndarray]:
    """Compute basic per-sample statistics from bag matrix.

    Args:
        preds_mat: Array of shape (n_samples, n_bags) with probabilities.
    """
    mean = preds_mat.mean(axis=1)
    std = preds_mat.std(axis=1)
    median = np.median(preds_mat, axis=1)

    # Signal space in [-1, 1]
    preds_signal = 2.0 * preds_mat - 1.0
    median_signal = np.median(preds_signal, axis=1)
    mad = np.median(np.abs(preds_signal - median_signal[:, np.newaxis]), axis=1)

    q10 = np.quantile(preds_mat, 0.10, axis=1)
    q25 = np.quantile(preds_mat, 0.25, axis=1)
    q75 = np.quantile(preds_mat, 0.75, axis=1)
    q90 = np.quantile(preds_mat, 0.90, axis=1)

    return {
        "mean": mean,
        "std": std,
        "median": median,
        "mad": mad,
        "q10": q10,
        "q25": q25,
        "q75": q75,
        "q90": q90,
        "median_signal": median_signal,
    }


def combine_bags(
    raw_preds: np.ndarray,
    methods: Optional[Iterable[str]] = None,
    mad_floor: float = 0.25,
) -> Dict[str, np.ndarray]:
    """Combine per-bag probabilities into multiple aggregate signals.

    Args:
        raw_preds: Array of shape (n_samples, n_bags) with probabilities.
        methods: Iterable of method names to compute. If None, uses
            BagAggregationConfig().methods.
        mad_floor: Floor for MAD in the mad_consensus recipe.

    Returns:
        Dict mapping method name -> 1D array of shape (n_samples,) in [0, 1].
    """
    if raw_preds.ndim != 2:
        raise ValueError(f"raw_preds must be 2D, got shape {raw_preds.shape}")

    n_samples, n_bags = raw_preds.shape
    if n_samples == 0 or n_bags == 0:
        return {}

    methods_seq = list(methods) if methods is not None else list(BagAggregationConfig().methods)
    if not methods_seq:
        return {}

    # Defensive clipping / NaN handling
    preds_mat = np.asarray(raw_preds, dtype=float)
    preds_mat = np.nan_to_num(preds_mat, nan=0.5, posinf=1.0, neginf=0.0)
    preds_mat = np.clip(preds_mat, 0.0, 1.0)

    stats = _compute_basic_stats(preds_mat)
    mean = stats["mean"]
    std = stats["std"]
    median = stats["median"]
    mad = stats["mad"]
    q10 = stats["q10"]
    q25 = stats["q25"]
    q75 = stats["q75"]
    q90 = stats["q90"]
    median_signal = stats["median_signal"]

    out: Dict[str, np.ndarray] = {}

    for name in methods_seq:
        key = name.lower()
        if key == "mean":
            out[key] = mean
        elif key in ("lower", "mean_minus_std"):
            out[key] = np.clip(mean - std, 0.0, 1.0)
        elif key == "mean_minus_std_1_5":
            out[key] = np.clip(mean - 1.5 * std, 0.0, 1.0)
        elif key == "mean_plus_std":
            out[key] = np.clip(mean + std, 0.0, 1.0)
        elif key == "median":
            out[key] = median
        elif key == "trimmed_mean_10":
            k = max(1, int(0.1 * n_bags))
            if k == 0:
                out[key] = mean
            else:
                sorted_preds = np.sort(preds_mat, axis=1)
                trimmed = sorted_preds[:, k:-k] if 2 * k < n_bags else sorted_preds
                out[key] = trimmed.mean(axis=1)
        elif key == "q10":
            out[key] = q10
        elif key == "q25":
            out[key] = q25
        elif key == "q75":
            out[key] = q75
        elif key == "q90":
            out[key] = q90
        elif key == "max":
            out[key] = preds_mat.max(axis=1)
        elif key == "min":
            out[key] = preds_mat.min(axis=1)
        elif key == "mad_consensus":
            # Consensus signal: shrink signal strength by MAD
            mad_eff = np.maximum(mad, mad_floor)
            signal_strength = np.abs(median_signal)
            raw_score = signal_strength - 1.1 * mad_eff
            score = np.clip(raw_score, 0.0, 1.0)
            out[key] = score
        else:
            # Unknown method: ignore silently (callers can check keys)
            continue

    return out


def evaluate_prob_variants(
    returns: pd.Series,
    prob_variants: Mapping[str, pd.Series],
    threshold: float = 0.6,
) -> pd.DataFrame:
    """Evaluate multiple probability variants at a fixed threshold.

    Args:
        returns: Series of realized returns (one per event).
        prob_variants: Mapping name -> probability Series.
        threshold: Probability gate threshold.

    Returns:
        DataFrame with one row per variant and columns:
        ['variant', 'threshold', 'n_events', 'n_trades', 'trades_per_day',
         'mean_return', 'sharpe_trade', 'max_drawdown', 'hit_rate'].
    """
    if not isinstance(returns, pd.Series):
        raise ValueError("returns must be a pandas Series")

    results: List[Dict[str, float]] = []

    # Base alignment index
    base_index = returns.index

    for name, probs in prob_variants.items():
        if probs is None:
            continue
        try:
            s = probs.astype(float)
        except Exception:
            s = pd.to_numeric(probs, errors="coerce")

        # Align by index; if lengths mismatch badly, fall back to tail-align
        s = s.reindex(base_index)
        if s.isna().all():
            # Try tail alignment as a fallback
            if len(s) != len(returns):
                arr = probs.to_numpy(dtype=float)
                if len(arr) > len(returns):
                    arr = arr[-len(returns):]
                elif len(arr) < len(returns):
                    pad = np.full(len(returns) - len(arr), 0.5, dtype=float)
                    arr = np.concatenate([pad, arr])
                s = pd.Series(arr, index=base_index)

        # Clean up NaNs
        s = s.replace([np.inf, -np.inf], np.nan).fillna(0.5)

        # Build gate
        gate_mask = s >= threshold

        # Apply same mask to returns
        ret = returns.copy().astype(float)
        ret = ret.replace([np.inf, -np.inf], np.nan).dropna()

        # Ensure gate_mask index matches ret
        gate_mask = gate_mask.reindex(ret.index).fillna(False)

        n_events = int(ret.size)
        gated = ret[gate_mask]
        n_trades = int(gated.size)

        if n_events == 0:
            continue

        if n_trades > 0:
            mean_ret = float(gated.mean())
            std_ret = float(gated.std(ddof=1)) if n_trades > 1 else 0.0
            sharpe = float(mean_ret / std_ret) * float(np.sqrt(n_trades)) if std_ret > 0 else 0.0
            hit_rate = float((gated > 0).mean())

            equity = (1.0 + gated).cumprod()
            running_max = equity.cummax()
            drawdown = equity / running_max - 1.0
            max_dd = float(drawdown.min()) if len(drawdown) > 0 else 0.0

            trades_per_day = None
            if isinstance(gated.index, pd.DatetimeIndex) and n_trades > 0:
                idx = gated.index.sort_values()
                start = idx[0]
                end = idx[-1]
                num_days = max(1, int((end.date() - start.date()).days) + 1)
                trades_per_day = float(n_trades) / float(num_days)
        else:
            mean_ret = 0.0
            std_ret = 0.0
            sharpe = 0.0
            hit_rate = 0.0
            max_dd = 0.0
            trades_per_day = 0.0

        row = {
            "variant": str(name),
            "threshold": float(threshold),
            "n_events": n_events,
            "n_trades": n_trades,
            "trades_per_day": float(trades_per_day) if trades_per_day is not None else float(0.0),
            "mean_return": float(mean_ret),
            "sharpe_trade": float(sharpe),
            "max_drawdown": float(max_dd),
            "hit_rate": float(hit_rate),
        }
        results.append(row)

    if not results:
        return pd.DataFrame(columns=[
            "variant",
            "threshold",
            "n_events",
            "n_trades",
            "trades_per_day",
            "mean_return",
            "sharpe_trade",
            "max_drawdown",
            "hit_rate",
        ])

    df_res = pd.DataFrame(results)
    df_res = df_res.sort_values(["mean_return", "sharpe_trade"], ascending=[False, False]).reset_index(drop=True)
    return df_res
