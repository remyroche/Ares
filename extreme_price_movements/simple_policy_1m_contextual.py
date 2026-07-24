"""Leakage-safe helpers for contextual one-minute policy research.

The functions here deliberately contain no outcome fitting.  They implement
unit-preserving ATR powers, robust train-fitted transforms, posterior mixtures,
and exposure-neutral Bayesian size multipliers used by the research runner.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class RobustState:
    median: float
    scale: float


def fit_robust_state(values: np.ndarray) -> RobustState:
    """Fit a median/IQR state on training rows only."""
    x = np.asarray(values, dtype=np.float64)
    x = x[np.isfinite(x)]
    if not len(x):
        return RobustState(0.0, 1.0)
    q25, median, q75 = np.quantile(x, (0.25, 0.50, 0.75))
    scale = max(float(q75 - q25) / 1.349, 1e-9)
    return RobustState(float(median), scale)


def apply_robust_state(values: np.ndarray, state: RobustState, clip: float = 3.0) -> np.ndarray:
    x = np.asarray(values, dtype=np.float64)
    out = (x - state.median) / state.scale
    out[~np.isfinite(out)] = 0.0
    return np.clip(out, -abs(float(clip)), abs(float(clip)))


def normalized_atr_power(atr_fraction: np.ndarray, reference: np.ndarray | float, power: float) -> np.ndarray:
    """Return ``reference * (ATR/reference)**power``; power=1 is identity."""
    atr = np.asarray(atr_fraction, dtype=np.float64)
    ref = np.asarray(reference, dtype=np.float64)
    safe_ref = np.maximum(ref, 1e-9)
    ratio = np.maximum(atr, 1e-9) / safe_ref
    return safe_ref * np.power(ratio, float(power))


def posterior_mixture_scale(posteriors: np.ndarray, component_scales: np.ndarray) -> np.ndarray:
    """Posterior expectation of positive per-component geometry scales."""
    p = np.asarray(posteriors, dtype=np.float64)
    scales = np.asarray(component_scales, dtype=np.float64)
    if p.ndim != 2 or scales.ndim != 1 or p.shape[1] != len(scales):
        raise ValueError("posterior/component shape mismatch")
    if not np.isfinite(p).all() or (p < -1e-8).any():
        raise ValueError("posteriors must be finite and non-negative")
    mass = p.sum(axis=1)
    if (mass <= 0.0).any():
        raise ValueError("posterior rows must have positive mass")
    normalized = p / mass[:, None]
    return normalized @ scales


def geometry_scaled_params(params: Mapping[str, Any], scale: float) -> dict[str, Any]:
    """Scale one coherent tight/loose geometry axis without changing family."""
    out = dict(params)
    s = float(np.clip(scale, 0.70, 1.35))
    bounds = {
        "sl_mult": (1.25, 5.0),
        "trailing_activation_mult": (0.35, 4.0),
        "giveback_beta": (0.10, 1.25),
        "entry_capital_ratio": (0.35, 0.97),
        "transition_center": (0.15, 8.0),
    }
    for key, (low, high) in bounds.items():
        if key in out:
            out[key] = float(np.clip(float(out[key]) * s, low, high))
    # Preserve the current-price clamp ordering if it is enabled.
    if float(out.get("current_distance_sl_ratio", 0.0)) > 0.0:
        out["current_distance_sl_ratio"] = min(
            float(out["current_distance_sl_ratio"]), float(out["entry_capital_ratio"])
        )
    return out


def quantize_scales(scales: np.ndarray, step: float = 0.025, low: float = 0.70, high: float = 1.35) -> np.ndarray:
    values = np.clip(np.asarray(scales, dtype=np.float64), low, high)
    return np.round(values / float(step)) * float(step)


def beta_binomial_lower_score(
    successes: np.ndarray,
    totals: np.ndarray,
    *,
    prior_success: float = 8.0,
    prior_failure: float = 8.0,
    uncertainty_aversion: float = 1.0,
) -> np.ndarray:
    """Normal-approximate conservative score from a Beta posterior."""
    a = np.asarray(successes, dtype=np.float64) + float(prior_success)
    b = np.asarray(totals, dtype=np.float64) - np.asarray(successes, dtype=np.float64) + float(prior_failure)
    mean = a / np.maximum(a + b, 1e-12)
    variance = a * b / np.maximum((a + b) ** 2 * (a + b + 1.0), 1e-12)
    return mean - float(uncertainty_aversion) * np.sqrt(np.maximum(variance, 0.0))


def exposure_neutral_size_multiplier(
    signal: np.ndarray,
    *,
    reference_mean: float,
    strength: float,
    lower: float = 0.50,
    upper: float = 1.35,
) -> np.ndarray:
    """Map a frozen signal to bounded multipliers with train-fitted exposure."""
    raw = np.exp(float(strength) * (np.asarray(signal, dtype=np.float64) - float(reference_mean)))
    raw = np.clip(raw, lower, upper)
    normalizer = max(float(np.mean(raw[np.isfinite(raw)])), 1e-9)
    return np.clip(raw / normalizer, lower, upper)


def support_shrink(raw_scale: float, support: int, prior_strength: float) -> float:
    weight = max(int(support), 0) / max(max(int(support), 0) + float(prior_strength), 1e-12)
    return float(np.exp(weight * np.log(max(float(raw_scale), 1e-9))))


def stable_fold_objective(values: Sequence[float]) -> float:
    x = np.asarray(values, dtype=np.float64)
    return float(x.mean() - 0.5 * x.std() + 0.25 * x.min())
