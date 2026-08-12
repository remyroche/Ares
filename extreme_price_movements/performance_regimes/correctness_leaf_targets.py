"""Causal targets for base-error leaf-regime discovery."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd


EPS = 1e-8
TARGET_FAMILIES = ("correctness", "positive", "negative", "entropy", "surprise")


@dataclass(frozen=True)
class CorrectnessScale:
    lower_bps: float
    upper_bps: float
    fit_rows: int


def fit_correctness_scale(residual_bps: Sequence[float], *, lower_q: float = .05, upper_q: float = .95) -> CorrectnessScale:
    """Fit side-local tail clips on already-resolved training residuals only."""
    value = np.asarray(residual_bps, dtype=float)
    value = value[np.isfinite(value)]
    if len(value) < 100:
        raise ValueError("need at least 100 resolved rows for a correctness scale")
    lower, upper = np.quantile(value, [lower_q, upper_q])
    if lower >= -EPS or upper <= EPS:
        raise ValueError("correctness target requires both over- and under-confidence support")
    return CorrectnessScale(float(lower), float(upper), int(len(value)))


def soft_correctness(residual_bps: Sequence[float], scale: CorrectnessScale) -> np.ndarray:
    """0=overconfident, .5=accurate, 1=underconfident; tails are clipped."""
    value = np.asarray(residual_bps, dtype=float)
    lower = .5 + .5 * np.clip(value / abs(scale.lower_bps), -1., 0.)
    upper = .5 + .5 * np.clip(value / scale.upper_bps, 0., 1.)
    return np.where(value <= 0., lower, upper).astype(np.float32)


def soft_positive_surprise(residual_bps: Sequence[float], *, onset_bps: float = 50., full_bps: float = 75.) -> np.ndarray:
    """Soft underestimation membership: zero through ``onset_bps``, one at ``full_bps``."""
    if full_bps <= onset_bps:
        raise ValueError("full_bps must exceed onset_bps")
    value = np.asarray(residual_bps, dtype=float)
    return np.clip((value - onset_bps) / (full_bps - onset_bps), 0., 1.).astype(np.float32)


def soft_negative_surprise(residual_bps: Sequence[float], *, onset_bps: float = -50., full_bps: float = -75.) -> np.ndarray:
    """Soft overestimation membership: zero through ``onset_bps``, one at ``full_bps``."""
    if full_bps >= onset_bps:
        raise ValueError("full_bps must be below onset_bps")
    value = np.asarray(residual_bps, dtype=float)
    return np.clip((onset_bps - value) / (onset_bps - full_bps), 0., 1.).astype(np.float32)


def binary_surprise(residual_bps: Sequence[float], *, threshold_bps: float = 50.) -> np.ndarray:
    """Whether the realised conversion error exceeds the declared economic band."""
    return (np.abs(np.asarray(residual_bps, dtype=float)) > threshold_bps).astype(np.float32)


def probability_entropy(probabilities: np.ndarray) -> np.ndarray:
    """Normalized Shannon entropy of a probability simplex, in [0, 1]."""
    p = np.asarray(probabilities, dtype=float)
    if p.ndim != 2 or p.shape[1] < 2:
        raise ValueError("probabilities must be a two-dimensional simplex")
    p = np.clip(p, EPS, np.inf)
    p = p / p.sum(axis=1, keepdims=True)
    return (-np.sum(p * np.log(p), axis=1) / np.log(p.shape[1])).astype(np.float32)


def select_top_base_per_timestamp(frame: pd.DataFrame, *, score_column: str, timestamp_column: str = "__ts__", candidate_column: str = "candidate_id", fraction: float = .05) -> pd.Series:
    """Stable global top-k base cohort per decision bar, never per side."""
    if not 0. < fraction <= 1.:
        raise ValueError("fraction must be in (0, 1]")
    result = pd.Series(False, index=frame.index)
    for _timestamp, part in frame.groupby(timestamp_column, observed=True, sort=False):
        count = max(1, int(np.ceil(len(part) * fraction)))
        take = part.sort_values([score_column, candidate_column], ascending=[False, True], kind="stable").head(count)
        result.loc[take.index] = True
    return result


def aggregate_correctness_periods(frame: pd.DataFrame, *, target_column: str, horizon_hours: int, timestamp_column: str = "__ts__", label_available_column: str = "label_available_ts") -> pd.DataFrame:
    """Assign a non-overlapping period label and its true availability time.

    The target is an equal-timestamp average: a busy timestamp cannot dominate
    a 72-hour correctness state merely because it has more candidates.
    """
    if horizon_hours <= 0:
        raise ValueError("horizon_hours must be positive")
    out = frame.copy()
    out[timestamp_column] = pd.to_datetime(out[timestamp_column], utc=True, errors="raise")
    out[label_available_column] = pd.to_datetime(out[label_available_column], utc=True, errors="raise")
    block = out[timestamp_column].dt.floor(f"{int(horizon_hours)}h")
    timestamp_mean = out.groupby([block, timestamp_column], observed=True)[target_column].mean().rename("__timestamp_target__")
    period = timestamp_mean.groupby(level=0, observed=True).mean().rename("period_correctness_target")
    available = out.groupby(block, observed=True)[label_available_column].max().rename("period_label_available_ts")
    out["correctness_period_start"] = block
    out = out.join(period, on="correctness_period_start").join(available, on="correctness_period_start")
    return out


__all__ = ["TARGET_FAMILIES", "CorrectnessScale", "aggregate_correctness_periods", "binary_surprise", "fit_correctness_scale", "probability_entropy", "select_top_base_per_timestamp", "soft_correctness", "soft_negative_surprise", "soft_positive_surprise"]
