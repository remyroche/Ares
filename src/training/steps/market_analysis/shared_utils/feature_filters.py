"""Reusable feature filtering and quality assessment utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class FilterResult:
    """Container for filter outputs and per-column metadata."""

    frame: pd.DataFrame
    column_metadata: Dict[str, Dict[str, float]]
    dropped_columns: List[str]


def winsorize_frame(
    frame: pd.DataFrame,
    lower_quantile: float,
    upper_quantile: float,
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, float]]]:
    """Winsorize each column of ``frame`` and return the capped frame and metadata."""

    capped = frame.copy()
    metadata: Dict[str, Dict[str, float]] = {}

    for column in frame.columns:
        series = frame[column].dropna()
        if series.empty:
            metadata[column] = {"lower": np.nan, "upper": np.nan}
            continue
        lower = float(series.quantile(lower_quantile))
        upper = float(series.quantile(upper_quantile))
        capped[column] = frame[column].clip(lower=lower, upper=upper)
        metadata[column] = {"lower": lower, "upper": upper}

    return capped, metadata


def filter_low_variance(frame: pd.DataFrame, min_variance: float) -> FilterResult:
    """Remove columns whose variance falls below ``min_variance``."""

    variances = frame.var(axis=0, skipna=True)
    keep_mask = variances >= min_variance
    kept_columns = variances.index[keep_mask].tolist()
    dropped_columns = variances.index[~keep_mask].tolist()
    metadata = {col: {"variance": float(variances[col])} for col in variances.index}

    filtered = frame[kept_columns]
    return FilterResult(filtered, metadata, dropped_columns)


def prune_correlated_features(frame: pd.DataFrame, threshold: float) -> FilterResult:
    """Drop columns that exceed the pairwise correlation ``threshold``."""

    if frame.empty or frame.shape[1] <= 1:
        return FilterResult(frame, {}, [])

    corr = frame.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    to_drop: List[str] = []

    for column in upper.columns:
        if any(upper[column] > threshold):
            to_drop.append(column)

    filtered = frame.drop(columns=to_drop) if to_drop else frame
    metadata = {col: {"max_correlation": float(upper[col].max()) if not upper[col].isna().all() else 0.0} for col in upper.columns}
    return FilterResult(filtered, metadata, to_drop)


def calculate_persistence(series: pd.Series) -> float:
    """Calculate a simple persistence score based on sign changes."""

    clean = series.dropna()
    if len(clean) < 2:
        return 0.0
    signs = np.sign(clean)
    sign_changes = np.count_nonzero(np.diff(signs) != 0)
    return 1.0 - (sign_changes / max(len(signs) - 1, 1))


def calculate_noise_ratio(series: pd.Series) -> float:
    """Estimate a noise ratio using the standard deviation relative to mean magnitude."""

    clean = series.dropna()
    if clean.empty:
        return float("inf")
    mean_abs = np.mean(np.abs(clean)) + 1e-12
    return float(np.std(clean) / mean_abs)


def calculate_temporal_stability(series: pd.Series) -> float:
    """Measure temporal stability via first-difference volatility."""

    clean = series.dropna()
    if len(clean) < 2:
        return 0.0
    diffs = np.diff(clean)
    denom = np.std(clean) + 1e-12
    return float(1.0 - (np.std(diffs) / denom))


def apply_quality_thresholds(
    frame: pd.DataFrame,
    min_persistence: float,
    max_noise_ratio: float,
    min_stability: float,
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, float]], Dict[str, List[str]]]:
    """Filter columns based on persistence, noise ratio, and temporal stability."""

    metrics: Dict[str, Dict[str, float]] = {}
    dropped: Dict[str, List[str]] = {}
    keep_columns: List[str] = []

    for column in frame.columns:
        series = frame[column]
        persistence = calculate_persistence(series)
        noise_ratio = calculate_noise_ratio(series)
        stability = calculate_temporal_stability(series)
        metrics[column] = {
            "persistence": persistence,
            "noise_ratio": noise_ratio,
            "stability": stability,
        }

        reasons: List[str] = []
        if persistence < min_persistence:
            reasons.append("persistence")
        if noise_ratio > max_noise_ratio:
            reasons.append("noise_ratio")
        if stability < min_stability:
            reasons.append("stability")

        if reasons:
            dropped[column] = reasons
        else:
            keep_columns.append(column)

    filtered = frame[keep_columns]
    return filtered, metrics, dropped


__all__ = [
    "FilterResult",
    "winsorize_frame",
    "filter_low_variance",
    "prune_correlated_features",
    "apply_quality_thresholds",
    "calculate_persistence",
    "calculate_noise_ratio",
    "calculate_temporal_stability",
]
