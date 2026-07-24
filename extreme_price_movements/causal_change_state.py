"""Causal multiscale change-point representation for market-state models.

This module deliberately produces a continuous representation rather than an
event alarm.  Every output compares recent observations with an immediately
preceding window contained in the supplied causal sequence.  It therefore
describes *what* changed, by how much, and with what temporal agreement while
leaving economic interpretation to a downstream side/archetype model.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd


EPS = np.float32(1e-4)


@dataclass(frozen=True)
class ChangeScale:
    name: str
    previous_bars: int
    recent_bars: int


DEFAULT_CHANGE_SCALES: tuple[ChangeScale, ...] = (
    ChangeScale("fast", 4, 2),
    ChangeScale("medium", 8, 4),
    ChangeScale("slow", 8, 8),
)


def _family(name: str) -> str:
    value = str(name).lower()
    if "fund" in value:
        return "funding"
    if "oi" in value or "open_interest" in value:
        return "leverage"
    if any(token in value for token in ("volume", "liquid", "depth", "spread", "amihud", "ob_")):
        return "volume_liquidity"
    if any(token in value for token in ("corr", "pc1", "eigen", "dispersion", "breadth", "xasset", "xs_", "cs_", "q_")):
        return "dependence_breadth"
    if any(token in value for token in ("rv", "vol", "atr", "range", "shock", "wick")):
        return "volatility"
    if any(token in value for token in ("ret", "price", "momentum", "trend", "recovery")):
        return "price"
    if any(token in value for token in ("entropy", "gmm", "aegmm", "reconstruction", "mahal", "ood", "drift", "support", "leaf")):
        return "state_uncertainty"
    return "other"


def _sigmoid(value: np.ndarray) -> np.ndarray:
    clipped = np.clip(value, -12.0, 12.0)
    return (1.0 / (1.0 + np.exp(-clipped))).astype(np.float32)


def _window_effect(
    sequence: np.ndarray,
    channel_positions: np.ndarray,
    previous: slice,
    recent: slice,
) -> tuple[np.ndarray, ...]:
    pre = sequence[:, previous, :][:, :, channel_positions]
    post = sequence[:, recent, :][:, :, channel_positions]
    # Normalize each channel before family aggregation.  Pooling raw funding,
    # return, OI and breadth values would make the representation depend on
    # arbitrary units and whichever feature has the largest numerical scale.
    pre_median = np.nanmedian(pre, axis=1).astype(np.float32)
    post_median = np.nanmedian(post, axis=1).astype(np.float32)
    mad = (
        np.nanmedian(np.abs(pre - pre_median[:, None, :]), axis=1).astype(np.float32)
        * np.float32(1.4826)
    )
    mad = np.maximum(mad, EPS)
    signed_by_channel = np.clip((post_median - pre_median) / mad, -8.0, 8.0).astype(np.float32)
    signed = np.nanmedian(signed_by_channel, axis=1).astype(np.float32)
    effect = np.nanmedian(np.abs(signed_by_channel), axis=1).astype(np.float32)
    p_change = _sigmoid(effect - np.float32(1.25))
    combined = np.concatenate([pre, post], axis=1)
    pre_percentile = np.nanmean(combined <= pre_median[:, None, :], axis=(1, 2)).astype(np.float32)
    post_percentile = np.nanmean(combined <= post_median[:, None, :], axis=(1, 2)).astype(np.float32)
    direction = np.sign(signed_by_channel)[:, None, :]
    persistence = np.nanmean(
        direction * (post - pre_median[:, None, :]) > np.float32(0.50) * mad[:, None, :],
        axis=(1, 2),
    ).astype(np.float32)
    valid = np.isfinite(pre_median).any(axis=1) & np.isfinite(post_median).any(axis=1)
    outputs = (signed, effect, p_change, pre_percentile, post_percentile, persistence)
    return tuple(np.where(valid, value, np.nan).astype(np.float32) for value in outputs)


def _boundary_distribution(sequence: np.ndarray) -> dict[str, np.ndarray]:
    """Approximate a run-length posterior from causal robust boundary scores."""
    length = sequence.shape[1]
    previous_bars, recent_bars = 4, 2
    boundaries = np.arange(previous_bars, length - recent_bars + 1, dtype=np.int16)
    if not len(boundaries):
        nan = np.full(len(sequence), np.nan, dtype=np.float32)
        return {name: nan.copy() for name in (
            "p_change_now", "p_change_recent_max", "change_score_area", "score_slope",
            "run_length_mean", "run_length_q05", "run_length_entropy", "change_age",
        )}
    scores: list[np.ndarray] = []
    all_channels = np.arange(sequence.shape[2], dtype=np.int32)
    for boundary in boundaries:
        _, effect, probability, *_ = _window_effect(
            sequence,
            all_channels,
            slice(boundary - previous_bars, boundary),
            slice(boundary, boundary + recent_bars),
        )
        scores.append(np.where(np.isfinite(effect), probability, 0.0).astype(np.float32))
    score = np.column_stack(scores).astype(np.float32)
    ages = (length - boundaries).astype(np.float32)
    logits = np.clip((score - np.nanmax(score, axis=1, keepdims=True)) * 5.0, -20.0, 0.0)
    weights = np.exp(logits).astype(np.float32)
    weights /= np.maximum(weights.sum(axis=1, keepdims=True), EPS)
    order = np.argsort(ages)
    cumulative = np.cumsum(weights[:, order], axis=1)
    q05_position = np.argmax(cumulative >= 0.05, axis=1)
    q05 = ages[order][q05_position]
    entropy = -np.sum(weights * np.log(np.maximum(weights, EPS)), axis=1)
    entropy /= np.log(max(score.shape[1], 2))
    slope_width = min(3, score.shape[1] - 1)
    slope = score[:, -1] - score[:, -(slope_width + 1)] if slope_width else np.zeros(len(score))
    age_mean = np.sum(weights * ages[None, :], axis=1)
    return {
        "p_change_now": score[:, -1].astype(np.float32),
        "p_change_recent_max": np.max(score[:, -min(4, score.shape[1]):], axis=1).astype(np.float32),
        "change_score_area": np.mean(score, axis=1).astype(np.float32),
        "score_slope": slope.astype(np.float32),
        "run_length_mean": age_mean.astype(np.float32),
        "run_length_q05": q05.astype(np.float32),
        "run_length_entropy": entropy.astype(np.float32),
        "change_age": age_mean.astype(np.float32),
    }


def build_causal_change_state(
    sequence: np.ndarray,
    channel_names: Sequence[str],
    *,
    scales: Sequence[ChangeScale] = DEFAULT_CHANGE_SCALES,
) -> tuple[np.ndarray, list[str]]:
    """Return a compact continuous CP vector for each causal input sequence.

    Parameters
    ----------
    sequence:
        ``[rows, time, channels]`` values ending at the decision timestamp.
    channel_names:
        Inference feature names in the final sequence axis.
    scales:
        Nested previous/recent comparisons.  Invalid scales are skipped.
    """
    values = np.asarray(sequence, dtype=np.float32)
    if values.ndim != 3 or values.shape[2] != len(channel_names):
        raise ValueError("sequence/channel contract mismatch")
    families: dict[str, list[int]] = {}
    for index, name in enumerate(channel_names):
        if str(name).startswith("__feature_staleness"):
            continue
        families.setdefault(_family(str(name)), []).append(index)
    columns: dict[str, np.ndarray] = {}
    family_scale_signed: dict[tuple[str, str], np.ndarray] = {}
    family_scale_probability: dict[tuple[str, str], np.ndarray] = {}
    for scale in scales:
        needed = int(scale.previous_bars + scale.recent_bars)
        if needed > values.shape[1]:
            continue
        boundary = values.shape[1] - scale.recent_bars
        for family, positions in families.items():
            signed, effect, probability, pre_pct, post_pct, persistence = _window_effect(
                values,
                np.asarray(positions, dtype=np.int32),
                slice(boundary - scale.previous_bars, boundary),
                slice(boundary, values.shape[1]),
            )
            prefix = f"cp_{family}_{scale.name}"
            columns[f"{prefix}__signed_shift"] = signed
            columns[f"{prefix}__effect_size"] = effect
            columns[f"{prefix}__p_change"] = probability
            columns[f"{prefix}__pre_state_percentile"] = pre_pct
            columns[f"{prefix}__post_state_percentile"] = post_pct
            columns[f"{prefix}__persistence"] = persistence
            family_scale_signed[(family, scale.name)] = signed
            family_scale_probability[(family, scale.name)] = probability
    columns.update({f"cp_global__{name}": value for name, value in _boundary_distribution(values).items()})
    probability_stack = np.column_stack(list(family_scale_probability.values()))
    signed_stack = np.column_stack(list(family_scale_signed.values()))
    columns["cp_global__cross_scale_agreement"] = np.abs(np.nanmean(np.sign(signed_stack), axis=1)).astype(np.float32)
    latest_scale = next((scale.name for scale in reversed(tuple(scales)) if any(key[1] == scale.name for key in family_scale_signed)), None)
    latest = [value for (family, scale), value in family_scale_signed.items() if scale == latest_scale]
    columns["cp_global__cross_family_agreement"] = (
        np.abs(np.nanmean(np.sign(np.column_stack(latest)), axis=1)).astype(np.float32)
        if latest else np.full(len(values), np.nan, dtype=np.float32)
    )
    columns["cp_global__novelty"] = np.sqrt(np.nanmean(np.square(signed_stack), axis=1)).astype(np.float32)
    columns["cp_global__mean_change_probability"] = np.nanmean(probability_stack, axis=1).astype(np.float32)

    def shift(family: str, scale: str = "medium") -> np.ndarray:
        return family_scale_signed.get((family, scale), np.zeros(len(values), dtype=np.float32))

    price = shift("price")
    leverage = shift("leverage")
    volume = shift("volume_liquidity")
    volatility = shift("volatility")
    funding = shift("funding")
    dependence = shift("dependence_breadth")
    pos = lambda value: np.maximum(value, 0.0).astype(np.float32)
    columns["cp_mechanism__compression_expansion"] = (
        pos(volatility) * (np.float32(0.5) + np.float32(0.5) * pos(volume))
    ).astype(np.float32)
    columns["cp_mechanism__long_liquidation"] = (
        pos(-price) * pos(-leverage) * (np.float32(0.5) + np.float32(0.5) * pos(volume))
    ).astype(np.float32)
    columns["cp_mechanism__short_covering"] = (
        pos(price) * pos(-leverage) * (np.float32(0.5) + np.float32(0.5) * pos(volume))
    ).astype(np.float32)
    columns["cp_mechanism__funding_transition"] = np.abs(funding).astype(np.float32)
    columns["cp_mechanism__correlation_fragmentation"] = (
        np.abs(dependence) * (np.float32(0.5) + np.float32(0.5) * pos(volatility))
    ).astype(np.float32)
    mechanism = np.column_stack([
        columns["cp_mechanism__compression_expansion"],
        columns["cp_mechanism__long_liquidation"],
        columns["cp_mechanism__short_covering"],
        columns["cp_mechanism__funding_transition"],
        columns["cp_mechanism__correlation_fragmentation"],
    ])
    mechanism_confidence = np.tanh(np.nanmax(mechanism, axis=1)).astype(np.float32)
    columns["cp_mechanism__unknown_transition"] = (
        columns["cp_global__p_change_now"] * (np.float32(1.0) - mechanism_confidence)
    ).astype(np.float32)
    names = list(columns)
    matrix = np.column_stack([columns[name] for name in names]).astype(np.float32)
    matrix[~np.isfinite(matrix)] = np.nan
    return matrix, names


def build_streaming_long_change_state(
    frame: pd.DataFrame,
    channel_names: Sequence[str],
    *,
    durations: Sequence[int] = (48, 96, 168),
    normalization_span: int = 720,
) -> pd.DataFrame:
    """Build compact 2--7 day change summaries without sequence expansion.

    Source channels use a one-bar-lagged EWM location and scale. They are then
    collapsed into economic families before rolling moments are calculated, so
    memory grows with ``time x families`` rather than
    ``rows x 168 x source_features``. Every output at ``t`` is causal.
    """
    if len(frame.columns) != len(channel_names):
        raise ValueError("streaming frame/channel contract mismatch")
    values = frame.apply(pd.to_numeric, errors="coerce").astype(np.float32, copy=False)
    minimum = max(24, int(normalization_span // 12))
    location = values.ewm(span=normalization_span, adjust=False, min_periods=minimum).mean().shift(1)
    scale = values.ewm(span=normalization_span, adjust=False, min_periods=minimum).std().shift(1)
    standardized = ((values - location) / scale.clip(lower=1e-4)).clip(-8.0, 8.0)
    families: dict[str, list[str]] = {}
    for name in channel_names:
        families.setdefault(_family(str(name)), []).append(str(name))
    family_frame = pd.DataFrame(index=values.index)
    for family, names in families.items():
        family_frame[family] = standardized.loc[:, names].median(axis=1, skipna=True).astype(np.float32)

    outputs: dict[str, pd.Series] = {}
    mechanism_by_duration: dict[str, dict[int, pd.Series]] = {}
    for duration_value in durations:
        duration = int(duration_value)
        if duration < 24:
            raise ValueError("long change durations must be at least 24 bars")
        label = f"{duration}h"
        min_periods = max(12, duration // 2)
        recent_mean = family_frame.rolling(duration, min_periods=min_periods).mean()
        previous_mean = recent_mean.shift(duration)
        previous_std = family_frame.rolling(duration, min_periods=min_periods).std().shift(duration)
        signed = ((recent_mean - previous_mean) / previous_std.clip(lower=0.20)).clip(-8.0, 8.0)
        effect = signed.abs()
        probability = 1.0 / (1.0 + np.exp(-(effect - 1.25).clip(-12.0, 12.0)))
        slope = recent_mean - recent_mean.shift(max(6, duration // 8))
        recent_max = probability.rolling(max(12, duration // 4), min_periods=1).max()
        area = probability.ewm(halflife=max(6, duration // 6), adjust=False).mean()
        for family in family_frame.columns:
            prefix = f"cp_long_{family}_{label}"
            outputs[f"{prefix}__signed_shift"] = signed[family].astype(np.float32)
            outputs[f"{prefix}__effect_size"] = effect[family].astype(np.float32)
            outputs[f"{prefix}__p_change"] = probability[family].astype(np.float32)
            outputs[f"{prefix}__pre_state"] = previous_mean[family].astype(np.float32)
            outputs[f"{prefix}__post_state"] = recent_mean[family].astype(np.float32)
            outputs[f"{prefix}__score_slope"] = slope[family].astype(np.float32)
            outputs[f"{prefix}__recent_max"] = recent_max[family].astype(np.float32)
            outputs[f"{prefix}__score_area"] = area[family].astype(np.float32)
        outputs[f"cp_long_global_{label}__cross_family_agreement"] = (
            signed.apply(np.sign).mean(axis=1).abs().astype(np.float32)
        )
        outputs[f"cp_long_global_{label}__novelty"] = (
            np.sqrt(signed.pow(2).mean(axis=1)).astype(np.float32)
        )
        outputs[f"cp_long_global_{label}__mean_change_probability"] = (
            probability.mean(axis=1).astype(np.float32)
        )
        zero = pd.Series(0.0, index=family_frame.index, dtype=np.float32)
        family_signed = lambda name: signed[name] if name in signed else zero
        family_probability = lambda name: probability[name] if name in probability else zero
        family_pre = lambda name: previous_mean[name] if name in previous_mean else zero
        family_slope = lambda name: slope[name] if name in slope else zero
        positive = lambda value: value.clip(lower=0.0, upper=8.0)
        price = family_signed("price")
        leverage = family_signed("leverage")
        volume = family_signed("volume_liquidity")
        volatility = family_signed("volatility")
        funding = family_signed("funding")
        dependence = family_signed("dependence_breadth")
        raw_mechanisms = {
            "compression_expansion": (
                positive(volatility)
                * (0.5 + 0.5 * positive(volume))
                * (0.5 + 0.5 * positive(-family_pre("volatility")))
            ),
            "long_liquidation": (
                positive(-price) * positive(-leverage)
                * (0.5 + 0.5 * positive(volume))
            ),
            "short_covering": (
                positive(price) * positive(-leverage)
                * (0.5 + 0.5 * positive(volume))
            ),
            "funding_transition": (
                funding.abs() * family_probability("funding")
            ),
            "correlation_break": (
                dependence.abs() * (0.5 + 0.5 * positive(volatility))
            ),
            "deleveraging_exhaustion": (
                positive(-leverage) * positive(family_slope("leverage"))
                * (0.5 + 0.5 * positive(price))
            ),
        }
        for mechanism, raw in raw_mechanisms.items():
            # Bounded, dimensionless outputs are deliberately MLP/GMM friendly.
            value = np.tanh(raw.clip(lower=0.0, upper=12.0)).astype(np.float32)
            outputs[f"cp_long_mechanism_{mechanism}_{label}"] = value
            mechanism_by_duration.setdefault(mechanism, {})[duration] = value
    for mechanism, values_by_duration in mechanism_by_duration.items():
        ordered = [values_by_duration[int(value)] for value in durations if int(value) in values_by_duration]
        if not ordered:
            continue
        matrix = pd.concat(ordered, axis=1)
        prefix = f"cp_long_mechanism_{mechanism}_cross_scale"
        outputs[f"{prefix}__mean"] = matrix.mean(axis=1).astype(np.float32)
        outputs[f"{prefix}__max"] = matrix.max(axis=1).astype(np.float32)
        outputs[f"{prefix}__trend"] = (matrix.iloc[:, -1] - matrix.iloc[:, 0]).astype(np.float32)
        dispersion = matrix.std(axis=1).fillna(0.0)
        outputs[f"{prefix}__agreement"] = (
            1.0 - (dispersion / (matrix.mean(axis=1).abs() + 0.25)).clip(0.0, 1.0)
        ).astype(np.float32)
    return pd.DataFrame(outputs, index=frame.index, dtype=np.float32)


__all__ = [
    "ChangeScale",
    "DEFAULT_CHANGE_SCALES",
    "build_causal_change_state",
    "build_streaming_long_change_state",
]
