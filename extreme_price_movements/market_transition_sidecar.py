"""Causal hourly market-transition features on a compact market spine.

This module is deliberately target-free.  It consumes one immutable hourly
market panel and produces reusable transition features which can be joined
backward to candidates.  Expensive online detectors run once per historical
hour, never per asset, candidate, or evaluation period.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from .bayesian_changepoint import BOCPDConfig, bocpd_student_t_run_summary


SCHEMA = "market_transition_sidecar_v1"
HOUR = pd.Timedelta(hours=1)

# Explicit mechanisms rather than outcome-selected columns.  The materializer
# resolves the first available causal source for every key and records it in
# its manifest.  All names refer to existing hourly multiview fields.
SPINE_SOURCE_PATTERNS: Mapping[str, tuple[str, ...]] = {
    "trend": ("breakout_efficiency", "market_trend", "trend_quality"),
    "volatility": ("breadth_dispersion__realized_vol", "market_realized_vol"),
    "vol_of_vol": ("breadth_dispersion__vol_of_vol", "vol_of_vol"),
    "breadth": ("downside_breadth_intensity", "breadth_dispersion__delta"),
    "dispersion": ("breadth_dispersion__robust_z", "btc_decoupling_dispersion"),
    "correlation": ("correlation_heterogeneity_dispersion", "correlation_breakdown_dispersion"),
    "activity": ("liquidity_xs__log_quote_volume__mean", "quote_volume"),
    "liquidity": ("liquidity_xs__amihud_illiq__mean", "ob_spread_bps_z_24h__mean"),
    "deleveraging": ("funding_deleveraging_divergence", "oi_contraction"),
    "rebound": ("broad_washout_recovery", "short_covering_score", "deleveraged_range_climax"),
    "funding": ("funding_confirmed_long_flush", "funding_deleveraging"),
    "exhaustion": ("short_breakout_exhaustion", "breakout_exhaustion"),
}


@dataclass(frozen=True)
class TransitionSidecarConfig:
    timestamp_col: str = "source_utc"
    robust_window_hours: int = 24 * 30
    min_reference_hours: int = 24 * 7
    ewma_half_lives: tuple[int, int, int, int] = (3, 12, 48, 168)
    bocpd_inputs: int = 8
    bocpd_run_hours: tuple[int, int] = (24 * 7, 24 * 30)
    bocpd_max_run_hours: int = 24 * 30
    covariance_fast_hours: int = 24
    covariance_medium_hours: int = 24 * 7
    covariance_slow_hours: int = 24 * 30
    covariance_update_hours: int = 6
    distribution_short_hours: int = 12
    distribution_reference_hours: int = 24 * 7
    source_coverage_floor: float = .80


def resolve_spine_sources(columns: Sequence[str]) -> dict[str, str]:
    """Resolve one level-like, causal source for each declared mechanism."""
    result: dict[str, str] = {}
    available = [str(column) for column in columns]
    # A raw/level proxy should win over a derived delta.  The final hourly
    # robust-z is still acceptable because it was materialised causally by the
    # multiview panel and is less noisy than a one-bar delta.
    preference = ("robust_z_1h", "delta_1h", "realized_vol_1h", "vol_of_vol_1h")
    for name, patterns in SPINE_SOURCE_PATTERNS.items():
        options = [column for column in available if any(pattern in column for pattern in patterns)]
        if not options:
            continue
        result[name] = min(
            options,
            key=lambda column: (
                next((rank for rank, suffix in enumerate(preference) if column.endswith(suffix)), len(preference)),
                len(column), column,
            ),
        )
    if len(result) < 8:
        raise ValueError(f"need at least eight compact market mechanisms; resolved only {result}")
    return result


def _strict_robust_z(series: pd.Series, config: TransitionSidecarConfig) -> tuple[pd.Series, pd.Series]:
    """Prior-only rolling median/MAD normalization and reference coverage."""
    prior = series.shift(1)
    median = prior.rolling(config.robust_window_hours, min_periods=config.min_reference_hours).median()
    # ``median`` at t is already prior-only; shift deviation once more so the
    # scale never incorporates x_t either.
    deviation = (series - median).abs().shift(1)
    mad = deviation.rolling(config.robust_window_hours, min_periods=config.min_reference_hours).median()
    scale = (1.4826 * mad).clip(lower=1e-5)
    z = ((series - median) / scale).clip(-10., 10.).astype("float32")
    coverage = prior.notna().rolling(config.robust_window_hours, min_periods=1).mean().astype("float32")
    return z, coverage


def _hours_since_change(sign: np.ndarray) -> np.ndarray:
    result = np.full(len(sign), np.nan, dtype=np.float32)
    last = -1
    previous = 0
    for index, value in enumerate(sign):
        if not np.isfinite(value):
            continue
        current = int(np.sign(value))
        if index and current != previous:
            last = index
        result[index] = 0. if last == index else (float(index - last) if last >= 0 else np.nan)
        previous = current
    return result


def _derivatives(z: pd.Series, config: TransitionSidecarConfig) -> pd.DataFrame:
    fast_h, medium_h, slow_h, structural_h = config.ewma_half_lives
    e_fast = z.ewm(halflife=fast_h, adjust=False, min_periods=1).mean()
    e_medium = z.ewm(halflife=medium_h, adjust=False, min_periods=1).mean()
    e_slow = z.ewm(halflife=slow_h, adjust=False, min_periods=1).mean()
    e_structural = z.ewm(halflife=structural_h, adjust=False, min_periods=1).mean()
    velocity_fast = e_fast - e_medium
    velocity_slow = e_medium - e_slow
    acceleration = velocity_fast - velocity_slow
    structural_acceleration = velocity_slow - (e_slow - e_structural)
    return pd.DataFrame({
        "level_z": z,
        "velocity_fast": velocity_fast,
        "velocity_slow": velocity_slow,
        "acceleration": acceleration,
        "structural_acceleration": structural_acceleration,
        "absolute_acceleration": acceleration.abs(),
        "acceleration_sign": np.sign(acceleration).astype("float32"),
        "hours_since_acceleration_sign_change": _hours_since_change(acceleration.to_numpy(float)),
    }, index=z.index).astype("float32")


def _bocpd_features(values: pd.DataFrame, config: TransitionSidecarConfig) -> pd.DataFrame:
    """Two-hazard BOCPD summaries over only the compact selected spine."""
    selected = list(values.columns)[: config.bocpd_inputs]
    per_hazard: list[np.ndarray] = []
    output = pd.DataFrame(index=values.index)
    for horizon in config.bocpd_run_hours:
        summaries = []
        for column in selected:
            array = values[column].to_numpy(np.float32)
            summary = bocpd_student_t_run_summary(
                array,
                BOCPDConfig(expected_run_hours=horizon, max_run_hours=config.bocpd_max_run_hours),
            )
            summaries.append(summary)
        cube = np.stack(summaries, axis=1)  # rows, bundles, [cp, mean, q05, entropy]
        suffix = f"h{horizon}"
        output[f"market_transition__bocpd_{suffix}__mean_cp"] = np.nanmean(cube[:, :, 0], axis=1)
        output[f"market_transition__bocpd_{suffix}__max_cp"] = np.nanmax(cube[:, :, 0], axis=1)
        output[f"market_transition__bocpd_{suffix}__breadth_cp_010"] = np.nanmean(cube[:, :, 0] >= .10, axis=1)
        output[f"market_transition__bocpd_{suffix}__mean_run_length"] = np.nanmean(cube[:, :, 1], axis=1)
        output[f"market_transition__bocpd_{suffix}__run_entropy"] = np.nanmean(cube[:, :, 3], axis=1)
        output[f"market_transition__bocpd_{suffix}__cp_max_6h"] = output[f"market_transition__bocpd_{suffix}__max_cp"].rolling(6, min_periods=1).max()
        per_hazard.append(cube[:, :, 0])
    if len(per_hazard) == 2:
        output["market_transition__bocpd__hazard_disagreement"] = np.nanmean(np.abs(per_hazard[0] - per_hazard[1]), axis=1)
    return output.astype("float32")


def _ewma_covariance_breaks(values: pd.DataFrame, config: TransitionSidecarConfig) -> pd.DataFrame:
    """Causal shrinkage covariance/correlation break summaries.

    Eigensystems are calculated only every six hours and forward carried.  A
    12-dimensional bundle spine makes this inexpensive and stable.
    """
    matrix = values.to_numpy(np.float64)
    n_rows, n_features = matrix.shape
    output = np.full((n_rows, 10), np.nan, dtype=np.float32)
    fast = np.eye(n_features, dtype=np.float64) * 1e-3
    medium = fast.copy(); slow = fast.copy()
    means = np.zeros(n_features, dtype=np.float64)
    alpha = [1. - np.exp(np.log(.5) / horizon) for horizon in (config.covariance_fast_hours, config.covariance_medium_hours, config.covariance_slow_hours)]
    last = np.full(10, np.nan, dtype=np.float32)
    for row in range(n_rows):
        x = matrix[row]
        finite = np.isfinite(x)
        if finite.mean() >= config.source_coverage_floor:
            filled = np.where(finite, x, means)
            means = .99 * means + .01 * filled
            centered = filled - means
            outer = np.outer(centered, centered)
            fast = (1. - alpha[0]) * fast + alpha[0] * outer
            medium = (1. - alpha[1]) * medium + alpha[1] * outer
            slow = (1. - alpha[2]) * slow + alpha[2] * outer
        if row % config.covariance_update_hours == 0 and np.isfinite(fast).all() and np.isfinite(slow).all():
            diagonal = np.diag(np.diag(slow)); shrink_slow = .9 * slow + .1 * diagonal
            diagonal_fast = np.diag(np.diag(fast)); shrink_fast = .9 * fast + .1 * diagonal_fast
            norm = max(float(np.linalg.norm(shrink_slow, "fro")), 1e-6)
            corr_fast = shrink_fast / np.sqrt(np.outer(np.diag(shrink_fast), np.diag(shrink_fast))).clip(1e-8)
            corr_slow = shrink_slow / np.sqrt(np.outer(np.diag(shrink_slow), np.diag(shrink_slow))).clip(1e-8)
            eig_fast = np.linalg.eigvalsh(shrink_fast); eig_slow = np.linalg.eigvalsh(shrink_slow)
            corr_eig = np.linalg.eigvalsh(corr_fast)
            off = corr_fast[np.triu_indices(n_features, 1)]
            last = np.asarray([
                np.linalg.norm(shrink_fast - shrink_slow, "fro") / norm,
                np.linalg.norm(medium - slow, "fro") / norm,
                np.trace(shrink_fast) / max(np.trace(shrink_slow), 1e-6),
                eig_fast[-1] / max(eig_slow[-1], 1e-6),
                eig_fast[-1] / max(eig_fast.sum(), 1e-6),
                (eig_fast.sum() ** 2) / max((eig_fast ** 2).sum(), 1e-6),
                np.linalg.norm(corr_fast - corr_slow, "fro") / max(np.linalg.norm(corr_slow, "fro"), 1e-6),
                np.mean(off), np.std(off), np.mean(off < 0.),
            ], dtype=np.float32)
        output[row] = last
    names = ("covariance_break_fast_slow", "covariance_break_medium_slow", "covariance_trace_ratio", "covariance_eig1_change", "covariance_eig1_share", "covariance_effective_rank", "correlation_frobenius_break", "correlation_mean_offdiag", "correlation_dispersion", "correlation_negative_share")
    return pd.DataFrame(output, columns=[f"market_transition__{name}" for name in names], index=values.index)


def _distribution_breaks(values: pd.DataFrame, config: TransitionSidecarConfig) -> pd.DataFrame:
    short = values.rolling(config.distribution_short_hours, min_periods=max(3, config.distribution_short_hours // 2))
    reference = values.shift(1).rolling(config.distribution_reference_hours, min_periods=config.min_reference_hours)
    short_median = short.median(); reference_median = reference.median()
    short_iqr = short.quantile(.75) - short.quantile(.25)
    reference_iqr = reference.quantile(.75) - reference.quantile(.25)
    median_shift = (short_median - reference_median).abs()
    scale_ratio = short_iqr / reference_iqr.clip(lower=1e-5)
    q90_shift = (short.quantile(.90) - reference.quantile(.90)).abs()
    return pd.DataFrame({
        "market_transition__distribution_median_shift_mean": median_shift.mean(axis=1),
        "market_transition__distribution_median_shift_max": median_shift.max(axis=1),
        "market_transition__distribution_scale_ratio_mean": scale_ratio.mean(axis=1),
        "market_transition__distribution_scale_ratio_max": scale_ratio.max(axis=1),
        "market_transition__distribution_q90_shift_mean": q90_shift.mean(axis=1),
        "market_transition__distribution_q90_shift_max": q90_shift.max(axis=1),
    }, index=values.index).astype("float32")


def _regime_probability_dynamics(frame: pd.DataFrame) -> pd.DataFrame:
    probabilities = [column for column in frame if str(column).startswith("market_regime__state_p_")]
    if len(probabilities) < 2:
        return pd.DataFrame(index=frame.index)
    p = frame[probabilities].to_numpy(np.float32)
    previous = np.vstack([np.full((1, p.shape[1]), np.nan, dtype=np.float32), p[:-1]])
    velocity = np.nansum(np.abs(p - previous), axis=1)
    entropy = -np.nansum(np.clip(p, 1e-8, 1.) * np.log(np.clip(p, 1e-8, 1.)), axis=1) / np.log(p.shape[1])
    margin = np.partition(p, -2, axis=1)[:, -1] - np.partition(p, -2, axis=1)[:, -2]
    changed = (np.nanargmax(p, axis=1) != np.r_[np.nanargmax(p[0]), np.nanargmax(p[:-1], axis=1)]).astype(np.float32)
    return pd.DataFrame({
        "market_transition__regime_entropy": entropy,
        "market_transition__regime_top2_margin": margin,
        "market_transition__regime_probability_velocity": velocity,
        "market_transition__regime_probability_acceleration": pd.Series(velocity).diff().to_numpy(),
        "market_transition__regime_state_changed": changed,
        "market_transition__regime_changes_24h": pd.Series(changed).rolling(24, min_periods=1).sum().to_numpy(),
    }, index=frame.index).astype("float32")


def market_transition_feature_names(spine_names: Sequence[str] = tuple(SPINE_SOURCE_PATTERNS)) -> list[str]:
    """Stable feature contract emitted when all declared spine sources exist."""
    fields = [
        f"market_transition__{name}__{operator}"
        for name in spine_names
        for operator in ("level_z", "velocity_fast", "velocity_slow", "acceleration", "structural_acceleration", "absolute_acceleration", "acceleration_sign", "hours_since_acceleration_sign_change")
    ]
    for horizon in (24 * 7, 24 * 30):
        fields += [
            f"market_transition__bocpd_h{horizon}__{operator}"
            for operator in ("mean_cp", "max_cp", "breadth_cp_010", "mean_run_length", "run_entropy", "cp_max_6h")
        ]
    fields += [
        "market_transition__bocpd__hazard_disagreement",
        "market_transition__covariance_break_fast_slow", "market_transition__covariance_break_medium_slow", "market_transition__covariance_trace_ratio", "market_transition__covariance_eig1_change", "market_transition__covariance_eig1_share", "market_transition__covariance_effective_rank", "market_transition__correlation_frobenius_break", "market_transition__correlation_mean_offdiag", "market_transition__correlation_dispersion", "market_transition__correlation_negative_share",
        "market_transition__distribution_median_shift_mean", "market_transition__distribution_median_shift_max", "market_transition__distribution_scale_ratio_mean", "market_transition__distribution_scale_ratio_max", "market_transition__distribution_q90_shift_mean", "market_transition__distribution_q90_shift_max",
        "market_transition__regime_entropy", "market_transition__regime_top2_margin", "market_transition__regime_probability_velocity", "market_transition__regime_probability_acceleration", "market_transition__regime_state_changed", "market_transition__regime_changes_24h",
        "market_transition__source_coverage", "market_transition__source_missing_share", "market_transition__acceleration_breadth", "market_transition__level_break", "market_transition__relationship_break", "market_transition__transition_velocity", "market_transition__transition_uncertainty", "market_transition__systemic_transition_breadth", "market_transition__transition_intensity",
    ]
    return fields


def build_market_transition_sidecar(frame: pd.DataFrame, *, source_columns: Mapping[str, str], config: TransitionSidecarConfig = TransitionSidecarConfig()) -> tuple[pd.DataFrame, list[str]]:
    """Build the immutable, hourly causal transition feature table."""
    if config.timestamp_col not in frame:
        raise KeyError(f"missing {config.timestamp_col}")
    missing = [column for column in source_columns.values() if column not in frame]
    if missing:
        raise KeyError(f"missing resolved market-spine sources: {missing}")
    panel = frame.sort_values(config.timestamp_col).reset_index(drop=True).copy()
    timestamp = pd.to_datetime(panel[config.timestamp_col], utc=True, errors="raise")
    if timestamp.duplicated().any() or not timestamp.is_monotonic_increasing:
        raise ValueError("market transition sidecar requires one sorted row per hourly timestamp")
    raw = pd.DataFrame({name: pd.to_numeric(panel[column], errors="coerce") for name, column in source_columns.items()})
    normalized = pd.DataFrame(index=panel.index); derivative_blocks = []
    coverage_columns = []
    for name in raw:
        z, coverage = _strict_robust_z(raw[name], config)
        normalized[name] = z
        coverage_columns.append(coverage)
        derivative = _derivatives(z, config).add_prefix(f"market_transition__{name}__")
        derivative_blocks.append(derivative)
    derivatives = pd.concat(derivative_blocks, axis=1)
    bocpd = _bocpd_features(normalized, config)
    covariance = _ewma_covariance_breaks(normalized, config)
    distribution = _distribution_breaks(normalized, config)
    regime = _regime_probability_dynamics(panel)
    acceleration_columns = [column for column in derivatives if column.endswith("__absolute_acceleration")]
    # A global median would let later periods change the earlier transition
    # breadth.  Every acceleration threshold is therefore a prior-only local
    # reference, exactly like the primitive normalizations.
    acceleration_reference = derivatives[acceleration_columns].shift(1).rolling(
        config.robust_window_hours, min_periods=config.min_reference_hours
    ).median()
    transition_breadth = (derivatives[acceleration_columns] > acceleration_reference).mean(axis=1)
    output = pd.concat([pd.DataFrame({config.timestamp_col: timestamp}), derivatives, bocpd, covariance, distribution, regime], axis=1)
    output["market_transition__source_coverage"] = pd.concat(coverage_columns, axis=1).mean(axis=1).astype("float32")
    output["market_transition__source_missing_share"] = raw.isna().mean(axis=1).astype("float32")
    output["market_transition__acceleration_breadth"] = transition_breadth.astype("float32")
    level_break = pd.concat([bocpd.filter(like="mean_cp"), distribution.filter(like="median_shift")], axis=1).mean(axis=1)
    relationship_break = covariance[["market_transition__covariance_break_fast_slow", "market_transition__correlation_frobenius_break"]].mean(axis=1)
    velocity = pd.concat([derivatives[acceleration_columns].mean(axis=1), regime.filter(like="probability_velocity")], axis=1).mean(axis=1)
    uncertainty = pd.concat([bocpd.filter(like="run_entropy"), regime.filter(like="regime_entropy")], axis=1).mean(axis=1)
    breadth = pd.concat([bocpd.filter(like="breadth_cp"), output[["market_transition__acceleration_breadth"]]], axis=1).mean(axis=1)
    output["market_transition__level_break"] = level_break.astype("float32")
    output["market_transition__relationship_break"] = relationship_break.astype("float32")
    output["market_transition__transition_velocity"] = velocity.astype("float32")
    output["market_transition__transition_uncertainty"] = uncertainty.astype("float32")
    output["market_transition__systemic_transition_breadth"] = breadth.astype("float32")
    output["market_transition__transition_intensity"] = pd.concat([output[["market_transition__level_break", "market_transition__relationship_break", "market_transition__transition_velocity", "market_transition__transition_uncertainty", "market_transition__systemic_transition_breadth"]]], axis=1).mean(axis=1).astype("float32")
    features = [column for column in output if column != config.timestamp_col]
    return output, features


__all__ = ["SCHEMA", "SPINE_SOURCE_PATTERNS", "TransitionSidecarConfig", "build_market_transition_sidecar", "market_transition_feature_names", "resolve_spine_sources"]
