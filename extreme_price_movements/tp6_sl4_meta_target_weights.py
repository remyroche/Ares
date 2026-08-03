"""Vectorised TP6/SL4 meta-target and sample-weight contracts.

This module deliberately does *not* fit a model.  It converts a panel that
contains cross-fitted, same-side base predictions and resolved TP6/SL4 H12
labels into well-defined meta learning targets.  Outcome columns are training
labels, never inference features.  Quantiles, priors, bin means, and class
frequencies are injected through :class:`MetaTrainingStatistics`, so callers
can fit them on each training fold and reuse them unchanged on validation/OOS.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Mapping

import numpy as np
import pandas as pd


MetaTargetName = Literal[
    "M0", "M1", "M2", "M3", "M5", "M6", "M7", "M8", "M9", "M10",
    "M11", "M12", "M13", "MC1", "MC2", "MC3",
]
MetaWeightName = Literal["MW0", "MW1", "MW2", "MW3", "MW4", "MW5", "MW6", "MW7", "MW8"]


@dataclass(frozen=True)
class MetaColumns:
    """Canonical names; override these rather than mutating the input panel."""

    realised_net_bps: str = "t4_tp6_sl4_net_bps"
    causal_base_expected_net_bps: str = "causal_base_expected_net_bps"
    entry_atr_bps: str = "entry_atr_bps"
    base_rank_percentile: str = "causal_base_rank_percentile"
    base_margin: str = "causal_base_margin"
    base_entropy: str = "causal_base_entropy"
    label_stability: str = "label_stability"
    base_target_stability: str = "base_target_stability"
    path_completeness: str = "path_completeness"
    event_conflict: str = "event_conflict"
    label_certainty: str = "label_certainty"
    class_label: str = "meta_class_label"
    side: str = "side_name"
    timestamp: str = "__ts__"


@dataclass(frozen=True)
class MetaTargetParameters:
    """Numerical policy only; all bps quantities are in realised-net units."""

    shrink_factor: float = 1.0
    soft_hurdle_bps: float = 0.0
    soft_tau_bps: float = 100.0
    robust_hurdle_bps: float = 25.0
    overestimate_margin_bps: float = 50.0
    trust_tau_bps: float = 100.0
    residual_clip_key: Literal["p1_99", "p3_97", "p5_95"] = "p1_99"
    ordinal_edges_bps: tuple[float, ...] = (-300.0, -100.0, 0.0, 100.0, 250.0)
    certainty_power: float = 1.0
    margin_scale: float = 1.0
    entropy_scale: float = 1.0


@dataclass(frozen=True)
class MetaWeightParameters:
    """All weights are made finite, positive, bounded and mean-one."""

    min_weight: float = 0.50
    max_weight: float = 2.00
    high_base_center: float = 0.80
    high_base_tau: float = 0.08
    high_base_floor: float = 0.50
    high_base_scale: float = 1.00
    economic_lambda: float = 0.25
    economic_cap_bps: float = 200.0
    boundary_hurdle_bps: float = 0.0
    boundary_tau_bps: float = 100.0
    boundary_floor: float = 0.50
    boundary_scale: float = 1.00
    class_balance_variant: Literal["natural", "sqrt_inverse", "effective"] = "natural"
    mild_failure_lambda: float = 0.25
    combined_exponents: tuple[float, float, float] = (1.0, 1.0, 1.0)


@dataclass(frozen=True)
class MetaTrainingStatistics:
    """Training-only estimates that must be frozen for validation/OOS use."""

    residual_clips: Mapping[str, tuple[float, float]] = field(default_factory=dict)
    ordinal_bin_means: tuple[float, ...] = ()
    prior_clear_rate: float | None = None
    class_counts: Mapping[int, int] = field(default_factory=dict)


@dataclass(frozen=True)
class MetaTargetBundle:
    """One or more finite targets plus conditional-head validity masks.

    ``primary`` is the direct score for single-head models.  Multi-part targets
    retain components in ``heads``: e.g. M9 has ``failure_probability`` and
    ``failure_severity`` with the latter valid only on failures.
    """

    name: str
    task: Literal["regression", "binary", "ordinal", "multi_head"]
    primary: np.ndarray
    heads: Mapping[str, np.ndarray] = field(default_factory=dict)
    valid_masks: Mapping[str, np.ndarray] = field(default_factory=dict)
    reconstruction: Mapping[str, object] = field(default_factory=dict)


def _require(frame: pd.DataFrame, *names: str) -> None:
    missing = [name for name in names if name not in frame.columns]
    if missing:
        raise KeyError(f"meta panel lacks required columns: {missing}")


def _numeric(frame: pd.DataFrame, *names: str) -> tuple[np.ndarray, ...]:
    _require(frame, *names)
    values: list[np.ndarray] = []
    for name in names:
        value = pd.to_numeric(frame[name], errors="raise").to_numpy(dtype=float)
        if not np.isfinite(value).all():
            raise ValueError(f"non-finite values in {name!r}")
        values.append(value)
    return tuple(values)


def _sigmoid(value: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(value, -35.0, 35.0)))


def _residual(frame: pd.DataFrame, columns: MetaColumns) -> np.ndarray:
    realised, base = _numeric(frame, columns.realised_net_bps, columns.causal_base_expected_net_bps)
    return realised - base


def _validate_target_parameters(p: MetaTargetParameters) -> None:
    if not 0.0 < p.shrink_factor <= 1.0:
        raise ValueError("shrink_factor must be in (0, 1]")
    if p.soft_tau_bps <= 0.0 or p.trust_tau_bps <= 0.0:
        raise ValueError("soft_tau_bps and trust_tau_bps must be positive")
    if p.margin_scale < 0.0 or p.entropy_scale < 0.0 or p.certainty_power <= 0.0:
        raise ValueError("certainty/margin/entropy scales must be non-negative (certainty positive)")


def _assert_finite_bundle(bundle: MetaTargetBundle) -> MetaTargetBundle:
    arrays = {"primary": bundle.primary, **bundle.heads}
    for key, value in arrays.items():
        if value.ndim != 1 or not np.isfinite(value).all():
            raise AssertionError(f"{bundle.name}:{key} must be a finite one-dimensional array")
    for key, mask in bundle.valid_masks.items():
        if mask.ndim != 1 or mask.dtype != bool or len(mask) != len(bundle.primary):
            raise AssertionError(f"{bundle.name}:{key} must be a boolean row mask")
    return bundle


def fit_meta_training_statistics(
    frame: pd.DataFrame, *, columns: MetaColumns = MetaColumns(),
    ordinal_edges_bps: tuple[float, ...] = (-300., -100., 0., 100., 250.),
    class_label: np.ndarray | None = None,
) -> MetaTrainingStatistics:
    """Fit reusable target statistics on *one training fold only*.

    Callers must not pass a validation/OOS panel here.  The separate function
    makes that lineage visible in the runner rather than hiding quantile
    fitting inside ``build_meta_target``.
    """
    residual = _residual(frame, columns)
    clips = {
        "p1_99": tuple(np.quantile(residual, [0.01, 0.99]).astype(float)),
        "p3_97": tuple(np.quantile(residual, [0.03, 0.97]).astype(float)),
        "p5_95": tuple(np.quantile(residual, [0.05, 0.95]).astype(float)),
    }
    realised = _numeric(frame, columns.realised_net_bps)[0]
    ordinal = np.digitize(realised, ordinal_edges_bps, right=True)
    means = tuple(float(realised[ordinal == index].mean()) if (ordinal == index).any() else float(np.nanmean(realised))
                  for index in range(len(ordinal_edges_bps) + 1))
    labels = ordinal if class_label is None else np.asarray(class_label)
    if len(labels) != len(frame):
        raise ValueError("class_label must align with frame")
    label_int = np.asarray(labels, dtype=int)
    counts = {int(key): int(value) for key, value in zip(*np.unique(label_int, return_counts=True))}
    return MetaTrainingStatistics(
        residual_clips=clips,
        ordinal_bin_means=means,
        prior_clear_rate=float(np.mean(realised > 0.0)),
        class_counts=counts,
    )


def build_tail_training_mask(
    frame: pd.DataFrame, top_fraction: float, *, columns: MetaColumns = MetaColumns(),
) -> np.ndarray:
    """M0-bis population gate from fold-local *per-side* base rank.

    ``base_rank_percentile`` must be rank-normalised independently by side on
    the training fold (1=best).  It is intentionally a caller materialisation
    requirement: global ranks would leak allocation decisions across sides.
    """
    if not 0.0 < top_fraction <= 1.0:
        raise ValueError("top_fraction must lie in (0, 1]")
    percentile = _numeric(frame, columns.base_rank_percentile)[0]
    if (percentile < 0.0).any() or (percentile > 1.0).any():
        raise ValueError("base_rank_percentile must be per-side values in [0, 1]")
    return percentile >= 1.0 - top_fraction


def build_meta_target(
    frame: pd.DataFrame, name: MetaTargetName, *, columns: MetaColumns = MetaColumns(),
    parameters: MetaTargetParameters = MetaTargetParameters(),
    statistics: MetaTrainingStatistics | None = None,
) -> MetaTargetBundle:
    """Build M0--MC3 targets without fitting or looking beyond this frame.

    M2 and M8 require ``statistics`` because their quantiles/bin means must be
    determined from the training fold then injected unchanged into OOS rows.
    M9--M11 expose component heads rather than silently training conditional
    losses on zero-filled non-applicable rows.
    """
    _validate_target_parameters(parameters)
    residual = _residual(frame, columns)
    realised = _numeric(frame, columns.realised_net_bps)[0]
    if name == "M0":
        bundle = MetaTargetBundle(name, "regression", residual)
    elif name == "M1":
        bundle = MetaTargetBundle(name, "regression", parameters.shrink_factor * residual)
    elif name == "M2":
        if statistics is None or parameters.residual_clip_key not in statistics.residual_clips:
            raise ValueError("M2 needs training-only residual clip statistics")
        lo, hi = statistics.residual_clips[parameters.residual_clip_key]
        if not np.isfinite([lo, hi]).all() or lo >= hi:
            raise ValueError("invalid frozen residual clip bounds")
        bundle = MetaTargetBundle(name, "regression", np.clip(residual, lo, hi), reconstruction={"clip_bps": (lo, hi)})
    elif name == "M3":
        atr = _numeric(frame, columns.entry_atr_bps)[0]
        if (atr <= 0.0).any():
            raise ValueError("entry_atr_bps must be strictly positive for M3")
        bundle = MetaTargetBundle(name, "regression", residual / atr, reconstruction={"multiply_by": columns.entry_atr_bps})
    elif name == "M5":
        bundle = MetaTargetBundle(name, "binary", (realised > 0.0).astype(float))
    elif name == "M6":
        bundle = MetaTargetBundle(name, "binary", (realised > parameters.robust_hurdle_bps).astype(float), reconstruction={"hurdle_bps": parameters.robust_hurdle_bps})
    elif name == "M7":
        bundle = MetaTargetBundle(name, "binary", _sigmoid((realised - parameters.soft_hurdle_bps) / parameters.soft_tau_bps), reconstruction={"hurdle_bps": parameters.soft_hurdle_bps, "tau_bps": parameters.soft_tau_bps})
    elif name == "M8":
        if statistics is None or len(statistics.ordinal_bin_means) != len(parameters.ordinal_edges_bps) + 1:
            raise ValueError("M8 needs frozen training-only ordinal bin means")
        ordinal = np.digitize(realised, parameters.ordinal_edges_bps, right=True).astype(float)
        bundle = MetaTargetBundle(name, "ordinal", ordinal, reconstruction={"edges_bps": parameters.ordinal_edges_bps, "bin_means_bps": statistics.ordinal_bin_means})
    elif name == "M9":
        failed = realised <= 0.0
        loss = np.maximum(-realised, 0.0)
        bundle = MetaTargetBundle(name, "multi_head", failed.astype(float),
            heads={"failure_probability": failed.astype(float), "failure_severity": loss},
            valid_masks={"failure_severity": failed}, reconstruction={"expected_failure_loss": "p_failure * conditional_loss"})
    elif name == "M10":
        success = realised > 0.0
        upside = np.maximum(realised, 0.0)
        bundle = MetaTargetBundle(name, "multi_head", success.astype(float),
            heads={"success_probability": success.astype(float), "success_upside": upside},
            valid_masks={"success_upside": success}, reconstruction={"expected_upside": "p_success * conditional_upside"})
    elif name == "M11":
        success, failed = realised > 0.0, realised <= 0.0
        bundle = MetaTargetBundle(name, "multi_head", realised,
            heads={"success_probability": success.astype(float), "success_upside": np.maximum(realised, 0.0),
                   "failure_probability": failed.astype(float), "failure_severity": np.maximum(-realised, 0.0)},
            valid_masks={"success_upside": success, "failure_severity": failed},
            reconstruction={"expected_net": "p_success * conditional_upside - p_failure * conditional_loss"})
    elif name == "M12":
        base = _numeric(frame, columns.causal_base_expected_net_bps)[0]
        bundle = MetaTargetBundle(name, "binary", (realised < base - parameters.overestimate_margin_bps).astype(float), reconstruction={"margin_bps": parameters.overestimate_margin_bps})
    elif name == "M13":
        bundle = MetaTargetBundle(name, "regression", np.exp(-np.abs(residual) / parameters.trust_tau_bps), reconstruction={"tau_bps": parameters.trust_tau_bps})
    elif name == "MC1":
        stability, base_stability, path, conflict = _numeric(frame, columns.label_stability, columns.base_target_stability, columns.path_completeness, columns.event_conflict)
        for label, value in ((columns.label_stability, stability), (columns.base_target_stability, base_stability), (columns.path_completeness, path)):
            if (value < 0.0).any() or (value > 1.0).any():
                raise ValueError(f"{label} must be in [0, 1]")
        if (conflict < 0.0).any() or (conflict > 1.0).any():
            raise ValueError("event_conflict must be in [0, 1]")
        certainty = np.power(np.clip(stability * base_stability * path * (1.0 - conflict), 0.0, 1.0), parameters.certainty_power)
        bundle = MetaTargetBundle(name, "regression", certainty * residual, reconstruction={"certainty": "label_stability * base_target_stability * path_completeness * (1-event_conflict)"})
    elif name == "MC2":
        if statistics is None or statistics.prior_clear_rate is None:
            raise ValueError("MC2 needs frozen training-only prior_clear_rate")
        prior = float(statistics.prior_clear_rate)
        if not 0.0 <= prior <= 1.0:
            raise ValueError("prior_clear_rate must be in [0, 1]")
        certainty = _numeric(frame, columns.label_certainty)[0]
        if (certainty < 0.0).any() or (certainty > 1.0).any():
            raise ValueError("label_certainty must be in [0, 1]")
        certainty = np.power(certainty, parameters.certainty_power)
        bundle = MetaTargetBundle(name, "binary", certainty * (realised > 0.0).astype(float) + (1.0 - certainty) * prior, reconstruction={"prior_clear_rate": prior, "certainty_power": parameters.certainty_power})
    elif name == "MC3":
        margin, entropy = _numeric(frame, columns.base_margin, columns.base_entropy)
        if (entropy < 0.0).any():
            raise ValueError("base_entropy must be non-negative")
        # A bounded confidence function: larger signed margin and lower entropy
        # retain more residual signal; it never changes the residual's sign.
        confidence = _sigmoid(parameters.margin_scale * margin - parameters.entropy_scale * entropy)
        bundle = MetaTargetBundle(name, "regression", residual * confidence, reconstruction={"confidence": "sigmoid(margin_scale*base_margin - entropy_scale*base_entropy)"})
    else:
        raise ValueError(f"unsupported meta target {name!r}")
    return _assert_finite_bundle(bundle)


def _normalise_weight(raw: np.ndarray, p: MetaWeightParameters) -> np.ndarray:
    if not 0.0 < p.min_weight <= p.max_weight:
        raise ValueError("require 0 < min_weight <= max_weight")
    raw = np.asarray(raw, dtype=float)
    if raw.ndim != 1 or not np.isfinite(raw).all() or (raw <= 0.0).any():
        raise ValueError("raw meta weights must be finite and strictly positive")
    weight = np.clip(raw, p.min_weight, p.max_weight)
    weight /= weight.mean()
    if not np.isfinite(weight).all() or (weight <= 0.0).any():
        raise AssertionError("meta weights must be finite and strictly positive")
    return weight


def build_meta_weight(
    frame: pd.DataFrame, name: MetaWeightName, *, columns: MetaColumns = MetaColumns(),
    parameters: MetaWeightParameters = MetaWeightParameters(),
    statistics: MetaTrainingStatistics | None = None, class_label: np.ndarray | None = None,
) -> np.ndarray:
    """Build MW0/MW1/MW3--MW7 using resolved labels only during fitting.

    The two requested formulas with a literal minus sign are made positive in
    the only compatible way: floor + scale × sigmoid(...).  MW5 therefore
    correctly *downweights* rows close to the net hurdle while never producing
    an invalid negative sample weight.
    """
    if parameters.high_base_tau <= 0.0 or parameters.boundary_tau_bps <= 0.0 or parameters.economic_cap_bps <= 0.0:
        raise ValueError("meta weight scales must be positive")
    if not 0.0 <= parameters.high_base_center <= 1.0 or parameters.high_base_floor <= 0.0 or parameters.boundary_floor <= 0.0:
        raise ValueError("invalid meta weight centre/floor")
    n = len(frame)
    if name in {"MW0", "MW2"}:
        # MW2's hard high-base selection is applied by the runner's
        # fold-local tail gate; its within-population weight is uniform.
        return np.ones(n, dtype=float)
    realised, base = _numeric(frame, columns.realised_net_bps, columns.causal_base_expected_net_bps)
    residual = realised - base
    if name == "MW1":
        percentile = _numeric(frame, columns.base_rank_percentile)[0]
        if (percentile < 0.0).any() or (percentile > 1.0).any():
            raise ValueError("base_rank_percentile must be in [0, 1]")
        raw = parameters.high_base_floor + parameters.high_base_scale * _sigmoid((percentile - parameters.high_base_center) / parameters.high_base_tau)
    elif name == "MW3":
        certainty = _numeric(frame, columns.label_certainty)[0]
        if (certainty < 0.0).any() or (certainty > 1.0).any():
            raise ValueError("label_certainty must be in [0, 1]")
        raw = 0.5 + 0.5 * certainty
    elif name == "MW4":
        if parameters.economic_lambda < 0.0:
            raise ValueError("economic_lambda must be non-negative")
        raw = 1.0 + parameters.economic_lambda * np.minimum(np.abs(residual), parameters.economic_cap_bps) / parameters.economic_cap_bps
    elif name == "MW5":
        distance = np.abs(realised - parameters.boundary_hurdle_bps)
        raw = parameters.boundary_floor + parameters.boundary_scale * _sigmoid(distance / parameters.boundary_tau_bps)
    elif name == "MW6":
        labels = np.asarray(class_label if class_label is not None else _numeric(frame, columns.class_label)[0], dtype=int)
        if len(labels) != n:
            raise ValueError("class_label must align with frame")
        if parameters.class_balance_variant == "natural":
            raw = np.ones(n, dtype=float)
        else:
            counts = statistics.class_counts if statistics and statistics.class_counts else {int(k): int(v) for k, v in zip(*np.unique(labels, return_counts=True))}
            by_row = np.asarray([counts.get(int(label), 0) for label in labels], dtype=float)
            if (by_row <= 0.0).any():
                raise ValueError("class-count statistics do not cover all labels")
            if parameters.class_balance_variant == "sqrt_inverse":
                raw = np.sqrt(n / (len(counts) * by_row))
            elif parameters.class_balance_variant == "effective":
                beta = (n - 1.0) / n
                all_counts = np.asarray(list(counts.values()), dtype=float)
                effective = (1.0 - beta ** all_counts) / max(1.0 - beta, 1e-12)
                mean_effective = effective.mean()
                raw = mean_effective / ((1.0 - beta ** by_row) / max(1.0 - beta, 1e-12))
            else:
                raise ValueError("unsupported MW6 class_balance_variant")
    elif name == "MW7":
        a, b, c = parameters.combined_exponents
        if min(a, b, c) < 0.0 or a + b + c <= 0.0 or parameters.mild_failure_lambda < 0.0:
            raise ValueError("invalid MW7 exponents or mild_failure_lambda")
        percentile, certainty = _numeric(frame, columns.base_rank_percentile, columns.label_certainty)
        if (percentile < 0.0).any() or (percentile > 1.0).any() or (certainty < 0.0).any() or (certainty > 1.0).any():
            raise ValueError("MW7 percentile/certainty must be in [0, 1]")
        relevance = parameters.high_base_floor + parameters.high_base_scale * _sigmoid((percentile - parameters.high_base_center) / parameters.high_base_tau)
        certainty_weight = .5 + .5 * certainty
        failure = np.maximum(-realised, 0.0)
        failure_weight = 1.0 + parameters.mild_failure_lambda * np.minimum(failure, parameters.economic_cap_bps) / parameters.economic_cap_bps
        raw = relevance ** a * certainty_weight ** b * failure_weight ** c
    elif name == "MW8":
        # Convex blend of independently meaningful winner candidates:
        # high-base relevance, label certainty, and economic-error magnitude.
        a, b, c = parameters.combined_exponents
        if min(a, b, c) < 0.0 or a + b + c <= 0.0 or parameters.economic_lambda < 0.0:
            raise ValueError("invalid MW8 blend coefficients")
        percentile, certainty = _numeric(frame, columns.base_rank_percentile, columns.label_certainty)
        if (percentile < 0.0).any() or (percentile > 1.0).any() or (certainty < 0.0).any() or (certainty > 1.0).any():
            raise ValueError("MW8 percentile/certainty must be in [0, 1]")
        relevance = parameters.high_base_floor + parameters.high_base_scale * _sigmoid((percentile - parameters.high_base_center) / parameters.high_base_tau)
        certainty_weight = .5 + .5 * certainty
        economic_weight = 1.0 + parameters.economic_lambda * np.minimum(np.abs(residual), parameters.economic_cap_bps) / parameters.economic_cap_bps
        raw = (a * relevance + b * certainty_weight + c * economic_weight) / (a + b + c)
    else:
        raise ValueError(f"unsupported meta weight {name!r}")
    return _normalise_weight(raw, parameters)


def meta_target_manifest(
    target: MetaTargetName, weight: MetaWeightName, *, columns: MetaColumns = MetaColumns(),
    target_parameters: MetaTargetParameters = MetaTargetParameters(), weight_parameters: MetaWeightParameters = MetaWeightParameters(),
) -> Mapping[str, object]:
    """Serializable lineage fragment for a per-side, OOF-meta run manifest."""
    return {
        "target": target, "weight": weight, "geometry": "TP6/SL4", "horizon": "H12",
        "columns": columns.__dict__, "target_parameters": target_parameters.__dict__,
        "weight_parameters": weight_parameters.__dict__,
        "causality": "base expected net must be same-side cross-fitted; target statistics fit on train rows only; resolved outcome fields never inference features",
        "ranking": "rank/calibrate per side before global allocation",
        "invariants": ["finite target heads", "positive mean-one sample weights"],
    }
