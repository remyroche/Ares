"""Validated TP6/SL4 H12 target and loss-weight construction.

This module intentionally contains no model fitting or feature selection.  It
turns a *resolved* panel sidecar into either a three-state simplex target or a
strictly positive per-row sample weight.  The sidecar fields are outcome/path
labels and must never be passed to an inference feature matrix.

Event convention is fixed: 0=upper first, 1=lower first, 2=H12 timeout.
Every ``B*`` target is a finite, non-negative row-simplex.  Every ``BW*``
weight is finite, strictly positive, clipped, and mean-normalised to one.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Mapping, Sequence

import numpy as np
import pandas as pd


TargetName = Literal["B0", "B1", "B2", "B3", "B5", "B6", "B7", "B8", "B9"]
WeightName = Literal["BW0", "BW1", "BW2", "BW3", "BW4", "BW5", "BW6", "BW8"]


@dataclass(frozen=True)
class TP6SL4Columns:
    """Resolved-sidecar column contract; override names rather than copying data."""

    event: str = "t2_tp6_sl4_event"
    exit_minute: str = "t2_tp6_sl4_exit_minute"
    mfe_atr: str = "t2_path_mfe_atr"
    mae_atr: str = "t2_path_mae_atr"
    terminal_atr: str = "t2_terminal_atr"
    gross_bps: str = "t4_tp6_sl4_gross_bps"
    net_bps: str = "t4_tp6_sl4_net_bps"
    upper_membership: str = "t2_tp6_sl4_upper_membership"
    lower_membership: str = "t2_tp6_sl4_lower_membership"
    timeout_membership: str = "t2_tp6_sl4_timeout_membership"
    label_certainty: str = "label_certainty"
    contract_consensus: str = "tp6_sl4_contract_consensus"
    effective_weight: str = "tp6_sl4_effective_weight"


@dataclass(frozen=True)
class TargetParameters:
    """Explicit numerical policy for all B targets.

    ``hard_floor`` retains the primary first-touch contract.  Values below
    0.50 are allowed but not recommended for base event training because they
    can let a terminal/path proxy overwrite the actual first barrier.
    """

    tp_atr: float = 6.0
    sl_atr: float = 4.0
    horizon_minutes: float = 720.0
    distance_tau_atr: float = 0.25
    first_touch_bonus: float = 2.0
    hard_floor: float = 0.75
    time_decay_hours: float = 4.0
    terminal_beta: float = 1.0
    terminal_clip_atr: float = 2.0
    peak_beta: float = 1.0
    peak_clip_atr: float = 3.0
    peak_excess_transform: Literal["clipped_linear", "sigmoid", "log1p"] = "log1p"
    adverse_beta: float = 1.0
    adverse_clip_atr: float = 3.0
    net_tau_bps: float = 100.0
    gross_tau_bps: float = 100.0
    economic_hurdle_bps: float = 0.0
    positive_hurdle_bps: float = 0.0
    adverse_hurdle_bps: float = 0.0
    economic_timeout_scale: float = 1.0


@dataclass(frozen=True)
class WeightParameters:
    """Bounded BW policies.  The output is always clipped then mean-one."""

    min_weight: float = 0.50
    max_weight: float = 2.00
    certainty_variant: Literal["half", "quarter"] = "half"
    class_support_variant: Literal["sqrt_inverse", "effective"] = "sqrt_inverse"
    side_balance_variant: Literal["natural", "mild"] = "natural"
    recency_half_life_days: float = 90.0
    mild_class_balance_power: float = 0.25
    time_decay_hours: float = 4.0
    economic_scale_bps: float = 100.0
    combined_exponents: tuple[float, float, float] = (1.0, 1.0, 1.0)


def _require(frame: pd.DataFrame, columns: Sequence[str]) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise KeyError(f"TP6/SL4 sidecar lacks required columns: {missing}")


def _finite(frame: pd.DataFrame, columns: Sequence[str]) -> np.ndarray:
    _require(frame, columns)
    values = frame.loc[:, columns].apply(pd.to_numeric, errors="raise").to_numpy(float)
    if not np.isfinite(values).all():
        raise ValueError(f"non-finite resolved target values in {list(columns)}")
    return values


def _softmax(logits: np.ndarray) -> np.ndarray:
    if logits.ndim != 2 or logits.shape[1] != 3:
        raise ValueError("three-class logits required")
    if not np.isfinite(logits).all():
        raise ValueError("target logits must be finite")
    z = logits - logits.max(axis=1, keepdims=True)
    weights = np.exp(np.clip(z, -60.0, 0.0))
    return weights / weights.sum(axis=1, keepdims=True)


def _hard(event: np.ndarray) -> np.ndarray:
    raw = np.asarray(event, dtype=float)
    rounded = raw.astype(int)
    if not np.isfinite(raw).all() or not np.allclose(raw, rounded, rtol=0.0, atol=0.0) or not np.isin(rounded, (0, 1, 2)).all():
        raise ValueError("event must use 0=upper, 1=lower, 2=timeout")
    target = np.zeros((len(rounded), 3), dtype=np.float64)
    target[np.arange(len(rounded)), rounded] = 1.0
    return target


def assert_simplex(target: np.ndarray, *, atol: float = 1e-7) -> None:
    """Fail closed rather than silently normalising an invalid target."""
    if target.ndim != 2 or target.shape[1] != 3:
        raise AssertionError("target must have shape (n_rows, 3)")
    if not np.isfinite(target).all() or (target < -atol).any():
        raise AssertionError("target must be finite and non-negative")
    if not np.allclose(target.sum(axis=1), 1.0, rtol=0.0, atol=atol):
        raise AssertionError("target rows must sum to one")


def assert_positive_weights(weight: np.ndarray) -> None:
    if weight.ndim != 1 or not np.isfinite(weight).all() or (weight <= 0.0).any():
        raise AssertionError("sample weights must be a finite strictly-positive vector")


def _blend_with_hard(hard: np.ndarray, soft: np.ndarray, floor: float) -> np.ndarray:
    if not 0.0 <= floor <= 1.0:
        raise ValueError("hard_floor must lie in [0, 1]")
    assert_simplex(soft)
    target = floor * hard + (1.0 - floor) * soft
    assert_simplex(target)
    return target


def _validate_target_parameters(p: TargetParameters) -> None:
    if p.tp_atr <= 0.0 or p.sl_atr <= 0.0 or p.horizon_minutes <= 0.0:
        raise ValueError("TP, SL, and H12 horizon parameters must be positive")


def _validate_weight_parameters(p: WeightParameters) -> None:
    if p.economic_scale_bps <= 0.0 or p.recency_half_life_days <= 0.0:
        raise ValueError("BW scale parameters must be positive")
    if not 0.0 < p.mild_class_balance_power <= 1.0:
        raise ValueError("mild_class_balance_power must lie in (0, 1]")


def _distance_soft(event: np.ndarray, mfe: np.ndarray, mae: np.ndarray, exit_minute: np.ndarray, p: TargetParameters) -> np.ndarray:
    if p.distance_tau_atr <= 0.0:
        raise ValueError("distance_tau_atr must be positive")
    upper = (mfe - p.tp_atr) / p.distance_tau_atr
    lower = (mae - p.sl_atr) / p.distance_tau_atr
    timeout = np.minimum((p.tp_atr - mfe) / p.distance_tau_atr, (p.sl_atr - mae) / p.distance_tau_atr)
    # The exact first touch disambiguates paths that later cross both levels.
    bonus = p.first_touch_bonus * (1.0 + 0.25 * (1.0 - np.minimum(exit_minute, p.horizon_minutes) / p.horizon_minutes))
    logits = np.column_stack((upper, lower, timeout))
    logits[np.arange(len(event)), event] += bonus
    return _softmax(logits)


def _event_logits(hard: np.ndarray, p: TargetParameters) -> np.ndarray:
    """Finite event-logit baseline, deliberately not log(one-hot)."""
    return p.first_touch_bonus * hard.copy()


def _transform_positive(value: np.ndarray, kind: str) -> np.ndarray:
    if kind == "clipped_linear":
        return value
    if kind == "sigmoid":
        return 1.0 / (1.0 + np.exp(-np.clip(value, -35., 35.)))
    if kind == "log1p":
        return np.log1p(value)
    raise ValueError(f"unsupported positive transform {kind!r}")


def build_target(frame: pd.DataFrame, name: TargetName, *, columns: TP6SL4Columns = TP6SL4Columns(), parameters: TargetParameters = TargetParameters()) -> np.ndarray:
    """Build B0/B1/B2/B3/B5/B6/B7/B8/B9 from resolved TP6/SL4 sidecar rows.

    B5--B9 always retain ``hard_floor`` first-touch mass.  This is essential:
    terminal, MFE, MAE and realised economics describe the completed path and
    must not redefine the event contract by themselves.
    """
    _validate_target_parameters(parameters)
    event = _finite(frame, [columns.event])[:, 0]
    hard = _hard(event)
    event = event.astype(int)
    if name == "B0":
        return hard
    if name == "B2":
        membership = _finite(frame, [columns.upper_membership, columns.lower_membership, columns.timeout_membership])
        if (membership < 0.0).any():
            raise ValueError("given memberships must be non-negative")
        sums = membership.sum(axis=1)
        if (sums <= 0.0).any():
            raise ValueError("given memberships must have positive row sums")
        target = membership / sums[:, None]
        assert_simplex(target)
        return target
    if name in ("B1", "B3"):
        mfe, mae, exit_minute = _finite(frame, [columns.mfe_atr, columns.mae_atr, columns.exit_minute]).T
        soft = _distance_soft(event, mfe, mae, exit_minute, parameters)
        if name == "B1":
            return soft
        if parameters.time_decay_hours <= 0.0:
            raise ValueError("time_decay_hours must be positive")
        elapsed_hours = np.minimum(exit_minute, parameters.horizon_minutes) / 60.0
        confidence = parameters.hard_floor + (1.0 - parameters.hard_floor) * np.exp(-elapsed_hours / parameters.time_decay_hours)
        # B3 is intentionally independent of B1: the exact winning state gets
        # its time-decayed confidence and the residual is evenly split between
        # the two alternatives, precisely as specified.
        target = np.full((len(event), 3), 0.0, dtype=float)
        target[:] = ((1.0 - confidence) / 2.0)[:, None]
        target[np.arange(len(event)), event] = confidence
        assert_simplex(target)
        return target
    if name == "B5":
        terminal = _finite(frame, [columns.terminal_atr])[:, 0]
        if parameters.terminal_clip_atr <= 0.0:
            raise ValueError("terminal_clip_atr must be positive")
        logits = _event_logits(hard, parameters)
        margin = terminal - parameters.tp_atr
        # Exact requested form: upper event logit minus beta times the
        # clipped side-normalised terminal margin.
        logits[:, 0] -= parameters.terminal_beta * np.clip(margin, -parameters.terminal_clip_atr, parameters.terminal_clip_atr)
        return _blend_with_hard(hard, _softmax(logits), parameters.hard_floor)
    if name == "B6":
        mfe = _finite(frame, [columns.mfe_atr])[:, 0]
        if parameters.peak_clip_atr <= 0.0:
            raise ValueError("peak_clip_atr must be positive")
        excess = np.clip(np.maximum(mfe - parameters.tp_atr, 0.0), 0.0, parameters.peak_clip_atr)
        logits = _event_logits(hard, parameters)
        logits[:, 0] += parameters.peak_beta * _transform_positive(excess, parameters.peak_excess_transform)
        return _blend_with_hard(hard, _softmax(logits), parameters.hard_floor)
    if name == "B7":
        mae = _finite(frame, [columns.mae_atr])[:, 0]
        if parameters.adverse_clip_atr <= 0.0:
            raise ValueError("adverse_clip_atr must be positive")
        severity = np.clip(np.maximum(mae - parameters.sl_atr, 0.0), 0.0, parameters.adverse_clip_atr)
        logits = _event_logits(hard, parameters)
        # More adverse severity increases lower-first confidence (never upper).
        logits[:, 1] += parameters.adverse_beta * severity
        return _blend_with_hard(hard, _softmax(logits), parameters.hard_floor)
    if name == "B8":
        gross = _finite(frame, [columns.gross_bps])[:, 0]
        if parameters.gross_tau_bps <= 0.0:
            raise ValueError("gross_tau_bps must be positive")
        upper = 1.0 / (1.0 + np.exp(-np.clip((gross - parameters.economic_hurdle_bps) / parameters.gross_tau_bps, -35., 35.)))
        lower = 1.0 / (1.0 + np.exp(-np.clip((-gross - parameters.economic_hurdle_bps) / parameters.gross_tau_bps, -35., 35.)))
        timeout = np.maximum(1.0 - upper - lower, 1e-8)
        soft = np.column_stack((upper, lower, timeout)); soft /= soft.sum(axis=1, keepdims=True)
        return _blend_with_hard(hard, soft, parameters.hard_floor)
    if name == "B9":
        gross = _finite(frame, [columns.gross_bps])[:, 0]
        if parameters.gross_tau_bps <= 0.0:
            raise ValueError("gross_tau_bps must be positive")
        positive = 1.0 / (1.0 + np.exp(-np.clip((gross - parameters.positive_hurdle_bps) / parameters.gross_tau_bps, -35., 35.)))
        negative = 1.0 / (1.0 + np.exp(-np.clip((-gross - parameters.adverse_hurdle_bps) / parameters.gross_tau_bps, -35., 35.)))
        mass = positive + negative
        # Preserve positive/negative relative mass; neutral receives the
        # remaining membership before final simplex normalisation.
        neutral = np.maximum(1.0 - mass, 1e-8)
        memberships = np.column_stack((positive, negative, neutral))
        soft = memberships / memberships.sum(axis=1, keepdims=True)
        return _blend_with_hard(hard, soft, parameters.hard_floor)
    raise ValueError(f"unsupported target {name!r}")


def _normalise_weight(raw: np.ndarray, p: WeightParameters) -> np.ndarray:
    if not 0.0 < p.min_weight <= p.max_weight:
        raise ValueError("weights require 0 < min_weight <= max_weight")
    if not np.isfinite(raw).all() or (raw <= 0.0).any():
        raise ValueError("raw sample weights must be finite and positive")
    clipped = np.clip(raw, p.min_weight, p.max_weight)
    weight = clipped / clipped.mean()
    assert_positive_weights(weight)
    return weight


def build_weight(frame: pd.DataFrame, name: WeightName, *, columns: TP6SL4Columns = TP6SL4Columns(), target: np.ndarray | None = None, target_parameters: TargetParameters = TargetParameters(), parameters: WeightParameters = WeightParameters()) -> np.ndarray:
    """Build the requested BW0--BW8 policies.

    The fields used here are resolved *training* labels only.  ``BW8`` is the
    requested certainty × mild-class-support × capped-recency product; it does
    not use realised MFE or PnL, which would make it a different policy.
    """
    _validate_target_parameters(target_parameters)
    _validate_weight_parameters(parameters)
    n = len(frame)
    if name == "BW0":
        return np.ones(n, dtype=float)
    event = _finite(frame, [columns.event])[:, 0]
    _hard(event)
    event = event.astype(int)
    if name == "BW1":
        if target is None:
            target = build_target(frame, "B1", columns=columns, parameters=target_parameters)
        assert_simplex(target)
        entropy = -(target * np.log(np.clip(target, 1e-12, 1.0))).sum(axis=1) / np.log(3.0)
        certainty = 1.0 - entropy
        raw = .5 + .5 * certainty if parameters.certainty_variant == "half" else .25 + .75 * certainty
    elif name == "BW2":
        exit_minute = _finite(frame, [columns.exit_minute])[:, 0]
        if parameters.time_decay_hours <= 0.0:
            raise ValueError("time_decay_hours must be positive")
        raw = .5 + .5 * np.exp(-np.minimum(exit_minute, target_parameters.horizon_minutes) / 60.0 / parameters.time_decay_hours)
    elif name == "BW3":
        consensus = _finite(frame, [columns.contract_consensus])[:, 0]
        if (consensus <= 0.0).any() or (consensus > 1.0).any():
            raise ValueError("BW3 consensus must be a strictly-positive fraction in (0, 1]")
        # Preserve the requested fraction exactly.  The central contract is in
        # the nine-way set, so it is already strictly positive; applying the
        # generic .50 floor would destroy its low-consensus information.
        weight = consensus / consensus.mean()
        assert_positive_weights(weight)
        return weight
    elif name == "BW4":
        gross = _finite(frame, [columns.gross_bps])[:, 0]
        if target_parameters.gross_tau_bps <= 0.0:
            raise ValueError("BW4 gross_tau_bps must be positive")
        # Positive bounded form of the requested economic-separation idea.
        # The literal ``min - scale * sigmoid`` is negative for ordinary
        # settings and cannot be a sample weight.
        separation = np.abs(gross - target_parameters.economic_hurdle_bps)
        raw = parameters.min_weight + (parameters.max_weight - parameters.min_weight) / (1. + np.exp(-separation / target_parameters.gross_tau_bps))
    elif name == "BW5":
        counts = np.bincount(event, minlength=3).astype(float)
        if parameters.class_support_variant == "sqrt_inverse":
            raw = np.sqrt(n / (3. * counts[event]))
        else:
            # Effective-number class weights (Cui et al.); beta is determined
            # from the observed training population rather than tuned on eval.
            beta = (n - 1.0) / n
            effective = (1.0 - np.power(beta, counts)) / max(1.0 - beta, 1e-12)
            raw = effective.mean() / effective[event]
    elif name == "BW6":
        if "side_name" not in frame: raise KeyError("BW6 requires side_name")
        if parameters.side_balance_variant == "natural":
            raw = np.ones(n, dtype=float)
        else:
            counts = frame.side_name.value_counts()
            raw = np.power(len(frame) / (len(counts) * frame.side_name.map(counts).to_numpy(float)), parameters.mild_class_balance_power)
    elif name == "BW8":
        if target is None:
            target = build_target(frame, "B1", columns=columns, parameters=target_parameters)
        assert_simplex(target)
        entropy = -(target * np.log(np.clip(target, 1e-12, 1.0))).sum(axis=1) / np.log(3.0)
        a, b, c = parameters.combined_exponents
        if min(a, b, c) < 0.0 or a + b + c <= 0.0:
            raise ValueError("combined BW8 exponents must be non-negative with positive total")
        certainty = .5 + .5 * (1.0 - entropy)
        counts = np.bincount(event, minlength=3).astype(float)
        class_support = np.power(n / (3. * counts[event]), parameters.mild_class_balance_power)
        if "__ts__" not in frame:
            raise KeyError("BW8 requires __ts__ for capped recency weighting")
        timestamp = pd.to_datetime(frame["__ts__"], utc=True)
        age_days = (timestamp.max() - timestamp).dt.total_seconds().to_numpy(float) / 86400.0
        recency = np.exp(-np.maximum(age_days, 0.0) / parameters.recency_half_life_days)
        # Capped so a remote row remains represented and a recent row cannot
        # dominate the loss solely because of timestamp.
        recency = np.clip(recency, parameters.min_weight, parameters.max_weight)
        raw = certainty**a * class_support**b * recency**c
    else:
        raise ValueError(f"unsupported weight {name!r}")
    return _normalise_weight(np.asarray(raw, dtype=float), parameters)


def target_manifest(name: TargetName, weight: WeightName, *, columns: TP6SL4Columns = TP6SL4Columns(), target_parameters: TargetParameters = TargetParameters(), weight_parameters: WeightParameters = WeightParameters()) -> Mapping[str, object]:
    """Serializable lineage fragment for callers' run manifests."""
    return {"target": name, "weight": weight, "geometry": "TP6/SL4", "horizon": "H12",
            "event_convention": "0=upper_first,1=lower_first,2=timeout",
            "columns": columns.__dict__, "target_parameters": target_parameters.__dict__,
            "weight_parameters": weight_parameters.__dict__,
            "causality": "sidecar/path/economic fields are resolved labels only; exclude them from inference features; fit rows require label_available_at before score boundary",
            "invariants": ["target finite nonnegative row-simplex", "weight finite positive clipped mean-one"]}
