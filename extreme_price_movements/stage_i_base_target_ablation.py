"""Leakage-safe Stage-I base-target ablation primitives.

The experiment intentionally separates three concerns:

* materialising exact H12 first-touch labels from an already-frozen, exact
  next-minute path pack;
* screening the complete 60-arm target grid with realised *oracle* economics;
* fitting either strict chronological OOF diagnostics or the fast single
  chronological development holdout, then mapping scores into common expected-
  net bps using only prior-resolved rows.

No feature is constructed from a realised path.  Path-derived values are
labels, training weights, or evaluation diagnostics only.  Invalid/incomplete
paths remain in the population audit but never enter supervised fitting.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
import json
import math
import os
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression


HORIZON_MINUTES = 12 * 60
COST_BPS = 100.0
TOP_FRACTIONS: tuple[float, ...] = (0.01, 0.05, 0.10)
ORDINAL_ALPHA: tuple[float, ...] = (0.25, 0.33, 0.50)
SL_GRID: tuple[int, ...] = (2, 3, 4)
TP_GRID: tuple[int, ...] = (3, 4, 5, 6, 7)
IDENTITY_COLUMNS: tuple[str, ...] = ("candidate_id", "__ts__", "__symbol__")
RANKING_POLICY = "pooled_global_common_bps_desc_then_decision_symbol_side_candidate"


class BaseTargetAblationError(ValueError):
    """Raised when target, chronology, or lineage contracts are violated."""


def _canonical_sha(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return sha256(encoded.encode("utf-8")).hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sigmoid(value: np.ndarray | float) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(np.asarray(value, dtype=np.float64), -60.0, 60.0)))


@dataclass(frozen=True, order=True)
class BarrierGeometry:
    k_sl: int
    k_tp: int
    horizon_minutes: int = HORIZON_MINUTES
    upper_floor_fraction: float = 0.015
    upper_cap_fraction: float = 0.040

    def __post_init__(self) -> None:
        if self.k_sl not in SL_GRID or self.k_tp not in TP_GRID:
            raise BaseTargetAblationError("geometry lies outside the preregistered grid")
        if self.horizon_minutes != HORIZON_MINUTES:
            raise BaseTargetAblationError("Stage-I target ablation is frozen to H12")
        if not 0.0 < self.upper_floor_fraction < self.upper_cap_fraction:
            raise BaseTargetAblationError("upper floor/cap are invalid")

    @property
    def key(self) -> str:
        return f"sl{self.k_sl}_tp{self.k_tp}"

    @property
    def promotion_eligible_geometry(self) -> bool:
        return self.k_sl < self.k_tp

    @property
    def disposition(self) -> str:
        return (
            "promotion_eligible_grid_geometry"
            if self.promotion_eligible_geometry
            else "diagnostic_non_promotable_grid_exception"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            **asdict(self), "key": self.key,
            "promotion_eligible_geometry": self.promotion_eligible_geometry,
            "disposition": self.disposition,
            "upper_formula": "clip(k_tp*ATR/entry, .015, .04)",
            "lower_formula": "k_sl*ATR/entry",
        }


def geometry_grid() -> tuple[BarrierGeometry, ...]:
    """Return all 15 audited geometries; only 12 may advance.

    The source specification independently requires the full 15/45/60 audit
    and ``k_sl < k_tp`` promotion.  Keeping all Cartesian cells reconciles
    those requirements without inventing a fifth target family.
    """

    result = tuple(BarrierGeometry(k_sl=sl, k_tp=tp) for sl in SL_GRID for tp in TP_GRID)
    if len(result) != 15 or sum(item.promotion_eligible_geometry for item in result) != 12:
        raise AssertionError("preregistered geometry inventory drift")
    return result


@dataclass(frozen=True)
class TargetArm:
    name: str
    family: str
    geometry: BarrierGeometry
    ordinal_alpha: float | None = None

    @property
    def promotion_eligible(self) -> bool:
        return self.geometry.promotion_eligible_geometry

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["promotion_eligible"] = self.promotion_eligible
        result["geometry_disposition"] = self.geometry.disposition
        return result


def target_arm_grid() -> tuple[TargetArm, ...]:
    arms: list[TargetArm] = []
    for geometry in geometry_grid():
        arms.append(TargetArm(f"S__{geometry.key}", "scalar_S", geometry))
        arms.extend(
            TargetArm(
                f"O_a{str(alpha).replace('.', 'p')}__{geometry.key}",
                "ordinal_O",
                geometry,
                ordinal_alpha=alpha,
            )
            for alpha in ORDINAL_ALPHA
        )
    if len(arms) != 60 or sum(item.promotion_eligible for item in arms) != 48:
        raise AssertionError("target-arm inventory must remain 60 audited / 48 promotable")
    return tuple(arms)


def target_column_for_arm(arm: TargetArm) -> str:
    """Return the compact per-geometry target column for an audited arm."""

    if arm.family == "scalar_S":
        return "S_target"
    if arm.family == "ordinal_O" and arm.ordinal_alpha in ORDINAL_ALPHA:
        return f"O_a{str(arm.ordinal_alpha).replace('.', 'p')}_target"
    raise BaseTargetAblationError(f"unsupported target arm {arm.name!r}")


@dataclass(frozen=True)
class PathLabels:
    valid: np.ndarray
    event: np.ndarray
    event_minute: np.ndarray
    favorable_progress: np.ndarray
    adverse_progress: np.ndarray
    dominance: np.ndarray
    gross_bps: np.ndarray
    net_bps: np.ndarray
    upper_fraction: np.ndarray
    lower_fraction: np.ndarray
    upper_floor_bound: np.ndarray
    upper_cap_bound: np.ndarray


@dataclass(frozen=True)
class H12PathPrimitivePack:
    """Target-neutral, side-normalised primitives computed once per path pack."""

    valid: np.ndarray
    atr_fraction: np.ndarray
    favorable: np.ndarray
    adverse: np.ndarray
    max_favorable: np.ndarray
    max_adverse: np.ndarray
    terminal_return: np.ndarray


@dataclass(frozen=True)
class H12GeometryTraversal:
    """First-touch cache for every distinct TP and SL in the frozen grid."""

    primitives: H12PathPrimitivePack
    upper_fraction: Mapping[int, np.ndarray]
    lower_fraction: Mapping[int, np.ndarray]
    upper_first: Mapping[int, np.ndarray]
    lower_first: Mapping[int, np.ndarray]


def materialize_h12_path_primitives(
    *, entry_price: np.ndarray, atr: np.ndarray, side_sign: np.ndarray,
    high: np.ndarray, low: np.ndarray, close: np.ndarray,
    path_complete: np.ndarray,
) -> H12PathPrimitivePack:
    """Perform expensive OHLC normalisation and side orientation exactly once."""

    entry = np.asarray(entry_price, dtype=np.float64)
    scale = np.asarray(atr, dtype=np.float64)
    sign = np.asarray(side_sign, dtype=np.int8)
    hi, lo, terminal = (np.asarray(value, dtype=np.float64) for value in (high, low, close))
    n = len(entry)
    complete_raw = np.asarray(path_complete)
    if complete_raw.shape != (n,):
        raise BaseTargetAblationError("path_complete must be row-aligned")
    if np.issubdtype(complete_raw.dtype, np.bool_):
        complete = complete_raw.astype(bool, copy=True)
    else:
        # NumPy casts NaN to True.  Parse explicitly so missing or malformed
        # completion flags fail closed instead of receiving future labels.
        parsed_complete = pd.to_numeric(
            pd.Series(complete_raw), errors="coerce"
        ).to_numpy(dtype=np.float64)
        complete = np.isfinite(parsed_complete) & (parsed_complete == 1.0)
    if hi.shape != (n, HORIZON_MINUTES) or lo.shape != hi.shape or terminal.shape != hi.shape:
        raise BaseTargetAblationError("path matrices must be rows x 720")
    if sign.shape != (n,) or not np.isin(sign, (-1, 1)).all():
        raise BaseTargetAblationError("side_sign must contain only -1/+1")
    finite_scalar = np.isfinite(entry) & np.isfinite(scale) & (entry > 0.0) & (scale > 0.0)
    finite_path = np.isfinite(hi).all(axis=1) & np.isfinite(lo).all(axis=1) & np.isfinite(terminal).all(axis=1)
    valid = complete & finite_scalar & finite_path
    long = sign > 0
    favorable = np.where(long[:, None], hi / entry[:, None] - 1.0, 1.0 - lo / entry[:, None])
    adverse = np.where(long[:, None], 1.0 - lo / entry[:, None], hi / entry[:, None] - 1.0)
    return H12PathPrimitivePack(
        valid=valid,
        atr_fraction=scale / entry,
        favorable=favorable,
        adverse=adverse,
        max_favorable=np.maximum(np.max(favorable, axis=1), 0.0),
        max_adverse=np.maximum(np.max(adverse, axis=1), 0.0),
        terminal_return=sign * (terminal[:, -1] / entry - 1.0),
    )


def materialize_h12_geometry_traversal(
    primitives: H12PathPrimitivePack,
    *, geometries: Sequence[BarrierGeometry] = geometry_grid(),
) -> H12GeometryTraversal:
    """Compute the eight distinct first touches shared by all 15 contracts."""

    selected = tuple(geometries)
    if not selected:
        raise BaseTargetAblationError("at least one geometry is required")
    sentinel = HORIZON_MINUTES
    upper_fraction: dict[int, np.ndarray] = {}
    lower_fraction: dict[int, np.ndarray] = {}
    upper_first: dict[int, np.ndarray] = {}
    lower_first: dict[int, np.ndarray] = {}
    for k_tp in sorted({item.k_tp for item in selected}):
        threshold = np.clip(
            k_tp * primitives.atr_fraction,
            selected[0].upper_floor_fraction,
            selected[0].upper_cap_fraction,
        )
        touch = primitives.favorable >= threshold[:, None]
        upper_fraction[k_tp] = threshold
        upper_first[k_tp] = np.where(touch.any(axis=1), touch.argmax(axis=1), sentinel).astype(np.int16)
    for k_sl in sorted({item.k_sl for item in selected}):
        threshold = k_sl * primitives.atr_fraction
        touch = primitives.adverse >= threshold[:, None]
        lower_fraction[k_sl] = threshold
        lower_first[k_sl] = np.where(touch.any(axis=1), touch.argmax(axis=1), sentinel).astype(np.int16)
    return H12GeometryTraversal(
        primitives=primitives, upper_fraction=upper_fraction,
        lower_fraction=lower_fraction, upper_first=upper_first, lower_first=lower_first,
    )


def materialize_geometry_labels_from_traversal(
    traversal: H12GeometryTraversal, geometry: BarrierGeometry,
) -> PathLabels:
    """Derive one target geometry without revisiting raw OHLC paths."""

    primitive = traversal.primitives
    upper = np.asarray(traversal.upper_fraction[geometry.k_tp], dtype=np.float64).copy()
    lower = np.asarray(traversal.lower_fraction[geometry.k_sl], dtype=np.float64).copy()
    upper_first = np.asarray(traversal.upper_first[geometry.k_tp], dtype=np.int16)
    lower_first = np.asarray(traversal.lower_first[geometry.k_sl], dtype=np.int16)
    sentinel = HORIZON_MINUTES
    # A same-minute *real* touch is adverse by contract.  The sentinel is not
    # a minute, however: equality at the sentinel means neither barrier was
    # touched and must remain a timeout.
    neither_touched = (lower_first == sentinel) & (upper_first == sentinel)
    adverse_first = (lower_first <= upper_first) & ~neither_touched
    event = np.where(
        neither_touched,
        1,
        np.where(adverse_first, 0, 2),
    ).astype(np.int8)
    event_minute = np.minimum(upper_first, lower_first).astype(np.int16)
    event_minute[event_minute == sentinel] = -1
    favorable_progress = np.clip(primitive.max_favorable / np.maximum(upper, 1e-12), 0.0, 1.0)
    adverse_progress = np.clip(primitive.max_adverse / np.maximum(lower, 1e-12), 0.0, 1.0)
    dominance = favorable_progress - adverse_progress
    gross_fraction = np.select(
        [event == 0, event == 2], [-lower, upper], default=primitive.terminal_return,
    )
    gross_bps = gross_fraction * 10_000.0
    net_bps = gross_bps - COST_BPS
    valid = primitive.valid
    for values in (favorable_progress, adverse_progress, dominance, gross_bps, net_bps, upper, lower):
        values[~valid] = np.nan
    event[~valid] = -1
    event_minute[~valid] = -1
    valid_event = event[valid]
    valid_minute = event_minute[valid]
    if (
        not np.isin(valid_event, (0, 1, 2)).all()
        or not np.array_equal(valid_event == 1, valid_minute == -1)
        or np.any((valid_minute[valid_event != 1] < 0) | (valid_minute[valid_event != 1] >= sentinel))
        or not np.allclose(net_bps[valid], gross_bps[valid] - COST_BPS, rtol=0.0, atol=1e-5)
    ):
        raise BaseTargetAblationError("materialized event/minute/economics invariants failed")
    unclipped_upper = geometry.k_tp * primitive.atr_fraction
    return PathLabels(
        valid=valid.copy(), event=event, event_minute=event_minute,
        favorable_progress=favorable_progress.astype(np.float32),
        adverse_progress=adverse_progress.astype(np.float32),
        dominance=dominance.astype(np.float32), gross_bps=gross_bps.astype(np.float32),
        net_bps=net_bps.astype(np.float32), upper_fraction=upper.astype(np.float32),
        lower_fraction=lower.astype(np.float32),
        upper_floor_bound=(unclipped_upper <= geometry.upper_floor_fraction),
        upper_cap_bound=(unclipped_upper >= geometry.upper_cap_fraction),
    )


def validate_entry_timing(
    signal_timestamp: Sequence[Any], decision_timestamp: Sequence[Any], entry_timestamp: Sequence[Any]
) -> None:
    """Enforce the frozen exact one-minute open at signal timestamp +1h."""

    signal = pd.to_datetime(pd.Series(signal_timestamp), utc=True, errors="raise")
    decision = pd.to_datetime(pd.Series(decision_timestamp), utc=True, errors="raise")
    entry = pd.to_datetime(pd.Series(entry_timestamp), utc=True, errors="raise")
    if not decision.eq(signal + pd.Timedelta(hours=1)).all():
        raise BaseTargetAblationError("decision must be the frozen signal close +1h")
    if not entry.eq(decision).all():
        raise BaseTargetAblationError("entry must be the exact minute-resolution open at signal +1h")


def materialize_geometry_labels(
    *,
    entry_price: np.ndarray,
    atr: np.ndarray,
    side_sign: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    path_complete: np.ndarray,
    geometry: BarrierGeometry,
) -> PathLabels:
    """Vectorised first-touch labels over exact post-entry minute paths.

    Arrays are ``rows x 720`` and start at the frozen exact minute-resolution
    open at signal timestamp +1h.  A bar touching both barriers is adverse, including a
    tie on the first touch minute.  Cost never changes a physical barrier; it
    is applied once only when producing the economic ``net_bps`` diagnostic.
    """

    primitives = materialize_h12_path_primitives(
        entry_price=entry_price, atr=atr, side_sign=side_sign, high=high, low=low,
        close=close, path_complete=path_complete,
    )
    traversal = materialize_h12_geometry_traversal(primitives, geometries=(geometry,))
    return materialize_geometry_labels_from_traversal(traversal, geometry)


def scalar_s_target(event: np.ndarray, dominance: np.ndarray) -> np.ndarray:
    """S target: lower=0, upper=1, soft timeout in [0.35, 0.65]."""

    state = np.asarray(event, dtype=np.int8)
    dom = np.asarray(dominance, dtype=np.float64)
    target = np.where(state == 0, 0.0, np.where(state == 2, 1.0, 0.35 + 0.30 * _sigmoid(dom / 0.20)))
    target[state < 0] = np.nan
    return target.astype(np.float32)


def ordinal_o_target(event: np.ndarray, dominance: np.ndarray, alpha: float) -> np.ndarray:
    """Five ordered states: adverse, weak-, neutral, weak+, favourable.

    Only unresolved paths use the dominance cut.  Barrier events anchor the
    two extremes, which keeps class meaning invariant across geometry cells.
    """

    if float(alpha) not in ORDINAL_ALPHA:
        raise BaseTargetAblationError("ordinal alpha is not preregistered")
    state = np.asarray(event, dtype=np.int8)
    dom = np.asarray(dominance, dtype=np.float64)
    timeout_class = np.where(dom < -alpha, 1, np.where(dom > alpha, 3, 2))
    target = np.where(state == 0, 0, np.where(state == 2, 4, timeout_class)).astype(np.int8)
    target[state < 0] = -1
    return target


def cumulative_ordinal_targets(classes: np.ndarray) -> np.ndarray:
    labels = np.asarray(classes, dtype=np.int8)
    if not np.isin(labels, (0, 1, 2, 3, 4)).all():
        raise BaseTargetAblationError("ordinal fitting classes must be in 0..4")
    return (labels[:, None] > np.arange(4, dtype=np.int8)[None, :]).astype(np.int8)


def recover_ordinal_simplex(cumulative_probability: np.ndarray) -> np.ndarray:
    """Project four cumulative P(Y>j) outputs to a monotone five-simplex."""

    raw = np.asarray(cumulative_probability, dtype=np.float64)
    if raw.ndim != 2 or raw.shape[1] != 4 or not np.isfinite(raw).all():
        raise BaseTargetAblationError("ordinal cumulative probabilities must be finite Nx4")
    # Minimum-violation projection for the required descending survival curve.
    survival = np.minimum.accumulate(np.clip(raw, 0.0, 1.0), axis=1)
    boundary = np.column_stack([np.ones(len(raw)), survival, np.zeros(len(raw))])
    simplex = boundary[:, :-1] - boundary[:, 1:]
    simplex = np.clip(simplex, 0.0, 1.0)
    simplex /= simplex.sum(axis=1, keepdims=True)
    return simplex.astype(np.float32)


def ordinal_score(simplex: np.ndarray) -> np.ndarray:
    probability = np.asarray(simplex, dtype=np.float64)
    if probability.ndim != 2 or probability.shape[1] != 5:
        raise BaseTargetAblationError("ordinal simplex must be Nx5")
    return (probability @ (np.arange(5, dtype=np.float64) / 4.0)).astype(np.float32)


def _require_family_target_support(
    values: Any, *, family: str, context: str
) -> None:
    target = pd.to_numeric(pd.Series(np.asarray(values)), errors="coerce").to_numpy(
        dtype=np.float64
    )
    target = target[np.isfinite(target)]
    unique = np.unique(target)
    if family == "scalar_S" and not np.any((target > 0.0) & (target < 1.0)):
        raise BaseTargetAblationError(
            f"{context}: scalar_S lacks soft timeout support"
        )
    if family == "ordinal_O" and (
        len(unique) <= 2 or not np.isin(unique, (1.0, 2.0, 3.0)).any()
    ):
        raise BaseTargetAblationError(
            f"{context}: ordinal_O is a degenerate hard first-touch target"
        )


@dataclass(frozen=True)
class Round1Gates:
    """Preregistered qualitative-gate thresholds; never estimated from results."""

    min_upper_support_rows: int = 100
    max_timeout_prevalence: float = 0.90
    min_worst_regime_upper_rate: float = 0.005
    min_oracle_top10_net_bps: float = 0.0

    def __post_init__(self) -> None:
        if self.min_upper_support_rows < 1:
            raise BaseTargetAblationError("min_upper_support_rows must be positive")
        if not 0.0 < self.max_timeout_prevalence < 1.0:
            raise BaseTargetAblationError("max_timeout_prevalence must be in (0,1)")
        if not 0.0 <= self.min_worst_regime_upper_rate < 1.0:
            raise BaseTargetAblationError("min_worst_regime_upper_rate must be in [0,1)")


def _stable_top(frame: pd.DataFrame, score_column: str, fraction: float) -> pd.DataFrame:
    work = frame.copy()
    work["__score__"] = pd.to_numeric(work[score_column], errors="coerce")
    work = work.loc[np.isfinite(work["__score__"])].copy()
    keys = ["__score__", "decision_ts", "__symbol__", "side_name", "candidate_id"]
    ascending = [False, True, True, True, True]
    work = work.sort_values(keys, ascending=ascending, kind="mergesort")
    count = min(len(work), max(1, int(math.ceil(len(work) * fraction))))
    return work.head(count)


def round1_screen(
    labels: pd.DataFrame,
    *,
    arms: Sequence[TargetArm] = target_arm_grid(),
    gates: Round1Gates,
    regime_column: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate all 60 label/oracle arms without fitting a model."""

    required = {
        *IDENTITY_COLUMNS, "decision_ts", "side_name", regime_column,
        "geometry", "event", "gross_bps", "net_bps", "target_valid",
        "upper_floor_bound", "upper_cap_bound",
    }
    if missing := sorted(required.difference(labels.columns)):
        raise BaseTargetAblationError(f"Round 1 labels lack {missing}")
    rows: list[dict[str, Any]] = []
    gate_rows: list[dict[str, Any]] = []
    for arm in arms:
        local = labels.loc[
            labels.geometry.eq(arm.geometry.key) & labels.target_valid.astype(bool)
        ].copy()
        target_column = target_column_for_arm(arm)
        if target_column not in local:
            raise BaseTargetAblationError(f"Round 1 lacks materialised target {target_column}")
        local["oracle_score"] = pd.to_numeric(local[target_column], errors="coerce")
        local = local.loc[np.isfinite(local.oracle_score)].copy()
        if local.empty:
            raise BaseTargetAblationError(f"{arm.name} has no valid target rows")
        upper_support = int(local.event.eq(2).sum())
        timeout_rate = float(local.event.eq(1).mean())
        target_values = local.oracle_score.to_numpy(dtype=np.float64)
        target_unique = np.unique(target_values)
        scalar_soft_rows = int(
            np.sum((target_values > 0.0) & (target_values < 1.0))
        )
        ordinal_interior_classes = int(
            np.intersect1d(target_unique, np.asarray([1.0, 2.0, 3.0])).size
        )
        if arm.family == "scalar_S":
            family_support_ok = scalar_soft_rows > 0
            family_support_reason = "scalar_soft_timeout_support"
        elif arm.family == "ordinal_O":
            family_support_ok = len(target_unique) > 2 and ordinal_interior_classes > 0
            family_support_reason = "ordinal_interior_class_support"
        else:
            family_support_ok = True
            family_support_reason = "control_family_not_applicable"
        supported_regime = local.loc[
            ~local[regime_column].astype(str).eq("causal_unknown")
        ]
        regime_upper = supported_regime.groupby(regime_column, observed=True).event.apply(
            lambda x: float(x.eq(2).mean())
        )
        worst_regime_upper = float(regime_upper.min()) if len(regime_upper) else 0.0
        unknown_regime_rows = int(
            local[regime_column].astype(str).eq("causal_unknown").sum()
        )
        arm_metrics: dict[float, tuple[float, float, int]] = {}
        for fraction in TOP_FRACTIONS:
            selected = _stable_top(local, "oracle_score", fraction)
            gross = float(selected.gross_bps.mean())
            net = float(selected.net_bps.mean())
            arm_metrics[fraction] = (gross, net, len(selected))
            rows.append({
                "arm": arm.name, "family": arm.family, "geometry": arm.geometry.key,
                "ordinal_alpha": arm.ordinal_alpha, "top_fraction": fraction,
                "rows": int(len(local)), "selected_rows": int(len(selected)),
                "gross_bps_per_trade": gross, "net_bps_per_trade": net,
                "upper_support_rows": upper_support, "upper_prevalence": float(local.event.eq(2).mean()),
                "timeout_prevalence": timeout_rate,
                "target_entropy": _discrete_entropy(local.oracle_score.to_numpy()),
                "target_effective_support": _effective_support(local.oracle_score.to_numpy()),
                "target_unique_values": int(len(target_unique)),
                "scalar_soft_timeout_rows": scalar_soft_rows,
                "ordinal_interior_classes": ordinal_interior_classes,
                "family_support_ok": bool(family_support_ok),
                "family_support_reason": family_support_reason,
                "worst_regime_upper_rate": worst_regime_upper,
                "unknown_causal_regime_rows": unknown_regime_rows,
                "supported_regime_count": int(len(regime_upper)),
                "upper_floor_binding_rate": float(local.upper_floor_bound.mean()),
                "upper_cap_binding_rate": float(local.upper_cap_bound.mean()),
                "promotion_eligible_geometry": arm.promotion_eligible,
                "geometry_disposition": arm.geometry.disposition,
            })
        failures: list[str] = []
        if upper_support < gates.min_upper_support_rows:
            failures.append("negligible_upper_support")
        if timeout_rate > gates.max_timeout_prevalence:
            failures.append("extreme_timeout_prevalence")
        if worst_regime_upper < gates.min_worst_regime_upper_rate:
            failures.append("poor_regime_stability")
        if arm_metrics[0.10][1] < gates.min_oracle_top10_net_bps:
            failures.append("weak_oracle_economics")
        if not family_support_ok:
            failures.append(
                "degenerate_hard_first_touch_scalar"
                if arm.family == "scalar_S"
                else "degenerate_hard_first_touch_ordinal"
            )
        if not arm.promotion_eligible:
            failures.append("diagnostic_non_promotable_grid_exception")
        gate_rows.append({
            "arm": arm.name, "promotion_eligible": not failures,
            "rejection_reasons": "|".join(failures), **asdict(gates),
        })
    metrics = pd.DataFrame(rows)
    gate_frame = pd.DataFrame(gate_rows)
    if metrics.arm.nunique() != 60 or len(gate_frame) != 60:
        raise AssertionError("Round 1 did not preserve the 60-arm inventory")
    return metrics, gate_frame


def _discrete_entropy(values: np.ndarray) -> float:
    value = np.asarray(values)
    value = value[np.isfinite(value)]
    if not len(value):
        return float("nan")
    _, count = np.unique(value, return_counts=True)
    p = count / count.sum()
    return float(-(p * np.log(np.maximum(p, 1e-12))).sum())


def _effective_support(values: np.ndarray) -> float:
    value = np.asarray(values)
    value = value[np.isfinite(value)]
    if not len(value):
        return 0.0
    _, count = np.unique(value, return_counts=True)
    p = count / count.sum()
    return float(1.0 / np.square(p).sum())


def robust_top10_lift_score(pooled_top10_lift: float, era_top10_lifts: Sequence[float]) -> float:
    """Exact preregistered robust score from the target-ablation specification."""

    eras = np.asarray(era_top10_lifts, dtype=np.float64)
    eras = eras[np.isfinite(eras)]
    if not np.isfinite(pooled_top10_lift) or not len(eras):
        return float("nan")
    median = float(np.median(eras))
    mad = float(np.median(np.abs(eras - median)))
    worst = float(np.min(eras))
    return float(0.5 * pooled_top10_lift + 0.5 * median - 0.5 * mad - max(0.0, -worst))


def fit_causal_common_bps_map(
    reference_score: np.ndarray,
    reference_net_bps: np.ndarray,
    current_score: np.ndarray,
    *,
    bins: int = 20,
    min_rows: int = 100,
) -> np.ndarray:
    """Prior-resolved equal-frequency bin means followed by isotonic mapping."""

    score = np.asarray(reference_score, dtype=np.float64)
    net = np.asarray(reference_net_bps, dtype=np.float64)
    current = np.asarray(current_score, dtype=np.float64)
    valid = np.isfinite(score) & np.isfinite(net)
    score, net = score[valid], net[valid]
    if len(score) < min_rows or np.unique(score).size < 4:
        return np.full(len(current), np.nan, dtype=np.float64)
    order = np.argsort(score, kind="mergesort")
    group = np.minimum(np.arange(len(order)) * bins // len(order), bins - 1)
    x, y, weight = [], [], []
    for item in range(bins):
        pos = order[group == item]
        if not len(pos):
            continue
        x.append(float(np.median(score[pos])))
        y.append(float(np.mean(net[pos])))
        weight.append(int(len(pos)))
    x_array = np.asarray(x)
    if len(x_array) < 4 or np.unique(x_array).size < 2:
        return np.full(len(current), np.nan, dtype=np.float64)
    model = IsotonicRegression(increasing=True, out_of_bounds="clip")
    model.fit(x_array, np.asarray(y), sample_weight=np.asarray(weight))
    result = np.full(len(current), np.nan, dtype=np.float64)
    finite = np.isfinite(current)
    result[finite] = model.predict(current[finite])
    return result


def causal_map_oof_scores(
    frame: pd.DataFrame,
    *,
    score_column: str,
    fold_column: str,
    decision_column: str = "decision_ts",
    available_column: str = "label_available_ts",
    net_column: str = "net_bps",
    min_rows: int = 100,
) -> pd.DataFrame:
    """Map each side/fold score with only labels resolved before fold start."""

    required = {"side_name", score_column, fold_column, decision_column, available_column, net_column}
    if missing := sorted(required.difference(frame.columns)):
        raise BaseTargetAblationError(f"causal mapping lacks {missing}")
    work = frame.copy()
    work[decision_column] = pd.to_datetime(work[decision_column], utc=True, errors="raise")
    work[available_column] = pd.to_datetime(work[available_column], utc=True, errors="raise")
    work["expected_net_bps"] = np.nan
    for fold_id, held in work.groupby(fold_column, sort=True):
        fold_start = held[decision_column].min()
        for side, current in held.groupby("side_name", sort=True):
            reference = work.loc[
                work.side_name.eq(side) & work[available_column].lt(fold_start)
            ]
            mapped = fit_causal_common_bps_map(
                reference[score_column].to_numpy(), reference[net_column].to_numpy(),
                current[score_column].to_numpy(), min_rows=min_rows,
            )
            work.loc[current.index, "expected_net_bps"] = mapped
            # The half-open prior-resolved condition is deliberately asserted
            # after fitting, so a future refactor cannot weaken it silently.
            if len(reference) and not reference[available_column].lt(fold_start).all():
                raise AssertionError("causal map consumed unresolved/future labels")
    return work


def pooled_global_tail_metrics(
    mapped: pd.DataFrame,
    *,
    score_column: str = "expected_net_bps",
    net_column: str = "net_bps",
    fractions: Sequence[float] = TOP_FRACTIONS,
) -> pd.DataFrame:
    """Rank once across both sides after common-bps mapping."""

    required = {*IDENTITY_COLUMNS, "decision_ts", "side_name", score_column, net_column}
    if missing := sorted(required.difference(mapped.columns)):
        raise BaseTargetAblationError(f"global ranking lacks {missing}")
    rows: list[dict[str, Any]] = []
    eligible = mapped.loc[np.isfinite(pd.to_numeric(mapped[score_column], errors="coerce"))].copy()
    for fraction in fractions:
        selected = _stable_top(eligible, score_column, float(fraction))
        rows.append({
            "top_fraction": float(fraction), "eligible_rows": int(len(eligible)),
            "selected_rows": int(len(selected)),
            "net_bps_per_trade": float(selected[net_column].mean()),
            "long_rows": int(selected.side_name.eq("long").sum()),
            "short_rows": int(selected.side_name.eq("short").sum()),
            "ranking_policy": RANKING_POLICY,
        })
    return pd.DataFrame(rows)


def require_selected_feature_contract(
    *, selector_dir: Path, base_selection_dir: Path
) -> dict[str, Any]:
    """Load the completed uncapped per-side selection and bind exact bytes."""

    selector_manifest_path = selector_dir / "manifest.json"
    selector_contract_path = selector_dir / "selector_feature_contract.json"
    feature_matrix_path = selector_dir / "selector_features.parquet"
    ledger_path = selector_dir / "selector_ledger.parquet"
    if not all(path.is_file() for path in (selector_manifest_path, selector_contract_path, feature_matrix_path, ledger_path)):
        raise BaseTargetAblationError("selector artifact is incomplete")
    selector = json.loads(selector_manifest_path.read_text(encoding="utf-8"))
    if selector.get("status") != "complete":
        raise BaseTargetAblationError("selector manifest is not complete")
    # An uncapped source is mandatory; legacy capped selectors cannot satisfy
    # the new target experiment even if a side manifest happens to point at it.
    selector_contract = json.loads(selector_contract_path.read_text(encoding="utf-8"))
    selector_integrity = selector.get("artifact_integrity")
    if (
        not isinstance(selector_integrity, dict)
        or selector_integrity.get("schema") != "stage_i_selector_artifact_integrity_v1"
        or selector_integrity.get("selector_ledger_sha256") != file_sha256(ledger_path)
        or selector_integrity.get("selector_features_sha256") != file_sha256(feature_matrix_path)
    ):
        raise BaseTargetAblationError("selector ledger/features fail artifact-integrity validation")
    source_cap = selector_contract.get("max_feature_columns")
    if source_cap not in (0, None):
        raise BaseTargetAblationError("base-target ablation requires the final uncapped selector")
    sides: dict[str, Any] = {}
    declared_selector_features = set(map(str, selector_contract.get("feature_columns", ())))
    if not declared_selector_features:
        raise BaseTargetAblationError("selector feature contract is empty")
    forbidden_inference_fields = {
        "r3_class", "r3_metric_target", "exact_gross_bps", "exact_net_bps",
        "label_available_ts", "label_valid", "target_invalid", "path_complete",
        "t2_tp6_sl4_event", "robust_clear_event_b25", "robust_clear_soft_b25_t50",
        "event", "event_minute", "gross_bps", "net_bps", "target_valid",
        "contract_certainty", "dominance", "favorable_progress", "adverse_progress",
    }
    selector_sha = file_sha256(selector_manifest_path)
    for side in ("long", "short"):
        path = base_selection_dir / side / "manifest.json"
        if not path.is_file():
            raise BaseTargetAblationError(f"{side}: selected-feature manifest is absent")
        manifest = json.loads(path.read_text(encoding="utf-8"))
        selected = tuple(map(str, manifest.get("selected_features", ())))
        base_input_contract = set(map(str, manifest.get("input_feature_contract", ())))
        if (
            manifest.get("schema") != "stage_i_base_feature_selection_v1"
            or manifest.get("status") != "complete"
            or str(manifest.get("side", "")).lower() != side
            or manifest.get("selector_sample_manifest_sha256") != selector_sha
            or not selected
            or manifest.get("selected_feature_contract") != list(selected)
            or not base_input_contract
            or not set(selected).issubset(base_input_contract)
            or not set(selected).issubset(declared_selector_features)
            or bool(set(selected).intersection(forbidden_inference_fields))
            or manifest.get("selector_artifact_integrity") != selector.get("artifact_integrity")
            or not isinstance(manifest.get("best_params"), dict)
        ):
            raise BaseTargetAblationError(f"{side}: selected-feature lineage is not final/hash-bound")
        sides[side] = {
            "manifest_path": str(path.resolve()), "manifest_sha256": file_sha256(path),
            "selected_features": list(selected),
            "selected_features_sha256": _canonical_sha(list(selected)),
            "fixed_params": manifest["best_params"],
            "fixed_params_sha256": _canonical_sha(manifest["best_params"]),
        }
    payload = {
        "schema": "stage_i_base_target_selected_feature_input_v1",
        "selector_manifest_path": str(selector_manifest_path.resolve()),
        "selector_manifest_sha256": selector_sha,
        "selector_feature_contract_path": str(selector_contract_path.resolve()),
        "selector_feature_contract_sha256": file_sha256(selector_contract_path),
        "selector_max_feature_columns": source_cap,
        "selector_features_sha256": file_sha256(feature_matrix_path),
        "selector_ledger_sha256": file_sha256(ledger_path),
        "sides": sides,
        "inference_inputs": "only selected per-side base features; no realised path/target fields",
    }
    payload["contract_sha256"] = _canonical_sha(payload)
    return payload


def verify_completed_manifest(root: Path, request_sha256: str) -> dict[str, Any] | None:
    """Fail-closed checkpoint/resume validation for published experiment cells."""

    path = root / "manifest.json"
    if not path.is_file():
        return None
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("status") != "complete" or manifest.get("request_sha256") != request_sha256:
        raise BaseTargetAblationError("completed base-target request drift")
    inventory = manifest.get("artifact_sha256")
    if not isinstance(inventory, dict) or not inventory:
        raise BaseTargetAblationError("completed base-target manifest lacks artifact inventory")
    for relative, expected in inventory.items():
        artifact = root / relative
        if not artifact.is_file() or file_sha256(artifact) != expected:
            raise BaseTargetAblationError(f"completed base-target artifact drift: {relative}")
    return manifest


def chronological_fold_vector(
    decision_timestamp: Sequence[Any],
    label_available_timestamp: Sequence[Any],
    *,
    folds: int,
    min_train_rows: int,
) -> np.ndarray:
    """Create expanding, whole-timestamp, purged H12 OOF folds.

    Rows before the first feasible held-out timestamp receive ``-1`` and form
    burn-in only.  A fit for fold ``j`` may use only rows whose label is
    available strictly before that fold's first decision timestamp.
    """

    if folds < 1 or min_train_rows < 3:
        raise BaseTargetAblationError("fold count/minimum training support are invalid")
    decision = pd.to_datetime(pd.Series(decision_timestamp), utc=True, errors="raise")
    available = pd.to_datetime(pd.Series(label_available_timestamp), utc=True, errors="raise")
    decision_ns = decision.astype("int64").to_numpy(dtype=np.int64)
    available_ns = available.astype("int64").to_numpy(dtype=np.int64)
    unique = np.unique(decision_ns)
    first = next(
        (index for index, timestamp in enumerate(unique) if int((available_ns < timestamp).sum()) >= min_train_rows),
        None,
    )
    if first is None or len(unique) - first < folds:
        raise BaseTargetAblationError("insufficient strict chronological OOF support")
    blocks = np.array_split(unique[first:], folds)
    output = np.full(len(decision), -1, dtype=np.int16)
    for fold_id, block in enumerate(blocks):
        if len(block):
            output[np.isin(decision_ns, block)] = fold_id
    return output


@dataclass(frozen=True)
class DevelopmentHoldoutSplit:
    """One preregistered large train/evaluation split for fast target selection."""

    train_mask: np.ndarray
    evaluation_mask: np.ndarray
    evaluation_start: pd.Timestamp
    evaluation_fraction: float
    purged_pre_evaluation_rows: int


def development_holdout_split(
    decision_timestamp: Sequence[Any],
    label_available_timestamp: Sequence[Any],
    *,
    evaluation_fraction: float = 0.25,
    min_train_rows: int = 500,
) -> DevelopmentHoldoutSplit:
    """Split whole timestamps and purge every unresolved pre-evaluation label."""

    if not 0.10 <= float(evaluation_fraction) <= 0.50:
        raise BaseTargetAblationError("development evaluation fraction must be in [0.10,0.50]")
    decision = pd.to_datetime(pd.Series(decision_timestamp), utc=True, errors="raise")
    available = pd.to_datetime(pd.Series(label_available_timestamp), utc=True, errors="raise")
    unique = np.sort(decision.unique())
    if len(unique) < 4:
        raise BaseTargetAblationError("development holdout needs at least four decision timestamps")
    evaluation_count = max(1, int(math.ceil(len(unique) * float(evaluation_fraction))))
    evaluation_start = pd.Timestamp(unique[-evaluation_count])
    before = decision.lt(evaluation_start).to_numpy()
    resolved = available.lt(evaluation_start).to_numpy()
    train_mask = before & resolved
    evaluation_mask = decision.ge(evaluation_start).to_numpy()
    if int(train_mask.sum()) < int(min_train_rows) or not evaluation_mask.any():
        raise BaseTargetAblationError("single development holdout lacks train/evaluation support")
    if not available.loc[train_mask].lt(evaluation_start).all():
        raise AssertionError("development training consumed unresolved labels")
    return DevelopmentHoldoutSplit(
        train_mask=train_mask,
        evaluation_mask=evaluation_mask,
        evaluation_start=evaluation_start,
        evaluation_fraction=float(evaluation_fraction),
        purged_pre_evaluation_rows=int((before & ~resolved).sum()),
    )


@dataclass
class _PreparedSideDevelopmentData:
    side: str
    feature_columns: tuple[str, ...]
    train_positions: np.ndarray
    evaluation_positions: np.ndarray
    x_train: np.ndarray
    x_evaluation: np.ndarray
    train_dataset: Any
    dataset_params: dict[str, Any]


class DevelopmentModelCache:
    """Immutable X/split/bin cache shared by every target arm in the funnel."""

    def __init__(
        self,
        frame: pd.DataFrame,
        *,
        selected_features: Mapping[str, Sequence[str]],
        fixed_params: Mapping[str, Mapping[str, Any]],
        evaluation_fraction: float,
        min_train_rows: int,
        seed: int,
    ) -> None:
        from lightgbm import Dataset

        self.split = development_holdout_split(
            frame.decision_ts, frame.label_available_ts,
            evaluation_fraction=evaluation_fraction, min_train_rows=min_train_rows,
        )
        contract_columns = [*IDENTITY_COLUMNS, "side_name", "decision_ts", "label_available_ts"]
        self._row_contract = pd.util.hash_pandas_object(
            frame.loc[:, contract_columns].astype(str), index=False,
        ).to_numpy(np.uint64)
        self.rows = len(frame)
        self.sides: dict[str, _PreparedSideDevelopmentData] = {}
        for side in ("long", "short"):
            columns = tuple(map(str, selected_features[side]))
            if missing := sorted(set(columns).difference(frame.columns)):
                raise BaseTargetAblationError(f"{side}: selected features absent from development frame: {missing[:8]}")
            side_mask = frame.side_name.eq(side).to_numpy()
            train_positions = np.flatnonzero(side_mask & self.split.train_mask)
            evaluation_positions = np.flatnonzero(side_mask & self.split.evaluation_mask)
            if len(train_positions) < min_train_rows or not len(evaluation_positions):
                raise BaseTargetAblationError(f"{side}: development split lacks required support")
            matrix = frame.loc[:, list(columns)].apply(pd.to_numeric, errors="coerce").to_numpy(np.float32)
            x_train = np.ascontiguousarray(matrix[train_positions])
            x_evaluation = np.ascontiguousarray(matrix[evaluation_positions])
            dataset_params = _fixed_params(fixed_params[side], seed=seed, objective="regression_l1")
            dataset_params.pop("n_estimators", None)
            dataset_params.pop("importance_type", None)
            train_dataset = Dataset(
                x_train, feature_name=list(columns), params=dataset_params, free_raw_data=False,
            )
            train_dataset.construct()
            self.sides[side] = _PreparedSideDevelopmentData(
                side=side, feature_columns=columns,
                train_positions=train_positions, evaluation_positions=evaluation_positions,
                x_train=x_train, x_evaluation=x_evaluation, train_dataset=train_dataset,
                dataset_params=dataset_params,
            )

    def require_compatible(self, frame: pd.DataFrame) -> None:
        contract_columns = [*IDENTITY_COLUMNS, "side_name", "decision_ts", "label_available_ts"]
        current = pd.util.hash_pandas_object(
            frame.loc[:, contract_columns].astype(str), index=False,
        ).to_numpy(np.uint64)
        if len(frame) != self.rows or not np.array_equal(current, self._row_contract):
            raise BaseTargetAblationError("target arm is not row-identical to the shared development cache")


def _core_lgbm_params(
    fixed_params: Mapping[str, Any], *, seed: int, objective: str,
) -> tuple[dict[str, Any], int]:
    params = _fixed_params(fixed_params, seed=seed, objective=objective)
    rounds = int(params.pop("n_estimators", 100))
    params.pop("importance_type", None)
    return params, rounds


def _fit_cached_development_model(
    prepared: _PreparedSideDevelopmentData,
    *, target: np.ndarray, weight: np.ndarray, family: str,
    fixed_params: Mapping[str, Any], seed: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Fit using preconstructed LightGBM bins and score train/evaluation rows."""

    import lightgbm as lgb

    def target_dataset(local_target: np.ndarray) -> Any:
        dataset = lgb.Dataset(
            prepared.x_train, label=local_target, weight=weight,
            feature_name=list(prepared.feature_columns),
            reference=prepared.train_dataset, params=prepared.dataset_params,
            free_raw_data=True,
        )
        dataset.construct()
        return dataset

    if family == "scalar_S":
        params, rounds = _core_lgbm_params(fixed_params, seed=seed, objective="regression_l1")
        model = lgb.train(params, target_dataset(target.astype(np.float32)), num_boost_round=rounds)
        train_score = np.clip(model.predict(prepared.x_train), 0.0, 1.0)
        eval_score = np.clip(model.predict(prepared.x_evaluation), 0.0, 1.0)
        return train_score.astype(np.float32), eval_score.astype(np.float32), {"models": 1}
    if family == "R3_control":
        params, rounds = _core_lgbm_params(fixed_params, seed=seed, objective="multiclass")
        params["num_class"] = 3
        model = lgb.train(params, target_dataset(target.astype(np.int8)), num_boost_round=rounds)
        train_probability = np.asarray(model.predict(prepared.x_train), dtype=np.float32)
        eval_probability = np.asarray(model.predict(prepared.x_evaluation), dtype=np.float32)
        if train_probability.shape[1] != 3 or eval_probability.shape[1] != 3:
            raise BaseTargetAblationError("R3 control did not emit a three-state simplex")
        return (
            (train_probability[:, 2] - train_probability[:, 0]).astype(np.float32),
            (eval_probability[:, 2] - eval_probability[:, 0]).astype(np.float32),
            {"models": 1},
        )
    if family != "ordinal_O":
        raise BaseTargetAblationError(f"unknown target family {family!r}")
    cumulative = cumulative_ordinal_targets(target.astype(np.int8))
    boundary_datasets: dict[int, Any] = {}
    for boundary in range(4):
        local = cumulative[:, boundary]
        if np.unique(local).size < 2:
            continue
        boundary_datasets[boundary] = target_dataset(local)

    def fit_boundary(boundary: int) -> tuple[int, np.ndarray, np.ndarray]:
        local = cumulative[:, boundary]
        if np.unique(local).size < 2:
            return (
                boundary,
                np.full(len(target), float(local[0]), dtype=np.float32),
                np.full(len(prepared.x_evaluation), float(local[0]), dtype=np.float32),
            )
        dataset = boundary_datasets[boundary]
        params, rounds = _core_lgbm_params(fixed_params, seed=seed + boundary, objective="binary")
        model = lgb.train(params, dataset, num_boost_round=rounds)
        return (
            boundary,
            np.asarray(model.predict(prepared.x_train), dtype=np.float32),
            np.asarray(model.predict(prepared.x_evaluation), dtype=np.float32),
        )

    # Four independent cumulative heads share immutable bins. Outer bounded
    # target-cell processes set this to one to avoid 3x4 oversubscription; the
    # sequential path retains the proven four-head batch.
    ordinal_workers = int(os.environ.get("STAGE_I_ORDINAL_WORKERS", "4"))
    ordinal_workers = min(max(ordinal_workers, 1), 4)
    with ThreadPoolExecutor(max_workers=ordinal_workers, thread_name_prefix="stage_i_ordinal") as pool:
        results = list(pool.map(fit_boundary, range(4)))
    train_probability = np.empty((len(target), 4), dtype=np.float32)
    eval_probability = np.empty((len(prepared.x_evaluation), 4), dtype=np.float32)
    for boundary, train_local, eval_local in results:
        train_probability[:, boundary] = train_local
        eval_probability[:, boundary] = eval_local
    return (
        ordinal_score(recover_ordinal_simplex(train_probability)),
        ordinal_score(recover_ordinal_simplex(eval_probability)),
        {"models": 4, "ordinal_batch_workers": ordinal_workers, "shared_lightgbm_bins": True},
    )


def training_weights(
    frame: pd.DataFrame,
    *,
    target: np.ndarray,
    mode: str,
    regime_column: str,
) -> np.ndarray:
    """Fit target-derived weights on a permitted training frame only."""

    if mode not in {"uniform", "contract_certainty", "hybrid"}:
        raise BaseTargetAblationError("unknown final training-weight arm")
    weight = np.ones(len(frame), dtype=np.float64)
    if mode in {"contract_certainty", "hybrid"}:
        if "contract_certainty" not in frame:
            raise BaseTargetAblationError("certainty weight requires training-only contract certainty")
        certainty = pd.to_numeric(frame.contract_certainty, errors="coerce").to_numpy(float)
        if not np.isfinite(certainty).all() or ((certainty < 0.0) | (certainty > 1.0)).any():
            raise BaseTargetAblationError("contract certainty is invalid")
        weight *= 0.5 + 0.5 * certainty
    if mode == "hybrid":
        # Mild recency: oldest=.75, newest=1.25.  It is based solely on causal
        # decision timestamps and is fitted independently inside every fold.
        timestamp = pd.to_datetime(frame.decision_ts, utc=True, errors="raise").astype("int64").to_numpy(float)
        span = max(float(timestamp.max() - timestamp.min()), 1.0)
        weight *= 0.75 + 0.50 * ((timestamp - timestamp.min()) / span)
        if regime_column not in frame:
            raise BaseTargetAblationError("hybrid weight requires preregistered causal environment")
        regime = frame[regime_column].astype(str)
        support = regime.value_counts()
        environment = regime.map(lambda value: math.sqrt(len(frame) / max(float(support[value]), 1.0))).to_numpy(float)
        environment /= max(float(environment.mean()), 1e-12)
        weight *= np.clip(environment, 0.75, 1.50)
        raw_target = np.asarray(target)
        if np.unique(raw_target).size > 20:
            # Continuous S labels are balanced by fold-train quintile, never
            # by each nearly-unique floating value.
            labels = pd.qcut(
                pd.Series(raw_target).rank(method="first"), 5, labels=False,
                duplicates="drop",
            ).astype(str)
        else:
            labels = pd.Series(raw_target.astype(str))
        class_support = labels.value_counts()
        balance = labels.map(lambda value: math.sqrt(len(labels) / max(float(class_support[value]), 1.0))).to_numpy(float)
        balance /= max(float(balance.mean()), 1e-12)
        weight *= np.clip(balance, 0.75, 1.50)
    weight = np.clip(weight, 0.25, 4.0)
    weight /= max(float(weight.mean()), 1e-12)
    return weight.astype(np.float32)


def _fixed_params(params: Mapping[str, Any], *, seed: int, objective: str) -> dict[str, Any]:
    result = dict(params)
    for key in (
        "objective", "num_class", "class_weight", "metric", "eval_metric",
        "random_state", "seed", "bagging_seed", "feature_fraction_seed",
        "num_threads", "num_thread", "nthread", "n_jobs",
    ):
        result.pop(key, None)
    result.update({"objective": objective, "random_state": int(seed), "verbosity": -1, "n_jobs": 1})
    return result


def fit_predict_target_arm(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    *,
    features: Sequence[str],
    target_column: str,
    family: str,
    fixed_params: Mapping[str, Any],
    seed: int,
    weight_mode: str,
    regime_column: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Fit S, O, or frozen-R3 using one fixed side-local feature contract."""

    from lightgbm import LGBMClassifier, LGBMRegressor

    columns = tuple(map(str, features))
    if missing := sorted(set(columns).difference(train.columns) | set(columns).difference(valid.columns)):
        raise BaseTargetAblationError(f"selected inference features are absent: {missing[:8]}")
    x_train = train.loc[:, list(columns)].apply(pd.to_numeric, errors="coerce")
    x_valid = valid.loc[:, list(columns)].apply(pd.to_numeric, errors="coerce")
    if x_train.shape[1] == 0:
        raise BaseTargetAblationError("selected feature contract is empty")
    target = pd.to_numeric(train[target_column], errors="coerce").to_numpy()
    if not np.isfinite(target).all():
        raise BaseTargetAblationError("invalid targets reached model fitting")
    weight = training_weights(train, target=target, mode=weight_mode, regime_column=regime_column)
    audit: dict[str, Any] = {
        "family": family, "seed": int(seed), "weight_mode": weight_mode,
        "train_rows": int(len(train)), "valid_rows": int(len(valid)),
        "features": list(columns), "weight_min": float(weight.min()),
        "weight_max": float(weight.max()), "weight_mean": float(weight.mean()),
    }
    if family == "scalar_S":
        model = LGBMRegressor(**_fixed_params(fixed_params, seed=seed, objective="regression_l1"))
        model.fit(x_train, target.astype(np.float32), sample_weight=weight)
        return np.clip(model.predict(x_valid), 0.0, 1.0).astype(np.float32), audit
    if family == "ordinal_O":
        classes = target.astype(np.int8)
        cumulative = cumulative_ordinal_targets(classes)
        probabilities = np.empty((len(valid), 4), dtype=np.float32)
        for boundary in range(4):
            local = cumulative[:, boundary]
            if np.unique(local).size < 2:
                probabilities[:, boundary] = float(local[0])
                continue
            model = LGBMClassifier(**_fixed_params(fixed_params, seed=seed + boundary, objective="binary"))
            model.fit(x_train, local, sample_weight=weight)
            probabilities[:, boundary] = model.predict_proba(x_valid)[:, 1]
        simplex = recover_ordinal_simplex(probabilities)
        audit["ordinal_probability_columns"] = [f"P(Y>{item})" for item in range(4)]
        return ordinal_score(simplex), audit
    if family == "R3_control":
        classes = target.astype(np.int8)
        params = _fixed_params(fixed_params, seed=seed, objective="multiclass")
        params["num_class"] = 3
        model = LGBMClassifier(**params)
        model.fit(x_train, classes, sample_weight=weight)
        probability = np.asarray(model.predict_proba(x_valid), dtype=np.float32)
        if probability.shape != (len(valid), 3):
            raise BaseTargetAblationError("R3 control did not emit a three-state simplex")
        return (probability[:, 2] - probability[:, 0]).astype(np.float32), audit
    raise BaseTargetAblationError(f"unknown target family {family!r}")


def run_strict_oof_arm(
    frame: pd.DataFrame,
    *,
    arm: TargetArm | None,
    target_column: str,
    family: str,
    selected_features: Mapping[str, Sequence[str]],
    fixed_params: Mapping[str, Mapping[str, Any]],
    folds: int,
    seeds: Sequence[int],
    min_train_rows: int,
    weight_mode: str,
    regime_column: str,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Generate side-local strict OOF scores for one arm; never same-fit scores."""

    work = frame.copy().reset_index(drop=True)
    required = {
        *IDENTITY_COLUMNS, "side_name", "decision_ts", "label_available_ts",
        "net_bps", target_column, regime_column,
    }
    if missing := sorted(required.difference(work.columns)):
        raise BaseTargetAblationError(f"OOF arm input lacks {missing}")
    valid_target = np.isfinite(pd.to_numeric(work[target_column], errors="coerce"))
    work = work.loc[valid_target].reset_index(drop=True)
    work["fold_id"] = chronological_fold_vector(
        work.decision_ts, work.label_available_ts, folds=folds, min_train_rows=min_train_rows,
    )
    predictions: list[pd.DataFrame] = []
    audits: list[dict[str, Any]] = []
    for seed in seeds:
        scored = work.loc[work.fold_id.ge(0), [
            *IDENTITY_COLUMNS, "side_name", "decision_ts", "label_available_ts",
            "net_bps", regime_column, "fold_id",
        ]].copy()
        scored["seed"] = int(seed)
        scored["raw_score"] = np.nan
        for side in ("long", "short"):
            side_features = tuple(selected_features[side])
            for fold_id in range(folds):
                valid_index = work.index[work.side_name.eq(side) & work.fold_id.eq(fold_id)]
                if not len(valid_index):
                    continue
                fold_start = pd.to_datetime(work.loc[valid_index, "decision_ts"], utc=True).min()
                train_index = work.index[
                    work.side_name.eq(side)
                    & pd.to_datetime(work.label_available_ts, utc=True).lt(fold_start)
                ]
                if len(train_index) < min_train_rows:
                    raise BaseTargetAblationError(f"{side}/fold{fold_id} lacks strict prior-resolved support")
                train, valid = work.loc[train_index], work.loc[valid_index]
                _require_family_target_support(
                    train[target_column],
                    family=family,
                    context=f"{side}/fold{fold_id}/train",
                )
                _require_family_target_support(
                    valid[target_column],
                    family=family,
                    context=f"{side}/fold{fold_id}/validation",
                )
                score, audit = fit_predict_target_arm(
                    train, valid, features=side_features, target_column=target_column,
                    family=family, fixed_params=fixed_params[side], seed=int(seed),
                    weight_mode=weight_mode, regime_column=regime_column,
                )
                keys = pd.MultiIndex.from_frame(valid.loc[:, list(IDENTITY_COLUMNS)])
                destination = pd.MultiIndex.from_frame(scored.loc[:, list(IDENTITY_COLUMNS)])
                positions = destination.get_indexer(keys)
                if (positions < 0).any():
                    raise AssertionError("OOF identity handoff drift")
                scored.iloc[positions, scored.columns.get_loc("raw_score")] = score
                audits.append({
                    **audit, "side": side, "fold_id": int(fold_id),
                    "fold_start": fold_start.isoformat(),
                    "train_label_available_max": pd.to_datetime(train.label_available_ts, utc=True).max().isoformat(),
                    "strict_prior_resolved": bool(pd.to_datetime(train.label_available_ts, utc=True).lt(fold_start).all()),
                })
        if scored.raw_score.isna().any():
            raise BaseTargetAblationError("strict OOF arm left scored rows without predictions")
        scored = causal_map_oof_scores(
            scored, score_column="raw_score", fold_column="fold_id",
            decision_column="decision_ts", available_column="label_available_ts",
            net_column="net_bps", min_rows=min_train_rows,
        )
        predictions.append(scored)
    result = pd.concat(predictions, ignore_index=True)
    result["arm"] = "R3_frozen_control" if arm is None else arm.name
    result["family"] = family
    result["weight_mode"] = weight_mode
    return result, audits


def run_development_holdout_arm(
    frame: pd.DataFrame,
    *,
    arm: TargetArm | None,
    target_column: str,
    family: str,
    selected_features: Mapping[str, Sequence[str]],
    fixed_params: Mapping[str, Mapping[str, Any]],
    seed: int,
    min_train_rows: int,
    weight_mode: str,
    regime_column: str,
    evaluation_fraction: float,
    model_cache: DevelopmentModelCache | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, Any]], DevelopmentModelCache]:
    """Fit once on a large prior-resolved window and score one later holdout.

    This is deliberately a fast *development-selection* result.  It is not
    walk-forward OOF evidence and cannot replace the later target-specific MDA
    and frozen 2024--2026 validation required by the roadmap.
    """

    work = frame.copy().reset_index(drop=True)
    required = {
        *IDENTITY_COLUMNS, "side_name", "decision_ts", "label_available_ts",
        "net_bps", target_column, regime_column,
    }
    if missing := sorted(required.difference(work.columns)):
        raise BaseTargetAblationError(f"development arm input lacks {missing}")
    valid_target = np.isfinite(pd.to_numeric(work[target_column], errors="coerce"))
    work = work.loc[valid_target].reset_index(drop=True)
    if model_cache is None:
        model_cache = DevelopmentModelCache(
            work, selected_features=selected_features, fixed_params=fixed_params,
            evaluation_fraction=evaluation_fraction, min_train_rows=min_train_rows,
            seed=seed,
        )
    else:
        model_cache.require_compatible(work)
        if model_cache.split.evaluation_fraction != float(evaluation_fraction):
            raise BaseTargetAblationError("development split fraction drift")
    evaluation_rows: list[pd.DataFrame] = []
    reference_rows: list[pd.DataFrame] = []
    audits: list[dict[str, Any]] = []
    for side in ("long", "short"):
        prepared = model_cache.sides[side]
        train = work.iloc[prepared.train_positions].copy()
        held = work.iloc[prepared.evaluation_positions].copy()
        target = pd.to_numeric(train[target_column], errors="coerce").to_numpy()
        if not np.isfinite(target).all():
            raise BaseTargetAblationError("invalid target reached development fitting")
        weight = training_weights(
            train, target=target, mode=weight_mode, regime_column=regime_column,
        )
        train_score, evaluation_score, model_audit = _fit_cached_development_model(
            prepared, target=target, weight=weight, family=family,
            fixed_params=fixed_params[side], seed=seed,
        )
        mapped_train = fit_causal_common_bps_map(
            train_score, train.net_bps.to_numpy(float), train_score, min_rows=min_train_rows,
        )
        mapped_evaluation = fit_causal_common_bps_map(
            train_score, train.net_bps.to_numpy(float), evaluation_score, min_rows=min_train_rows,
        )
        base_columns = [
            *IDENTITY_COLUMNS, "side_name", "decision_ts", "label_available_ts",
            "net_bps", regime_column,
        ]
        reference = train.loc[:, base_columns].copy()
        reference["seed"] = int(seed)
        reference["fold_id"] = 0
        reference["split_role"] = "prior_resolved_training_reference"
        reference["raw_score"] = train_score
        reference["expected_net_bps"] = mapped_train
        current = held.loc[:, base_columns].copy()
        current["seed"] = int(seed)
        current["fold_id"] = 0
        current["split_role"] = "held_out_development_evaluation"
        current["raw_score"] = evaluation_score
        current["expected_net_bps"] = mapped_evaluation
        reference_rows.append(reference)
        evaluation_rows.append(current)
        audits.append({
            "side": side, "seed": int(seed), "weight_mode": weight_mode,
            "train_rows": int(len(train)), "evaluation_rows": int(len(held)),
            "evaluation_start": model_cache.split.evaluation_start.isoformat(),
            "train_decision_min": pd.to_datetime(train.decision_ts, utc=True).min().isoformat(),
            "train_decision_max": pd.to_datetime(train.decision_ts, utc=True).max().isoformat(),
            "train_label_available_max": pd.to_datetime(train.label_available_ts, utc=True).max().isoformat(),
            "evaluation_decision_min": pd.to_datetime(held.decision_ts, utc=True).min().isoformat(),
            "strict_prior_resolved": bool(
                pd.to_datetime(train.label_available_ts, utc=True).lt(model_cache.split.evaluation_start).all()
            ),
            "purged_pre_evaluation_rows_global": model_cache.split.purged_pre_evaluation_rows,
            "features": list(prepared.feature_columns),
            "shared_feature_matrix": True, "shared_lightgbm_bins": True,
            **model_audit,
        })
    prediction = pd.concat(evaluation_rows, ignore_index=True)
    reference = pd.concat(reference_rows, ignore_index=True)
    arm_name = "R3_frozen_control" if arm is None else arm.name
    for output in (prediction, reference):
        output["arm"] = arm_name
        output["family"] = family
        output["weight_mode"] = weight_mode
    return prediction, reference, audits, model_cache


__all__ = [
    "BaseTargetAblationError", "BarrierGeometry", "TargetArm", "Round1Gates", "PathLabels",
    "geometry_grid", "target_arm_grid", "validate_entry_timing", "materialize_geometry_labels",
    "target_column_for_arm",
    "scalar_s_target", "ordinal_o_target", "cumulative_ordinal_targets",
    "recover_ordinal_simplex", "ordinal_score", "round1_screen", "robust_top10_lift_score",
    "fit_causal_common_bps_map", "causal_map_oof_scores", "pooled_global_tail_metrics",
    "require_selected_feature_contract", "verify_completed_manifest", "file_sha256",
    "chronological_fold_vector", "training_weights", "fit_predict_target_arm", "run_strict_oof_arm",
    "H12PathPrimitivePack", "H12GeometryTraversal", "materialize_h12_path_primitives",
    "materialize_h12_geometry_traversal", "materialize_geometry_labels_from_traversal",
    "DevelopmentHoldoutSplit", "development_holdout_split", "DevelopmentModelCache",
    "run_development_holdout_arm",
]
