"""Leakage-safe target variants for base-model target research.

The production S59 labels already expose a net executable return in
``__first_touch_capture_net__`` / ``__y_ret__``.  This module deliberately
does *not* subtract transaction costs from those columns again.  It provides a
small, explicit target space for chronological ablations:

* the existing first-touch soft label;
* a raw net-return soft label;
* a volatility-normalized net-return soft label; and
* a bounded blend of the raw and normalized labels; and
* side-relative soft economic labels fit on training rows and frozen for the
  corresponding OOS fold.

Volatility sensitivity belongs primarily in training weights, not in the
economic target.  The weight reference is fit on training rows only and can be
serialized and reused for the corresponding OOS fold.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping

import numpy as np
import pandas as pd


NET_RETURN_CANDIDATES = (
    "__first_touch_capture_net__",
    "__u_policy_net__",
    "__y_ret__",
)
VOLATILITY_CANDIDATES = (
    "__barrier_pct__",
    "__sl__",
    "__tp__",
)


@dataclass(frozen=True)
class TargetDesignSpec:
    """One deliberately small target-design arm.

    ``cost_mode`` is a guardrail rather than an optimization knob.  The
    canonical materialized labels use ``already_net``.  A gross input can be
    made explicit for a future data source, but that source must provide a
    documented cost column or explicit ``round_trip_cost``.
    """

    name: str
    target_kind: str
    raw_temperature: float = 0.020
    vol_temperature: float = 1.00
    margin: float = 0.0
    dual_raw_weight: float = 0.50
    vol_clip: float = 8.0
    cost_mode: str = "already_net"
    round_trip_cost: float = 0.01
    weight_mode: str = "timestamp_balanced"
    vol_weight_power: float = 0.50
    vol_weight_lower: float = 0.50
    vol_weight_upper: float = 2.00
    side_temperature: float = 1.0
    side_ecdf_knots: int = 129

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


DEFAULT_TARGET_SPECS: tuple[TargetDesignSpec, ...] = (
    TargetDesignSpec(
        name="current_first_touch_soft",
        target_kind="existing_first_touch",
    ),
    TargetDesignSpec(
        name="raw_net_soft",
        target_kind="raw_net",
    ),
    TargetDesignSpec(
        name="raw_net_vol_weighted",
        target_kind="raw_net",
        weight_mode="timestamp_balanced_vol_damped",
    ),
    TargetDesignSpec(
        name="vol_norm_net_soft",
        target_kind="vol_norm_net",
    ),
    TargetDesignSpec(
        name="dual_raw_vol_soft",
        target_kind="dual_raw_vol",
        dual_raw_weight=0.50,
    ),
    TargetDesignSpec(
        name="side_robust_net_soft",
        target_kind="side_robust_net",
        side_temperature=1.0,
    ),
    TargetDesignSpec(
        name="side_net_ecdf_soft",
        target_kind="side_net_ecdf",
        side_ecdf_knots=129,
    ),
)


def _numeric_column(frame: pd.DataFrame, candidates: tuple[str, ...]) -> tuple[pd.Series, str]:
    for column in candidates:
        if column in frame.columns:
            values = pd.to_numeric(frame[column], errors="coerce")
            if values.notna().any():
                return values.astype(np.float64), column
    raise KeyError(f"None of the required columns are present: {list(candidates)}")


def _sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(values, -60.0, 60.0)))


def _side_values(frame: pd.DataFrame) -> np.ndarray:
    """Resolve canonical side labels without assuming one upstream schema."""

    if "side_name" in frame.columns:
        values = frame["side_name"].astype(str).str.lower().to_numpy(copy=False)
        return np.where(np.isin(values, ("long", "short")), values, "unknown")
    if "side" in frame.columns:
        numeric = pd.to_numeric(frame["side"], errors="coerce").fillna(1.0).to_numpy()
        return np.where(numeric < 0.0, "short", "long")
    return np.full(len(frame), "unknown", dtype=object)


def fit_target_reference(
    train_frame: pd.DataFrame,
    spec: TargetDesignSpec,
) -> dict[str, Any]:
    """Fit compact side-relative target transforms on permitted train rows.

    The reference stores robust location/scale and ECDF knots rather than
    labels.  Passing it into OOS target construction makes fold-local fitting
    impossible by contract.
    """

    net, net_meta = resolve_net_return(train_frame, spec)
    sides = _side_values(train_frame)
    levels = np.linspace(0.0, 1.0, max(int(spec.side_ecdf_knots), 5), dtype=np.float64)
    global_finite = net[np.isfinite(net)]
    fallback = global_finite if global_finite.size else np.array([0.0], dtype=np.float64)
    reference: dict[str, Any] = {
        "schema": "side_economic_target_reference_v1",
        "quantile_levels": levels.tolist(),
        "net_return_column": net_meta["net_return_column"],
        "sides": {},
    }
    for side in ("long", "short", "unknown"):
        local = net[(sides == side) & np.isfinite(net)]
        source = local if local.size >= 32 else fallback
        q25, q75 = np.quantile(source, (0.25, 0.75))
        reference["sides"][side] = {
            "rows": int(local.size),
            "uses_global_fallback": bool(local.size < 32),
            "median": float(np.median(source)),
            "iqr": max(float(q75 - q25), 1e-5),
            "ecdf_knots": np.maximum.accumulate(np.quantile(source, levels)).astype(float).tolist(),
        }
    return reference


def _side_reference_arrays(
    frame: pd.DataFrame,
    reference: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
    sides = _side_values(frame)
    records = dict(reference.get("sides") or {})
    fallback = records.get("unknown") or next(
        iter(records.values()), {"median": 0.0, "iqr": 1.0, "ecdf_knots": [0.0, 1.0]}
    )
    medians = np.empty(len(frame), dtype=np.float64)
    iqrs = np.empty(len(frame), dtype=np.float64)
    knots: list[np.ndarray] = []
    for idx, side in enumerate(sides):
        row = records.get(str(side), fallback)
        medians[idx] = float(row.get("median", 0.0))
        iqrs[idx] = max(float(row.get("iqr", 1.0)), 1e-5)
        knots.append(np.asarray(row.get("ecdf_knots", [0.0, 1.0]), dtype=np.float64))
    return medians, iqrs, knots


def resolve_net_return(
    frame: pd.DataFrame,
    spec: TargetDesignSpec,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Return an executable *net* return and record the cost treatment."""

    values, source_column = _numeric_column(frame, NET_RETURN_CANDIDATES)
    net = values.fillna(0.0).to_numpy(dtype=np.float64, copy=False)
    mode = str(spec.cost_mode).strip().lower()
    if mode == "already_net":
        return net, {
            "net_return_column": source_column,
            "cost_mode": "already_net",
            "additional_cost_subtracted": 0.0,
        }
    if mode != "gross_minus_cost":
        raise ValueError(f"Unsupported cost_mode={spec.cost_mode!r}")

    cost = float(spec.round_trip_cost)
    if "__first_touch_round_trip_cost__" in frame.columns:
        observed = pd.to_numeric(
            frame["__first_touch_round_trip_cost__"], errors="coerce"
        ).to_numpy(dtype=np.float64, copy=False)
        finite = observed[np.isfinite(observed)]
        if finite.size:
            cost = float(np.median(finite))
    return net - cost, {
        "net_return_column": source_column,
        "cost_mode": "gross_minus_cost",
        "additional_cost_subtracted": float(cost),
    }


def resolve_volatility(frame: pd.DataFrame) -> tuple[np.ndarray, dict[str, Any]]:
    """Resolve a positive, bounded geometry scale without target leakage."""

    values, source_column = _numeric_column(frame, VOLATILITY_CANDIDATES)
    raw = np.abs(values.to_numpy(dtype=np.float64, copy=False))
    finite_positive = raw[np.isfinite(raw) & (raw > 1e-8)]
    fallback = float(np.median(finite_positive)) if finite_positive.size else 0.01
    vol = np.nan_to_num(raw, nan=fallback, posinf=fallback, neginf=fallback)
    vol = np.maximum(vol, 1e-5)
    return vol, {"volatility_column": source_column, "volatility_fallback": fallback}


def build_target(
    frame: pd.DataFrame,
    spec: TargetDesignSpec,
    reference: Mapping[str, Any] | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Build a bounded soft target.

    Side-relative variants require an explicit train-fitted reference.  This
    prevents an OOS label distribution from being fit silently.
    """

    kind = str(spec.target_kind).strip().lower()
    if kind == "existing_first_touch":
        existing, source_column = _numeric_column(frame, ("__first_touch_target_soft__",))
        target = np.clip(existing.fillna(0.5).to_numpy(dtype=np.float64, copy=False), 0.0, 1.0)
        return target.astype(np.float32), {
            "target_kind": kind,
            "target_source_column": source_column,
            "cost_mode": "materialized_label",
            "additional_cost_subtracted": 0.0,
        }

    net, net_meta = resolve_net_return(frame, spec)
    vol, vol_meta = resolve_volatility(frame)
    margin = float(spec.margin)
    raw = _sigmoid((net - margin) / max(float(spec.raw_temperature), 1e-6))
    standardized = np.clip((net - margin) / vol, -float(spec.vol_clip), float(spec.vol_clip))
    vol_norm = _sigmoid(standardized / max(float(spec.vol_temperature), 1e-6))
    if kind == "raw_net":
        target = raw
    elif kind == "vol_norm_net":
        target = vol_norm
    elif kind == "dual_raw_vol":
        alpha = float(np.clip(spec.dual_raw_weight, 0.0, 1.0))
        target = alpha * raw + (1.0 - alpha) * vol_norm
    elif kind in {"side_robust_net", "side_net_ecdf"}:
        if reference is None:
            raise ValueError(f"{kind} requires a train-fitted target reference")
        medians, iqrs, knots = _side_reference_arrays(frame, reference)
        if kind == "side_robust_net":
            target = _sigmoid(
                (net - medians) / (iqrs * max(float(spec.side_temperature), 1e-6))
            )
        else:
            levels = np.asarray(reference.get("quantile_levels", []), dtype=np.float64)
            if levels.size < 2:
                raise ValueError("side_net_ecdf requires quantile_levels in its target reference")
            target = np.empty(len(net), dtype=np.float64)
            for idx, local_knots in enumerate(knots):
                target[idx] = np.interp(net[idx], local_knots, levels, left=0.0, right=1.0)
    else:
        raise ValueError(f"Unsupported target_kind={spec.target_kind!r}")
    target = np.clip(np.nan_to_num(target, nan=0.5, posinf=1.0, neginf=0.0), 0.0, 1.0)
    return target.astype(np.float32), {
        "target_kind": kind,
        "raw_target_mean": float(np.mean(raw)),
        "vol_target_mean": float(np.mean(vol_norm)),
        "net_return_mean": float(np.mean(net)),
        "net_return_p90": float(np.quantile(net, 0.90)),
        "reference_schema": None if reference is None else reference.get("schema"),
        **net_meta,
        **vol_meta,
    }


def fit_training_weight_reference(train_frame: pd.DataFrame) -> dict[str, float]:
    """Fit the only data-derived scale on permitted training rows."""

    vol, _ = resolve_volatility(train_frame)
    median = float(np.median(vol[np.isfinite(vol) & (vol > 1e-8)])) if len(vol) else 0.01
    return {"volatility_train_median": max(median, 1e-5)}


def build_training_weights(
    train_frame: pd.DataFrame,
    spec: TargetDesignSpec,
    reference: Mapping[str, float] | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Return timestamp-balanced, optionally volatility-damped train weights.

    This routine must only be called for fitting rows.  It intentionally has no
    OOS equivalent, because sample weights are not a prediction-time feature.
    """

    n_rows = len(train_frame)
    if n_rows == 0:
        return np.empty(0, dtype=np.float32), {"rows": 0}
    mode = str(spec.weight_mode).strip().lower()
    weights = np.ones(n_rows, dtype=np.float64)
    if "timestamp_balanced" in mode:
        if "__ts__" not in train_frame.columns:
            raise KeyError("Timestamp-balanced weights require __ts__")
        groups = pd.to_datetime(train_frame["__ts__"], utc=True, errors="coerce")
        counts = groups.groupby(groups, dropna=False).transform("size").to_numpy(dtype=np.float64)
        # Match the repository's W7 contract exactly: it clips the raw inverse
        # timestamp count before mean normalization.  On this dense global
        # universe W7 often collapses to unit weights; changing that silently
        # would confound a target-design comparison with a new weight policy.
        weights = np.clip(1.0 / np.maximum(counts, 1.0), 0.10, 5.0)
    ref = dict(reference or fit_training_weight_reference(train_frame))
    if "vol_damped" in mode:
        vol, _ = resolve_volatility(train_frame)
        median = max(float(ref["volatility_train_median"]), 1e-5)
        relative = np.maximum(vol / median, 1e-5)
        damped = np.power(relative, -float(spec.vol_weight_power))
        damped = np.clip(damped, float(spec.vol_weight_lower), float(spec.vol_weight_upper))
        weights *= damped
    weights = np.nan_to_num(weights, nan=1.0, posinf=1.0, neginf=1.0)
    weights *= float(n_rows) / max(float(weights.sum()), 1e-12)
    weights = np.clip(weights, 0.10, 5.0)
    return weights.astype(np.float32), {
        "weight_mode": mode,
        "rows": int(n_rows),
        "weight_mean": float(np.mean(weights)),
        "weight_std": float(np.std(weights)),
        "effective_n": float((weights.sum() ** 2) / max(np.square(weights).sum(), 1e-12)),
        **ref,
    }


__all__ = [
    "DEFAULT_TARGET_SPECS",
    "NET_RETURN_CANDIDATES",
    "TargetDesignSpec",
    "build_target",
    "build_training_weights",
    "fit_target_reference",
    "fit_training_weight_reference",
    "resolve_net_return",
    "resolve_volatility",
]
