"""Causal shared-bps calibration for a regime-aware residual stack.

This module intentionally implements a *small additive correction*, not a
regime-routed model.  A shared residual expert owns the candidate ranking.  The
calibrator only estimates how much that common-bps prediction is biased in the
prior-resolved global, side, or side x *soft* regime population:

``prediction_bps + global_bias + side_bias + soft_regime_bias``.

The C2 term is an expectation under regime probabilities, with each component
strongly shrunk to its side correction.  There are no fitted per-regime models,
no hard-regime gates, and no regime-local prediction path.

The optional C3 form is still one shared mapping, expressed as a strongly
shrunk hierarchy of affine *deviations* in the same common-bps space:

``global_intercept + global_slope * prediction_bps``
``+ side_intercept + side_slope * prediction_bps``
``+ Σ p(regime) * (side×regime_intercept + side×regime_slope * prediction_bps)``.

It is not a collection of local experts: every score follows the same formula
and soft regime probabilities only form an expectation over correction terms.
All fit entry points require an explicit ``fit_before_utc`` and reject rows
whose outcomes were not already resolved at that cutoff.  The prequential
helper makes that temporal contract convenient for chronological OOF/replay
materialisation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Sequence

import numpy as np
import pandas as pd


CalibrationMode = Literal[
    "C0_global",
    "C1_side",
    "C2_side_soft_regime",
    "C3_hierarchical_affine_soft_regime",
]


class CausalCalibrationError(ValueError):
    """Raised when a calibration fit or application violates its lineage."""


@dataclass(frozen=True)
class SharedBpsCalibration:
    """Frozen additive calibration corrections expressed in common bps.

    ``side_corrections_bps`` are deviations from the global correction.  The
    optional regime matrix stores deviations from each side correction, one
    column for each soft regime probability.  This representation makes C2
    explicitly hierarchical and prevents double-counting the parent terms.
    """

    mode: CalibrationMode
    global_correction_bps: float
    side_corrections_bps: dict[str, float]
    side_support: dict[str, int]
    regime_corrections_bps: dict[str, tuple[float, ...]]
    regime_effective_support: dict[str, tuple[float, ...]]
    regime_probability_columns: tuple[str, ...]
    fit_before_utc: pd.Timestamp
    max_resolution_utc: pd.Timestamp
    global_support: int
    side_shrink_rows: float
    regime_shrink_rows: float
    regime_weight_cap: float
    # C3 affine terms.  Additive C0--C2 retain their historical behaviour:
    # their slope fields stay at the identity / zero-deviation defaults.
    global_slope: float = 1.0
    side_slope_corrections: dict[str, float] = field(default_factory=dict)
    regime_slope_corrections: dict[str, tuple[float, ...]] = field(default_factory=dict)
    global_shrink_rows: float = 0.0
    contract: str = "shared_additive_common_bps_v1"


def _utc(value: object, *, name: str) -> pd.Timestamp:
    parsed = pd.Timestamp(value)
    if parsed.tzinfo is None:
        parsed = parsed.tz_localize("UTC")
    else:
        parsed = parsed.tz_convert("UTC")
    if pd.isna(parsed):
        raise CausalCalibrationError(f"{name} is not a finite UTC timestamp")
    return parsed


def _utc_series(frame: pd.DataFrame, column: str, *, name: str) -> pd.Series:
    if column not in frame:
        raise CausalCalibrationError(f"calibration frame lacks {name} column {column!r}")
    parsed = pd.to_datetime(frame[column], utc=True, errors="coerce")
    if parsed.isna().any():
        raise CausalCalibrationError(f"{name} column {column!r} contains invalid timestamps")
    return parsed


def _require_common_inputs(
    frame: pd.DataFrame,
    raw_prediction_bps: Sequence[float] | np.ndarray,
    realised_net_bps: Sequence[float] | np.ndarray | None,
    *,
    side_column: str,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray]:
    if side_column not in frame:
        raise CausalCalibrationError(f"calibration frame lacks side column {side_column!r}")
    raw = np.asarray(raw_prediction_bps, dtype=np.float64)
    if raw.ndim != 1 or len(raw) != len(frame):
        raise CausalCalibrationError("raw_prediction_bps must be a one-dimensional row-aligned array")
    if not np.isfinite(raw).all():
        raise CausalCalibrationError("raw_prediction_bps must be finite common-bps values")
    target: np.ndarray | None = None
    if realised_net_bps is not None:
        target = np.asarray(realised_net_bps, dtype=np.float64)
        if target.ndim != 1 or len(target) != len(frame):
            raise CausalCalibrationError("realised_net_bps must be a one-dimensional row-aligned array")
        if not np.isfinite(target).all():
            raise CausalCalibrationError("realised_net_bps must be finite common-bps values")
    side = frame[side_column].astype(str).str.lower().to_numpy(dtype=object)
    if pd.Series(side).str.strip().eq("").any():
        raise CausalCalibrationError("side values must be non-empty")
    return raw, target, side


def _soft_matrix(
    frame: pd.DataFrame,
    columns: Sequence[str],
    *,
    required: bool,
) -> tuple[np.ndarray, tuple[str, ...]]:
    fields = tuple(str(column) for column in columns)
    if not fields:
        if required:
            raise CausalCalibrationError("C2 requires at least two soft regime probability columns")
        return np.empty((len(frame), 0), dtype=np.float64), fields
    if len(fields) < 2:
        raise CausalCalibrationError("soft regime calibration requires at least two probability columns")
    missing = [column for column in fields if column not in frame]
    if missing:
        raise CausalCalibrationError(f"calibration frame lacks soft regime columns: {missing}")
    values = frame.loc[:, fields].apply(pd.to_numeric, errors="coerce").to_numpy(np.float64)
    if not np.isfinite(values).all() or (values < -1e-8).any():
        raise CausalCalibrationError("soft regime probabilities must be finite and non-negative")
    if not np.allclose(values.sum(axis=1), 1.0, rtol=0.0, atol=1e-6):
        raise CausalCalibrationError("soft regime probabilities must sum to one on every row")
    return np.clip(values, 0.0, 1.0), fields


def _weighted_affine(
    x: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray | None = None,
    *,
    prior_slope: float = 0.0,
) -> tuple[float, float]:
    """Return an intercept/slope fit, with a stable identity-slope fallback.

    This intentionally uses only a two-parameter weighted least-squares
    summary.  Hierarchy and shrinkage are applied by the caller; no per-regime
    regressor or routing model is created here.
    """
    if weights is None:
        weights = np.ones(len(x), dtype=np.float64)
    total = float(weights.sum())
    if total <= 0.0:
        return 0.0, 0.0
    mean_x = float(np.dot(weights, x) / total)
    mean_y = float(np.dot(weights, y) / total)
    centered_x = x - mean_x
    denominator = float(np.dot(weights, centered_x * centered_x))
    if denominator <= 1e-12:
        return float(mean_y - prior_slope * mean_x), float(prior_slope)
    slope = float(np.dot(weights, centered_x * (y - mean_y)) / denominator)
    return float(mean_y - slope * mean_x), slope


def _shrink_weight(support: float, shrink_rows: float, *, cap: float = 1.0) -> float:
    return float(min(cap, support / (support + shrink_rows))) if support > 0.0 else 0.0


def fit_shared_bps_calibration(
    frame: pd.DataFrame,
    raw_prediction_bps: Sequence[float] | np.ndarray,
    realised_net_bps: Sequence[float] | np.ndarray,
    *,
    fit_before_utc: object,
    mode: CalibrationMode = "C2_side_soft_regime",
    resolution_column: str = "outcome_resolved_at",
    side_column: str = "side_name",
    soft_regime_columns: Sequence[str] = (),
    min_global_rows: int = 32,
    global_shrink_rows: float = 5_000.0,
    side_shrink_rows: float = 1_500.0,
    regime_shrink_rows: float = 3_000.0,
    regime_weight_cap: float = 0.50,
) -> SharedBpsCalibration:
    """Fit C0/C1/C2 on labels resolved strictly before a declared cutoff.

    ``fit_before_utc`` is normally the first decision timestamp in the later
    OOF/OOS anchor.  Requiring it instead of inferring it prevents a caller
    from accidentally treating a frame's last timestamp as a causal boundary.
    """

    if mode not in ("C0_global", "C1_side", "C2_side_soft_regime", "C3_hierarchical_affine_soft_regime"):
        raise CausalCalibrationError(f"unsupported calibration mode {mode!r}")
    if min_global_rows < 1 or global_shrink_rows <= 0 or side_shrink_rows <= 0 or regime_shrink_rows <= 0:
        raise CausalCalibrationError("minimum support and shrinkage constants must be positive")
    if not 0.0 < regime_weight_cap <= 1.0:
        raise CausalCalibrationError("regime_weight_cap must be in (0, 1]")

    raw, target, side = _require_common_inputs(
        frame, raw_prediction_bps, realised_net_bps, side_column=side_column
    )
    assert target is not None
    resolution = _utc_series(frame, resolution_column, name="outcome resolution")
    cutoff = _utc(fit_before_utc, name="fit_before_utc")
    # ``<`` (not ``<=``) avoids co-timestamp outcome use under unknown within-
    # timestamp ordering.  It is deliberately more conservative than a daily
    # as-of join.
    if not (resolution < cutoff).all():
        latest = resolution.max()
        raise CausalCalibrationError(
            "calibration fit includes unresolved/current/future outcomes: "
            f"max_resolution_utc={latest} fit_before_utc={cutoff}"
        )
    if len(frame) < min_global_rows:
        raise CausalCalibrationError(
            f"calibration requires at least {min_global_rows} prior-resolved rows; got {len(frame)}"
        )
    if mode in ("C2_side_soft_regime", "C3_hierarchical_affine_soft_regime"):
        soft, soft_columns = _soft_matrix(frame, soft_regime_columns, required=True)
    else:
        # Validate optional fields when supplied, while C0/C1 remain explicitly
        # independent of regime membership.
        soft, soft_columns = _soft_matrix(frame, soft_regime_columns, required=False)

    residual = target - raw
    affine = mode == "C3_hierarchical_affine_soft_regime"
    if affine:
        # The global slope is anchored at identity and the global intercept at
        # zero.  ``global_shrink_rows`` is deliberately high: C3 may correct a
        # shared score scale, but cannot freely replace it with a local model.
        _, raw_slope = _weighted_affine(raw, target, prior_slope=1.0)
        global_weight = _shrink_weight(float(len(frame)), float(global_shrink_rows))
        global_slope = 1.0 + global_weight * (raw_slope - 1.0)
        global_correction = float(global_weight * np.mean(target - global_slope * raw))
    else:
        global_slope = 1.0
        global_correction = float(np.mean(residual))
    global_support = int(len(frame))
    side_corrections: dict[str, float] = {}
    side_support: dict[str, int] = {}
    side_slope_corrections: dict[str, float] = {}
    for key in sorted(pd.unique(side)):
        pos = side == key
        count = int(pos.sum())
        side_support[str(key)] = count
        weight = _shrink_weight(float(count), float(side_shrink_rows))
        if affine:
            parent_residual = target[pos] - (global_correction + global_slope * raw[pos])
            _, raw_delta_slope = _weighted_affine(raw[pos], parent_residual)
            delta_slope = weight * raw_delta_slope
            side_slope_corrections[str(key)] = float(delta_slope)
            side_corrections[str(key)] = float(weight * np.mean(parent_residual - delta_slope * raw[pos]))
        else:
            # C1's correction is a strongly-shrunk *deviation* from C0, not an
            # independently calibrated score scale.
            side_corrections[str(key)] = float(weight * (residual[pos].mean() - global_correction))

    regime_corrections: dict[str, tuple[float, ...]] = {}
    regime_support: dict[str, tuple[float, ...]] = {}
    regime_slope_corrections: dict[str, tuple[float, ...]] = {}
    if mode in ("C2_side_soft_regime", "C3_hierarchical_affine_soft_regime"):
        for key in sorted(pd.unique(side)):
            pos = side == key
            p = soft[pos]
            r = residual[pos]
            parent = global_correction + side_corrections[str(key)]
            effective_n = p.sum(axis=0)
            weight = np.asarray([
                _shrink_weight(float(value), float(regime_shrink_rows), cap=float(regime_weight_cap))
                for value in effective_n
            ])
            if affine:
                parent_value = parent + (global_slope + side_slope_corrections[str(key)]) * raw[pos]
                parent_residual = target[pos] - parent_value
                intercepts = np.zeros(p.shape[1], dtype=np.float64)
                slopes = np.zeros(p.shape[1], dtype=np.float64)
                for col in range(p.shape[1]):
                    raw_intercept, raw_slope = _weighted_affine(raw[pos], parent_residual, p[:, col])
                    slopes[col] = weight[col] * raw_slope
                    # Condition the intercept on its similarly shrunk slope to
                    # preserve a proper affine correction rather than mixing
                    # an uncentred slope with a separately shrunk mean.
                    support = effective_n[col]
                    mean_residual = (float(np.dot(p[:, col], parent_residual)) / support) if support > 0 else 0.0
                    mean_raw = (float(np.dot(p[:, col], raw[pos])) / support) if support > 0 else 0.0
                    intercepts[col] = weight[col] * (mean_residual - slopes[col] * mean_raw)
                regime_corrections[str(key)] = tuple(intercepts.tolist())
                regime_slope_corrections[str(key)] = tuple(slopes.tolist())
            else:
                weighted_mean = np.divide(
                    (p * r[:, None]).sum(axis=0),
                    effective_n,
                    out=np.full(p.shape[1], parent, dtype=np.float64),
                    where=effective_n > 0,
                )
                # Store only the incremental side x soft-regime deviation. At
                # prediction it is averaged by the contemporaneous soft simplex.
                regime_corrections[str(key)] = tuple((weight * (weighted_mean - parent)).tolist())
            regime_support[str(key)] = tuple(effective_n.astype(float).tolist())

    return SharedBpsCalibration(
        mode=mode,
        global_correction_bps=global_correction,
        side_corrections_bps=side_corrections,
        side_support=side_support,
        regime_corrections_bps=regime_corrections,
        regime_effective_support=regime_support,
        regime_probability_columns=soft_columns if mode in ("C2_side_soft_regime", "C3_hierarchical_affine_soft_regime") else (),
        fit_before_utc=cutoff,
        max_resolution_utc=pd.Timestamp(resolution.max()),
        global_support=global_support,
        side_shrink_rows=float(side_shrink_rows),
        regime_shrink_rows=float(regime_shrink_rows),
        regime_weight_cap=float(regime_weight_cap),
        global_slope=float(global_slope),
        side_slope_corrections=side_slope_corrections,
        regime_slope_corrections=regime_slope_corrections,
        global_shrink_rows=float(global_shrink_rows) if affine else 0.0,
        contract=("shared_hierarchical_affine_common_bps_v1" if affine else "shared_additive_common_bps_v1"),
    )


def predict_shared_bps_calibration(
    calibrator: SharedBpsCalibration,
    frame: pd.DataFrame,
    raw_prediction_bps: Sequence[float] | np.ndarray,
    *,
    decision_timestamp_column: str = "__ts__",
    side_column: str = "side_name",
    require_after_fit_boundary: bool = True,
    return_details: bool = False,
) -> np.ndarray | pd.DataFrame:
    """Apply a frozen correction and return a globally comparable bps score.

    The default boundary check disallows using a calibrator to re-score its own
    fit rows.  Set it to ``False`` only for training diagnostics; it does not
    weaken the fit-time prior-resolution check.
    """

    raw, _, side = _require_common_inputs(frame, raw_prediction_bps, None, side_column=side_column)
    decision = _utc_series(frame, decision_timestamp_column, name="decision timestamp")
    if require_after_fit_boundary and not (decision >= calibrator.fit_before_utc).all():
        raise CausalCalibrationError(
            "calibrator may only be applied at/after its fit boundary; "
            "use require_after_fit_boundary=False for fit diagnostics"
        )
    affine = calibrator.mode == "C3_hierarchical_affine_soft_regime"
    # Keep components as *corrections* to raw prediction so the returned
    # quantity remains a directly comparable common-bps score for every side.
    global_component = (
        calibrator.global_correction_bps + (calibrator.global_slope - 1.0) * raw
        if affine else np.full(len(frame), calibrator.global_correction_bps, dtype=np.float64)
    )
    side_component = np.asarray(
        [
            calibrator.side_corrections_bps.get(str(key), 0.0)
            + (calibrator.side_slope_corrections.get(str(key), 0.0) * raw[row] if affine else 0.0)
            for row, key in enumerate(side)
        ], dtype=np.float64
    ) if calibrator.mode != "C0_global" else np.zeros(len(frame), dtype=np.float64)
    regime_component = np.zeros(len(frame), dtype=np.float64)
    if calibrator.mode in ("C2_side_soft_regime", "C3_hierarchical_affine_soft_regime"):
        soft, fields = _soft_matrix(frame, calibrator.regime_probability_columns, required=True)
        if fields != calibrator.regime_probability_columns:
            raise CausalCalibrationError("soft regime probability contract changed after fitting")
        for key in pd.unique(side):
            pos = side == key
            correction = calibrator.regime_corrections_bps.get(str(key))
            if correction is not None:
                intercept = soft[pos] @ np.asarray(correction, dtype=np.float64)
                if affine:
                    slopes = calibrator.regime_slope_corrections.get(str(key), ())
                    slope = soft[pos] @ np.asarray(slopes, dtype=np.float64)
                    regime_component[pos] = intercept + slope * raw[pos]
                else:
                    regime_component[pos] = intercept
    calibrated = raw + global_component + side_component + regime_component
    if return_details:
        return pd.DataFrame(
            {
                "raw_common_bps": raw,
                "calibration_global_correction_bps": global_component,
                "calibration_side_correction_bps": side_component,
                "calibration_soft_regime_correction_bps": regime_component,
                "calibrated_common_bps": calibrated,
                "calibration_mode": calibrator.mode,
                "calibration_fit_before_utc": calibrator.fit_before_utc,
                "calibration_max_resolution_utc": calibrator.max_resolution_utc,
            },
            index=frame.index,
        )
    return calibrated.astype(np.float64, copy=False)


def prequential_shared_bps_calibration(
    frame: pd.DataFrame,
    raw_prediction_bps: Sequence[float] | np.ndarray,
    realised_net_bps: Sequence[float] | np.ndarray,
    *,
    mode: CalibrationMode = "C2_side_soft_regime",
    decision_timestamp_column: str = "__ts__",
    resolution_column: str = "outcome_resolved_at",
    side_column: str = "side_name",
    soft_regime_columns: Sequence[str] = (),
    anchor: Literal["day", "timestamp"] = "day",
    min_global_rows: int = 32,
    global_shrink_rows: float = 5_000.0,
    side_shrink_rows: float = 1_500.0,
    regime_shrink_rows: float = 3_000.0,
    regime_weight_cap: float = 0.50,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Generate chronological calibration OOF predictions from prior outcomes.

    The first anchors with insufficient resolved support are exact identity
    fallbacks.  This is intentional: no global/side/regime prior may be
    manufactured from current or later labels.
    """

    if anchor not in ("day", "timestamp"):
        raise CausalCalibrationError("anchor must be 'day' or 'timestamp'")
    raw, target, _ = _require_common_inputs(
        frame, raw_prediction_bps, realised_net_bps, side_column=side_column
    )
    assert target is not None
    decision = _utc_series(frame, decision_timestamp_column, name="decision timestamp")
    resolution = _utc_series(frame, resolution_column, name="outcome resolution")
    # Explicitly validate the regime contract before iterating so a malformed
    # later group cannot leave a half-materialised artifact behind.
    if mode in ("C2_side_soft_regime", "C3_hierarchical_affine_soft_regime"):
        _soft_matrix(frame, soft_regime_columns, required=True)

    group_key = decision.dt.normalize() if anchor == "day" else decision
    order = np.argsort(decision.to_numpy(dtype="datetime64[ns]"), kind="stable")
    output = raw.copy()
    audit: list[dict[str, object]] = []
    keys = group_key.to_numpy()[order]
    start = 0
    while start < len(order):
        stop = start + 1
        while stop < len(order) and keys[stop] == keys[start]:
            stop += 1
        positions = order[start:stop]
        cutoff = pd.Timestamp(decision.iloc[positions].min())
        prior = resolution < cutoff
        # The decision timestamp should never occur after its own path resolves.
        # This check catches accidentally supplied post-entry decisions.
        if (resolution.iloc[positions] < decision.iloc[positions]).any():
            raise CausalCalibrationError("a row resolves before its decision timestamp")
        if int(prior.sum()) < min_global_rows:
            audit.append(
                {
                    "anchor_utc": cutoff,
                    "status": "identity_no_prior_resolved_support",
                    "prior_rows": int(prior.sum()),
                    "max_resolution_utc": pd.NaT if not prior.any() else pd.Timestamp(resolution[prior].max()),
                    "prediction_rows": int(len(positions)),
                    "mode": mode,
                }
            )
        else:
            fit = fit_shared_bps_calibration(
                frame.loc[prior], raw[prior], target[prior], fit_before_utc=cutoff,
                mode=mode, resolution_column=resolution_column, side_column=side_column,
                soft_regime_columns=soft_regime_columns, min_global_rows=min_global_rows,
                global_shrink_rows=global_shrink_rows,
                side_shrink_rows=side_shrink_rows, regime_shrink_rows=regime_shrink_rows,
                regime_weight_cap=regime_weight_cap,
            )
            output[positions] = predict_shared_bps_calibration(
                fit, frame.iloc[positions], raw[positions],
                decision_timestamp_column=decision_timestamp_column, side_column=side_column,
            )
            audit.append(
                {
                    "anchor_utc": cutoff,
                    "status": "prior_resolved_hierarchical_calibration",
                    "prior_rows": fit.global_support,
                    "max_resolution_utc": fit.max_resolution_utc,
                    "prediction_rows": int(len(positions)),
                    "mode": mode,
                    "global_correction_bps": fit.global_correction_bps,
                }
            )
        start = stop
    return output, pd.DataFrame(audit)


__all__ = [
    "CalibrationMode",
    "CausalCalibrationError",
    "SharedBpsCalibration",
    "fit_shared_bps_calibration",
    "predict_shared_bps_calibration",
    "prequential_shared_bps_calibration",
]
