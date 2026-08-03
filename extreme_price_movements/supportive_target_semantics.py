"""Explicit, non-promotional semantics for path-supportive training targets.

The historical ``supportive_labels.parquet`` pack deliberately contains realised
H12 paths.  Those columns are labels, never inference features.  This module
creates a compact *target sidecar* with three safeguards the frozen pack did
not encode uniformly:

* conditional Peak-MFE and pre-MFE-MAE regression targets are null unless the
  meaningful-MFE event was actually reached and the source path is valid;
* time-to-opportunity and time-to-meaningful-MFE are represented as right-
  censored event time, reach, cumulative-incidence and interval-hazard labels,
  rather than treating an unreached row's 12h sentinel as an observed time;
* opportunity, adverse, persistence and adverse-recovery support labels carry
  explicit validity masks and retain the exact source-path semantics.

The output is a label contract for future strict-OOF head experiments.  It is
not a model feature contract and must not be added to base/execution inference
inputs directly.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd


SCHEMA = "supportive_target_semantics_v1"
DEFAULT_HORIZON_HOURS = 12.0
DEFAULT_HAZARD_BOUNDARIES_HOURS = (1.0, 2.0, 4.0, 8.0, 12.0)
IDENTITY = ("candidate_id", "symbol", "side", "decision_ts", "label_end_ts", "label_available_ts")
REQUIRED_SOURCE_COLUMNS = (
    *IDENTITY,
    "__path_auxiliary_target_valid__", "__time_to_first_meaningful_mfe_target_valid__",
    "__meaningful_mfe_reached_12h__", "__peak_mfe_atr_12h__",
    "__mae_before_meaningful_mfe_atr_12h__", "__time_to_first_meaningful_mfe_hours_12h__",
    "clean_economic_favorable_first", "adverse_first", "same_minute_favorable_adverse_conflict",
    "first_favorable_minute", "__mfe_persistence_path_efficiency_12h__",
    "__adverse_trough_atr_12h__", "__adverse_trough_recovery_50pct_confirmed_2bars_12h__",
)


class SupportiveTargetContractError(ValueError):
    """The frozen label pack cannot prove the requested target semantics."""


@dataclass(frozen=True)
class CensoredTimeTarget:
    """Names for one reach/event-time target family in the materialised sidecar."""

    prefix: str
    event_observed: str
    observed_time_hours: str
    censor_time_hours: str


def _require(frame: pd.DataFrame, names: Iterable[str], *, context: str = "supportive label pack") -> None:
    missing = sorted(set(names).difference(frame.columns))
    if missing:
        raise SupportiveTargetContractError(f"{context} is missing columns: {missing}")


def _binary(frame: pd.DataFrame, name: str) -> np.ndarray:
    _require(frame, (name,))
    values = pd.to_numeric(frame[name], errors="coerce").to_numpy(float)
    if not np.isfinite(values).all() or not np.isin(values, (0.0, 1.0)).all():
        raise SupportiveTargetContractError(f"{name} must be a finite binary label")
    return values.astype(bool)


def _numeric(frame: pd.DataFrame, name: str, *, allow_missing: bool = False) -> np.ndarray:
    _require(frame, (name,))
    values = pd.to_numeric(frame[name], errors="coerce").to_numpy(float)
    if not allow_missing and not np.isfinite(values).all():
        raise SupportiveTargetContractError(f"{name} must be finite")
    return values


def _name_number(hours: float) -> str:
    return str(int(hours)) if float(hours).is_integer() else str(hours).replace(".", "p")


def _validate_hazard_boundaries(boundaries_hours: Sequence[float], *, horizon_hours: float) -> tuple[float, ...]:
    result = tuple(float(value) for value in boundaries_hours)
    if not result or not all(np.isfinite(value) and 0.0 < value <= horizon_hours for value in result):
        raise SupportiveTargetContractError("hazard boundaries must be finite values in (0, horizon]")
    if tuple(sorted(set(result))) != result or result[-1] != horizon_hours:
        raise SupportiveTargetContractError("hazard boundaries must be strictly increasing and end at horizon_hours")
    return result


def censored_time_labels(
    *,
    reached: np.ndarray | Sequence[bool],
    time_hours: np.ndarray | Sequence[float],
    valid: np.ndarray | Sequence[bool],
    prefix: str,
    horizon_hours: float = DEFAULT_HORIZON_HOURS,
    hazard_boundaries_hours: Sequence[float] = DEFAULT_HAZARD_BOUNDARIES_HOURS,
) -> pd.DataFrame:
    """Materialise right-censored reach, cumulative and discrete-hazard labels.

    A reached row has an observed event time in ``[0, horizon]``.  An unreached
    row remains an event-time *censoring* observation at the horizon: its
    observed time is null and it receives a zero cumulative-incidence target.
    Hazard labels are valid only while the row is still at risk at the start of
    the interval, avoiding the common error of assigning post-event zeroes.
    """
    horizon = float(horizon_hours)
    if not np.isfinite(horizon) or horizon <= 0.0:
        raise SupportiveTargetContractError("horizon_hours must be finite and positive")
    boundaries = _validate_hazard_boundaries(hazard_boundaries_hours, horizon_hours=horizon)
    event = np.asarray(reached, dtype=bool)
    time = np.asarray(time_hours, dtype=float)
    base_valid = np.asarray(valid, dtype=bool)
    if not (event.shape == time.shape == base_valid.shape):
        raise SupportiveTargetContractError("censored time inputs must have identical shapes")
    bad_event_time = base_valid & event & (~np.isfinite(time) | (time < 0.0) | (time > horizon))
    if bad_event_time.any():
        raise SupportiveTargetContractError(f"{prefix} reached rows require an observed time within the horizon")
    event_time = np.where(event, time, horizon)
    observed_time = np.where(base_valid & event, time, np.nan)
    rows: dict[str, Any] = {
        f"target_{prefix}_valid": base_valid.astype(np.int8),
        f"target_{prefix}_reached": np.where(base_valid, event, 0).astype(np.int8),
        f"target_{prefix}_event_observed": (base_valid & event).astype(np.int8),
        f"target_{prefix}_observed_time_hours": observed_time.astype(np.float32),
        f"target_{prefix}_censor_time_hours": np.where(base_valid, event_time, np.nan).astype(np.float32),
    }
    start = 0.0
    for end in boundaries:
        end_name = _name_number(end)
        rows[f"target_{prefix}_cumulative_reach_by_{end_name}h"] = (
            base_valid & event & (time <= end)
        ).astype(np.int8)
        interval_name = f"{_name_number(start)}_{end_name}h"
        # An event at t=0 is assigned to the first discrete interval so the
        # interval hazards remain exhaustive with cumulative incidence.
        at_risk = base_valid & ((~event) | (event_time > start) | ((start == 0.0) & (event_time == 0.0)))
        occurred = base_valid & event & (event_time <= end) & (
            (event_time > start) | ((start == 0.0) & (event_time == 0.0))
        )
        rows[f"target_{prefix}_hazard_{interval_name}_valid"] = at_risk.astype(np.int8)
        rows[f"target_{prefix}_hazard_{interval_name}"] = np.where(at_risk, occurred.astype(np.int8), np.nan).astype(np.float32)
        start = end
    return pd.DataFrame(rows)


def materialize_supportive_target_semantics(
    frame: pd.DataFrame,
    *,
    horizon_hours: float = DEFAULT_HORIZON_HOURS,
    hazard_boundaries_hours: Sequence[float] = DEFAULT_HAZARD_BOUNDARIES_HOURS,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Return a compact target-only sidecar and its machine-readable contract."""
    required = set(REQUIRED_SOURCE_COLUMNS)
    _require(frame, required)
    if frame.candidate_id.isna().any() or frame.candidate_id.astype(str).duplicated().any():
        raise SupportiveTargetContractError("candidate_id must be non-null and one-to-one")
    decision = pd.to_datetime(frame.decision_ts, utc=True, errors="coerce")
    label_end = pd.to_datetime(frame.label_end_ts, utc=True, errors="coerce")
    label_available = pd.to_datetime(frame.label_available_ts, utc=True, errors="coerce")
    if decision.isna().any() or label_end.isna().any() or label_available.isna().any():
        raise SupportiveTargetContractError("identity timestamps must be valid UTC")
    horizon = pd.Timedelta(hours=float(horizon_hours))
    if not label_end.eq(decision + horizon).all() or not label_available.eq(label_end).all():
        raise SupportiveTargetContractError("support label availability must equal the declared H-horizon end")

    path_valid = _binary(frame, "__path_auxiliary_target_valid__")
    time_source_valid = _binary(frame, "__time_to_first_meaningful_mfe_target_valid__")
    meaningful_reached = _binary(frame, "__meaningful_mfe_reached_12h__")
    peak = _numeric(frame, "__peak_mfe_atr_12h__")
    mae = _numeric(frame, "__mae_before_meaningful_mfe_atr_12h__")
    time_to_mfe = _numeric(frame, "__time_to_first_meaningful_mfe_hours_12h__", allow_missing=True)
    clean = _binary(frame, "clean_economic_favorable_first")
    adverse_first = _binary(frame, "adverse_first")
    conflict = _binary(frame, "same_minute_favorable_adverse_conflict")
    opportunity = clean
    adverse = adverse_first | conflict
    if (opportunity & adverse).any():
        raise SupportiveTargetContractError("opportunity and adverse support labels must be mutually exclusive")
    first_favorable_minutes = _numeric(frame, "first_favorable_minute", allow_missing=True)
    persistence = _numeric(frame, "__mfe_persistence_path_efficiency_12h__", allow_missing=True)
    trough = _numeric(frame, "__adverse_trough_atr_12h__")
    # Recovery is intentionally missing when a trough/recovery observation is
    # not available in the frozen path pack.  Preserve that missingness as an
    # invalid conditional target; do not turn it into a no-recovery zero.
    recovery_50 = _numeric(frame, "__adverse_trough_recovery_50pct_confirmed_2bars_12h__", allow_missing=True)
    finite_recovery = recovery_50[np.isfinite(recovery_50)]
    if not np.isin(finite_recovery, (0.0, 1.0)).all():
        raise SupportiveTargetContractError("confirmed adverse recovery must be binary wherever observed")

    result = frame.loc[:, IDENTITY].copy()
    for timestamp in ("decision_ts", "label_end_ts", "label_available_ts"):
        result[timestamp] = pd.to_datetime(result[timestamp], utc=True, errors="raise")
    result["target_source_path_valid"] = path_valid.astype(np.int8)
    result["target_meaningful_mfe_reached_12h"] = (path_valid & meaningful_reached).astype(np.int8)

    # Conditional path magnitudes deliberately retain nulls for unreached or
    # invalid rows: zero is a legitimate economic magnitude, not a censoring
    # substitute.  Regression fits must use the accompanying validity mask.
    conditional_mfe = path_valid & meaningful_reached
    result["target_peak_mfe_atr_given_meaningful_mfe_valid"] = conditional_mfe.astype(np.int8)
    result["target_peak_mfe_atr_given_meaningful_mfe"] = np.where(conditional_mfe, peak, np.nan).astype(np.float32)
    result["target_mae_before_meaningful_mfe_atr_given_meaningful_mfe_valid"] = conditional_mfe.astype(np.int8)
    result["target_mae_before_meaningful_mfe_atr_given_meaningful_mfe"] = np.where(conditional_mfe, mae, np.nan).astype(np.float32)

    mfe_time = censored_time_labels(
        reached=meaningful_reached,
        time_hours=time_to_mfe,
        valid=path_valid & time_source_valid,
        prefix="meaningful_mfe",
        horizon_hours=horizon_hours,
        hazard_boundaries_hours=hazard_boundaries_hours,
    )
    # Opportunity is a competing-risk event: an adverse/timeout row is right
    # censored for this single-event time target rather than labelled as a
    # 12-hour opportunity.  Its binary adverse support label remains separate.
    opportunity_time = censored_time_labels(
        reached=opportunity,
        time_hours=first_favorable_minutes / 60.0,
        valid=path_valid,
        prefix="opportunity",
        horizon_hours=horizon_hours,
        hazard_boundaries_hours=hazard_boundaries_hours,
    )
    result = pd.concat([result.reset_index(drop=True), mfe_time, opportunity_time], axis=1)

    result["support_opportunity_valid"] = path_valid.astype(np.int8)
    result["support_opportunity"] = np.where(path_valid, opportunity, np.nan).astype(np.float32)
    result["support_adverse_valid"] = path_valid.astype(np.int8)
    result["support_adverse"] = np.where(path_valid, adverse, np.nan).astype(np.float32)
    persistence_valid = conditional_mfe & np.isfinite(persistence)
    if (persistence[persistence_valid] < 0.0).any() or (persistence[persistence_valid] > 1.0).any():
        raise SupportiveTargetContractError("MFE persistence efficiency must be in [0, 1] on valid rows")
    result["support_persistence_given_meaningful_mfe_valid"] = persistence_valid.astype(np.int8)
    result["support_persistence_given_meaningful_mfe"] = np.where(persistence_valid, persistence, np.nan).astype(np.float32)
    recovery_valid = path_valid & np.isfinite(trough) & (trough > 0.0) & np.isfinite(recovery_50)
    result["support_adverse_recovery_50pct_confirmed_valid"] = recovery_valid.astype(np.int8)
    result["support_adverse_recovery_50pct_confirmed"] = np.where(recovery_valid, recovery_50, np.nan).astype(np.float32)

    contract = {
        "schema": SCHEMA,
        "promotion_eligible": False,
        "model_input_eligible": False,
        "horizon_hours": float(horizon_hours),
        "hazard_boundaries_hours": [float(value) for value in hazard_boundaries_hours],
        "identity": list(IDENTITY),
        "source_required_columns": sorted(required),
        "conditional_magnitude_contract": {
            "peak_mfe": "target_peak_mfe_atr_given_meaningful_mfe is observed only where path_valid AND meaningful_mfe_reached; otherwise null with explicit valid=0",
            "mae": "target_mae_before_meaningful_mfe_atr_given_meaningful_mfe is observed only where path_valid AND meaningful_mfe_reached; otherwise null with explicit valid=0",
        },
        "censoring_contract": {
            "meaningful_mfe": "unreached candidates are right-censored at H; they are not assigned H as an observed event time",
            "opportunity": "non-opportunity candidates are right-censored at H for the opportunity event; adverse is exposed as a separate support label",
            "hazards": "interval target is null when no longer at risk at interval start; cumulative incidence remains zero after a non-event",
        },
        "support_label_contract": {
            "opportunity": "clean_economic_favorable_first",
            "adverse": "adverse_first OR same_minute_favorable_adverse_conflict",
            "persistence": "MFE persistence path efficiency conditional on meaningful-MFE reach",
            "recovery": "confirmed 50% recovery after a non-zero adverse trough",
        },
        "prohibition": "all output columns are future-resolved labels or validity masks. They may be training targets or strict OOF head labels, never raw inference features.",
    }
    return result, contract
