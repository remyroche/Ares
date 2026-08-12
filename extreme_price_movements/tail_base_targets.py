"""Vectorised, leakage-safe labels for the second (tail) base ranker.

These labels are *training targets only*.  All inputs other than
``label_valid`` are realised-path quantities and must never be projected into
the base or meta feature contract.  The functions intentionally return ``-1``
for invalid rows rather than turning missing paths into ordinary losing labels.

The triple-barrier target uses two nested contracts:

* first resolve the outer ``+6 / -6 ATR`` contract, with an adverse tie break;
* for paths which do not reach either outer barrier, resolve ``+4 / -4 ATR``;
* otherwise assign the H12 timeout grade.

That is the same severe-first ordering already used by the path-label
materialiser, expressed here as a compact reusable utility.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd


INVALID_GRADE = np.int8(-1)
NET_GRADE_BOUNDS_BPS: tuple[float, ...] = (-50.0, 50.0, 150.0, 250.0, 350.0)
ATR_GRADE_BOUNDS: tuple[float, ...] = (-1.0, 0.0, 1.0, 2.0, 3.0)
DEFAULT_HORIZON_MINUTES = 12 * 60


class TailBaseTargetError(ValueError):
    """Raised when the tail-target row or path contract is violated."""


@dataclass(frozen=True)
class TailBaseTargetColumns:
    """Column contract for :func:`build_tail_base_targets`.

    The first-touch minute fields use ``-1`` for a valid timeout/no-touch and
    values in ``[0, horizon_minutes)`` for a touched barrier.  They are
    deliberately explicit rather than inferred from outcome values.
    """

    candidate_id: str = "candidate_id"
    decision_ts: str = "__decision_ts__"
    symbol: str = "__symbol__"
    side: str = "side_name"
    label_valid: str = "label_valid"
    exact_net_bps: str = "exact_net_bps"
    atr_bps: str = "atr_bps"
    first_tp4_minute: str = "first_tp4_minute"
    first_tp6_minute: str = "first_tp6_minute"
    first_sl4_minute: str = "first_sl4_minute"
    first_sl6_minute: str = "first_sl6_minute"

    @property
    def identity(self) -> tuple[str, str, str, str]:
        return (self.candidate_id, self.decision_ts, self.symbol, self.side)

    @property
    def required(self) -> tuple[str, ...]:
        return (
            *self.identity,
            self.label_valid,
            self.exact_net_bps,
            self.atr_bps,
            self.first_tp4_minute,
            self.first_tp6_minute,
            self.first_sl4_minute,
            self.first_sl6_minute,
        )


def _as_1d(values: Iterable[object] | np.ndarray, *, name: str) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 1:
        raise TailBaseTargetError(f"{name} must be one-dimensional")
    return array


def _numeric(values: Iterable[object] | np.ndarray, *, name: str) -> np.ndarray:
    array = _as_1d(values, name=name)
    return pd.to_numeric(pd.Series(array), errors="coerce").to_numpy(dtype=np.float64)


def _valid_mask(valid: Iterable[object] | np.ndarray, *, n: int) -> np.ndarray:
    raw = _as_1d(valid, name="label_valid")
    if len(raw) != n:
        raise TailBaseTargetError("label_valid must be row-aligned")
    if np.issubdtype(raw.dtype, np.bool_):
        return raw.astype(bool, copy=False)
    numeric = pd.to_numeric(pd.Series(raw), errors="coerce").to_numpy(dtype=np.float64)
    return np.isfinite(numeric) & (numeric == 1.0)


def _validated_numeric_target_inputs(
    net_bps: Iterable[object] | np.ndarray,
    atr_bps: Iterable[object] | np.ndarray,
    label_valid: Iterable[object] | np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    net = _numeric(net_bps, name="exact_net_bps")
    atr = _numeric(atr_bps, name="atr_bps")
    if len(atr) != len(net):
        raise TailBaseTargetError("exact_net_bps and atr_bps must be row-aligned")
    valid = _valid_mask(label_valid, n=len(net))
    bad = valid & (~np.isfinite(net) | ~np.isfinite(atr) | (atr <= 0.0))
    if bad.any():
        raise TailBaseTargetError(
            "valid rows require finite exact_net_bps and positive finite atr_bps"
        )
    return net, atr, valid


def grade_exact_net_bps(
    net_bps: Iterable[object] | np.ndarray,
    label_valid: Iterable[object] | np.ndarray,
) -> np.ndarray:
    """Return the six exact-net grades with thresholds in basis points.

    Grades are ``0..5`` for ``<=-50, (-50,50], (50,150], (150,250],
    (250,350], >350`` bps.  Invalid rows are ``-1``.
    """

    net = _numeric(net_bps, name="exact_net_bps")
    valid = _valid_mask(label_valid, n=len(net))
    if (valid & ~np.isfinite(net)).any():
        raise TailBaseTargetError("valid rows require finite exact_net_bps")
    grades = np.full(len(net), INVALID_GRADE, dtype=np.int8)
    grades[valid] = np.digitize(net[valid], NET_GRADE_BOUNDS_BPS, right=True).astype(np.int8)
    return grades


def grade_atr_normalized_net(
    net_bps: Iterable[object] | np.ndarray,
    atr_bps: Iterable[object] | np.ndarray,
    label_valid: Iterable[object] | np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return six grades of exact net normalised by decision-time ATR.

    ``z_atr = exact_net_bps / atr_bps``.  The returned tuple is
    ``(grades, z_atr)``; invalid rows have grade ``-1`` and ``NaN`` z-score.
    """

    net, atr, valid = _validated_numeric_target_inputs(net_bps, atr_bps, label_valid)
    z_atr = np.full(len(net), np.nan, dtype=np.float32)
    z_atr[valid] = (net[valid] / atr[valid]).astype(np.float32)
    grades = np.full(len(net), INVALID_GRADE, dtype=np.int8)
    grades[valid] = np.digitize(z_atr[valid], ATR_GRADE_BOUNDS, right=True).astype(np.int8)
    return grades, z_atr


def _first_touch_minutes(
    values: Iterable[object] | np.ndarray,
    *, name: str,
    valid: np.ndarray,
    horizon_minutes: int,
) -> np.ndarray:
    minute = _numeric(values, name=name)
    if len(minute) != len(valid):
        raise TailBaseTargetError(f"{name} must be row-aligned")
    # Valid no-touch is -1.  Avoid silently accepting fractional or out of
    # horizon offsets because that can scramble first-touch ordering.
    bad = valid & (
        ~np.isfinite(minute)
        | (minute < -1.0)
        | (minute >= float(horizon_minutes))
        | (minute != np.floor(minute))
    )
    if bad.any():
        raise TailBaseTargetError(
            f"valid {name} values must be integer -1 or minutes in [0, {horizon_minutes})"
        )
    return minute.astype(np.int32, copy=False)


def grade_first_touch_tbm(
    first_tp4_minute: Iterable[object] | np.ndarray,
    first_tp6_minute: Iterable[object] | np.ndarray,
    first_sl4_minute: Iterable[object] | np.ndarray,
    first_sl6_minute: Iterable[object] | np.ndarray,
    label_valid: Iterable[object] | np.ndarray,
    *,
    horizon_minutes: int = DEFAULT_HORIZON_MINUTES,
) -> np.ndarray:
    """Return the five first-touch triple-barrier grades.

    The nested-contract ordering is: severe adverse ``-6 ATR`` (grade 0),
    strong favourable ``+6 ATR`` (grade 4), then moderate adverse ``-4 ATR``
    (grade 1), moderate favourable ``+4 ATR`` (grade 3), and a neutral H12
    timeout (grade 2).  Ties resolve adversely.  Validity and nesting of the
    four first-touch inputs are checked before grading.
    """

    if not isinstance(horizon_minutes, (int, np.integer)) or horizon_minutes <= 0:
        raise TailBaseTargetError("horizon_minutes must be a positive integer")
    n = len(_as_1d(first_tp4_minute, name="first_tp4_minute"))
    valid = _valid_mask(label_valid, n=n)
    tp4 = _first_touch_minutes(first_tp4_minute, name="first_tp4_minute", valid=valid, horizon_minutes=horizon_minutes)
    tp6 = _first_touch_minutes(first_tp6_minute, name="first_tp6_minute", valid=valid, horizon_minutes=horizon_minutes)
    sl4 = _first_touch_minutes(first_sl4_minute, name="first_sl4_minute", valid=valid, horizon_minutes=horizon_minutes)
    sl6 = _first_touch_minutes(first_sl6_minute, name="first_sl6_minute", valid=valid, horizon_minutes=horizon_minutes)

    invalid_nesting = valid & (
        ((tp6 >= 0) & ((tp4 < 0) | (tp4 > tp6)))
        | ((sl6 >= 0) & ((sl4 < 0) | (sl4 > sl6)))
    )
    if invalid_nesting.any():
        raise TailBaseTargetError(
            "a +6/-6 first touch requires its +4/-4 parent touch at or before it"
        )

    grades = np.full(n, INVALID_GRADE, dtype=np.int8)
    grades[valid] = 2
    # Outer contract gets precedence.  An equal-minute opposing OHLC crossing
    # is ambiguous intrabar and therefore resolves against the candidate.
    severe_adverse = valid & (sl6 >= 0) & ((tp6 < 0) | (sl6 <= tp6))
    strong_favourable = valid & ~severe_adverse & (tp6 >= 0) & ((sl6 < 0) | (tp6 < sl6))
    moderate_adverse = valid & ~severe_adverse & ~strong_favourable & (sl4 >= 0) & ((tp4 < 0) | (sl4 <= tp4))
    moderate_favourable = valid & ~severe_adverse & ~strong_favourable & ~moderate_adverse & (tp4 >= 0) & ((sl4 < 0) | (tp4 < sl4))
    grades[severe_adverse] = 0
    grades[moderate_adverse] = 1
    grades[moderate_favourable] = 3
    grades[strong_favourable] = 4
    return grades


def _validate_identity(frame: pd.DataFrame, columns: TailBaseTargetColumns) -> None:
    missing = [column for column in columns.required if column not in frame.columns]
    if missing:
        raise TailBaseTargetError(f"missing required tail-target columns: {missing}")
    identity = list(columns.identity)
    if frame.loc[:, identity].isna().any().any():
        raise TailBaseTargetError("tail-target identity fields cannot be null")
    if frame.duplicated(identity).any():
        raise TailBaseTargetError("tail-target identity must be unique per candidate/timestamp/symbol/side")
    decision_ts = pd.to_datetime(frame[columns.decision_ts], errors="coerce", utc=True)
    if decision_ts.isna().any():
        raise TailBaseTargetError("decision_ts must be parseable as UTC")


def build_tail_base_targets(
    frame: pd.DataFrame,
    *,
    columns: TailBaseTargetColumns = TailBaseTargetColumns(),
    horizon_minutes: int = DEFAULT_HORIZON_MINUTES,
) -> pd.DataFrame:
    """Materialise all requested tail-base ranker targets on one identity-safe frame.

    The returned frame preserves the canonical identity and emits independent
    validity flags so a path invalid for every target never becomes a grade-0
    loss.  In valid H12 labels, the three validity flags are identical; keeping
    them explicit makes an incomplete source contract auditable.
    """

    if not isinstance(frame, pd.DataFrame):
        raise TypeError("frame must be a pandas DataFrame")
    _validate_identity(frame, columns)
    valid = _valid_mask(frame[columns.label_valid].to_numpy(), n=len(frame))
    net_grade = grade_exact_net_bps(frame[columns.exact_net_bps].to_numpy(), valid)
    atr_grade, z_atr = grade_atr_normalized_net(
        frame[columns.exact_net_bps].to_numpy(), frame[columns.atr_bps].to_numpy(), valid
    )
    tbm_grade = grade_first_touch_tbm(
        frame[columns.first_tp4_minute].to_numpy(),
        frame[columns.first_tp6_minute].to_numpy(),
        frame[columns.first_sl4_minute].to_numpy(),
        frame[columns.first_sl6_minute].to_numpy(),
        valid,
        horizon_minutes=horizon_minutes,
    )
    output = frame.loc[:, list(columns.identity)].copy()
    output["tail_target_valid"] = valid
    output["tail_target_net_grade_0_5"] = net_grade
    output["tail_target_atr_grade_0_5"] = atr_grade
    output["tail_target_atr_z"] = z_atr
    output["tail_target_tbm_grade_0_4"] = tbm_grade
    return output


__all__ = [
    "ATR_GRADE_BOUNDS",
    "DEFAULT_HORIZON_MINUTES",
    "INVALID_GRADE",
    "NET_GRADE_BOUNDS_BPS",
    "TailBaseTargetColumns",
    "TailBaseTargetError",
    "build_tail_base_targets",
    "grade_atr_normalized_net",
    "grade_exact_net_bps",
    "grade_first_touch_tbm",
]
