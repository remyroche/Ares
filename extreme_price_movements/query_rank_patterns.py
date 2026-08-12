"""Predeclared economic relevance grades for query-construction ablations.

These are label functions only.  They never enter inference features, and
their required path primitives are deliberately explicit so a caller cannot
quietly substitute terminal returns for a first-touch contract.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class GradePattern:
    name: str
    family: str
    spacing: float
    lower_atr: float | None = None
    upper_atr: float | None = None
    minimum_absolute_bps: float = 0.0
    horizon_hours: int = 12


def grade_pattern_grid() -> tuple[GradePattern, ...]:
    """Small, declared grade grid requested for the development funnel."""
    result: list[GradePattern] = []
    for spacing in (1.0, 1.5, 2.0):
        result.append(GradePattern(f"atr_spacing_{spacing:g}", "atr_spacing", spacing))
        result.append(GradePattern(f"absolute_spacing_{spacing:g}pct", "absolute_spacing", spacing))
    for lower in (2.0, 3.0, 4.0):
        for upper in (2.0, 3.0, 4.0, 5.0, 6.0):
            if lower <= upper:
                result.append(GradePattern(
                    f"tbm_sl{lower:g}_tp{upper:g}_min1pct", "triple_barrier",
                    spacing=1.0, lower_atr=lower, upper_atr=upper,
                    minimum_absolute_bps=100.0,
                ))
    return tuple(result)


def _require(frame: pd.DataFrame, columns: tuple[str, ...]) -> None:
    missing = [column for column in columns if column not in frame]
    if missing:
        raise KeyError(f"grade pattern needs materialized H12 path fields: {missing}")


def construct_grades(frame: pd.DataFrame, pattern: GradePattern) -> np.ndarray:
    """Return grades 0..4 using gross/net and first-touch path primitives.

    Grade 0 is gross-negative. Grade 1 is gross-positive but net-negative or
    unresolved. Grades 2--4 require respectively 1%, 1.5%, and 2% gross
    clearance, in addition to the declared ATR/absolute path condition.
    """
    _require(frame, ("gross_bps", "net_bps", "path_timeout", "favorable_first", "adverse_first"))
    gross = pd.to_numeric(frame["gross_bps"], errors="coerce").to_numpy(float)
    net = pd.to_numeric(frame["net_bps"], errors="coerce").to_numpy(float)
    favorable_first = frame["favorable_first"].fillna(False).to_numpy(bool)
    adverse_first = frame["adverse_first"].fillna(False).to_numpy(bool)
    timeout = frame["path_timeout"].fillna(True).to_numpy(bool)
    valid = np.isfinite(gross) & np.isfinite(net) & (favorable_first | adverse_first | timeout)
    grade = np.zeros(len(frame), dtype=np.int8)
    weak = valid & (gross > 0.0) & ((net <= 0.0) | timeout)
    grade[weak] = 1
    if pattern.family == "absolute_spacing":
        threshold_bps = pattern.spacing * 100.0
        clear = gross >= threshold_bps
        levels = np.array([100.0, 150.0, 200.0])
        for level, minimum in enumerate(levels, start=2):
            grade[valid & clear & (gross >= minimum)] = level
    else:
        upper = pattern.upper_atr if pattern.family == "triple_barrier" else pattern.spacing
        lower = pattern.lower_atr if pattern.family == "triple_barrier" else pattern.spacing
        if lower is not None and upper is not None and lower > upper:
            raise ValueError("lower ATR threshold cannot exceed upper ATR threshold")
        # ``favorable_first`` is materialised separately for this exact
        # TP/SL contract.  It includes the adverse same-minute tie break.
        wins = valid & favorable_first & ~adverse_first
        # Economics guardrail applies even to large ATR moves in low ATR-bps regimes.
        for level, minimum in enumerate((100.0, 150.0, 200.0), start=2):
            grade[wins & (gross >= max(minimum, pattern.minimum_absolute_bps))] = level
    return grade


def construct_first_touch_grades(*, gross_bps: np.ndarray, net_bps: np.ndarray,
                                 favorable_minutes: np.ndarray, adverse_minutes: np.ndarray,
                                 thresholds: np.ndarray, lower_atr: float,
                                 upper_atr: float) -> np.ndarray:
    """Build the requested 0--4 first-touch grade from one H12 grid pass.

    The 1%, 1.5% and 2% gross guardrails are deliberately *additional* to the
    positive-first contract.  This prevents a high grade in tiny-ATR regimes
    where a nominal multi-ATR crossing is economically immaterial.
    """
    if lower_atr > upper_atr:
        raise ValueError("lower ATR limit cannot exceed upper ATR limit")
    threshold=np.asarray(thresholds,dtype=float)
    lower_index=np.flatnonzero(np.isclose(threshold,lower_atr))
    upper_index=np.flatnonzero(np.isclose(threshold,upper_atr))
    if len(lower_index)!=1 or len(upper_index)!=1:
        raise KeyError("declared threshold is absent from materialized first-touch grid")
    favorable=np.asarray(favorable_minutes)[:,upper_index[0]]
    adverse=np.asarray(adverse_minutes)[:,lower_index[0]]
    positive=(favorable>=0)&((adverse<0)|(favorable<adverse))
    negative=(adverse>=0)&((favorable<0)|(adverse<=favorable))
    gross=np.asarray(gross_bps,dtype=float); net=np.asarray(net_bps,dtype=float)
    result=np.zeros(len(gross),dtype=np.int8)
    result[(gross>0.)&((net<=0.)|(~positive&~negative))]=1
    for grade,floor in ((2,100.),(3,150.),(4,200.)):
        result[positive&(gross>=floor)]=grade
    return result
