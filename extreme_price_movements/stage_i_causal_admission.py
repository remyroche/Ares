"""Canonical causal 21-day EV admission for Stage-I predictions.

The component calibrates each side's model score to common expected *net bps*
from only already-resolved observations in the preceding 21 calendar days.
There is deliberately no pooled fallback: weak side support remains unmapped
and therefore inadmissible.  Admission is a floor on mapped expected net;
ranking happens afterwards across the surviving combined long/short population.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression


SIDES = ("long", "short")


@dataclass(frozen=True)
class Causal21dAdmissionSpec:
    window_days: int = 21
    min_reference_rows: int = 500
    bins: int = 20
    trim_fraction: float = 0.05
    net_floor_bps: float = 50.0

    def __post_init__(self) -> None:
        if self.window_days != 21:
            raise ValueError("canonical admission requires a full 21-calendar-day window")
        if self.min_reference_rows < 4 or self.bins < 4:
            raise ValueError("admission needs at least four reference rows/bins")
        if not 0.0 <= self.trim_fraction < 0.5:
            raise ValueError("trim_fraction must be in [0, .5)")


def _robust_mean(values: np.ndarray, trim_fraction: float) -> float:
    ordered = np.sort(np.asarray(values, dtype=float))
    trim = int(np.floor(len(ordered) * trim_fraction))
    kept = ordered[trim:len(ordered) - trim] if len(ordered) - 2 * trim else ordered
    return float(kept.mean())


def _fit_predict_robust_isotonic(
    reference_score: np.ndarray,
    reference_net_bps: np.ndarray,
    current_score: np.ndarray,
    spec: Causal21dAdmissionSpec,
) -> np.ndarray:
    """Equal-frequency score bins → trimmed conditional means → isotonic.

    The binning stabilises heavy-tailed trade outcomes while the weighted
    isotonic fit preserves the monotone score-to-common-bps contract.
    """
    valid = np.isfinite(reference_score) & np.isfinite(reference_net_bps)
    score, target = reference_score[valid], reference_net_bps[valid]
    if len(score) < spec.min_reference_rows or np.unique(score).size < 4:
        return np.full(len(current_score), np.nan)
    order = np.argsort(score, kind="stable")
    groups = np.minimum(np.arange(len(order)) * spec.bins // len(order), spec.bins - 1)
    table: list[tuple[float, float, int]] = []
    for bin_id in range(spec.bins):
        position = order[groups == bin_id]
        if not len(position):
            continue
        table.append((float(np.median(score[position])), _robust_mean(target[position], spec.trim_fraction), int(len(position))))
    x = np.asarray([row[0] for row in table], dtype=float)
    y = np.asarray([row[1] for row in table], dtype=float)
    w = np.asarray([row[2] for row in table], dtype=float)
    if len(x) < 4 or np.unique(x).size < 2:
        return np.full(len(current_score), np.nan)
    model = IsotonicRegression(increasing=True, out_of_bounds="clip")
    model.fit(x, y, sample_weight=w)
    output = np.full(len(current_score), np.nan)
    current_valid = np.isfinite(current_score)
    output[current_valid] = model.predict(current_score[current_valid])
    return output


def _validate_input(
    frame: pd.DataFrame, *, score_column: str, net_column: str,
    decision_column: str, label_available_column: str, identity_column: str,
) -> pd.DataFrame:
    required = {identity_column, "side_name", score_column, net_column, decision_column, label_available_column}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"causal admission lacks required columns: {missing}")
    if frame[identity_column].isna().any() or frame[identity_column].duplicated().any():
        raise ValueError("causal admission requires immutable unique candidate identities")
    out = frame.copy()
    out["side_name"] = out["side_name"].astype(str).str.lower()
    if not out["side_name"].isin(SIDES).all():
        raise ValueError("causal admission requires canonical long/short sides")
    out[decision_column] = pd.to_datetime(out[decision_column], utc=True, errors="raise")
    out[label_available_column] = pd.to_datetime(out[label_available_column], utc=True, errors="raise")
    if out[label_available_column].isna().any() or out[decision_column].isna().any():
        raise ValueError("decision and exact label availability timestamps must be finite")
    if (out[label_available_column] <= out[decision_column]).any():
        raise ValueError("label availability must be strictly after its decision")
    return out


def apply_causal_21d_side_admission(
    frame: pd.DataFrame,
    *,
    score_column: str,
    net_column: str = "net_bps",
    decision_column: str = "__ts__",
    label_available_column: str = "label_available_ts",
    identity_column: str = "candidate_id",
    spec: Causal21dAdmissionSpec = Causal21dAdmissionSpec(),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Map scores and apply the 50-bps floor without changing population.

    Every input candidate appears exactly once in the returned frame.  Rows
    with insufficient *side-local* resolved support are explicitly unmapped
    and rejected; no global or zero-valued fallback is permitted.
    """
    out = _validate_input(
        frame, score_column=score_column, net_column=net_column,
        decision_column=decision_column, label_available_column=label_available_column,
        identity_column=identity_column,
    )
    original_index = out.index.copy()
    out["__admission_original_position__"] = np.arange(len(out), dtype=np.int64)
    out = out.sort_values([decision_column, identity_column], kind="stable").reset_index(drop=True)
    row_count = len(out)
    expected_net_bps = np.full(row_count, np.nan, dtype=float)
    reference_rows = np.zeros(row_count, dtype=np.int64)
    mapping_status = np.full(
        row_count, "unmapped_insufficient_side_support", dtype=object,
    )
    admitted = np.zeros(row_count, dtype=bool)
    score = pd.to_numeric(out[score_column], errors="coerce").to_numpy(dtype=float)
    target = pd.to_numeric(out[net_column], errors="coerce").to_numpy(dtype=float)
    decision = out[decision_column]
    available = out[label_available_column]
    # Parquet-backed timestamps may retain microsecond rather than nanosecond
    # storage. Normalize both integer indexes to nanoseconds before combining
    # them with Timedelta.value (which is always nanoseconds).
    available_ns = available.array.as_unit("ns").asi8
    snapshot_values = decision.dt.normalize()
    snapshot_ns = snapshot_values.array.as_unit("ns").asi8
    side_values = out["side_name"].to_numpy(dtype=object)
    finite_reference = np.isfinite(score) & np.isfinite(target)

    # `out` is decision-sorted, so each normalized decision date occupies one
    # contiguous slice.  Build those slices once rather than asking pandas to
    # materialise a group index for every snapshot.
    if row_count:
        snapshot_starts = np.r_[0, np.flatnonzero(snapshot_ns[1:] != snapshot_ns[:-1]) + 1]
        snapshot_ends = np.r_[snapshot_starts[1:], row_count]
    else:
        snapshot_starts = snapshot_ends = np.empty(0, dtype=np.int64)

    # The old implementation scanned the complete ledger twice for every
    # calendar day.  Instead, maintain one side-local index sorted by exact
    # label availability and locate the 21-day half-open window with two binary
    # searches.  Validation guarantees label_available_ts > decision_ts, so
    # availability < snapshot also proves decision < snapshot.  Before fitting,
    # reference positions are restored to canonical decision/identity order;
    # this preserves stable tie behaviour in the equal-frequency score bins.
    reference_indices: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for side in SIDES:
        side_pos = np.flatnonzero((side_values == side) & finite_reference)
        availability_order = np.argsort(available_ns[side_pos], kind="stable")
        sorted_pos = side_pos[availability_order]
        reference_indices[side] = (available_ns[sorted_pos], sorted_pos)

    window_ns = pd.Timedelta(days=spec.window_days).value
    audit: list[dict[str, object]] = []
    for current_start, current_end in zip(snapshot_starts, snapshot_ends, strict=True):
        snapshot_value = int(snapshot_ns[current_start])
        snapshot = pd.Timestamp(snapshot_values.iloc[current_start])
        window_start_value = snapshot_value - window_ns
        window_start = snapshot - pd.Timedelta(days=spec.window_days)
        day_pos = np.arange(current_start, current_end, dtype=np.int64)
        for side in SIDES:
            current_pos = day_pos[side_values[day_pos] == side]
            sorted_available, sorted_reference_pos = reference_indices[side]
            lower = int(np.searchsorted(sorted_available, window_start_value, side="left"))
            upper = int(np.searchsorted(sorted_available, snapshot_value, side="left"))
            reference_count = upper - lower
            status = "mapped"
            if reference_count < spec.min_reference_rows:
                status = "unmapped_insufficient_side_support"
            elif not len(current_pos):
                status = "mapped_no_current_side_rows"
            else:
                reference_pos = np.sort(sorted_reference_pos[lower:upper])
                mapped = _fit_predict_robust_isotonic(score[reference_pos], target[reference_pos], score[current_pos], spec)
                if not np.isfinite(mapped).all():
                    status = "unmapped_degenerate_score_support"
                else:
                    expected_net_bps[current_pos] = mapped
                    admitted[current_pos] = mapped >= spec.net_floor_bps
            if len(current_pos):
                reference_rows[current_pos] = reference_count
                mapping_status[current_pos] = status
            reference_max = (
                pd.Timestamp(available.iloc[sorted_reference_pos[upper - 1]])
                if reference_count else pd.NaT
            )
            audit.append({
                "snapshot_utc": snapshot, "side_name": side,
                "window_start_utc": window_start, "window_end_exclusive_utc": snapshot,
                "reference_rows": int(reference_count), "current_rows": int(len(current_pos)),
                "reference_max_label_available_ts": reference_max,
                "strictly_prior_resolved": bool(reference_max < snapshot) if reference_count else True,
                "mapping_status": status,
            })
    out["causal_21d_side_expected_net_bps"] = expected_net_bps
    out["causal_21d_side_reference_rows"] = reference_rows
    out["causal_21d_side_mapping_status"] = mapping_status
    out["causal_21d_side_admitted_ge_50bps"] = admitted
    if not all(row["strictly_prior_resolved"] for row in audit):
        raise AssertionError("causal admission included an unresolved reference label")
    if out[identity_column].duplicated().any() or len(out) != len(frame):
        raise AssertionError("causal admission changed candidate population")
    out = out.sort_values("__admission_original_position__", kind="stable").drop(columns="__admission_original_position__")
    out.index = original_index
    if out.loc[~out["causal_21d_side_expected_net_bps"].notna(), "causal_21d_side_admitted_ge_50bps"].any():
        raise AssertionError("unmapped rows cannot be admitted")
    return out, pd.DataFrame(audit)


def pooled_global_admission_comparison(
    frame: pd.DataFrame,
    *,
    raw_score_column: str,
    net_column: str = "net_bps",
    gross_column: str | None = None,
    identity_column: str = "candidate_id",
    top_fractions: Sequence[float] = (0.01, 0.05, 0.10),
) -> pd.DataFrame:
    """Compare raw global ranks with post-admission pooled-global ranks.

    There is intentionally no timestamp or side grouping in either selection.
    """
    required = {raw_score_column, net_column, identity_column, "causal_21d_side_expected_net_bps", "causal_21d_side_admitted_ge_50bps"}
    if gross_column is not None:
        required.add(gross_column)
    if missing := sorted(required - set(frame.columns)):
        raise ValueError(f"admission comparison lacks columns: {missing}")
    rows: list[dict[str, object]] = []
    raw = frame[np.isfinite(pd.to_numeric(frame[raw_score_column], errors="coerce"))]
    admitted = frame[frame["causal_21d_side_admitted_ge_50bps"].astype(bool) & frame["causal_21d_side_expected_net_bps"].notna()]
    for fraction in top_fractions:
        if not 0.0 < float(fraction) <= 1.0:
            raise ValueError("top fractions must lie in (0, 1]")
        for label, population, score in (
            ("without_admission_raw_global", raw, raw_score_column),
            ("with_admission_mapped_pooled_global", admitted, "causal_21d_side_expected_net_bps"),
        ):
            n = min(len(population), max(1, int(np.ceil(len(raw) * float(fraction))))) if len(population) else 0
            selected = population.sort_values([score, identity_column], ascending=[False, True], kind="stable").head(n)
            rows.append({
                "comparison": label, "top_fraction_of_original_population": float(fraction),
                "original_population_rows": int(len(raw)), "eligible_rows": int(len(population)), "selected_rows": int(len(selected)),
                "mean_realised_net_bps": float(pd.to_numeric(selected[net_column], errors="coerce").mean()) if len(selected) else np.nan,
                "mean_realised_gross_bps": (
                    float(pd.to_numeric(selected[gross_column], errors="coerce").mean())
                    if len(selected) and gross_column is not None else np.nan
                ),
                "selected_long_rows": int(selected.side_name.eq("long").sum()) if len(selected) else 0,
                "selected_short_rows": int(selected.side_name.eq("short").sum()) if len(selected) else 0,
            })
    return pd.DataFrame(rows)


__all__ = ["Causal21dAdmissionSpec", "apply_causal_21d_side_admission", "pooled_global_admission_comparison"]
