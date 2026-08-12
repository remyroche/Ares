"""Causal producer-local health checks for an exact-reserve EV map.

The exact 42-day producer reserve is the primary score-to-policy-net map.  It
is intentionally static from the producer's first live hour, so it cannot
know about a later abrupt failure of its *actionable* tail.  This module adds
an optional, post-map safety overlay.  It never pools raw scores or outcomes
between producers, and it only consumes policy labels available before the
current decision timestamp.

The overlay is deliberately separate from the base EV map.  It can be tested
or disabled without changing bundle semantics or creating an admission drought
at a refit boundary.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_PRIOR_COLUMN = "ev_bridge_prior_expected_net_bps"
DEFAULT_EXPECTED_COLUMN = "tail_health_expected_net_bps"
DEFAULT_LCB_COLUMN = "tail_health_lcb_bps"


@dataclass(frozen=True)
class TailHealthSpec:
    """Frozen, causal tail-health overlay parameters.

    ``tail_prior_floor_bps`` is evaluated from the exact reserve map before a
    current outcome exists.  It defines the only population whose realised
    residual can update the overlay.  This prevents broad-universe residuals
    from obscuring a failure of the admitted high-score tail.
    """

    tail_prior_floor_bps: float = 50.0
    admission_floor_bps: float = 50.0
    residual_windows_days: tuple[int, ...] = (14, 7, 3)
    residual_shrinkage_rows: tuple[float, ...] = (100.0, 50.0, 25.0)
    minimum_residual_rows: int = 20
    trim_fraction: float = 0.05
    lower_confidence_z: float = 0.0

    def __post_init__(self) -> None:
        if not self.residual_windows_days:
            raise ValueError("tail-health overlay needs one or more residual windows")
        if tuple(sorted(self.residual_windows_days, reverse=True)) != self.residual_windows_days:
            raise ValueError("tail-health windows must be broad-to-recent")
        if len(self.residual_windows_days) != len(self.residual_shrinkage_rows):
            raise ValueError("tail-health needs one shrinkage value per window")
        if any(days <= 0 for days in self.residual_windows_days):
            raise ValueError("tail-health windows must be positive")
        if any(value <= 0.0 for value in self.residual_shrinkage_rows):
            raise ValueError("tail-health shrinkage must be positive")
        if self.minimum_residual_rows < 4:
            raise ValueError("tail-health minimum support must be at least four rows")
        if not 0.0 <= self.trim_fraction < 0.5:
            raise ValueError("tail-health trim fraction must be in [0, .5)")
        if self.lower_confidence_z < 0.0:
            raise ValueError("tail-health lower-confidence multiplier cannot be negative")


def _trimmed_mean_and_se(values: np.ndarray, trim_fraction: float) -> tuple[float, float]:
    ordered = np.sort(np.asarray(values, dtype=float))
    trim = int(np.floor(len(ordered) * trim_fraction))
    kept = ordered[trim:len(ordered) - trim] if len(ordered) > 2 * trim else ordered
    if not len(kept):  # pragma: no cover - defensive after support validation
        return float("nan"), float("nan")
    if len(kept) < 2:
        return float(kept.mean()), 0.0
    return float(kept.mean()), float(kept.std(ddof=1) / np.sqrt(len(kept)))


def apply_exact_producer_tail_health(
    frame: pd.DataFrame,
    *,
    spec: TailHealthSpec = TailHealthSpec(),
    prior_column: str = DEFAULT_PRIOR_COLUMN,
    net_column: str = "policy_net_bps",
    path_valid_column: str = "policy_path_valid",
    decision_column: str = "__decision_ts__",
    label_available_column: str = "policy_label_available_ts",
    producer_column: str = "producer_bundle_id",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply a causal, producer-local high-tail residual safety overlay.

    ``frame`` may combine resolved history and an outcome-free live snapshot.
    A snapshot row always receives a finite initial result if its exact reserve
    prior is finite; unresolved rows never enter residual calculations.  The
    returned audit gives one record per producer × side × decision hour.
    """

    required = {
        "candidate_id", "side_name", decision_column, label_available_column,
        producer_column, prior_column, net_column, path_valid_column,
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"tail-health frame lacks: {missing}")
    if frame["candidate_id"].isna().any() or frame["candidate_id"].duplicated().any():
        raise ValueError("tail-health requires immutable unique candidate identities")

    original = frame.copy()
    original["__tail_health_position__"] = np.arange(len(original), dtype=np.int64)
    work = original.sort_values(
        [producer_column, "side_name", decision_column, "candidate_id"], kind="stable",
    ).reset_index(drop=True)
    decision = pd.to_datetime(work[decision_column], utc=True, errors="raise")
    label_available = pd.to_datetime(work[label_available_column], utc=True, errors="coerce")
    prior = pd.to_numeric(work[prior_column], errors="coerce").to_numpy(float)
    realised = pd.to_numeric(work[net_column], errors="coerce").to_numpy(float)
    valid_path = work[path_valid_column].fillna(False).astype(bool).to_numpy()
    eligible = np.isfinite(prior) & (prior >= spec.tail_prior_floor_bps)
    resolved = valid_path & np.isfinite(realised) & label_available.notna().to_numpy()
    if ((label_available <= decision) & label_available.notna()).any():
        raise ValueError("tail-health requires policy labels to resolve after their decision")

    correction = np.zeros(len(work), dtype=float)
    standard_error = np.full(len(work), np.nan, dtype=float)
    status = np.full(len(work), "reserve_prior_only_no_tail_residual_support", dtype=object)
    support = {
        window: np.zeros(len(work), dtype=np.int64)
        for window in spec.residual_windows_days
    }
    audit_rows: list[dict[str, Any]] = []

    group_columns = [producer_column, "side_name"]
    for values, group in work.groupby(group_columns, sort=False, observed=True):
        positions = group.index.to_numpy(np.int64)
        # The only historical state available to this producer is its own
        # resolved high-tail observations.  Sort this index by label readiness,
        # rather than decision time, to enforce point-in-time availability.
        usable = positions[resolved[positions] & eligible[positions]]
        available_ns = label_available.iloc[usable].astype("int64").to_numpy()
        order = np.argsort(available_ns, kind="stable")
        available_ns = available_ns[order]
        usable = usable[order]

        producer, side = values
        hours = decision.iloc[positions].drop_duplicates().sort_values()
        for timestamp in hours:
            current = positions[decision.iloc[positions].eq(timestamp).to_numpy()]
            cutoff_ns = int(timestamp.value)
            upper = int(np.searchsorted(available_ns, cutoff_ns, side="left"))
            estimate = 0.0
            any_support = False
            latest_se = float("nan")
            counts: dict[int, int] = {}
            max_available = pd.NaT
            for window, shrinkage in zip(
                spec.residual_windows_days, spec.residual_shrinkage_rows, strict=True,
            ):
                lower = int(np.searchsorted(
                    available_ns,
                    cutoff_ns - pd.Timedelta(days=window).value,
                    side="left",
                ))
                history = usable[lower:upper]
                count = int(len(history))
                counts[window] = count
                support[window][current] = count
                if count:
                    max_available = pd.Timestamp(available_ns[upper - 1], unit="ns", tz="UTC")
                if count >= spec.minimum_residual_rows:
                    mean, latest_se = _trimmed_mean_and_se(
                        realised[history] - prior[history], spec.trim_fraction,
                    )
                    weight = count / (count + float(shrinkage))
                    estimate = weight * mean + (1.0 - weight) * estimate
                    any_support = True
            correction[current] = estimate
            standard_error[current] = latest_se
            status[current] = (
                "exact_producer_tail_prior_plus_causal_residual"
                if any_support else "reserve_prior_only_no_tail_residual_support"
            )
            audit_rows.append({
                producer_column: producer,
                "side_name": side,
                "snapshot_utc": timestamp,
                "current_rows": int(len(current)),
                "tail_residual_correction_bps": float(estimate),
                "tail_residual_standard_error_bps": latest_se,
                "tail_residual_mapping_status": (
                    "mapped" if any_support else "prior_only_insufficient_tail_support"
                ),
                "strictly_prior_resolved": bool(max_available < timestamp) if pd.notna(max_available) else True,
                "reference_max_label_available_ts": max_available,
                **{f"tail_residual_reference_rows_{window}d": counts[window] for window in spec.residual_windows_days},
            })

    if audit_rows and not all(row["strictly_prior_resolved"] for row in audit_rows):
        raise AssertionError("tail-health overlay consumed an unresolved outcome")
    expected = prior + correction
    lcb = expected - np.where(
        np.isfinite(standard_error), spec.lower_confidence_z * standard_error, 0.0,
    )
    work["tail_health_reserve_eligible"] = eligible
    work["tail_health_recent_residual_bps"] = correction
    work["tail_health_recent_residual_se_bps"] = standard_error
    work[DEFAULT_EXPECTED_COLUMN] = expected
    work[DEFAULT_LCB_COLUMN] = lcb
    work["tail_health_mapping_status"] = status
    for window, values in support.items():
        work[f"tail_health_reference_rows_{window}d"] = values
    work["tail_health_admitted_ge_50bps"] = (
        eligible & np.isfinite(lcb) & (lcb >= spec.admission_floor_bps)
    )
    work["tail_health_contract"] = (
        "producer-local reserve-defined high-tail residual overlay; "
        "prior-resolved policy outcomes only"
    )
    out = work.sort_values("__tail_health_position__", kind="stable").drop(
        columns="__tail_health_position__",
    ).reset_index(drop=True)
    return out, pd.DataFrame(audit_rows)


__all__ = [
    "DEFAULT_EXPECTED_COLUMN",
    "DEFAULT_LCB_COLUMN",
    "TailHealthSpec",
    "apply_exact_producer_tail_health",
]
