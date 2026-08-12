"""Canonical equal-day, robust strict-R3 EV admission.

The active producer owns a frozen 28-day OOS reserve.  That reserve fixes the
score-cell coordinate system and seeds one policy-net observation per UTC day
and cell.  During the producer's lifetime, newly resolved same-producer
outcomes are appended causally.  At a decision on day ``d`` only labels whose
availability timestamp is strictly before ``d 00:00 UTC`` may contribute.

Each cell drops its highest and lowest 15% of daily means, receives equal
weight per retained day, and is projected onto a monotone score-to-EV curve.
The executable hurdle remains +50 net bps.  Raw scores and outcomes are never
pooled across upstream or conversion producers.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

from .strict_r3_ev_bridge import (
    EXACT_PRODUCER_RESERVE_CALIBRATION_MODE,
    StrictR3EVBridgeBundle,
)


CELL_DAY_TRIM_15_42D_CALIBRATION_MODE = (
    "strict_oof_exact_producer_cell_day_trim15_42d_v1"
)
CELL_DAY_TRIM_15_28D_CALIBRATION_MODE = (
    "strict_oof_exact_producer_cell_day_trim15_28d_v1"
)
# Active canonical mode.  The 42-day name remains exported solely so old,
# immutable research artifacts can still be identified accurately.
CELL_DAY_TRIM_15_CALIBRATION_MODE = CELL_DAY_TRIM_15_28D_CALIBRATION_MODE


@dataclass(frozen=True)
class CellDayAdmissionSpec:
    window_days: int = 28
    trim_fraction: float = 0.15
    net_floor_bps: float = 50.0

    def __post_init__(self) -> None:
        if self.window_days <= 0:
            raise ValueError("cell-day admission window must be positive")
        if not 0.0 <= self.trim_fraction < 0.5:
            raise ValueError("cell-day trim fraction must be in [0, .5)")
        if not np.isfinite(self.net_floor_bps):
            raise ValueError("cell-day EV floor must be finite")


def _cell_ids(reference: np.ndarray, values: np.ndarray, bins: int) -> np.ndarray:
    reference = np.asarray(reference, dtype=float)
    current = np.asarray(values, dtype=float)
    if len(reference) < bins * 4 or not np.isfinite(reference).all():
        raise ValueError("cell-day admission lacks its frozen score reference")
    output = np.full(len(current), -1, dtype=np.int16)
    finite = np.isfinite(current)
    rank = np.searchsorted(reference, current[finite], side="right")
    output[finite] = np.minimum(rank * bins // len(reference), bins - 1)
    return output


def _trim(values: np.ndarray, fraction: float) -> np.ndarray:
    ordered = np.sort(np.asarray(values, dtype=float))
    count = int(math.floor(len(ordered) * fraction))
    if not count:
        return ordered
    retained = ordered[count:len(ordered) - count]
    return retained if len(retained) else ordered


def _curve(table: pd.DataFrame, *, bins: int, trim_fraction: float) -> tuple[np.ndarray, np.ndarray]:
    means = np.full(bins, np.nan, dtype=float)
    support = np.zeros(bins, dtype=np.int64)
    for cell in range(bins):
        values = table.loc[
            table["__cell__"].eq(cell), "cell_day_ev_bps"
        ].to_numpy(float)
        retained = _trim(values, trim_fraction)
        support[cell] = len(retained)
        if len(retained):
            means[cell] = float(retained.mean())
    usable = np.isfinite(means) & (support > 0)
    if usable.sum() >= 2:
        x = (np.arange(bins, dtype=float) + 0.5) / bins
        model = IsotonicRegression(increasing=True, out_of_bounds="clip")
        fitted = means.copy()
        fitted[usable] = model.fit(
            x[usable], means[usable], sample_weight=support[usable],
        ).predict(x[usable])
        if not usable.all():
            fitted[~usable] = np.interp(x[~usable], x[usable], fitted[usable])
        means = fitted
    return means, support


def apply_cell_day_trim15_admission_snapshot(
    *,
    resolved_score_ledger: pd.DataFrame,
    current_scores: pd.DataFrame,
    bundle: StrictR3EVBridgeBundle,
    spec: CellDayAdmissionSpec = CellDayAdmissionSpec(),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply the frozen-producer cell-day map to one live UTC snapshot."""
    if bundle.calibration_mode != EXACT_PRODUCER_RESERVE_CALIBRATION_MODE:
        raise ValueError("cell-day admission requires an exact-producer reserve bundle")
    references = getattr(bundle, "side_score_references", {})
    seed = getattr(bundle, "cell_day_seed", pd.DataFrame())
    if not references or seed.empty:
        raise ValueError(
            "exact-producer bundle predates canonical cell-day state; refit the calibrator",
        )
    required_current = {
        "candidate_id", "__decision_ts__", "side_name", bundle.score_column,
        "ev_score_family_id", "geometry_bundle_sha256",
        "conversion_bundle_sha256", "upstream_bundle_sha256",
        "stack_is_prequential",
    }
    required_resolved = {
        *required_current, bundle.net_column, "policy_label_available_ts",
    }
    missing = sorted(required_current.difference(current_scores.columns))
    if missing:
        raise ValueError(f"current cell-day snapshot lacks: {missing}")
    missing = sorted(required_resolved.difference(resolved_score_ledger.columns))
    if missing:
        raise ValueError(f"resolved cell-day ledger lacks: {missing}")
    current = current_scores.copy()
    resolved = resolved_score_ledger.copy()
    if current["candidate_id"].isna().any() or current["candidate_id"].duplicated().any():
        raise ValueError("cell-day snapshot requires unique candidate identities")
    if not current["stack_is_prequential"].fillna(False).astype(bool).all():
        raise ValueError("current cell-day scores are not strict prequential")
    if not resolved["stack_is_prequential"].fillna(False).astype(bool).all():
        raise ValueError("resolved cell-day scores are not strict prequential")
    current["__decision_ts__"] = pd.to_datetime(
        current["__decision_ts__"], utc=True, errors="raise",
    )
    resolved["__decision_ts__"] = pd.to_datetime(
        resolved["__decision_ts__"], utc=True, errors="raise",
    )
    resolved["policy_label_available_ts"] = pd.to_datetime(
        resolved["policy_label_available_ts"], utc=True, errors="raise",
    )
    snapshots = current["__decision_ts__"].dt.normalize().unique()
    if len(snapshots) != 1:
        raise ValueError("one cell-day admission call must contain one UTC day")
    snapshot_day = pd.Timestamp(snapshots[0])
    activation = pd.Timestamp(bundle.fit_cutoff)
    if activation.tzinfo is None:
        activation = activation.tz_localize("UTC")
    else:
        activation = activation.tz_convert("UTC")
    if activation > current["__decision_ts__"].min():
        raise ValueError("cell-day producer activates after the current decision")

    expected_lineage = {
        "ev_score_family_id": bundle.ev_score_family_id,
        "geometry_bundle_sha256": bundle.geometry_bundle_sha256,
        **dict(bundle.producer_lineage),
    }
    for column, expected in expected_lineage.items():
        if not current[column].astype(str).eq(str(expected)).all():
            raise ValueError(f"cell-day snapshot rejects mismatched {column}")
        resolved = resolved.loc[resolved[column].astype(str).eq(str(expected))].copy()

    bins = bundle.spec.prior_bins
    seed = seed.copy()
    seed["side_name"] = seed["side_name"].astype(str).str.lower()
    seed["__day__"] = pd.to_datetime(seed["__day__"], utc=True, errors="raise")
    seed = seed.loc[
        seed["__day__"].ge(snapshot_day - pd.Timedelta(days=spec.window_days))
        & seed["__day__"].lt(activation)
    ].copy()

    dynamic_parts: list[pd.DataFrame] = []
    resolved = resolved.loc[
        resolved["__decision_ts__"].ge(activation)
        & resolved["__decision_ts__"].ge(snapshot_day - pd.Timedelta(days=spec.window_days))
        & resolved["policy_label_available_ts"].lt(snapshot_day)
        & np.isfinite(pd.to_numeric(resolved[bundle.net_column], errors="coerce"))
    ].copy()
    if "policy_path_valid" in resolved:
        resolved = resolved.loc[
            resolved["policy_path_valid"].fillna(False).astype(bool)
        ].copy()
    for side, block in resolved.groupby(
        resolved["side_name"].astype(str).str.lower(), observed=True, sort=True,
    ):
        reference = np.asarray(references.get(side, []), dtype=float)
        cell = _cell_ids(
            reference,
            pd.to_numeric(block[bundle.score_column], errors="coerce").to_numpy(float),
            bins,
        )
        dynamic = pd.DataFrame({
            "side_name": side,
            "__day__": block["__decision_ts__"].dt.normalize().to_numpy(),
            "__cell__": cell,
            "policy_net_bps": pd.to_numeric(
                block[bundle.net_column], errors="coerce",
            ).to_numpy(float),
        })
        dynamic = dynamic.loc[dynamic["__cell__"].ge(0)]
        dynamic_parts.append(
            dynamic.groupby(
                ["side_name", "__day__", "__cell__"],
                observed=True, sort=True,
            ).agg(
                cell_day_ev_bps=("policy_net_bps", "mean"),
                cell_day_trades=("policy_net_bps", "size"),
            ).reset_index()
        )
    history = pd.concat([seed, *dynamic_parts], ignore_index=True, sort=False)

    output = current.copy()
    output["causal_21d_side_expected_net_bps"] = np.nan
    output["cell_day_fixed_score_cell"] = -1
    output["cell_day_retained_day_support"] = 0
    audits: list[dict[str, object]] = []
    for side, positions in output.groupby(
        output["side_name"].astype(str).str.lower(), observed=True, sort=True,
    ).groups.items():
        reference = np.asarray(references.get(side, []), dtype=float)
        cells = _cell_ids(
            reference,
            pd.to_numeric(
                output.loc[positions, bundle.score_column], errors="coerce",
            ).to_numpy(float),
            bins,
        )
        table = history.loc[history["side_name"].eq(side)]
        curve, support = _curve(
            table, bins=bins, trim_fraction=spec.trim_fraction,
        )
        values = np.where(cells >= 0, curve[np.maximum(cells, 0)], np.nan)
        output.loc[positions, "causal_21d_side_expected_net_bps"] = values
        output.loc[positions, "cell_day_fixed_score_cell"] = cells
        output.loc[positions, "cell_day_retained_day_support"] = np.where(
            cells >= 0, support[np.maximum(cells, 0)], 0,
        )
        audits.append({
            "snapshot_utc": snapshot_day,
            "side_name": side,
            "current_rows": int(len(positions)),
            "seed_cell_days": int(len(seed.loc[seed["side_name"].eq(side)])),
            "dynamic_cell_days": int(sum(len(value) for value in dynamic_parts)),
            "reference_max_label_available_ts": (
                resolved["policy_label_available_ts"].max()
                if len(resolved) else pd.NaT
            ),
            "strictly_prior_resolved": bool(
                resolved["policy_label_available_ts"].max() < snapshot_day
            ) if len(resolved) else True,
            "mapped_curve_min_bps": float(np.nanmin(curve)),
            "mapped_curve_max_bps": float(np.nanmax(curve)),
            "admission_floor_bps": spec.net_floor_bps,
        })
    output["causal_21d_side_mapping_status"] = CELL_DAY_TRIM_15_CALIBRATION_MODE
    output["causal_21d_side_admitted_ge_50bps"] = (
        np.isfinite(output["causal_21d_side_expected_net_bps"])
        & output["causal_21d_side_expected_net_bps"].ge(spec.net_floor_bps)
    )
    output["ev_mapping_score_family_id"] = bundle.ev_score_family_id
    output["ev_mapping_geometry_bundle_sha256"] = bundle.geometry_bundle_sha256
    output["ev_mapping_conversion_vintage"] = output[
        "conversion_bundle_sha256"
    ].astype(str)
    output["ev_mapping_upstream_vintage"] = output[
        "upstream_bundle_sha256"
    ].astype(str)
    output["ev_mapping_vintage_mode"] = CELL_DAY_TRIM_15_CALIBRATION_MODE
    output["ev_bridge_bundle_identity"] = str(
        bundle.manifest.get("bundle_identity", "unpersisted")
    )
    audit = pd.DataFrame(audits)
    if not audit.empty and not audit["strictly_prior_resolved"].all():
        raise AssertionError("cell-day admission consumed an unresolved outcome")
    return output, audit


__all__ = [
    "CELL_DAY_TRIM_15_28D_CALIBRATION_MODE",
    "CELL_DAY_TRIM_15_42D_CALIBRATION_MODE",
    "CELL_DAY_TRIM_15_CALIBRATION_MODE",
    "CellDayAdmissionSpec",
    "apply_cell_day_trim15_admission_snapshot",
]
